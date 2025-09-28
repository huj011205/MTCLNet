# -*- coding: UTF-8 -*-
"""
@Project ：MC-Net-main-2
@File    ：train_mtclnet_3d.py.py
@IDE     ：PyCharm
@Author  ：HYZ
@Date    ：2024/1/10 9:11
"""
import argparse
import logging
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tensorboardX import SummaryWriter
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.sdf import compute_sdf
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import losses, ramps, test_mtclnet_patch


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

def get_current_gamma(iter_num):
    if iter_num < 6000:
        return 0
    elif 6000 <= iter_num < 15000:
        # 从0逐渐增加到0.3，选择一个合适的函数即可，这里以线性增长为例
        return 0.3 * (iter_num - 6000) / (15000 - 6000)
    else:
        # 达到15000次以后，gamma值保持不变
        return 0.3

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../', help='dataset_path')
parser.add_argument('--exp', type=str, default='tmax-20000-16', help='exp_name')
parser.add_argument('--model', type=str, default='mtclnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of all  data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='lr rate for training')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='labeled samples')
parser.add_argument('--seed', type=int, default=1338, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=50.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--alpha', type=float, default=0.1, help='weight to balance all losses')
parser.add_argument('--beta', type=float, default=0.3, help='weight to balance all losses')
parser.add_argument('--gamma', type=float, default=0.2, help='weight to balance all losses')
parser.add_argument('--lamda', type=float, default=0.4, help='weight to balance all losses')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Pancreas'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == '__main__':
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + '/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    if args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model.encoder.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model.decoder1.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(model.decoder2.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    optimizer4 = optim.SGD(model.decoder3.parameters(), lr=0.5, momentum=0.9, weight_decay=0.0001)

    # 设置 CosineAnnealingWarmRestarts 调度器
    max_epoch = max_iterations // len(trainloader) + 1
    T_0 = max_epoch // 5  # 这里假设 T_0 是总轮数的 1/5，你可以根据需要调整这个值
    scheduler1 = CosineAnnealingWarmRestarts(optimizer1, T_0=T_0, T_mult=2, eta_min=0.0001)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=T_0, T_mult=2, eta_min=0.0001)
    scheduler3 = CosineAnnealingWarmRestarts(optimizer3, T_0=T_0, T_mult=2, eta_min=0.0001)
    scheduler4 = CosineAnnealingWarmRestarts(optimizer4, T_0=T_0, T_mult=2, eta_min=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    iter_num = 0
    best_dice = 0
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()

            outputs_seg, outputs_sdf, outputs_rec = model(volume_batch)

            outputs_seg_soft = torch.sigmoid(outputs_seg)
            outputs_rec_soft = torch.sigmoid(outputs_rec)

            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu().numpy(), outputs_seg[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_sdf[:labeled_bs, 0, ...], gt_dis)
            loss_seg = losses.dice_loss(outputs_seg_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

            dis_to_mask = torch.sigmoid(-1500 * outputs_sdf)
            loss_consistency = torch.mean((dis_to_mask - outputs_seg_soft) ** 2)

            fore_volume = outputs_seg_soft * volume_batch
            back_volume = (1 - outputs_seg_soft) * volume_batch  # 或者volume_batch -
            fore_rec = outputs_seg_soft * outputs_rec
            back_rec = (1 - outputs_seg_soft) * outputs_rec  # 或者outputs_rec - fore_rec
            compose_volume = fore_volume + back_volume

            w1 = torch.sum((1 - outputs_seg_soft))
            w2 = torch.sum(outputs_seg_soft)
            total = w1 + w2
            w1 = w1 / total
            w2 = w2 / total

            foreground_loss = mse_loss(fore_rec, fore_volume)
            background_loss = mse_loss(back_rec, back_volume)
            loss_rec = w1 * background_loss + w2 * foreground_loss
            loss_rec_2 = mse_loss(outputs_rec, compose_volume)

            supervised_loss = loss_seg + args.beta * loss_sdf

            unsupervised_loss = args.lamda * loss_consistency + args.gamma * loss_rec
            consistency_weight = get_current_consistency_weight(iter_num // 500)
            loss_total = supervised_loss + consistency_weight * unsupervised_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            loss_total.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_total', loss_total, iter_num)
            writer.add_scalar('loss/loss_sdf', loss_sdf, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_consistency', loss_consistency, iter_num)
            writer.add_scalar('loss/loss_rec', loss_rec, iter_num)
            writer.add_scalar('loss/loss_rec_2', loss_rec_2, iter_num)
            writer.add_scalar('loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('loss/unsupervised_loss', unsupervised_loss, iter_num)
            writer.add_scalar('loss/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d : loss_total : %f, loss_sdf: %f, loss_seg: %f, loss_consistency: %f, loss_rec: %f, loss_rec_2: %f' % (
                iter_num, loss_total.item(), loss_sdf.item(), loss_seg.item(), loss_consistency.item(), loss_rec.item(), loss_rec_2.item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_seg_soft[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = outputs_rec_soft[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/outputs_rec_soft_1', grid_image, iter_num)

                image = outputs_seg[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/out_seg', grid_image, iter_num)

                image = outputs_rec[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/rec_image', grid_image, iter_num)

                image = fore_volume[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/fore_volume', grid_image, iter_num)

                image = fore_rec[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/fore_rec', grid_image, iter_num)

                image = back_volume[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/back_volume', grid_image, iter_num)

                image = back_rec[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/back_rec', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outputs_sdf[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap', grid_image, iter_num)

            # 更新调度器
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_mtclnet_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_mtclnet_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
