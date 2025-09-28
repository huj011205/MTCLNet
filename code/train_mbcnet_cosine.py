# -*- coding: UTF-8 -*-
"""
@Project ：MBCNet3d_v1
@File    ：train_mbcnet3d_v1.py
@IDE     ：PyCharm
@Author  ：HYZ (adapted by Grok)
@Date    ：2025/09/12
"""
import argparse
import logging
import os
import shutil
import sys
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# 假设以下文件存在于项目目录
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import ramps,test_mbcnet_patch
from utils.sdf1 import compute_sdf



# 损失函数定义
class MBCNet3dLoss(torch.nn.Module):
    def __init__(self, seg_weight=1.0, edge_weight=0.5, embed_weight=0.5, dice_weight=0.5, ce_weight=0.5):
        super(MBCNet3dLoss, self).__init__()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
        self.embed_weight = embed_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
        target = target.float()
        if pred.shape[1] > 1:
            dice = 0
            for c in range(pred.shape[1]):
                pred_c = pred[:, c].contiguous().view(-1)
                target_c = (target == c).float().view(-1)
                intersection = (pred_c * target_c).sum()
                dice += (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            return 1 - dice / pred.shape[1]
        else:
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def forward(self, outputs, seg_target, edge_target, embed_target):
        segA, edgeA, segB, embedB, segC = outputs
        batch_size = seg_target.size(0)
        segA_ce = self.ce_loss(segA, seg_target).clamp(min=0)
        segA_dice = self.dice_loss(segA, seg_target).clamp(min=0)
        segA_loss = self.ce_weight * segA_ce + self.dice_weight * segA_dice

        segB_ce = self.ce_loss(segB, seg_target).clamp(min=0)
        segB_dice = self.dice_loss(segB, seg_target).clamp(min=0)
        segB_loss = self.ce_weight * segB_ce + self.dice_weight * segB_dice

        segC_ce = self.ce_loss(segC, seg_target).clamp(min=0)
        segC_dice = self.dice_loss(segC, seg_target).clamp(min=0)
        segC_loss = self.ce_weight * segC_ce + self.dice_weight * segC_dice

        edge_ce = self.bce_loss(edgeA.squeeze(1), edge_target.float()).clamp(min=0)
        edge_dice = self.dice_loss(edgeA, edge_target).clamp(min=0)
        edge_loss = self.ce_weight * edge_ce + self.dice_weight * edge_dice

        embedB_flat = embedB.permute(0, 2, 3, 4, 1).reshape(-1, embedB.shape[1])
        embed_target_flat = embed_target.permute(0, 2, 3, 4, 1).reshape(-1, embed_target.shape[1])
        # 调整 labels 为基于 seg_target 的相似性（示例：相同类别为 1，不同为 -1）
        labels = torch.ones_like(embedB_flat[:, 0]) * 2 - 1  # 暂设为 1，需根据任务调整
        embed_loss = self.cosine_loss(embedB_flat, embed_target_flat, labels).clamp(min=0)

        total_loss = (self.seg_weight * (segA_loss + segB_loss + segC_loss) / 3 +
                      self.edge_weight * edge_loss +
                      self.embed_weight * embed_loss)
        return total_loss


def get_current_consistency_weight(iter_num):
    # 基础 sigmoid rampup
    base_weight = ramps.sigmoid_rampup(iter_num // 500, args.consistency_rampup)
    # 根据标签数量调整
    unlabeled_ratio = 1.0 - (args.labelnum / args.max_samples)
    adjusted_weight = args.consistency * base_weight * (1.0 + (1.0 - unlabeled_ratio) * 0.5)
    return adjusted_weight


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='数据集名称')
parser.add_argument('--root_path', type=str, default='../', help='数据集路径')
parser.add_argument('--exp', type=str, default='mbc_16_new', help='实验名称')
parser.add_argument('--model', type=str, default='mbcnet3d_v1', help='模型名称')
parser.add_argument('--max_iteration', type=int, default=35000, help='最大训练迭代次数')
parser.add_argument('--max_samples', type=int, default=80, help='最大训练样本数')
parser.add_argument('--labeled_bs', type=int, default=2, help='每 GPU 标记数据的批次大小')
parser.add_argument('--batch_size', type=int, default=4, help='每 GPU 总批次大小')
parser.add_argument('--base_lr', type=float, default=0.1, help='基础学习率')
parser.add_argument('--deterministic', type=int, default=1, help='是否使用确定性训练')
parser.add_argument('--labelnum', type=int, default=16, help='标记样本数')
parser.add_argument('--seed', type=int, default=1338, help='随机种子')
parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU')
parser.add_argument('--temperature', type=float, default=0.1, help='锐化温度参数')
parser.add_argument('--seg_weight', type=float, default=1.0, help='分割损失权重')
parser.add_argument('--edge_weight', type=float, default=0.5, help='边界损失权重')
parser.add_argument('--embed_weight', type=float, default=0.5, help='嵌入损失权重')
args = parser.parse_args()

if args.labelnum <= 8:
    consistency = 1.0
    consistency_rampup = 50.0
    decoder_c_lr = 0.5
elif args.labelnum <= 16:
    consistency = 1.5
    consistency_rampup = 30.0
    decoder_c_lr = 0.3
else:
    consistency = 2.0
    consistency_rampup = 20.0
    decoder_c_lr = 0.2

args.consistency = consistency
args.consistency_rampup = consistency_rampup

snapshot_path = os.path.join(args.root_path,
                             f"model/{args.dataset_name}_{args.exp}_{args.labelnum}_labeled/{args.model}")
num_classes = 2

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = os.path.join(args.root_path, 'data/LA')
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = os.path.join(args.root_path, 'data/Pancreas')
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
    if os.path.exists(os.path.join(snapshot_path, 'code')):
        shutil.rmtree(os.path.join(snapshot_path, 'code'))
    shutil.copytree('.', os.path.join(snapshot_path, 'code'), shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = model.cuda()

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model.encoder.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model.decoderA.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(model.decoderB.parameters(), lr=base_lr*0.05 , momentum=0.9, weight_decay=0.0001)
    optimizer4 = optim.SGD(model.decoderC.parameters(), lr=decoder_c_lr, momentum=0.9, weight_decay=0.0001)


    loss_fn = MBCNet3dLoss(seg_weight=args.seg_weight, edge_weight=args.edge_weight, embed_weight=args.embed_weight)
    mse_loss = torch.nn.MSELoss()

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num = 0
    best_dice = 0
    lr_ = base_lr
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            segA, edgeA, segB, embedB, segC = model(volume_batch)

            # 监督损失
            edge_target = compute_sdf(label_batch[:labeled_bs].cpu().numpy(), segA[:labeled_bs].shape[2:])
            edge_target = torch.from_numpy(edge_target).float().cuda()
            embed_target = torch.zeros_like(embedB[:labeled_bs])
            supervised_loss = loss_fn((segA[:labeled_bs], edgeA[:labeled_bs], segB[:labeled_bs],
                                       embedB[:labeled_bs], segC[:labeled_bs]),
                                      label_batch[:labeled_bs], edge_target, embed_target)

            # 一致性损失
            segA_soft = torch.softmax(segA, dim=1)
            segB_soft = torch.softmax(segB, dim=1)
            segC_soft = torch.softmax(segC, dim=1)
            consistency_loss = (mse_loss(segA_soft[labeled_bs:], segB_soft[labeled_bs:]) +
                                mse_loss(segB_soft[labeled_bs:], segC_soft[labeled_bs:]) +
                                mse_loss(segC_soft[labeled_bs:], segA_soft[labeled_bs:])) / 3

            consistency_weight = get_current_consistency_weight(iter_num // 500)
            loss_total = supervised_loss + consistency_weight * consistency_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            iter_num += 1

            if iter_num % 100 == 0:
                print(
                    f"Iter {iter_num}: total_loss: {loss_total.item():.4f}, supervised: {supervised_loss.item():.4f}, consistency: {consistency_loss.item():.4f}")

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_total', loss_total, iter_num)
            writer.add_scalar('loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('loss/consistency_weight', consistency_weight, iter_num)

            logging.info(f'iteration {iter_num} : loss_total: {loss_total.item():.5f}, '
                         f'supervised_loss: {supervised_loss.item():.5f}, '
                         f'consistency_loss: {consistency_loss.item():.5f}')

            # if iter_num % 20 == 0:
            #     # 输入图像
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     # edgeA
            #     edge_slice = torch.sigmoid(edgeA[0, 0, :, :, 20])
            #     image = edge_slice.unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 1, normalize=False)
            #     writer.add_image('train/EdgeA', grid_image, iter_num)
            #
            #     # segA
            #     image = torch.softmax(segA[0:1], dim=1)[0, 1:2, :, :, 20].unsqueeze(0).permute(2, 0, 1, 3).repeat(1, 3,
            #                                                                                                       1, 1)
            #     grid_image = make_grid(image, 1, normalize=False)
            #     writer.add_image('train/SegA', grid_image, iter_num)
            #
            #     # segB
            #     image = torch.softmax(segB[0:1], dim=1)[0, 1:2, :, :, 20].unsqueeze(0).permute(2, 0, 1, 3).repeat(1, 3,
            #                                                                                                       1, 1)
            #     grid_image = make_grid(image, 1, normalize=False)
            #     writer.add_image('train/SegB', grid_image, iter_num)
            #
            #     # embedB
            #     image = embedB[0, 0:1, :, :, 20].unsqueeze(0).permute(2, 0, 1, 3).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 1, normalize=True)
            #     writer.add_image('train/EmbedB', grid_image, iter_num)
            #
            #     # segC
            #     image = torch.softmax(segC[0:1], dim=1)[0, 1:2, :, :, 20].unsqueeze(0).permute(2, 0, 1, 3).repeat(1, 3,
            #                                                                                                       1, 1)
            #     grid_image = make_grid(image, 1, normalize=False)
            #     writer.add_image('train/SegC', grid_image, iter_num)
            #
            #     # 真值标签
            #     label_slice = label_batch[0, :, :, 20]
            #     image = label_slice.unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 1, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)
            if iter_num % 20 == 0:
                depth = label_batch.shape[2]

                # 寻找包含前景的切片索引（基于有标签样本）
                foreground_slices = []
                for z in range(depth):
                    if torch.any(label_batch[0, :, :, z] > 0):
                        foreground_slices.append(z)
                        if len(foreground_slices) >= 5:  # 最多取 5 个前景切片
                            break

                # 如果前景切片不足 5 个，补齐或使用默认索引
                if len(foreground_slices) < 5:
                    print(f"Warning: Only {len(foreground_slices)} foreground slices found in label_batch[0]")
                    while len(foreground_slices) < 5 and foreground_slices:
                        foreground_slices.append(foreground_slices[-1])

                if not foreground_slices:
                    print("Warning: No foreground slice found in label_batch[0], using default slices")
                    foreground_slices = list(range(20, min(61, depth), 10))[:5]

                # 确保切片索引不超过深度
                foreground_slices = [min(z, depth - 1) for z in foreground_slices][:5]

                # 输入图像 (volume_batch, 有标签样本)
                image = volume_batch[0, 0:1, :, :, foreground_slices]
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)  # [5, 1, H, W] -> [5, 3, H, W]
                grid_image = make_grid(image, 5, normalize=True)  # 全局归一化
                writer.add_image('train/Image_Labeled', grid_image, iter_num)

                # 真值标签 (label_batch, 有标签样本)
                image = label_batch[0, :, :, foreground_slices].float().unsqueeze(0)
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)  # 标签无需归一化
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                # segA (有标签样本)
                image = torch.softmax(segA[0:1], dim=1)[0, 1:2, :, :, foreground_slices]
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)  # softmax 已归一化
                writer.add_image('train/SegA_Labeled', grid_image, iter_num)

                # segB (有标签样本)
                image = torch.softmax(segB[0:1], dim=1)[0, 1:2, :, :, foreground_slices]
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/SegB_Labeled', grid_image, iter_num)

                # segC (有标签样本)
                image = torch.softmax(segC[0:1], dim=1)[0, 1:2, :, :, foreground_slices]
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/SegC_Labeled', grid_image, iter_num)

                # edgeA (有标签样本)
                image = torch.sigmoid(edgeA[0, 0:1, :, :, foreground_slices])
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)  # sigmoid 已归一化
                writer.add_image('train/EdgeA_Labeled', grid_image, iter_num)

                # embedB (有标签样本)
                image = embedB[0, 0:1, :, :, foreground_slices]
                image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)  # 全局归一化
                writer.add_image('train/EmbedB_Labeled', grid_image, iter_num)

                # 无标签样本的预测（使用 batch_size - labeled_bs 的第一个无标签样本，例如 index=2）
                if args.batch_size > args.labeled_bs:  # 确保有无标签样本
                    # 输入图像 (volume_batch, 无标签样本)
                    image = volume_batch[args.labeled_bs, 0:1, :, :, foreground_slices]
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('train/Image_Unlabeled', grid_image, iter_num)

                    # segA (无标签样本)
                    image = torch.softmax(segA[args.labeled_bs:args.labeled_bs + 1], dim=1)[0, 1:2, :, :,
                            foreground_slices]
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/SegA_Unlabeled', grid_image, iter_num)

                    # segB (无标签样本)
                    image = torch.softmax(segB[args.labeled_bs:args.labeled_bs + 1], dim=1)[0, 1:2, :, :,
                            foreground_slices]
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/SegB_Unlabeled', grid_image, iter_num)

                    # segC (无标签样本)
                    image = torch.softmax(segC[args.labeled_bs:args.labeled_bs + 1], dim=1)[0, 1:2, :, :,
                            foreground_slices]
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/SegC_Unlabeled', grid_image, iter_num)

                    # edgeA (无标签样本)
                    image = torch.sigmoid(edgeA[args.labeled_bs, 0:1, :, :, foreground_slices])
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/EdgeA_Unlabeled', grid_image, iter_num)

                    # embedB (无标签样本)
                    image = embedB[args.labeled_bs, 0:1, :, :, foreground_slices]
                    image = image.permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('train/EmbedB_Unlabeled', grid_image, iter_num)
            # 学习率调度
            if iter_num % 15000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 15000)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer3.param_groups:
                    param_group['lr'] = 0.005 * 0.1 ** (iter_num // 15000)
                for param_group in optimizer4.param_groups:
                    param_group['lr'] = 0.5 * 0.1 ** (iter_num // 15000)

            if iter_num % 30000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 30000)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = 0.01 * 0.1 ** (iter_num // 30000)
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = 0.01 * 0.1 ** (iter_num // 30000)
                for param_group in optimizer3.param_groups:
                    param_group['lr'] = 0.0005 * 0.1 ** (iter_num // 30000)
                for param_group in optimizer4.param_groups:
                    param_group['lr'] = 0.005 * 0.1 ** (iter_num // 30000)

            # 定期验证
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                dice_sample =test_mbcnet_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                           stride_xy=18 if args.dataset_name == "LA" else 16,
                                           stride_z=4 if args.dataset_name == "LA" else 16,
                                           dataset_name=args.dataset_name)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice:.4f}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info(f"save model to {save_mode_path}")
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()