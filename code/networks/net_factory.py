# networks/net_factory.py
from .MTCLNet import MTCLNet3d_v1
from .VNet import VNet, MCNet3d_v1, MCNet3d_v2
from .unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3
from .MBCNet import MBCNet3d_v1

def net_factory(net_type="unet", in_chns=1, class_num=4, mode="train", n_filters=16):
    net_type = net_type.lower()
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', 
                   has_dropout=(mode == "train")).cuda()
    elif net_type == "mcnet3d_v1":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', 
                         has_dropout=(mode == "train")).cuda()
    elif net_type == "mcnet3d_v2":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', 
                         has_dropout=(mode == "train")).cuda()
    elif net_type == "mtclnet3d_v1":
        net = MTCLNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', 
                           has_dropout=(mode == "train")).cuda()

    else:
        raise ValueError(f"Unsupported net_type: {net_type}")
    return net