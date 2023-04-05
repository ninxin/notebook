import os
import time
import datetime

import torch

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
from torchvision import transforms

# from unet import UNet as Model        # ✓ GN last
# from FANet import FANet as Model        # 要mask
# from VNet import VNet as Model
# from R2_Attu import R2U_Net as Model        # OOM √ GN last
# from R2_Attu import AttU_Net as Model       # ✓ GN last
# from R2_Attu import R2AttU_Net as Model     # OOM √ GN last
# from NestedUNet import NestedUNet as Model        # ✓ GN last
# from doublenet import doubleNet as Model
# from KiU_Net import kiunet as Model     # OOM OOM

# from ResUnetPlus import ResUnetPlusPlus as Model        # ✓ GN
# from MC_UNet import MCUNet as Model     # ✓ GN
# from RC_Net import RCNet as Model       # ✓ GN
# from DC_UNet import DcUnet as Model     # OOM √ GN last
# from XNet import XNet as Model
# from Unet3plus import UNet3Plus as Model
# from HistoSeg import HistoSeg as Model        # 不行
# from NanoNet import NanoNet as Model      # √
# from IterNet import Iternet as Model        # √, GN
# from AGNet import AG_Net as Model           # √, GN
# from CE_Net import CE_Net_OCT as Model      # √, GN
# from DUNet import DUNetV1V2 as Model         # √, GN
# from XNet import MyNet as Model
# from FRNet import FR_UNet as Model
from Unet3plus import UNet3Plus as Model



class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            # transforms.Resize([128, 128]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # transforms.Resize([128, 128]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


# def create_model(num_classes):
#     model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
#     return model


def main(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"            # PCI_BUS_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # -1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #device = args.device
    print(device)
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])       # cpu的数量
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # model = create_model(num_classes=num_classes)

    # model = Model(in_channels=3, num_classes=num_classes, base_c=32)  # unet
    model = Model()     # FANet FR_UNet UNet3Plus
    # model = Model(num_classes=2)        # NestedNet
    # model = Model()         # XNet, doublenet, Kiunet, CE-Net, FANet, UNet3Plus, HistoSeg, NanoNet
    # model = Model(channel=3)        # ResUnetPlusPlus
    # model = Model(inchan=3, num_class=2)        # MC-UNet
    # model = Model(in_chann=3, mid_chann=32, num_class=2)        # RCNet
    # model = Model(output_ch=2)      # R2U_Net AttU_Net R2AttU_Net
    # model = Model(input_channels=3)     # DcUnet
    # model = Model(n_channels=3, n_classes=2)        # Iternet
    # model = Model(n_classes=2)          # AG_Net
    # model = Model(n_channels=3, n_classes=2)        # DUNet
    # model = Model(3, 2)         # MyNet


    model.to(device)        # 输出为2通道

    # 用来保存训练以及验证过程中信息
    # results_file = "result/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = "result/{}{}.txt".format(datetime.datetime.now().strftime("%m%d-%H%M-"), 'UNet3Plus')

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    #
    # AGNet
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # AGNet


    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_UNet3Plus_model.pth")
        else:
            torch.save(save_file, "save_weights/UNet3Plus_model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="dataset/", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")     # 200

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
