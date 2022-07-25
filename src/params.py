import argparse

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Compression Artifacts Removal")
# training
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--step", type=int, default=80, help="decay of the learning rate after step of epochs")
parser.add_argument("--loss", default="L1", type=str, help="L1, L2")

# hardware
parser.add_argument("--cuda", default=True, help="use GPU")
parser.add_argument("--threads", default=4, type=int, help="number of threads for dataloader")
parser.add_argument("--gpus", default="[1]", type=str, help="gpu id list, multi-gpu such as [0,1,2,3]")

# network
parser.add_argument("--net", default="hinet", type=str, help="network")
parser.add_argument("--n_colors", default=3, type=int, help="number of color channels")
parser.add_argument("--in_channel", default=5, type=int, help="number of input channels")
parser.add_argument("--out_channel", default=3, type=int, help="number of output channels")
parser.add_argument("--n_resblocks", default=64, type=int, help="number of ResNet blocks")
parser.add_argument("--n_feats", default=64, type=int, help="number of feature channels")
parser.add_argument("--res_scale", default=0.1, type=float, help="ResNet scaling")
parser.add_argument("--in_glo_channel", default=3, type=int, help="number of global input channels")

# data path
parser.add_argument("--imgGTRoot", default="", type=str, help="Path of training GT images")
parser.add_argument("--imgValidRoot", default="", type=str, help="Path of validation images")

# training data
parser.add_argument("--dataloader", default="jpeg_qm", type=str, help="dataloader")
parser.add_argument("--batchSize", default=16, type=int, help="training batch size")
parser.add_argument("--size", default=64, type=int, help="size of training patch")
parser.add_argument("--quality", default=[10, 50], type=str, help="the range of quality factors when training")
parser.add_argument("--colorMode", default="RGB", type=str, help="L, RGB")

# others
parser.add_argument("--resume", default="", type=str, help="path to checkpoint, resume training")
# parser.add_argument("--pretrained", default="work_dirs/hinet_patch256_lr_1e_05_fineture/checkpoint/results/model_epoch_11.pth", type=str, help="path to pretrained model, for fine-tuning")
parser.add_argument("--pretrained", default="work_dirs/hinet_patch256_flikr2k_fineture/checkpoint/model_epoch_14.pth", type=str, help="path to pretrained model")


parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--savedir", default="./results", type=str, help="save checkpoint")
parser.add_argument("--validation", action="store_true", help="validation during training")


opt = parser.parse_args()

if opt.colorMode == 'L':
    opt.n_colors = 1
elif opt.colorMode == 'RGB':
    opt.n_colors = 3
else:
    raise RuntimeError('Color mode is invalid.')

if opt.in_channel == 0:
    if opt.net in ['qmar','qmarg']:
        if opt.colorMode == 'L':
            opt.in_channel = 2
            opt.in_glo_channel = 1
        else:
            opt.in_channel = 5
            opt.in_glo_channel = 3
    else:
        opt.in_channel = opt.n_colors

if opt.out_channel == 0:
    opt.out_channel = opt.n_colors

if opt.colorMode == 'L':
    opt.in_glo_channel = 1

# default data path
# opt.imgGTRoot = r'/data1/AIM2022/compress_image_track/div2k_patch_64_crop_datasets/gt' if opt.imgGTRoot == '' else opt.imgGTRoot
opt.imgGTRoot = '/data1/Datasets/Flickr2K/Flickr2K_HR' if opt.imgGTRoot == '' else opt.imgGTRoot
# opt.imgGTRoot = '/data1/AIM2022/DIV2K/gt' if opt.imgGTRoot == '' else opt.imgGTRoot   # /data1/AIM2022/Video_data/video_train_gt_frames_sr_gt
opt.imgValidRoot = '/data1/AIM2022/LIVE1/jpeg_qf_10/live1/gt' if opt.imgValidRoot == '' else opt.imgValidRoot
