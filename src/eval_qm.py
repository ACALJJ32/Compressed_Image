import os
import argparse
import torch
import numpy as np
import time
import skimage
import glob
import utils.util_image as util
from utils.util_quan import getQM
from dataset import *

parser = argparse.ArgumentParser(description="PyTorch QCN Demo")
parser.add_argument("--cuda", default=False, help="use cuda?")
parser.add_argument("--model", default="weight/dqt_hinet/dqt_hinet_flikr2k_p512_epoch57.pth", type=str, help="model path")
parser.add_argument("--realdatapath", default='/data1/AIM2022/compress_image_track/image_val', type=str, help="test data path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--GT", default=False, help="evaluate GT image dataset")
parser.add_argument("--quality", default=10, type=int, help="JPEG quality when evaluating GT images")
parser.add_argument("--save", default="div2k_image_val", type=str, help="save path") 
parser.add_argument("--colorMode", default="RGB", type=str, help="L, RGB")


def padding8(imgGT):
        H, W = imgGT.shape[0:2]
        pad_h = 8 - H % 8 if H % 8 != 0 else 0
        pad_w = 8 - W % 8 if W % 8 != 0 else 0
        imgGT = np.pad(imgGT, ((0, pad_h),(0,pad_w),(0,0)), 'edge')
        return imgGT


# parameters
opt = parser.parse_args()
print(opt)
cuda = opt.cuda

if cuda:
    print(f"=> use gpu id: '{opt.gpus}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


fullPaths = sorted(glob.glob(os.path.join(opt.realdatapath, '*.jpg')))
imgGTRoot = opt.realdatapath

start_time_all = time.time()
gputime = 0

# loading model =====
modelData = torch.load(opt.model)
model = modelData['model']
if cuda:
    model = model.cuda()
else:
    model = model.cpu()
model.eval()

trainOpt = modelData['opt'] if 'opt' in modelData else None
if trainOpt:
    if opt.colorMode != trainOpt['colorMode']:
        print(f'WARNING: The color mode of this model is {trainOpt["colorMode"]}, not {opt.colorMode}.')
        opt.colorMode = trainOpt['colorMode']
    dataloader = trainOpt['dataloader'] if 'dataloader' in trainOpt else ''

valid_set = eval(dataloader).ValidData(imgValidRoot=imgGTRoot, 
    quality=opt.quality, 
    colorMode=opt.colorMode)

for i in range(len(valid_set)):
    currStartTime = time.time()
    print(f'\n[{i+1}/{len(valid_set)}]', valid_set.imgNames[i])
    img_name = os.path.basename(valid_set.imgNames[i]).split(".")[0]

    currImg = util.imread_uint8(fullPaths[i], mode=opt.colorMode)
    with open(fullPaths[i],"rb") as f:
        img_bytes=f.read()

    ori_h, ori_w = currImg.shape[0:2]
    currImg = padding8(currImg)
    QMs = getQM(img_bytes)

    # Quantization map
    QM = np.zeros([8,8,2], dtype=np.uint8)
    QM[:,:,0] = QMs[0]
    QM[:,:,1] = QMs[1]
    cnt_h, cnt_w = np.array(currImg.shape[0:2]) // 8
    QMimg = np.tile(QM, (cnt_h, cnt_w, 1))
    currImg = np.concatenate((currImg, QMimg), axis=-1)
    currImg = util.uint2tensor(currImg)
    currImg = util.tensor3to4(currImg)

    start_time = time.time()
    if cuda:
        currImg = currImg.cuda()
    HQImg = model(currImg)
    HQImg = util.tensor2uint(HQImg)
    elapsed_time = time.time() - start_time
    gputime += elapsed_time

    HQImg = HQImg[0:ori_h, 0:ori_w, ...]
    if not os.path.exists(os.path.join('./results/', opt.save)):
        os.makedirs(os.path.join('./results/', opt.save))

    skimage.io.imsave(os.path.join('./results/', opt.save, img_name + '.png'), HQImg)
    currSpendTime = time.time() - currStartTime
    print(f'Time cost: {currSpendTime:.4f}s')

elapsed_time_all = time.time() - start_time_all
print(f'All time: {elapsed_time_all:.4f}s, primary time: {gputime:.4f}s')
print(f'============= END ==============')
