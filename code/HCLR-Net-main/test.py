import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import torchvision
import torch.nn as nn
import kornia.metrics.psnr as PSNR
import kornia.metrics.ssim as SSIM
import pytorch_lightning as pl
from train1 import CoolSystem

from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Unet_transmission_map(underwater_path,T_Unet):
    
    original = Image.open(underwater_path).convert('RGB')


    enhance_transforms = transforms.Compose([
    transforms.Resize((256,256), Image.BICUBIC),
    transforms.ToTensor()
    ])
    with torch.no_grad():
        original = enhance_transforms(original)
        original = original.cuda().unsqueeze(0)
        torch.cuda.synchronize()
        start = time.time()
        enhanced= T_Unet(original)
        torch.cuda.synchronize()
        end = time.time()
 
    return enhanced,original,end-start

if __name__ == '__main__':

    test_path=r'/share/zhangdan2013/code/datasets/UIEB_end/val/input'

    pth_path=r'/share/zhangdan2013/code/torch-nn/UCR-Net/epoch428-psnr23.192-ssim0.925-train_loss0.203-val_loss0.073.ckpt'

    checkpoint = torch.load(pth_path, map_location='cuda:0')
    print(checkpoint)
    args = {
        'epochs': 500,
        # datasetsw
        'train_datasets': r'/share/zhangdan2013/code/datasets/UIEB_end/train',
        'test_datasets': None,
        'val_datasets': r'/share/zhangdan2013/code/datasets/UIEB_end/val',
        # bs
        'train_bs': 16,
        # 'train_bs':4,
        'test_bs': 1,
        'val_bs': 8,
        # 'initlr':0.0002,
        'initlr': 0.0001,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 16,
        # Net
        'model_blocks': 5,
        'chns': 64
    }
    hparams = Namespace(**args)
    Model = CoolSystem(hparams)
    Model.load_state_dict(checkpoint['state_dict'])
    Model = Model.cuda()

    test_list =os.listdir(test_path)

    fps_avg = []
    for i,image in enumerate(test_list):
        print(image)
        enhanced,original,time_num = Unet_transmission_map(os.path.join(test_path,image),Model)
        torchvision.utils.save_image(enhanced, r'./test_results/' + image.replace('.tif', '.jpg'))
        fps_avg.append(1/time_num)
    print('avg fps:', np.mean(fps_avg))