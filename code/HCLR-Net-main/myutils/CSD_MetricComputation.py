from unicodedata import name
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import os
from torchvision import utils as vutils


def psnr1(original,enhanced):
    vutils.save_image(original,'a.png')
    original = cv2.imread('a.png')
    vutils.save_image(enhanced, 'a.png')
    enhanced = cv2.imread('a.png')
    # original = original.cpu().numpy()
    # enhanced = enhanced.cpu().numpy()
    original = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(enhanced,cv2.COLOR_BGR2GRAY)
    return peak_signal_noise_ratio(original,enhanced)
def ssim1(original,enhanced):
    vutils.save_image(original,'a.png')
    original = cv2.imread('a.png')
    vutils.save_image(enhanced, 'a.png')
    enhanced = cv2.imread('a.png')
    # original = original.cpu().numpy()
    # enhanced = enhanced.cpu().numpy()
    # original = np.array(original.getdata(), dtype='uint8')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return structural_similarity(original,enhanced,full=True, multichannel=True)[0]
#
# def ComputePSNR_SSIM(img_dir,gt_path):
#     error_list_ssim, error_list_psnr = [], []
#     for dir_path in img_dir:
#         enhanced_name = dir_path.split('\\')[-1]
#         gt_name = enhanced_name
#         enhanced = cv2.imread(dir_path)
#         gt = cv2.imread(os.path.join(gt_path,gt_name))
#         error_psnr = psnr(enhanced,gt)
#         error_ssim = ssim(enhanced,gt)
#         print(enhanced_name,error_psnr,error_ssim)
#         error_list_psnr.append(error_psnr)
#         error_list_ssim.append(error_ssim)
#     return np.array(error_list_ssim), np.array(error_list_psnr)
