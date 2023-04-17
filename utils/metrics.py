import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity

def SSIM(estim, gt):
    estim = estim.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    maxval = gt.max()
    ssim = 0
    for idx in range(gt.shape[0]):
        ssim += structural_similarity(gt[idx], estim[idx], data_range=maxval, channel_axis=0)
    ssim /= gt.shape[0]
    return ssim

def PSNR(estim, gt):
    estim = estim.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    maxval = gt.max()
    mse = np.mean((estim - gt)**2)
    psnr = 10 * math.log10(maxval / mse)
    return psnr