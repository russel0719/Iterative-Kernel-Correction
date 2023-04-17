import time
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.plot_image import plot
from utils.metrics import SSIM, PSNR
from utils.utils import tensor2imgtensor, imgtensor2img

def test(test_dl, len_test, sftmd, predictor, corrector, args):
    loss_fn = F.mse_loss
    num_iter = args['num_iter']
    device = args['device']
    
    sftmd.to(device)
    predictor.to(device)
    corrector.to(device)
    sftmd.eval()
    predictor.eval()
    corrector.eval()
    count = 0
    
    total_mse = 0
    total_ssim = 0
    total_psnr = 0
    
    with torch.no_grad():
        start_time = time.time()
        for test_data in test_dl:
            LR = test_data['LR']
            HR = test_data['HR']

            n_batch = len(LR)
            count += n_batch

            LR = LR.to(device)
            HR = HR.to(device)

            estim_kernel = predictor(LR)
            SR = sftmd(LR, estim_kernel)
            for i in range(num_iter):
                delta_h = corrector(SR, estim_kernel)
                estim_kernel += delta_h
                SR = sftmd(LR, estim_kernel)

            mse = loss_fn(SR, HR)
            ssim = SSIM(SR, HR)
            psnr = PSNR(SR, HR)
            total_mse += mse
            total_ssim += ssim
            total_psnr += psnr

            msg = "{}\t[{}/{}]\ttest [mse: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                time.ctime(), count, len_test, mse, ssim, psnr
            )
            print(msg)
            result_lr = tensor2imgtensor(LR[0].cpu())
            result_sr = tensor2imgtensor(SR[0].cpu())
            result_hr = tensor2imgtensor(HR[0].cpu())
            img = imgtensor2img(result_sr)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"./IKC/results/{args['test'][12:]}_{count}_x{args['scale']}.jpg", img)
            plot([result_lr, result_sr, result_hr], ["LR", "SR", "HR"], (1, 3))
        end_time = time.time()
    
    print(
        "avg mse: {:.5f}\navg ssim: {:.5f}\navg psnr: {:.5f}\ntotal time: {:.5f} sec".format(
            total_mse/len_test, total_ssim/len_test, total_psnr/len_test, end_time - start_time
        )
    )