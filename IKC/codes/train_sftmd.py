import os
import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.metrics import SSIM, PSNR

def train_sftmd(train_dl, valid_dl, len_train, sftmd, args):
    optimizer = optim.Adam(sftmd.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    loss_fn = F.mse_loss
    num_epoch = args['num_epoch_sftmd']
    device = args['device']
    log_interval = args['log_interval']
    
    ckpt_model_path = ''
    best_val_loss = math.inf
    
    for epoch in range(num_epoch):
        sftmd.to(device)
        sftmd.train()
        train_loss = 0
        count = 0
        
        for batch_id, train_data in enumerate(train_dl):
            LR = train_data['LR']
            HR = train_data['HR']
            reduced_kernel = train_data['reduced_kernel']
            
            n_batch = len(LR)
            count += n_batch
            
            LR = LR.to(device)
            HR = HR.to(device)
            reduced_kernel = reduced_kernel.to(device)
            
            SR = sftmd(LR, reduced_kernel)
            
            loss = loss_fn(SR, HR)
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (batch_id + 1) % log_interval == 0:
                msg = "{}\tEpoch {}:[{}/{}]\ttrain [mse: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                    time.ctime(), epoch + 1, count, len_train,
                    train_loss / (batch_id + 1), SSIM(SR, HR), PSNR(SR, HR)
                )
                print(msg)
        scheduler.step()
        
        with torch.no_grad():
            sftmd.eval()
            valid_loss = 0
            ssim = 0
            psnr = 0
            for valid_data in valid_dl:
                LR = valid_data['LR']
                HR = valid_data['HR']
                kernel = valid_data['kernel']
                reduced_kernel = valid_data['reduced_kernel']

                n_batch = len(LR)
                LR = LR.to(device)
                HR = HR.to(device)
                reduced_kernel = reduced_kernel.to(device)
                
                SR = sftmd(LR, reduced_kernel)

                loss = loss_fn(SR, HR)
                valid_loss += loss
                
                ssim += SSIM(SR, HR)
                psnr += PSNR(SR, HR)
            msg = "{}\tEpoch {}:\t\tvalid [mse: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                time.ctime(), epoch + 1, valid_loss / len(valid_dl), ssim / len(valid_dl), psnr / len(valid_dl)
            )
            print(msg)

            # save best model
            if (valid_loss / len(valid_dl) < best_val_loss):
                best_val_loss = valid_loss / len(valid_dl)
                print("sftmd : new best validation loss!")
                sftmd.eval().cpu()
                ckpt_model_filename = "ckpt_sftmd_x" + str(args['scale']) + ".pth"
                ckpt_model_path = os.path.join(args['ckpt_dir'], ckpt_model_filename)
                torch.save({
                    'model_state_dict': sftmd.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_loss': best_val_loss
                }, ckpt_model_path)