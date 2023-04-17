import os
import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.metrics import SSIM, PSNR

def train_IKC(train_dl, valid_dl, len_train, sftmd, predictor, corrector, args):
    optimizer_p = optim.Adam(predictor.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']))
    optimizer_c = optim.Adam(corrector.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']))
    scheduler_p = lr_scheduler.StepLR(optimizer_p, step_size=args['step_size'], gamma=args['gamma'])
    scheduler_c = lr_scheduler.StepLR(optimizer_c, step_size=args['step_size'], gamma=args['gamma'])
    loss_fn = F.mse_loss
    num_epoch = args['num_epoch_IKC']
    num_iter = args['num_iter']
    device = args['device']
    log_interval = args['log_interval']
    
    ckpt_model_path_p = ''
    ckpt_model_path_c = ''
    best_val_p_loss = math.inf
    best_val_c_loss = math.inf
    
    for epoch in range(num_epoch):
        sftmd.to(device)
        sftmd.eval()
        predictor.to(device)
        predictor.train()
        corrector.to(device)
        corrector.train()
        train_p_loss = 0
        train_c_loss = 0
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
            
            # update predictor
            estimated_kernel = predictor(LR)
            loss_p = loss_fn(estimated_kernel, reduced_kernel)
            train_p_loss += loss_p
            loss_p.backward()
            optimizer_p.step()
            optimizer_p.zero_grad()
            
            SR = sftmd(LR, estimated_kernel.detach())
            # IKC Algorithm
            for i in range(num_iter):
                # update corrector
                delta_h = corrector(SR, estimated_kernel.detach())
                loss_c = loss_fn((estimated_kernel.detach() + delta_h), reduced_kernel)
                train_c_loss += loss_c
                loss_c.backward()
                optimizer_c.step()
                optimizer_c.zero_grad()
                
                estimated_kernel += delta_h.detach()
                SR = sftmd(LR, estimated_kernel.detach())

            if (batch_id + 1) % log_interval == 0:
                msg = "{}\tEpoch {}:[{}/{}]\ttrain [predictor: {:.5f}  corrector: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                    time.ctime(), epoch + 1, count, len_train,
                    train_p_loss / (batch_id + 1), train_c_loss / (batch_id + 1) / num_iter,
                    SSIM(SR, HR), PSNR(SR, HR)
                )
                print(msg)
        scheduler_p.step()
        scheduler_c.step()
        
        with torch.no_grad():
            predictor.eval()
            corrector.eval()
            valid_p_loss = 0
            valid_c_loss = 0
            ssim = 0
            psnr = 0
            for valid_data in valid_dl:
                LR = valid_data['LR']
                HR = valid_data['HR']
                reduced_kernel = valid_data['reduced_kernel']

                n_batch = len(LR)
                LR = LR.to(device)
                HR = HR.to(device)
                reduced_kernel = reduced_kernel.to(device)
                
                estimated_kernel = predictor(LR)
                loss_p = loss_fn(estimated_kernel, reduced_kernel)
                valid_p_loss += loss_p
                
                SR = sftmd(LR, estimated_kernel)
                # IKC Algorithm
                for i in range(num_iter):
                    delta_h = corrector(SR, estimated_kernel)
                    loss_c = loss_fn((estimated_kernel + delta_h), reduced_kernel)
                    valid_c_loss += loss_c

                    estimated_kernel += delta_h
                    SR = sftmd(LR, estimated_kernel)
                ssim += SSIM(SR, HR)
                psnr += PSNR(SR, HR)
            msg = "{}\tEpoch {}:\t\tvalid [predictor: {:.5f}  corrector: {:.5f}  ssim: {:.5f}  psnr: {:.5f}]".format(
                time.ctime(), epoch + 1, valid_p_loss / len(valid_dl), valid_c_loss / len(valid_dl) / num_iter,
                ssim / len(valid_dl), psnr / len(valid_dl)
            )
            print(msg)

            # save best model
            if (valid_p_loss / len(valid_dl) < best_val_p_loss):
                best_val_p_loss = valid_p_loss / len(valid_dl)
                print("predictor : new best validation loss!")
                predictor.eval().cpu()
                ckpt_model_filename_p = "ckpt_predictor_x" + str(args['scale']) + ".pth"
                ckpt_model_path_p = os.path.join(args['ckpt_dir'], ckpt_model_filename_p)
                torch.save({
                    'model_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer_p.state_dict(),
                    'total_loss': best_val_p_loss
                }, ckpt_model_path_p)
                
            if (valid_c_loss / len(valid_dl) < best_val_c_loss):
                best_val_c_loss = valid_c_loss / len(valid_dl)
                print("corrector : new best validation loss!")
                corrector.eval().cpu()
                ckpt_model_filename_c = "ckpt_corrector_x" + str(args['scale']) + ".pth"
                ckpt_model_path_c = os.path.join(args['ckpt_dir'], ckpt_model_filename_c)
                torch.save({
                    'model_state_dict': corrector.state_dict(),
                    'optimizer_state_dict': optimizer_c.state_dict(),
                    'total_loss': best_val_c_loss
                }, ckpt_model_path_c)