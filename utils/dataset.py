import os, glob
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision import datasets

from utils.utils import img2tensor, double2tensor

SEED = 0
KERNEL_SIZE = 21
PATCH_SIZE = 144

class ImageKernelDataset(torch.utils.data.Dataset): 
    def __init__(self, imgs:list, kernel_path:str, scale:int, train:bool):
        self.imgs = imgs
        self.scale = scale
        self.train = train
        self.random = np.random.RandomState(SEED)
        
        self.kernel_dict = torch.load(f"{kernel_path}/kernel_scale_x{scale}_dim21_diverse.pth")
        self.kernels = self.kernel_dict['kernels']
        self.reduced_kernels = self.kernel_dict['reduced_kernels']
        self.stddevs1 = self.kernel_dict['stddevs1']
        self.stddevs2 = self.kernel_dict['stddevs2']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.train:
            # randomly patch image
            img_from, img_to = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
            for i in range(2):
                img_from[i] = self.random.randint(0, img.shape[i] - (KERNEL_SIZE + 1 + PATCH_SIZE))
                img_to[i] = img_from[i] + KERNEL_SIZE + 1 + PATCH_SIZE
            img_patch = img[img_from[0]:img_to[0], img_from[1]:img_to[1]]
        else:
            img_patch = img
            
        # augmentation
        img_aug = img_patch
        half = KERNEL_SIZE // 2 + 1
        if self.train:
            aug1 = self.random.randint(2)
            aug2 = self.random.randint(3)
            if aug1 == 1:
                img_aug = cv2.flip(img_aug, 0)
            if aug2 == 1:
                img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_CLOCKWISE)
            elif aug2 == 2:
                img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_hr = img_aug[half:-half, half:-half]
        else:
            img_hr = img_aug
                
        # apply kernel
        kernel_idx = self.random.randint(len(self.kernels))
        kernel = self.kernels[kernel_idx].reshape(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)
        reduced_kernel = self.reduced_kernels[kernel_idx].astype(np.float32)
        stddev1 = self.stddevs1[kernel_idx]
        stddev2 = self.stddevs2[kernel_idx]
        
        img_blur = cv2.filter2D(img_aug, ddepth=-1, kernel=kernel)
        
        if self.train:
            img_blur = img_blur[half:-half, half:-half]
            img_lr = cv2.resize(img_blur, (PATCH_SIZE//self.scale, PATCH_SIZE//self.scale), cv2.INTER_NEAREST)
        else:
            filename = f"{filename[:-6]}LR.png"
            img_lr = cv2.imread(filename, cv2.IMREAD_COLOR)
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        
        # make dict
        data_dict = dict(
            HR = img2tensor(img_hr),
            LR = img2tensor(img_lr),
            kernel = kernel,
            reduced_kernel = reduced_kernel,
            stddev1 = stddev1,
            stddev2 = stddev2
        )
        return data_dict

class TestImageDataset(torch.utils.data.Dataset): 
    def __init__(self, imgs_hr:list, imgs_lr:list):
        self.imgs_hr = imgs_hr
        self.imgs_lr = imgs_lr

    def __len__(self):
        return len(self.imgs_hr)

    def __getitem__(self, idx):
        filename_hr = self.imgs_hr[idx]
        filename_lr = self.imgs_lr[idx]
        img_hr = cv2.imread(filename_hr, cv2.IMREAD_COLOR)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.imread(filename_lr, cv2.IMREAD_COLOR)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        
        data_dict = dict(
            HR = img2tensor(img_hr),
            LR = img2tensor(img_lr),
        )
        return data_dict
    
def get_datasets(args):
    test_path = f"{args['test']}/image_SRF_{args['scale']}/" 
    train_imgs = glob.glob(args['train'] + "/**/*.png", recursive=True)
    valid_imgs = glob.glob(args['valid'] + "/**/*.png", recursive=True)
    test_imgs_hr = glob.glob(test_path + "/*HR.png", recursive=True)
    test_imgs_lr = glob.glob(test_path + "/*LR.png", recursive=True)
    
    train_dataset = ImageKernelDataset(
        imgs=train_imgs, kernel_path=args['kernel_path'], scale=args['scale'], train=True
    )
    valid_dataset = ImageKernelDataset(
        imgs=valid_imgs, kernel_path=args['kernel_path'], scale=args['scale'], train=True
    )
    test_dataset = TestImageDataset(
        imgs_hr=test_imgs_hr, imgs_lr=test_imgs_lr
    )
    
    return train_dataset, valid_dataset, test_dataset