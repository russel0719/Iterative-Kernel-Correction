import os
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA
from scipy import ndimage

def make_gaussian_kernels(scale:int, num:int):
    stddevs1 = []
    stddevs2 = []
    kernels = []
    for i in range(num):
        stddev1 = np.random.uniform(0.2, scale)
        stddev2 = np.random.uniform(0.2, scale)
        stddevs1.append(stddev1)
        stddevs2.append(stddev2)
        kernel_x = cv2.getGaussianKernel(21, stddev1)
        kernel_y = cv2.getGaussianKernel(21, stddev1)
        kernel = np.outer(kernel_x, kernel_y.T)
        rotate = np.random.uniform(0, 45)
        kernel = ndimage.rotate(kernel, rotate, reshape=False).reshape(-1)
        kernels.append(kernel)
        
    return stddevs1, stddevs2, kernels

def save_gaussian_kernels(
    kernel_path:str, reduced_kernel_dim:int, scale:int, stddevs1:list, stddevs2:list, kernels:list
):
    pca = PCA(n_components=reduced_kernel_dim)
    pca.fit(kernels)
    reduced_kernels = pca.transform(kernels)
    recon_kernels = pca.inverse_transform(reduced_kernels)
    
    kernel_dict = dict(
        pca = pca,
        kernels = kernels,
        stddevs1 = stddevs1,
        stddevs2 = stddevs2,
        reduced_kernels = reduced_kernels,
        recon_kernels = recon_kernels,
        scale = scale,
    )
    
    torch.save(
        kernel_dict,
        f"{kernel_path}/kernel_scale_x{scale}_dim21_diverse.pth"
    )
    
def make_and_save(args):
    stddevs1, stddevs2, kernels = make_gaussian_kernels(args['scale'], args['kernel_num'])
    save_gaussian_kernels(
        args['kernel_path'],args['reduced_kernel_dim'],
        args['scale'], stddevs1, stddevs2, kernels
    )
    print("creating kernel dict is complete")