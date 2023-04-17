import numpy as np
import torch
from torchvision import transforms

def img2tensor(img):
    tf = transforms.ToTensor()
    return tf(img)

def tensor2imgtensor(tensor):
    return tensor.clamp_(0, 1).permute(1, 2, 0)

def imgtensor2img(imgtensor):
    return (255 * imgtensor.numpy()).astype(np.uint8)

def double2tensor(data):
    return torch.tensor(data, dtype=torch.float32)