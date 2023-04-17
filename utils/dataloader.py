import torch
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size:int, shuffle:bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)