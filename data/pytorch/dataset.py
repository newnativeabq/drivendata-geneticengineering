#dataset.py

"""
Given aligned x, y, produce a dataset that can feed pytorch models
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader



class QuickDataset(Dataset):
    """ Root Dataset for simple x, y """

    def __init__(self, x, y, transforms=None):
        self.x = x 
        self.y = y
        self.transform = Compose(transforms=transforms)


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.transform(self.x[idx])
        y = self.y[idx]

        return x, y




class Compose():
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            if t is not None:
                x = t(x)
        return x