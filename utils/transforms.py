# transforms.py


import numpy as np
import torch


class np_to_tensor():

    def __call__(self, x: np.ndarray):
        x = x.astype('float')
        return torch.tensor(x).float()
