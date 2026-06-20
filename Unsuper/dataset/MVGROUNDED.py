import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from ..utils.utils import enhance
from scipy.stats import truncnorm

class MVGROUNDED(Dataset):
    def __init__(self, opt):
        self.root_dir = opt.data_path
        self.file_paths = sorted(glob(os.path.join(self.root_dir, '**', '*.npy'), recursive=True))
        self.config = opt

        # beta sample
        self.alpha = 5
        self.beta = 2
        self.lower = 0
        self.upper = 19
        self.beta_dist = torch.distributions.Beta(self.alpha, self.beta)

    def __len__(self):
        return len(self.file_paths)

    def norm(self, x):
        # x: numpy array, shape [C, H, W]
        min_val = x.reshape(x.shape[0], -1).min(axis=1).reshape(-1, 1, 1)
        max_val = x.reshape(x.shape[0], -1).max(axis=1).reshape(-1, 1, 1)

        x_norm = (x - min_val) / (max_val - min_val)  # [0, 1]
        x_norm = x_norm * 2 - 1  # [-1, 1]
        return x_norm

    def __getitem__(self, idx):
        sample = int((self.upper - self.lower + 1) * self.beta_dist.sample((1,)))
        file_path = self.file_paths[idx]
        data = self.norm(np.load(file_path))  # shape: [20, H, W]
        return  data[0,:,:], data[sample,:,:]
