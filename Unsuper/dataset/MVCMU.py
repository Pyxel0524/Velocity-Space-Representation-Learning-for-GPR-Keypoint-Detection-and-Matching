import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from ..utils.utils import enhance
from scipy.stats import truncnorm

class MVCMU(Dataset):
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

    def __getitem__(self, idx):
        sample = int((self.upper - self.lower + 1) * self.beta_dist.sample((1,)))
        file_path = self.file_paths[idx]
        data = np.load(file_path)  # shape: [20, H, W]

        return data[0,:,:], data[sample,:,:]
        # import matplotlib.pyplot as plt
        # plt.figure();plt.imshow(data[0]);
        # plt.figure();plt.imshow(data[sample]);
        # plt.show()
