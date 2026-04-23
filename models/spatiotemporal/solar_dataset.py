import torch
from torch.utils.data import Dataset
import numpy as np

class SpatioTemporalDataset(Dataset):
    def __init__(self, data, targets, future_sza, seq_len, flatten=False):
        self.data = data
        self.targets = targets
        self.future_sza = future_sza
        self.seq_len = seq_len
        self.flatten = flatten

        valid = np.arange(len(targets))
        self.valid = valid[valid >= seq_len - 1]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        t = self.valid[idx]
        seq = self.data[t - self.seq_len + 1 : t + 1]
        target = self.targets[t]

        if self.flatten:
            seq = seq.flatten()

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
