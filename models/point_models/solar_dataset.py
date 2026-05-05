from torch.utils.data import Dataset
import torch

class SolarDataset(Dataset):
    def __init__(self, data, targets, seq_len, device='cpu', flatten=False):
        self.data = torch.tensor(data, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.seq_len = seq_len
        self.flatten = flatten
        self.valid = torch.arange(seq_len - 1, len(targets)).cpu().numpy()

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        t = self.valid[idx]
        seq = self.data[t - self.seq_len + 1 : t + 1]
        target = self.targets[t]

        if self.flatten:
            seq = seq.view(-1)

        return seq, target