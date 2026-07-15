from __future__ import annotations

import torch
from torch.utils.data import Dataset

class LOBDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        
        self.y = torch.tensor(y-1, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]