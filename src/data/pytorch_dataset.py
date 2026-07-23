from __future__ import annotations

import torch
from torch.utils.data import Dataset

class LOBDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)

        if self.X.ndim == 3:
            self.X = self.X.unsqueeze(1)
        
        self.y = torch.tensor(y-1, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]