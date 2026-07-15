from __future__ import annotations

import numpy as np
import pandas as pd

class SlidingWindowGenerator:

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def generate(self, X: pd.DataFrame, y: pd.Series):
        
        X = X.to_numpy(dtype=np.float32)
        y = y.to_numpy(dtype=np.int64)

        samples = len(X) - self.window_size + 1

        windows = np.zeros(
            (
                samples,
                self.window_size,
                X.shape[1],
            ),
            dtype=np.float32,
        )

        labels = np.zeros(samples, dtype=np.int64)

        for i in range(samples):
            windows[i] = X[i:i+self.window_size]
            labels[i] = y[i+self.window_size-1]
        
        return windows, labels