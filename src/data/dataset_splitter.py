from __future__ import annotations

class DatasetSplitter:

    @staticmethod
    def split(X, y, train_ratio=0.70, val_ratio=0.15):
        n = len(X)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )