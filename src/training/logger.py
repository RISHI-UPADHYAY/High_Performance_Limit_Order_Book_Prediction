from pathlib import Path
import csv

class CSVLogger:

    def __init__(self, filepath):
        self.filepath = Path(filepath)

        self.filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                ]
            )

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.filepath, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    epoch, 
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ]
            )