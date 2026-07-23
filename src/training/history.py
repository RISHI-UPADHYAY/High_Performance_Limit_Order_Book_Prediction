import csv
import json
from pathlib import Path

class TrainingHistory:

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(float(train_loss))
        self.history["train_acc"].append(float(train_acc))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_acc"].append(float(val_acc))

    def save_json(self, path):
        path = Path(path)

        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)

    def save_csv(self, path):
        path = Path(path)

        epochs = len(self.history["train_loss"])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ])
            for i in range(epochs):
                writer.writerow([
                    i+1,
                    self.history["train_loss"][i],
                    self.history["train_acc"][i],
                    self.history["val_loss"][i],
                    self.history["val_acc"][i],
                ])