from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:

    def __init__(self, log_dir="logs/tensorboard"):

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, epoch, train_loss, train_acc, val_loss, val_acc):

        self.writer.add_scalar("Loss/Trainer", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)

        self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    def close(self):
        self.writer.close()