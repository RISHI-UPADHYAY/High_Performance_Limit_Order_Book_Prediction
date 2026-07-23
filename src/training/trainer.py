import torch 
from tqdm import tqdm

from src.training.metrics import AverageMeter
from src.training.checkpoint import save_checkpoint
from src.training.logger import CSVLogger
from src.training.tensorboard_logger import TensorBoardLogger
from src.training.history import TrainingHistory

class Trainer:

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = CSVLogger("logs/metrics.py")
        self.tb = TensorBoardLogger()

    def train_epoch(self, loader):
        self.model.train()

        loss_meter = AverageMeter()

        correct = 0
        total = 0

        for X, y in tqdm(loader, desc="Training"):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(X)

            loss = self.criterion(logits, y)

            loss.backward()

            self.optimizer.step()

            loss_meter.update(loss.item(), X.size(0))

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()

            total += y.size(0)

        accuracy = correct / total
        return loss_meter.average, accuracy
    
    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        loss_meter = AverageMeter()

        correct = 0
        total = 0

        for X, y in tqdm(loader, desc="Validation"):
            X = X.to(self.device)
            y = y.to(self.device)


            logits = self.model(X)

            loss = self.criterion(logits, y)

            loss_meter.update(loss.item(), X.size(0))

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()

            total += y.size(0)

        accuracy = correct / total

        return loss_meter.average, accuracy
    
    def fit(self, train_loader, val_loader, epochs, scheduler=None, early_stopping=None):
        best_loss = float("inf")

        history = TrainingHistory()

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc = self.validate(val_loader)

            history.update(
                train_loss,
                train_acc,
                val_acc,
                val_loss,
            )

            print(f"Epoch {epoch+1} / {epochs}")

            print(f"Train Loss: {train_loss: 4f}" f" | Val Acc: {val_acc: 4f}")

            self.logger.log(
                epoch+1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            self.tb.log_scalars(
                epoch+1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if scheduler:
                scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    "best_model.pt",
                )

            if early_stopping:
                if early_stopping.step(val_loss):
                    print("Early stopping")
                    break
        self.tb.close()
        history.save_json("logs/history.json")
        history.save_csv("logs/history.csv")
                