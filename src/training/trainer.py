import torch 
from tqdm import tqdm

from src.training.metrics import AverageMeter
from src.training.checkpoint import save_checkpoint

class Trainer:

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()

        loss_meter = AverageMeter()

        correct = 0
        total = 0

        for X, y in loader:
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

        for X, y in loader:
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

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc = self.validate(val_loader)

            print(f"Epoch {epoch+1} / {epochs}")

            print(f"Train Loss: {train_loss: 4f}" f" | Val Acc: {val_acc: 4f}")

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
                