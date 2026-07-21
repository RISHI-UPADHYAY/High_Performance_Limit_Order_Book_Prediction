import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config

from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.dataset_splitter import DatasetSplitter
from src.data.pytorch_dataset import LOBDataset

from src.models.deeplob import DeepLOB

from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping


def main():

    print("=" * 60)
    print("High Performance Limit Order Book Prediction")
    print("=" * 60)

    device = torch.device(config.DEVICE)

    print(f"\nUsing Device: {device}")

    # ======================================================
    # Dataset
    # ======================================================

    parser = FI2010Parser(config.DATA_DIR)

    df = parser.load(config.TRAIN_FILE)

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(
        df,
        horizon=config.PREDICTION_HORIZON,
    )

    generator = SlidingWindowGenerator(
        window_size=config.WINDOW_SIZE
    )

    windows, labels = generator.generate(X, y)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = DatasetSplitter.split(windows, labels)

    X_train = X_train[:2048]
    y_train = y_train[:2048]

    X_val = X_val[:512]
    y_val = y_val[:512]

    train_dataset = LOBDataset(X_train, y_train)
    val_dataset = LOBDataset(X_val, y_val)
    test_dataset = LOBDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # ======================================================
    # Model
    # ======================================================

    model = DeepLOB().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        scheduler=scheduler,
        early_stopping=early_stopping,
    )

    print("\nTraining Complete")


if __name__ == "__main__":
    main()