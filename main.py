from torch.utils.data import DataLoader

from src.config import settings

from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.dataset_splitter import DatasetSplitter
from src.data.pytorch_dataset import LOBDataset


def main():

    parser = FI2010Parser(settings.DATA_DIR / "FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(
        df,
        horizon="10",
    )

    window_generator = SlidingWindowGenerator(
        window_size=settings.WINDOW_SIZE
    )

    windows, labels = window_generator.generate(
        X,
        y,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = DatasetSplitter.split(
        windows,
        labels,
        train_ratio=settings.TRAIN_RATIO,
        val_ratio=settings.VALIDATION_RATIO,
    )

    train_dataset = LOBDataset(
        X_train,
        y_train,
    )

    val_dataset = LOBDataset(
        X_val,
        y_val,
    )

    test_dataset = LOBDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )

    print("\n========== CONFIG ==========")

    print(f"Window Size : {settings.WINDOW_SIZE}")
    print(f"Batch Size  : {settings.BATCH_SIZE}")
    print(f"Epochs      : {settings.EPOCHS}")
    print(f"Device      : {settings.DEVICE}")

    print("\n======= DATA SUMMARY =======")

    print(f"Train Samples      : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")
    print(f"Test Samples       : {len(test_dataset)}")

    print(f"\nTrain Batches      : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")
    print(f"Test Batches       : {len(test_loader)}")

    print("\n============================\n")


if __name__ == "__main__":
    main()