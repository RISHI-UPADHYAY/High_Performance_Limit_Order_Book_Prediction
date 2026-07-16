from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.pytorch_dataset import LOBDataset
from torch.utils.data import DataLoader
from src.data.dataset_splitter import DatasetSplitter

def main():

    parser = FI2010Parser("datasets/FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(df, horizon="10")

    window_generator = SlidingWindowGenerator(window_size=100)

    windows, labels = window_generator.generate(X, y)

    (X_train, y_train, X_val, y_val, X_test, y_test) = DatasetSplitter.split(windows, labels)

    train_dataset = LOBDataset(X_train, y_train)
    val_dataset = LOBDataset(X_val, y_val)
    test_dataset = LOBDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\nTrain batches: ", len(train_loader))
    print("Validation batches: ", len(val_loader))
    print("Test Batches: ", len(test_loader))

    print("\n=========================\n")

if __name__ == "__main__":
    main()