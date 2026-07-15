from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.pytorch_dataset import LOBDataset
from torch.utils.data import DataLoader

def main():

    parser = FI2010Parser("datasets/FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(df, horizon="10")

    window_generator = SlidingWindowGenerator(window_size=100)

    windows, labels = window_generator.generate(X, y)

    dataset = LOBDataset(windows, labels)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("\nDataset Size:")
    print(len(dataset))
    print("\nNumber of Batches:")
    print(len(loader))
    batch_x, batch_y = next(iter(loader))

    print("\nBatch Shapes")
    print(batch_x.shape)
    print(batch_y.shape)
    

    print("\n=========================\n")

if __name__ == "__main__":
    main()