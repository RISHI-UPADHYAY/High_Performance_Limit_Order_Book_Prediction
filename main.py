from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator

def main():

    parser = FI2010Parser("datasets/FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(df, horizon="10")

    window_generator = SlidingWindowGenerator(window_size=100)

    windows, labels = window_generator.generate(X, y)

    print("\n============= Dataset Summary=========\n")

    print(f"Feature Matrix Shape: {X.shape}\n")
    print(f"Label Vector Shape: {y.shape}")

    print("\nSliding Window Shape:")

    print(windows.shape)
    print(labels.shape)
    

    print("\n=========================\n")

if __name__ == "__main__":
    main()