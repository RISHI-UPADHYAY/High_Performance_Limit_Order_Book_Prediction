from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor

def main():

    parser = FI2010Parser("datasets/FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(df, horizon="10")

    print("\n============= Dataset Summary=========\n")

    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Label Vector Shape: {y.shape}")

    print("\nFirst Five Labels")
    print(y.head())

    print("\nLabel Distribution:")
    print(y.value_counts().sort_index())

    print("\n=========================\n")

if __name__ == "__main__":
    main()