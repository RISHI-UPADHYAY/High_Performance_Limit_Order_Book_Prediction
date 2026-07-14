from src.data.dataset_loader import FI2010Parser
from src.data.dataset_inspector import DatasetInspector
from src.data.feature_extractor import FeatureExtractor

parser = FI2010Parser("datasets/FI2010")

df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

X = FeatureExtractor.extract(df)

print(X.head())
print(X.shape)