import pandas as pd

class FeatureExtractor:
    FEATURE_ROWS = 40

    @staticmethod
    def extract(df: pd.DataFrame) -> pd.DataFrame:

        #Extract the first 40 rows (LOB features)
        features = df.iloc[:FeatureExtractor.FEATURE_ROWS]

        #transpose
        features = features.T

        features.columns = [
            f"feature_{i+1}" for i in range(features.shape[1])
        ]

        return features.reset_index(drop=True)