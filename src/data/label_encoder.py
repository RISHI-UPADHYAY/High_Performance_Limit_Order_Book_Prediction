import pandas as pd

class LabelEncoder:
    mapping = {
        1:0,
        2:1,
        3:2,
    }

    @classmethod
    def encode(cls, labels: pd.Series) -> pd.Series:
        return labels.map(cls.mapping)