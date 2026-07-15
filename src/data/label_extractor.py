import pandas as pd

class LabelExtractor:
    LABEL_ROWS = {
        "10": 144,
        "20": 145,
        "30": 146,
        "50": 147,
        "100": 148,
    }

    @classmethod
    def extract(cls, df: pd.DataFrame, horizon="10"):
        
        if horizon not in cls.LABEL_ROWS:
            raise ValueError(f"Unknown horizon: {horizon}")

        row = cls.LABEL_ROWS[horizon]

        labels = df.iloc[row].astype(int)

        labels.name = f"label_{horizon}"

        return labels.reset_index(drop=True)