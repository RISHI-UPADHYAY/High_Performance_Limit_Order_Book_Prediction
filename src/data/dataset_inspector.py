from src.data.dataset_info import DatasetInfo

class DatasetInspector:

    FEATURE_ROWS = 40

    @staticmethod
    def inspect(df):
        rows, cols = df.shape

        return DatasetInfo(
            total_rows=rows,
            total_columns=cols,
            feature_rows=40,
            label_rows=rows - 40,
            samples=cols,
        )