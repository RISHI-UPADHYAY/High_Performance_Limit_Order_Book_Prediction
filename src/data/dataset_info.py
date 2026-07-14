from dataclasses import dataclass

@dataclass
class DatasetInfo:
    total_rows: int
    total_columns: int
    feature_rows: int
    label_rows: int
    samples: int