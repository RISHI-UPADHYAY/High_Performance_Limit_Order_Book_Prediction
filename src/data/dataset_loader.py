from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

class FI2010Parser:

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load(self, filename: str):

        file_path = self.dataset_path / filename

        if not file_path.exists():
            raise FileNotFoundError(file_path)
        
        logger.info(f"Loading {filename}")

        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

        logger.info(f"Raw Shape: {len(df)}")

        return df