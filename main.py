from config import config

from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.dataset_splitter import DatasetSplitter

from src.data.feature_flattener import FeatureFlattener
from src.data.feature_scaler import FeatureScaler

from src.models.logistic_regression import LogisticRegressionModel

from src.training.evaluation import Evaluator


def main():

    print("=" * 60)
    print("High-Performance Limit Order Book Prediction Platform")
    print("=" * 60)

    parser = FI2010Parser(config.DATA_DIR / "FI2010")

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(
        df,
        horizon="10",
    )

    window_generator = SlidingWindowGenerator(
        window_size=config.WINDOW_SIZE,
    )

    windows, labels = window_generator.generate(X, y)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = DatasetSplitter.split(
        windows,
        labels,
    )

    print("\nFlattening windows...")

    flattener = FeatureFlattener()

    X_train = flattener.flatten(X_train)
    X_val = flattener.flatten(X_val)
    X_test = flattener.flatten(X_test)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    print("\nScaling features...")

    scaler = FeatureScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")

    model = LogisticRegressionModel()

    model.fit(
        X_train,
        y_train - 1,
    )

    print("\nPredicting...")

    predictions = model.predict(X_test)

    accuracy, matrix, report = Evaluator.evaluate(
        y_test - 1,
        predictions,
    )

    print("\n================ RESULTS ================")

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\nConfusion Matrix")

    print(matrix)

    print("\nClassification Report")

    print(report)

    print("=" * 60)


if __name__ == "__main__":
    main()