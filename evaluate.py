import torch
from torch.utils.data import DataLoader

from src.config import config

from src.data.dataset_loader import FI2010Parser
from src.data.feature_extractor import FeatureExtractor
from src.data.label_extractor import LabelExtractor
from src.data.sliding_window import SlidingWindowGenerator
from src.data.dataset_splitter import DatasetSplitter
from src.data.pytorch_dataset import LOBDataset

from src.models.deeplob import DeepLOB

from src.evaluation.evaluator import Evaluator
from src.evaluation.confusion_matrix import ConfusionMatrixPlotter


def main():

    print("=" * 60)
    print("DeepLOB Evaluation")
    print("=" * 60)

    device = torch.device(config.DEVICE)

    parser = FI2010Parser(config.DATA_DIR)

    df = parser.load("Train_Dst_NoAuction_DecPre_CF_7.txt")

    X = FeatureExtractor.extract(df)

    y = LabelExtractor.extract(
        df,
        horizon="10",
    )

    generator = SlidingWindowGenerator(
        window_size=config.WINDOW_SIZE,
    )

    windows, labels = generator.generate(X, y)

    (
        _,
        _,
        _,
        _,
        X_test,
        y_test,
    ) = DatasetSplitter.split(
        windows,
        labels,
    )

    test_dataset = LOBDataset(
        X_test,
        y_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    model = DeepLOB().to(device)

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)


            pred = logits.argmax(dim=1)

            predictions.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    metrics = Evaluator.evaluate(
        targets,
        predictions,
    )

    print("\nAccuracy :", metrics["accuracy"])
    print("\nPrecision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1 Score :", metrics["f1"])

    print("\nClassification Report\n")
    print(metrics["classification_report"])

    ConfusionMatrixPlotter.plot(
        metrics["confusion_matrix"],
        save_path="evaluation_confusion_matrix.png",
    )

    print("\nEvaluation Complete")


if __name__ == "__main__":
    main()