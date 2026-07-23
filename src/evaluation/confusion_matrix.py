import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class ConfusionMatrixPlotter:

    @staticmethod
    def plot(confusion_matrix, save_path="confusion_matrix.png"):

        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=["Down", "Stay", "Up"],
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        display.plot(
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()

        print(f"Saved confusion matrix to {save_path}")