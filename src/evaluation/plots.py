import matplotlib.pyplot as plt

class TrainingPlotter:

    @staticmethod
    def plot(history):

        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(8, 5))

        plt.plot(epochs, history["train_loss"], label="Train", linewidth=2)

        plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.legend()

        plt.tight_layout()

        plt.savefig("training_loss.png")

        plt.close()

        #Accuracy
        plt.figure(figsize=(8, 5))

        plt.plot(epochs, history["train_acc"], label="Train", linewidth=2)
        plt.plot(epochs, history["val_acc"], label="Validation", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.legend()
        plt.tight_layout()

        plt.savefig("training_accuracy.png")

        plt.close()

        print("Saved training curves.")
