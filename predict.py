import torch
import torch.nn.functional as F

from src.models.deeplob import DeepLOB
from src.config import config

CLASS_NAMES = {
    0: "DOWN",
    1: "STATIONARY",
    2: "UP",
}

def main():
    
    print("=" * 60)
    print("DeepLOB Inference")
    print("=" * 60)

    device = torch.device(config.DEVICE)

    model = DeepLOB().to(device)

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    sample = torch.randn(
        1,
        1,
        config.WINDOW_SIZE,
        config.NUM_FEATURES,
    ).to(device)

    with torch.no_grad():

        logits = model(sample)

        probabilities = F.softmax(
            logits, 
            dim=1,
        )

        prediction = probabilities.argmax(dim=1).item()
    print()

    print("Prediction: ", CLASS_NAMES[prediction])

    print()

    print("Proabilities")

    for index, probability in enumerate(probabilities[0]):

        print(f"{CLASS_NAMES[index]:12s}: {probability.item():.4f}")


if __name__ == "__main__":
    main()