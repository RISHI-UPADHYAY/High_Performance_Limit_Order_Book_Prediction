import torch

from src.models.deeplob import DeepLOB


def main():

    model = DeepLOB()

    x = torch.randn(8, 1, 100, 40)

    y = model(x)

    print("=" * 60)
    print("DeepLOB Test")
    print("=" * 60)

    print("Input :", x.shape)
    print("Output:", y.shape)

    assert y.shape == (8, 48, 100, 40)

    print("\n✓ Stage 2 Passed")


if __name__ == "__main__":
    main()