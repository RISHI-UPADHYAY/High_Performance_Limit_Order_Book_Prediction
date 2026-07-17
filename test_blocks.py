import torch

from src.models.blocks.conv_block import ConvBlock
from src.models.blocks.inception_block import InceptionBlock


def main():

    # Batch = 8
    # Channels = 1
    # Window = 100
    # Features = 40

    x = torch.randn(8, 1, 100, 40)

    print("=" * 60)
    print("Testing ConvBlock")
    print("=" * 60)

    conv = ConvBlock(
        in_channels=1,
        out_channels=32,
    )

    conv_out = conv(x)

    print("Input Shape : ", x.shape)
    print("Output Shape:", conv_out.shape)

    assert conv_out.shape == (8, 32, 100, 40)

    print("\n✓ ConvBlock Passed")

    print("\n" + "=" * 60)
    print("Testing InceptionBlock")
    print("=" * 60)

    inception = InceptionBlock(
        in_channels=32,
        out_channels=48,
    )

    inception_out = inception(conv_out)

    print("Input Shape : ", conv_out.shape)
    print("Output Shape:", inception_out.shape)

    assert inception_out.shape == (8, 48, 100, 40)

    print("\n✓ InceptionBlock Passed")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()