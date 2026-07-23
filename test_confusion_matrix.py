from src.evaluation.confusion_matrix import ConfusionMatrixPlotter

import numpy as np


def main():

    matrix = np.array([
        [20, 2, 1],
        [3, 18, 2],
        [0, 4, 21],
    ])

    ConfusionMatrixPlotter.plot(matrix)

    print("=" * 50)
    print("CONFUSION MATRIX TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()