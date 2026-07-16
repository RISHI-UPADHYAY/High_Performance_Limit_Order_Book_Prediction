from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 100
NUM_FEAUTURES = 40
NUM_CLASSES = 3
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
NUM_WORKERS = 4
PIN_MEMEORY = True


DROPOUT = 0.20
HIDDEN_SIZE = 128