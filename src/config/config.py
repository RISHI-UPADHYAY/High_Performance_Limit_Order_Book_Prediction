from pathlib import Path
import random

import numpy as np
import torch


# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datasets" / "FI2010"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"


# Create directories automatically
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Dataset
# ==========================================================

TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILE = "Test_Dst_NoAuction_DecPre_CF_7.txt"

WINDOW_SIZE = 100
NUM_FEATURES = 40
NUM_CLASSES = 3

PREDICTION_HORIZON = "10"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 1

WEIGHT_DECAY = 1e-5

PATIENCE = 5

RANDOM_SEED = 42


# ==========================================================
# Hardware
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_WORKERS = 0
PIN_MEMORY = True


# ==========================================================
# DeepLOB
# ==========================================================

DROPOUT = 0.20
HIDDEN_SIZE = 128


# ==========================================================
# Logistic Regression Baseline
# ==========================================================

LOGISTIC_MAX_ITER = 1000
LOGISTIC_SOLVER = "lbfgs"


# ==========================================================
# Reproducibility
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)