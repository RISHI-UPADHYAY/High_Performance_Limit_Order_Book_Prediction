# High-Performance Limit Order Book Prediction Platform

A production-oriented implementation of a Deep Learning pipeline for **Limit Order Book (LOB) mid-price movement prediction** using the **FI-2010 benchmark dataset**.

The project implements an end-to-end machine learning workflow including:

- Dataset preprocessing
- Sliding-window generation
- DeepLOB neural network
- Training pipeline
- Evaluation
- Inference
- Model checkpointing
- Visualization

The goal is to build a research-quality prediction system similar to those used in quantitative trading firms for market microstructure analysis.

---

# Features

## Data Pipeline

- FI-2010 dataset parser
- Feature extraction
- Label extraction
- Sliding window generation
- Dataset splitting
- PyTorch Dataset implementation
- Automatic preprocessing pipeline

---

## Deep Learning Model

Current implementation:

- CNN Feature Extractor
- Inception Module
- LSTM Sequence Model
- Fully Connected Classification Head

Predicts:

- DOWN
- STATIONARY
- UP

---

## Training Pipeline

Implemented training engine including

- Mini-batch training
- Validation
- CrossEntropy Loss
- AdamW Optimizer
- ReduceLROnPlateau Scheduler
- Early Stopping
- Model Checkpoint Saving
- Accuracy Monitoring

---

## Evaluation

Complete evaluation pipeline with

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Inference

Supports inference from trained checkpoints.

Example output:

```
Prediction: STATIONARY

Probabilities

DOWN        : 0.2645
STATIONARY  : 0.4756
UP          : 0.2598
```

---

# Project Structure

```
src/
│
├── config/
├── data/
│   ├── dataset_loader.py
│   ├── feature_extractor.py
│   ├── sliding_window.py
│   ├── dataset_splitter.py
│   ├── pytorch_dataset.py
│   └── ...
│
├── models/
│   ├── deeplob.py
│   ├── blocks/
│   │   ├── conv_block.py
│   │   ├── inception_block.py
│   │   └── lstm_block.py
│
├── training/
│   ├── trainer.py
│   ├── checkpoint.py
│   ├── metrics.py
│   └── early_stopping.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── inference.py
│   ├── confusion_matrix.py
│   └── plots.py
│
└── utils/
```

---

# Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Loguru

---

# Dataset

This project uses the

**FI-2010 Limit Order Book Benchmark Dataset**

containing

- Multiple Nordic stocks
- 10 levels of bid/ask prices
- 10 levels of bid/ask volumes
- Mid-price movement labels

---

# Training

Run

```bash
python train.py
```

Example output

```
Epoch 1 / 1

Train Loss : 1.0596

Validation Accuracy : 54.88%
```

---

# Evaluation

Run

```bash
python evaluate.py
```

Outputs

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# Inference

Run

```bash
python predict.py
```

Produces

```
Prediction

DOWN
STATIONARY
UP
```

along with prediction probabilities.

---

# Current Status

Implemented

- End-to-end preprocessing pipeline
- DeepLOB architecture
- Training engine
- Evaluation pipeline
- Inference pipeline
- Model checkpointing

---

# Roadmap

Planned improvements include

- Transformer-based LOB encoder
- DeepLOB++ architecture
- DeepLOB + Attention
- DeepLOB + Temporal CNN
- Multi-horizon prediction
- Hyperparameter optimization
- Mixed Precision Training
- CUDA optimization
- ONNX / TorchScript deployment
- Real-time inference engine
- NASDAQ ITCH parser
- Binance order book streaming
- Feature engineering for HFT
- Model explainability
- Experiment tracking
- Docker deployment
- CI/CD pipeline
- Comprehensive benchmarking

---

# Research Inspiration

The implementation is inspired by the DeepLOB architecture proposed in

> Zhang, Zohren & Roberts
> "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"

while being developed as an extensible production-oriented research platform.

---



Building quantitative finance, machine learning, and high-performance systems.