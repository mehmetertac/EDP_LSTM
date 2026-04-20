"""Paths, hyperparameters, and feature column list."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

# Default matches Kaggle when the dataset is added as an input.
DEFAULT_CSV = (
    "/kaggle/input/smart-city-energy-dataset/smart_city_energy_dataset.csv"
)
CSV_PATH = os.environ.get("EDP_CSV_PATH", DEFAULT_CSV)

RANDOM_SEED = 42
SEQUENCE_LENGTH = 24
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# test = remainder

TARGET_COL = "electricity_demand_mw"

# Multivariate inputs per timestep (includes current demand and engineered features).
FEATURE_COLS = [
    TARGET_COL,
    "rolling_mean_24",
    "rolling_std_24",
    "temp_rolling_mean_24",
    "temp_lag_24",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_24",
    "humidity_decimal",
    "temperature_humidity_index",
    "cloud_sol",
]

CHECKPOINT_NAME = "best_lstm.pt"
PLOT_NAME = "pred_vs_actual.png"
