"""Chronological splits, fit-on-train scalers, and sequence DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import load_raw_csv
from src.features import add_features


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    input_size: int
    sequence_length: int


def _build_sequences(
    scaled_X: np.ndarray,
    scaled_y: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(scaled_X) - sequence_length
    X_list = []
    y_list = []
    for i in range(n):
        X_list.append(scaled_X[i : i + sequence_length])
        y_list.append(scaled_y[i + sequence_length])
    X_arr = np.asarray(X_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
    return X_arr, y_arr


def prepare_data_bundle(
    csv_path: str | None = None,
    sequence_length: int | None = None,
    batch_size: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> DataBundle:
    csv_path = csv_path or config.CSV_PATH
    sequence_length = sequence_length or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
    val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO

    df = load_raw_csv(csv_path)
    df = add_features(df)
    df = df.dropna(subset=config.FEATURE_COLS).reset_index(drop=True)

    n_rows = len(df)
    n_seq = n_rows - sequence_length
    if n_seq < 10:
        raise ValueError(
            f"Not enough rows after cleaning for sequences: n_seq={n_seq}"
        )

    n_train = int(n_seq * train_ratio)
    n_val = int(n_seq * val_ratio)
    n_test = n_seq - n_train - n_val
    if n_test < 1 or n_val < 1:
        raise ValueError(
            f"Bad split sizes: n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    # Fit scalers only on rows used by training sequences (no future leakage).
    fit_end = n_train + sequence_length
    raw_X = df[config.FEATURE_COLS].to_numpy(dtype=np.float64)
    raw_y = df[[config.TARGET_COL]].to_numpy(dtype=np.float64)

    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)
    scaler_X.fit(raw_X[:fit_end])
    scaler_y.fit(raw_y[:fit_end])

    scaled_X = scaler_X.transform(raw_X).astype(np.float32)
    scaled_y = scaler_y.transform(raw_y).astype(np.float32).ravel()

    X_arr, y_arr = _build_sequences(scaled_X, scaled_y, sequence_length)

    X_train = torch.from_numpy(X_arr[:n_train])
    y_train = torch.from_numpy(y_arr[:n_train])
    X_val = torch.from_numpy(X_arr[n_train : n_train + n_val])
    y_val = torch.from_numpy(y_arr[n_train : n_train + n_val])
    X_test = torch.from_numpy(X_arr[n_train + n_val :])
    y_test = torch.from_numpy(y_arr[n_train + n_val :])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        input_size=len(config.FEATURE_COLS),
        sequence_length=sequence_length,
    )
