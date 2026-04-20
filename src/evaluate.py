"""Metrics, inverse transform, and saved plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            acts.append(yb.cpu().numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(acts)
    return y_true, y_pred


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def inverse_transform_y(
    scaler_y, y: np.ndarray
) -> np.ndarray:
    return scaler_y.inverse_transform(
        np.asarray(y, dtype=np.float64).reshape(-1, 1)
    )


def save_prediction_plot(
    actuals: np.ndarray,
    predictions: np.ndarray,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or (config.ARTIFACTS_DIR / config.PLOT_NAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.plot(actuals.ravel(), label="Actual Demand")
    plt.plot(predictions.ravel(), label="Predicted Demand")
    plt.title("LSTM: Actual vs. Predicted Electricity Demand")
    plt.xlabel("Time Step")
    plt.ylabel("Electricity Demand (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def print_metrics(m: dict[str, float]) -> None:
    print("\n--- LSTM Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {m['mae']:.2f}")
    print(f"Mean Squared Error (MSE): {m['mse']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {m['rmse']:.2f}")
    print(f"R-squared (R2): {m['r2']:.2f}")
