"""Training loop with validation, epoch-mean loss, and best checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import config
from src.dataset import prepare_data_bundle
from src.evaluate import (
    collect_predictions,
    inverse_transform_y,
    metrics_report,
    print_metrics,
    save_prediction_plot,
)
from src.model import LSTMModel
from src.utils import get_device, set_seed


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    set_seed(config.RANDOM_SEED)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = prepare_data_bundle()
    device = get_device()

    model = LSTMModel(
        input_size=bundle.input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        output_size=1,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    ckpt_path = Path(config.ARTIFACTS_DIR) / config.CHECKPOINT_NAME
    best_val = float("inf")

    print("Starting LSTM model training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = _run_epoch(
            model,
            bundle.train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )
        val_loss = _run_epoch(
            model,
            bundle.val_loader,
            criterion,
            None,
            device,
            train=False,
        )
        scheduler.step(val_loss)
        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], "
            f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": bundle.input_size,
                    "hidden_size": config.HIDDEN_SIZE,
                    "num_layers": config.NUM_LSTM_LAYERS,
                    "dropout": config.DROPOUT,
                },
                ckpt_path,
            )

    print("Training complete.\n")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    y_true_s, y_pred_s = collect_predictions(model, bundle.test_loader, device)
    y_true = inverse_transform_y(bundle.scaler_y, y_true_s)
    y_pred = inverse_transform_y(bundle.scaler_y, y_pred_s)

    m = metrics_report(y_true, y_pred)
    print_metrics(m)

    plot_path = save_prediction_plot(y_true, y_pred)
    print(f"Saved plot to {plot_path}")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
