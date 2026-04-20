"""Load and sort raw CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {path}. Set EDP_CSV_PATH or add the Kaggle dataset input."
        )
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)
