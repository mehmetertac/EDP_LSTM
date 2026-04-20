"""Rolling, lag, cyclical, and weather-derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rolling_mean_24"] = out["electricity_demand_mw"].rolling(24).mean()
    out["rolling_std_24"] = out["electricity_demand_mw"].rolling(24).std()
    out["temp_rolling_mean_24"] = out["temperature_c"].rolling(24).mean()
    out["temp_lag_24"] = out["temperature_c"].shift(24)

    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["day_of_year"] = out["datetime"].dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["lag_1"] = out["electricity_demand_mw"].shift(1)
    out["lag_24"] = out["electricity_demand_mw"].shift(24)

    out["humidity_decimal"] = out["humidity_pct"] / 100.0
    out["temperature_humidity_index"] = out["temperature_c"] - (
        0.55 * (1 - out["humidity_decimal"]) * (out["temperature_c"] - 14.5)
    )
    out["cloud_sol"] = out["cloud_cover"] * out["solar_irradiance_wm2"]

    return out
