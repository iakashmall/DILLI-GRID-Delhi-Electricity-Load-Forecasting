"""
SHARED FEATURE ENGINEERING
==========================
One place that turns the raw hourly table into model-ready features,
used by training, export, and the FastAPI app so they can never drift apart.
"""

import numpy as np
import pandas as pd
import holidays

def build_features(df):
    """Turn the raw hourly table into model-ready features."""
    out = df.copy()
    idx = out.index

    # --- calendar features ---
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek            # 0 = Monday
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # cyclical encoding: tells the model that hour 23 is next to hour 0
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Indian public holidays (load behaves like a Sunday on these days)
    in_holidays = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    out["is_holiday"] = pd.Series(idx.date, index=idx).map(
        lambda d: 1 if d in in_holidays else 0).values

    # --- weather features ---
    # Cooling/heating "degree hours": AC load rises above ~24 C,
    # heating load rises below ~18 C. Very intuitive for load models.
    out["cooling_dh"] = np.clip(out["temp_c"] - 24, 0, None)
    out["heating_dh"] = np.clip(18 - out["temp_c"], 0, None)
    # Humid heat feels hotter -> more AC. Simple interaction term.
    out["temp_x_humidity"] = out["temp_c"] * out["humidity"] / 100

    # --- history (lag) features: the strongest predictors ---
    out["lag_24h"] = out["load_mw"].shift(24)      # same hour yesterday
    out["lag_168h"] = out["load_mw"].shift(168)    # same hour last week
    out["roll_mean_24h"] = out["load_mw"].shift(1).rolling(24).mean()

    return out.dropna()

FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos",
            "dayofweek", "is_weekend", "is_holiday",
            "temp_c", "humidity", "wind_kmh", "rain", "fog",
            "cooling_dh", "heating_dh", "temp_x_humidity",
            "lag_24h", "lag_168h", "roll_mean_24h"]
TARGET = "load_mw"

