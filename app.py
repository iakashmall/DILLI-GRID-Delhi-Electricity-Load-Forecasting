"""
app.py — FastAPI server for the Delhi Load Forecasting project.

Serves:
  GET  /                 -> the dashboard (outputs/dashboard.html)
  POST /api/predict      -> high-accuracy prediction using the FULL model
                            (XGBoost relative-target, needs recent load history)
  POST /api/predict_anytime -> prediction for ANY date/time using the
                            lag-free deploy model (same one the browser uses)

Run:
  uvicorn app:app --reload --port 8000
"""

import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
app = FastAPI(title="Delhi Load Forecast API")

# ---------------------------------------------------------------- load models
with open(ROOT / "models" / "best_model.pkl", "rb") as f:
    BEST = pickle.load(f)          # XGBoost trained on load / roll_mean_24h

with open(ROOT / "models" / "deploy_model.json") as f:
    DEPLOY = json.load(f)          # flattened trees, lag-free features

with open(ROOT / "outputs" / "climatology.json") as f:
    CLIMO = json.load(f)           # typical temp/humidity per (month, hour)

IN_HOLIDAYS = holidays_lib.India(years=range(2010, 2036))

# Hourly history (used to build lag features for the full model)
HIST = pd.read_csv(ROOT / "data" / "delhi_hourly.csv", parse_dates=["timestamp"])
HIST = HIST.set_index("timestamp").sort_index()


# ---------------------------------------------------------------- request bodies
class AnytimeRequest(BaseModel):
    """Forecast for any date/time. Weather is optional — climatology fills it."""
    date: str = Field(..., example="2026-06-15")
    hour: int = Field(..., ge=0, le=23)
    temp_c: float | None = None
    humidity: float | None = None
    growth_pct_per_year: float = 3.6   # demand growth vs the 2012 training era


class HistoryRequest(BaseModel):
    """High-accuracy forecast for a timestamp inside the historical dataset."""
    timestamp: str = Field(..., example="2012-06-15 15:00")


# ---------------------------------------------------------------- helpers
def deploy_features(dt: datetime, temp: float, hum: float) -> list[float]:
    """Build the 12 lag-free features in the exact training order."""
    hour, month = dt.hour, dt.month
    dow = dt.weekday()                       # Mon=0 like pandas
    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        dow,
        1.0 if dow >= 5 else 0.0,
        1.0 if dt.date() in IN_HOLIDAYS else 0.0,
        temp,
        hum,
        max(temp - 24, 0.0),
        max(18 - temp, 0.0),
        temp * hum,
    ]


def predict_deploy(x: list[float]) -> float:
    """Walk the flattened trees (mirrors the JavaScript in the dashboard)."""
    total = DEPLOY["base_score"]
    fr = np.float32
    for t in DEPLOY["trees"]:
        n = 0
        while t["feat"][n] >= 0:
            n = t["left"][n] if fr(x[t["feat"][n]]) < fr(t["thr"][n]) else t["right"][n]
        total += t["leaf"][n]
    return total


# ---------------------------------------------------------------- routes
@app.get("/")
def dashboard():
    return FileResponse(ROOT / "outputs" / "dashboard.html")


@app.post("/api/predict_anytime")
def predict_anytime(req: AnytimeRequest):
    try:
        dt = datetime.strptime(f"{req.date} {req.hour:02d}:00", "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    key = f"{dt.month}-{dt.hour}"
    typ_temp, typ_hum = CLIMO.get(key, [25.0, 55.0])
    temp = req.temp_c if req.temp_c is not None else typ_temp
    hum = req.humidity if req.humidity is not None else typ_hum

    base_mw = predict_deploy(deploy_features(dt, temp, hum))
    growth = (1 + req.growth_pct_per_year / 100) ** (dt.year - 2012)
    mw = base_mw * growth

    return {
        "timestamp": dt.isoformat(),
        "predicted_mw": round(mw, 1),
        "band_mw": [round(mw * 0.95, 1), round(mw * 1.05, 1)],
        "weather_used": {"temp_c": temp, "humidity": hum,
                         "from_climatology": req.temp_c is None},
        "growth_multiplier": round(growth, 3),
        "is_holiday": dt.date() in IN_HOLIDAYS,
        "model": "xgboost-anytime (lag-free), holdout MAPE 5.1%",
    }


@app.post("/api/predict")
def predict_full(req: HistoryRequest):
    """Best-accuracy model — needs the previous 7 days of real load."""
    try:
        ts = pd.Timestamp(req.timestamp)
    except ValueError:
        raise HTTPException(400, "timestamp must be parseable, e.g. 2012-06-15 15:00")

    if ts not in HIST.index:
        raise HTTPException(404, "timestamp outside the historical dataset "
                                 "(2010-03-29 .. 2013-03-10) — use /api/predict_anytime")

    # Build lag features from real history (same as training)
    from src.features import build_features, FEATURES
    window = HIST.loc[ts - pd.Timedelta(days=8): ts]
    feats = build_features(window)
    if ts not in feats.index or feats.loc[ts, FEATURES].isna().any():
        raise HTTPException(422, "not enough history before this timestamp")

    row = feats.loc[[ts]]
    X = row[BEST["features"]]
    if BEST["needs_scaling"]:
        X = BEST["scaler"].transform(X)
    ratio = float(BEST["model"].predict(X)[0])
    mw = ratio * float(row["roll_mean_24h"].iloc[0])

    return {
        "timestamp": str(ts),
        "predicted_mw": round(mw, 1),
        "actual_mw": round(float(HIST.loc[ts, "load_mw"]), 1),
        "model": "xgboost-relative (deployed), test MAPE 2.89%",
    }
