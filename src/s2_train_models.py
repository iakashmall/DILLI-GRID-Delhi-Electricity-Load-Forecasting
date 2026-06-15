"""
STEP 2 — TRAIN & COMPARE MODELS
===============================
Builds features, trains every candidate model, and evaluates each one on:
  * the full test set, AND
  * "hard" subsets: peak-load hours, heatwave hours, and statistical outlier days

so that we pick the model that is best EVEN in outlier conditions.

Models compared:
  0. Naive (yesterday same hour)      -> baseline everyone must beat
  1. Naive (last week same hour)      -> baseline
  2. Linear Regression
  3. Ridge Regression
  4. Random Forest
  5. XGBoost (gradient boosted trees)
  6. Neural Network (MLP)

Output: outputs/metrics.json, outputs/test_predictions.csv, models/best_model.pkl
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RNG = 42

from s0_features import build_features, FEATURES, TARGET

# ---------------------------------------------------------------
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT (never shuffle time series!)
# ---------------------------------------------------------------
df = pd.read_csv("data/delhi_hourly.csv", parse_dates=["timestamp"],
                 index_col="timestamp")
df = build_features(df)

split = "2012-03-11"               # train ~2 years, test = 1 full year (covers summer heatwaves + winter)
train, test = df[df.index < split], df[df.index >= split]
X_tr, y_tr = train[FEATURES], train[TARGET]
X_te, y_te = test[FEATURES], test[TARGET]
print(f"train: {len(train)} rows | test: {len(test)} rows\n")

# ---------------------------------------------------------------
# 3. EVALUATION HELPERS  (incl. outlier conditions)
# ---------------------------------------------------------------
def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)

# "hard" test subsets ------------------------------------------------
peak_mask = y_te >= y_te.quantile(0.95)                  # top 5 % load hours
hot_mask = test["temp_c"] >= 40                          # heatwave hours
daily_mean = y_te.groupby(y_te.index.date).mean()
z = np.abs((daily_mean - daily_mean.mean()) / daily_mean.std())
outlier_days = set(daily_mean[z > 2].index)              # statistically unusual days
outday_mask = pd.Series(y_te.index.date, index=y_te.index).isin(outlier_days)

def evaluate(name, pred):
    pred = np.asarray(pred)
    return {
        "model": name,
        "mape": round(mape(y_te, pred), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 1),
        "mae": round(float(mean_absolute_error(y_te, pred)), 1),
        "r2": round(float(r2_score(y_te, pred)), 4),
        "mape_peak_hours": round(mape(y_te[peak_mask], pred[peak_mask]), 2),
        "mape_heatwave": round(mape(y_te[hot_mask], pred[hot_mask]), 2) if hot_mask.sum() else None,
        "mape_outlier_days": round(mape(y_te[outday_mask], pred[outday_mask.values]), 2),
    }

results, predictions = [], {}

# ---------------------------------------------------------------
# 4. BASELINES
# ---------------------------------------------------------------
results.append(evaluate("Naive (24h ago)", X_te["lag_24h"]))
predictions["Naive (24h ago)"] = X_te["lag_24h"].values
results.append(evaluate("Naive (168h ago)", X_te["lag_168h"]))

# ---------------------------------------------------------------
# 5. ML MODELS
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_tr)          # needed by linear & neural models

models = {
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=10.0), True),
    "Random Forest": (RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=4,
        n_jobs=-1, random_state=RNG), False),
    "XGBoost": (XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=7,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:absoluteerror",   # MAE objective = robust to outliers
        n_jobs=-1, random_state=RNG), False),
    "Neural Network (MLP)": (MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=400,
        early_stopping=True, random_state=RNG), True),
}

fitted = {}
for name, (model, needs_scaling) in models.items():
    print(f"training {name} ...")
    Xa, Xb = (scaler.transform(X_tr), scaler.transform(X_te)) if needs_scaling else (X_tr, X_te)
    model.fit(Xa, y_tr)
    pred = model.predict(Xb)
    fitted[name] = model
    predictions[name] = pred
    results.append(evaluate(name, pred))

# ---------------------------------------------------------------
# 5b. XGBoost (relative target) — the extrapolation fix
# ---------------------------------------------------------------
# Trees cannot predict values HIGHER than anything seen in training,
# so plain XGBoost under-predicts record-breaking summer peaks.
# Fix: predict the RATIO  load / rolling-24h-average  (a stationary
# quantity around 1.0), then multiply back. Demand can grow year after
# year and the model stays valid.
print("training XGBoost (relative target) ...")
ratio_tr = y_tr / X_tr["roll_mean_24h"]
xgb_rel = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=7,
                       subsample=0.9, colsample_bytree=0.9,
                       objective="reg:absoluteerror", n_jobs=-1, random_state=RNG)
xgb_rel.fit(X_tr, ratio_tr)
pred_rel = xgb_rel.predict(X_te) * X_te["roll_mean_24h"].values
fitted["XGBoost (relative)"] = xgb_rel
predictions["XGBoost (relative)"] = pred_rel
results.append(evaluate("XGBoost (relative)", pred_rel))
models["XGBoost (relative)"] = (xgb_rel, False)

# ---------------------------------------------------------------
# 6. PICK THE WINNER — accuracy overall AND in outlier conditions
# ---------------------------------------------------------------
# Score = average of normal MAPE and the hard-condition MAPEs.
def combined(r):
    hard = [r["mape_peak_hours"], r["mape_outlier_days"]]
    if r["mape_heatwave"] is not None:
        hard.append(r["mape_heatwave"])
    return 0.5 * r["mape"] + 0.5 * float(np.mean(hard))

ml_results = [r for r in results if not r["model"].startswith("Naive")]
best = min(ml_results, key=combined)
best_name = best["model"]
print("\n=== RESULTS (sorted by combined score) ===")
for r in sorted(results, key=combined):
    print(f"{r['model']:24s} MAPE {r['mape']:5.2f}%  peak {r['mape_peak_hours']:5.2f}%  heat {r['mape_heatwave'] or 0:5.2f}%  "
          f"outlier-days {r['mape_outlier_days']:5.2f}%  R2 {r['r2']}")
print(f"\nBEST MODEL: {best_name}")

# ---------------------------------------------------------------
# 7. SAVE EVERYTHING THE DASHBOARD / API NEEDS
# ---------------------------------------------------------------
with open("outputs/metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_name,
               "split_date": split,
               "n_train": len(train), "n_test": len(test)}, f, indent=2)

out = pd.DataFrame({"timestamp": y_te.index, "actual": y_te.values,
                    "predicted": predictions[best_name],
                    "temp_c": test["temp_c"].values})
out.to_csv("outputs/test_predictions.csv", index=False)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"model": fitted[best_name], "scaler": scaler,
                 "features": FEATURES, "needs_scaling": models[best_name][1],
                 "name": best_name}, f)

# Feature importance (for tree models) — nice for the dashboard
if hasattr(fitted[best_name], "feature_importances_"):
    imp = sorted(zip(FEATURES, fitted[best_name].feature_importances_.tolist()),
                 key=lambda t: -t[1])
    with open("outputs/feature_importance.json", "w") as f:
        json.dump([{"feature": a, "importance": round(b, 4)} for a, b in imp], f, indent=2)

print("saved: outputs/metrics.json, outputs/test_predictions.csv, models/best_model.pkl")
