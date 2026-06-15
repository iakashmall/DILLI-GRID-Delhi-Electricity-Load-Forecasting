"""
STEP 3 — EXPORT A MODEL THE WEBSITE CAN RUN
===========================================
The winning model uses lag features (yesterday's load), which exist only
for historical hours. For the dashboard's "forecast ANY date & time"
feature we train a second XGBoost on calendar + weather features only,
then export its trees as plain JSON so JavaScript can run it in the
browser — no server needed.

Outputs:
  models/deploy_model.json   -> trees as arrays (feature, threshold, children, leaf)
  outputs/climatology.json   -> typical temp/humidity per (month, hour) to pre-fill the widget
"""

import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from s0_features import build_features  # same features as training

DEPLOY_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos",
                   "dayofweek", "is_weekend", "is_holiday",
                   "temp_c", "humidity", "cooling_dh", "heating_dh",
                   "temp_x_humidity"]

df = pd.read_csv("data/delhi_hourly.csv", parse_dates=["timestamp"],
                 index_col="timestamp")
df = build_features(df)

X, y = df[DEPLOY_FEATURES], df["load_mw"]

# quick honest check on the last 6 months before fitting on everything
split = "2012-09-10"
m_check = XGBRegressor(n_estimators=220, learning_rate=0.07, max_depth=6,
                       subsample=0.9, n_jobs=-1, random_state=42)
m_check.fit(X[df.index < split], y[df.index < split])
p = m_check.predict(X[df.index >= split])
mape = np.mean(np.abs((y[df.index >= split] - p) / y[df.index >= split])) * 100
print(f"deploy model holdout MAPE (no lag features): {mape:.2f}%")

# final fit on ALL data
model = XGBRegressor(n_estimators=220, learning_rate=0.07, max_depth=6,
                     subsample=0.9, n_jobs=-1, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------------
# export trees -> JSON arrays a tiny JS function can evaluate
# ---------------------------------------------------------------
def flatten(node, arrs):
    """Depth-first flatten of one xgboost tree into parallel arrays."""
    i = len(arrs["feat"])
    for k in ("feat", "thr", "left", "right", "leaf"):
        arrs[k].append(0)
    if "leaf" in node:
        arrs["feat"][i] = -1
        arrs["leaf"][i] = round(node["leaf"], 3)
    else:
        arrs["feat"][i] = int(node["split"][1:]) if node["split"].startswith("f") \
            else DEPLOY_FEATURES.index(node["split"])
        arrs["thr"][i] = node["split_condition"]  # full precision! rounding flips knife-edge splits
        kids = {c["nodeid"]: c for c in node["children"]}
        arrs["left"][i] = flatten(kids[node["yes"]], arrs)
        arrs["right"][i] = flatten(kids[node["no"]], arrs)
    return i

trees = []
for dump in model.get_booster().get_dump(dump_format="json"):
    arrs = {"feat": [], "thr": [], "left": [], "right": [], "leaf": []}
    flatten(json.loads(dump), arrs)
    trees.append(arrs)

base_raw = (json.loads(model.get_booster().save_config())
            ["learner"]["learner_model_param"]["base_score"])
base = float(base_raw.strip("[]"))  # xgboost stores it like "[2.977079E3]"

with open("models/deploy_model.json", "w") as f:
    json.dump({"base_score": base, "features": DEPLOY_FEATURES,
               "trees": trees}, f)


# ---------------------------------------------------------------
# sanity check: JS-style tree walk must match xgboost's predict()
# ---------------------------------------------------------------
def js_predict(x):
    s = base
    for t in trees:
        i = 0
        while t["feat"][i] != -1:
            i = t["left"][i] if x[t["feat"][i]] < t["thr"][i] else t["right"][i]
        s += t["leaf"][i]
    return s

sample = X.iloc[:300].values.astype(np.float32)  # xgboost compares in float32; JS uses Math.fround
pp = model.predict(X.iloc[:300])
diffs = [abs(js_predict(r) - pp[k]) for k, r in enumerate(sample)]
import numpy as _np
print(f"|python - js| mean {_np.mean(diffs):.3f} MW, median {_np.median(diffs):.3f}, max {max(diffs):.2f}")

# ---------------------------------------------------------------
# climatology: typical Delhi temp & humidity for each (month, hour)
# ---------------------------------------------------------------
clim = (df.groupby([df.index.month, df.index.hour])[["temp_c", "humidity"]]
          .mean().round(1))
clim_out = {f"{m}-{h}": [row["temp_c"], row["humidity"]]
            for (m, h), row in clim.iterrows()}
with open("outputs/climatology.json", "w") as f:
    json.dump(clim_out, f)

size = len(open("models/deploy_model.json").read()) / 1024
print(f"exported {len(trees)} trees ({size:.0f} KB) + climatology table")
