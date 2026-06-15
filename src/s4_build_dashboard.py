"""
STEP 4 — BUILD THE DASHBOARD
============================
Gathers everything the website needs and injects it into the HTML template
as embedded JSON, producing ONE self-contained file you can open anywhere
(double-click, host on GitHub Pages, or serve from FastAPI).

Output: outputs/dashboard.html
"""

import json
import numpy as np
import pandas as pd
import holidays

# ---------------------------------------------------------------
# 1. hourly test-set actual vs predicted (compact arrays)
# ---------------------------------------------------------------
tp = pd.read_csv("outputs/test_predictions.csv", parse_dates=["timestamp"])
t0 = tp["timestamp"].iloc[0]
hourly = {
    "start": t0.strftime("%Y-%m-%dT%H:00"),
    # hour offsets from start (handles small gaps in the data)
    "offsets": ((tp["timestamp"] - t0).dt.total_seconds() // 3600).astype(int).tolist(),
    "actual": tp["actual"].round(0).astype(int).tolist(),
    "predicted": tp["predicted"].round(0).astype(int).tolist(),
    "temp": tp["temp_c"].round(1).tolist(),
}

# ---------------------------------------------------------------
# 2. accuracy diagnostics computed once here, in Python
# ---------------------------------------------------------------
err = tp["predicted"] - tp["actual"]
pct = err / tp["actual"] * 100

bins = np.arange(-12, 12.5, 1.0)
hist, _ = np.histogram(pct.clip(-12, 12), bins=bins)
histogram = {"bins": bins[:-1].tolist(), "counts": hist.tolist()}

monthly = (pd.DataFrame({"m": tp["timestamp"].dt.strftime("%Y-%m"),
                         "ape": np.abs(pct)})
           .groupby("m")["ape"].mean().round(2))
monthly_mape = {"labels": monthly.index.tolist(), "values": monthly.values.tolist()}

# daily peaks (actual vs predicted) — what grid operators actually care about
d = tp.set_index("timestamp").resample("1D").max()
daily_peak = {
    "dates": d.index.strftime("%Y-%m-%d").tolist(),
    "actual": d["actual"].round(0).tolist(),
    "predicted": d["predicted"].round(0).tolist(),
}

# ---------------------------------------------------------------
# 3. temperature -> load curve (the classic "AC kicks in" U-shape)
# ---------------------------------------------------------------
full = pd.read_csv("data/delhi_hourly.csv", parse_dates=["timestamp"], index_col="timestamp")
tb = (full.assign(bin=(full["temp_c"] // 2 * 2))
          .groupby("bin")["load_mw"].agg(["mean", "count"]))
tb = tb[tb["count"] > 30]
temp_curve = {"temp": tb.index.tolist(), "load": tb["mean"].round(0).tolist()}

# ---------------------------------------------------------------
# 4. everything else: metrics, importance, model, climatology, holidays
# ---------------------------------------------------------------
metrics = json.load(open("outputs/metrics.json"))
importance = json.load(open("outputs/feature_importance.json"))
deploy_model = json.load(open("models/deploy_model.json"))
climatology = json.load(open("outputs/climatology.json"))

hol = holidays.India(years=range(2010, 2036))
holiday_list = sorted(d.strftime("%Y-%m-%d") for d in hol)

payload = {
    "hourly": hourly, "histogram": histogram, "monthly_mape": monthly_mape,
    "daily_peak": daily_peak, "temp_curve": temp_curve,
    "metrics": metrics, "importance": importance,
    "climatology": climatology, "holidays": holiday_list,
}

html = open("src/dashboard_template.html").read()
html = html.replace("/*__DATA__*/", "const DATA = " + json.dumps(payload) + ";")
html = html.replace("/*__MODEL__*/", "const MODEL = " + json.dumps(deploy_model) + ";")
open("outputs/dashboard.html", "w").write(html)

size = len(html) / 1024
print(f"dashboard.html written ({size:.0f} KB) — open it in any browser")
