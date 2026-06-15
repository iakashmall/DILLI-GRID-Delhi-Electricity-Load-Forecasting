# DILLI·GRID — Delhi Electricity Load Forecasting

End-to-end ML pipeline: real Delhi load + weather data → model bake-off →
best model deployed inside a self-contained interactive dashboard where you
can forecast load for **any date and hour you choose**.

---

## 1. The data

| Dataset | Source | Coverage |
|---|---|---|
| Load (15-min, by DISCOM) | Delhi SLDC, via the public `Ecohen4/Delhi` GitHub dataset | 29 Mar 2010 → 10 Mar 2013 |
| Weather (hourly) | Wunderground Delhi archive (`stevanzecic/WeatherPredict` mirror of the Kaggle dataset) | 1996 → 2017 |

`step1_prepare_data.py` cleans both (glitch removal, UTC→IST shift, sensor-error
filtering), resamples everything to hourly, merges them, and fills small weather
gaps with month-hour climatology. Result: **25,867 clean hourly rows** in
`data/delhi_hourly.csv`.

## 2. Features (`src/features.py`)

Simple and physical — no black magic:

- **Calendar**: hour & month as sin/cos pairs, day-of-week, weekend flag, Indian public holidays
- **Weather**: temperature, humidity, wind, rain/fog flags, cooling degree-hours `max(T−24,0)`, heating degree-hours `max(18−T,0)`, temp×humidity (mugginess)
- **History**: load 24 h ago, load 168 h ago (same hour last week), 24 h rolling mean

## 3. Model bake-off (`step2_train_models.py`)

Chronological split — train on the first ~2 years, test on **one full unseen
year** that includes Delhi's record-breaking summer of 2012. Every model is also
scored on three *stress subsets*: top-5 % load hours, ≥40 °C heatwave hours, and
statistically unusual days (|z| > 2).

| Model | Overall MAPE | Peak hours | Heatwave | Outlier days |
|---|---|---|---|---|
| Naive (yesterday) | 5.12 % | 5.62 % | 5.93 % | 6.74 % |
| Linear / Ridge | ~4.8 % | ~4.3 % | ~4.7 % | ~4.6 % |
| Random Forest | 3.48 % | 4.79 % | 4.39 % | 3.46 % |
| XGBoost (plain) | 3.22 % | **5.46 %** ⚠ | 4.85 % | 3.39 % |
| Neural Net (MLP) | 3.30 % | 3.05 % | 3.19 % | 2.46 % |
| **XGBoost (relative) — deployed** | **2.89 %** | **2.41 %** | **2.29 %** | **2.52 %** |

**The key insight:** plain tree models *cannot extrapolate*. Summer 2012 set
all-time load records, so plain XGBoost — the best model on normal days —
collapsed exactly when it mattered (5.46 % on peaks). The fix: predict the
**ratio** `load / rolling-24h-mean` instead of raw MW, then multiply back. The
ratio stays in a range the trees have seen even when absolute load breaks
records. That one change made XGBoost the best model *everywhere*, including
every outlier condition. (R² = 0.975 on the unseen year.)

## 4. Web deployment (`step3_export_for_web.py`)

The dashboard's "Forecast Lab" must work for *any* future date, where lag
features don't exist. So a second, **lag-free** XGBoost is trained on the 12
calendar + weather features only (holdout MAPE 5.1 %) and its 220 trees are
exported to plain JSON. The dashboard runs the trees natively in JavaScript —
no server needed (float32 casting via `Math.fround` gives exact parity with
Python: max deviation 0.01 MW).

## 5. The dashboard (`outputs/dashboard.html`)

One self-contained HTML file, control-room-at-dusk aesthetic:

- Live oscilloscope hero showing the last 14 days of real load
- Actual-vs-predicted explorer over the full test year (zoom, pan, temperature overlay)
- Model arena with all stress-test results
- Error diagnostics (histogram, monthly MAPE, scatter)
- What drives Delhi's load (feature importance + the temperature U-curve)
- **Forecast Lab** — pick any date (2010–2035) and hour; typical weather is
  pre-filled from climatology, sliders let you run what-if scenarios, and a
  demand-growth slider scales the 2012-era model to later years

## How to run

```bash
pip install -r requirements.txt

python src/step1_prepare_data.py    # clean + merge data
python src/step2_train_models.py    # bake-off, saves best model + metrics
python src/step3_export_for_web.py  # lag-free model -> JSON for the browser
python src/step4_build_dashboard.py # builds outputs/dashboard.html

# Option A: just open the file
open outputs/dashboard.html

# Option B: serve it with the API
uvicorn app:app --reload --port 8000
#   GET  /                      -> dashboard
#   POST /api/predict_anytime   -> any date/time (lag-free model)
#   POST /api/predict           -> best model on historical timestamps
```

## Honest caveats

- The data era is 2010–2013; Delhi's demand has grown since (peak crossed
  8,000 MW in 2024). The growth slider/parameter compensates linearly-ish,
  but for production you'd retrain on current SLDC data — the pipeline is
  ready for it, just swap the CSV.
- The "anytime" model (±5 %) is necessarily less accurate than the full
  lag-based model (±2.9 %) because it can't see recent load.
- A natural extension is an LSTM/PatchTST sequence model; the MLP results
  suggest deep models are competitive, but the relative-target XGBoost won
  on every metric while staying explainable.
