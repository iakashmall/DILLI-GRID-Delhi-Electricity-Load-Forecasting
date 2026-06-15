"""
STEP 1 — DATA PREPARATION
=========================
Takes two raw files and produces ONE clean hourly dataset:

  data/delhi_load_15min_raw.csv  -> Delhi SLDC load, 15-min, by DISCOM (2010-2013)
  data/delhi_weather_raw.csv     -> Delhi (Safdarjung) hourly weather (1996-2017)

Output: data/delhi_hourly.csv  (one row per hour: load + weather)
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. LOAD DATA  (Delhi State Load Dispatch Centre, 15-min interval)
# ---------------------------------------------------------------
print("Reading load data ...")
load = pd.read_csv("data/delhi_load_15min_raw.csv")

# Keep only the city total (sum of all DISCOMs: BRPL, BYPL, NDPL, NDMC, MES)
load = load[load["discom"] == "Total"].copy()

# Build a proper timestamp: 'date' is the day, 'timepoint' is the 15-min
# slot number (1..96).  Slot 1 = 00:00-00:15, so slot start = (timepoint-1)*15 min.
load["date"] = pd.to_datetime(load["date"], format="%m/%d/%y")
load["timestamp"] = load["date"] + pd.to_timedelta((load["timepoint"] - 1) * 15, unit="min")
load = load[["timestamp", "mW"]].rename(columns={"mW": "load_mw"})

# Remove impossible values (sensor glitches). Delhi load in this period
# is roughly 1000-5500 MW.
load = load[(load["load_mw"] > 500) & (load["load_mw"] < 7000)]

# 15-min -> hourly average
load = (load.set_index("timestamp")
            .resample("1h").mean())

print(f"  load: {load.index.min()}  ->  {load.index.max()}   ({len(load)} hours)")

# ---------------------------------------------------------------
# 2. WEATHER DATA  (wunderground, Delhi; timestamps are UTC)
# ---------------------------------------------------------------
print("Reading weather data ...")
wx = pd.read_csv("data/delhi_weather_raw.csv")
wx.columns = [c.strip() for c in wx.columns]

wx["timestamp"] = pd.to_datetime(wx["datetime_utc"], format="%Y%m%d-%H:%M")
# Convert UTC -> IST (+5:30) so it lines up with the load data
wx["timestamp"] = wx["timestamp"] + pd.Timedelta(hours=5, minutes=30)

wx = wx.rename(columns={"_tempm": "temp_c", "_hum": "humidity",
                        "_wspdm": "wind_kmh", "_rain": "rain", "_fog": "fog"})
wx = wx[["timestamp", "temp_c", "humidity", "wind_kmh", "rain", "fog"]]

# Clean obvious sensor errors
wx.loc[(wx["temp_c"] < -5) | (wx["temp_c"] > 52), "temp_c"] = np.nan
wx.loc[(wx["humidity"] < 1) | (wx["humidity"] > 100), "humidity"] = np.nan
wx.loc[(wx["wind_kmh"] < 0) | (wx["wind_kmh"] > 120), "wind_kmh"] = np.nan

# Hourly average (raw data sometimes has several reports per hour)
wx = (wx.set_index("timestamp")
        .resample("1h").mean())

# Fill small gaps by interpolation (max 6 hours)
wx = wx.interpolate(limit=6)

# ---------------------------------------------------------------
# 3. MERGE + FINAL CLEANING
# ---------------------------------------------------------------
df = load.join(wx, how="inner")

# Fill any leftover weather gaps with the month-hour average (climatology)
for col in ["temp_c", "humidity", "wind_kmh", "rain", "fog"]:
    clim = df.groupby([df.index.month, df.index.hour])[col].transform("mean")
    df[col] = df[col].fillna(clim)

# Drop hours where load itself is missing — we cannot train on those
df = df.dropna(subset=["load_mw"])

df.index.name = "timestamp"
df.to_csv("data/delhi_hourly.csv")

print(f"\nFinal dataset: {len(df)} hourly rows  "
      f"({df.index.min().date()} -> {df.index.max().date()})")
print(df.describe().round(1))
