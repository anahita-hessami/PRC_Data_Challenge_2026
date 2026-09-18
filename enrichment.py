"""
PRC Data Challenge 2026 - Data Enrichment
Author: Harsh (External Data & Feature Enrichment Lead)

Builds leakage-safe features for each departing flight: time of day, weekend
flag, recent traffic (15/30/60 min), weather (visibility, wind, temperature,
fog, snow/ice) and a parking zone. Features are anchored on the actual
off-block time (AOBT_3_flt), with the scheduled time (SCHED_TIME_UTC_mvt) as a
fallback, because the test set has no BLOCK_TIME_UTC_mvt. Weather is read from a
saved cache so the process runs offline and reproducibly.
"""

import os, io, time
from datetime import date
import requests
import pandas as pd

WEATHER_CACHE = "weather_cache.parquet"


def build_weather_cache(airports, start, end, path=WEATHER_CACHE):
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    dates = (f"year1={start.year}&month1={start.month}&day1={start.day}"
             f"&year2={end.year}&month2={end.month}&day2={end.day}")
    frames = []
    for i, code in enumerate(airports, 1):
        print(f"[{i}/{len(airports)}] {code} ...", end=" ", flush=True)
        url = f"{base}?data=all&tz=Etc/UTC&format=comma&latlon=no&missing=M&trace=T&{dates}&station={code}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60); r.raise_for_status()
                if "station,valid" not in r.text: raise ValueError("empty reply")
                w = pd.read_csv(io.StringIO(r.text), comment="#", na_values="M").assign(airport=code)
                frames.append(w); print(f"ok ({len(w)} rows)"); break
            except Exception as e:
                if attempt < 2: print("retry...", end=" ", flush=True); time.sleep(4)
                else: print("FAILED:", repr(e))
        time.sleep(1)
    wx = pd.concat(frames, ignore_index=True)
    wx["valid"] = pd.to_datetime(wx["valid"])
    codes = wx["wxcodes"].fillna("")
    wx["is_fog"]      = codes.str.contains("FG").astype(int)
    wx["is_snow_ice"] = codes.str.contains("SN|FZ").astype(int)
    wx["temp_c"]      = (wx["tmpf"] - 32) * 5/9
    wx = wx.rename(columns={"vsby": "visibility_miles", "sknt": "wind_kt"})
    wx = wx[["airport", "valid", "visibility_miles", "wind_kt", "temp_c", "is_fog", "is_snow_ice"]]
    wx.to_parquet(path, index=False)
    print(f"saved {len(wx)} real weather rows -> {path}")
    return wx


def load_weather(path=WEATHER_CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run build_weather_cache(...) once first.")
    wx = pd.read_parquet(path)
    wx["valid"] = pd.to_datetime(wx["valid"])
    if wx["valid"].dt.tz is None:
        wx["valid"] = wx["valid"].dt.tz_localize("UTC")
    return wx.sort_values("valid")


def build_features(df, weather=None):
    needed = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}.")
    if weather is None:
        weather = load_weather()

    d = df.copy()
    if "PHASE_mvt" in d.columns:
        d = d[d["PHASE_mvt"] == "DEP"].copy()

    aobt  = pd.to_datetime(d["AOBT_3_flt"])
    sched = pd.to_datetime(d["SCHED_TIME_UTC_mvt"])
    anchor = aobt.fillna(sched)
    if anchor.dt.tz is None:
        anchor = anchor.dt.tz_localize("UTC")
    d["anchor_time"] = anchor

    d["stand_area"] = d["STAND_mvt"].astype("string").str.strip().str[0].fillna("UNK")

    ok = d[d["anchor_time"].notna()].copy()
    ok["pushback_hour"] = ok["anchor_time"].dt.hour
    ok["is_weekend"]    = (ok["anchor_time"].dt.dayofweek >= 5).astype(int)

    ok = ok.sort_values(["ADEP_mvt", "anchor_time"])
    for win in ["15min", "30min", "60min"]:
        def _recent(g, win=win):
            return g.set_index("anchor_time")["MVT_ID_mvt"].rolling(win, closed="left").count()
        ok[f"planes_pushed_last_{win}"] = ok.groupby("ADEP_mvt", group_keys=False).apply(_recent).values
        ok[f"planes_pushed_last_{win}"] = ok[f"planes_pushed_last_{win}"].fillna(0)

    ok = ok.sort_values("anchor_time")
    ok = pd.merge_asof(ok, weather, left_on="anchor_time", right_on="valid",
                       left_by="ADEP_mvt", right_by="airport",
                       direction="backward", tolerance=pd.Timedelta("2h"))

    feat_cols = ["pushback_hour", "is_weekend",
                 "planes_pushed_last_15min", "planes_pushed_last_30min", "planes_pushed_last_60min",
                 "visibility_miles", "wind_kt", "temp_c", "is_fog", "is_snow_ice"]
    out = d[["MVT_ID_mvt", "stand_area"]].merge(ok[["MVT_ID_mvt"] + feat_cols], on="MVT_ID_mvt", how="left")

    cols = ["MVT_ID_mvt", "pushback_hour", "is_weekend",
            "planes_pushed_last_15min", "planes_pushed_last_30min", "planes_pushed_last_60min",
            "visibility_miles", "wind_kt", "temp_c", "is_fog", "is_snow_ice", "stand_area"]
    return out[cols].reset_index(drop=True)
