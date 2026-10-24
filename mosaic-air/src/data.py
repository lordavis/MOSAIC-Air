"""Download and prepare the UCI Beijing multi-station air-quality time series."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/beijing_multi_site_air_quality.zip"
OUT = ROOT / "data/processed/series.npz"
URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
LOOKBACK = 24
HORIZONS = (1, 6, 24)
SPLITS = {"train_end": "2015-12-31 23:00", "val_end": "2016-06-30 23:00"}
NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
FEATURES = NUMERIC + ["WD_sin", "WD_cos"]
FEATURE_NAMES = FEATURES + [f"missing_{name}" for name in FEATURES]

WIND_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def ensure_archive() -> None:
    if ARCHIVE.exists():
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    ARCHIVE.write_bytes(response.content)


def prepare(force=False) -> Path:
    if OUT.exists() and not force:
        return OUT
    ensure_archive()
    stations_file = ROOT / "src/stations.json"
    station_meta = json.loads(stations_file.read_text())['stations']
    with zipfile.ZipFile(ARCHIVE) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("PRSA2017_Data_20130301-20170228.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            csv_names = [n for n in inner.namelist() if n.endswith(".csv")]
            frames = []
            for name in csv_names:
                frame = pd.read_csv(inner.open(name))
                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data[["year", "month", "day", "hour"]].rename(columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}))
    station_names = sorted(data["station"].unique().tolist())
    missing_coords = sorted(set(station_names) - set(station_meta))
    if missing_coords:
        raise ValueError(f"Missing coordinate metadata for {missing_coords}")
    times = pd.date_range(data.time.min(), data.time.max(), freq="h")
    n_time, n_station = len(times), len(station_names)
    features_raw = np.zeros((n_time, n_station, len(FEATURES)), dtype=np.float32)
    feature_missing = np.ones((n_time, n_station, len(FEATURES)), dtype=np.float32)
    targets = np.full((n_time, n_station), np.nan, dtype=np.float32)
    wind = np.zeros((n_time, n_station, 2), dtype=np.float32)

    for si, station in enumerate(station_names):
        df = data.loc[data.station == station].set_index("time").reindex(times)
        numeric = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
        raw_wd = df["wd"].map(WIND_DEGREES).astype(float)
        rad = np.deg2rad(raw_wd)
        numeric["WD_sin"] = np.sin(rad)
        numeric["WD_cos"] = np.cos(rad)
        numeric = numeric[FEATURES]
        feature_missing[:, si, :] = numeric.isna().to_numpy(dtype=np.float32)
        targets[:, si] = pd.to_numeric(df["PM2.5"], errors="coerce").to_numpy(dtype=np.float32)
        # Causal forward-fill is limited to three hours; longer gaps are median-imputed below.
        filled = numeric.ffill(limit=3)
        features_raw[:, si, :] = filled.to_numpy(dtype=np.float32)
        wd_deg = raw_wd.to_numpy(dtype=np.float32)
        wind[:, si, 0] = np.deg2rad(np.nan_to_num(wd_deg, nan=0.0))
        ws = pd.to_numeric(df["WSPM"], errors="coerce").ffill(limit=3)
        wind[:, si, 1] = ws.to_numpy(dtype=np.float32)

    index = pd.DatetimeIndex(times)
    train_end = pd.Timestamp(SPLITS["train_end"])
    train_time_mask = index <= train_end
    medians = np.nanmedian(features_raw[train_time_mask], axis=(0, 1)).astype(np.float32)
    # At least one direction record is always available; use 0 for any all-missing transformed feature.
    medians = np.nan_to_num(medians, nan=0.0)
    for fi in range(features_raw.shape[-1]):
        arr = features_raw[:, :, fi]
        bad = ~np.isfinite(arr)
        arr[bad] = medians[fi]
    for si in range(n_station):
        station_missing = ~np.isfinite(wind[:, si, 1])
        station_speed_median = np.nanmedian(wind[train_time_mask, si, 1])
        if not np.isfinite(station_speed_median):
            station_speed_median = 0.0
        wind[station_missing, si, 1] = station_speed_median
    mean = features_raw[train_time_mask].mean(axis=(0, 1))
    std = features_raw[train_time_mask].std(axis=(0, 1))
    std[std < 1e-6] = 1.0
    normalized = (features_raw - mean) / std
    # Missing flags preserve sensor availability information; direction channels are also included.
    model_features = np.concatenate([normalized, feature_missing], axis=-1).astype(np.float32)
    target_train = targets[train_time_mask]
    target_mean = float(np.nanmean(target_train))
    target_std = float(np.nanstd(target_train))
    if target_std < 1e-6:
        target_std = 1.0
    y_filled = targets.copy()
    for si in range(n_station):
        s = pd.Series(y_filled[:, si], index=index).ffill(limit=3).to_numpy(dtype=np.float32)
        s[~np.isfinite(s)] = float(np.nanmedian(target_train))
        y_filled[:, si] = s

    coords = np.array([[station_meta[s]["lat"], station_meta[s]["lon"]] for s in station_names], dtype=np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        x=model_features,
        y=targets,
        y_filled=y_filled,
        wind=wind,
        times_ns=index.asi8,
        station_names=np.array(station_names),
        coords=coords,
        feature_names=np.array(FEATURE_NAMES),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        target_mean=np.array(target_mean, dtype=np.float32),
        target_std=np.array(target_std, dtype=np.float32),
    )
    return OUT


if __name__ == "__main__":
    path = prepare(force=True)
    with np.load(path, allow_pickle=False) as pack:
        print(f"Prepared {path.relative_to(ROOT)}")
        print(f"Series shape: {pack['x'].shape}; stations: {', '.join(pack['station_names'].tolist())}")
