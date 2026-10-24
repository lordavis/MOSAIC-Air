"""FastAPI inference service for the MOSAIC-Air historical replay demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .data import HORIZONS, LOOKBACK
from .model import WindAwareGraphTemporalNet

ROOT = Path(__file__).resolve().parents[1]
PACK = np.load(ROOT / "data/processed/series.npz", allow_pickle=False)
CHECKPOINT = torch.load(ROOT / "models/wagtf_net.pt", map_location="cpu", weights_only=False)
MODEL = WindAwareGraphTemporalNet(CHECKPOINT["n_features"], CHECKPOINT["coords"], CHECKPOINT["horizons"])
MODEL.load_state_dict(CHECKPOINT["state_dict"])
MODEL.eval()
X = PACK["x"]
Y = PACK["y"]
Y_FILL = PACK["y_filled"]
WIND = PACK["wind"]
TIMES = pd.to_datetime(PACK["times_ns"])
STATIONS = PACK["station_names"].tolist()
COORDS = PACK["coords"]
FEATURE_MEAN = PACK["feature_mean"]
FEATURE_STD = PACK["feature_std"]
TARGET_MEAN = float(CHECKPOINT["target_mean"])
TARGET_STD = float(CHECKPOINT["target_std"])
EVENT_THRESHOLD = float(CHECKPOINT["event_threshold"])
EVENT_CUTS = CHECKPOINT["event_cuts"]

app = FastAPI(title="MOSAIC-Air inference API", version="1.0.0")


@app.get("/")
def home():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "historical_replay", "model": "WAGTF-Net"}


@app.get("/api/metadata")
def metadata():
    # Test-period timestamps avoid presenting training records as unseen forecasts.
    start = int(np.searchsorted(TIMES.values, np.datetime64("2016-07-01T00:00")))
    end = len(TIMES) - max(HORIZONS) - 1
    default_idx = min(start + (end - start) * 3 // 5, end)
    return {
        "stations": [{"name": s, "lat": float(COORDS[i, 0]), "lon": float(COORDS[i, 1])} for i, s in enumerate(STATIONS)],
        "horizons_hours": list(HORIZONS), "event_threshold_ug_m3": EVENT_THRESHOLD,
        "period_start": TIMES[start].strftime("%Y-%m-%dT%H:%M"),
        "period_end": TIMES[end].strftime("%Y-%m-%dT%H:%M"),
        "default_time": TIMES[default_idx].strftime("%Y-%m-%dT%H:%M"),
        "timezone_note": "Dataset timestamps are Beijing local time (UTC+08:00).",
    }


@app.get("/api/forecast")
def forecast(timestamp: str = Query(..., description="Historical issue time, YYYY-MM-DDTHH:00")):
    try:
        requested = pd.Timestamp(timestamp)
        idx = int(TIMES.get_indexer([requested])[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Use a valid hourly timestamp such as 2016-12-01T06:00.") from exc
    if idx < LOOKBACK - 1 or idx + max(HORIZONS) >= len(TIMES):
        raise HTTPException(status_code=422, detail="The selected time does not have a complete input window and forecast horizon.")
    if requested < pd.Timestamp("2016-07-01"):
        raise HTTPException(status_code=422, detail="Choose a timestamp from the held-out test period beginning 2016-07-01.")

    x = torch.from_numpy(X[idx - LOOKBACK + 1:idx + 1]).unsqueeze(0)
    wind = torch.from_numpy(WIND[idx]).unsqueeze(0)
    with torch.no_grad():
        q, event_logits = MODEL(x, wind)
        q = (q[0].numpy() * TARGET_STD + TARGET_MEAN).clip(min=0.0)
        event_prob = torch.sigmoid(event_logits[0]).numpy()

    forecasts = []
    for si, name in enumerate(STATIONS):
        temp = float(X[idx, si, 6] * FEATURE_STD[6] + FEATURE_MEAN[6])
        ws = float(WIND[idx, si, 1])
        wd = float(np.degrees(WIND[idx, si, 0]) % 360)
        horizons = []
        for hi, horizon in enumerate(HORIZONS):
            probability = float(event_prob[si, hi])
            cut = float(EVENT_CUTS[str(horizon)])
            horizons.append({
                "hours": int(horizon), "p10": float(q[si, hi, 0]), "p50": float(q[si, hi, 1]), "p90": float(q[si, hi, 2]),
                "event_probability": probability, "episode_alert": probability >= cut, "event_cut": cut,
            })
        forecasts.append({
            "station": name, "lat": float(COORDS[si, 0]), "lon": float(COORDS[si, 1]),
            "current_pm25": None if not np.isfinite(Y[idx, si]) else float(Y[idx, si]),
            "pm25_input_imputed": not bool(np.isfinite(Y[idx, si])),
            "temperature_c": temp, "wind_speed_m_s": ws, "wind_from_deg": wd, "horizons": horizons,
        })
    return {
        "issue_time": requested.strftime("%Y-%m-%dT%H:%M"),
        "timezone": "Asia/Shanghai (UTC+08:00)", "mode": "historical_replay",
        "episode_threshold_ug_m3": EVENT_THRESHOLD, "forecasts": forecasts,
    }
