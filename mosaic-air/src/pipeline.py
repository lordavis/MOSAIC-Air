"""Reproducible training and evaluation pipeline for MOSAIC-Air."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import HORIZONS, LOOKBACK, SPLITS, prepare
from .model import WindAwareGraphTemporalNet

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
THRESHOLD = 75.0  # Project-defined high-pollution episode threshold (not a health guideline).
QUANTILES = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)


class WindowDataset(Dataset):
    def __init__(self, x, y, wind, anchors, target_mean, target_std):
        self.x, self.y, self.wind = x, y, wind
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.target_mean, self.target_std = float(target_mean), float(target_std)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        t = int(self.anchors[i])
        x = self.x[t - LOOKBACK + 1:t + 1]
        targets = np.stack([self.y[t + h] for h in HORIZONS])
        mask = np.isfinite(targets)
        scaled = (np.nan_to_num(targets, nan=self.target_mean) - self.target_mean) / self.target_std
        wind = self.wind[t]
        return (torch.from_numpy(x), torch.from_numpy(wind), torch.from_numpy(scaled.astype(np.float32)),
                torch.from_numpy(mask), torch.from_numpy(np.nan_to_num(targets, nan=0.0).astype(np.float32)))


def split_anchors(times, stride=6):
    idx = pd.to_datetime(times)
    train_end = pd.Timestamp(SPLITS["train_end"])
    val_end = pd.Timestamp(SPLITS["val_end"])
    last = len(idx) - max(HORIZONS) - 1
    base = np.arange(LOOKBACK - 1, last + 1, dtype=np.int64)
    train = base[(idx[base + max(HORIZONS)] <= train_end)]
    val = base[(idx[base + 1] > train_end) & (idx[base + max(HORIZONS)] <= val_end)]
    test = base[(idx[base + 1] > val_end)]
    return train[::stride], val[::stride], test[::stride]


def masked_pinball(quantiles, target, mask):
    # quantiles [B,N,H,Q], target/mask [B,H,N]
    pred = quantiles.permute(0, 2, 1, 3)
    y = target.unsqueeze(-1)
    q = QUANTILES.to(pred.device).view(1, 1, 1, -1)
    error = y - pred
    loss = torch.maximum(q * error, (q - 1.0) * error)
    m = mask.unsqueeze(-1).float()
    return (loss * m).sum() / (m.sum().clamp(min=1.0) * 3.0)


def masked_bce(logits, target_raw, mask, pos_weight):
    # logits [B,N,H], event targets [B,H,N]
    labels = (target_raw >= THRESHOLD).float().permute(0, 2, 1)
    pred = logits
    m = mask.permute(0, 2, 1).float()
    loss = nn.functional.binary_cross_entropy_with_logits(pred, labels, reduction="none", pos_weight=pos_weight)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def run_epoch(model, loader, optimizer, device, target_mean, target_std, pos_weight, train=True):
    model.train(train)
    losses, val_abs, val_count = [], 0.0, 0.0
    for x, wind, y, mask, raw in loader:
        x, wind, y, mask, raw = x.to(device), wind.to(device), y.to(device), mask.to(device), raw.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            quantiles, events = model(x, wind)
            loss = masked_pinball(quantiles, y, mask) + 0.25 * masked_bce(events, raw, mask, pos_weight)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        med = (quantiles[..., 1].permute(0, 2, 1) * target_std + target_mean)
        val_abs += (torch.abs(med - raw) * mask.float()).sum().item()
        val_count += mask.sum().item()
    return float(np.mean(losses)), val_abs / max(val_count, 1.0)


@torch.no_grad()
def predict_windows(model, loader, device, mean, std):
    model.eval()
    qs, probs, ys, masks = [], [], [], []
    for x, wind, _, mask, raw in loader:
        q, e = model(x.to(device), wind.to(device))
        q = (q.cpu().numpy() * std + mean).clip(min=0.0)
        qs.append(q)
        probs.append(torch.sigmoid(e).cpu().numpy())
        ys.append(raw.numpy())
        masks.append(mask.numpy())
    # Quantiles [samples,stations,horizons,quantiles], probs [samples,stations,horizons]
    return np.concatenate(qs), np.concatenate(probs), np.concatenate(ys), np.concatenate(masks)


def make_baseline_features(x, y_fill, anchors, times, station_count):
    rows = []
    dates = pd.to_datetime(times)
    for t in anchors:
        # Include causal PM lags, concurrent weather/other pollutants, cyclic time, and station identity.
        vals = []
        for lag in (0, 1, 3, 6, 12, 23):
            vals.append(y_fill[t - lag, :, None] / 100.0)
        current = x[t, :, :13]
        hour = dates[t].hour
        doy = dates[t].dayofyear
        time_vars = np.tile([math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                             math.sin(2 * math.pi * doy / 366), math.cos(2 * math.pi * doy / 366)], (station_count, 1))
        station_id = np.eye(station_count, dtype=np.float32)
        rows.append(np.concatenate(vals + [current, time_vars, station_id], axis=1))
    return np.concatenate(rows, axis=0)


def metric_row(model_name, horizon, y, pred, mask, prob=None, event_cut=0.5, lower=None, upper=None, score_cut=None):
    actual = y[mask]
    estimate = pred[mask]
    labels = (actual >= THRESHOLD).astype(int)
    out = {
        "model": model_name, "horizon_hours": int(horizon), "n": int(mask.sum()),
        "MAE": float(mean_absolute_error(actual, estimate)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, estimate))),
        "R2": float(r2_score(actual, estimate)),
        "event_F1": float(f1_score(labels, estimate >= (THRESHOLD if score_cut is None else score_cut), zero_division=0)) if prob is None else float(f1_score(labels, prob[mask] >= event_cut, zero_division=0)),
        "event_cut": float(event_cut if prob is not None else (THRESHOLD if score_cut is None else score_cut)),
        "event_cut_type": "probability" if prob is not None else "PM2.5 µg/m³",
    }
    if prob is not None:
        out["event_Brier"] = float(brier_score_loss(labels, prob[mask]))
    if lower is not None and upper is not None:
        lo, hi = lower[mask], upper[mask]
        out["P10_P90_coverage"] = float(np.mean((actual >= lo) & (actual <= hi)))
        out["P10_P90_width"] = float(np.mean(hi - lo))
    return out


def best_event_cut(y, prob, mask):
    labels = (y[mask] >= THRESHOLD).astype(int)
    p = prob[mask]
    if len(labels) == 0 or labels.sum() == 0:
        return 0.5
    cuts = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(labels, p >= c, zero_division=0) for c in cuts]
    return float(cuts[int(np.argmax(scores))])


def best_regression_event_cut(y, estimate, mask):
    labels = (y[mask] >= THRESHOLD).astype(int)
    scores = estimate[mask]
    if len(labels) == 0 or labels.sum() == 0:
        return THRESHOLD
    cuts = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(labels, scores >= c, zero_division=0) for c in cuts]
    return float(cuts[int(np.argmax(f1s))])


def plot_results(metrics, examples, station_names, coords):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#f7f6f2")
    for ax in axes:
        ax.set_facecolor("#f7f6f2")
    standard = frame[~frame.model.str.contains("missing", case=False)]
    for model_name, grp in standard.groupby("model"):
        axes[0].plot(grp.horizon_hours, grp.MAE, marker="o", linewidth=2, label=model_name)
    axes[0].set(title="PM₂.₅ forecast error by horizon", xlabel="Forecast horizon (hours)", ylabel="MAE (µg/m³)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    models = ["Persistence", "HistGradientBoosting", "WAGTF-Net"]
    colors = {"Persistence": "#94a6a0", "HistGradientBoosting": "#c8a46d", "WAGTF-Net": "#397d6c"}
    centers = np.arange(3)
    width = .23
    for mi, model_name in enumerate(models):
        grp = standard[standard.model == model_name].set_index("horizon_hours").reindex(HORIZONS)
        axes[1].bar(centers + (mi - 1) * width, grp.event_F1.to_numpy(), width=width, color=colors[model_name], label=model_name)
    axes[1].set(title="High-episode detection", ylabel="F1 score", ylim=(0, 1))
    axes[1].set_xticks(centers, [f"+{h} h" for h in HORIZONS])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    q, prob, y, mask, times, example_anchor = examples
    median = q[:, :, 0, 1]
    target = y[:, 0, :]
    valid = mask[:, 0, :]
    station = int(np.nanargmax(np.nanmean(np.where(valid, target, np.nan), axis=0)))
    first = example_anchor
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor="#f7f6f2")
    ax.set_facecolor("#f7f6f2")
    ax.plot(np.arange(len(target)), np.where(valid[:, station], target[:, station], np.nan), color="#344a53", lw=1.4, label="Observed PM₂.₅")
    ax.plot(np.arange(len(median)), median[:, station], color="#de855e", lw=1.2, label="WAGTF-Net · 1 h")
    ax.set(title=f"Historical test-window replay · {station_names[station]}", xlabel="Forecast issue time (every 6 hours)", ylabel="PM₂.₅ (µg/m³)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "example_forecast.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Long-term average hotspot map provides geographic context, not a model claim.
    avg = np.nanmean(y[:, 0, :], axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor="#f7f6f2")
    ax.set_facecolor("#f7f6f2")
    sc = ax.scatter(coords[:, 1], coords[:, 0], c=avg, s=85 + np.clip(avg, 0, None) * 1.1, cmap="YlOrRd", edgecolor="white", linewidth=1.5)
    short_names = {
        "Aotizhongxin": "Aotiz", "Changping": "Chang", "Dingling": "Ding",
        "Dongsi": "Dongsi", "Guanyuan": "Guany", "Gucheng": "Guch",
        "Huairou": "Huairou", "Nongzhanguan": "Nongz", "Shunyi": "Shunyi",
        "Tiantan": "Tiantan", "Wanliu": "Wanliu", "Wanshouxigong": "Wanshou",
    }
    offsets = {
        "Aotizhongxin": (5, 11), "Changping": (5, 7), "Dingling": (5, 7),
        "Dongsi": (4, 13), "Guanyuan": (-35, 8), "Gucheng": (5, -14),
        "Huairou": (5, 4), "Nongzhanguan": (5, -14), "Shunyi": (5, -13),
        "Tiantan": (6, 8), "Wanliu": (-31, 10), "Wanshouxigong": (-34, -14),
    }
    for i, name in enumerate(station_names):
        ax.annotate(short_names.get(name, name), (coords[i, 1], coords[i, 0]), xytext=offsets.get(name, (5, 4)), textcoords="offset points", fontsize=7)
    ax.set(title="Beijing monitoring network · mean test-set PM₂.₅", xlabel="Longitude", ylabel="Latitude")
    cb = fig.colorbar(sc, ax=ax, pad=.02)
    cb.set_label("µg/m³")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "station_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pack_path = prepare()
    p = np.load(pack_path, allow_pickle=False)
    x, y, y_fill, wind = p["x"], p["y"], p["y_filled"], p["wind"]
    times, station_names, coords = p["times_ns"], p["station_names"].tolist(), p["coords"]
    mean, std = float(p["target_mean"]), float(p["target_std"])
    train_a, val_a, test_a = split_anchors(times)
    print(f"Dataset: {len(times)} hourly timestamps · {len(station_names)} stations")
    print(f"Window anchors: train={len(train_a)} validation={len(val_a)} test={len(test_a)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WindAwareGraphTemporalNet(x.shape[-1], coords).to(device)
    train_ds = WindowDataset(x, y, wind, train_a, mean, std)
    val_ds = WindowDataset(x, y, wind, val_a, mean, std)
    test_ds = WindowDataset(x, y, wind, test_a, mean, std)
    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=96, shuffle=False, num_workers=0)
    observed_train = np.concatenate([y[t + h] for t in train_a for h in HORIZONS])
    valid_train = observed_train[np.isfinite(observed_train)]
    n_pos = int(np.sum(valid_train >= THRESHOLD))
    n_neg = int(np.sum(valid_train < THRESHOLD))
    pos_weight = torch.tensor(min(10.0, n_neg / max(n_pos, 1)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(1, 13):
        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer, device, mean, std, pos_weight, train=True)
        va_loss, va_mae = run_epoch(model, val_loader, optimizer, device, mean, std, pos_weight, train=False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "validation_loss": va_loss, "train_mae": tr_mae, "validation_mae": va_mae})
        print(f"epoch {epoch:02d}/12  train loss {tr_loss:.4f} · val MAE {va_mae:.2f} µg/m³")
        if va_mae < best_val - 0.02:
            best_val, best_state, patience = va_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    val_q, val_prob, val_y, val_mask = predict_windows(model, DataLoader(val_ds, batch_size=96), device, mean, std)
    event_cuts = {}
    for hi, horizon in enumerate(HORIZONS):
        event_cuts[str(horizon)] = best_event_cut(val_y[:, hi, :], val_prob[:, :, hi], val_mask[:, hi, :])
    q, prob, actual, mask = predict_windows(model, test_loader, device, mean, std)

    # Spatial-temporal baseline: a shared gradient-boosted model with causal lags, weather, time, and station ID.
    baseline_train_anchors = train_a
    baseline_test_anchors = test_a
    baseline_train_x = make_baseline_features(x, y_fill, baseline_train_anchors, times, len(station_names))
    baseline_test_x = make_baseline_features(x, y_fill, baseline_test_anchors, times, len(station_names))
    baseline_val_x = make_baseline_features(x, y_fill, val_a, times, len(station_names))
    gbdt_predictions = np.zeros((len(test_a), len(HORIZONS), len(station_names)), dtype=np.float32)
    gbdt_val_predictions = np.zeros((len(val_a), len(HORIZONS), len(station_names)), dtype=np.float32)
    for hi, horizon in enumerate(HORIZONS):
        target_train = np.stack([y[t + horizon] for t in baseline_train_anchors]).reshape(-1)
        valid = np.isfinite(target_train)
        reg = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=24, learning_rate=0.08, l2_regularization=1.0, random_state=23)
        reg.fit(baseline_train_x[valid], target_train[valid])
        pred = reg.predict(baseline_test_x)
        gbdt_predictions[:, hi, :] = pred.reshape(len(test_a), len(station_names))
        val_pred = reg.predict(baseline_val_x)
        gbdt_val_predictions[:, hi, :] = val_pred.reshape(len(val_a), len(station_names))
    persistence = np.repeat(y_fill[test_a, :, None], len(HORIZONS), axis=2).transpose(0, 2, 1)
    persistence_val = np.repeat(y_fill[val_a, :, None], len(HORIZONS), axis=2).transpose(0, 2, 1)
    reg_event_cuts, persistence_event_cuts = {}, {}
    for hi, horizon in enumerate(HORIZONS):
        persistence_event_cuts[str(horizon)] = best_regression_event_cut(val_y[:, hi, :], persistence_val[:, hi, :], val_mask[:, hi, :])
        reg_event_cuts[str(horizon)] = best_regression_event_cut(val_y[:, hi, :], gbdt_val_predictions[:, hi, :], val_mask[:, hi, :])

    # WAGTF event probability threshold is selected on validation data, then applied unchanged to test data.
    metrics = []
    for hi, horizon in enumerate(HORIZONS):
        valid = mask[:, hi, :]
        metrics.append(metric_row("Persistence", horizon, actual[:, hi, :], persistence[:, hi, :], valid, score_cut=persistence_event_cuts[str(horizon)]))
        metrics.append(metric_row("HistGradientBoosting", horizon, actual[:, hi, :], gbdt_predictions[:, hi, :], valid, score_cut=reg_event_cuts[str(horizon)]))
        metrics.append(metric_row("WAGTF-Net", horizon, actual[:, hi, :], q[:, :, hi, 1], valid,
                                  prob[:, :, hi], event_cuts[str(horizon)], q[:, :, hi, 0], q[:, :, hi, 2]))

    # Controlled 20% additional PM2.5 input masking on the test window.
    stress_x = x.copy()
    rng = np.random.default_rng(923)
    stress_start = int(test_a[0] - LOOKBACK + 1)
    stress_end = int(test_a[-1] + 1)
    observed_pm = stress_x[stress_start:stress_end, :, 13] < 0.5
    additional_missing = (rng.random(observed_pm.shape) < 0.20) & observed_pm
    segment = stress_x[stress_start:stress_end]
    segment[:, :, 0] = np.where(additional_missing, 0.0, segment[:, :, 0])
    segment[:, :, 13] = np.where(additional_missing, 1.0, segment[:, :, 13])
    stress_ds = WindowDataset(stress_x, y, wind, test_a, mean, std)
    stress_q, stress_prob, _, stress_mask = predict_windows(model, DataLoader(stress_ds, batch_size=96), device, mean, std)
    stress_rows = []
    for hi, horizon in enumerate(HORIZONS):
        stress_row = metric_row("WAGTF-Net +20% PM2.5 input missing", horizon, actual[:, hi, :], stress_q[:, :, hi, 1],
                                stress_mask[:, hi, :], stress_prob[:, :, hi], event_cuts[str(horizon)],
                                stress_q[:, :, hi, 0], stress_q[:, :, hi, 2])
        stress_rows.append(stress_row)
        metrics.append(stress_row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "missingness_stress.json").write_text(json.dumps({
        "experiment": "Additional random masking of 20% of originally observed PM2.5 input values in held-out test windows; missingness flags set; model unchanged.",
        "masked_fraction_of_observed_test_inputs": float(additional_missing.sum() / max(observed_pm.sum(), 1)),
        "metrics": stress_rows,
    }, indent=2))

    checkpoint = {
        "state_dict": model.cpu().state_dict(), "n_features": int(x.shape[-1]),
        "coords": coords, "horizons": HORIZONS, "target_mean": mean, "target_std": std,
        "station_names": station_names, "event_threshold": THRESHOLD, "event_cuts": event_cuts,
    }
    torch.save(checkpoint, MODEL_DIR / "wagtf_net.pt")
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    (RESULTS_DIR / "experiment_config.json").write_text(json.dumps({
        "seed": 23, "lookback_hours": LOOKBACK, "horizons_hours": HORIZONS,
        "train_end": SPLITS["train_end"], "validation_end": SPLITS["val_end"],
        "test_start": "2016-07-01 00:00", "episode_threshold_ug_m3": THRESHOLD,
        "additional_missingness_stress": "Randomly mask 20% of observed held-out PM2.5 input values; indicate mask explicitly; no retraining.",
        "coordinate_note": "Approximate station coordinates from published literature; see docs/sources.md."
    }, indent=2))
    example_anchor = int(test_a[0])
    plot_results(metrics, (q, prob, actual, mask, times, example_anchor), station_names, coords)
    print("\nTest metrics")
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print(f"\nSaved model: {MODEL_DIR / 'wagtf_net.pt'}")
    print(f"Saved evaluation artifacts in {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
