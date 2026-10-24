"""Wind conditioned graph and temporal model used in the MOSAIC-Air pilot."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def haversine_and_bearing(coords: np.ndarray):
    """Return destination/source distance (km) and source-to-destination bearing (radians)."""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    n = len(coords)
    distance = np.zeros((n, n), dtype=np.float32)
    bearing = np.zeros((n, n), dtype=np.float32)
    for dest in range(n):
        for source in range(n):
            dlat = lat[dest] - lat[source]
            dlon = lon[dest] - lon[source]
            a = np.sin(dlat / 2) ** 2 + np.cos(lat[source]) * np.cos(lat[dest]) * np.sin(dlon / 2) ** 2
            distance[dest, source] = 6371.0 * 2 * np.arcsin(np.sqrt(a))
            y = np.sin(dlon) * np.cos(lat[dest])
            x = np.cos(lat[source]) * np.sin(lat[dest]) - np.sin(lat[source]) * np.cos(lat[dest]) * np.cos(dlon)
            bearing[dest, source] = np.arctan2(y, x)
    return distance, bearing


class WindAwareGraphTemporalNet(nn.Module):
    """Shared temporal convolution, wind-biased graph attention, quantile/event heads."""

    def __init__(self, n_features: int, coords: np.ndarray, horizons=(1, 6, 24), hidden=32, heads=4):
        super().__init__()
        self.horizons = tuple(horizons)
        self.n_stations = int(len(coords))
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        dist, bearing = haversine_and_bearing(coords)
        self.register_buffer("dist_km", torch.tensor(dist))
        self.register_buffer("bearing", torch.tensor(bearing))
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32))

        self.input_proj = nn.Linear(n_features, hidden)
        self.tconv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.tconv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.coord_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.graph_out = nn.Linear(hidden, hidden)
        self.graph_norm = nn.LayerNorm(hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden * 2, hidden), nn.GELU())
        self.quantile_head = nn.Linear(hidden, len(self.horizons) * 3)
        self.event_head = nn.Linear(hidden, len(self.horizons))

    def forward(self, x, wind):
        # x: [batch, lookback, stations, features]; wind: [batch, stations, (direction-from rad, speed m/s)]
        b, steps, n, _ = x.shape
        z = self.input_proj(x).permute(0, 2, 3, 1).reshape(b * n, self.hidden, steps)
        z = F.gelu(self.tconv1(z))
        z = F.gelu(self.tconv2(z))
        temporal = z[:, :, -1].reshape(b, n, self.hidden)
        # Center and scale coordinates in km so the station encoder is numerically stable.
        coords_km = self.coords.clone()
        coords_km[:, 0] = (coords_km[:, 0] - coords_km[:, 0].mean()) * 111.0
        coords_km[:, 1] = (coords_km[:, 1] - coords_km[:, 1].mean()) * 85.0
        coords_km = coords_km / 40.0
        temporal = temporal + self.coord_proj(coords_km).unsqueeze(0)

        q = self.q_proj(temporal).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(temporal).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(temporal).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,destination,source]

        direction_from = wind[:, :, 0]  # [B,source]
        speed = wind[:, :, 1].clamp(min=0.0, max=12.0)
        downwind = direction_from + math.pi
        delta = self.bearing.unsqueeze(0) - downwind[:, None, :]
        alignment = torch.cos(delta)
        wind_bias = (speed[:, None, :] / 5.0).clamp(max=2.0) * alignment
        distance_bias = -self.dist_km / 45.0
        graph_bias = (0.9 * wind_bias + 0.35 * distance_bias).unsqueeze(1)
        far = self.dist_km > 100.0
        logits = logits + graph_bias
        logits = logits.masked_fill(far[None, None, :, :], -1e4)
        attn = torch.softmax(logits, dim=-1)
        spatial = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.hidden)
        spatial = self.graph_norm(temporal + self.graph_out(spatial))
        fused = self.fusion(torch.cat([temporal, spatial], dim=-1))

        raw = self.quantile_head(fused).view(b, n, len(self.horizons), 3)
        median = raw[..., 0]
        lo = median - F.softplus(raw[..., 1])
        hi = median + F.softplus(raw[..., 2])
        quantiles = torch.stack([lo, median, hi], dim=-1)  # [B,N,H,Q]
        events = self.event_head(fused)  # [B,N,H]
        return quantiles, events
