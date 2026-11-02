"""Vectorised feature extraction for fixed-length sensor windows.

The same extractor is applied to every raw-signal dataset so that differences between
datasets come from the data, not from dataset-specific feature engineering.

For each channel (and for the Euclidean magnitude of every 3-axis sensor):
  time domain : mean, std, min, max, median, q25, q75, IQR, MAD, RMS, skewness,
                kurtosis, mean-crossing rate, std of first difference (jerk proxy)
  frequency   : dominant frequency, spectral centroid, normalised spectral entropy,
                relative band power in [0,1), [1,3), [3,5) and [5, fs/2] Hz
For each 3-axis sensor: Pearson correlations between the three axes.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TIME_FEATS = ["mean", "std", "min", "max", "median", "q25", "q75", "iqr", "mad", "rms",
              "skew", "kurt", "mcr", "diffstd"]
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, np.inf)]
FREQ_FEATS = ["domfreq", "centroid", "spentropy"] + [f"band{lo:g}-{hi:g}" for lo, hi in BANDS]


def _channel_features(W: np.ndarray, fs: float) -> np.ndarray:
    """W: (n_windows, length, n_channels) -> (n_windows, n_channels, n_feats)."""
    W = W.astype(np.float64, copy=False)
    mean = W.mean(axis=1)
    std = W.std(axis=1)
    q25, med, q75 = np.percentile(W, [25, 50, 75], axis=1)
    mad = np.median(np.abs(W - med[:, None, :]), axis=1)
    rms = np.sqrt((W ** 2).mean(axis=1))
    with np.errstate(all="ignore"):
        skew = stats.skew(W, axis=1, bias=False)
        kurt = stats.kurtosis(W, axis=1, bias=False)
    centred = W - mean[:, None, :]
    signs = np.signbit(centred)
    mcr = (signs[:, 1:, :] != signs[:, :-1, :]).mean(axis=1)
    diffstd = np.diff(W, axis=1).std(axis=1)
    time = [mean, std, W.min(axis=1), W.max(axis=1), med, q25, q75, q75 - q25, mad, rms,
            skew, kurt, mcr, diffstd]

    L = W.shape[1]
    spec = np.abs(np.fft.rfft(centred, axis=1)) ** 2          # power spectrum, DC removed
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    spec = spec[:, 1:, :]
    freqs = freqs[1:]
    total = spec.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    p = spec / safe_total[:, None, :]
    domfreq = freqs[np.argmax(spec, axis=1)]
    centroid = (p * freqs[None, :, None]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1) / np.log(max(len(freqs), 2))
    bands = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bands.append(p[:, m, :].sum(axis=1) if m.any() else np.zeros_like(mean))
    freq = np.stack([domfreq, centroid, ent] + bands, axis=-1)
    freq[total <= 0] = 0.0                                     # constant signal: no spectrum
    return np.concatenate([np.stack(time, axis=-1), freq], axis=-1)


def extract_features(W: np.ndarray, fs: float, channel_names: list[str],
                     triads: list[tuple[int, int, int]]) -> tuple[np.ndarray, list[str]]:
    """Return (features, names) for windows W of shape (n, length, n_channels).

    triads: index triples of channels that form one 3-axis sensor (e.g. acc x/y/z).
    """
    n = W.shape[0]
    mags, mag_names = [], []
    for (a, b, c) in triads:
        mags.append(np.sqrt(W[:, :, a] ** 2 + W[:, :, b] ** 2 + W[:, :, c] ** 2))
        base = channel_names[a].rsplit("_", 1)[0]
        mag_names.append(f"{base}_mag")
    full = np.concatenate([W] + [m[:, :, None] for m in mags], axis=2) if mags else W
    names_ch = list(channel_names) + mag_names
    cf = _channel_features(full, fs)                      # (n, C, F)
    feat_names = [f"{ch}__{f}" for ch in names_ch for f in TIME_FEATS + FREQ_FEATS]
    parts = [cf.reshape(n, -1)]
    corr_names = []
    for (a, b, c) in triads:
        base = channel_names[a].rsplit("_", 1)[0]
        for i, j, tag in [(a, b, "xy"), (a, c, "xz"), (b, c, "yz")]:
            x = W[:, :, i] - W[:, :, i].mean(axis=1, keepdims=True)
            y = W[:, :, j] - W[:, :, j].mean(axis=1, keepdims=True)
            den = np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))
            with np.errstate(all="ignore"):
                r = np.where(den > 0, (x * y).sum(axis=1) / den, 0.0)
            parts.append(r[:, None])
            corr_names.append(f"{base}__corr_{tag}")
    X = np.concatenate(parts, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, feat_names + corr_names
