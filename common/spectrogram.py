"""spectrogram.py

Unified, single-pass spectrogram pipeline (EEG-only).

Pipeline stages:
    1. robust_normalize_trace   – per-sample median/IQR scaling of the raw 1D signal
    2. stft -> |.| -> log1p      – scipy ShortTimeFFT, magnitude only, log compression
    3. consumer-specific norm    – either global z-score (diffusion) or per-sample
                                   min-max to [0,1] + bilinear resize (classifier)

The legacy pipeline stacked three normalizations sometimes; here each consumer
picks exactly one downstream norm and the intermediate (log-magnitude) form is
what gets cached / stored.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import SFT, KEEP_FREQ_IDX, F_KEEP, CLASSIFIER_IMG_HW


# -----------------------------------------------------------------------------
# Raw signal normalization
# -----------------------------------------------------------------------------

def robust_normalize_trace(x: np.ndarray) -> np.ndarray:
    """Robustly normalize a 1D signal by median and IQR."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    scale = q75 - q25
    if scale == 0:
        scale = 1.0
    return (x - med) / scale


# -----------------------------------------------------------------------------
# Signal -> log-magnitude spectrogram
# -----------------------------------------------------------------------------

def signal_to_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Convert a 1D signal into a log1p-magnitude spectrogram.

    Returns
    -------
    spec : np.ndarray, shape (F_KEEP, T)
        Frequency bins up to MAX_FREQ_HZ, log1p of |STFT|.
    """
    signal = robust_normalize_trace(signal)
    S = SFT.stft(signal, padding="zeros")
    mag = np.abs(S[KEEP_FREQ_IDX, :]).astype(np.float32)
    return np.log1p(mag)


def signals_to_spectrograms(signals: np.ndarray) -> np.ndarray:
    """Vectorized wrapper. Input (N, L) -> output (N, F_KEEP, T)."""
    specs = [signal_to_spectrogram(row) for row in signals]
    return np.stack(specs, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Consumer-specific normalizations
# -----------------------------------------------------------------------------

def compute_zscore_stats(specs: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) over all elements of a (N,F,T) spectrogram array."""
    mean = float(np.mean(specs))
    std = float(np.std(specs))
    if std < 1e-8:
        std = 1.0
    return mean, std


def zscore_apply(specs: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply global z-score (used for diffusion inputs)."""
    return ((specs - mean) / (std + 1e-8)).astype(np.float32)


def zscore_invert(specs: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert global z-score back to log1p-magnitude domain."""
    return specs * std + mean


# -----------------------------------------------------------------------------
# Classifier-side preprocessing (differentiable, works on torch tensors too)
# -----------------------------------------------------------------------------

def spec_to_classifier_input_np(spec: np.ndarray,
                                out_hw: tuple = CLASSIFIER_IMG_HW) -> torch.Tensor:
    """Numpy -> torch [1,1,H,W] for ResNet18. Per-sample min-max + bilinear."""
    x = torch.from_numpy(np.asarray(spec, dtype=np.float32))[None, None]
    x = _minmax_and_resize(x, out_hw)
    return x[0]  # [1, H, W]


def spec_to_classifier_input_torch(
    spec: torch.Tensor,
    out_hw: tuple = CLASSIFIER_IMG_HW,
) -> torch.Tensor:
    """Torch [B,F,T] (or [B,1,F,T]) -> torch [B,1,H,W], differentiable.

    Used for the auxiliary classifier loss during diffusion training where
    gradients need to flow from the classifier all the way back into the UNet.
    """
    if spec.dim() == 3:
        spec = spec.unsqueeze(1)
    return _minmax_and_resize(spec, out_hw)


def _minmax_and_resize(x: torch.Tensor, out_hw: tuple) -> torch.Tensor:
    """Per-sample min-max to [0,1] then bilinear resize."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    mn = x_flat.min(dim=1, keepdim=True).values
    mx = x_flat.max(dim=1, keepdim=True).values
    x_flat = (x_flat - mn) / (mx - mn + 1e-8)
    x = x_flat.view_as(x)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x
