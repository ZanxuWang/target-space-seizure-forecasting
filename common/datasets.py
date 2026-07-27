"""datasets.py

EEG-only datasets for classifier and diffusion training.

- HorizonSpecDataset: serves single-subwindow spectrograms for classifier training.
  Each sample is a 3s spectrogram (e.g. 0-3s post-onset target, or a preictal
  horizon like -6 to -3s), normalized for ResNet18 consumption.

- DiffusionDataset: serves concatenated (cond | target) spectrograms for
  infill-diffusion training, together with a mask that's True for the
  conditioning frames and False for the target frames.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DATA_ROOT, RT_REGION, FS, HOP_LENGTH,
    ONSET_IDX, ROW_SAMPLES, SUBWINDOW_SAMPLES, F_KEEP,
)
from .spectrogram import (
    signal_to_spectrogram, signals_to_spectrograms,
    compute_zscore_stats, zscore_apply,
    spec_to_classifier_input_np,
)


# -----------------------------------------------------------------------------
# Raw data loading
# -----------------------------------------------------------------------------

def _mouse_dir(mouse_id: str, region: str = RT_REGION) -> str:
    return os.path.join(DATA_ROOT, region, mouse_id)


def load_mouse_eeg(mouse_id: str, region: str = RT_REGION) -> tuple[np.ndarray, np.ndarray]:
    """Load eeg.csv (rows, samples) and labels.csv for one mouse."""
    d = _mouse_dir(mouse_id, region)
    X = pd.read_csv(os.path.join(d, "eeg.csv"), header=None).values.astype(np.float32)
    y = pd.read_csv(os.path.join(d, "labels.csv"), header=None).values.squeeze().astype(np.int64)
    if X.shape[1] != ROW_SAMPLES:
        raise ValueError(
            f"Unexpected row length for {mouse_id}: got {X.shape[1]}, expected {ROW_SAMPLES}"
        )
    if len(y) != len(X):
        raise ValueError(f"Label/row count mismatch for {mouse_id}")
    return X, y


def load_mice_eeg(mouse_ids: list, region: str = RT_REGION) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate rows and labels across multiple mice. Returns (X, y, mouse_idx)."""
    Xs, ys, mids = [], [], []
    for i, m in enumerate(mouse_ids):
        X, y = load_mouse_eeg(m, region)
        Xs.append(X)
        ys.append(y)
        mids.append(np.full(len(y), i, dtype=np.int64))
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(mids, axis=0),
    )


# -----------------------------------------------------------------------------
# Sub-window extraction (onset-relative)
# -----------------------------------------------------------------------------

def subwindow_slice(start_sec: float, end_sec: float) -> tuple[int, int]:
    """Convert onset-relative seconds -> absolute sample indices in a row."""
    s = int(round(ONSET_IDX + start_sec * FS))
    e = int(round(ONSET_IDX + end_sec * FS))
    if s < 0 or e > ROW_SAMPLES or s >= e:
        raise ValueError(
            f"Sub-window [{start_sec}, {end_sec}]s maps to [{s}, {e}] "
            f"out of row range [0, {ROW_SAMPLES}]"
        )
    return s, e


def extract_subwindow(rows: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    """Slice (N, ROW_SAMPLES) -> (N, L) for the given onset-relative seconds."""
    s, e = subwindow_slice(start_sec, end_sec)
    return rows[:, s:e]


# -----------------------------------------------------------------------------
# Classifier dataset (one subwindow -> one spectrogram)
# -----------------------------------------------------------------------------

class HorizonSpecDataset(Dataset):
    """Per-subwindow spectrogram dataset for classifier training.

    Each sample returns (spec_tensor [1,H,W], label) where the spec has been
    resized to CLASSIFIER_IMG_HW (typically 224x224) for ResNet18 input.
    No cross-sample statistics are used (normalization is per-sample min-max
    inside spec_to_classifier_input_np) so train/val/test can be built
    independently without leakage.
    """

    def __init__(self, rows: np.ndarray, labels: np.ndarray,
                 start_sec: float, end_sec: float):
        subs = extract_subwindow(rows, start_sec, end_sec)
        self.specs = signals_to_spectrograms(subs)          # (N, F, T)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        x = spec_to_classifier_input_np(spec)               # [1, H, W]
        y = int(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.long)


# -----------------------------------------------------------------------------
# Diffusion dataset (cond || target spectrograms with mask)
# -----------------------------------------------------------------------------

class DiffusionDataset(Dataset):
    """Concatenate cond and target 3s spectrograms along the time axis.

    Returns (full_spec [F,T_total], mask [F,T_total], label) where mask is
    True over the conditioning frames and False over the target frames.
    Both specs are log1p-magnitude and then globally z-scored using
    (mean, std) computed on the training fold.

    Parameters
    ----------
    rows : (N, ROW_SAMPLES)
    labels : (N,)
    cond_range : (start_sec, end_sec) onset-relative
    target_range : (start_sec, end_sec) onset-relative, defaults to (0, 3)
    mean, std : optional, if None computed from this dataset's specs
    """

    def __init__(
        self,
        rows: np.ndarray,
        labels: np.ndarray,
        cond_range: tuple,
        target_range: tuple = (0.0, 3.0),
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        cond_sub = extract_subwindow(rows, *cond_range)
        tgt_sub = extract_subwindow(rows, *target_range)

        cond_specs = signals_to_spectrograms(cond_sub)      # (N, F, Tc)
        tgt_specs = signals_to_spectrograms(tgt_sub)        # (N, F, Tt)

        full = np.concatenate([cond_specs, tgt_specs], axis=2)  # (N, F, Tc+Tt)

        if mean is None or std is None:
            mean, std = compute_zscore_stats(full)
        self.mean = float(mean)
        self.std = float(std)

        self.specs = zscore_apply(full, self.mean, self.std)    # (N, F, T)
        self.labels = labels.astype(np.int64)

        # Time-axis split point (frames): conditioning frames are the first
        # cond_specs.shape[2] frames, target frames follow.
        self.cond_frames = int(cond_specs.shape[2])
        self.target_frames = int(tgt_specs.shape[2])
        self.total_frames = self.specs.shape[2]

        # Mask: True = conditioning (kept fixed), False = target (generated)
        mask = np.ones_like(self.specs, dtype=bool)
        mask[:, :, self.cond_frames:] = False
        self.masks = mask

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        spec = torch.from_numpy(self.specs[idx]).float()       # [F, T]
        mask = torch.from_numpy(self.masks[idx])               # [F, T]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return spec, mask, label


class ConditionalTargetDataset(Dataset):
    """Target-only conditional diffusion dataset.

    Returns separate conditioning and target spectrograms:

    - cond_spec:   [F, T_cond] z-scored log1p preictal spectrogram
    - target_spec: [F, T_target] z-scored log1p future target spectrogram
    - label:       class label, used only for training/evaluation losses

    The global z-score stats are computed over the concatenated
    (cond || target) training specs for compatibility with the current
    diffusion representation, but the model receives the two parts separately.
    """

    def __init__(
        self,
        rows: np.ndarray,
        labels: np.ndarray,
        cond_range: tuple,
        target_range: tuple = (0.0, 3.0),
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        cond_sub = extract_subwindow(rows, *cond_range)
        tgt_sub = extract_subwindow(rows, *target_range)

        cond_specs = signals_to_spectrograms(cond_sub)      # (N, F, Tc)
        tgt_specs = signals_to_spectrograms(tgt_sub)        # (N, F, Tt)

        if mean is None or std is None:
            full = np.concatenate([cond_specs, tgt_specs], axis=2)
            mean, std = compute_zscore_stats(full)
        self.mean = float(mean)
        self.std = float(std)

        self.cond_specs = zscore_apply(cond_specs, self.mean, self.std)
        self.target_specs = zscore_apply(tgt_specs, self.mean, self.std)
        self.labels = labels.astype(np.int64)

        self.cond_frames = int(cond_specs.shape[2])
        self.target_frames = int(tgt_specs.shape[2])
        self.freq_bins = int(cond_specs.shape[1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        cond = torch.from_numpy(self.cond_specs[idx]).float()       # [F, Tc]
        target = torch.from_numpy(self.target_specs[idx]).float()   # [F, Tt]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return cond, target, label
