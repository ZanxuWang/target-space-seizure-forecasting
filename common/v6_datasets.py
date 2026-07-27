"""v6_datasets.py

Multimodal conditional-diffusion datasets with optional per-sample
standardization and SpecAugment-style augmentation.

The key v6 changes:

- ``norm_mode``:
    - ``"global"``: legacy behavior - subtract a single train-set (mean, std).
    - ``"per_sample"``: each spectrogram is z-scored independently so
      per-mouse scale variation doesn't leak into the input distribution.

- Optional ``include_emg``, ``include_photometry`` flags. EMG is STFT'd
  exactly like EEG and stacked as an extra channel of the past condition.
  Photometry is NOT STFT'd (changes too slowly): we resample its 3 s
  window to the same time-frame grid as the EEG spectrogram and stack as
  a single broadcast "spectro-like" channel of constant value across
  frequency bins (acts as an extra time-series cue with shape [F, T_cond]).

- SpecAugment-style time/frequency masking on past + target at train time.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DATA_ROOT,
    RT_REGION,
    FS,
    HOP_LENGTH,
    ONSET_IDX,
    ROW_SAMPLES,
    SUBWINDOW_SAMPLES,
    F_KEEP,
)
from .datasets import extract_subwindow
from .spectrogram import signals_to_spectrograms


# -----------------------------------------------------------------------------
# Raw multimodal loaders
# -----------------------------------------------------------------------------


def _mouse_dir(mouse_id: str, region: str = RT_REGION) -> str:
    return os.path.join(DATA_ROOT, region, mouse_id)


def _load_one_modality(mouse_id: str, fname: str, region: str = RT_REGION) -> np.ndarray:
    path = os.path.join(_mouse_dir(mouse_id, region), fname)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def load_mouse_multimodal(
    mouse_id: str,
    region: str = RT_REGION,
    include_emg: bool = False,
    include_photometry: bool = False,
) -> dict:
    out = {
        "eeg": _load_one_modality(mouse_id, "eeg.csv", region),
        "labels": pd.read_csv(
            os.path.join(_mouse_dir(mouse_id, region), "labels.csv"), header=None
        ).values.squeeze().astype(np.int64),
    }
    if include_emg:
        out["emg"] = _load_one_modality(mouse_id, "emg.csv", region)
    if include_photometry:
        out["photometry"] = _load_one_modality(mouse_id, "photometry.csv", region)
    for k, arr in out.items():
        if k == "labels":
            continue
        if arr.shape[1] != ROW_SAMPLES:
            raise ValueError(
                f"Unexpected row length for {mouse_id}/{k}: got {arr.shape[1]}, expected {ROW_SAMPLES}"
            )
        if len(arr) != len(out["labels"]):
            raise ValueError(f"Label/row count mismatch for {mouse_id}/{k}")
    return out


def load_mice_multimodal(
    mouse_ids: list,
    region: str = RT_REGION,
    include_emg: bool = False,
    include_photometry: bool = False,
) -> dict:
    bank = {"eeg": [], "labels": [], "mouse_idx": []}
    if include_emg:
        bank["emg"] = []
    if include_photometry:
        bank["photometry"] = []
    for i, m in enumerate(mouse_ids):
        rec = load_mouse_multimodal(m, region, include_emg, include_photometry)
        for k in bank:
            if k == "mouse_idx":
                bank[k].append(np.full(len(rec["labels"]), i, dtype=np.int64))
            else:
                bank[k].append(rec[k])
    return {k: np.concatenate(v, axis=0) for k, v in bank.items()}


# -----------------------------------------------------------------------------
# Spectrogram-side helpers
# -----------------------------------------------------------------------------


def per_sample_zscore(specs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each (F, T) spectrogram independently."""
    s = specs.astype(np.float32)
    flat = s.reshape(s.shape[0], -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True) + eps
    flat = (flat - mu) / sd
    return flat.reshape(s.shape)


def _photometry_summary_channel(
    raw_subwin: np.ndarray,
    F_target: int,
    T_target: int,
) -> np.ndarray:
    """Convert per-sample photometry waveform [N, L] to a fake spectrogram-shaped
    channel by binning into ``T_target`` time bins and broadcasting across F.

    Returns [N, F_target, T_target] (per-sample z-scored on the time-bin axis).
    """
    N, L = raw_subwin.shape
    if L == T_target:
        binned = raw_subwin
    else:
        idx = np.linspace(0, L, T_target + 1).astype(int)
        idx = np.clip(idx, 0, L)
        binned = np.stack(
            [raw_subwin[:, idx[t]:idx[t + 1]].mean(axis=1) for t in range(T_target)],
            axis=1,
        )
    mu = binned.mean(axis=1, keepdims=True)
    sd = binned.std(axis=1, keepdims=True) + 1e-6
    binned = ((binned - mu) / sd).astype(np.float32)
    return np.broadcast_to(binned[:, None, :], (N, F_target, T_target)).copy()


# -----------------------------------------------------------------------------
# SpecAugment
# -----------------------------------------------------------------------------


def spec_augment_torch(
    spec: torch.Tensor,
    n_freq_masks: int = 2,
    max_freq_width: int = 6,
    n_time_masks: int = 2,
    max_time_width: int = 4,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """In-place SpecAugment on a [..., F, T] tensor."""
    if spec.numel() == 0:
        return spec
    F_dim = spec.shape[-2]
    T_dim = spec.shape[-1]
    for _ in range(n_freq_masks):
        w = int(torch.randint(0, max_freq_width + 1, (1,)).item())
        if w == 0 or F_dim - w <= 0:
            continue
        f0 = int(torch.randint(0, F_dim - w, (1,)).item())
        spec[..., f0:f0 + w, :] = mask_value
    for _ in range(n_time_masks):
        w = int(torch.randint(0, max_time_width + 1, (1,)).item())
        if w == 0 or T_dim - w <= 0:
            continue
        t0 = int(torch.randint(0, T_dim - w, (1,)).item())
        spec[..., :, t0:t0 + w] = mask_value
    return spec


# -----------------------------------------------------------------------------
# Main dataset
# -----------------------------------------------------------------------------


class V6MultimodalDataset(Dataset):
    """Multimodal conditional-diffusion dataset for v6.

    Returns per-item:

        past   : [C_past, F, T_cond]  (C_past = 1 + include_emg + include_photometry)
        target : [F, T_target]
        label  : long scalar

    Photometry, when included, is built as a [F, T_cond] channel summarising
    the past photometry waveform via time-bin averaging then per-sample
    z-score, broadcast across the frequency axis.

    Augmentation (``augment=True``) applies SpecAugment masks to a *copy*
    of past and target each ``__getitem__`` call.
    """

    def __init__(
        self,
        data_bank: dict,
        cond_range: tuple,
        target_range: tuple = (0.0, 3.0),
        norm_mode: str = "per_sample",
        global_mean: Optional[float] = None,
        global_std: Optional[float] = None,
        include_emg: bool = False,
        include_photometry: bool = False,
        augment: bool = False,
        aug_freq_masks: int = 2,
        aug_max_freq_width: int = 6,
        aug_time_masks: int = 2,
        aug_max_time_width: int = 4,
    ):
        self.norm_mode = norm_mode
        self.include_emg = include_emg
        self.include_photometry = include_photometry
        self.augment = augment
        self.aug_kwargs = dict(
            n_freq_masks=aug_freq_masks,
            max_freq_width=aug_max_freq_width,
            n_time_masks=aug_time_masks,
            max_time_width=aug_max_time_width,
        )

        # --- EEG past + target ---
        eeg = data_bank["eeg"]
        eeg_cond = extract_subwindow(eeg, *cond_range)
        eeg_tgt = extract_subwindow(eeg, *target_range)
        cond_specs = signals_to_spectrograms(eeg_cond)
        tgt_specs = signals_to_spectrograms(eeg_tgt)

        # Stats
        if norm_mode == "global":
            if global_mean is None or global_std is None:
                full = np.concatenate([cond_specs, tgt_specs], axis=2)
                global_mean = float(np.mean(full))
                global_std = float(np.std(full)) or 1.0
            self.mean = float(global_mean)
            self.std = float(global_std) if global_std > 1e-8 else 1.0
            cond_specs_n = (cond_specs - self.mean) / self.std
            tgt_specs_n = (tgt_specs - self.mean) / self.std
        elif norm_mode == "per_sample":
            cond_specs_n = per_sample_zscore(cond_specs)
            tgt_specs_n = per_sample_zscore(tgt_specs)
            # Bookkeeping (some downstream code expects a scalar pair):
            self.mean = 0.0
            self.std = 1.0
        else:
            raise ValueError(f"Unknown norm_mode {norm_mode!r}")

        # --- EMG STFT past as extra past channel ---
        past_chans = [cond_specs_n.astype(np.float32)[:, None]]
        if include_emg:
            emg = data_bank["emg"]
            emg_cond = extract_subwindow(emg, *cond_range)
            emg_specs = signals_to_spectrograms(emg_cond)
            if norm_mode == "global":
                emg_full = np.concatenate(
                    [emg_specs, signals_to_spectrograms(extract_subwindow(emg, *target_range))],
                    axis=2,
                )
                emg_mean = float(np.mean(emg_full))
                emg_std = float(np.std(emg_full)) or 1.0
                emg_specs_n = (emg_specs - emg_mean) / emg_std
            else:
                emg_specs_n = per_sample_zscore(emg_specs)
            past_chans.append(emg_specs_n.astype(np.float32)[:, None])

        if include_photometry:
            photo = data_bank["photometry"]
            photo_cond = extract_subwindow(photo, *cond_range)
            T_cond = cond_specs_n.shape[2]
            F_dim = cond_specs_n.shape[1]
            photo_chan = _photometry_summary_channel(photo_cond, F_dim, T_cond)
            past_chans.append(photo_chan.astype(np.float32)[:, None])

        # [N, C_past, F, T_cond]
        self.past_specs = np.concatenate(past_chans, axis=1)
        self.target_specs = tgt_specs_n.astype(np.float32)
        self.labels = data_bank["labels"].astype(np.int64)
        self.mouse_idx = data_bank.get("mouse_idx", np.zeros(len(self.labels), dtype=np.int64))

        self.cond_frames = int(cond_specs_n.shape[2])
        self.target_frames = int(tgt_specs_n.shape[2])
        self.freq_bins = int(cond_specs_n.shape[1])
        self.n_past_channels = int(self.past_specs.shape[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        past = torch.from_numpy(self.past_specs[idx]).float()  # [C_past, F, T_cond]
        target = torch.from_numpy(self.target_specs[idx]).float()  # [F, T_target]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        mouse = torch.tensor(int(self.mouse_idx[idx]), dtype=torch.long)
        if self.augment:
            past = past.clone()
            target = target.clone()
            spec_augment_torch(past, **self.aug_kwargs)
            spec_augment_torch(target, **self.aug_kwargs)
        return past, target, label, mouse
