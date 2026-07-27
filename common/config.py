"""config.py

Central configuration for the EEG-only multi-horizon diffusion pipeline.

All STFT parameters, sampling rate, horizon definitions, paths, and mouse IDs
live here so that the rest of the codebase imports a single source of truth.
"""

import os
import numpy as np
from scipy.signal import ShortTimeFFT


# =============================================================================
# Signal / STFT parameters
# =============================================================================
FS: int = 256                     # Sampling rate (Hz)
N_FFT: int = 128                  # STFT window size
HOP_LENGTH: int = 32              # STFT hop length
MAX_FREQ_HZ: float = 60.0         # Upper bound on frequency bins to keep

SFT = ShortTimeFFT.from_window(
    "hann", FS, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
    fft_mode="onesided", phase_shift=None,
)
_KEEP_MASK = SFT.f <= MAX_FREQ_HZ
KEEP_FREQ_IDX = np.where(_KEEP_MASK)[0]
F_FULL: int = len(SFT.f)
F_KEEP: int = int(_KEEP_MASK.sum())


# =============================================================================
# Windowing
# =============================================================================
# Each prepared row is PRE_WINDOW_SEC + POST_WINDOW_SEC long, centered so that
# the seizure onset sits at sample index ONSET_IDX = PRE_WINDOW_SEC * FS.
PRE_WINDOW_SEC: int = 18
POST_WINDOW_SEC: int = 3
ROW_SECONDS: int = PRE_WINDOW_SEC + POST_WINDOW_SEC      # 21
ROW_SAMPLES: int = ROW_SECONDS * FS                      # 5376
ONSET_IDX: int = PRE_WINDOW_SEC * FS                     # 4608

# Target region: always the 3s immediately after onset
TARGET_SEC: tuple = (0.0, 3.0)

# Conditioning horizons (start_sec, end_sec) relative to onset. Each is a 3s
# sub-window to the LEFT of onset. Horizon index 0 = closest to onset.
HORIZONS: list = [
    (-3.0, 0.0),
    (-6.0, -3.0),
    (-9.0, -6.0),
    (-12.0, -9.0),
    (-15.0, -12.0),
    (-18.0, -15.0),
]

SUBWINDOW_SEC: float = 3.0
SUBWINDOW_SAMPLES: int = int(SUBWINDOW_SEC * FS)         # 768


# =============================================================================
# Classifier input shape (ResNet18 expects 224x224)
# =============================================================================
CLASSIFIER_IMG_HW: tuple = (224, 224)


# =============================================================================
# Paths / cohort
# =============================================================================
DATA_ROOT: str = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(DATA_ROOT, "..", "data", "processed"))

RT_REGION: str = "RT"
RT_MICE: list = ["M229", "M233", "M234", "M237", "M238"]

VPM_REGION: str = "VPM"
VPM_MICE: list = ["M710", "M711", "M1079", "M1080"]

# First-class region concept so train scripts can take --region and stay generic.
REGIONS: dict = {
    RT_REGION: RT_MICE,
    VPM_REGION: VPM_MICE,
}


def mice_for_region(region: str) -> list:
    """Return the list of mouse IDs for a region; raises on unknown region."""
    if region not in REGIONS:
        raise ValueError(
            f"Unknown region {region!r}. Available: {sorted(REGIONS.keys())}"
        )
    return REGIONS[region]


def pick_val_mouse(region: str, fold: str) -> str:
    """Deterministic validation-mouse rule for one held-out fold.

    Uses the first non-fold mouse alphabetically among the region's cohort.
    Matches the historical M234 -> val=M229 convention while extending to VPM.
    """
    mice = mice_for_region(region)
    candidates = sorted(m for m in mice if m != fold)
    if not candidates:
        raise ValueError(f"No non-fold mice available for region={region} fold={fold}")
    return candidates[0]

# Pipeline version. Bumping this string redirects all artifacts (classifier,
# diffusion, inference checkpoints/logs/npz files) into a fresh subdir of
# outputs/, so previous versions stay reproducibly intact on disk. See
# VERSIONS.md at the repo root for the changelog.
VERSION: str = "v6_xattn_m234_h0"

# Override via env var if you want to point a single run at a custom version
# directory without editing this file.
VERSION = os.environ.get("PIPELINE_VERSION", VERSION)

_OUTPUTS_BASE: str = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
))
OUTPUTS_ROOT: str = os.path.join(_OUTPUTS_BASE, VERSION)


# =============================================================================
# Training defaults
# =============================================================================
DEFAULT_SEED: int = 42
