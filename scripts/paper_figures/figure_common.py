"""Shared helpers for final-paper figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "results" / "paper" / "artifacts"
FIGURE_DATA_DIR = ROOT / "results" / "paper" / "figure_data"
OUTPUT_DIR = ROOT / "results" / "paper" / "reproduced_figures"

CLASS_COLORS = {
    0: "#4C78A8",
    1: "#D55E00",
}

METHOD_COLORS = {
    "eeg_preictal": "#4C78A8",
    "eeg_diffusion": "#1F77B4",
    "mm_preictal": "#F2A65A",
    "mm_diffusion": "#C0392B",
    "target": "#27AE60",
}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def display_region(region: str) -> str:
    """Use TC in final-paper labels while preserving VPM on disk internally."""
    return "TC" if region.upper() == "VPM" else region.upper()


def save_figure(fig: plt.Figure, stem: Path, aliases: tuple[Path, ...] = ()) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for base in (stem, *aliases):
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=400)
        print(f"wrote {base.with_suffix('.pdf')}")
        print(f"wrote {base.with_suffix('.png')}")
    plt.close(fig)


def load_test_probs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        y = payload["test_y"].astype(int)
        p = payload["test_probs"].astype(float)
    if len(y) != len(p):
        raise ValueError(f"Mismatched labels/probabilities in {path}")
    return y, p


def p_to_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "n.s."


def benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n, 0, -1):
        idx = rank - 1
        running = min(running, ranked[idx] * n / rank)
        adjusted[order[idx]] = running
    return np.clip(adjusted, 0.0, 1.0)
