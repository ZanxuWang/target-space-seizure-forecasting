"""Generate final-paper ROC panels for Figures 4 and 5.

The script redraws the six-horizon ROC grids from saved test-set
probability arrays. It does not retrain classifiers or resample diffusion
models.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paper_figures.figure_common import (  # type: ignore
        ARTIFACT_DIR,
        FIGURE_DATA_DIR,
        METHOD_COLORS,
        OUTPUT_DIR,
        ROOT,
        configure_matplotlib,
        display_region,
        load_test_probs,
        save_figure,
    )
else:
    from .figure_common import (
        ARTIFACT_DIR,
        FIGURE_DATA_DIR,
        METHOD_COLORS,
        OUTPUT_DIR,
        ROOT,
        configure_matplotlib,
        display_region,
        load_test_probs,
        save_figure,
    )

from common.config import HORIZONS


FULL_MM_MODALITY = "eeg_emg_photo"
EEG_MODALITY = "eeg"


@dataclass(frozen=True)
class RocCase:
    figure_no: int
    region: str
    mouse: str
    diffusion_version: str
    eeg_classifier_version: str
    mm_classifier_version: str
    target_alignment_npz: str


CASES = (
    RocCase(
        figure_no=4,
        region="RT",
        mouse="M238",
        diffusion_version="v6_paper_RT_M238_mm_allh",
        eeg_classifier_version="v6_paper_RT_M238_eeg_classifiers_valauc",
        mm_classifier_version="v6_rt_h0_fold_screen",
        target_alignment_npz="RT_M238_target_space_alignment_data.npz",
    ),
    RocCase(
        figure_no=5,
        region="VPM",
        mouse="M710",
        diffusion_version="v6_paper_VPM_M710_allh",
        eeg_classifier_version="v6_paper_VPM_M710_eeg_classifiers_valauc",
        mm_classifier_version="v6_paper_VPM_classifiers",
        target_alignment_npz="VPM_M710_target_space_alignment_data.npz",
    ),
)


def _fold_dir(mouse: str) -> str:
    return f"fold_{mouse}"


def _ensemble_path(case: RocCase, horizon_idx: int, modality: str) -> Path:
    root = ARTIFACT_DIR / case.diffusion_version / "ensembles"
    if case.region == "RT":
        return root / _fold_dir(case.mouse) / f"horizon_{horizon_idx}" / modality / "ensemble_probs.npz"
    return (
        root
        / case.region
        / _fold_dir(case.mouse)
        / f"horizon_{horizon_idx}"
        / modality
        / "ensemble_probs.npz"
    )


def _eeg_classifier_path(case: RocCase, horizon_idx: int) -> Path:
    root = ARTIFACT_DIR / case.eeg_classifier_version / "classifiers"
    if case.region == "RT":
        return root / _fold_dir(case.mouse) / f"horizon_{horizon_idx}" / "final_eval_probs.npz"
    return (
        root
        / case.region
        / _fold_dir(case.mouse)
        / f"horizon_{horizon_idx}"
        / "final_eval_probs.npz"
    )


def _mm_classifier_path(case: RocCase, horizon_idx: int) -> Path:
    return (
        ARTIFACT_DIR
        / case.mm_classifier_version
        / "classifiers_mm"
        / case.region
        / _fold_dir(case.mouse)
        / FULL_MM_MODALITY
        / f"horizon_{horizon_idx}"
        / "final_eval_probs.npz"
    )


def _target_probs(case: RocCase) -> tuple[np.ndarray, np.ndarray]:
    with np.load(FIGURE_DATA_DIR / case.target_alignment_npz) as payload:
        return payload["labels"].astype(int), payload["real_probs"].astype(float)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def _series(case: RocCase, horizon_idx: int) -> list[dict]:
    items: list[dict] = []
    for modality, label, color, linestyle, linewidth in (
        (FULL_MM_MODALITY, "MM diffusion", METHOD_COLORS["mm_diffusion"], "-", 2.05),
        (EEG_MODALITY, "EEG diffusion", METHOD_COLORS["eeg_diffusion"], "-", 1.75),
    ):
        y, p = load_test_probs(_ensemble_path(case, horizon_idx, modality))
        fpr, tpr, _ = roc_curve(y, p)
        items.append({
            "label": label,
            "auc": _auc(y, p),
            "fpr": fpr,
            "tpr": tpr,
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
        })

    for label, path, color in (
        ("EEG preictal window", _eeg_classifier_path(case, horizon_idx), METHOD_COLORS["eeg_preictal"]),
        ("MM preictal window", _mm_classifier_path(case, horizon_idx), METHOD_COLORS["mm_preictal"]),
    ):
        y, p = load_test_probs(path)
        fpr, tpr, _ = roc_curve(y, p)
        items.append({
            "label": label,
            "auc": _auc(y, p),
            "fpr": fpr,
            "tpr": tpr,
            "color": color,
            "linestyle": "--",
            "linewidth": 1.25,
        })

    y, p = _target_probs(case)
    fpr, tpr, _ = roc_curve(y, p)
    items.append({
        "label": "Target reference",
        "auc": _auc(y, p),
        "fpr": fpr,
        "tpr": tpr,
        "color": METHOD_COLORS["target"],
        "linestyle": ":",
        "linewidth": 1.45,
    })
    return items


def _sec(value: float) -> str:
    return f"{value:g}"


def _draw_axis(ax: plt.Axes, series: list[dict], horizon_idx: int, show_xlabel: bool, show_ylabel: bool) -> None:
    for item in series:
        ax.plot(
            item["fpr"],
            item["tpr"],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=item["linewidth"],
            label=f"{item['label']} {item['auc']:.2f}",
        )
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=0.65, linestyle=(0, (2, 2)))
    start, end = HORIZONS[horizon_idx]
    ax.set_title(f"$h_{horizon_idx}$: {_sec(start)} to {_sec(end)} s")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("False-positive rate" if show_xlabel else "")
    ax.set_ylabel("True-positive rate" if show_ylabel else "")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#EDEDED", linewidth=0.45)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=5.0,
        handlelength=1.55,
        labelspacing=0.18,
        borderaxespad=0.2,
    )


def draw_case(case: RocCase, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.9), sharex=True, sharey=True)
    for panel_idx, (ax, horizon_idx) in enumerate(zip(axes.flat, range(6))):
        _draw_axis(
            ax,
            _series(case, horizon_idx),
            horizon_idx,
            show_xlabel=panel_idx >= 3,
            show_ylabel=panel_idx % 3 == 0,
        )
        ax.text(
            0.03,
            0.96,
            chr(ord("a") + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9,
        )

    disp = display_region(case.region)
    fig.suptitle(
        f"{disp} cohort, held-out test mouse {case.mouse}",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.55, h_pad=0.7)

    stem = out_dir / f"figure{case.figure_no}_{disp}_{case.mouse}_roc_performance"
    legacy = out_dir / f"{case.region}_{case.mouse}_roc_all_horizons"
    save_figure(fig, stem, aliases=(legacy,))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    for case in CASES:
        draw_case(case, args.out_dir)


if __name__ == "__main__":
    main()
