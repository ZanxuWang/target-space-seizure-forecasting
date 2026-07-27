"""Generate final-paper target-space diagnostic panel.

The top row shows PCA projections of frozen target-classifier features.
The bottom row shows class-conditional score distributions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paper_figures.figure_common import (  # type: ignore
        CLASS_COLORS,
        FIGURE_DATA_DIR,
        OUTPUT_DIR,
        configure_matplotlib,
        display_region,
        save_figure,
    )
else:
    from .figure_common import (
        CLASS_COLORS,
        FIGURE_DATA_DIR,
        OUTPUT_DIR,
        configure_matplotlib,
        display_region,
        save_figure,
    )


@dataclass(frozen=True)
class DiagnosticCase:
    region: str
    mouse: str
    alignment_npz: str


CASES = (
    DiagnosticCase("RT", "M238", "RT_M238_target_space_alignment_data.npz"),
    DiagnosticCase("VPM", "M710", "VPM_M710_target_space_alignment_data.npz"),
)

METHODS = (
    ("Preictal\nwindow", "direct_probs"),
    ("Real target", "real_probs"),
    ("Diffusion", "ensemble_probs"),
)


def _load_case(case: DiagnosticCase, out_dir: Path) -> dict[str, np.ndarray]:
    path = FIGURE_DATA_DIR / case.alignment_npz
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _confidence_ellipse(
    points: np.ndarray,
    ax: plt.Axes,
    color: str,
    linestyle: str,
    level: float = 1.665,
) -> None:
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * level * np.sqrt(np.maximum(values, 1e-12))
    ax.add_patch(Ellipse(
        points.mean(axis=0),
        width,
        height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=1.15,
        linestyle=linestyle,
        alpha=0.9,
    ))


def draw(out_dir: Path) -> None:
    all_data = {case.region: _load_case(case, out_dir) for case in CASES}

    regions = [case.region for case in CASES]
    fig, axes = plt.subplots(
        2,
        len(regions),
        figsize=(7.25, 5.55),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )
    rng = np.random.default_rng(2026)

    for col, case in enumerate(CASES):
        data = all_data[case.region]
        labels = data["labels"].astype(int)
        generated_labels = data["generated_labels"].astype(int)
        real_features = data["real_features"].astype(float)
        generated_features = data["generated_features"].astype(float)

        joint = np.concatenate([real_features, generated_features], axis=0)
        joint = StandardScaler().fit_transform(joint)
        projection = PCA(n_components=2, random_state=0).fit_transform(joint)
        real_xy = projection[:len(real_features)]
        generated_xy = projection[len(real_features):]

        ax = axes[0, col]
        for label in (0, 1):
            color = CLASS_COLORS[label]
            real_points = real_xy[labels == label]
            generated_points = generated_xy[generated_labels == label]
            ax.scatter(
                generated_points[:, 0],
                generated_points[:, 1],
                s=7,
                marker="x",
                linewidth=0.45,
                color=color,
                alpha=0.23,
            )
            ax.scatter(
                real_points[:, 0],
                real_points[:, 1],
                s=9,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.62,
            )
            _confidence_ellipse(real_points, ax, color, "-")
            _confidence_ellipse(generated_points, ax, color, "--")
        ax.set_title(
            f"{display_region(case.region)}: frozen target-classifier features",
            fontweight="bold",
        )
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.grid(color="#EEEEEE", linewidth=0.45)

        ax = axes[1, col]
        positions: list[int] = []
        box_data: list[np.ndarray] = []
        colors: list[str] = []
        for method_idx, (_, key) in enumerate(METHODS):
            values = data[key].astype(float)
            for label in (0, 1):
                positions.append(method_idx * 3 + label + 1)
                box_data.append(values[labels == label])
                colors.append(CLASS_COLORS[label])
        boxplot = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            boxprops={"linewidth": 0.7},
        )
        for patch, color, values, position in zip(boxplot["boxes"], colors, box_data, positions):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
            sample_count = min(70, len(values))
            chosen = rng.choice(len(values), sample_count, replace=False)
            x = position + rng.uniform(-0.18, 0.18, size=sample_count)
            ax.scatter(x, values[chosen], s=5, color=color, alpha=0.35, linewidth=0)

        ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel(r"$p_{\mathrm{ictal}}$")
        ax.set_xticks(positions)
        ax.set_xticklabels(["N", "I", "N", "I", "N", "I"], fontsize=7)
        for center, (method_label, _) in zip((1.5, 4.5, 7.5), METHODS):
            ax.text(
                center,
                -0.16,
                method_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
            )
        ax.set_title("Cohort-wide score distributions", fontweight="bold")
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[0],
               markeredgecolor="none", label="Non-ictal", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[1],
               markeredgecolor="none", label="Ictal", markersize=5),
        Line2D([0], [0], marker="o", color="#444444", linestyle="none",
               label="Real target", markersize=4),
        Line2D([0], [0], marker="x", color="#444444", linestyle="none",
               label="Generated sample", markersize=4),
        Line2D([0], [0], color="#444444", linestyle="-", label="Real 75% ellipse"),
        Line2D([0], [0], color="#444444", linestyle="--", label="Generated 75% ellipse"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=7,
    )
    fig.suptitle(
        "Target-space feature comparison and classifier-score separation at h0",
        fontsize=10.5,
        fontweight="bold",
        y=1.028,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.0, h_pad=1.2)

    save_figure(
        fig,
        out_dir / "figure8_RT_TC_target_space_diagnostic",
        aliases=(out_dir / "RT_VPM_target_space_diagnostic",),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    draw(args.out_dir)


if __name__ == "__main__":
    main()
