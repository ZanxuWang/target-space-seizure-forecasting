"""Generate final-paper class-conditional target-structure panels.

Figures 6 and 7 are redrawn from the compact prototype arrays saved during
the diffusion diagnostic run. This keeps the final plotting step lightweight
and exactly reproducible.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
class GenerationCase:
    figure_no: int
    region: str
    mouse: str
    prototype_npz: str
    legacy_stem: str


CASES = (
    GenerationCase(
        figure_no=6,
        region="RT",
        mouse="M238",
        prototype_npz="RT_M238_generation_quality_prototype_data.npz",
        legacy_stem="RT_M238_generation_quality",
    ),
    GenerationCase(
        figure_no=7,
        region="VPM",
        mouse="M710",
        prototype_npz="VPM_M710_generation_quality_prototype_data.npz",
        legacy_stem="VPM_M710_generation_quality",
    ),
)

ROW_PREFIXES = (
    ("Real targets", "real_targets"),
    ("Generated targets", "generated_targets"),
)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _frequency_ylim(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for _, prefix in ROW_PREFIXES:
        for suffix in ("frequency_effect", "frequency_ci_low", "frequency_ci_high"):
            values.append(np.asarray(arrays[f"{prefix}_{suffix}"]))
    max_abs = max(float(np.nanmax(np.abs(v))) for v in values)
    limit = max(1.75, min(2.8, max_abs + 0.18))
    return -limit, limit


def draw_case(case: GenerationCase, out_dir: Path) -> None:
    arrays = _load_arrays(FIGURE_DATA_DIR / case.prototype_npz)
    frequencies = np.linspace(0, 60, arrays["real_targets_mean_nonictal"].shape[0])
    ylim = _frequency_ylim(arrays)

    fig = plt.figure(figsize=(7.8, 4.75))
    grid = fig.add_gridspec(
        2,
        8,
        width_ratios=[1.0, 1.0, 0.06, 0.15, 1.0, 0.06, 0.18, 1.25],
        wspace=0.16,
        hspace=0.43,
    )
    axes: dict[tuple[int, int], plt.Axes] = {}
    mean_color_axes: list[plt.Axes] = []
    effect_color_axes: list[plt.Axes] = []
    for row in range(2):
        axes[row, 0] = fig.add_subplot(grid[row, 0])
        axes[row, 1] = fig.add_subplot(grid[row, 1])
        mean_color_axes.append(fig.add_subplot(grid[row, 2]))
        axes[row, 2] = fig.add_subplot(grid[row, 4])
        effect_color_axes.append(fig.add_subplot(grid[row, 5]))
        axes[row, 3] = fig.add_subplot(grid[row, 7])

    titles = [
        "Mean non-ictal target",
        "Mean ictal target",
        "Ictal minus non-ictal\nstandardized contrast",
        "Frequency-wise class effect",
    ]

    for row, (row_name, prefix) in enumerate(ROW_PREFIXES):
        mean0 = arrays[f"{prefix}_mean_nonictal"]
        mean1 = arrays[f"{prefix}_mean_ictal"]
        contrast = arrays[f"{prefix}_effect_map"]
        freq_effect = arrays[f"{prefix}_frequency_effect"]
        freq_lower = arrays[f"{prefix}_frequency_ci_low"]
        freq_upper = arrays[f"{prefix}_frequency_ci_high"]

        mean_values = np.concatenate([mean0.ravel(), mean1.ravel()])
        mean_vmin, mean_vmax = np.percentile(mean_values, [2, 98])
        for col, image in enumerate((mean0, mean1)):
            ax = axes[row, col]
            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=mean_vmin,
                vmax=mean_vmax,
                extent=[0, 3, 0, 60],
                interpolation="nearest",
            )
            ax.set_xlabel("Time from onset (s)")
            if col == 0:
                ax.set_ylabel(f"{row_name}\nFrequency (Hz)", fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, width=0.6, pad=1.5)
            if col == 1:
                colorbar = fig.colorbar(im, cax=mean_color_axes[row])
                colorbar.ax.set_title("Normalized\nintensity", fontsize=5.8, pad=3.0)
                colorbar.ax.yaxis.set_ticks_position("right")
                colorbar.ax.yaxis.set_label_position("right")
                colorbar.ax.tick_params(
                    axis="y",
                    labelsize=5.7,
                    length=2,
                    pad=1.0,
                    labelleft=False,
                    labelright=True,
                )

        ax = axes[row, 2]
        effect_image = ax.imshow(
            contrast,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1.5,
            vmax=1.5,
            extent=[0, 3, 0, 60],
            interpolation="nearest",
        )
        ax.set_xlabel("Time from onset (s)")
        ax.set_yticklabels([])
        ax.tick_params(length=2, width=0.6, pad=1.5)
        colorbar = fig.colorbar(effect_image, cax=effect_color_axes[row])
        colorbar.ax.set_title("Cohen's\n$d$", fontsize=5.8, pad=3.0)
        colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1.0)

        ax = axes[row, 3]
        ax.fill_between(
            frequencies,
            freq_lower,
            freq_upper,
            color=CLASS_COLORS[1],
            alpha=0.18,
            linewidth=0,
            label="Bootstrap 95% CI",
        )
        ax.plot(frequencies, freq_effect, color=CLASS_COLORS[1], linewidth=1.5)
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.axhline(0.5, color="#999999", linewidth=0.6, linestyle=":")
        ax.axhline(-0.5, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, 60)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(color="#EEEEEE", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.6, pad=1.5)

        if row == 0:
            for col, title in enumerate(titles):
                axes[row, col].set_title(title, fontweight="bold")

    disp = display_region(case.region)
    fig.suptitle(
        f"{disp} cohort, held-out test mouse {case.mouse}: "
        "class-conditional h0 target structure",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.875)

    stem = out_dir / f"figure{case.figure_no}_{disp}_{case.mouse}_generation_quality"
    aliases = [out_dir / case.legacy_stem]
    if case.region == "VPM":
        aliases.append(out_dir / f"TC_{case.mouse}_generation_quality")
    save_figure(fig, stem, aliases=tuple(aliases))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--case", choices=["all", "RT", "TC"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    selected = CASES
    if args.case == "RT":
        selected = (CASES[0],)
    elif args.case == "TC":
        selected = (CASES[1],)
    for case in selected:
        draw_case(case, args.out_dir)


if __name__ == "__main__":
    main()
