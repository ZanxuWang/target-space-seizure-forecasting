"""Generate delta-AUC statistics for diffusion vs matched preictal classifiers.

The analysis is event-level and paired: each comparison uses the same held-out
test events for the diffusion ensemble and its matched preictal-window
classifier. Confidence intervals use stratified event bootstrap resampling.
P-values use a paired permutation test that swaps method scores within event
under the null hypothesis of exchangeable methods. P-values are corrected with
Benjamini-Hochberg FDR across all planned horizon-level comparisons.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paper_figures.figure_common import (  # type: ignore
        ARTIFACT_DIR,
        METHOD_COLORS,
        OUTPUT_DIR,
        ROOT,
        benjamini_hochberg,
        configure_matplotlib,
        display_region,
        load_test_probs,
        p_to_stars,
        save_figure,
    )
else:
    from .figure_common import (
        ARTIFACT_DIR,
        METHOD_COLORS,
        OUTPUT_DIR,
        ROOT,
        benjamini_hochberg,
        configure_matplotlib,
        display_region,
        load_test_probs,
        p_to_stars,
        save_figure,
    )


FULL_MM_MODALITY = "eeg_emg_photo"
EEG_MODALITY = "eeg"


@dataclass(frozen=True)
class Case:
    region: str
    mouse: str
    diffusion_version: str
    eeg_classifier_version: str
    mm_classifier_version: str


CASES = (
    Case(
        region="RT",
        mouse="M238",
        diffusion_version="v6_paper_RT_M238_mm_allh",
        eeg_classifier_version="v6_paper_RT_M238_eeg_classifiers_valauc",
        mm_classifier_version="v6_rt_h0_fold_screen",
    ),
    Case(
        region="VPM",
        mouse="M710",
        diffusion_version="v6_paper_VPM_M710_allh",
        eeg_classifier_version="v6_paper_VPM_M710_eeg_classifiers_valauc",
        mm_classifier_version="v6_paper_VPM_classifiers",
    ),
)

COMPARISONS = (
    ("EEG", EEG_MODALITY, "EEG classifier", "EEG diffusion", METHOD_COLORS["eeg_diffusion"]),
    ("Full-MM", FULL_MM_MODALITY, "Full-MM classifier", "Full-MM diffusion", METHOD_COLORS["mm_diffusion"]),
)


def _fold_dir(mouse: str) -> str:
    return f"fold_{mouse}"


def _ensemble_path(case: Case, horizon_idx: int, modality: str) -> Path:
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


def _eeg_classifier_path(case: Case, horizon_idx: int) -> Path:
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


def _mm_classifier_path(case: Case, horizon_idx: int) -> Path:
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


def _classifier_path(case: Case, horizon_idx: int, modality: str) -> Path:
    if modality == EEG_MODALITY:
        return _eeg_classifier_path(case, horizon_idx)
    if modality == FULL_MM_MODALITY:
        return _mm_classifier_path(case, horizon_idx)
    raise ValueError(f"Unsupported modality: {modality}")


def _auc_fast(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _delta_auc(y: np.ndarray, diffusion_prob: np.ndarray, baseline_prob: np.ndarray) -> float:
    return _auc_fast(y, diffusion_prob) - _auc_fast(y, baseline_prob)


def _stratified_bootstrap_ci(
    y: np.ndarray,
    diffusion_prob: np.ndarray,
    baseline_prob: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    class0 = np.flatnonzero(y == 0)
    class1 = np.flatnonzero(y == 1)
    deltas = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(class0, size=len(class0), replace=True),
            rng.choice(class1, size=len(class1), replace=True),
        ])
        deltas[draw] = _delta_auc(y[idx], diffusion_prob[idx], baseline_prob[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_permutation_p(
    y: np.ndarray,
    diffusion_prob: np.ndarray,
    baseline_prob: np.ndarray,
    observed_delta: float,
    rng: np.random.Generator,
    n_permutation: int,
) -> float:
    null_abs = np.empty(n_permutation, dtype=float)
    for draw in range(n_permutation):
        swap = rng.random(len(y)) < 0.5
        perm_diffusion = np.where(swap, baseline_prob, diffusion_prob)
        perm_baseline = np.where(swap, diffusion_prob, baseline_prob)
        null_abs[draw] = abs(_delta_auc(y, perm_diffusion, perm_baseline))
    return float((np.sum(null_abs >= abs(observed_delta)) + 1) / (n_permutation + 1))


def _analyze(
    n_bootstrap: int,
    n_permutation: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for case in CASES:
        for modality_label, modality, baseline_label, diffusion_label, _ in COMPARISONS:
            for horizon_idx in range(6):
                y_diff, p_diff = load_test_probs(_ensemble_path(case, horizon_idx, modality))
                y_base, p_base = load_test_probs(_classifier_path(case, horizon_idx, modality))
                if not np.array_equal(y_diff, y_base):
                    raise RuntimeError(
                        f"Label mismatch for {case.region}/{case.mouse}, "
                        f"{modality_label}, h{horizon_idx}"
                    )
                observed_delta = _delta_auc(y_diff, p_diff, p_base)
                ci_low, ci_high = _stratified_bootstrap_ci(
                    y_diff, p_diff, p_base, rng, n_bootstrap
                )
                p_value = _paired_permutation_p(
                    y_diff, p_diff, p_base, observed_delta, rng, n_permutation
                )
                rows.append({
                    "cohort": display_region(case.region),
                    "mouse": case.mouse,
                    "modality": modality_label,
                    "horizon": f"h{horizon_idx}",
                    "horizon_idx": horizon_idx,
                    "baseline_method": baseline_label,
                    "diffusion_method": diffusion_label,
                    "baseline_auc": _auc_fast(y_base, p_base),
                    "diffusion_auc": _auc_fast(y_diff, p_diff),
                    "delta_auc": observed_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_value,
                    "n_nonictal": int(np.sum(y_diff == 0)),
                    "n_ictal": int(np.sum(y_diff == 1)),
                })
    q_values = benjamini_hochberg([float(row["permutation_p"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_bh"] = float(q_value)
        row["significance"] = p_to_stars(float(q_value))
    return rows


def _write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cohort",
        "mouse",
        "modality",
        "horizon",
        "baseline_method",
        "diffusion_method",
        "baseline_auc",
        "diffusion_auc",
        "delta_auc",
        "ci_low",
        "ci_high",
        "permutation_p",
        "q_bh",
        "significance",
        "n_nonictal",
        "n_ictal",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"wrote {path}")


def _row_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (str(row["cohort"]), str(row["modality"]), int(row["horizon_idx"])): row
        for row in rows
    }


def draw(rows: list[dict[str, object]], out_dir: Path) -> None:
    lookup = _row_lookup(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    x = np.arange(6)
    offsets = {"EEG": -0.075, "Full-MM": 0.075}
    label_colors = {label: color for label, _, _, _, color in COMPARISONS}

    ci_values = np.asarray([[float(row["ci_low"]), float(row["ci_high"])] for row in rows])
    ymin = min(-0.06, float(np.nanmin(ci_values[:, 0])) - 0.025)
    ymax = max(0.08, float(np.nanmax(ci_values[:, 1])) + 0.055)

    mouse_by_cohort = {
        display_region(case.region): case.mouse for case in CASES
    }
    for ax, cohort in zip(axes, ("RT", "TC")):
        for modality_label, _, _, _, _ in COMPARISONS:
            ordered = [lookup[(cohort, modality_label, h)] for h in range(6)]
            y = np.asarray([float(row["delta_auc"]) for row in ordered])
            lo = np.asarray([float(row["ci_low"]) for row in ordered])
            hi = np.asarray([float(row["ci_high"]) for row in ordered])
            xpos = x + offsets[modality_label]
            ax.errorbar(
                xpos,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o",
                markersize=4.0,
                capsize=2.4,
                linewidth=1.45,
                elinewidth=1.0,
                color=label_colors[modality_label],
                label=f"{modality_label} diffusion - classifier",
            )
            mean_delta = float(np.mean(y))
            ax.text(
                xpos[-1] + 0.19,
                y[-1],
                f"mean {mean_delta:+.3f}",
                color=label_colors[modality_label],
                fontsize=6.2,
                va="center",
            )
            for point_x, point_y, point_hi, row in zip(xpos, y, hi, ordered):
                if float(row["q_bh"]) < 0.05:
                    ax.text(
                        point_x,
                        point_hi + 0.012,
                        str(row["significance"]),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=label_colors[modality_label],
                        fontweight="bold",
                    )

        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$h_{h}$" for h in range(6)])
        ax.set_xlim(-0.45, 5.62)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Preictal horizon")
        ax.set_title(
            f"{cohort} held-out test mouse {mouse_by_cohort[cohort]}",
            fontweight="bold",
        )
        ax.grid(axis="y", color="#EAEAEA", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"$\Delta$AUC: diffusion $-$ preictal classifier")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        "Paired event-level AUC gain over matched preictal-window classifiers",
        y=1.14,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Error bars: stratified event-bootstrap 95% CI. Stars: paired permutation test, BH-FDR corrected.",
        ha="center",
        fontsize=6.6,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=0.8)
    save_figure(fig, out_dir / "figure3_delta_auc_statistics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--n_permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    rows = _analyze(
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        seed=args.seed,
    )
    _write_table(rows, args.out_dir / "figure3_delta_auc_statistics.tsv")
    draw(rows, args.out_dir)


if __name__ == "__main__":
    main()
