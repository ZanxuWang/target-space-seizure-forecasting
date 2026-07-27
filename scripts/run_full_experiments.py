"""Run the complete paper pipeline from processed CSVs.

The default mode prints the commands. Pass ``--execute`` to launch the full
GPU workload. Existing completed classifier/diffusion outputs are skipped by
their underlying scripts, so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
HORIZONS = "0,1,2,3,4,5"
SEEDS = "0,1,2,3,4"


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def dispatch(command: list[str], execute: bool) -> None:
    print("+", command_text(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def module(python: str, name: str, *args: str) -> list[str]:
    return [python, "-m", name, *args]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run instead of only printing commands.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    py = str(Path(args.python).resolve())

    versions = {
        "rt_target": "v6_xattn_m234_h0",
        "rt_eeg_cls": "v6_paper_RT_M238_eeg_classifiers_valauc",
        "rt_mm_cls": "v6_rt_h0_fold_screen",
        "rt_eeg_diff": "v6_paper_RT_M238_eeg_allh",
        "rt_mm_diff": "v6_paper_RT_M238_mm_allh",
        "vpm_cls": "v6_paper_VPM_classifiers",
        "vpm_eeg_cls": "v6_paper_VPM_M710_eeg_classifiers_valauc",
        "vpm_eeg_diff": "v6_paper_VPM_M710_eeg_allh",
        "vpm_mm_diff": "v6_paper_VPM_M710_allh",
    }

    # Target-space classifiers.
    for region, fold, version in (
        ("RT", "M238", versions["rt_target"]),
        ("VPM", "M710", versions["vpm_cls"]),
    ):
        dispatch(
            module(
                py,
                "scripts.train_classifier",
                "--region", region,
                "--fold", fold,
                "--kind", "target",
                "--epochs", "30",
                "--out_root", str(OUTPUTS / version),
            ),
            args.execute,
        )

    # Direct EEG and full-multimodal preictal classifiers.
    for region, fold, eeg_version, mm_version in (
        ("RT", "M238", versions["rt_eeg_cls"], versions["rt_mm_cls"]),
        ("VPM", "M710", versions["vpm_eeg_cls"], versions["vpm_cls"]),
    ):
        for horizon in range(6):
            dispatch(
                module(
                    py,
                    "scripts.train_classifier",
                    "--region", region,
                    "--fold", fold,
                    "--kind", "horizon",
                    "--horizon_idx", str(horizon),
                    "--select_best_by", "val_auc",
                    "--epochs", "30",
                    "--out_root", str(OUTPUTS / eeg_version),
                ),
                args.execute,
            )
            dispatch(
                module(
                    py,
                    "scripts.train_classifier_mm",
                    "--region", region,
                    "--fold", fold,
                    "--kind", "horizon",
                    "--horizon_idx", str(horizon),
                    "--include_emg",
                    "--include_photometry",
                    "--epochs", "30",
                    "--out_root", str(OUTPUTS / mm_version),
                ),
                args.execute,
            )

    # Conditional diffusion: two cohorts x two conditioning settings x
    # six horizons x five seeds. The queue definitions contain the exact
    # architecture, loss, augmentation, DDIM and final-evaluation settings.
    diffusion_jobs = (
        ("RT", "M238", "M229", "anchor_b02", versions["rt_mm_diff"], versions["rt_target"]),
        ("RT", "M238", "M229", "anchor_eeg", versions["rt_eeg_diff"], versions["rt_target"]),
        ("VPM", "M710", "M1079", "anchor_vpm", versions["vpm_mm_diff"], versions["vpm_cls"]),
        ("VPM", "M710", "M1079", "anchor_eeg", versions["vpm_eeg_diff"], versions["vpm_cls"]),
    )
    for region, fold, val_mouse, queue, version, target_version in diffusion_jobs:
        target_path = OUTPUTS / target_version / "classifiers"
        if region != "RT":
            target_path /= region
        target_path = target_path / f"fold_{fold}" / "target" / "best.pth"
        dispatch(
            module(
                py,
                "scripts.v6_loop",
                "--region", region,
                "--fold", fold,
                "--val_holdout_mouse", val_mouse,
                "--horizons", HORIZONS,
                "--seeds", SEEDS,
                "--queue", queue,
                "--max_runs", "0",
                "--out_root", str(OUTPUTS / version),
                "--classifier_ckpt", str(target_path),
                "--python", py,
            ),
            args.execute,
        )

    # Aggregate the EEG and multimodal ensembles with matched baselines.
    aggregate_jobs = (
        (
            versions["rt_mm_diff"],
            "RT",
            "M238",
            versions["rt_eeg_diff"],
            [versions["rt_eeg_cls"], versions["rt_mm_cls"], versions["rt_target"]],
        ),
        (
            versions["vpm_mm_diff"],
            "VPM",
            "M710",
            versions["vpm_eeg_diff"],
            [versions["vpm_eeg_cls"], versions["vpm_cls"]],
        ),
    )
    for version, region, fold, eeg_diffusion, classifier_versions in aggregate_jobs:
        command = module(
            py,
            "scripts.v6_aggregate_results",
            "--version", version,
            "--region", region,
            "--fold", fold,
            "--n_bootstrap", "1000",
            "--diffusion_root", eeg_diffusion,
        )
        for classifier_version in classifier_versions:
            command.extend(["--classifier_root", classifier_version])
        dispatch(command, args.execute)

    for version, region, fold, _, classifier_versions in aggregate_jobs:
        command = module(
            py,
            "scripts.v6_qc_audit",
            "--version", version,
            "--region", region,
            "--fold", fold,
        )
        for classifier_version in classifier_versions:
            command.extend(["--classifier_root", classifier_version])
        dispatch(command, args.execute)

    if args.execute:
        print("Full training and aggregation completed. Generate checkpoint-dependent")
        print("target-space arrays with: python -m scripts.v6_make_generation_figure")
        print("Then render final panels with: python -m scripts.paper_figures.make_all_paper_figures")
    else:
        print("\nDry command listing only. Re-run with --execute to launch the full workload.")


if __name__ == "__main__":
    main()
