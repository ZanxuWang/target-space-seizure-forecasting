"""Recompute the paper's event counts and Table 4 AUCs from bundled artifacts.

This is the fast, checkpoint-free integrity test for the public repository.
It reads the processed labels and saved held-out probabilities, recomputes all
reported AUCs, checks the five-seed/80-target ensemble metadata, and emits
machine-readable tables under ``results/paper/tables``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "results" / "paper" / "artifacts"
FIGURE_DATA = ROOT / "results" / "paper" / "figure_data"
TABLES = ROOT / "results" / "paper" / "tables"
MANIFESTS = ROOT / "results" / "paper" / "manifests"
FULL_MM = "eeg_emg_photo"


@dataclass(frozen=True)
class Case:
    region: str
    paper_cohort: str
    mouse: str
    diffusion_version: str
    eeg_classifier_version: str
    mm_classifier_version: str
    target_alignment: str


CASES = (
    Case(
        "RT",
        "RT",
        "M238",
        "v6_paper_RT_M238_mm_allh",
        "v6_paper_RT_M238_eeg_classifiers_valauc",
        "v6_rt_h0_fold_screen",
        "RT_M238_target_space_alignment_data.npz",
    ),
    Case(
        "VPM",
        "TC",
        "M710",
        "v6_paper_VPM_M710_allh",
        "v6_paper_VPM_M710_eeg_classifiers_valauc",
        "v6_paper_VPM_classifiers",
        "VPM_M710_target_space_alignment_data.npz",
    ),
)

# Values printed in Table 4 of the polished paper, in h0...h5 order.
PAPER_AUC = {
    ("RT", "Target classifier"): [0.999] * 6,
    ("RT", "EEG classifier"): [0.763, 0.852, 0.861, 0.595, 0.861, 0.839],
    ("RT", "EEG diffusion"): [0.906, 0.876, 0.873, 0.832, 0.858, 0.809],
    ("RT", "Full-MM classifier"): [0.785, 0.856, 0.816, 0.834, 0.768, 0.844],
    ("RT", "Full-MM diffusion"): [0.903, 0.862, 0.888, 0.877, 0.892, 0.855],
    ("TC", "Target classifier"): [0.923] * 6,
    ("TC", "EEG classifier"): [0.715, 0.731, 0.707, 0.755, 0.749, 0.735],
    ("TC", "EEG diffusion"): [0.748, 0.764, 0.758, 0.755, 0.768, 0.723],
    ("TC", "Full-MM classifier"): [0.726, 0.723, 0.684, 0.620, 0.548, 0.576],
    ("TC", "Full-MM diffusion"): [0.777, 0.729, 0.754, 0.714, 0.742, 0.694],
}


def load_probs(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        y = payload["test_y"].astype(int)
        p = payload["test_probs"].astype(float)
        meta: dict[str, object] = {}
        for key in ("n_seeds", "seed_ids", "n_samples_per_seed", "n_total_generated_per_event"):
            if key in payload:
                value = payload[key]
                meta[key] = value.tolist() if value.ndim else value.item()
    if y.shape != p.shape:
        raise AssertionError(f"Label/probability shape mismatch: {path}")
    return y, p, meta


def ensemble_path(case: Case, horizon: int, modality: str) -> Path:
    path = ARTIFACTS / case.diffusion_version / "ensembles"
    if case.region != "RT":
        path /= case.region
    return path / f"fold_{case.mouse}" / f"horizon_{horizon}" / modality / "ensemble_probs.npz"


def eeg_classifier_path(case: Case, horizon: int) -> Path:
    path = ARTIFACTS / case.eeg_classifier_version / "classifiers"
    if case.region != "RT":
        path /= case.region
    return path / f"fold_{case.mouse}" / f"horizon_{horizon}" / "final_eval_probs.npz"


def mm_classifier_path(case: Case, horizon: int) -> Path:
    return (
        ARTIFACTS
        / case.mm_classifier_version
        / "classifiers_mm"
        / case.region
        / f"fold_{case.mouse}"
        / FULL_MM
        / f"horizon_{horizon}"
        / "final_eval_probs.npz"
    )


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def assert_same_labels(reference: np.ndarray, candidate: np.ndarray, source: Path) -> None:
    if not np.array_equal(reference, candidate):
        raise AssertionError(f"Held-out label ordering differs in {source}")


def audit_event_counts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for labels_path in sorted(DATA.glob("*/*/labels.csv")):
        region = labels_path.parent.parent.name
        mouse = labels_path.parent.name
        with labels_path.open("r", newline="", encoding="utf-8-sig") as handle:
            labels = [int(row[0]) for row in csv.reader(handle) if row]
        signal_shapes = {}
        for modality in ("eeg", "emg", "photometry"):
            signal_path = labels_path.with_name(f"{modality}.csv")
            with signal_path.open("r", newline="", encoding="utf-8-sig") as handle:
                n_columns = len(next(csv.reader(handle)))
            signal_shapes[modality] = [len(labels), n_columns]
            if n_columns != 5376:
                raise AssertionError(f"Expected 5376 signal columns in {signal_path}, got {n_columns}")
        rows.append(
            {
                "cohort": "TC" if region == "VPM" else region,
                "disk_region": region,
                "mouse": mouse,
                "ictal": int(sum(labels)),
                "nonictal": int(len(labels) - sum(labels)),
                "total": len(labels),
                "signal_shapes": signal_shapes,
            }
        )
        if sum(labels) * 2 != len(labels):
            raise AssertionError(f"Expected balanced labels in {labels_path}")
    if len(rows) != 9:
        raise AssertionError(f"Expected 9 mice, found {len(rows)}")
    return rows


def audit_auc() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for case in CASES:
        target_path = FIGURE_DATA / case.target_alignment
        with np.load(target_path, allow_pickle=False) as payload:
            target_y = payload["labels"].astype(int)
            target_p = payload["real_probs"].astype(float)
        target_auc = auc(target_y, target_p)
        values: dict[str, list[float]] = {
            "Target classifier": [target_auc] * 6,
            "EEG classifier": [],
            "EEG diffusion": [],
            "Full-MM classifier": [],
            "Full-MM diffusion": [],
        }
        for horizon in range(6):
            sources = {
                "EEG classifier": eeg_classifier_path(case, horizon),
                "EEG diffusion": ensemble_path(case, horizon, "eeg"),
                "Full-MM classifier": mm_classifier_path(case, horizon),
                "Full-MM diffusion": ensemble_path(case, horizon, FULL_MM),
            }
            horizon_y: np.ndarray | None = None
            for method, source in sources.items():
                y, p, meta = load_probs(source)
                if horizon_y is None:
                    horizon_y = y
                else:
                    assert_same_labels(horizon_y, y, source)
                assert_same_labels(target_y, y, source)
                score = auc(y, p)
                values[method].append(score)
                if "diffusion" in method.lower():
                    expected_meta = {
                        "n_seeds": 5,
                        "seed_ids": [0, 1, 2, 3, 4],
                        "n_samples_per_seed": 16,
                        "n_total_generated_per_event": 80,
                    }
                    if meta != expected_meta:
                        raise AssertionError(f"Unexpected ensemble metadata in {source}: {meta}")
        for method, scores in values.items():
            paper_scores = PAPER_AUC[(case.paper_cohort, method)]
            for horizon, (score, paper_score) in enumerate(zip(scores, paper_scores)):
                if round(score + 1e-12, 3) != paper_score:
                    raise AssertionError(
                        f"{case.paper_cohort} {method} h{horizon}: "
                        f"recomputed {score:.9f}, paper {paper_score:.3f}"
                    )
                long_rows.append(
                    {
                        "cohort": case.paper_cohort,
                        "mouse": case.mouse,
                        "method": method,
                        "horizon": f"h{horizon}",
                        "auc_full_precision": score,
                        "auc_paper_3dp": paper_score,
                    }
                )
            wide_rows.append(
                {
                    "cohort": case.paper_cohort,
                    "mouse": case.mouse,
                    "method": method,
                    **{f"h{i}": score for i, score in enumerate(scores)},
                    "mean": float(np.mean(scores)),
                }
            )
    return wide_rows, long_rows


def audit_delta_table(long_rows: list[dict[str, object]]) -> None:
    source = FIGURE_DATA / "figure8_delta_auc_statistics.tsv"
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 24:
        raise AssertionError(f"Expected 24 paired delta-AUC rows in {source}, found {len(rows)}")
    lookup = {
        (str(row["cohort"]), str(row["modality"]), str(row["horizon"])): row
        for row in rows
    }
    auc_lookup = {
        (str(row["cohort"]), str(row["method"]), str(row["horizon"])): float(row["auc_full_precision"])
        for row in long_rows
    }
    for cohort in ("RT", "TC"):
        for modality, baseline, diffusion in (
            ("EEG", "EEG classifier", "EEG diffusion"),
            ("Full-MM", "Full-MM classifier", "Full-MM diffusion"),
        ):
            for horizon in range(6):
                row = lookup[(cohort, modality, f"h{horizon}")]
                baseline_auc = auc_lookup[(cohort, baseline, f"h{horizon}")]
                diffusion_auc = auc_lookup[(cohort, diffusion, f"h{horizon}")]
                if not np.isclose(float(row["baseline_auc"]), baseline_auc):
                    raise AssertionError(f"Baseline AUC mismatch in delta table: {row}")
                if not np.isclose(float(row["diffusion_auc"]), diffusion_auc):
                    raise AssertionError(f"Diffusion AUC mismatch in delta table: {row}")
                if not np.isclose(float(row["delta_auc"]), diffusion_auc - baseline_auc):
                    raise AssertionError(f"Delta AUC mismatch in delta table: {row}")


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    events = audit_event_counts()
    wide_auc, long_auc = audit_auc()
    audit_delta_table(long_auc)
    event_rows = [
        {key: row[key] for key in ("cohort", "disk_region", "mouse", "ictal", "nonictal", "total")}
        for row in events
    ]
    write_tsv(
        TABLES / "table1_event_counts.tsv",
        event_rows,
        ["cohort", "disk_region", "mouse", "ictal", "nonictal", "total"],
    )
    write_tsv(
        TABLES / "table4_auc.tsv",
        wide_auc,
        ["cohort", "mouse", "method", "h0", "h1", "h2", "h3", "h4", "h5", "mean"],
    )
    write_tsv(
        TABLES / "table4_auc_long.tsv",
        long_auc,
        ["cohort", "mouse", "method", "horizon", "auc_full_precision", "auc_paper_3dp"],
    )
    report = {
        "status": "PASS",
        "processed_mice": len(events),
        "processed_events": int(sum(int(row["total"]) for row in events)),
        "table4_auc_cells_verified": len(long_auc),
        "paired_delta_auc_rows_verified": 24,
        "ensemble_seeds": [0, 1, 2, 3, 4],
        "samples_per_seed": 16,
        "generated_targets_per_event": 80,
    }
    (MANIFESTS / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
