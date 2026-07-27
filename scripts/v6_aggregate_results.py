"""v6_aggregate_results.py

Walk one or more v6 diffusion experiment trees, ensemble per-event probabilities
across all seeds for each (region, fold, horizon, diffusion_modality) cell, and emit a single
``master_results.tsv`` with paper-grade metrics plus bootstrap 95% CIs.

The script also writes per-cell pooled NPZ files so notebooks can re-plot
ROC curves without redoing any sampling.

What it reads:
    <outputs>/<VERSION>/diffusion/[<REGION>/]fold_<MOUSE>/horizon_<H>/<RUN>/
        ├── summary.json
        ├── final_eval_probs.npz   (val_y, val_probs, test_y, test_probs)

What it writes:
    <outputs>/<VERSION>/master_results.tsv
    <outputs>/<VERSION>/ensembles/[<REGION>/]fold_<MOUSE>/horizon_<H>/<MODALITY>/ensemble_probs.npz

Each TSV row covers one (region, fold, horizon, diffusion_modality, run_kind) cell, where
run_kind is either ``single_ckpt`` (per-seed) or ``ensemble`` (mean over
all seeds in that cell).

Preictal baselines + target ceilings are also pulled in from the matching
classifier ``summary.json`` files for context.

Usage:
    python -m scripts.v6_aggregate_results
    python -m scripts.v6_aggregate_results --version v6_paper_M234_allh
    python -m scripts.v6_aggregate_results --version v6_paper_VPM_M710_allh --region VPM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import (
    HORIZONS,
    OUTPUTS_ROOT,
    REGIONS,
    RT_REGION,
    _OUTPUTS_BASE,
)
from common.loo_splits import fold_name


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    if y.size == 0:
        return {"auc": float("nan"), "acc": float("nan"),
                "balanced_acc": float("nan"),
                "n": 0, "n_pos": 0, "n_neg": 0}
    pred = (p >= threshold).astype(np.int64)
    acc = float((pred == y).mean())
    auc = float("nan")
    if np.unique(y).size > 1:
        try:
            from sklearn.metrics import roc_auc_score, balanced_accuracy_score
            auc = float(roc_auc_score(y, p))
            bacc = float(balanced_accuracy_score(y, pred))
        except Exception:
            bacc = float("nan")
    else:
        bacc = float("nan")
    return {
        "auc": auc,
        "acc": acc,
        "balanced_acc": bacc,
        "n": int(y.size),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def _bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str = "auc",
    n_resamples: int = 1000,
    seed: int = 0,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Percentile bootstrap CI on a binary metric."""
    if y.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = y.size
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        ps = p[idx]
        if np.unique(ys).size < 2 and metric == "auc":
            continue
        m = _binary_metrics(ys, ps, threshold=threshold)
        v = m.get(metric, float("nan"))
        if v == v:
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    arr = np.asarray(stats)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return (float(lo), float(hi))


# -----------------------------------------------------------------------------
# Tree walking
# -----------------------------------------------------------------------------


def _diffusion_root(out_root: Path, region: str) -> Path:
    if region == RT_REGION:
        return out_root / "diffusion"
    return out_root / "diffusion" / region


def _iter_run_dirs(out_root: Path, region: str,
                   fold: str | None = None) -> Iterable[Path]:
    base = _diffusion_root(out_root, region)
    if not base.is_dir():
        return []
    pattern = "*" if fold is None else fold_name(fold)
    for fold_dir in sorted(base.glob(pattern)):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        for horizon_dir in sorted(fold_dir.glob("horizon_*")):
            for run_dir in sorted(horizon_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "summary.json").is_file():
                    yield run_dir


def _classifier_summary(out_root: Path, region: str, fold: str,
                        sub: str) -> dict | None:
    """Locate the classifier summary.json. RT uses legacy layout, VPM uses
    region-segregated layout. Returns None if the file is missing.
    """
    if region == RT_REGION:
        path = out_root / "classifiers" / fold_name(fold) / sub / "summary.json"
    else:
        path = out_root / "classifiers" / region / fold_name(fold) / sub / "summary.json"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _classifier_summary_multi(classifier_roots: list[Path], region: str,
                              fold: str, sub: str) -> dict | None:
    """Try each candidate classifier root in order; return the first hit."""
    for root in classifier_roots:
        s = _classifier_summary(root, region, fold, sub)
        if s is not None:
            return s
    return None


# Modality tags emitted by ``scripts/train_classifier_mm.py``. The "eeg"
# (1-channel) tag is excluded here because the EEG-only baseline is read from
# the legacy ``classifiers/`` tree by ``_classifier_summary``.
MM_TAGS = ("eeg_emg", "eeg_photo", "eeg_emg_photo")
DIFFUSION_MODALITIES = ("eeg", "eeg_emg", "eeg_photo", "eeg_emg_photo")


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _as_int_or_none(v) -> int | None:
    try:
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            v = v.item()
        if isinstance(v, str) and not v.strip():
            return None
        return int(v)
    except Exception:
        return None


def _infer_diffusion_modality(summary: dict) -> str:
    """Infer the diffusion conditioning modality tag from run config.

    Current paper methods need ``eeg`` and ``eeg_emg_photo``. The intermediate
    tags are kept so older exploratory runs remain representable without being
    accidentally pooled into the full multimodal ensemble.
    """
    args = summary.get("args", {}) if isinstance(summary.get("args", {}), dict) else {}
    include_emg = _as_bool(args.get("include_emg", summary.get("include_emg", False)))
    include_photo = _as_bool(args.get("include_photometry", summary.get("include_photometry", False)))

    if not include_emg and not include_photo:
        n_ch = _as_int_or_none(summary.get("n_past_channels"))
        if n_ch == 3:
            return "eeg_emg_photo"
        if n_ch == 2:
            # Historical 2-channel runs should have include flags in args; if
            # they do not, keep them out of the EEG-only pool.
            return "eeg_unknown2"

    parts = ["eeg"]
    if include_emg:
        parts.append("emg")
    if include_photo:
        parts.append("photo")
    return "_".join(parts)


def _resolve_outputs_root(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0].lower() == "outputs":
        parts = parts[1:]
    if not parts:
        return Path(_OUTPUTS_BASE)
    return Path(_OUTPUTS_BASE).joinpath(*parts)


def _mm_classifier_summary(classifier_roots: list[Path], region: str,
                           fold: str, sub: str, tag: str) -> dict | None:
    """Return the summary.json for a multi-modal preictal/target classifier.

    Layout (set by ``train_classifier_mm.py``):
        <root>/classifiers_mm/<REGION>/fold_<MOUSE>/<TAG>/<SUB>/summary.json
    """
    for root in classifier_roots:
        path = root / "classifiers_mm" / region / fold_name(fold) / tag / sub / "summary.json"
        if path.is_file():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def _mm_baseline_columns(classifier_roots: list[Path], region: str,
                         fold: str, sub: str) -> dict:
    """Return {f'preictal_mm_<tag>_test_auc'/'_acc': value} for each known tag.

    Missing entries are filled with NaN so the schema is stable regardless of
    which MM classifiers have been trained yet.
    """
    out: dict = {}
    for tag in MM_TAGS:
        s = _mm_classifier_summary(classifier_roots, region, fold, sub, tag)
        if s is None:
            out[f"preictal_mm_{tag}_test_auc"] = float("nan")
            out[f"preictal_mm_{tag}_test_acc"] = float("nan")
        else:
            t = s.get("test", {})
            out[f"preictal_mm_{tag}_test_auc"] = float(t.get("auc", float("nan")))
            out[f"preictal_mm_{tag}_test_acc"] = float(t.get("acc", float("nan")))
    return out


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    probs_path = run_dir / "final_eval_probs.npz"
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception:
        return None
    final_eval_metrics = summary.get("final_eval_metrics", {})
    args = summary.get("args", {})
    n_samples = _as_int_or_none(final_eval_metrics.get("final_n_samples"))
    if n_samples is None:
        n_samples = _as_int_or_none(args.get("final_eval_samples"))
    record = {
        "run_dir": str(run_dir),
        "run_name": summary.get("run_name", run_dir.name),
        "region": summary.get("region")
                  or summary.get("args", {}).get("region")
                  or RT_REGION,
        "fold": summary.get("fold")
                 or summary.get("args", {}).get("fold"),
        "horizon_idx": summary.get("horizon_idx")
                        if summary.get("horizon_idx") is not None
                        else summary.get("args", {}).get("horizon_idx"),
        "seed": summary.get("seed")
                or summary.get("args", {}).get("seed"),
        "val_holdout_mouse": summary.get("val_holdout_mouse")
                              or summary.get("args", {}).get("val_holdout_mouse"),
        "best_metrics": summary.get("best_metrics", {}),
        "final_eval_metrics": final_eval_metrics,
        "args": args,
        "diffusion_modality": _infer_diffusion_modality(summary),
        "n_samples_per_seed": n_samples,
    }
    if probs_path.is_file():
        try:
            with np.load(probs_path, allow_pickle=True) as data:
                record["val_y"] = data["val_y"].astype(np.int64)
                record["val_probs"] = data["val_probs"].astype(np.float32)
                record["test_y"] = data["test_y"].astype(np.int64)
                record["test_probs"] = data["test_probs"].astype(np.float32)
                if "n_samples" in data.files:
                    record["n_samples_per_seed"] = _as_int_or_none(data["n_samples"])
        except Exception as e:
            print(f"[warn] failed to load probs from {probs_path}: {e}",
                  file=sys.stderr)
            record["test_y"] = None
            record["test_probs"] = None
            record["val_y"] = None
            record["val_probs"] = None
    else:
        record["test_y"] = None
        record["test_probs"] = None
        record["val_y"] = None
        record["val_probs"] = None
    return record


# -----------------------------------------------------------------------------
# Row builders
# -----------------------------------------------------------------------------


COLUMNS = [
    "region", "fold", "horizon_idx", "run_kind", "run_name", "seed",
    "diffusion_modality",
    "n_seeds_in_ensemble", "n_samples_per_seed", "n_total_generated_per_event",
    "n_test", "n_test_pos", "n_test_neg",
    "test_auc", "test_auc_ci_lo", "test_auc_ci_hi",
    "test_acc", "test_acc_ci_lo", "test_acc_ci_hi",
    "test_balanced_acc",
    "val_auc", "val_acc",
    "preictal_test_auc", "preictal_test_acc",
    # Multi-modal preictal baselines (NaN until corresponding MM classifier
    # is trained via scripts/train_classifier_mm.py).
    "preictal_mm_eeg_emg_test_auc", "preictal_mm_eeg_emg_test_acc",
    "preictal_mm_eeg_photo_test_auc", "preictal_mm_eeg_photo_test_acc",
    "preictal_mm_eeg_emg_photo_test_auc", "preictal_mm_eeg_emg_photo_test_acc",
    "target_test_auc", "target_test_acc",
    "best_score", "best_select_by",
    "val_holdout_mouse",
    "run_dir",
]


def _row_from_single(record: dict, classifier_roots: list[Path], n_boot: int) -> dict:
    region = record["region"]
    fold = record["fold"]
    horizon = int(record["horizon_idx"]) if record["horizon_idx"] is not None else None
    modality = record.get("diffusion_modality", "")
    n_samples = _as_int_or_none(record.get("n_samples_per_seed"))
    test_y = record.get("test_y")
    test_p = record.get("test_probs")
    if test_y is None or test_p is None or len(test_y) == 0:
        # No probs file - fall back to summary-level metrics.
        fm = record.get("final_eval_metrics", {})
        bm = record.get("best_metrics", {})
        test_auc = float(fm.get("final_test_gen_auc", bm.get("test_gen_auc_avg", float("nan"))))
        test_acc = float(fm.get("final_test_gen_acc", bm.get("test_gen_acc_avg", float("nan"))))
        m = {"auc": test_auc, "acc": test_acc, "balanced_acc": float("nan"),
             "n": 0, "n_pos": 0, "n_neg": 0}
        auc_ci = (float("nan"), float("nan"))
        acc_ci = (float("nan"), float("nan"))
        val_auc = float(fm.get("final_val_gen_auc", bm.get("val_gen_auc_avg", float("nan"))))
        val_acc = float(fm.get("final_val_gen_acc", bm.get("val_gen_acc_avg", float("nan"))))
    else:
        m = _binary_metrics(test_y, test_p)
        auc_ci = _bootstrap_ci(test_y, test_p, "auc", n_boot)
        acc_ci = _bootstrap_ci(test_y, test_p, "acc", n_boot)
        val_y = record.get("val_y")
        val_p = record.get("val_probs")
        if val_y is not None and val_p is not None and len(val_y) > 0:
            vm = _binary_metrics(val_y, val_p)
            val_auc, val_acc = vm["auc"], vm["acc"]
        else:
            val_auc = val_acc = float("nan")

    pre = _classifier_summary_multi(classifier_roots, region, fold, f"horizon_{horizon}") if horizon is not None else None
    tgt = _classifier_summary_multi(classifier_roots, region, fold, "target")
    mm_cols = _mm_baseline_columns(classifier_roots, region, fold,
                                   f"horizon_{horizon}") if horizon is not None else {
        f"preictal_mm_{t}_test_auc": float("nan") for t in MM_TAGS
    } | {f"preictal_mm_{t}_test_acc": float("nan") for t in MM_TAGS}
    row = {
        "region": region,
        "fold": fold,
        "horizon_idx": horizon,
        "run_kind": "single_ckpt",
        "run_name": record["run_name"],
        "seed": record.get("seed", ""),
        "diffusion_modality": modality,
        "n_seeds_in_ensemble": 1,
        "n_samples_per_seed": n_samples if n_samples is not None else "",
        "n_total_generated_per_event": n_samples if n_samples is not None else "",
        "n_test": m["n"], "n_test_pos": m["n_pos"], "n_test_neg": m["n_neg"],
        "test_auc": m["auc"], "test_auc_ci_lo": auc_ci[0], "test_auc_ci_hi": auc_ci[1],
        "test_acc": m["acc"], "test_acc_ci_lo": acc_ci[0], "test_acc_ci_hi": acc_ci[1],
        "test_balanced_acc": m.get("balanced_acc", float("nan")),
        "val_auc": val_auc, "val_acc": val_acc,
        "preictal_test_auc": float(pre.get("test", {}).get("auc", float("nan"))) if pre else float("nan"),
        "preictal_test_acc": float(pre.get("test", {}).get("acc", float("nan"))) if pre else float("nan"),
        "target_test_auc": float(tgt.get("test", {}).get("auc", float("nan"))) if tgt else float("nan"),
        "target_test_acc": float(tgt.get("test", {}).get("acc", float("nan"))) if tgt else float("nan"),
        "best_score": record.get("args", {}).get("best_score", float("nan")),
        "best_select_by": record.get("args", {}).get("select_best_by", ""),
        "val_holdout_mouse": record.get("val_holdout_mouse", ""),
        "run_dir": record.get("run_dir", ""),
    }
    row.update(mm_cols)
    return row


def _row_from_ensemble(records: list[dict], classifier_roots: list[Path],
                       ensemble_root: Path, n_boot: int) -> dict | None:
    records = [r for r in records
               if r.get("test_y") is not None and r.get("test_probs") is not None
               and len(r["test_y"]) > 0]
    if not records:
        return None
    region = records[0]["region"]
    fold = records[0]["fold"]
    horizon = int(records[0]["horizon_idx"]) if records[0]["horizon_idx"] is not None else None
    modality = records[0].get("diffusion_modality", "")

    test_y = records[0]["test_y"]
    test_probs_stack = []
    val_probs_stack = []
    val_y = records[0].get("val_y")
    seed_ids = []
    sample_counts = []
    for r in records:
        if r["test_y"] is None or r["test_probs"] is None:
            continue
        if r.get("diffusion_modality", "") != modality:
            print(f"[warn] mismatched modality inside ensemble group at "
                  f"region={region} fold={fold} h={horizon}; skipping {r['run_dir']}",
                  file=sys.stderr)
            continue
        if not np.array_equal(r["test_y"], test_y):
            print(f"[warn] mismatched test_y across seeds at region={region} "
                  f"fold={fold} h={horizon}; skipping {r['run_dir']}",
                  file=sys.stderr)
            continue
        test_probs_stack.append(r["test_probs"])
        seed_ids.append(r.get("seed", -1))
        n_samples = _as_int_or_none(r.get("n_samples_per_seed"))
        if n_samples is not None:
            sample_counts.append(n_samples)
        if val_y is not None and r.get("val_y") is not None and np.array_equal(r["val_y"], val_y):
            val_probs_stack.append(r["val_probs"])
    if not test_probs_stack:
        return None
    test_probs = np.mean(np.stack(test_probs_stack, axis=0), axis=0)
    val_probs = (np.mean(np.stack(val_probs_stack, axis=0), axis=0)
                 if val_probs_stack else None)

    m = _binary_metrics(test_y, test_probs)
    auc_ci = _bootstrap_ci(test_y, test_probs, "auc", n_boot)
    acc_ci = _bootstrap_ci(test_y, test_probs, "acc", n_boot)
    if val_probs is not None and val_y is not None:
        vm = _binary_metrics(val_y, val_probs)
        val_auc, val_acc = vm["auc"], vm["acc"]
    else:
        val_auc = val_acc = float("nan")

    if region == RT_REGION:
        ens_dir = ensemble_root / fold_name(fold) / f"horizon_{horizon}" / modality
    else:
        ens_dir = ensemble_root / region / fold_name(fold) / f"horizon_{horizon}" / modality
    ens_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ens_dir / "ensemble_probs.npz"
    unique_sample_counts = sorted(set(sample_counts))
    n_samples_per_seed = unique_sample_counts[0] if len(unique_sample_counts) == 1 else None
    n_total_generated = (
        len(test_probs_stack) * n_samples_per_seed
        if n_samples_per_seed is not None else None
    )
    np.savez(
        npz_path,
        val_y=val_y if val_y is not None else np.array([], dtype=np.int64),
        val_probs=val_probs if val_probs is not None else np.array([], dtype=np.float32),
        test_y=test_y,
        test_probs=test_probs.astype(np.float32),
        n_seeds=np.int64(len(test_probs_stack)),
        seed_ids=np.array(seed_ids),
        diffusion_modality=np.array(modality),
        n_samples_per_seed=np.int64(n_samples_per_seed) if n_samples_per_seed is not None else np.array(-1, dtype=np.int64),
        n_total_generated_per_event=np.int64(n_total_generated) if n_total_generated is not None else np.array(-1, dtype=np.int64),
    )
    # Preserve the historical path for existing downstream notebooks that
    # expect the full multimodal ensemble at horizon_<H>/ensemble_probs.npz.
    if modality == "eeg_emg_photo":
        legacy_npz_path = ens_dir.parent / "ensemble_probs.npz"
        np.savez(
            legacy_npz_path,
            val_y=val_y if val_y is not None else np.array([], dtype=np.int64),
            val_probs=val_probs if val_probs is not None else np.array([], dtype=np.float32),
            test_y=test_y,
            test_probs=test_probs.astype(np.float32),
            n_seeds=np.int64(len(test_probs_stack)),
            seed_ids=np.array(seed_ids),
            diffusion_modality=np.array(modality),
            n_samples_per_seed=np.int64(n_samples_per_seed) if n_samples_per_seed is not None else np.array(-1, dtype=np.int64),
            n_total_generated_per_event=np.int64(n_total_generated) if n_total_generated is not None else np.array(-1, dtype=np.int64),
        )

    pre = _classifier_summary_multi(classifier_roots, region, fold, f"horizon_{horizon}") if horizon is not None else None
    tgt = _classifier_summary_multi(classifier_roots, region, fold, "target")
    mm_cols = _mm_baseline_columns(classifier_roots, region, fold,
                                   f"horizon_{horizon}") if horizon is not None else {
        f"preictal_mm_{t}_test_auc": float("nan") for t in MM_TAGS
    } | {f"preictal_mm_{t}_test_acc": float("nan") for t in MM_TAGS}
    row = {
        "region": region,
        "fold": fold,
        "horizon_idx": horizon,
        "run_kind": "ensemble",
        "run_name": f"ensemble_{len(test_probs_stack)}seed",
        "seed": "",
        "diffusion_modality": modality,
        "n_seeds_in_ensemble": len(test_probs_stack),
        "n_samples_per_seed": n_samples_per_seed if n_samples_per_seed is not None else ",".join(map(str, unique_sample_counts)),
        "n_total_generated_per_event": n_total_generated if n_total_generated is not None else "",
        "n_test": m["n"], "n_test_pos": m["n_pos"], "n_test_neg": m["n_neg"],
        "test_auc": m["auc"], "test_auc_ci_lo": auc_ci[0], "test_auc_ci_hi": auc_ci[1],
        "test_acc": m["acc"], "test_acc_ci_lo": acc_ci[0], "test_acc_ci_hi": acc_ci[1],
        "test_balanced_acc": m.get("balanced_acc", float("nan")),
        "val_auc": val_auc, "val_acc": val_acc,
        "preictal_test_auc": float(pre.get("test", {}).get("auc", float("nan"))) if pre else float("nan"),
        "preictal_test_acc": float(pre.get("test", {}).get("acc", float("nan"))) if pre else float("nan"),
        "target_test_auc": float(tgt.get("test", {}).get("auc", float("nan"))) if tgt else float("nan"),
        "target_test_acc": float(tgt.get("test", {}).get("acc", float("nan"))) if tgt else float("nan"),
        "best_score": "",
        "best_select_by": "",
        "val_holdout_mouse": ",".join(sorted({str(r.get("val_holdout_mouse", "")) for r in records})),
        "run_dir": str(ens_dir),
    }
    row.update(mm_cols)
    return row


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None,
                   help="Override OUTPUTS_ROOT subdir, e.g. v6_paper_M234_allh.")
    p.add_argument("--region", default=None,
                   help="Filter to one region (RT/VPM). Default: all regions present.")
    p.add_argument("--fold", default=None,
                   help="Filter to one fold (mouse ID). Default: all folds present.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--out_path", default=None,
                   help="Override TSV destination path. Default: <version>/master_results.tsv")
    p.add_argument("--diffusion_root", action="append", default=None,
                   help="Extra outputs/<VERSION> dirs whose diffusion runs should "
                        "be scanned and merged into this master table. Repeat to "
                        "combine EEG-only and multimodal result roots.")
    p.add_argument("--classifier_root", action="append", default=None,
                   help="Extra outputs/<VERSION> dirs to search for classifier summary.json "
                        "files (target + preictal). Repeat to add more. The current "
                        "--version's classifiers dir is always searched first.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.version is not None:
        out_root = Path(_OUTPUTS_BASE) / args.version
    else:
        out_root = Path(OUTPUTS_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    ensemble_root = out_root / "ensembles"

    scan_roots = [out_root]
    if args.diffusion_root:
        for extra in args.diffusion_root:
            extra_path = _resolve_outputs_root(extra)
            if extra_path not in scan_roots:
                scan_roots.append(extra_path)
    print(f"[aggregate] diffusion_roots={[str(r) for r in scan_roots]}")

    classifier_roots = [out_root]
    if args.classifier_root:
        for extra in args.classifier_root:
            extra_path = _resolve_outputs_root(extra)
            if extra_path not in classifier_roots:
                classifier_roots.append(extra_path)
    print(f"[aggregate] classifier_roots={[str(r) for r in classifier_roots]}")

    regions = [args.region] if args.region else list(REGIONS.keys())

    rows: list[dict] = []
    groups: dict[tuple, list[dict]] = {}
    n_runs = 0
    for scan_root in scan_roots:
        for region in regions:
            for run_dir in _iter_run_dirs(scan_root, region, args.fold):
                rec = _load_run(run_dir)
                if rec is None:
                    continue
                if rec.get("fold") is None or rec.get("horizon_idx") is None:
                    continue
                n_runs += 1
                rows.append(_row_from_single(rec, classifier_roots, args.n_bootstrap))
                key = (
                    rec["region"],
                    rec["fold"],
                    int(rec["horizon_idx"]),
                    rec.get("diffusion_modality", ""),
                )
                groups.setdefault(key, []).append(rec)
    print(f"[aggregate] scanned {n_runs} runs")

    for key, records in groups.items():
        if len(records) < 2:
            continue
        ens = _row_from_ensemble(records, classifier_roots, ensemble_root, args.n_bootstrap)
        if ens is not None:
            rows.append(ens)

    # Sort for stable output.
    rows.sort(key=lambda r: (
        r.get("region", ""), r.get("fold", ""),
        r.get("horizon_idx", -1) if r.get("horizon_idx") is not None else -1,
        r.get("diffusion_modality", ""),
        r.get("run_kind", ""), str(r.get("seed", "")),
    ))

    tsv_path = Path(args.out_path) if args.out_path else (out_root / "master_results.tsv")
    with open(tsv_path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            cells = []
            for c in COLUMNS:
                v = r.get(c, "")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v) if v is not None else "")
            f.write("\t".join(cells) + "\n")

    n_single = sum(1 for r in rows if r["run_kind"] == "single_ckpt")
    n_ens = sum(1 for r in rows if r["run_kind"] == "ensemble")
    print(f"[aggregate] wrote {tsv_path} ({n_single} single rows + {n_ens} ensemble rows)")


if __name__ == "__main__":
    main()
