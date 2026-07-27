"""v6_qc_audit.py

Quality-control and diagnostic summaries for the final v6 paper tables.

Primary paper results remain validation-selected. This script only adds tags
and diagnostic "what did the logs contain?" tables so unusual behavior is
visible during manuscript preparation.

Outputs:
    <outputs>/<VERSION>/audit/qc_tags.tsv
    <outputs>/<VERSION>/audit/classifier_diagnostic_bestcase.tsv
    <outputs>/<VERSION>/audit/diffusion_diagnostic_bestcase.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import OUTPUTS_ROOT, RT_REGION, _OUTPUTS_BASE
from common.loo_splits import fold_name


EEG_MODALITY = "eeg"
FULL_MM_MODALITY = "eeg_emg_photo"


def _resolve_outputs_root(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0].lower() == "outputs":
        parts = parts[1:]
    return Path(_OUTPUTS_BASE).joinpath(*parts)


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_log(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None,
                   help="outputs/<VERSION> containing master_results.tsv.")
    p.add_argument("--region", default=None)
    p.add_argument("--fold", default=None)
    p.add_argument("--classifier_root", action="append", default=None,
                   help="outputs/<VERSION> roots to scan for classifier logs.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--required_seeds", type=int, default=5)
    p.add_argument("--required_generated", type=int, default=80)
    p.add_argument("--weak_auc_threshold", type=float, default=0.55)
    p.add_argument("--delta_threshold", type=float, default=0.02)
    p.add_argument("--condition_gap_threshold", type=float, default=0.05)
    p.add_argument("--nonmonotonic_delta", type=float, default=0.05)
    return p.parse_args()


def _load_master(version: str | None) -> tuple[Path, pd.DataFrame]:
    root = Path(_OUTPUTS_BASE) / version if version else Path(OUTPUTS_ROOT)
    path = root / "master_results.tsv"
    df = pd.read_csv(path, sep="\t")
    if "diffusion_modality" not in df.columns:
        df["diffusion_modality"] = FULL_MM_MODALITY
    return root, df


def _condition_gap_by_cell(single_rows: pd.DataFrame) -> dict[tuple, float]:
    gaps: dict[tuple, list[float]] = {}
    for _, row in single_rows.iterrows():
        summary = _read_json(Path(str(row.get("run_dir", ""))) / "summary.json")
        gap = _safe_float(summary.get("best_metrics", {}).get("val_condition_gap_auc"))
        if np.isfinite(gap):
            key = (
                row.get("region"),
                row.get("fold"),
                int(row.get("horizon_idx")),
                row.get("diffusion_modality"),
            )
            gaps.setdefault(key, []).append(gap)
    return {k: float(np.mean(v)) for k, v in gaps.items() if v}


def qc_tags(df: pd.DataFrame, args) -> pd.DataFrame:
    ens = df[df.run_kind.eq("ensemble")].copy()
    if args.region:
        ens = ens[ens.region.eq(args.region)]
    if args.fold:
        ens = ens[ens.fold.eq(args.fold)]

    single = df[df.run_kind.eq("single_ckpt")].copy()
    gap_map = _condition_gap_by_cell(single)
    tag_rows = []

    for _, row in ens.iterrows():
        tags: list[str] = []
        auc = _safe_float(row.get("test_auc"))
        eeg_cls_auc = _safe_float(row.get("preictal_test_auc"))
        mm_cls_auc = _safe_float(row.get("preictal_mm_eeg_emg_photo_test_auc"))
        auc_values = [v for v in [auc, eeg_cls_auc, mm_cls_auc] if np.isfinite(v)]
        if any(v < 0.5 for v in auc_values):
            tags.append("below_chance_auc")
        if any(v < args.weak_auc_threshold for v in auc_values):
            tags.append("weak_auc")

        n_seeds = _safe_float(row.get("n_seeds_in_ensemble"))
        n_generated = _safe_float(row.get("n_total_generated_per_event"))
        if (
            not np.isfinite(n_seeds) or n_seeds < args.required_seeds
            or not np.isfinite(n_generated) or n_generated < args.required_generated
        ):
            tags.append("incomplete_ensemble")

        modality = row.get("diffusion_modality")
        if modality == EEG_MODALITY:
            matched = eeg_cls_auc
        elif modality == FULL_MM_MODALITY:
            matched = mm_cls_auc
        else:
            matched = np.nan
        if np.isfinite(auc) and np.isfinite(matched) and auc <= matched:
            tags.append("diffusion_not_above_matched_classifier")

        gap_key = (
            row.get("region"),
            row.get("fold"),
            int(row.get("horizon_idx")),
            modality,
        )
        mean_gap = gap_map.get(gap_key, np.nan)
        if np.isfinite(mean_gap) and mean_gap < args.condition_gap_threshold:
            tags.append("weak_condition_gap")

        tag_rows.append({
            **{c: row.get(c) for c in [
                "region", "fold", "horizon_idx", "diffusion_modality",
                "test_auc", "preictal_test_auc",
                "preictal_mm_eeg_emg_photo_test_auc", "target_test_auc",
                "n_seeds_in_ensemble", "n_samples_per_seed",
                "n_total_generated_per_event",
            ]},
            "mean_val_condition_gap_auc": mean_gap,
            "qc_tags": ";".join(tags),
        })

    out = pd.DataFrame(tag_rows)
    if out.empty:
        return out

    extra_tags: dict[int, list[str]] = {i: [] for i in out.index}
    for (region, fold), group in out.groupby(["region", "fold"]):
        for h, h_group in group.groupby("horizon_idx"):
            if {EEG_MODALITY, FULL_MM_MODALITY}.issubset(set(h_group.diffusion_modality)):
                eeg_row = h_group[h_group.diffusion_modality.eq(EEG_MODALITY)].iloc[0]
                mm_row = h_group[h_group.diffusion_modality.eq(FULL_MM_MODALITY)].iloc[0]
                eeg_auc = _safe_float(eeg_row.test_auc)
                mm_auc = _safe_float(mm_row.test_auc)
                if np.isfinite(eeg_auc) and np.isfinite(mm_auc):
                    if mm_auc < eeg_auc - args.delta_threshold:
                        extra_tags[int(mm_row.name)].append("mm_diffusion_under_eeg")

                eeg_cls = _safe_float(eeg_row.preictal_test_auc)
                mm_cls = _safe_float(eeg_row.preictal_mm_eeg_emg_photo_test_auc)
                if np.isfinite(eeg_cls) and np.isfinite(mm_cls):
                    if mm_cls < eeg_cls - args.delta_threshold:
                        for idx in h_group.index:
                            extra_tags[int(idx)].append("mm_classifier_under_eeg")

        for modality, m_group in group.groupby("diffusion_modality"):
            m_group = m_group.sort_values("horizon_idx")
            prev_h = None
            prev_auc = None
            for idx, row in m_group.iterrows():
                h = int(row.horizon_idx)
                auc = _safe_float(row.test_auc)
                if prev_h is not None and h == prev_h + 1:
                    if np.isfinite(prev_auc) and np.isfinite(auc):
                        if auc > prev_auc + args.nonmonotonic_delta:
                            extra_tags[int(idx)].append("nonmonotonic_horizon")
                prev_h = h
                prev_auc = auc

    for idx, tags in extra_tags.items():
        if tags:
            current = out.at[idx, "qc_tags"]
            merged = [t for t in str(current).split(";") if t] + tags
            out.at[idx, "qc_tags"] = ";".join(sorted(set(merged)))
    return out


def _classifier_dir(root: Path, region: str, fold: str, sub: str) -> Path:
    if region == RT_REGION:
        return root / "classifiers" / fold_name(fold) / sub
    return root / "classifiers" / region / fold_name(fold) / sub


def _mm_classifier_dir(root: Path, region: str, fold: str, sub: str) -> Path:
    return root / "classifiers_mm" / region / fold_name(fold) / FULL_MM_MODALITY / sub


def classifier_diagnostics(df: pd.DataFrame, roots: list[Path], args) -> pd.DataFrame:
    ens = df[df.run_kind.eq("ensemble")].copy()
    if args.region:
        ens = ens[ens.region.eq(args.region)]
    if args.fold:
        ens = ens[ens.fold.eq(args.fold)]
    cells = sorted({(r.region, r.fold, int(r.horizon_idx)) for r in ens.itertuples()})
    rows = []
    seen: set[Path] = set()
    for region, fold, h in cells:
        sub = f"horizon_{h}"
        for root in roots:
            for kind, directory in [
                ("eeg_preictal", _classifier_dir(root, region, fold, sub)),
                ("full_mm_preictal", _mm_classifier_dir(root, region, fold, sub)),
            ]:
                directory = directory.resolve()
                if directory in seen:
                    continue
                seen.add(directory)
                summary = _read_json(directory / "summary.json")
                log = _read_log(directory / "training_log.csv")
                if not summary or log is None or log.empty:
                    continue
                best_val_auc = log.loc[log.val_auc.idxmax()]
                best_test_auc = log.loc[log.test_auc.idxmax()]
                rows.append({
                    "region": region,
                    "fold": fold,
                    "horizon_idx": h,
                    "classifier_kind": kind,
                    "selection_metric": summary.get("selection_metric", "legacy_or_unknown"),
                    "paper_val_auc": summary.get("val", {}).get("auc", np.nan),
                    "paper_test_auc": summary.get("test", {}).get("auc", np.nan),
                    "best_val_auc_epoch": int(best_val_auc.epoch),
                    "best_val_auc": float(best_val_auc.val_auc),
                    "test_auc_at_best_val_auc": float(best_val_auc.test_auc),
                    "best_test_auc_epoch": int(best_test_auc.epoch),
                    "best_test_auc_diagnostic_only": float(best_test_auc.test_auc),
                    "val_auc_at_best_test_auc": float(best_test_auc.val_auc),
                    "source_dir": str(directory),
                })
    return pd.DataFrame(rows)


def diffusion_diagnostics(df: pd.DataFrame, args) -> pd.DataFrame:
    single = df[df.run_kind.eq("single_ckpt")].copy()
    if args.region:
        single = single[single.region.eq(args.region)]
    if args.fold:
        single = single[single.fold.eq(args.fold)]
    rows = []
    for _, row in single.iterrows():
        run_dir = Path(str(row.get("run_dir", "")))
        summary = _read_json(run_dir / "summary.json")
        log = _read_log(run_dir / "training_log.csv")
        rec = {
            "region": row.get("region"),
            "fold": row.get("fold"),
            "horizon_idx": row.get("horizon_idx"),
            "diffusion_modality": row.get("diffusion_modality"),
            "run_name": row.get("run_name"),
            "seed": row.get("seed"),
            "saved_best_epoch": summary.get("final_eval_metrics", {}).get("final_best_epoch", np.nan),
            "saved_final_eval_test_auc": summary.get("final_eval_metrics", {}).get("final_test_gen_auc", np.nan),
            "saved_best_metric": summary.get("select_best_by", summary.get("args", {}).get("select_best_by", "")),
            "best_model_exists": (run_dir / "best_model.pth").is_file(),
            "final_model_exists": (run_dir / "final_model.pth").is_file(),
            "source_dir": str(run_dir),
        }
        if log is not None and not log.empty:
            best_val = log.loc[log.val_gen_auc_avg.idxmax()]
            best_test = log.loc[log.test_gen_auc_avg.idxmax()]
            rec.update({
                "best_val_auc_epoch": int(best_val.epoch),
                "best_val_auc": float(best_val.val_gen_auc_avg),
                "test_auc_at_best_val_auc": float(best_val.test_gen_auc_avg),
                "condition_gap_at_best_val_auc": float(best_val.val_condition_gap_auc),
                "best_test_auc_epoch": int(best_test.epoch),
                "best_test_auc_diagnostic_only": float(best_test.test_gen_auc_avg),
                "val_auc_at_best_test_auc": float(best_test.val_gen_auc_avg),
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root, df = _load_master(args.version)
    out_dir = Path(args.out_dir) if args.out_dir else root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier_roots = [root]
    for r in args.classifier_root or []:
        resolved = _resolve_outputs_root(r)
        if resolved not in classifier_roots:
            classifier_roots.append(resolved)

    qc = qc_tags(df, args)
    cls = classifier_diagnostics(df, classifier_roots, args)
    diff = diffusion_diagnostics(df, args)

    qc_path = out_dir / "qc_tags.tsv"
    cls_path = out_dir / "classifier_diagnostic_bestcase.tsv"
    diff_path = out_dir / "diffusion_diagnostic_bestcase.tsv"
    qc.to_csv(qc_path, sep="\t", index=False)
    cls.to_csv(cls_path, sep="\t", index=False)
    diff.to_csv(diff_path, sep="\t", index=False)

    print(f"[qc] wrote {qc_path} ({len(qc)} rows)")
    print(f"[qc] wrote {cls_path} ({len(cls)} rows)")
    print(f"[qc] wrote {diff_path} ({len(diff)} rows)")
    if not qc.empty:
        flagged = qc[qc.qc_tags.fillna("").astype(str).str.len() > 0]
        print(f"[qc] flagged rows: {len(flagged)}")


if __name__ == "__main__":
    main()
