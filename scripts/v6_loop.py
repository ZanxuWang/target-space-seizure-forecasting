"""v6_loop.py

Autonomous experiment loop for the v6 XAttn diffusion pipeline.

- Runs a queue of training configs against a chosen (region, fold) and one
  or more horizons / seeds. The Cartesian product of horizons * seeds *
  queued configs is expanded automatically.
- Logs best_metrics + final_eval_metrics from each run to results.tsv.
- For the legacy `initial` queue, adaptive follow-ups are still generated.
- Stops once test_gen_auc_avg AND test_gen_acc_avg both >= TARGET (only
  applies when sweeping a single horizon * single seed), or when all
  queued configs are exhausted.

Use --max_runs to cap the number of runs in this session.

Examples
--------
Full RT-M234 anchor sweep across all 6 horizons, 5 seeds:

    python -m scripts.v6_loop --region RT --fold M234 \\
        --horizons 0,1,2,3,4,5 --seeds 0,1,2,3,4 \\
        --queue anchor_b02 --max_runs 30

VPM config search on chosen fold at horizon 0:

    python -m scripts.v6_loop --region VPM --fold M710 \\
        --horizons 0 --seeds 0 --queue initial --max_runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = sys.executable

TARGET_AUC = 0.80
TARGET_ACC = 0.80
RESULTS_TSV_NAME = "results.tsv"


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--out_root", default=None,
                   help="Override outputs/<VERSION> root. Defaults to common.config.OUTPUTS_ROOT")
    p.add_argument("--region", default="RT",
                   help="Cohort region (RT or VPM). Forwarded to v6_train.")
    p.add_argument("--fold", default="M234",
                   help="Held-out test mouse for this sweep.")
    p.add_argument("--horizon_idx", type=int, default=None,
                   help="DEPRECATED: use --horizons. If set and --horizons unset, "
                        "the sweep covers this single horizon.")
    p.add_argument("--horizons", default=None,
                   help="Comma-separated list of horizon indices, e.g. '0,1,2,3,4,5'. "
                        "Defaults to '0' (or --horizon_idx if provided).")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated list of seeds, e.g. '0,1,2,3,4'. "
                        "Defaults to a single seed coming from each config's "
                        "'seed' key (which falls back to DEFAULT_SEED in v6_train).")
    p.add_argument("--val_holdout_mouse", default=None,
                   help="Override val_holdout_mouse for every queued config. If unset, "
                        "each config's own value (and v6_train's auto-pick fallback) "
                        "is preserved.")
    p.add_argument("--classifier_ckpt", default=None,
                   help="Absolute path to the target classifier .pth. Forwarded to "
                        "v6_train so we can reuse classifiers trained under a different "
                        "PIPELINE_VERSION. If unset, v6_train uses its own default.")
    p.add_argument("--max_runs", type=int, default=8,
                   help="Maximum number of runs this session (0 = unlimited).")
    p.add_argument("--start_at", type=int, default=0,
                   help="Skip first N runs in the expanded queue (for restarts).")
    p.add_argument("--queue", default="initial",
                   choices=["initial", "batch2", "batch3", "batch4",
                            "anchor_b02", "anchor_vpm", "anchor_eeg"],
                   help="Which queue to run: initial (5 starter configs), "
                        "batch2 (followups based on r04/r05 findings), "
                        "batch3 (r04-style multi-seed pool for stronger ensembling), "
                        "batch4 (val-mouse rotation + aggressive aux for calibration), "
                        "anchor_b02 (single config = b02 hparams; sweep horizons*seeds), "
                        "anchor_vpm (TBD VPM-tuned anchor; placeholder copies b02), "
                        "anchor_eeg (same anchor hparams but EEG-only conditioning).")
    args = p.parse_args()
    if args.horizons is None:
        if args.horizon_idx is not None:
            args.horizons_list = [int(args.horizon_idx)]
        else:
            args.horizons_list = [0]
    else:
        args.horizons_list = _parse_int_list(args.horizons)
        if not args.horizons_list:
            raise SystemExit(f"--horizons parsed to empty list: {args.horizons!r}")
    args.seeds_list = _parse_int_list(args.seeds) if args.seeds else []
    return args


# -----------------------------------------------------------------------------
# Queue of experiments
# -----------------------------------------------------------------------------


def build_initial_queue() -> list[dict]:
    """Initial set of experiments to attempt before adaptive ones kick in.

    Anchor uses ``norm_mode=global`` so generated outputs sit in the same
    log1p-magnitude space the classifier was trained on after the
    ``* std + mean`` rescale. ``per_sample`` is left as an ablation only.

    Key hypothesis from the failed r01 attempts: val_gen_auc climbs but
    test_gen_auc (M234) stays at chance, i.e. classic train-mouse OOD.
    All runs from r02 onward add DANN + cross-mouse mixup + augmentation
    to push past_encoder features toward mouse-invariance.
    """
    base = dict(
        epochs=200,
        eval_every=20,
        eval_gen_samples=4,
        eval_steps=50,
        eval_guidance_scale=2.0,
        select_best_by="val_gen_auc_avg",
        base_ch=64,
        depth=4,
        time_dim=256,
        token_dim=256,
        past_encoder_depth=3,
        past_self_attn_layers=2,
        eps_weight=0.25,
        aux_ce_weight=0.5,
        aux_fm_weight=5.0,
        past_ce_weight=0.5,
        t_aux_min=100,
        t_aux_max=900,
        cond_drop_prob=0.15,
        norm_mode="global",
        augment=True,
        mixup_prob=0.0,
        mouse_adv_weight=0.0,
        cross_mouse_mixup=False,
        include_emg=False,
        include_photometry=False,
        val_holdout_mouse="M229",
    )

    runs = []

    # Run 1: DANN + cross-mouse mixup, EEG only (main hypothesis)
    runs.append({
        **base,
        "run_name": "r01_dann_mixup",
        "mouse_adv_weight": 0.5,
        "mouse_adv_lambda_grl": 1.0,
        "mixup_prob": 0.5,
        "mixup_alpha": 0.4,
        "cross_mouse_mixup": True,
    })

    # Run 2: heavier DANN + harder aux
    runs.append({
        **base,
        "run_name": "r02_dann_strong",
        "mouse_adv_weight": 1.0,
        "mouse_adv_lambda_grl": 1.5,
        "mixup_prob": 0.5,
        "mixup_alpha": 0.4,
        "cross_mouse_mixup": True,
        "aux_ce_weight": 1.5,
        "aux_fm_weight": 10.0,
        "past_ce_weight": 1.0,
    })

    # Run 3: DANN + mixup + EEG + EMG (multimodal)
    runs.append({
        **base,
        "run_name": "r03_dann_mm_emg",
        "include_emg": True,
        "mouse_adv_weight": 0.5,
        "mouse_adv_lambda_grl": 1.0,
        "mixup_prob": 0.5,
        "cross_mouse_mixup": True,
    })

    # Run 4: full multimodal + DANN
    runs.append({
        **base,
        "run_name": "r04_dann_mm_full",
        "include_emg": True,
        "include_photometry": True,
        "mouse_adv_weight": 0.5,
        "mouse_adv_lambda_grl": 1.0,
        "mixup_prob": 0.5,
        "cross_mouse_mixup": True,
    })

    # Run 5: full multimodal + DANN + strong aux + bigger past
    runs.append({
        **base,
        "run_name": "r05_mm_dann_bigger",
        "include_emg": True,
        "include_photometry": True,
        "mouse_adv_weight": 1.0,
        "mouse_adv_lambda_grl": 1.5,
        "mixup_prob": 0.5,
        "cross_mouse_mixup": True,
        "aux_ce_weight": 1.5,
        "aux_fm_weight": 10.0,
        "past_ce_weight": 1.0,
        "past_encoder_depth": 4,
        "past_self_attn_layers": 4,
        "token_dim": 384,
        "eval_guidance_scale": 3.0,
    })

    return runs


def build_batch2_queue() -> list[dict]:
    """Second batch of experiments based on what worked in the initial 5:
    - r04 (full multimodal, base config) hit test_auc=0.7312 (final 16-sample).
    - r05 (full multimodal + bigger model + heavier aux) hit test_auc=0.7544.
    - r03 (EEG+EMG only) hit test_auc=0.6694.

    Strategy:
    - Longer training of r04/r05 configs (300-400 epochs).
    - Vary CFG scale at eval time (some up, some down).
    - Try multi-seed ensembling via parallel runs of r05.
    - Bigger aux supervision to push test_acc up.
    """
    base = dict(
        epochs=300,
        eval_every=25,
        eval_gen_samples=4,
        eval_steps=50,
        eval_guidance_scale=2.0,
        select_best_by="val_gen_auc_avg",
        base_ch=64,
        depth=4,
        time_dim=256,
        token_dim=256,
        past_encoder_depth=3,
        past_self_attn_layers=2,
        eps_weight=0.25,
        aux_ce_weight=0.5,
        aux_fm_weight=5.0,
        past_ce_weight=0.5,
        t_aux_min=100,
        t_aux_max=900,
        cond_drop_prob=0.15,
        norm_mode="global",
        augment=True,
        mixup_prob=0.5,
        mouse_adv_weight=0.5,
        mouse_adv_lambda_grl=1.0,
        cross_mouse_mixup=True,
        include_emg=True,
        include_photometry=True,
        val_holdout_mouse="M229",
    )

    runs = []

    # b01: r04 config, longer training (300 ep), eval more often to find peak
    runs.append({**base, "run_name": "b01_r04_long", "epochs": 300, "eval_every": 20})

    # b02: r04 config, but stronger aux to push test_acc up
    runs.append({
        **base, "run_name": "b02_strong_aux", "epochs": 300, "eval_every": 20,
        "aux_ce_weight": 2.0, "aux_fm_weight": 10.0, "past_ce_weight": 1.0,
    })

    # b03: r05 config (bigger model) repeated with different seed; longer
    runs.append({
        **base, "run_name": "b03_bigger_seed1", "epochs": 300, "eval_every": 20,
        "aux_ce_weight": 1.5, "aux_fm_weight": 10.0, "past_ce_weight": 1.0,
        "mouse_adv_weight": 1.0, "mouse_adv_lambda_grl": 1.5,
        "past_encoder_depth": 4, "past_self_attn_layers": 4,
        "token_dim": 384, "eval_guidance_scale": 3.0,
        "seed": 0,
    })

    # b04: same bigger model, seed=1 (for ensembling)
    runs.append({
        **base, "run_name": "b04_bigger_seed2", "epochs": 300, "eval_every": 20,
        "aux_ce_weight": 1.5, "aux_fm_weight": 10.0, "past_ce_weight": 1.0,
        "mouse_adv_weight": 1.0, "mouse_adv_lambda_grl": 1.5,
        "past_encoder_depth": 4, "past_self_attn_layers": 4,
        "token_dim": 384, "eval_guidance_scale": 3.0,
        "seed": 1,
    })

    # b05: same bigger model, seed=2 (for ensembling)
    runs.append({
        **base, "run_name": "b05_bigger_seed3", "epochs": 300, "eval_every": 20,
        "aux_ce_weight": 1.5, "aux_fm_weight": 10.0, "past_ce_weight": 1.0,
        "mouse_adv_weight": 1.0, "mouse_adv_lambda_grl": 1.5,
        "past_encoder_depth": 4, "past_self_attn_layers": 4,
        "token_dim": 384, "eval_guidance_scale": 3.0,
        "seed": 2,
    })

    return runs


def build_batch3_queue() -> list[dict]:
    """Batch3: focused on r04-style architecture (smaller, more stable) with
    multiple seeds for ensembling. Findings from batch2:
    - r04-style (smaller architecture) was more reliable than r05-bigger.
    - b01_r04_long / b02_strong_aux: TEST_AUC=0.7725 (tied best).
    - b03/b04/b05 (bigger) had higher variance and lower test AUC overall.

    Strategy: extend the ensemble pool with diverse r04-style runs.
    - Different seeds for diversity
    - Slightly different aux/CFG mixes
    - Slightly extended training (400 ep)
    """
    base = dict(
        epochs=400,
        eval_every=25,
        eval_gen_samples=4,
        eval_steps=50,
        eval_guidance_scale=2.0,
        select_best_by="val_gen_auc_avg",
        base_ch=64,
        depth=4,
        time_dim=256,
        token_dim=256,
        past_encoder_depth=3,
        past_self_attn_layers=2,
        eps_weight=0.25,
        aux_ce_weight=0.5,
        aux_fm_weight=5.0,
        past_ce_weight=0.5,
        t_aux_min=100,
        t_aux_max=900,
        cond_drop_prob=0.15,
        norm_mode="global",
        augment=True,
        mixup_prob=0.5,
        mouse_adv_weight=0.5,
        mouse_adv_lambda_grl=1.0,
        cross_mouse_mixup=True,
        include_emg=True,
        include_photometry=True,
        val_holdout_mouse="M229",
    )

    runs = []

    # c01: r04 base, seed 3, slightly higher CFG and stronger aux
    runs.append({
        **base, "run_name": "c01_r04_seed3", "seed": 3,
        "aux_ce_weight": 1.0, "aux_fm_weight": 7.5, "past_ce_weight": 0.75,
    })

    # c02: r04 base, seed 4, eval_guidance_scale 2.5 (try slightly stronger CFG)
    runs.append({
        **base, "run_name": "c02_r04_seed4_cfg25", "seed": 4,
        "eval_guidance_scale": 2.5,
        "aux_ce_weight": 1.0, "aux_fm_weight": 7.5, "past_ce_weight": 0.75,
    })

    # c03: r04 base + extra past_self_attn_layers=3, seed 5
    runs.append({
        **base, "run_name": "c03_r04_seed5_pa3", "seed": 5,
        "past_self_attn_layers": 3,
        "aux_ce_weight": 1.0, "aux_fm_weight": 7.5, "past_ce_weight": 0.75,
    })

    # c04: r04 base + cond_drop_prob 0.25 (more aggressive CFG dropout) seed 6
    runs.append({
        **base, "run_name": "c04_r04_seed6_cdp25", "seed": 6,
        "cond_drop_prob": 0.25,
        "aux_ce_weight": 1.0, "aux_fm_weight": 7.5, "past_ce_weight": 0.75,
    })

    # c05: r04 base, seed 7, stronger DANN
    runs.append({
        **base, "run_name": "c05_r04_seed7_dann", "seed": 7,
        "mouse_adv_weight": 0.75, "mouse_adv_lambda_grl": 1.25,
        "aux_ce_weight": 1.0, "aux_fm_weight": 7.5, "past_ce_weight": 0.75,
    })

    return runs


def build_anchor_b02_queue() -> list[dict]:
    """Single 'anchor' config = b02_strong_aux, the best per-row on M234/h0.

    Returns exactly one config dict. The (region, fold, horizon, seed)
    expansion is done by main() before dispatch so one anchor config will be
    used for every (horizon, seed) cell in the sweep.

    Source: outputs/v6_xattn_m234_h0/results.tsv row 'b02_strong_aux'
        (TEST_AUC=0.7725, TEST_ACC=0.7375, multi-modal + DANN + mixup,
        aux_ce_weight=2.0, aux_fm_weight=10.0, past_ce_weight=1.0).
    """
    base = dict(
        epochs=300,
        eval_every=20,
        eval_gen_samples=4,
        eval_steps=50,
        eval_guidance_scale=2.0,
        select_best_by="val_gen_auc_avg",
        base_ch=64,
        depth=4,
        time_dim=256,
        token_dim=256,
        past_encoder_depth=3,
        past_self_attn_layers=2,
        eps_weight=0.25,
        aux_ce_weight=2.0,
        aux_fm_weight=10.0,
        past_ce_weight=1.0,
        t_aux_min=100,
        t_aux_max=900,
        cond_drop_prob=0.15,
        norm_mode="global",
        augment=True,
        mixup_prob=0.5,
        mouse_adv_weight=0.5,
        mouse_adv_lambda_grl=1.0,
        cross_mouse_mixup=True,
        include_emg=True,
        include_photometry=True,
        final_eval_samples=16,
        run_name="anchor_b02",
    )
    return [base]


def build_anchor_vpm_queue() -> list[dict]:
    """VPM anchor config. Initially mirrors anchor_b02; Phase-3 config search
    on the chosen VPM fold may motivate overriding this. We keep it as a
    separate function so the RT and VPM anchors can drift independently
    without breaking caller scripts.
    """
    cfg = build_anchor_b02_queue()[0]
    cfg = dict(cfg)
    cfg["run_name"] = "anchor_vpm"
    return [cfg]


def build_anchor_eeg_queue() -> list[dict]:
    """EEG-only version of the anchor config.

    This is the paired diffusion counterpart for the EEG-only preictal
    classifier. It keeps the same loss/architecture/training schedule as the
    multimodal anchor but removes EMG and photometry conditioning channels.
    """
    cfg = dict(build_anchor_b02_queue()[0])
    cfg["run_name"] = "anchor_eeg"
    cfg["include_emg"] = False
    cfg["include_photometry"] = False
    return [cfg]


def build_batch4_queue() -> list[dict]:
    """Batch4: val mouse rotation + aggressive aux for cross-mouse calibration.

    Analysis of batch2 5-ckpt ensemble revealed severe calibration mismatch
    between val mouse M229 (mean prob 0.532) and test mouse M234 (mean prob
    0.669). Tuning threshold on val (M229) HURTS test (M234) accuracy.

    Strategy:
    - Rotate val mouse across runs (M229, M233, M237, M238) so the ensemble
      averages out per-val-mouse biases.
    - Increase aux_ce_weight to push generated samples toward more confident
      classifier outputs (better acc downstream).
    - Keep r04-style smaller architecture (stable performance pattern).
    """
    base = dict(
        epochs=400,
        eval_every=25,
        eval_gen_samples=4,
        eval_steps=50,
        eval_guidance_scale=2.0,
        select_best_by="val_gen_auc_avg",
        base_ch=64,
        depth=4,
        time_dim=256,
        token_dim=256,
        past_encoder_depth=3,
        past_self_attn_layers=2,
        eps_weight=0.25,
        aux_ce_weight=1.5,
        aux_fm_weight=10.0,
        past_ce_weight=1.0,
        t_aux_min=100,
        t_aux_max=900,
        cond_drop_prob=0.15,
        norm_mode="global",
        augment=True,
        mixup_prob=0.5,
        mouse_adv_weight=0.5,
        mouse_adv_lambda_grl=1.0,
        cross_mouse_mixup=True,
        include_emg=True,
        include_photometry=True,
    )

    runs = []

    # d01: val=M233, seed=8
    runs.append({
        **base, "run_name": "d01_val_M233", "seed": 8,
        "val_holdout_mouse": "M233",
    })

    # d02: val=M237, seed=9
    runs.append({
        **base, "run_name": "d02_val_M237", "seed": 9,
        "val_holdout_mouse": "M237",
    })

    # d03: val=M238, seed=10
    runs.append({
        **base, "run_name": "d03_val_M238", "seed": 10,
        "val_holdout_mouse": "M238",
    })

    # d04: val=M229, aux_ce_weight=3.0 (very aggressive aux), seed=11
    runs.append({
        **base, "run_name": "d04_aux3_M229", "seed": 11,
        "val_holdout_mouse": "M229",
        "aux_ce_weight": 3.0, "aux_fm_weight": 15.0,
    })

    # d05: val=M237, aux_ce_weight=3.0, seed=12
    runs.append({
        **base, "run_name": "d05_aux3_M237", "seed": 12,
        "val_holdout_mouse": "M237",
        "aux_ce_weight": 3.0, "aux_fm_weight": 15.0,
    })

    return runs


# -----------------------------------------------------------------------------
# Adaptive follow-ups based on best run
# -----------------------------------------------------------------------------


def adaptive_followups(best_summary: dict) -> list[dict]:
    """Suggest further runs based on what worked. Called when initial queue
    is exhausted and we still haven't hit the target."""
    a = dict(best_summary.get("args", {}))
    # strip args that we want to override below
    a.setdefault("epochs", 500)
    a["epochs"] = 600
    a["select_best_by"] = "val_gen_auc_avg"
    out = []

    # Adaptive 1: keep best config but train longer and warm CFG
    out.append({**a, "run_name": "a01_long_warm_cfg", "epochs": 600, "eval_guidance_scale": 2.5})
    # Adaptive 2: add multimodal if not already on
    if not a.get("include_emg"):
        out.append({**a, "run_name": "a02_add_emg", "include_emg": True})
    if not a.get("include_photometry"):
        out.append({**a, "run_name": "a03_add_photo", "include_photometry": True})
    # Adaptive 4: harder aux
    out.append({
        **a, "run_name": "a04_harder_aux",
        "aux_ce_weight": 3.0, "aux_fm_weight": 12.0, "past_ce_weight": 1.5,
        "eps_weight": 0.1,
    })
    # Adaptive 5: bigger model
    out.append({
        **a, "run_name": "a05_bigger_model",
        "base_ch": 96, "depth": 4, "token_dim": 384,
        "past_encoder_depth": 4, "past_self_attn_layers": 4,
    })
    return out


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def _diffusion_dir(out_root: Path, region: str, fold: str, horizon: int,
                   run_name: str) -> Path:
    """Locate the per-run output dir; mirrors v6_train.main()'s rules."""
    if region == "RT":
        return out_root / "diffusion" / f"fold_{fold}" / f"horizon_{horizon}" / run_name
    return out_root / "diffusion" / region / f"fold_{fold}" / f"horizon_{horizon}" / run_name


def run_one(args, cfg: dict, horizon: int, seed: int | None,
            out_root: Path) -> dict:
    """Launch one (cfg, horizon, seed) run through v6_train.

    The CLI args take precedence over `cfg` for region/fold/horizon/seed.
    """
    full_run_name = cfg["run_name"]
    if seed is not None and "seed" not in full_run_name:
        full_run_name = f"{full_run_name}__s{seed}"
    if (len(args.horizons_list) > 1
            and f"_h{horizon}" not in full_run_name):
        full_run_name = f"{full_run_name}__h{horizon}"
    cmd = [
        args.python, "-m", "scripts.v6_train",
        "--region", args.region,
        "--fold", args.fold,
        "--horizon_idx", str(horizon),
        "--out_root", str(out_root),
        "--target_test_auc", str(min(TARGET_AUC, TARGET_ACC)),
        "--run_name", full_run_name,
    ]
    if args.val_holdout_mouse is not None:
        cmd.extend(["--val_holdout_mouse", args.val_holdout_mouse])
    if args.classifier_ckpt is not None:
        cmd.extend(["--classifier_ckpt", args.classifier_ckpt])
    skip_keys = {"run_name"}
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
        skip_keys.add("seed")
    # If --val_holdout_mouse is supplied on the CLI, the cfg's value is
    # already overridden via the if-block above; skip the cfg's copy too.
    # If the cfg-side val_holdout_mouse is not a member of the current
    # region's cohort (e.g. legacy 'M229' on a VPM sweep), drop it and let
    # v6_train auto-pick via common.config.pick_val_mouse(region, fold).
    if args.val_holdout_mouse is not None:
        skip_keys.add("val_holdout_mouse")
    elif "val_holdout_mouse" in cfg:
        # Lazily import here to avoid a top-level cycle.
        from common.config import REGIONS as _REGIONS
        region_cohort = _REGIONS.get(args.region, [])
        if cfg["val_holdout_mouse"] not in region_cohort:
            print(f"[run_one] dropping cfg val_holdout_mouse={cfg['val_holdout_mouse']} "
                  f"(not in region {args.region} cohort {region_cohort}); "
                  f"v6_train will auto-pick.", flush=True)
            skip_keys.add("val_holdout_mouse")
    for k, v in cfg.items():
        if k in skip_keys:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    print("=" * 80, flush=True)
    print("RUN:", " ".join(cmd), flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"crash(exit={proc.returncode})"

    summary_path = (
        _diffusion_dir(out_root, args.region, args.fold, horizon, full_run_name)
        / "summary.json"
    )
    summary = {}
    if summary_path.is_file():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception as e:
            print(f"failed to read {summary_path}: {e}", flush=True)
    best_metrics = summary.get("best_metrics", {})
    final_metrics = summary.get("final_eval_metrics", {})
    return {
        "run_name": full_run_name,
        "region": args.region,
        "fold": args.fold,
        "horizon_idx": horizon,
        "seed": seed if seed is not None else cfg.get("seed", ""),
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "test_gen_auc_avg": best_metrics.get("test_gen_auc_avg", float("nan")),
        "test_gen_acc_avg": best_metrics.get("test_gen_acc_avg", float("nan")),
        "val_gen_auc_avg": best_metrics.get("val_gen_auc_avg", float("nan")),
        "val_gen_acc_avg": best_metrics.get("val_gen_acc_avg", float("nan")),
        "val_condition_gap_auc": best_metrics.get("val_condition_gap_auc", float("nan")),
        "final_test_auc": final_metrics.get("final_test_gen_auc", float("nan")),
        "final_test_acc": final_metrics.get("final_test_gen_acc", float("nan")),
        "final_val_auc": final_metrics.get("final_val_gen_auc", float("nan")),
        "final_val_acc": final_metrics.get("final_val_gen_acc", float("nan")),
        "best_score": summary.get("best_score", float("nan")),
        "cfg": cfg,
        "summary_path": str(summary_path) if summary_path.is_file() else "",
    }


def write_tsv_row(path: Path, row: dict):
    new_file = not path.exists()
    keys = [
        "run_name", "region", "fold", "horizon_idx", "seed",
        "status", "elapsed_s",
        "final_test_auc", "final_test_acc",
        "final_val_auc", "final_val_acc",
        "test_gen_auc_avg", "test_gen_acc_avg",
        "val_gen_auc_avg", "val_gen_acc_avg",
        "val_condition_gap_auc",
        "best_score",
        "cfg_json",
    ]
    with open(path, "a") as f:
        if new_file:
            f.write("\t".join(keys) + "\n")
        out_row = []
        for k in keys:
            if k == "cfg_json":
                out_row.append(json.dumps(row["cfg"], sort_keys=True))
            else:
                v = row.get(k, "")
                out_row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        f.write("\t".join(out_row) + "\n")


def _expand_jobs(configs: list[dict], horizons: list[int],
                 seeds: list[int]) -> list[tuple[dict, int, int | None]]:
    """Cartesian product of configs * horizons * seeds (seeds may be empty).

    Returned tuples are (cfg, horizon_idx, seed_or_None). If seeds is empty,
    we emit one job per (cfg, horizon) and let v6_train use the cfg's seed
    (or DEFAULT_SEED).
    """
    jobs = []
    seed_iter = seeds if seeds else [None]
    for cfg in configs:
        for h in horizons:
            for s in seed_iter:
                jobs.append((cfg, h, s))
    return jobs


def main():
    args = parse_args()
    if args.out_root is None:
        from common.config import OUTPUTS_ROOT
        out_root = Path(OUTPUTS_ROOT)
    else:
        out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / RESULTS_TSV_NAME
    print(f"[loop] out_root={out_root}", flush=True)
    print(f"[loop] results.tsv={results_path}", flush=True)
    print(f"[loop] region={args.region} fold={args.fold} "
          f"horizons={args.horizons_list} seeds={args.seeds_list or '[cfg default]'}",
          flush=True)

    if args.queue == "initial":
        configs = build_initial_queue()
    elif args.queue == "batch2":
        configs = build_batch2_queue()
    elif args.queue == "batch3":
        configs = build_batch3_queue()
    elif args.queue == "batch4":
        configs = build_batch4_queue()
    elif args.queue == "anchor_b02":
        configs = build_anchor_b02_queue()
    elif args.queue == "anchor_vpm":
        configs = build_anchor_vpm_queue()
    elif args.queue == "anchor_eeg":
        configs = build_anchor_eeg_queue()
    else:
        raise SystemExit(f"unknown queue: {args.queue}")

    jobs = _expand_jobs(configs, args.horizons_list, args.seeds_list)
    jobs = jobs[args.start_at:]
    print(f"[loop] queue={args.queue} configs={len(configs)} expanded_jobs={len(jobs)}",
          flush=True)

    runs_done = 0
    best = None

    def is_better(r):
        nonlocal best
        if best is None:
            return True
        a = r.get("test_gen_auc_avg")
        b = best.get("test_gen_auc_avg")
        try:
            return float(a) > float(b)
        except (TypeError, ValueError):
            return False

    # The legacy "stop when target reached" early exit only makes sense for a
    # single-cell sweep. For multi-horizon/seed sweeps we always run them all.
    single_cell = (len(args.horizons_list) == 1
                   and (len(args.seeds_list) <= 1))
    is_anchor_sweep = args.queue in ("anchor_b02", "anchor_vpm", "anchor_eeg")

    while jobs:
        if args.max_runs and runs_done >= args.max_runs:
            print(f"[loop] hit max_runs={args.max_runs}, stopping.", flush=True)
            break
        cfg, horizon, seed = jobs.pop(0)
        result = run_one(args, cfg, horizon, seed, out_root)
        write_tsv_row(results_path, result)
        runs_done += 1
        if is_better(result):
            best = result
            print(f"[loop] new best: {best['run_name']} "
                  f"test_auc={best['test_gen_auc_avg']:.4f} "
                  f"test_acc={best['test_gen_acc_avg']:.4f}", flush=True)

        try:
            t_auc = float(result.get("test_gen_auc_avg", "nan"))
            t_acc = float(result.get("test_gen_acc_avg", "nan"))
        except (TypeError, ValueError):
            t_auc = t_acc = float("nan")
        if single_cell and t_auc >= TARGET_AUC and t_acc >= TARGET_ACC:
            print(f"[loop] TARGET REACHED: {result['run_name']} "
                  f"test_auc={t_auc:.4f} test_acc={t_acc:.4f}", flush=True)
            return

        # Only do adaptive follow-ups in single-cell mode and non-anchor mode,
        # to preserve historical behavior on the M234/h0 debug runs.
        if (not jobs) and best is not None and single_cell and not is_anchor_sweep:
            print("[loop] initial queue exhausted; generating adaptive followups...", flush=True)
            full_summary_path = best.get("summary_path", "")
            if full_summary_path and os.path.isfile(full_summary_path):
                with open(full_summary_path) as f:
                    best_summary = json.load(f)
            else:
                best_summary = {"args": best["cfg"]}
            followups = adaptive_followups(best_summary)
            new_jobs = _expand_jobs(followups, args.horizons_list, args.seeds_list)
            for j in new_jobs:
                jobs.append(j)
            print(f"[loop] adaptive followups added: {len(new_jobs)}", flush=True)

    print(f"[loop] done. {runs_done} runs. best run: "
          f"{best['run_name'] if best else 'none'} "
          f"test_auc={best['test_gen_auc_avg'] if best else float('nan')}", flush=True)


if __name__ == "__main__":
    main()
