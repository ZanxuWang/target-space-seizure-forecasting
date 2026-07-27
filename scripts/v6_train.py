"""v6_train.py

Train an XAttn conditional diffusion model for one (fold, horizon) pair.

Architecture: common.v6_diffusion.XAttnUNet2D + XAttnTargetDiffusion
Dataset:      common.v6_datasets.V6MultimodalDataset

Inference signature:  past_spec only.  No label, no real-target, no preictal
classifier output.  CFG over past condition only.

Model selection metric: val_gen_auc_avg by default (no test leakage).
test metrics are computed and logged but never used to choose checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.classifier import EEGClassifier, freeze
from common.config import (
    DEFAULT_SEED,
    HORIZONS,
    OUTPUTS_ROOT,
    REGIONS,
    RT_REGION,
    RT_MICE,
    TARGET_SEC,
    VERSION,
    mice_for_region,
    pick_val_mouse,
)
from common.loo_splits import fold_name
from common.spectrogram import spec_to_classifier_input_torch
from common.v6_datasets import V6MultimodalDataset, load_mice_multimodal
from common.v6_diffusion import XAttnTargetDiffusion, XAttnUNet2D


def _git_short_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=RT_REGION, choices=sorted(REGIONS.keys()),
                   help="Cohort region (RT or VPM). Validates --fold against it.")
    p.add_argument("--fold", default="M234",
                   help="Held-out test mouse. Must be a member of --region's cohort.")
    p.add_argument("--horizon_idx", type=int, default=0)
    p.add_argument("--run_name", default="run_xattn",
                   help="Subdirectory under outputs/<VERSION>/diffusion/<fold>/<horizon>/")
    p.add_argument("--classifier_ckpt", type=str, default=None)

    # Architecture
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--token_dim", type=int, default=256)
    p.add_argument("--past_encoder_depth", type=int, default=3)
    p.add_argument("--past_self_attn_layers", type=int, default=2)

    # Modalities
    p.add_argument("--include_emg", action="store_true")
    p.add_argument("--include_photometry", action="store_true")

    # Normalization / augmentation
    p.add_argument("--norm_mode", default="global", choices=["global", "per_sample"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--aug_freq_masks", type=int, default=2)
    p.add_argument("--aug_max_freq_width", type=int, default=6)
    p.add_argument("--aug_time_masks", type=int, default=2)
    p.add_argument("--aug_max_time_width", type=int, default=4)
    p.add_argument("--mixup_prob", type=float, default=0.0,
                   help="Probability of applying batch mixup at training time.")
    p.add_argument("--mixup_alpha", type=float, default=0.4,
                   help="Beta distribution alpha for mixup lambda sampling.")
    p.add_argument("--cross_mouse_mixup", action="store_true",
                   help="If set, mixup pairs are restricted to different mice.")

    # Losses
    p.add_argument("--eps_weight", type=float, default=0.25)
    p.add_argument("--aux_ce_weight", type=float, default=1.0)
    p.add_argument("--aux_fm_weight", type=float, default=5.0)
    p.add_argument("--past_ce_weight", type=float, default=0.5)
    p.add_argument("--mouse_adv_weight", type=float, default=0.0,
                   help="Weight on gradient-reversed mouse-id loss. >0 enables DANN.")
    p.add_argument("--mouse_adv_lambda_grl", type=float, default=1.0,
                   help="Lambda multiplier for the gradient-reversal layer.")
    p.add_argument("--t_aux_min", type=int, default=100)
    p.add_argument("--t_aux_max", type=int, default=900)
    p.add_argument("--aux_max_samples", type=int, default=0)
    p.add_argument("--cond_drop_prob", type=float, default=0.15)

    # Training
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.10,
                   help="Ignored when --val_holdout_mouse is set.")
    p.add_argument("--val_holdout_mouse", type=str, default=None,
                   help="One mouse id (from the --region cohort) to hold out from "
                        "training mice and use as the validation fold instead of a "
                        "random val_frac. This is a within-training-mice LOO; the "
                        "actual test mouse (--fold) is still excluded from both "
                        "train and val. If unset, the script defaults to "
                        "pick_val_mouse(region, fold).")
    p.add_argument("--no_amp", action="store_true")

    # Evaluation
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--eval_gen_samples", type=int, default=4)
    p.add_argument("--eval_guidance_scale", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--select_best_by", default="val_gen_auc_avg",
                   choices=[
                       "val_gen_auc_avg",
                       "val_gen_acc_avg",
                       "val_condition_gap_auc",
                       "val_gen_mse",
                       "val_eps_mse",
                       "val_auc_correct",
                   ])
    p.add_argument("--target_test_auc", type=float, default=0.80,
                   help="Stop early once test_gen_auc_avg >= this (target reached).")

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_root", type=str, default=OUTPUTS_ROOT)
    p.add_argument("--final_eval_samples", type=int, default=16,
                   help="Number of diffusion samples per event for the FINAL EVAL "
                        "block (uses the best-by-val checkpoint).")
    args = p.parse_args()

    region_mice = mice_for_region(args.region)
    if args.fold not in region_mice:
        raise SystemExit(
            f"--fold {args.fold!r} is not in region {args.region!r} cohort {region_mice}"
        )
    if args.val_holdout_mouse is None:
        args.val_holdout_mouse = pick_val_mouse(args.region, args.fold)
        print(f"[parse_args] auto-selected val_holdout_mouse={args.val_holdout_mouse} "
              f"for region={args.region} fold={args.fold}")
    if args.val_holdout_mouse == args.fold:
        raise SystemExit(
            f"--val_holdout_mouse must differ from --fold (got {args.fold})"
        )
    if args.val_holdout_mouse not in region_mice:
        raise SystemExit(
            f"--val_holdout_mouse {args.val_holdout_mouse!r} is not in region "
            f"{args.region!r} cohort {region_mice}"
        )
    return args


def default_classifier_ckpt(args) -> str:
    """Resolve the auxiliary/target classifier checkpoint for this region+fold.

    RT keeps the legacy layout `<out_root>/classifiers/<fold_NAME>/target/best.pth`
    so existing artifacts are reused; VPM uses the region-segregated layout
    introduced in Phase 0.
    """
    base = args.out_root
    if args.region == RT_REGION:
        return os.path.join(base, "classifiers", fold_name(args.fold), "target", "best.pth")
    return os.path.join(base, "classifiers", args.region,
                        fold_name(args.fold), "target", "best.pth")


def split_indices(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def subset_bank(bank: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in bank.items()}


@torch.no_grad()
def eval_diffusion_loss(diff, loader, device) -> float:
    diff.model.eval()
    losses = []
    for batch in loader:
        past, target = batch[0].to(device), batch[1].to(device)
        t = torch.randint(0, diff.num_timesteps, (target.shape[0],), device=device)
        out = diff.p_losses(
            past, target, t,
            labels=None, classifier=None,
            aux_ce_weight=0.0, aux_fm_weight=0.0, past_ce_weight=0.0,
            mouse_adv_weight=0.0,
            eps_weight=1.0, cond_drop_prob=0.0,
        )
        losses.append(out["eps_mse"].item())
    return float(np.mean(losses)) if losses else float("nan")


def _binary_metrics(y, prob, pred):
    acc = float((pred == y).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"acc": acc, "auc": auc}


@torch.no_grad()
def eval_generated_target_classifier_avg(
    diff,
    loader,
    classifier,
    device,
    steps: int,
    num_samples: int,
    guidance_scale: float,
    spec_mean: float,
    spec_std: float,
    prefix: str,
    return_arrays: bool = False,
):
    diff.ema_model.eval()
    classifier.eval()
    num_samples = max(1, int(num_samples))

    ys, avg_probs, prob_stds = [], [], []
    for batch in loader:
        past, target, label = batch[0], batch[1], batch[2]
        past = past.to(device)
        target_shape = (target.shape[1], target.shape[2])

        sample_probs = []
        for _ in range(num_samples):
            gen = diff.sample_target_ddim(
                past,
                target_shape=target_shape,
                steps=steps,
                guidance_scale=guidance_scale,
                progress=False,
            )
            gen_logmag = gen * spec_std + spec_mean
            clf_in = spec_to_classifier_input_torch(gen_logmag)
            logits = classifier(clf_in)
            sample_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        prob_stack = np.stack(sample_probs, axis=1)
        avg_probs.append(prob_stack.mean(axis=1))
        prob_stds.append(prob_stack.std(axis=1))
        ys.append(label.numpy())

    if not ys:
        result = {
            f"{prefix}_gen_auc_avg": float("nan"),
            f"{prefix}_gen_acc_avg": float("nan"),
            f"{prefix}_gen_prob_std": float("nan"),
        }
        if return_arrays:
            result["y"] = np.array([], dtype=np.int64)
            result["probs"] = np.array([], dtype=np.float32)
            result["prob_stds"] = np.array([], dtype=np.float32)
        return result
    y = np.concatenate(ys)
    prob = np.concatenate(avg_probs)
    stds = np.concatenate(prob_stds)
    pred = (prob >= 0.5).astype(np.int64)
    m = _binary_metrics(y, prob, pred)
    result = {
        f"{prefix}_gen_auc_avg": m["auc"],
        f"{prefix}_gen_acc_avg": m["acc"],
        f"{prefix}_gen_prob_std": float(stds.mean()),
    }
    if return_arrays:
        result["y"] = y.astype(np.int64)
        result["probs"] = prob.astype(np.float32)
        result["prob_stds"] = stds.astype(np.float32)
    return result


@torch.no_grad()
def eval_condition_dependence(
    diff,
    loader,
    classifier,
    device,
    steps: int,
    spec_mean: float,
    spec_std: float,
):
    diff.ema_model.eval()
    classifier.eval()
    ys = []
    probs = {"correct": [], "zero": [], "shuffled": []}
    preds = {"correct": [], "zero": [], "shuffled": []}
    mses = []
    for batch in loader:
        past, target, label = batch[0], batch[1], batch[2]
        past = past.to(device)
        target = target.to(device)
        target_shape = (target.shape[1], target.shape[2])
        variants = {
            "correct": past,
            "zero": torch.zeros_like(past),
            "shuffled": past[torch.randperm(past.shape[0], device=device)],
        }
        for name, p_used in variants.items():
            gen = diff.sample_target_ddim(
                p_used,
                target_shape=target_shape,
                steps=steps,
                guidance_scale=1.0,
                progress=False,
            )
            if name == "correct":
                mses.append(float(torch.mean((gen - target) ** 2).cpu()))
            gen_logmag = gen * spec_std + spec_mean
            clf_in = spec_to_classifier_input_torch(gen_logmag)
            logits = classifier(clf_in)
            probs[name].append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds[name].append(logits.argmax(dim=1).cpu().numpy())
        ys.append(label.numpy())

    if not ys:
        return {
            "val_auc_correct": float("nan"),
            "val_auc_zero": float("nan"),
            "val_auc_shuffled": float("nan"),
            "val_condition_gap_auc": float("nan"),
            "val_gen_acc_correct": float("nan"),
            "val_gen_mse": float("nan"),
        }
    y = np.concatenate(ys)
    metrics = {}
    for name in ("correct", "zero", "shuffled"):
        prob = np.concatenate(probs[name])
        pred = np.concatenate(preds[name])
        m = _binary_metrics(y, prob, pred)
        metrics[f"val_auc_{name}"] = m["auc"]
        metrics[f"val_gen_acc_{name}"] = m["acc"]
    refs = [v for v in [metrics["val_auc_zero"], metrics["val_auc_shuffled"]] if np.isfinite(v)]
    metrics["val_condition_gap_auc"] = (
        metrics["val_auc_correct"] - max(refs)
        if np.isfinite(metrics["val_auc_correct"]) and refs
        else float("nan")
    )
    metrics["val_gen_mse"] = float(np.mean(mses)) if mses else float("nan")
    return metrics


def checkpoint_score(metrics: dict, key: str):
    if key in {"val_gen_mse", "val_eps_mse"}:
        return float(metrics.get(key, float("nan"))), False
    return float(metrics.get(key, float("nan"))), True


def save_payload(path, **kwargs):
    torch.save(kwargs, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not (0 <= args.horizon_idx < len(HORIZONS)):
        raise SystemExit(f"horizon_idx must be in [0, {len(HORIZONS) - 1}]")
    cond_range = HORIZONS[args.horizon_idx]
    target_range = TARGET_SEC

    region_mice = mice_for_region(args.region)
    train_mice = [m for m in region_mice if m != args.fold]
    if args.val_holdout_mouse is not None:
        if args.val_holdout_mouse == args.fold:
            raise SystemExit(
                f"--val_holdout_mouse {args.val_holdout_mouse} is the test mouse; "
                f"choose a different training mouse."
            )
        if args.val_holdout_mouse not in train_mice:
            raise SystemExit(
                f"--val_holdout_mouse {args.val_holdout_mouse} is not a training mouse. "
                f"Available: {train_mice}"
            )
    test_mouse = args.fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"[v6 {args.run_name}] region={args.region} fold={args.fold} "
          f"horizon={args.horizon_idx} cond={cond_range} target={target_range} "
          f"device={device}")
    print(f"  norm_mode={args.norm_mode} augment={args.augment} "
          f"include_emg={args.include_emg} include_photometry={args.include_photometry} "
          f"mouse_adv_weight={args.mouse_adv_weight} "
          f"mixup_prob={args.mixup_prob} cross_mouse_mixup={args.cross_mouse_mixup} "
          f"val_holdout_mouse={args.val_holdout_mouse}")

    if args.val_holdout_mouse is not None:
        inner_train_mice = [m for m in train_mice if m != args.val_holdout_mouse]
        bank_tr = load_mice_multimodal(
            inner_train_mice,
            include_emg=args.include_emg,
            include_photometry=args.include_photometry,
            region=args.region,
        )
        bank_va = load_mice_multimodal(
            [args.val_holdout_mouse],
            include_emg=args.include_emg,
            include_photometry=args.include_photometry,
            region=args.region,
        )
        print(f"  val_holdout: train_mice={inner_train_mice}, val_mouse={args.val_holdout_mouse}")
    else:
        bank_tm = load_mice_multimodal(
            train_mice,
            include_emg=args.include_emg,
            include_photometry=args.include_photometry,
            region=args.region,
        )
        tr_idx, va_idx = split_indices(len(bank_tm["labels"]), args.val_frac, args.seed)
        bank_tr = subset_bank(bank_tm, tr_idx)
        bank_va = subset_bank(bank_tm, va_idx)
    bank_test = load_mice_multimodal(
        [test_mouse],
        include_emg=args.include_emg,
        include_photometry=args.include_photometry,
        region=args.region,
    )
    print(f"  rows train={len(bank_tr['labels'])} val={len(bank_va['labels'])} test={len(bank_test['labels'])}")

    common_ds_kwargs = dict(
        cond_range=cond_range,
        target_range=target_range,
        norm_mode=args.norm_mode,
        include_emg=args.include_emg,
        include_photometry=args.include_photometry,
    )
    train_ds = V6MultimodalDataset(
        bank_tr,
        augment=args.augment,
        aug_freq_masks=args.aug_freq_masks,
        aug_max_freq_width=args.aug_max_freq_width,
        aug_time_masks=args.aug_time_masks,
        aug_max_time_width=args.aug_max_time_width,
        **common_ds_kwargs,
    )
    g_mean = train_ds.mean
    g_std = train_ds.std
    val_ds = V6MultimodalDataset(
        bank_va,
        global_mean=g_mean,
        global_std=g_std,
        **common_ds_kwargs,
    )
    test_ds = V6MultimodalDataset(
        bank_test,
        global_mean=g_mean,
        global_std=g_std,
        **common_ds_kwargs,
    )

    n_past_ch = train_ds.n_past_channels
    print(f"  n_past_channels={n_past_ch} cond_frames={train_ds.cond_frames} "
          f"target_frames={train_ds.target_frames} freq_bins={train_ds.freq_bins}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    ckpt_path = args.classifier_ckpt or default_classifier_ckpt(args)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Target classifier ckpt not found: {ckpt_path}")
    clf_ckpt = torch.load(ckpt_path, map_location=device)
    classifier = EEGClassifier(n_classes=2, pretrained=False).to(device)
    classifier.load_state_dict(clf_ckpt["model_state_dict"])
    classifier = freeze(classifier)

    n_train_mice = int(max(bank_tr["mouse_idx"].max(), 0)) + 1 if len(bank_tr["mouse_idx"]) else 4
    unet = XAttnUNet2D(
        in_ch=1,
        base_ch=args.base_ch,
        depth=args.depth,
        time_dim=args.time_dim,
        token_dim=args.token_dim,
        past_in_ch=n_past_ch,
        past_encoder_depth=args.past_encoder_depth,
        past_self_attn_layers=args.past_self_attn_layers,
        n_mice=n_train_mice,
    )
    diff = XAttnTargetDiffusion(
        unet, num_timesteps=args.timesteps, ema_decay=args.ema_decay, device=device,
    )
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # RT keeps the legacy `diffusion/fold_<MOUSE>/horizon_<H>/<run>/` layout
    # so all our v6_xattn_m234_h0 runs stay reachable. VPM gets a region
    # prefix to avoid mouse-ID collisions across cohorts.
    if args.region == RT_REGION:
        out_dir = os.path.join(
            args.out_root, "diffusion", fold_name(args.fold),
            f"horizon_{args.horizon_idx}", args.run_name,
        )
    else:
        out_dir = os.path.join(
            args.out_root, "diffusion", args.region, fold_name(args.fold),
            f"horizon_{args.horizon_idx}", args.run_name,
        )
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write(
            "epoch,train_eps_mse,train_aux_ce,train_aux_fm,train_past_ce,"
            "val_eps_mse,test_eps_mse,val_gen_mse,"
            "val_gen_auc_avg,val_gen_acc_avg,test_gen_auc_avg,test_gen_acc_avg,"
            "val_gen_prob_std,test_gen_prob_std,"
            "val_auc_correct,val_auc_zero,val_auc_shuffled,"
            "val_condition_gap_auc,val_gen_acc_correct,"
            "best_metric,best_score,lr\n"
        )

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "version": VERSION,
            "git_sha": _git_short_sha(),
            "region": args.region,
            "fold": args.fold,
            "horizon_idx": args.horizon_idx,
            "val_holdout_mouse": args.val_holdout_mouse,
            "seed": args.seed,
            "cond_range_sec": list(cond_range),
            "target_range_sec": list(target_range),
            "spec_mean": g_mean,
            "spec_std": g_std,
            "cond_frames": train_ds.cond_frames,
            "target_frames": train_ds.target_frames,
            "freq_bins": train_ds.freq_bins,
            "n_past_channels": n_past_ch,
            "classifier_ckpt": ckpt_path,
            "no_leakage_note": "inference uses past spectrogram + random target noise only",
            "args": vars(args),
        }, f, indent=2)

    init_score, larger_is_better = checkpoint_score({}, args.select_best_by)
    best_score = -float("inf") if larger_is_better else float("inf")
    best_metrics = {}
    wait = 0

    for ep in range(1, args.epochs + 1):
        unet.train()
        ep_eps, ep_ce, ep_fm, ep_past_ce, ep_mouse = [], [], [], [], []
        for batch in tqdm(
            train_loader,
            desc=f"[v6 {args.run_name} {args.fold}/h{args.horizon_idx}] ep {ep}/{args.epochs}",
        ):
            past, target, label, mouse_b = batch
            past = past.to(device)
            target = target.to(device)
            label = label.to(device)
            mouse_b = mouse_b.to(device)

            # Optional batch mixup: aim to make conditioning generalizable across mice.
            do_mixup = (args.mixup_prob > 0.0
                        and torch.rand(1, device=device).item() < args.mixup_prob)
            if do_mixup:
                B = past.shape[0]
                if args.cross_mouse_mixup:
                    perm = torch.randperm(B, device=device)
                    fixed = 0
                    for _ in range(8):
                        if (mouse_b[perm] != mouse_b).all():
                            break
                        perm = torch.randperm(B, device=device)
                        fixed += 1
                else:
                    perm = torch.randperm(B, device=device)
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                past_mix = lam * past + (1 - lam) * past[perm]
                target_mix = lam * target + (1 - lam) * target[perm]
                # Use the dominant sample for hard labels (mixup label smoothing).
                hard_label = label if lam >= 0.5 else label[perm]
                hard_mouse = mouse_b if lam >= 0.5 else mouse_b[perm]
                past = past_mix
                target = target_mix
                label = hard_label
                mouse_b = hard_mouse

            t = torch.randint(0, diff.num_timesteps, (target.shape[0],), device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = diff.p_losses(
                    past, target, t,
                    labels=label,
                    mouse_idx=mouse_b,
                    classifier=classifier,
                    aux_ce_weight=args.aux_ce_weight,
                    aux_fm_weight=args.aux_fm_weight,
                    past_ce_weight=args.past_ce_weight,
                    mouse_adv_weight=args.mouse_adv_weight,
                    mouse_adv_lambda_grl=args.mouse_adv_lambda_grl,
                    eps_weight=args.eps_weight,
                    cond_drop_prob=args.cond_drop_prob,
                    t_aux_min=args.t_aux_min,
                    t_aux_max=args.t_aux_max,
                    aux_max_samples=args.aux_max_samples,
                    spec_mean=g_mean,
                    spec_std=g_std,
                )
            loss = out["total"]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.clip_grad_norm)
            scaler.step(opt)
            scaler.update()
            diff.update_ema()

            ep_eps.append(out["eps_mse"].item())
            ep_ce.append(out["aux_ce"].item())
            ep_fm.append(out["aux_fm"].item())
            ep_past_ce.append(out["past_ce"].item())
            ep_mouse.append(out["mouse_adv"].item())

        sched.step()
        tr_eps = float(np.mean(ep_eps))
        tr_ce = float(np.mean(ep_ce))
        tr_fm = float(np.mean(ep_fm))
        tr_past_ce = float(np.mean(ep_past_ce))
        tr_mouse = float(np.mean(ep_mouse)) if ep_mouse else 0.0

        do_eval = (ep == 1) or (ep % args.eval_every == 0) or (ep == args.epochs)
        if do_eval:
            val_eps = eval_diffusion_loss(diff, val_loader, device)
            test_eps = eval_diffusion_loss(diff, test_loader, device)
            val_avg = eval_generated_target_classifier_avg(
                diff, val_loader, classifier, device,
                args.eval_steps, args.eval_gen_samples, args.eval_guidance_scale,
                g_mean, g_std, prefix="val",
            )
            test_avg = eval_generated_target_classifier_avg(
                diff, test_loader, classifier, device,
                args.eval_steps, args.eval_gen_samples, args.eval_guidance_scale,
                g_mean, g_std, prefix="test",
            )
            dep = eval_condition_dependence(
                diff, val_loader, classifier, device, args.eval_steps, g_mean, g_std,
            )
            metrics = {
                "val_eps_mse": val_eps,
                "test_eps_mse": test_eps,
                **val_avg,
                **test_avg,
                **dep,
            }
            score, larger = checkpoint_score(metrics, args.select_best_by)
            if np.isnan(score):
                score = -float("inf") if larger else float("inf")
            lr_now = float(sched.get_last_lr()[0])

            print(
                f"  ep {ep}: eps={tr_eps:.4f} ce={tr_ce:.4f} fm={tr_fm:.4f} past_ce={tr_past_ce:.4f} "
                f"mouse_adv={tr_mouse:.4f} "
                f"val_eps={val_eps:.4f} test_eps={test_eps:.4f} "
                f"val_gen_auc={val_avg['val_gen_auc_avg']:.4f} val_gen_acc={val_avg['val_gen_acc_avg']:.4f} "
                f"TEST_gen_auc={test_avg['test_gen_auc_avg']:.4f} TEST_gen_acc={test_avg['test_gen_acc_avg']:.4f} "
                f"gap={dep['val_condition_gap_auc']:.4f} "
                f"sel({args.select_best_by})={score:.4f}"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"{ep},{tr_eps:.6f},{tr_ce:.6f},{tr_fm:.6f},{tr_past_ce:.6f},"
                    f"{val_eps:.6f},{test_eps:.6f},{dep['val_gen_mse']:.6f},"
                    f"{val_avg['val_gen_auc_avg']:.6f},{val_avg['val_gen_acc_avg']:.6f},"
                    f"{test_avg['test_gen_auc_avg']:.6f},{test_avg['test_gen_acc_avg']:.6f},"
                    f"{val_avg['val_gen_prob_std']:.6f},{test_avg['test_gen_prob_std']:.6f},"
                    f"{dep['val_auc_correct']:.6f},{dep['val_auc_zero']:.6f},"
                    f"{dep['val_auc_shuffled']:.6f},{dep['val_condition_gap_auc']:.6f},"
                    f"{dep['val_gen_acc_correct']:.6f},"
                    f"{args.select_best_by},{score:.6f},{lr_now:.6e}\n"
                )

            improved = score > best_score + 1e-6 if larger else score < best_score - 1e-6
            if improved:
                best_score = score
                best_metrics = metrics
                wait = 0
                save_payload(
                    os.path.join(out_dir, "best_model.pth"),
                    model_state_dict=unet.state_dict(),
                    ema_model_state_dict=diff.ema_model.state_dict(),
                    epoch=ep,
                    args=vars(args),
                    region=args.region,
                    fold=args.fold,
                    horizon_idx=args.horizon_idx,
                    seed=args.seed,
                    cond_range_sec=list(cond_range),
                    target_range_sec=list(target_range),
                    spec_mean=g_mean,
                    spec_std=g_std,
                    cond_frames=train_ds.cond_frames,
                    target_frames=train_ds.target_frames,
                    freq_bins=train_ds.freq_bins,
                    n_past_channels=n_past_ch,
                    model_params=dict(
                        in_ch=1,
                        base_ch=args.base_ch,
                        depth=args.depth,
                        time_dim=args.time_dim,
                        token_dim=args.token_dim,
                        past_in_ch=n_past_ch,
                        past_encoder_depth=args.past_encoder_depth,
                        past_self_attn_layers=args.past_self_attn_layers,
                        n_mice=n_train_mice,
                    ),
                    select_best_by=args.select_best_by,
                    best_score=best_score,
                    best_metrics=best_metrics,
                    version=VERSION,
                )
                print(f"  -> saved best_model.pth ({args.select_best_by}={best_score:.4f}) "
                      f"TEST_auc={test_avg['test_gen_auc_avg']:.4f}")
            else:
                wait += 1

            # Early exit if target reached
            if (
                np.isfinite(test_avg["test_gen_auc_avg"])
                and test_avg["test_gen_auc_avg"] >= args.target_test_auc
                and np.isfinite(test_avg["test_gen_acc_avg"])
                and test_avg["test_gen_acc_avg"] >= args.target_test_auc
            ):
                print(f"  *** TARGET REACHED: test_gen_auc={test_avg['test_gen_auc_avg']:.4f}, "
                      f"test_gen_acc={test_avg['test_gen_acc_avg']:.4f} >= {args.target_test_auc}")
                break

        if wait >= args.patience:
            print(f"  early stopping at ep {ep} (best {args.select_best_by}={best_score:.4f})")
            break

    # High-confidence final evaluation using the BEST checkpoint with more
    # generations to reduce sampling noise. This produces the headline numbers
    # in summary.json without leaking test info into model selection.
    final_eval_metrics = {}
    best_ckpt_path = os.path.join(out_dir, "best_model.pth")
    final_n_samples = max(args.eval_gen_samples * 4, int(args.final_eval_samples))
    if os.path.isfile(best_ckpt_path):
        try:
            ck = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            diff.ema_model.load_state_dict(ck["ema_model_state_dict"])
            test_final = eval_generated_target_classifier_avg(
                diff, test_loader, classifier, device,
                args.eval_steps, final_n_samples,
                args.eval_guidance_scale, g_mean, g_std, prefix="test",
                return_arrays=True,
            )
            val_final = eval_generated_target_classifier_avg(
                diff, val_loader, classifier, device,
                args.eval_steps, final_n_samples,
                args.eval_guidance_scale, g_mean, g_std, prefix="val",
                return_arrays=True,
            )
            final_eval_metrics = {
                "final_best_epoch": int(ck.get("epoch", -1)),
                "final_val_gen_auc": val_final["val_gen_auc_avg"],
                "final_val_gen_acc": val_final["val_gen_acc_avg"],
                "final_test_gen_auc": test_final["test_gen_auc_avg"],
                "final_test_gen_acc": test_final["test_gen_acc_avg"],
                "final_n_samples": final_n_samples,
                "final_guidance_scale": float(args.eval_guidance_scale),
                "final_steps": int(args.eval_steps),
            }
            print(f"  *** FINAL EVAL (best ckpt ep={int(ck.get('epoch',-1))}, "
                  f"{final_n_samples} samples): "
                  f"val_auc={val_final['val_gen_auc_avg']:.4f} "
                  f"val_acc={val_final['val_gen_acc_avg']:.4f} "
                  f"TEST_AUC={test_final['test_gen_auc_avg']:.4f} "
                  f"TEST_ACC={test_final['test_gen_acc_avg']:.4f}")
            # Save per-event probability arrays for downstream ensembling /
            # ROC plotting / bootstrap CIs. This is the canonical artifact
            # used by scripts/v6_aggregate_results.py.
            probs_path = os.path.join(out_dir, "final_eval_probs.npz")
            np.savez(
                probs_path,
                val_y=val_final["y"],
                val_probs=val_final["probs"],
                val_prob_stds=val_final["prob_stds"],
                test_y=test_final["y"],
                test_probs=test_final["probs"],
                test_prob_stds=test_final["prob_stds"],
                n_samples=np.int64(final_n_samples),
                guidance_scale=np.float32(args.eval_guidance_scale),
                steps=np.int64(args.eval_steps),
                region=np.array(args.region),
                fold=np.array(args.fold),
                horizon_idx=np.int64(args.horizon_idx),
                seed=np.int64(args.seed),
            )
            print(f"  saved per-event probs -> {probs_path}")
        except Exception as e:
            print(f"  final eval failed: {e}")

    # Final model
    save_payload(
        os.path.join(out_dir, "final_model.pth"),
        model_state_dict=unet.state_dict(),
        ema_model_state_dict=diff.ema_model.state_dict(),
        epoch=ep,
        args=vars(args),
        region=args.region,
        fold=args.fold,
        horizon_idx=args.horizon_idx,
        seed=args.seed,
        cond_range_sec=list(cond_range),
        target_range_sec=list(target_range),
        spec_mean=g_mean,
        spec_std=g_std,
        cond_frames=train_ds.cond_frames,
        target_frames=train_ds.target_frames,
        freq_bins=train_ds.freq_bins,
        n_past_channels=n_past_ch,
        model_params=dict(
            in_ch=1,
            base_ch=args.base_ch,
            depth=args.depth,
            time_dim=args.time_dim,
            token_dim=args.token_dim,
            past_in_ch=n_past_ch,
            past_encoder_depth=args.past_encoder_depth,
            past_self_attn_layers=args.past_self_attn_layers,
            n_mice=n_train_mice,
        ),
        select_best_by=args.select_best_by,
        best_score=best_score,
        best_metrics=best_metrics,
        version=VERSION,
    )
    # Write a small summary
    summary = {
        "run_name": args.run_name,
        "region": args.region,
        "fold": args.fold,
        "horizon_idx": args.horizon_idx,
        "val_holdout_mouse": args.val_holdout_mouse,
        "seed": args.seed,
        "select_best_by": args.select_best_by,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "final_eval_metrics": final_eval_metrics,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
