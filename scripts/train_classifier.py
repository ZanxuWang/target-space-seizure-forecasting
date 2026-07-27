"""train_classifier.py

Train an EEG classifier on a single sub-window (target or preictal horizon)
for one leave-one-out fold.

Roles:
- --kind target : 0-3s post-onset window. Becomes the frozen auxiliary
  classifier for diffusion training AND the fixed evaluator of generated
  spectrograms.
- --kind horizon --horizon_idx I : one of the six preictal horizons. Used
  solely to report baseline predictive signal from real preictal windows.

Usage:
    python -m scripts.train_classifier --fold M229 --kind target
    python -m scripts.train_classifier --fold M229 --kind horizon --horizon_idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as `python scripts/train_classifier.py` from the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common.config import (
    HORIZONS, TARGET_SEC, OUTPUTS_ROOT, RT_REGION, REGIONS, DEFAULT_SEED,
    mice_for_region,
)
from common.datasets import HorizonSpecDataset, load_mice_eeg
from common.classifier import EEGClassifier
from common.loo_splits import fold_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=RT_REGION, choices=sorted(REGIONS.keys()),
                   help="Cohort region (RT or VPM).")
    p.add_argument("--fold", required=True,
                   help="Held-out mouse ID (leave-one-out test mouse). Must be one "
                        "of the mice for --region.")
    p.add_argument("--kind", required=True, choices=["target", "horizon"])
    p.add_argument("--horizon_idx", type=int, default=None,
                   help="Horizon index 0..5 (required if --kind horizon)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--select_best_by", default="val_acc",
                   choices=["val_acc", "val_auc"],
                   help="Checkpoint selection metric. Use val_auc for final "
                        "paper preictal baselines; val_acc preserves legacy runs.")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_root", type=str, default=OUTPUTS_ROOT)
    args = p.parse_args()
    region_mice = mice_for_region(args.region)
    if args.fold not in region_mice:
        raise SystemExit(
            f"--fold {args.fold} not in region {args.region} cohort {region_mice}"
        )
    return args


def resolve_window(args) -> tuple[tuple, str]:
    """Return (start_sec, end_sec) and a subdir name."""
    if args.kind == "target":
        return TARGET_SEC, "target"
    if args.horizon_idx is None:
        raise SystemExit("--horizon_idx is required when --kind horizon")
    if not (0 <= args.horizon_idx < len(HORIZONS)):
        raise SystemExit(f"--horizon_idx must be in [0, {len(HORIZONS) - 1}]")
    h = HORIZONS[args.horizon_idx]
    return h, f"horizon_{args.horizon_idx}"


def make_splits(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    prob = np.concatenate(probs)
    return y, prob


def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    y, prob = collect_probs(model, loader, device)
    p = (prob >= 0.5).astype(np.int64)
    acc = float((p == y).mean())
    # Simple AUC without sklearn dependency
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"acc": acc, "auc": auc}


def selection_score(metrics: dict, select_best_by: str) -> float:
    key = "auc" if select_best_by == "val_auc" else "acc"
    value = float(metrics.get(key, float("nan")))
    return value if np.isfinite(value) else -1.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (start_sec, end_sec), sub_name = resolve_window(args)
    region_mice = mice_for_region(args.region)
    train_mice = [m for m in region_mice if m != args.fold]
    test_mouse = args.fold

    print(f"[region={args.region} fold={args.fold}] kind={args.kind} "
          f"window=({start_sec},{end_sec})s")
    print(f"  train mice: {train_mice}")
    print(f"  test  mouse: {test_mouse}")

    X_trainmice, y_trainmice, _ = load_mice_eeg(train_mice, region=args.region)
    X_test, y_test, _ = load_mice_eeg([test_mouse], region=args.region)

    # Internal train/val split from the training mice
    tr_idx, val_idx = make_splits(len(X_trainmice), args.val_frac, args.seed)
    X_tr, y_tr = X_trainmice[tr_idx], y_trainmice[tr_idx]
    X_va, y_va = X_trainmice[val_idx], y_trainmice[val_idx]

    print(f"  rows: train={len(X_tr)} val={len(X_va)} test={len(X_test)}")
    print(f"  train pos rate: {y_tr.mean():.3f}, val pos rate: {y_va.mean():.3f}, test pos rate: {y_test.mean():.3f}")

    train_ds = HorizonSpecDataset(X_tr, y_tr, start_sec, end_sec)
    val_ds = HorizonSpecDataset(X_va, y_va, start_sec, end_sec)
    test_ds = HorizonSpecDataset(X_test, y_test, start_sec, end_sec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGClassifier(n_classes=2, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # When running on multiple regions under one VERSION, keep classifier
    # outputs region-segregated to avoid collisions (fold_M229 in RT vs VPM
    # would otherwise overwrite each other if mouse ID overlapped). For RT the
    # legacy layout `classifiers/fold_<MOUSE>/<sub_name>/` is preserved.
    if args.region == RT_REGION:
        out_dir = os.path.join(args.out_root, "classifiers", fold_name(args.fold), sub_name)
    else:
        out_dir = os.path.join(args.out_root, "classifiers", args.region,
                               fold_name(args.fold), sub_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_acc,val_auc,test_acc,test_auc,lr\n")

    best_val = -1.0
    wait = 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"[{args.fold}/{sub_name}] epoch {ep}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        val_m = evaluate(model, val_loader, device)
        test_m = evaluate(model, test_loader, device)
        lr_now = float(sched.get_last_lr()[0])

        train_loss = float(np.mean(losses))
        print(f"  epoch {ep}: train_loss={train_loss:.4f} "
              f"val_acc={val_m['acc']:.3f} val_auc={val_m['auc']:.3f} "
              f"test_acc={test_m['acc']:.3f} test_auc={test_m['auc']:.3f}")

        with open(log_path, "a") as f:
            f.write(f"{ep},{train_loss:.6f},{val_m['acc']:.4f},{val_m['auc']:.4f},"
                    f"{test_m['acc']:.4f},{test_m['auc']:.4f},{lr_now:.6e}\n")

        val_score = selection_score(val_m, args.select_best_by)
        if val_score > best_val + 1e-6:
            best_val = val_score
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stopping at epoch {ep} "
                      f"(best {args.select_best_by}={best_val:.3f})")
                break

    # Load best weights for final save / test eval
    if best_state is not None:
        model.load_state_dict(best_state)
    val_y, val_probs = collect_probs(model, val_loader, device)
    test_y, test_probs = collect_probs(model, test_loader, device)
    final_test = evaluate(model, test_loader, device)
    final_val = evaluate(model, val_loader, device)
    print(f"  [best] val_acc={final_val['acc']:.3f} val_auc={final_val['auc']:.3f} "
          f"test_acc={final_test['acc']:.3f} test_auc={final_test['auc']:.3f}")

    ckpt_path = os.path.join(out_dir, "best.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "region": args.region,
        "fold": args.fold,
        "kind": args.kind,
        "horizon_idx": args.horizon_idx,
        "window_sec": [start_sec, end_sec],
        "val_metrics": final_val,
        "test_metrics": final_test,
        "selection_metric": args.select_best_by,
        "best_selection_score": best_val,
    }, ckpt_path)
    np.savez(
        os.path.join(out_dir, "final_eval_probs.npz"),
        val_y=val_y.astype(np.int64),
        val_probs=val_probs.astype(np.float32),
        test_y=test_y.astype(np.int64),
        test_probs=test_probs.astype(np.float32),
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "region": args.region,
            "fold": args.fold,
            "kind": args.kind,
            "horizon_idx": args.horizon_idx,
            "window_sec": [start_sec, end_sec],
            "val": final_val,
            "test": final_test,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_test)),
            "pos_rate_test": float(y_test.mean()),
            "selection_metric": args.select_best_by,
            "best_selection_score": float(best_val),
            "ckpt": ckpt_path,
        }, f, indent=2)
    print(f"  saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
