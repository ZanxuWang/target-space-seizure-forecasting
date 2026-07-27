"""train_classifier_mm.py

Train a multi-modal preictal/target classifier as a *fair baseline* for the
multi-modal diffusion pipeline. Mirrors ``scripts/train_classifier.py`` but
uses ``common.classifier_mm.MMClassifier`` (parameterised input channels)
and a multi-modal spectrogram stack:

    channel 0 : EEG  log1p STFT (per-sample min-max + resize to 224x224)
    channel 1 : EMG  log1p STFT (per-sample min-max + resize)   [if --include_emg]
    channel ? : Photo time-binned summary, broadcast across freq, then resize
                                                                [if --include_photometry]

The output classifier is what we compare the multi-modal diffusion pipeline
against, controlling for input modalities. To keep the existing EEG-only
artifacts intact, this script writes under

    <out_root>/classifiers_mm/<REGION>/fold_<MOUSE>/<MODALITY_TAG>/<SUB>/

where MODALITY_TAG is one of {eeg, eeg_emg, eeg_photo, eeg_emg_photo}
(auto-derived from --include_emg / --include_photometry), and SUB is
``target`` or ``horizon_<H>`` exactly like ``train_classifier.py``.

Usage examples
--------------

    # EEG+Photo preictal classifier at horizon 0 for VPM/M710:
    python -m scripts.train_classifier_mm \
        --region VPM --fold M710 --kind horizon --horizon_idx 0 \
        --include_photometry

    # Full multi-modal preictal classifier:
    python -m scripts.train_classifier_mm \
        --region RT --fold M234 --kind horizon --horizon_idx 0 \
        --include_emg --include_photometry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common.config import (
    HORIZONS, TARGET_SEC, OUTPUTS_ROOT, RT_REGION, REGIONS, DEFAULT_SEED,
    CLASSIFIER_IMG_HW, mice_for_region,
)
from common.datasets import extract_subwindow
from common.spectrogram import signals_to_spectrograms
from common.v6_datasets import (
    _photometry_summary_channel,
    load_mice_multimodal,
)
from common.classifier_mm import MMClassifier
from common.loo_splits import fold_name


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def _per_sample_minmax_resize(x: np.ndarray, out_hw: tuple) -> np.ndarray:
    """Per-sample min-max to [0,1] then bilinear resize to ``out_hw``.

    x: [N, F, T] float32  ->  [N, H, W] float32.

    Matches the normalization used by ``spec_to_classifier_input_np`` so
    each modality channel is on the same numerical footing as the EEG
    channel of the legacy EEG-only baseline.
    """
    t = torch.from_numpy(x.astype(np.float32))[:, None]  # [N,1,F,T]
    N = t.shape[0]
    flat = t.view(N, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-8)
    t = flat.view_as(t)
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[:, 0].cpu().numpy()


class MultimodalSpecDataset(Dataset):
    """One subwindow -> a [C_in, H, W] spectrogram stack + binary label.

    Each channel is per-sample min-maxed and resized to CLASSIFIER_IMG_HW
    (typically 224x224). No cross-sample statistics are used so train/val/
    test splits can be built independently without leakage.

    ``data_bank`` is the dict returned by ``load_mice_multimodal``.
    """

    def __init__(
        self,
        data_bank: dict,
        start_sec: float,
        end_sec: float,
        include_emg: bool,
        include_photometry: bool,
        out_hw: tuple = CLASSIFIER_IMG_HW,
    ):
        self.include_emg = include_emg
        self.include_photometry = include_photometry

        eeg = data_bank["eeg"]
        eeg_sub = extract_subwindow(eeg, start_sec, end_sec)
        eeg_specs = signals_to_spectrograms(eeg_sub)
        F_dim, T_dim = eeg_specs.shape[1], eeg_specs.shape[2]
        chans = [_per_sample_minmax_resize(eeg_specs, out_hw)]

        if include_emg:
            emg = data_bank["emg"]
            emg_sub = extract_subwindow(emg, start_sec, end_sec)
            emg_specs = signals_to_spectrograms(emg_sub)
            chans.append(_per_sample_minmax_resize(emg_specs, out_hw))

        if include_photometry:
            photo = data_bank["photometry"]
            photo_sub = extract_subwindow(photo, start_sec, end_sec)
            photo_chan = _photometry_summary_channel(photo_sub, F_dim, T_dim)
            chans.append(_per_sample_minmax_resize(photo_chan, out_hw))

        self.X = np.stack(chans, axis=1).astype(np.float32)  # [N, C, H, W]
        self.labels = data_bank["labels"].astype(np.int64)
        self.n_channels = int(self.X.shape[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=RT_REGION, choices=sorted(REGIONS.keys()))
    p.add_argument("--fold", required=True,
                   help="Held-out test mouse. Must be a member of --region's cohort.")
    p.add_argument("--kind", required=True, choices=["target", "horizon"])
    p.add_argument("--horizon_idx", type=int, default=None,
                   help="Horizon index 0..5 (required if --kind horizon)")
    p.add_argument("--include_emg", action="store_true")
    p.add_argument("--include_photometry", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
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
    if args.kind == "target":
        return TARGET_SEC, "target"
    if args.horizon_idx is None:
        raise SystemExit("--horizon_idx is required when --kind horizon")
    if not (0 <= args.horizon_idx < len(HORIZONS)):
        raise SystemExit(f"--horizon_idx must be in [0, {len(HORIZONS) - 1}]")
    return HORIZONS[args.horizon_idx], f"horizon_{args.horizon_idx}"


def modality_tag(args) -> str:
    tags = ["eeg"]
    if args.include_emg:
        tags.append("emg")
    if args.include_photometry:
        tags.append("photo")
    return "_".join(tags)


def make_splits(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate(model, loader, device) -> dict:
    y, prob = collect_probs(model, loader, device)
    p = (prob >= 0.5).astype(np.int64)
    acc = float((p == y).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"acc": acc, "auc": auc}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (start_sec, end_sec), sub_name = resolve_window(args)
    region_mice = mice_for_region(args.region)
    train_mice = [m for m in region_mice if m != args.fold]
    test_mouse = args.fold
    tag = modality_tag(args)

    print(f"[region={args.region} fold={args.fold}] kind={args.kind} "
          f"window=({start_sec},{end_sec})s modality={tag}")
    print(f"  train mice: {train_mice}")
    print(f"  test  mouse: {test_mouse}")

    bank_tr_full = load_mice_multimodal(
        train_mice, region=args.region,
        include_emg=args.include_emg, include_photometry=args.include_photometry,
    )
    bank_test = load_mice_multimodal(
        [test_mouse], region=args.region,
        include_emg=args.include_emg, include_photometry=args.include_photometry,
    )

    tr_idx, va_idx = make_splits(len(bank_tr_full["labels"]), args.val_frac, args.seed)
    bank_tr = {k: v[tr_idx] for k, v in bank_tr_full.items()}
    bank_va = {k: v[va_idx] for k, v in bank_tr_full.items()}

    print(f"  rows: train={len(bank_tr['labels'])} val={len(bank_va['labels'])} "
          f"test={len(bank_test['labels'])}")
    print(f"  train pos rate: {bank_tr['labels'].mean():.3f}, "
          f"val pos rate: {bank_va['labels'].mean():.3f}, "
          f"test pos rate: {bank_test['labels'].mean():.3f}")

    train_ds = MultimodalSpecDataset(
        bank_tr, start_sec, end_sec, args.include_emg, args.include_photometry,
    )
    val_ds = MultimodalSpecDataset(
        bank_va, start_sec, end_sec, args.include_emg, args.include_photometry,
    )
    test_ds = MultimodalSpecDataset(
        bank_test, start_sec, end_sec, args.include_emg, args.include_photometry,
    )
    n_channels = train_ds.n_channels
    print(f"  n_input_channels={n_channels}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMClassifier(
        n_classes=2, pretrained=True, in_channels=n_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = os.path.join(
        args.out_root, "classifiers_mm", args.region, fold_name(args.fold), tag, sub_name,
    )
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
        for x, y in tqdm(train_loader, desc=f"[{tag}/{args.fold}/{sub_name}] ep {ep}/{args.epochs}"):
            x = x.to(device); y = y.to(device)
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
        print(f"  ep {ep}: train_loss={train_loss:.4f} "
              f"val_acc={val_m['acc']:.3f} val_auc={val_m['auc']:.3f} "
              f"test_acc={test_m['acc']:.3f} test_auc={test_m['auc']:.3f}")
        with open(log_path, "a") as f:
            f.write(f"{ep},{train_loss:.6f},{val_m['acc']:.4f},{val_m['auc']:.4f},"
                    f"{test_m['acc']:.4f},{test_m['auc']:.4f},{lr_now:.6e}\n")
        val_score = float(val_m["auc"]) if np.isfinite(val_m["auc"]) else -1.0
        if val_score > best_val + 1e-6:
            best_val = val_score
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stopping at epoch {ep} (best val_auc={best_val:.3f})")
                break

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
        "modality_tag": tag,
        "in_channels": n_channels,
        "val_metrics": final_val,
        "test_metrics": final_test,
        "selection_metric": "val_auc",
        "best_val_auc": best_val,
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
            "modality_tag": tag,
            "include_emg": bool(args.include_emg),
            "include_photometry": bool(args.include_photometry),
            "in_channels": n_channels,
            "window_sec": [start_sec, end_sec],
            "val": final_val,
            "test": final_test,
            "n_train": int(len(bank_tr["labels"])),
            "n_val": int(len(bank_va["labels"])),
            "n_test": int(len(bank_test["labels"])),
            "pos_rate_test": float(bank_test["labels"].mean()),
            "selection_metric": "val_auc",
            "best_val_auc": float(best_val),
            "ckpt": ckpt_path,
        }, f, indent=2)
    print(f"  saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
