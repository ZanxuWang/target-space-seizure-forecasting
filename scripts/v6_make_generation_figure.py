"""Create classifier-aligned qualitative and target-space diagnostic figures.

The diffusion model generates a distribution of plausible post-onset targets;
it is not trained as a paired future reconstructor. Accordingly, this script
never averages generated spectrograms for display. It shows an actual generated
sample, the probability distribution across all five selected checkpoints, and
cohort-wide target-classifier feature/probability diagnostics.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.classifier import EEGClassifier, freeze
from common.config import HORIZONS, TARGET_SEC
from common.spectrogram import spec_to_classifier_input_torch
from common.v6_datasets import V6MultimodalDataset, load_mice_multimodal
from common.v6_diffusion import XAttnTargetDiffusion, XAttnUNet2D


OUT_DIR = ROOT / "results" / "paper" / "generated_target_artifacts"

CASES = {
    "RT": {
        "mouse": "M238",
        "run_root": ROOT / "outputs" / "v6_paper_RT_M238_mm_allh"
        / "diffusion" / "fold_M238" / "horizon_0",
        "run_name": "anchor_b02__s{seed}__h0",
        "ensemble": ROOT / "outputs" / "v6_paper_RT_M238_mm_allh"
        / "ensembles" / "fold_M238" / "horizon_0"
        / "eeg_emg_photo" / "ensemble_probs.npz",
        "target_classifier": ROOT / "outputs" / "v6_xattn_m234_h0"
        / "classifiers" / "fold_M238" / "target" / "best.pth",
        "direct": ROOT / "outputs" / "v6_rt_h0_fold_screen"
        / "classifiers_mm" / "RT" / "fold_M238" / "eeg_emg_photo"
        / "horizon_0" / "final_eval_probs.npz",
    },
    "VPM": {
        "mouse": "M710",
        "run_root": ROOT / "outputs" / "v6_paper_VPM_M710_allh"
        / "diffusion" / "VPM" / "fold_M710" / "horizon_0",
        "run_name": "anchor_vpm__s{seed}__h0",
        "ensemble": ROOT / "outputs" / "v6_paper_VPM_M710_allh"
        / "ensembles" / "VPM" / "fold_M710" / "horizon_0"
        / "eeg_emg_photo" / "ensemble_probs.npz",
        "target_classifier": ROOT / "outputs" / "v6_paper_VPM_classifiers"
        / "classifiers" / "VPM" / "fold_M710" / "target" / "best.pth",
        "direct": ROOT / "outputs" / "v6_paper_VPM_classifiers"
        / "classifiers_mm" / "VPM" / "fold_M710" / "eeg_emg_photo"
        / "horizon_0" / "final_eval_probs.npz",
    },
}

CLASS_COLORS = {0: "#0072B2", 1: "#D55E00"}
DOMAIN_COLORS = {"Direct": "#999999", "Real target": "#009E73", "Diffusion": "#CC79A7"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.2,
    "axes.titlesize": 9,
    "axes.labelsize": 8.2,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.7,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="RT,VPM")
    parser.add_argument("--samples_per_checkpoint", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--refresh_alignment",
        action="store_true",
        help="Regenerate one target per event and checkpoint for PCA diagnostics.",
    )
    parser.add_argument(
        "--skip_alignment",
        action="store_true",
        help="Only create the two per-cohort qualitative figures.",
    )
    return parser.parse_args()


def _torch_load(path: Path, device: torch.device | str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_path(case: dict, seed: int) -> Path:
    return case["run_root"] / case["run_name"].format(seed=seed) / "best_model.pth"


def _load_diffusion(path: Path, device: torch.device) -> tuple[XAttnTargetDiffusion, dict]:
    checkpoint = _torch_load(path, device)
    model = XAttnUNet2D(**checkpoint["model_params"])
    diffusion = XAttnTargetDiffusion(
        model,
        num_timesteps=int(checkpoint["args"]["timesteps"]),
        ema_decay=float(checkpoint["args"].get("ema_decay", 0.995)),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    diffusion.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    model.eval()
    diffusion.ema_model.eval()
    return diffusion, checkpoint


def _load_classifier(path: Path, device: torch.device) -> EEGClassifier:
    payload = _torch_load(path, device)
    classifier = EEGClassifier(pretrained=False).to(device)
    classifier.load_state_dict(payload["model_state_dict"])
    return freeze(classifier)


def _load_region(region: str, device: torch.device) -> dict:
    case = CASES[region]
    seed0 = _torch_load(_checkpoint_path(case, 0), "cpu")
    spec_mean = float(seed0["spec_mean"])
    spec_std = float(seed0["spec_std"])
    bank = load_mice_multimodal(
        [case["mouse"]],
        region=region,
        include_emg=True,
        include_photometry=True,
    )
    dataset = V6MultimodalDataset(
        bank,
        HORIZONS[0],
        TARGET_SEC,
        norm_mode="global",
        global_mean=spec_mean,
        global_std=spec_std,
        include_emg=True,
        include_photometry=True,
        augment=False,
    )
    with np.load(case["ensemble"]) as payload:
        labels = payload["test_y"].astype(np.int64)
        ensemble_probs = payload["test_probs"].astype(np.float32)
    with np.load(case["direct"]) as payload:
        direct_labels = payload["test_y"].astype(np.int64)
        direct_probs = payload["test_probs"].astype(np.float32)
    if not np.array_equal(labels, dataset.labels):
        raise RuntimeError(f"Ensemble and dataset order differ for {region}")
    if not np.array_equal(labels, direct_labels):
        raise RuntimeError(f"Direct-classifier and dataset order differ for {region}")

    classifier = _load_classifier(case["target_classifier"], device)
    real_logmag = dataset.target_specs * spec_std + spec_mean
    real_probs, real_features = _classify_specs(
        classifier, real_logmag, device, batch_size=64
    )
    return {
        "region": region,
        "mouse": case["mouse"],
        "case": case,
        "dataset": dataset,
        "classifier": classifier,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "labels": labels,
        "ensemble_probs": ensemble_probs,
        "direct_probs": direct_probs,
        "real_probs": real_probs,
        "real_features": real_features,
    }


@torch.inference_mode()
def _classify_specs(
    classifier: EEGClassifier,
    specs: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(specs, dtype=torch.float32)
    probs, features = [], []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size].to(device)
        classifier_input = spec_to_classifier_input_torch(batch)
        feat = classifier.encode(classifier_input)
        logits = classifier.classifier(feat)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        features.append(feat.cpu().numpy())
    return np.concatenate(probs), np.concatenate(features)


def _select_high_confidence_examples(region_data: dict) -> np.ndarray:
    """Select one transparent, jointly class-consistent event per class."""
    labels = region_data["labels"]
    ensemble = region_data["ensemble_probs"]
    real = region_data["real_probs"]
    direct = region_data["direct_probs"]
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if label == 0:
            joint_error = ensemble[candidates] + real[candidates] + direct[candidates]
        else:
            joint_error = (
                (1.0 - ensemble[candidates])
                + (1.0 - real[candidates])
                + (1.0 - direct[candidates])
            )
        selected.append(int(candidates[np.argmin(joint_error)]))
    return np.asarray(selected, dtype=np.int64)


@torch.inference_mode()
def _generate_selected(
    region_data: dict,
    selected: np.ndarray,
    samples_per_checkpoint: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> dict:
    dataset = region_data["dataset"]
    classifier = region_data["classifier"]
    spec_mean = region_data["spec_mean"]
    spec_std = region_data["spec_std"]
    all_specs, all_probs, all_seeds = [], [], []

    for checkpoint_seed in range(5):
        checkpoint_path = _checkpoint_path(region_data["case"], checkpoint_seed)
        diffusion, checkpoint = _load_diffusion(checkpoint_path, device)
        if float(checkpoint["spec_mean"]) != spec_mean or float(checkpoint["spec_std"]) != spec_std:
            raise RuntimeError(f"Normalization mismatch in {checkpoint_path}")

        torch.manual_seed(seed + checkpoint_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + checkpoint_seed)
        seed_specs, seed_probs = [], []
        for idx in selected:
            past = torch.from_numpy(dataset.past_specs[idx]).float().to(device)
            past = past.unsqueeze(0).repeat(samples_per_checkpoint, 1, 1, 1)
            generated = diffusion.sample_target_ddim(
                past,
                target_shape=(dataset.freq_bins, dataset.target_frames),
                steps=steps,
                guidance_scale=guidance_scale,
                progress=False,
            )
            generated_logmag = generated * spec_std + spec_mean
            classifier_input = spec_to_classifier_input_torch(generated_logmag)
            logits = classifier(classifier_input)
            seed_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            seed_specs.append(generated_logmag.cpu().numpy())
        all_specs.append(np.stack(seed_specs, axis=0))
        all_probs.append(np.stack(seed_probs, axis=0))
        all_seeds.append(np.full((len(selected), samples_per_checkpoint), checkpoint_seed))

        del diffusion, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    specs = np.concatenate(all_specs, axis=1)
    probs = np.concatenate(all_probs, axis=1)
    checkpoint_seeds = np.concatenate(all_seeds, axis=1)
    exemplar_indices = np.asarray([
        int(np.argmin(np.abs(probs[row] - region_data["ensemble_probs"][idx])))
        for row, idx in enumerate(selected)
    ])
    exemplars = np.stack([
        specs[row, exemplar_indices[row]] for row in range(len(selected))
    ])
    exemplar_probs = np.asarray([
        probs[row, exemplar_indices[row]] for row in range(len(selected))
    ])
    exemplar_seeds = np.asarray([
        checkpoint_seeds[row, exemplar_indices[row]] for row in range(len(selected))
    ])
    return {
        "selected": selected,
        "generated_specs": specs,
        "generated_probs": probs,
        "checkpoint_seeds": checkpoint_seeds,
        "exemplars": exemplars,
        "exemplar_probs": exemplar_probs,
        "exemplar_seeds": exemplar_seeds,
    }


def _classifier_view(spec: np.ndarray) -> np.ndarray:
    spec = np.asarray(spec, dtype=np.float32)
    lo, hi = float(spec.min()), float(spec.max())
    return (spec - lo) / (hi - lo + 1e-8)


def _normalize_spec_batch(specs: np.ndarray) -> np.ndarray:
    specs = np.asarray(specs, dtype=np.float32)
    lo = specs.min(axis=(1, 2), keepdims=True)
    hi = specs.max(axis=(1, 2), keepdims=True)
    return (specs - lo) / (hi - lo + 1e-8)


def _cohens_d(class0: np.ndarray, class1: np.ndarray) -> np.ndarray:
    n0, n1 = len(class0), len(class1)
    pooled = np.sqrt(
        (
            (n0 - 1) * class0.var(axis=0, ddof=1)
            + (n1 - 1) * class1.var(axis=0, ddof=1)
        )
        / max(n0 + n1 - 2, 1)
        + 1e-8
    )
    return (class1.mean(axis=0) - class0.mean(axis=0)) / pooled


def _frequency_effect_with_ci(
    specs: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_bootstrap: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = specs.mean(axis=2)
    class0 = profiles[labels == 0]
    class1 = profiles[labels == 1]
    effect = _cohens_d(class0, class1)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((n_bootstrap, profiles.shape[1]), dtype=np.float32)
    for draw in range(n_bootstrap):
        sample0 = class0[rng.integers(0, len(class0), len(class0))]
        sample1 = class1[rng.integers(0, len(class1), len(class1))]
        bootstrap[draw] = _cohens_d(sample0, sample1)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return effect, lower, upper


def _event_averaged_generated_specs(data: dict) -> np.ndarray:
    """Average checkpoints within event after classifier-side normalization."""
    labels = data["labels"]
    generated = data["generated_specs"]
    seeds = data["generated_seeds"]
    per_seed = []
    for checkpoint_seed in sorted(np.unique(seeds)):
        seed_specs = generated[seeds == checkpoint_seed]
        if len(seed_specs) != len(labels):
            raise RuntimeError(
                f"Expected one generated sample per event for seed {checkpoint_seed}"
            )
        per_seed.append(_normalize_spec_batch(seed_specs))
    return np.stack(per_seed, axis=0).mean(axis=0)


def _draw_class_prototypes(
    region: str,
    mouse: str,
    data: dict,
    out_dir: Path,
) -> None:
    """Draw all-event class prototypes and standardized contrast maps."""
    labels = data["labels"]
    real_specs = _normalize_spec_batch(data["real_specs"])
    generated_specs = _event_averaged_generated_specs(data)
    rows = [
        ("Real targets", real_specs, 1201),
        ("Generated targets", generated_specs, 1202),
    ]

    fig = plt.figure(figsize=(7.15, 4.6))
    grid = fig.add_gridspec(
        2, 7,
        width_ratios=[1.0, 1.0, 0.045, 1.0, 0.045, 0.09, 1.16],
        left=0.075, right=0.985, bottom=0.11, top=0.83,
        wspace=0.24, hspace=0.48,
    )
    axes = np.empty((2, 4), dtype=object)
    mean_color_axes = []
    effect_color_axes = []
    for row in range(2):
        axes[row, 0] = fig.add_subplot(grid[row, 0])
        axes[row, 1] = fig.add_subplot(grid[row, 1])
        mean_color_axes.append(fig.add_subplot(grid[row, 2]))
        axes[row, 2] = fig.add_subplot(grid[row, 3])
        effect_color_axes.append(fig.add_subplot(grid[row, 4]))
        axes[row, 3] = fig.add_subplot(grid[row, 6])
    frequencies = np.linspace(0, 60, real_specs.shape[1])
    titles = [
        "Mean non-ictal target",
        "Mean ictal target",
        "Ictal minus non-ictal\nstandardized contrast",
        "Frequency-wise class effect",
    ]

    saved = {}
    for row, (row_name, specs, bootstrap_seed) in enumerate(rows):
        class0 = specs[labels == 0]
        class1 = specs[labels == 1]
        mean0 = class0.mean(axis=0)
        mean1 = class1.mean(axis=0)
        contrast = _cohens_d(class0, class1)
        freq_effect, freq_lower, freq_upper = _frequency_effect_with_ci(
            specs, labels, seed=bootstrap_seed
        )
        saved[f"{row_name.lower().replace(' ', '_')}_mean_nonictal"] = mean0
        saved[f"{row_name.lower().replace(' ', '_')}_mean_ictal"] = mean1
        saved[f"{row_name.lower().replace(' ', '_')}_effect_map"] = contrast
        saved[f"{row_name.lower().replace(' ', '_')}_frequency_effect"] = freq_effect
        saved[f"{row_name.lower().replace(' ', '_')}_frequency_ci_low"] = freq_lower
        saved[f"{row_name.lower().replace(' ', '_')}_frequency_ci_high"] = freq_upper

        mean_values = np.concatenate([mean0.ravel(), mean1.ravel()])
        mean_vmin, mean_vmax = np.percentile(mean_values, [2, 98])
        for col, image in enumerate((mean0, mean1)):
            ax = axes[row, col]
            im = ax.imshow(
                image, origin="lower", aspect="auto", cmap="magma",
                vmin=mean_vmin, vmax=mean_vmax,
                extent=[0, 3, 0, 60], interpolation="nearest",
            )
            ax.set_xlabel("Time from onset (s)")
            if col == 0:
                ax.set_ylabel(f"{row_name}\nFrequency (Hz)", fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, width=0.6)
            if col == 1:
                colorbar = fig.colorbar(im, cax=mean_color_axes[row])
                colorbar.ax.set_title(
                    "Normalized\nintensity",
                    fontsize=5.8,
                    pad=3.0,
                )
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
                colorbar.ax.tick_params(
                    axis="y",
                    labelsize=5.7,
                    length=2,
                    pad=1.0,
                    labelleft=True,
                    labelright=False,
                )

        ax = axes[row, 2]
        effect_image = ax.imshow(
            contrast, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-1.5, vmax=1.5, extent=[0, 3, 0, 60],
            interpolation="nearest",
        )
        ax.set_xlabel("Time from onset (s)")
        ax.set_yticklabels([])
        ax.tick_params(length=2, width=0.6)
        colorbar = fig.colorbar(effect_image, cax=effect_color_axes[row])
        colorbar.ax.set_title(
            "Cohen's\n$d$",
            fontsize=5.8,
            pad=3.0,
        )
        colorbar.ax.tick_params(labelsize=6, length=2, pad=1.5)

        ax = axes[row, 3]
        ax.fill_between(
            frequencies, freq_lower, freq_upper,
            color=CLASS_COLORS[1], alpha=0.18, linewidth=0,
            label="Bootstrap 95% CI",
        )
        ax.plot(frequencies, freq_effect, color=CLASS_COLORS[1], linewidth=1.5)
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.axhline(0.5, color="#999999", linewidth=0.6, linestyle=":")
        ax.axhline(-0.5, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, 60)
        ax.set_ylim(-1.7, 1.7)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(color="#EEEEEE", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.6)

        if row == 0:
            for col, title in enumerate(titles):
                axes[row, col].set_title(title, fontweight="bold")

    fig.suptitle(
        f"{region} cohort, held-out test mouse {mouse}: "
        "class-conditional h0 target structure",
        fontsize=10.5, fontweight="bold", y=0.995,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{region}_{mouse}_generation_quality"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    np.savez_compressed(
        stem.with_name(stem.name + "_prototype_data.npz"),
        labels=labels,
        **saved,
    )
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def _draw_qualitative(region_data: dict, generated: dict, out_dir: Path) -> None:
    region = region_data["region"]
    mouse = region_data["mouse"]
    selected = generated["selected"]
    labels = region_data["labels"][selected]
    class_names = {0: "Non-ictal", 1: "Ictal"}

    fig = plt.figure(figsize=(7.15, 4.05))
    grid = fig.add_gridspec(
        2, 4, width_ratios=[1.0, 1.0, 1.0, 1.25],
        left=0.075, right=0.985, bottom=0.12, top=0.83,
        wspace=0.36, hspace=0.46,
    )
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(grid[row, col])

    heatmap_titles = [
        "Preictal EEG condition",
        "Real post-onset target",
        "Generated target sample",
    ]
    for row, (idx, label) in enumerate(zip(selected, labels)):
        past = region_data["dataset"].past_specs[idx, 0]
        past = past * region_data["spec_std"] + region_data["spec_mean"]
        real = region_data["dataset"].target_specs[idx]
        real = real * region_data["spec_std"] + region_data["spec_mean"]
        exemplar = generated["exemplars"][row]
        images = [past, real, exemplar]
        extents = [(-3, 0, 0, 60), (0, 3, 0, 60), (0, 3, 0, 60)]

        for col, (image, extent) in enumerate(zip(images, extents)):
            ax = axes[row, col]
            ax.imshow(
                _classifier_view(image),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                extent=extent,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(heatmap_titles[col], fontweight="bold")
            ax.set_xlabel("Time from onset (s)")
            if col == 0:
                ax.set_ylabel(f"{class_names[int(label)]}\nFrequency (Hz)", fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, width=0.6)

        axes[row, 0].text(
            0.03, 0.96,
            f"event {idx}\nmultimodal direct $p$={region_data['direct_probs'][idx]:.3f}",
            transform=axes[row, 0].transAxes,
            ha="left", va="top", color="white", fontsize=6.4,
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )
        axes[row, 1].text(
            0.03, 0.96,
            f"target classifier $p$={region_data['real_probs'][idx]:.3f}",
            transform=axes[row, 1].transAxes,
            ha="left", va="top", color="white", fontsize=6.4,
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )
        axes[row, 2].text(
            0.03, 0.96,
            f"checkpoint {generated['exemplar_seeds'][row]}\n"
            f"sample $p$={generated['exemplar_probs'][row]:.3f}",
            transform=axes[row, 2].transAxes,
            ha="left", va="top", color="white", fontsize=6.4,
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )

        ax = axes[row, 3]
        probs = generated["generated_probs"][row]
        checkpoint_seeds = generated["checkpoint_seeds"][row]
        rng = np.random.default_rng(991 + row)
        for checkpoint_seed in range(5):
            values = probs[checkpoint_seeds == checkpoint_seed]
            x = checkpoint_seed + 1 + rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                x, values, s=16, color=CLASS_COLORS[int(label)],
                alpha=0.72, edgecolor="white", linewidth=0.3, zorder=3,
            )
            ax.plot(
                [checkpoint_seed + 0.83, checkpoint_seed + 1.17],
                [np.median(values)] * 2,
                color="black", linewidth=1.1, zorder=4,
            )
        ensemble_prob = region_data["ensemble_probs"][idx]
        ax.axhline(
            ensemble_prob, color="#7B3294", linewidth=1.3,
            label=f"saved ensemble={ensemble_prob:.3f}",
        )
        ax.axhline(0.5, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
        ax.scatter(
            [6], [region_data["real_probs"][idx]], marker="D", s=25,
            color=DOMAIN_COLORS["Real target"], edgecolor="black", linewidth=0.4,
            zorder=5,
        )
        ax.scatter(
            [7], [region_data["direct_probs"][idx]], marker="s", s=25,
            color=DOMAIN_COLORS["Direct"], edgecolor="black", linewidth=0.4,
            zorder=5,
        )
        if row == 0:
            ax.set_title("Per-sample classifier scores", fontweight="bold")
        ax.set_xlim(0.55, 7.45)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(r"$p_{\mathrm{ictal}}$")
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels(["S0", "S1", "S2", "S3", "S4", "Real", "Direct"], rotation=32)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02, 0.04, f"ensemble $p$={ensemble_prob:.3f}",
            transform=ax.transAxes, fontsize=6.5, color="#7B3294",
        )

    fig.suptitle(
        f"{region} cohort, held-out test mouse {mouse}: "
        "deterministically selected h0 examples",
        fontsize=10.5, fontweight="bold", y=0.96,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{region}_{mouse}_generation_quality"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    np.savez_compressed(
        stem.with_name(stem.name + "_data.npz"),
        selected=selected,
        labels=labels,
        direct_probs=region_data["direct_probs"][selected],
        real_probs=region_data["real_probs"][selected],
        ensemble_probs=region_data["ensemble_probs"][selected],
        generated_specs=generated["generated_specs"],
        generated_probs=generated["generated_probs"],
        checkpoint_seeds=generated["checkpoint_seeds"],
        exemplar_probs=generated["exemplar_probs"],
        exemplar_seeds=generated["exemplar_seeds"],
    )
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


@torch.inference_mode()
def _alignment_data(
    region_data: dict,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    out_dir: Path,
    refresh: bool,
    device: torch.device,
) -> dict:
    region = region_data["region"]
    mouse = region_data["mouse"]
    cache_path = out_dir / f"{region}_{mouse}_target_space_alignment_data.npz"
    if cache_path.is_file() and not refresh:
        with np.load(cache_path) as payload:
            return {key: payload[key] for key in payload.files}

    dataset = region_data["dataset"]
    classifier = region_data["classifier"]
    labels = region_data["labels"]
    generated_specs, generated_features = [], []
    generated_probs, generated_labels, generated_seeds = [], [], []
    for checkpoint_seed in range(5):
        checkpoint_path = _checkpoint_path(region_data["case"], checkpoint_seed)
        diffusion, checkpoint = _load_diffusion(checkpoint_path, device)
        print(f"  alignment {region}: checkpoint {checkpoint_seed}")
        torch.manual_seed(seed + 100 + checkpoint_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100 + checkpoint_seed)

        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            past = torch.from_numpy(dataset.past_specs[start:end]).float().to(device)
            generated = diffusion.sample_target_ddim(
                past,
                target_shape=(dataset.freq_bins, dataset.target_frames),
                steps=steps,
                guidance_scale=guidance_scale,
                progress=False,
            )
            generated_logmag = generated * region_data["spec_std"] + region_data["spec_mean"]
            classifier_input = spec_to_classifier_input_torch(generated_logmag)
            feat = classifier.encode(classifier_input)
            logits = classifier.classifier(feat)
            generated_specs.append(generated_logmag.cpu().numpy())
            generated_features.append(feat.cpu().numpy())
            generated_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            generated_labels.append(labels[start:end])
            generated_seeds.append(np.full(end - start, checkpoint_seed, dtype=np.int64))

        del diffusion, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "labels": labels,
        "real_features": region_data["real_features"].astype(np.float32),
        "real_probs": region_data["real_probs"].astype(np.float32),
        "direct_probs": region_data["direct_probs"].astype(np.float32),
        "ensemble_probs": region_data["ensemble_probs"].astype(np.float32),
        "real_specs": (
            region_data["dataset"].target_specs * region_data["spec_std"]
            + region_data["spec_mean"]
        ).astype(np.float32),
        "generated_specs": np.concatenate(generated_specs).astype(np.float32),
        "generated_features": np.concatenate(generated_features).astype(np.float32),
        "generated_probs": np.concatenate(generated_probs).astype(np.float32),
        "generated_labels": np.concatenate(generated_labels).astype(np.int64),
        "generated_seeds": np.concatenate(generated_seeds).astype(np.int64),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    print(f"wrote {cache_path}")
    return result


def _confidence_ellipse(
    points: np.ndarray,
    ax: plt.Axes,
    color: str,
    linestyle: str,
    level: float = 1.665,
) -> None:
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * level * np.sqrt(np.maximum(values, 1e-12))
    center = points.mean(axis=0)
    ax.add_patch(Ellipse(
        center, width, height, angle=angle, fill=False,
        edgecolor=color, linewidth=1.15, linestyle=linestyle, alpha=0.9,
    ))


def _draw_alignment(all_data: dict[str, dict], out_dir: Path) -> None:
    regions = [region for region in ("RT", "VPM") if region in all_data]
    fig, axes = plt.subplots(
        2, len(regions), figsize=(7.15, 5.25), squeeze=False,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    rng = np.random.default_rng(2026)

    for col, region in enumerate(regions):
        data = all_data[region]
        labels = data["labels"]
        generated_labels = data["generated_labels"]
        real_features = data["real_features"]
        generated_features = data["generated_features"]
        joint = np.concatenate([real_features, generated_features], axis=0)
        joint = StandardScaler().fit_transform(joint)
        projection = PCA(n_components=2, random_state=0).fit_transform(joint)
        real_xy = projection[:len(real_features)]
        generated_xy = projection[len(real_features):]

        ax = axes[0, col]
        for label in (0, 1):
            color = CLASS_COLORS[label]
            real_points = real_xy[labels == label]
            generated_points = generated_xy[generated_labels == label]
            ax.scatter(
                generated_points[:, 0], generated_points[:, 1],
                s=7, marker="x", linewidth=0.45, color=color, alpha=0.23,
            )
            ax.scatter(
                real_points[:, 0], real_points[:, 1],
                s=9, marker="o", facecolor=color, edgecolor="white",
                linewidth=0.25, alpha=0.62,
            )
            _confidence_ellipse(real_points, ax, color, "-")
            _confidence_ellipse(generated_points, ax, color, "--")
        ax.set_title(f"{region}: frozen target-classifier features", fontweight="bold")
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#EEEEEE", linewidth=0.45)

        ax = axes[1, col]
        positions, box_data, colors = [], [], []
        methods = [
            ("Direct", data["direct_probs"]),
            ("Real target", data["real_probs"]),
            ("Diffusion", data["ensemble_probs"]),
        ]
        for method_idx, (method, probabilities) in enumerate(methods):
            for label in (0, 1):
                positions.append(method_idx * 3 + label + 1)
                box_data.append(probabilities[labels == label])
                colors.append(CLASS_COLORS[label])
        boxplot = ax.boxplot(
            box_data, positions=positions, widths=0.72, patch_artist=True,
            showfliers=False, medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
            boxprops={"linewidth": 0.7},
        )
        for patch, color, values, position in zip(
            boxplot["boxes"], colors, box_data, positions
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
            sample_count = min(70, len(values))
            chosen = rng.choice(len(values), sample_count, replace=False)
            x = position + rng.uniform(-0.18, 0.18, size=sample_count)
            ax.scatter(x, values[chosen], s=5, color=color, alpha=0.35, linewidth=0)
        ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.035, 1.035)
        ax.set_ylabel(r"$p_{\mathrm{ictal}}$")
        ax.set_xticks(positions)
        ax.set_xticklabels(["N", "I", "N", "I", "N", "I"], fontsize=7)
        for center, method in zip((1.5, 4.5, 7.5), ("Direct", "Real target", "Diffusion")):
            ax.text(
                center, -0.15, method, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7,
            )
        ax.set_title("Cohort-wide score distributions", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[0],
               markeredgecolor="none", label="Non-ictal", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[1],
               markeredgecolor="none", label="Ictal", markersize=5),
        Line2D([0], [0], marker="o", color="#444444", linestyle="none",
               label="Real target", markersize=4),
        Line2D([0], [0], marker="x", color="#444444", linestyle="none",
               label="Generated sample", markersize=4),
        Line2D([0], [0], color="#444444", linestyle="-", label="Real 75% ellipse"),
        Line2D([0], [0], color="#444444", linestyle="--", label="Generated 75% ellipse"),
    ]
    fig.legend(
        handles=legend, loc="upper center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0.995), fontsize=7,
    )
    fig.suptitle(
        "Target-space feature comparison and classifier-score separation at h0",
        fontsize=10.5, fontweight="bold", y=1.035,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.0, h_pad=1.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "RT_VPM_target_space_diagnostic"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    regions = [item.strip().upper() for item in args.regions.split(",") if item.strip()]
    for region in regions:
        if region not in CASES:
            raise SystemExit(f"Unknown region {region!r}; choose from {sorted(CASES)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    alignment = {}
    for region in regions:
        print(f"[generation diagnostics] {region}")
        region_data = _load_region(region, device)
        alignment[region] = _alignment_data(
            region_data,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            out_dir=args.out_dir,
            refresh=args.refresh_alignment,
            device=device,
        )
        _draw_class_prototypes(
            region, region_data["mouse"], alignment[region], args.out_dir
        )
        del region_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if alignment and not args.skip_alignment:
        _draw_alignment(alignment, args.out_dir)


if __name__ == "__main__":
    main()
