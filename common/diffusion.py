"""diffusion.py

Unconditional infill diffusion on 2D spectrograms.

Differences from the legacy (eeg_diffusion_STFT_conditional.py) implementation:
- No class conditioning: the UNet has no class embedding, no null token, no
  CFG branches. This models p(x_target | x_cond) by infilling the target
  region while the conditioning region is kept fixed.
- Auxiliary classifier loss: during training, the predicted clean sample
  x_hat_0 is reconstructed from the noise prediction. The target region of
  x_hat_0 is extracted, denormalized to log1p-magnitude space, reshaped for
  a frozen ResNet18 classifier, and classified against the true label with
  cross-entropy. This supplies a discriminative training signal WITHOUT the
  label being fed in at inference time.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# Beta schedule
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008, device=None) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)


# =============================================================================
# UNet components (EEG-only, no class conditioning)
# =============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = F.gelu(emb)
        emb = self.linear2(emb)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_emb = nn.Linear(time_dim, out_ch * 2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        t_out = self.time_emb(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        return h + self.shortcut(x)


class UNet2D(nn.Module):
    """2D UNet for spectrogram denoising, no class conditioning."""

    def __init__(self, in_ch: int = 1, base_ch: int = 64, depth: int = 4, time_dim: int = 256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.downs.append(ConvBlock(prev, c, time_dim))
            prev = c

        self.mid = ConvBlock(prev, prev, time_dim)
        self.attention = nn.MultiheadAttention(prev, num_heads=8, batch_first=True)

        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, 2))
            self.ups.append(ConvBlock(c * 2, c, time_dim))
            prev = c

        self.final = nn.Sequential(
            nn.GroupNorm(8, prev),
            nn.GELU(),
            nn.Conv2d(prev, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        x = self.init_conv(x)
        skips = []
        for down in self.downs:
            x = down(x, t_emb)
            skips.append(x)
            x = F.avg_pool2d(x, 2)

        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_att, _ = self.attention(x_flat, x_flat, x_flat)
        x = x + x_att.transpose(1, 2).view(B, C, H, W)

        x = self.mid(x, t_emb)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips.pop()
            h, w = skip.shape[2:]
            dh, dw = h - x.shape[2], w - x.shape[3]
            if dh or dw:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x, t_emb)

        return self.final(x)


# =============================================================================
# Diffusion wrapper
# =============================================================================

class UnconditionalInfillDiffusion:
    """DDPM training + DDIM infill sampling with optional aux classifier loss."""

    def __init__(
        self,
        model: UNet2D,
        num_timesteps: int = 1000,
        ema_decay: float = 0.995,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_timesteps = num_timesteps

        self.betas = cosine_beta_schedule(num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    # -- EMA ---------------------------------------------------------------
    def update_ema(self):
        with torch.no_grad():
            for ep, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ep.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # -- Forward process --------------------------------------------------
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.alphas_cumprod[t][:, None, None, None]
        x_noisy = torch.sqrt(a) * x_start + torch.sqrt(1.0 - a) * noise
        return x_noisy, noise

    # -- Training loss ----------------------------------------------------
    def p_losses(
        self,
        x_full: torch.Tensor,                   # [B, F, T]  (z-scored log1p specs)
        mask: torch.Tensor,                     # [B, F, T]  True=cond, False=target
        t: torch.Tensor,                        # [B]
        labels: Optional[torch.Tensor] = None,  # [B]
        classifier: Optional[nn.Module] = None,
        aux_ce_weight: float = 0.0,
        aux_fm_weight: float = 0.0,
        eps_weight: float = 1.0,
        eps_cond_weight: float = 1.0,
        eps_target_weight: float = 1.0,
        t_aux_min: int = 300,
        t_aux_max: int = 800,
        spec_mean: Optional[float] = None,
        spec_std: Optional[float] = None,
        freq_weighted: bool = True,
    ) -> dict:
        """Compute total loss and return components in a dict.

        The epsilon-MSE can be globally downweighted (``eps_weight``) and
        region-weighted (``eps_cond_weight`` vs. ``eps_target_weight``). This
        lets aux-heavy ablations emphasize target separability without fully
        discarding the denoising objective.

        Two complementary auxiliary losses on the predicted clean sample
        ``x_hat_0`` at the target region (both label-free at inference time):

        - aux_ce: cross-entropy of the frozen classifier's class probabilities
          on x_hat_0 vs. the true label. Provides class-discriminative pressure
          but is sparse (most timesteps make this tiny).
        - aux_fm: L2 between the classifier's penultimate (encoder) features
          on the generated target and on the matched real target. Dense per
          sample, does not use ``labels`` directly, but inherits class-aware
          structure from how the classifier was trained.

        Both are restricted to ``t_aux_min <= t <= t_aux_max`` (default
        300..800). At very low t, ``x_hat_0 ~= x_start`` algebraically and the
        gradient through the UNet vanishes; at very high t, ``x_hat_0`` is too
        noisy to be informative. The mid-t window is where the UNet's
        prediction actually drives ``x_hat_0`` and the aux gradient is useful.
        """
        B, F_dim, T = x_full.shape
        x_start = x_full.unsqueeze(1)
        x_noisy, noise = self.q_sample(x_start, t)

        pred_noise = self.model(x_noisy, t)

        # Main eps-prediction MSE
        mse = F.mse_loss(pred_noise, noise, reduction="none")
        if freq_weighted:
            w = torch.linspace(2.0, 0.5, F_dim, device=x_full.device).view(1, 1, -1, 1)
            mse = mse * w
        if eps_cond_weight != 1.0 or eps_target_weight != 1.0:
            region_w = torch.where(
                mask.unsqueeze(1),
                torch.as_tensor(eps_cond_weight, device=x_full.device, dtype=mse.dtype),
                torch.as_tensor(eps_target_weight, device=x_full.device, dtype=mse.dtype),
            )
            mse = mse * region_w
        eps_mse = mse.mean()

        zero = torch.tensor(0.0, device=x_full.device)
        out = {"eps_mse": eps_mse, "aux_ce": zero.clone(), "aux_fm": zero.clone()}

        use_ce = (classifier is not None) and (aux_ce_weight > 0.0) and (labels is not None)
        use_fm = (classifier is not None) and (aux_fm_weight > 0.0)
        if (use_ce or use_fm):
            a = self.alphas_cumprod[t][:, None, None, None]
            x0_hat = (x_noisy - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a)
            x0_hat = torch.clamp(x0_hat, -3.0, 3.0)

            sel = (t >= t_aux_min) & (t <= t_aux_max)
            if sel.any():
                x0_sel = x0_hat[sel].squeeze(1)            # [Bs, F, T]
                mask_sel = mask[sel]                       # [Bs, F, T]
                # Same mask pattern across the batch -> read target idx once.
                target_frames = (~mask_sel[0]).any(dim=0)  # [T]
                tgt_idx = target_frames.nonzero(as_tuple=False).squeeze(-1)

                x0_gen = x0_sel[:, :, tgt_idx]             # [Bs, F, T_target]
                x0_real = x_start[sel].squeeze(1)[:, :, tgt_idx]

                if spec_mean is not None and spec_std is not None:
                    x0_gen = x0_gen * spec_std + spec_mean
                    x0_real = x0_real * spec_std + spec_mean

                from .spectrogram import spec_to_classifier_input_torch
                clf_gen = spec_to_classifier_input_torch(x0_gen)
                clf_real = spec_to_classifier_input_torch(x0_real)

                if use_ce:
                    logits = classifier(clf_gen)
                    out["aux_ce"] = F.cross_entropy(logits, labels[sel])

                if use_fm:
                    # Classifier is frozen, so encoder features for the real
                    # target need no grad; the generated branch keeps grad.
                    f_gen = classifier.encode(clf_gen)
                    with torch.no_grad():
                        f_real = classifier.encode(clf_real)
                    out["aux_fm"] = F.mse_loss(f_gen, f_real)

        out["total"] = eps_weight * eps_mse + aux_ce_weight * out["aux_ce"] + aux_fm_weight * out["aux_fm"]
        return out

    # -- DDIM infill sampling --------------------------------------------
    @torch.no_grad()
    def sample_infill_ddim(
        self,
        target: torch.Tensor,   # [B, F, T]  known full spectrogram (cond region used)
        mask: torch.Tensor,     # [B, F, T]  True = keep from target (cond)
        steps: int = 100,
        eta: float = 0.0,
        progress: bool = True,
    ) -> torch.Tensor:
        """Infill the masked-out (target) region while keeping cond fixed.

        Returns the final full spectrogram with the target region replaced by
        generated content. Fully unconditional: no label inputs, no CFG.
        """
        self.ema_model.eval()
        B, F_dim, T = target.shape
        target_4d = target.unsqueeze(1)
        mask_4d = mask.unsqueeze(1)

        x = torch.randn_like(target_4d)

        c = self.num_timesteps // steps
        seq = list(range(0, self.num_timesteps, c))
        seq_next = [-1] + list(seq[:-1])

        iterator = zip(reversed(seq), reversed(seq_next))
        if progress:
            iterator = tqdm(iterator, total=len(seq), desc="DDIM infill")

        for i, j in iterator:
            t = torch.full((B,), i, dtype=torch.long, device=x.device)

            pred_noise = self.ema_model(x, t)
            at = self.alphas_cumprod[i]
            at_next = self.alphas_cumprod[j] if j >= 0 else torch.tensor(1.0, device=x.device)

            x0_pred = (x - torch.sqrt(1 - at) * pred_noise) / torch.sqrt(at)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if j < 0:
                x = x0_pred
            else:
                c1 = torch.sqrt(at_next)
                c2 = torch.sqrt(1 - at_next)
                x = c1 * x0_pred + c2 * pred_noise

                # Renoise the known region to t=j and merge
                t_next = torch.full((B,), j, dtype=torch.long, device=x.device)
                target_noisy, _ = self.q_sample(target_4d, t_next)
                x = torch.where(mask_4d, target_noisy, x)

        x = torch.where(mask_4d, target_4d, x)
        return x.squeeze(1)
