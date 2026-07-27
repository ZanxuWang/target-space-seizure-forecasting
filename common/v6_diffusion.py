"""v6_diffusion.py

Cross-attention conditional diffusion for spectrogram forecasting.

Key design choices vs. v5 (common/conditional_diffusion.py):

1.  Past encoder produces a SEQUENCE of spatial tokens [B, N, D] instead of a
    pooled FiLM vector. This preserves time-frequency structure of the past
    window, which the v5 condition_gap diagnostic showed was being squashed.

2.  Each UNet block does cross-attention from spatial target features to past
    tokens. The conditioning is forced through every level of the denoiser.

3.  Optional auxiliary classification head on past tokens (training-only)
    pushes the past encoder to produce class-discriminative features. This
    head is unused at inference - it just shapes representations.

4.  Inference is strictly label-free. Class label is never an input. We rely
    on classifier-free guidance (condition dropout during training, scale
    amplification during sampling) to amplify the past signal.

5.  Optional extra past channels: emg spectrogram + photometry envelope, all
    stacked along the spectrogram channel dim of the past encoder. EEG only
    is the default; multimodal is a flag.

Inference signature: model(x_target_noisy, t, past_spec)
  - past_spec shape: [B, C_past, F, T_cond] (C_past in {1, 2, 3})
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .diffusion import cosine_beta_schedule


# =============================================================================
# Building blocks
# =============================================================================


class GradientReversal(torch.autograd.Function):
    """Gradient-reversal layer for domain-adversarial training (DANN).

    Forward: identity. Backward: multiplies grad by ``-lambda``.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


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
        return self.linear2(emb)


class PastTokenEncoder(nn.Module):
    """Encode past spectrogram(s) into a sequence of spatial tokens.

    Output: [B, N, token_dim]. The token grid covers the down-sampled past
    spectrogram, so each token corresponds to a (freq-band, time-segment)
    region of the past window.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        token_dim: int = 256,
        depth: int = 3,
        n_self_attn_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        layers = []
        c = base_ch
        layers.append(nn.Conv2d(in_ch, c, 3, padding=1))
        layers.append(nn.GroupNorm(8, c))
        layers.append(nn.GELU())
        for _ in range(depth):
            c_next = min(c * 2, token_dim)
            layers.append(nn.Conv2d(c, c_next, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(8, c_next), c_next))
            layers.append(nn.GELU())
            c = c_next
        layers.append(nn.Conv2d(c, token_dim, 1))
        self.net = nn.Sequential(*layers)
        self.token_dim = token_dim

        self.self_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(token_dim),
                "attn": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(token_dim),
                "mlp": nn.Sequential(
                    nn.Linear(token_dim, token_dim * 2),
                    nn.GELU(),
                    nn.Linear(token_dim * 2, token_dim),
                ),
            })
            for _ in range(n_self_attn_layers)
        ])

    def forward(self, past_spec: torch.Tensor) -> torch.Tensor:
        if past_spec.dim() == 3:
            past_spec = past_spec.unsqueeze(1)
        h = self.net(past_spec)
        B, D, H, W = h.shape
        tokens = h.view(B, D, H * W).transpose(1, 2)
        for blk in self.self_attn_blocks:
            x = blk["ln1"](tokens)
            a, _ = blk["attn"](x, x, x)
            tokens = tokens + a
            x = blk["ln2"](tokens)
            tokens = tokens + blk["mlp"](x)
        return tokens


class XAttnFiLMConvBlock(nn.Module):
    """Conv residual block with FiLM(time) and cross-attention to past tokens."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        token_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_emb = nn.Linear(time_dim, out_ch * 2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.xattn_norm = nn.GroupNorm(8, out_ch)
        self.q_proj = nn.Linear(out_ch, out_ch)
        self.kv_proj = nn.Linear(token_dim, out_ch * 2)
        self.xattn = nn.MultiheadAttention(out_ch, n_heads, batch_first=True)
        self.xattn_out = nn.Linear(out_ch, out_ch)
        # Gate starts >0 so cross-attention contributes from epoch 1.
        self.xattn_gate = nn.Parameter(torch.full((1,), 0.5))

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        past_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)

        film = self.time_emb(t_emb)[:, :, None, None]
        scale, shift = film.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = h + self.shortcut(x)

        if past_tokens is not None:
            B, C, H, W = h.shape
            q_in = self.xattn_norm(h).view(B, C, H * W).transpose(1, 2)
            q = self.q_proj(q_in)
            kv = self.kv_proj(past_tokens)
            k, v = kv.chunk(2, dim=-1)
            a, _ = self.xattn(q, k, v)
            a = self.xattn_out(a)
            # tanh keeps the gate in (-1, 1); init=0.5 -> ~0.46 from epoch 1.
            gate = torch.tanh(self.xattn_gate)
            h = h + gate * a.transpose(1, 2).view(B, C, H, W)
        return h


class XAttnUNet2D(nn.Module):
    """UNet denoiser for target spectrogram with past cross-attention."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        depth: int = 4,
        time_dim: int = 256,
        token_dim: int = 256,
        past_in_ch: int = 1,
        past_encoder_depth: int = 3,
        past_self_attn_layers: int = 2,
        n_mice: int = 4,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.depth = depth
        self.time_dim = time_dim
        self.token_dim = token_dim
        self.past_in_ch = past_in_ch
        self.n_mice = n_mice

        self.time_embed = TimeEmbedding(time_dim)
        self.past_encoder = PastTokenEncoder(
            in_ch=past_in_ch,
            base_ch=max(base_ch // 2, 16),
            token_dim=token_dim,
            depth=past_encoder_depth,
            n_self_attn_layers=past_self_attn_layers,
        )

        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.downs.append(XAttnFiLMConvBlock(prev, c, time_dim, token_dim))
            prev = c

        self.mid = XAttnFiLMConvBlock(prev, prev, time_dim, token_dim)
        self.mid_self_attn = nn.MultiheadAttention(prev, num_heads=8, batch_first=True)

        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, 2))
            self.ups.append(XAttnFiLMConvBlock(c * 2, c, time_dim, token_dim))
            prev = c

        self.final = nn.Sequential(
            nn.GroupNorm(8, prev),
            nn.GELU(),
            nn.Conv2d(prev, 1, 1),
        )

        self.past_logit_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2),
        )
        # Mouse-domain head: predicts which training mouse produced this past.
        # We attach it via gradient reversal so the past encoder is pushed
        # toward mouse-id-invariant representations (DANN-style).
        self.mouse_logit_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, max(2, n_mice)),
        )

    def encode_past(self, past_spec: torch.Tensor) -> torch.Tensor:
        return self.past_encoder(past_spec)

    def past_logits(self, past_tokens: torch.Tensor) -> torch.Tensor:
        pooled = past_tokens.mean(dim=1)
        return self.past_logit_head(pooled)

    def mouse_logits(self, past_tokens: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
        pooled = past_tokens.mean(dim=1)
        pooled = grad_reverse(pooled, lambda_grl)
        return self.mouse_logit_head(pooled)

    def forward(
        self,
        x_target: torch.Tensor,
        t: torch.Tensor,
        past_spec: Optional[torch.Tensor],
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if past_tokens is None and past_spec is not None:
            past_tokens = self.encode_past(past_spec)

        x = self.init_conv(x_target)
        skips = []
        for down in self.downs:
            x = down(x, t_emb, past_tokens)
            skips.append(x)
            x = F.avg_pool2d(x, 2)

        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_att, _ = self.mid_self_attn(x_flat, x_flat, x_flat)
        x = x + x_att.transpose(1, 2).view(B, C, H, W)
        x = self.mid(x, t_emb, past_tokens)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips.pop()
            h, w = skip.shape[2:]
            dh, dw = h - x.shape[2], w - x.shape[3]
            if dh or dw:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x, t_emb, past_tokens)

        return self.final(x)


# =============================================================================
# Diffusion wrapper
# =============================================================================


class XAttnTargetDiffusion:
    """DDPM training + DDIM target-only sampling with cross-attention past."""

    def __init__(
        self,
        model: XAttnUNet2D,
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

    def update_ema(self):
        with torch.no_grad():
            for ep, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ep.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.alphas_cumprod[t][:, None, None, None]
        x_noisy = torch.sqrt(a) * x_start + torch.sqrt(1.0 - a) * noise
        return x_noisy, noise

    @staticmethod
    def apply_condition_dropout(
        past_spec: torch.Tensor,
        cond_drop_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly null the past condition for CFG. Returns past_used, keep_mask."""
        B = past_spec.shape[0]
        keep = torch.ones(B, device=past_spec.device, dtype=torch.bool)
        if cond_drop_prob <= 0.0:
            return past_spec, keep
        drop = torch.rand(B, device=past_spec.device) < cond_drop_prob
        if not drop.any():
            return past_spec, keep
        keep = ~drop
        past_used = past_spec.clone()
        past_used[drop] = 0.0
        return past_used, keep

    def p_losses(
        self,
        past_spec: torch.Tensor,
        target_spec: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mouse_idx: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
        aux_ce_weight: float = 0.0,
        aux_fm_weight: float = 0.0,
        past_ce_weight: float = 0.0,
        mouse_adv_weight: float = 0.0,
        mouse_adv_lambda_grl: float = 1.0,
        eps_weight: float = 1.0,
        cond_drop_prob: float = 0.0,
        t_aux_min: int = 50,
        t_aux_max: int = 950,
        aux_max_samples: Optional[int] = None,
        spec_mean: Optional[float] = None,
        spec_std: Optional[float] = None,
        freq_weighted: bool = True,
    ) -> dict:
        """Return dict of loss components; 'total' is the weighted sum.

        past_spec : [B, C_past, F, T_cond] (or [B, F, T_cond] -> auto unsqueeze)
        target_spec : [B, F, T_target]
        """
        if past_spec.dim() == 3:
            past_spec = past_spec.unsqueeze(1)
        x_start = target_spec.unsqueeze(1)
        x_noisy, noise = self.q_sample(x_start, t)

        if self.model.training:
            past_used, cond_keep = self.apply_condition_dropout(past_spec, cond_drop_prob)
        else:
            past_used = past_spec
            cond_keep = torch.ones(past_spec.shape[0], device=past_spec.device, dtype=torch.bool)

        past_tokens = self.model.encode_past(past_used)
        pred_noise = self.model(x_noisy, t, past_spec=None, past_tokens=past_tokens)

        mse = F.mse_loss(pred_noise, noise, reduction="none")
        if freq_weighted:
            F_dim = target_spec.shape[1]
            w = torch.linspace(2.0, 0.5, F_dim, device=target_spec.device).view(1, 1, -1, 1)
            mse = mse * w
        eps_mse = mse.mean()

        zero = torch.tensor(0.0, device=target_spec.device)
        out = {
            "eps_mse": eps_mse,
            "aux_ce": zero.clone(),
            "aux_fm": zero.clone(),
            "past_ce": zero.clone(),
            "mouse_adv": zero.clone(),
        }

        use_past_ce = (
            past_ce_weight > 0.0 and labels is not None
        )
        if use_past_ce:
            past_logits = self.model.past_logits(past_tokens[cond_keep]) if cond_keep.any() else None
            if past_logits is not None and past_logits.shape[0] > 0:
                out["past_ce"] = F.cross_entropy(past_logits, labels[cond_keep])

        use_mouse_adv = (
            mouse_adv_weight > 0.0 and mouse_idx is not None
        )
        if use_mouse_adv and cond_keep.any():
            mouse_logits = self.model.mouse_logits(past_tokens[cond_keep], mouse_adv_lambda_grl)
            out["mouse_adv"] = F.cross_entropy(mouse_logits, mouse_idx[cond_keep])

        use_ce = (classifier is not None) and (aux_ce_weight > 0.0) and (labels is not None)
        use_fm = (classifier is not None) and (aux_fm_weight > 0.0)
        if use_ce or use_fm:
            a = self.alphas_cumprod[t][:, None, None, None]
            x0_hat = (x_noisy - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a)
            x0_hat = torch.clamp(x0_hat, -3.0, 3.0)

            sel = (t >= t_aux_min) & (t <= t_aux_max) & cond_keep
            if sel.any():
                sel_idx = sel.nonzero(as_tuple=False).squeeze(1)
                if aux_max_samples is not None and aux_max_samples > 0 and sel_idx.numel() > aux_max_samples:
                    perm = torch.randperm(sel_idx.numel(), device=sel_idx.device)[:aux_max_samples]
                    sel_idx = sel_idx[perm]

                x0_gen = x0_hat[sel_idx].squeeze(1)
                x0_real = x_start[sel_idx].squeeze(1)

                if spec_mean is not None and spec_std is not None:
                    x0_gen = x0_gen * spec_std + spec_mean
                    x0_real = x0_real * spec_std + spec_mean

                from .spectrogram import spec_to_classifier_input_torch
                clf_gen = spec_to_classifier_input_torch(x0_gen)
                clf_real = spec_to_classifier_input_torch(x0_real)

                if use_ce:
                    logits = classifier(clf_gen)
                    out["aux_ce"] = F.cross_entropy(logits, labels[sel_idx])

                if use_fm:
                    f_gen = classifier.encode(clf_gen)
                    with torch.no_grad():
                        f_real = classifier.encode(clf_real)
                    out["aux_fm"] = F.mse_loss(f_gen, f_real)

        out["total"] = (
            eps_weight * eps_mse
            + aux_ce_weight * out["aux_ce"]
            + aux_fm_weight * out["aux_fm"]
            + past_ce_weight * out["past_ce"]
            + mouse_adv_weight * out["mouse_adv"]
        )
        return out

    @torch.no_grad()
    def sample_target_ddim(
        self,
        past_spec: torch.Tensor,
        target_shape: tuple[int, int],
        steps: int = 100,
        guidance_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate target spectrogram from past condition only.

        past_spec : [B, C_past, F, T_cond] OR [B, F, T_cond] (auto-unsqueeze)
        """
        self.ema_model.eval()
        if past_spec.dim() == 3:
            past_spec = past_spec.unsqueeze(1)
        B = past_spec.shape[0]
        F_dim, T_target = target_shape
        x = torch.randn((B, 1, F_dim, T_target), device=past_spec.device)
        past_spec = past_spec.to(self.device)
        null_past = torch.zeros_like(past_spec)

        cond_tokens = self.ema_model.encode_past(past_spec)
        uncond_tokens = self.ema_model.encode_past(null_past) if guidance_scale != 1.0 else None

        step = max(1, self.num_timesteps // steps)
        seq = list(range(0, self.num_timesteps, step))
        seq_next = [-1] + list(seq[:-1])

        iterator = zip(reversed(seq), reversed(seq_next))
        if progress:
            iterator = tqdm(iterator, total=len(seq), desc="DDIM xattn")

        for i, j in iterator:
            t = torch.full((B,), i, dtype=torch.long, device=x.device)

            if guidance_scale == 1.0:
                pred_noise = self.ema_model(x, t, past_spec=None, past_tokens=cond_tokens)
            else:
                eps_cond = self.ema_model(x, t, past_spec=None, past_tokens=cond_tokens)
                eps_uncond = self.ema_model(x, t, past_spec=None, past_tokens=uncond_tokens)
                pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            at = self.alphas_cumprod[i]
            at_next = self.alphas_cumprod[j] if j >= 0 else torch.tensor(1.0, device=x.device)
            x0_pred = (x - torch.sqrt(1 - at) * pred_noise) / torch.sqrt(at)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if j < 0:
                x = x0_pred
            else:
                x = torch.sqrt(at_next) * x0_pred + torch.sqrt(1 - at_next) * pred_noise

        return x.squeeze(1)
