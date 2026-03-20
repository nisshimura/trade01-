"""
Mamba3-inspired Selective State Space Model (pure PyTorch, MPS/CPU compatible)

Based on: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
(Dao AI Lab, Goombalab, 2026)

The original uses Triton/TileLang/CuteDSL CUDA kernels.
This preserves the mathematical structure in pure PyTorch:
  - Selective SSM with input-dependent A, B, C, D
  - Multi-head architecture with RoPE on B/C
  - Gating (SiLU) + Trapezoidal integration + RMSNorm
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SelectiveSSMLayer(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 64, expand: int = 2, headdim: int = 32,
        dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4, rope_fraction: float = 0.5, dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.A_floor = A_floor

        self.d_inner = max(int(expand * d_model), headdim)
        self.d_inner = (self.d_inner // headdim) * headdim
        self.nheads = self.d_inner // headdim

        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = max(self.split_tensor_size // 2, 1)

        d_in_proj = 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        dt_init = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))

        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _selective_scan(self, x, dt, A, B, C, D, trap):
        orig_device = x.device
        batch, seqlen, nheads, headdim = x.shape

        xc, dtc, Ac, Bc, Cc, Dc, tc = (
            t.float().cpu() for t in (x, dt, A, B, C, D, trap)
        )
        A_bar = torch.exp(Ac * dtc)
        B_bar = dtc.unsqueeze(-1) * Bc

        h = torch.zeros(batch, nheads, headdim, B.shape[-1])
        outputs = torch.empty(batch, seqlen, nheads, headdim)

        for t in range(seqlen):
            xt = xc[:, t]
            abt = A_bar[:, t, :, None, None]
            bbt = B_bar[:, t]
            ct = Cc[:, t]
            trt = tc[:, t, :, None, None]

            xb = xt.unsqueeze(-1) * bbt.unsqueeze(-2)
            h_new = abt * h + xb
            h = (1 - trt) * h_new + trt * (h + h_new) * 0.5
            outputs[:, t] = (h * ct.unsqueeze(-2)).sum(-1) + Dc[None, :, None] * xt

        return outputs.to(orig_device).to(x.dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = u.shape
        proj = self.in_proj(u)

        z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angles = proj.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles], dim=-1)

        x = x.view(batch, seqlen, self.nheads, self.headdim)
        z = z.view(batch, seqlen, self.nheads, self.headdim)

        B = self.B_norm(B_raw).unsqueeze(2).expand(-1, -1, self.nheads, -1) + self.B_bias
        C = self.C_norm(C_raw).unsqueeze(2).expand(-1, -1, self.nheads, -1) + self.C_bias

        cos_a = torch.cos(angles).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        sin_a = torch.sin(angles).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        rd = min(self.num_rope_angles, B.shape[-1] // 2)
        if rd > 0:
            B = torch.cat([apply_rotary_emb(B[..., :rd*2], cos_a[..., :rd], sin_a[..., :rd]), B[..., rd*2:]], -1)
            C = torch.cat([apply_rotary_emb(C[..., :rd*2], cos_a[..., :rd], sin_a[..., :rd]), C[..., rd*2:]], -1)

        A = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_raw)

        y = self._selective_scan(x, dt, A, B, C, self.D, trap)
        y = y * F.silu(z)
        return self.out_proj(self.dropout(y.reshape(batch, seqlen, self.d_inner)))


class Mamba3Forecaster(nn.Module):
    def __init__(
        self, n_us: int = 11, n_jp: int = 17, d_model: int = 64, d_state: int = 32,
        n_layers: int = 2, expand: int = 2, headdim: int = 16, dropout: float = 0.1,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(n_us + n_jp, d_model), RMSNorm(d_model))
        self.layers = nn.ModuleList([
            SelectiveSSMLayer(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            RMSNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_jp),
        )

    def forward(self, us_ret: torch.Tensor, jp_ret: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(torch.cat([us_ret, jp_ret], dim=-1))
        for layer, norm in zip(self.layers, self.norm_layers):
            x = layer(norm(x)) + x
        return self.head(x[:, -1, :])
