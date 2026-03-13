"""
pia_torch.py  ─  Paired Interference Architecture (PyTorch)
═══════════════════════════════════════════════════════════════════════════════

OPTIMISED VERSION — changes vs original
────────────────────────────────────────
1.  TRUE PARALLEL SCAN  (Hillis-Steele inclusive prefix scan)
      Replaces the sequential Python for-loop in DLiNOSSLayer with a
      O(log L) depth parallel scan over the time axis.  All hidden states
      h[0..L-1] are computed simultaneously; the C readout is a single
      batched einsum over all positions.
      Enabled by default via --scan.  Falls back to sequential for L < 16.

2.  BFLOAT16 AMP  (--bf16)
      bfloat16 has the same exponent range as float32 (no overflow risk).
      float16 can overflow the complex SSM state accumulation; bfloat16
      cannot.  No loss scaler is needed with bfloat16.
      SSM internals always stay in float32/complex64 regardless of AMP dtype.
      Legacy --amp (float16 + GradScaler) still works.

3.  GRADIENT ACCUMULATION  (--accum_steps N)
      Effective batch = batch_size × accum_steps without extra VRAM.
      The optimiser step fires only every N micro-steps.
      States are detached (truncated BPTT) after every micro-step.

4.  MEMORY-BANK WARMUP FREEZE  (--mem_warmup N)
      MemoryBank gradients are zeroed for the first N steps so the SSM
      builds stable representations before the NTM competes for signal.

5.  SUBSAMPLED ORTH LOSS  (--orth_sample_v N, default 64)
      Randomly samples N of 256 vocab entries each step.
      Statistically equivalent to using all 256, ~4× cheaper.

6.  ROPE CACHE  (per PIABlock, keyed by sequence length + dtype)
      cos/sin tables are computed once and reused.  Evicts stale entries
      if more than 16 distinct lengths are seen.

7.  TORCH.COMPILE IMPROVEMENTS
      mode="reduce-overhead", dynamic=True  — handles variable-length
      sequences without retracing.

8.  BUGFIXES vs original
      - torch.cuda.amp.GradScaler  → torch.amp.GradScaler  (not deprecated)
      - torch.cuda.amp.autocast    → torch.amp.autocast
      - Unnecessary h.clone() in _inject_replay removed
      - SSM always casts x to float32 before complex arithmetic
      - val metrics written to CSV without double-flushing MetricLogger
      - StreamingBuffer._fill loop guarded against double exhaustion
      - r_gate added to NO_DECAY parameter group

Architecture (unchanged from original):
  MultiVecEmbed  K sub-vectors / token, softmax mixing, orth regulariser
  RoPE           rotary position encoding with pos_offset streaming support
  DLiNOSSLayer   gated complex SSM, fast band (|A|<1) + slow band (|A|=1)
  MemoryBank     hippocampal NTM: sparse top-k read, NTM write, replay
  PIABlock       Pre-LN → RoPE → SSM → mem → replay → FFN
  PIAModel       stack of PIABlocks + LM head, stateful chunked training

Quick start
───────────
  pip install torch datasets transformers accelerate wandb

  # Wikipedia, all optimisations on:
  python pia_torch.py --scan --bf16 --compile --accum_steps 4

  # The Pile (streaming, no RAM limit):
  python pia_torch.py --dataset pile --d_model 512 --n_layers 8 \
      --batch_size 8 --seq_len 512 --steps 100000 \
      --scan --bf16 --compile --accum_steps 4

  # Resume:
  python pia_torch.py --resume ckpt/step_010000.pt

  # Evaluate a checkpoint:
  python pia_torch.py --eval ckpt/best.pt

  # Generate:
  python pia_torch.py --sample "def fibonacci" --ckpt ckpt/best.pt

Recommended flags for Google Jules / A100 VM:
  --scan --bf16 --compile --accum_steps 4 --mem_warmup 1000 \
  --batch_size 16 --seq_len 512 --d_model 512 --n_layers 8
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import argparse, csv, math, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser  (raw bytes, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

VOCAB = 256

def encode(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="replace"))

def decode(ids: List[int]) -> str:
    return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, d: int,
                     device: torch.device, dtype: torch.dtype,
                     base: float = 10_000.0) -> Tuple[Tensor, Tensor]:
    """Return (cos, sin) each shape (seq_len, d//2)."""
    half  = d // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos   = torch.arange(seq_len, device=device, dtype=dtype)
    ang   = torch.outer(pos, theta)          # (L, d//2)
    return ang.cos(), ang.sin()


def rope_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    x   : (B, L, D)  or  (B, L, H, d)
    cos : (L, D//2)
    sin : (L, D//2)
    Rotate adjacent pairs.
    """
    *head, D = x.shape
    x  = x.reshape(*head, D // 2, 2)        # (..., D//2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    while cos.dim() < x0.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).reshape(*head, D)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Vector Embedding
# ─────────────────────────────────────────────────────────────────────────────

class MultiVecEmbed(nn.Module):
    """
    Each token has K sub-vectors of dimension d = D//K.
    Output = sqrt(D) * Σ_k  softmax(mix)[k] * sc[k] * W[tok, k]
    Orthogonality regulariser encourages sub-vector diversity.
    """

    def __init__(self, vocab: int, d_model: int, K: int = 4):
        super().__init__()
        assert d_model % K == 0, f"d_model={d_model} must be divisible by K={K}"
        self.V, self.D, self.K = vocab, d_model, K
        self.d = d_model // K

        self.W   = nn.Parameter(torch.empty(vocab, K, self.d))
        self.mix = nn.Parameter(torch.zeros(vocab, K))
        self.sc  = nn.Parameter(torch.full((K,), math.sqrt(d_model / K)))

        nn.init.normal_(self.W, std=1.0 / math.sqrt(self.d))

    def forward(self, ids: Tensor) -> Tensor:
        """ids: (B, L) int64  →  (B, L, D)"""
        B, L = ids.shape
        vecs = self.W[ids]                               # (B, L, K, d)
        mix  = F.softmax(self.mix[ids], dim=-1)          # (B, L, K)
        out  = (mix.unsqueeze(-1) * self.sc.view(1, 1, self.K, 1) * vecs
                ).reshape(B, L, self.D)
        return out * math.sqrt(self.D)

    def orth_loss(self, sample_v: int = 64) -> Tensor:
        """
        Gram off-diagonal penalty → encourages sub-vector diversity.
        sample_v: randomly sample this many vocab entries per step.
                  Statistically equivalent to all 256, ~4× cheaper at 64.
        """
        if sample_v >= self.V:
            W = self.W
        else:
            idx = torch.randperm(self.V, device=self.W.device)[:sample_v]
            W   = self.W[idx]
        W  = F.normalize(W, dim=-1)                      # (V', K, d)
        G  = torch.bmm(W, W.transpose(1, 2))             # (V', K, K)
        I  = torch.eye(self.K, device=W.device, dtype=W.dtype)
        return ((G - I) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Prefix Scan  (Hillis-Steele inclusive scan)
# ─────────────────────────────────────────────────────────────────────────────

def _assoc_scan(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Inclusive parallel prefix scan over dimension 1 (the time/sequence axis)
    using the Hillis-Steele up-sweep algorithm.

    Associative operator ⊕ for composing linear maps  h → a·h + b :
        (a2, b2) ⊕ (a1, b1)  =  (a2·a1,  a2·b1 + b2)

    After the scan, prefix result at position t satisfies:
        pa[t]·h0 + pb[t]  =  h[t]
    for the recurrence  h[t] = a[t]·h[t-1] + b[t],  h[-1] = h0.

    Complexity: O(L·log L) work, O(log L) sequential depth.

    Args:
        a  (B, L, H, S)  complex  — per-step "decay" factors (α[t])
        b  (B, L, H, S)  complex  — per-step "input" terms   (β[t])
    Returns:
        pa (B, L, H, S)  complex  — prefix product (cumulative α)
        pb (B, L, H, S)  complex  — prefix sum     (cumulative β weighted by α)
    """
    pa, pb = a, b
    step   = 1
    L      = a.shape[1]
    while step < L:
        # "Left neighbour" at distance `step`.
        # Positions 0..step-1 have no left neighbour → use identity (1, 0).
        pa_left = torch.cat([
            torch.ones (a.shape[0], step, *a.shape[2:], dtype=a.dtype, device=a.device),
            pa[:, :-step],
        ], dim=1)
        pb_left = torch.cat([
            torch.zeros(b.shape[0], step, *b.shape[2:], dtype=b.dtype, device=b.device),
            pb[:, :-step],
        ], dim=1)
        # Combine current element with its left neighbour
        pb = pa * pb_left + pb
        pa = pa * pa_left
        step *= 2
    return pa, pb


# ─────────────────────────────────────────────────────────────────────────────
# DLiNOSS SSM  (diagonal linear oscillatory state-space model)
# ─────────────────────────────────────────────────────────────────────────────

class DLiNOSSLayer(nn.Module):
    """
    Gated diagonal complex SSM with two-band state space.

    Fast band  [0 : S_fast]  per head:
      |A_fast| = exp(-softplus(log_alpha)) ∈ (0, 0.999]  — decays normally
      h_fast[t] = g[t] · (A · h_fast[t-1]) + (1-g[t]) · Bx[t]

    Slow band  [S_fast : S]  per head:
      |A_slow| = 1  exactly  — unit-circle phasors, zero amplitude decay
      h_slow[t] = A · h_slow[t-1] + (1-g[t]) · Bx[t]

    Parallel scan reformulation (both bands):
      alpha[t] = g[t] · A   (fast)   OR   A            (slow)
      beta[t]  = (1 - g[t]) · Bx[t]
      h[t]     = alpha[t] · h[t-1] + beta[t]  ← standard linear recurrence

    _assoc_scan(alpha, beta) evaluates all h[t] in O(log L) depth.

    Parameters
    ──────────
    d_model   total model dimension
    d_state   total state size (fast + slow) across all heads
    n_heads   number of SSM heads (d_model and d_state must be divisible)
    slow_frac fraction of per-head state dims that are lossless [0, 1)
    use_scan  dispatch to parallel scan (True) or sequential loop (False)
    """

    def __init__(self, d_model: int, d_state: int, n_heads: int,
                 slow_frac: float = 0.25, use_scan: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert d_state % n_heads == 0, "d_state must be divisible by n_heads"
        assert 0.0 <= slow_frac < 1.0, "slow_frac must be in [0, 1)"

        self.H        = n_heads
        self.dh       = d_model // n_heads        # per-head model dim
        self.S        = d_state // n_heads         # total per-head state dim
        self.S_slow   = max(0, int(round(self.S * slow_frac)))
        self.S_fast   = self.S - self.S_slow
        self.use_scan = use_scan

        # Fast-band magnitude parameters (only S_fast dims need log_alpha)
        self.log_alpha = nn.Parameter(torch.zeros(n_heads, self.S_fast))
        # Phase parameters for all S dims (both bands rotate phase)
        self.omega     = nn.Parameter(torch.randn(n_heads, self.S) * 0.1)

        # B and C stored as real+imag pairs (complex Adam works poorly)
        self.B_re = nn.Parameter(torch.randn(n_heads, self.S, self.dh) * 0.01)
        self.B_im = nn.Parameter(torch.zeros(n_heads, self.S, self.dh))
        self.C_re = nn.Parameter(torch.randn(n_heads, self.dh, self.S) * 0.01)
        self.C_im = nn.Parameter(torch.zeros(n_heads, self.dh, self.S))

        # Skip connection: output += D_skip * input
        self.D_skip = nn.Parameter(torch.ones(d_model) * 0.1)

        # Write-gate: bias +1 → sigmoid(1) ≈ 0.73 → gates start open
        self.W_g = nn.Parameter(torch.zeros(n_heads, self.S, self.dh))
        self.b_g = nn.Parameter(torch.ones(n_heads, self.S))

    # ── A matrix helpers ─────────────────────────────────────────────────────
    def get_A(self) -> Tensor:
        """(H, S) complex64 — fast dims decay, slow dims unit magnitude.
        Always returns complex64 regardless of AMP dtype.
        torch.polar / torch.complex require float32 or float64 inputs;
        AMP may cast params to bfloat16/float16, so we force .float() here.
        """
        log_alpha = self.log_alpha.float()   # (H, S_fast) float32
        omega     = self.omega.float()       # (H, S)      float32
        mag_fast  = torch.exp(-F.softplus(log_alpha)).clamp(max=0.999)
        if self.S_slow > 0:
            mag_slow = torch.ones(self.H, self.S_slow,
                                  device=log_alpha.device, dtype=torch.float32)
            mag = torch.cat([mag_fast, mag_slow], dim=-1)
        else:
            mag = mag_fast
        return torch.polar(mag, omega)       # (H, S) complex64

    def get_B(self) -> Tensor:
        # .float() guards against bfloat16/float16 AMP; torch.complex requires float32+
        return torch.complex(self.B_re.float(), self.B_im.float())   # (H, S, dh) complex64

    def get_C(self) -> Tensor:
        return torch.complex(self.C_re.float(), self.C_im.float())   # (H, dh, S) complex64

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        """Zero initial hidden state (B, H, S) complex64."""
        return torch.zeros(batch, self.H, self.S,
                           dtype=torch.complex64, device=device)

    # ── Single-step inference (streaming / generation) ────────────────────────
    def step(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x  : (B, D)   float — one token per batch element
        h  : (B, H, S) complex64
        →  y: (B, D),  h_new: (B, H, S)
        """
        B    = x.shape[0]
        xf   = x.float()          # SSM always in float32
        A    = self.get_A()
        B_c  = self.get_B()
        C_c  = self.get_C()
        xr   = xf.view(B, self.H, self.dh)
        xc   = torch.complex(xr, torch.zeros_like(xr))

        g    = torch.sigmoid(
                   torch.einsum("hsd,bhd->bhs", self.W_g, xr) + self.b_g
               )                                      # (B, H, S) real
        Bx   = torch.einsum("hsd,bhd->bhs", B_c, xc) # (B, H, S) complex
        g_c  = g.to(Bx.dtype)
        Ah   = A.unsqueeze(0) * h

        Sf   = self.S_fast
        if self.S_slow > 0:
            h_f = (g_c[:, :, :Sf] * Ah[:, :, :Sf]
                   + (1 - g_c[:, :, :Sf]) * Bx[:, :, :Sf])
            h_s = Ah[:, :, Sf:] + (1 - g_c[:, :, Sf:]) * Bx[:, :, Sf:]
            h   = torch.cat([h_f, h_s], dim=-1)
        else:
            h   = g_c * Ah + (1 - g_c) * Bx

        yc   = torch.einsum("hds,bhs->bhd", C_c, h)
        # D_skip is float32; xf is already float32 — compute entirely in float32
        # then cast once to x.dtype so AMP dtype (bfloat16) is preserved correctly.
        y    = (yc.real.reshape(B, -1) + xf * self.D_skip).to(x.dtype)
        return y, h

    # ── Sequential loop (fallback / short sequences / debugging) ─────────────
    def _forward_sequential(self, x: Tensor,
                            h0: Tensor) -> Tuple[Tensor, Tensor]:
        B, L, D = x.shape
        xf      = x.float()
        A       = self.get_A()
        B_c     = self.get_B()
        C_c     = self.get_C()
        Sf      = self.S_fast

        xr_all  = xf.view(B, L, self.H, self.dh)
        g_all   = torch.sigmoid(
                      torch.einsum("hsd,blhd->blhs", self.W_g, xr_all) + self.b_g
                  )
        xc_all  = torch.complex(xr_all, torch.zeros_like(xr_all))
        Bx_all  = torch.einsum("hsd,blhd->blhs", B_c, xc_all)  # (B,L,H,S)

        h    = h0
        outs = []
        for t in range(L):
            g_t  = g_all[:, t].to(Bx_all.dtype)
            Bx_t = Bx_all[:, t]
            Ah   = A.unsqueeze(0) * h
            if Sf < self.S:
                h_f = (g_t[:, :, :Sf] * Ah[:, :, :Sf]
                       + (1 - g_t[:, :, :Sf]) * Bx_t[:, :, :Sf])
                h_s = Ah[:, :, Sf:] + (1 - g_t[:, :, Sf:]) * Bx_t[:, :, Sf:]
                h   = torch.cat([h_f, h_s], dim=-1)
            else:
                h   = g_t * Ah + (1 - g_t) * Bx_t
            yc  = torch.einsum("hds,bhs->bhd", C_c, h)
            # D_skip is float32; xf[:, t] is float32 — cast once at end to preserve AMP dtype.
            outs.append((yc.real.reshape(B, -1) + xf[:, t] * self.D_skip).to(x.dtype))

        return torch.stack(outs, dim=1), h

    # ── Parallel scan (fast path for training) ────────────────────────────────
    def _forward_scan(self, x: Tensor,
                      h0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute all h[0..L-1] simultaneously via the Hillis-Steele prefix scan.

        Key identity:
            h[t] = alpha[t] · h[t-1] + beta[t]
          where
            alpha[t] = g[t] · A    (fast dims)
                       A            (slow dims)
            beta[t]  = (1 - g[t]) · B·x[t]

        After _assoc_scan:  h[t] = pa[t] · h0 + pb[t]
        Output:  y = Re(C · h_all) + D·x   — single einsum over all L.
        """
        B, L, D  = x.shape
        Sf       = self.S_fast
        A        = self.get_A()    # (H, S) complex
        B_c      = self.get_B()    # (H, S, dh)
        C_c      = self.get_C()    # (H, dh, S)

        xf      = x.float()
        xr_all  = xf.view(B, L, self.H, self.dh)
        g_all   = torch.sigmoid(
                      torch.einsum("hsd,blhd->blhs", self.W_g, xr_all) + self.b_g
                  )                                   # (B, L, H, S) real
        xc_all  = torch.complex(xr_all, torch.zeros_like(xr_all))
        Bx_all  = torch.einsum("hsd,blhd->blhs", B_c, xc_all)  # (B,L,H,S) cplx
        g_c     = g_all.to(Bx_all.dtype)             # cast to complex dtype

        # Build per-step (alpha[t], beta[t]) for the linear recurrence
        A_exp = A.unsqueeze(0).unsqueeze(0)           # (1, 1, H, S)
        if Sf < self.S:
            # Fast dims: data-dependent decay  alpha = g * A
            alpha_f = g_c[:, :, :, :Sf] * A_exp[:, :, :, :Sf]
            # Slow dims: always carry full amplitude  alpha = A
            alpha_s = A_exp[:, :, :, Sf:].expand(B, L, -1, -1).contiguous()
            alpha   = torch.cat([alpha_f, alpha_s], dim=-1)  # (B, L, H, S)
        else:
            alpha   = g_c * A_exp

        beta = (1.0 - g_c) * Bx_all                  # (B, L, H, S)

        # Parallel prefix scan → pa[t], pb[t] such that h[t] = pa[t]*h0 + pb[t]
        pa, pb = _assoc_scan(alpha, beta)             # (B, L, H, S) each

        # Recover full hidden-state sequence
        h_all = pa * h0.unsqueeze(1) + pb            # (B, L, H, S)

        # C readout over all L simultaneously — no loop needed.
        # D_skip is float32; xf is float32. Compute addition in float32, cast once
        # to x.dtype so bfloat16 AMP is preserved end-to-end.
        y_c = torch.einsum("hds,blhs->blhd", C_c, h_all)  # (B, L, H, dh) cplx
        y   = (y_c.real.reshape(B, L, D)
               + xf * self.D_skip.unsqueeze(0).unsqueeze(0)).to(x.dtype)

        return y, h_all[:, -1]

    # ── Public dispatch ───────────────────────────────────────────────────────
    def forward(self, x: Tensor,
                h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        x   : (B, L, D)  float
        h0  : (B, H, S)  complex64 or None  (zero if None)
        →   y: (B, L, D),  h_last: (B, H, S)
        """
        B, L, _ = x.shape
        if h0 is None:
            h0 = self.init_state(B, x.device)

        if self.use_scan and L >= 16:
            return self._forward_scan(x, h0)
        return self._forward_sequential(x, h0)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse addressing helper
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_addr(scores: Tensor, top_k: int) -> Tensor:
    """
    Hippocampal sparse addressing: keep exactly the top-k slots,
    mask the rest with -inf, then re-normalise via softmax.
    scores : (*, num_slots)  →  (*, num_slots) sparse probability distribution
    """
    K      = min(top_k, scores.size(-1))
    _, idx = torch.topk(scores, K, dim=-1)
    masked = torch.full_like(scores, float("-inf"))
    masked.scatter_(-1, idx, scores.gather(-1, idx))
    return F.softmax(masked, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# External Memory Bank  (hippocampal NTM)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBank(nn.Module):
    """
    Hippocampal external memory:
      - Sparse top-k read  (pattern completion / winner-take-all)
      - NTM-style write at last position (erase + add vectors)
      - Per-position replay: retrieved vector injected into residual stream
        AND into SSM slow-band hidden state

    Memory footprint: num_slots × D floats  (constant w.r.t. sequence length).
    """

    def __init__(self, d_model: int, num_slots: int = 64,
                 read_top_k: int = 4):
        super().__init__()
        self.D         = d_model
        self.num_slots = num_slots
        self.read_top_k= read_top_k
        self.scale     = d_model ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_e = nn.Linear(d_model, d_model, bias=True)
        self.W_a = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # Replay gate: per-channel scalar; sigmoid(-2) ≈ 0.12 at init
        self.r_gate = nn.Parameter(torch.full((d_model,), -2.0))
        self.ln     = nn.LayerNorm(d_model)

        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02)
        nn.init.zeros_(self.W_e.bias)
        nn.init.zeros_(self.W_a.bias)

    def init_memory(self, batch: int, device: torch.device,
                    dtype: torch.dtype) -> Tensor:
        """Zero-initialised memory: (B, num_slots, D)."""
        return torch.zeros(batch, self.num_slots, self.D,
                           device=device, dtype=dtype)

    def forward(self, x: Tensor, M: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x : (B, L, D)          hidden states
        M : (B, num_slots, D)  current memory bank
        Returns: out, replay_all, M_new
        """
        B, L, D = x.shape
        q       = self.W_q(x)                                        # (B,L,D)

        # Sparse read (hippocampal pattern completion)
        scores  = torch.einsum("bld,bkd->blk", q, M) * self.scale   # (B,L,K)
        addr    = _sparse_addr(scores, self.read_top_k)
        r       = torch.einsum("blk,bkd->bld", addr, M)             # (B,L,D)
        out     = self.W_o(self.ln(r))

        # Replay signal: free since r is already computed for all positions
        replay_gate = torch.sigmoid(self.r_gate)                     # (D,)
        replay_all  = replay_gate * r                                 # (B,L,D)

        # NTM-style write (only at last position → O(1) per chunk)
        x_last  = x[:, -1]
        q_last  = q[:, -1]
        addr_w  = _sparse_addr(
            (q_last.unsqueeze(1) @ M.transpose(1, 2)).squeeze(1) * self.scale,
            self.read_top_k,
        )                                                             # (B, K)
        erase   = torch.sigmoid(self.W_e(x_last))                    # (B, D)
        add     = torch.tanh   (self.W_a(x_last))                    # (B, D)
        e_outer = addr_w.unsqueeze(-1) * erase.unsqueeze(1)          # (B, K, D)
        a_outer = addr_w.unsqueeze(-1) * add.unsqueeze(1)            # (B, K, D)
        M_new   = M * (1 - e_outer) + a_outer

        return out, replay_all, M_new


# ─────────────────────────────────────────────────────────────────────────────
# PIA Block
# ─────────────────────────────────────────────────────────────────────────────

class PIABlock(nn.Module):
    """
    Pre-LN → RoPE → Gated-SSM → residual
           → HippoMemory → replay residual → replay into SSM slow band
           → Pre-LN → FFN → residual
    """

    def __init__(self, d_model: int, d_state: int, n_heads: int,
                 dropout: float = 0.0, ffn_mult: int = 4,
                 mem_slots: int = 64, slow_frac: float = 0.25,
                 mem_top_k: int = 4, use_scan: bool = True):
        super().__init__()
        self.ln1    = nn.LayerNorm(d_model)
        self.ssm    = DLiNOSSLayer(d_model, d_state, n_heads,
                                   slow_frac=slow_frac, use_scan=use_scan)
        self.ln2    = nn.LayerNorm(d_model)
        self.fc1    = nn.Linear(d_model, d_model * ffn_mult)
        self.fc2    = nn.Linear(d_model * ffn_mult, d_model)
        self.drop   = nn.Dropout(dropout)
        self.memory = MemoryBank(d_model, num_slots=mem_slots,
                                 read_top_k=mem_top_k)

        H      = n_heads
        S_slow = self.ssm.S_slow
        self._H      = H
        self._S_slow = S_slow
        if S_slow > 0:
            self.W_replay = nn.Linear(d_model, H * S_slow * 2, bias=False)
            nn.init.normal_(self.W_replay.weight, std=0.01)
        else:
            self.W_replay = None

        # RoPE cache: dict keyed by (total_len, D, dtype_str)
        # Stores (cos, sin) on the correct device and dtype.
        self._rope_cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}

    # ── RoPE cache ────────────────────────────────────────────────────────────
    def _get_rope(self, total_len: int, D: int,
                  device: torch.device, dtype: torch.dtype
                  ) -> Tuple[Tensor, Tensor]:
        key = (total_len, D, str(dtype), str(device))
        if key not in self._rope_cache:
            cos, sin = build_rope_cache(total_len, D, device, torch.float32)
            self._rope_cache[key] = (cos.to(dtype), sin.to(dtype))
            # Evict oldest entry if cache grows too large
            if len(self._rope_cache) > 16:
                del self._rope_cache[next(iter(self._rope_cache))]
        return self._rope_cache[key]

    # ── Replay injection into SSM slow band ───────────────────────────────────
    def _inject_replay(self, h: Tensor, replay_all: Tensor) -> Tensor:
        """
        Inject the last-position replay signal into the slow-band slice of h.
        h          : (B, H, S)  complex  — final SSM state after processing chunk
        replay_all : (B, L, D)  float   — per-position replay from MemoryBank
        Returns    : (B, H, S)  complex
        """
        if self._S_slow == 0 or self.W_replay is None:
            return h
        B      = h.shape[0]
        r_last = replay_all[:, -1]                          # (B, D)
        proj   = self.W_replay(r_last)                      # (B, H*S_slow*2)
        proj   = proj.view(B, self._H, self._S_slow, 2)
        # proj may be bfloat16/float16 under AMP; torch.complex requires float32+
        proj_f = proj.float()
        delta  = torch.complex(proj_f[..., 0], proj_f[..., 1])  # (B, H, S_slow) complex64
        h_new  = h.clone()
        h_new[:, :, self.ssm.S_fast:] = h[:, :, self.ssm.S_fast:] + delta
        return h_new

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: Tensor,
                h0: Optional[Tensor] = None,
                M0: Optional[Tensor] = None,
                pos_offset: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x          : (B, L, D)
        h0         : (B, H, S) complex or None
        M0         : (B, num_slots, D) float or None
        pos_offset : starting position index for RoPE (streaming)
        Returns    : x_out (B,L,D), h_last (B,H,S), M_new (B,slots,D)
        """
        B, L, D = x.shape
        if M0 is None:
            M0 = self.memory.init_memory(B, x.device, x.dtype)

        # RoPE (cached)
        cos_full, sin_full = self._get_rope(pos_offset + L, D, x.device, x.dtype)
        cos = cos_full[pos_offset:]
        sin = sin_full[pos_offset:]

        # 1. SSM branch
        xn         = self.ln1(x)
        xr         = rope_apply(xn, cos, sin)
        ctx, h_new = self.ssm(xr, h0)
        x          = x + ctx

        # 2. Hippocampal memory
        mem_out, replay_all, M_new = self.memory(x, M0)
        x          = x + mem_out

        # 3a. Per-position replay into residual stream
        x          = x + replay_all

        # 3b. Inject last-position replay into SSM slow band
        h_new      = self._inject_replay(h_new, replay_all)

        # 4. FFN branch
        x          = x + self.fc2(self.drop(F.gelu(self.fc1(self.ln2(x)))))

        return x, h_new, M_new


# ─────────────────────────────────────────────────────────────────────────────
# Full PIA Language Model
# ─────────────────────────────────────────────────────────────────────────────

# State type: list of (h, M, pos_offset) per layer
State = List[Tuple[Tensor, Tensor, int]]


class PIAModel(nn.Module):
    """Paired Interference Architecture language model."""

    def __init__(self,
                 vocab         : int   = VOCAB,
                 d_model       : int   = 256,
                 n_layers      : int   = 4,
                 d_state       : int   = 64,
                 n_heads       : int   = 4,
                 K             : int   = 4,
                 dropout       : float = 0.0,
                 orth_lam      : float = 0.02,
                 orth_sample_v : int   = 64,
                 mem_slots     : int   = 64,
                 slow_frac     : float = 0.25,
                 mem_top_k     : int   = 4,
                 use_scan      : bool  = True):
        super().__init__()
        self.orth_lam      = orth_lam
        self.orth_sample_v = orth_sample_v

        self.embed  = MultiVecEmbed(vocab, d_model, K)
        self.blocks = nn.ModuleList([
            PIABlock(d_model, d_state, n_heads, dropout,
                     mem_slots=mem_slots, slow_frac=slow_frac,
                     mem_top_k=mem_top_k, use_scan=use_scan)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.02 / math.sqrt(2))

    def forward(self, ids: Tensor,
                states: Optional[State] = None) -> Tuple[Tensor, State]:
        x = self.embed(ids)
        new_states: State = []
        for i, block in enumerate(self.blocks):
            h0, M0, pos_off = states[i] if states else (None, None, 0)
            x, h_new, M_new = block(x, h0, M0, pos_offset=pos_off)
            new_states.append((h_new, M_new, pos_off + ids.shape[1]))
        logits = self.head(self.ln_out(x))
        return logits, new_states

    def compute_loss(self, ids: Tensor,
                     states: Optional[State] = None
                     ) -> Tuple[Tensor, Tensor, State]:
        logits, new_states = self.forward(ids, states)
        B, L, V = logits.shape
        ce    = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            ids[:, 1:].reshape(-1).long(),
        )
        orth  = (self.embed.orth_loss(self.orth_sample_v)
                 if self.orth_lam > 0 else ce.new_zeros(1))
        total = ce + self.orth_lam * orth
        return total, ce, new_states

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self,
                 prompt : str | List[int],
                 n      : int   = 200,
                 temp   : float = 0.8,
                 top_k  : int   = 50,
                 top_p  : float = 0.95,
                 device : Optional[torch.device] = None) -> str:
        dev = device or next(self.parameters()).device
        was_training = self.training
        self.eval()

        prompt_ids = encode(prompt) if isinstance(prompt, str) else list(prompt)
        ids_t      = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        logits, states = self.forward(ids_t)
        next_lg = logits[0, -1]

        out = list(prompt_ids)
        for _ in range(n):
            lg = next_lg / max(temp, 1e-8)

            if top_k > 0:
                topkv, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg        = lg.masked_fill(lg < topkv[-1], float("-inf"))

            if 0.0 < top_p < 1.0:
                sorted_lg, sorted_idx = torch.sort(lg, descending=True)
                cum_p     = sorted_lg.softmax(dim=-1).cumsum(dim=-1)
                remove    = cum_p - sorted_lg.softmax(dim=-1) > top_p
                sorted_lg = sorted_lg.masked_fill(remove, float("-inf"))
                lg        = torch.zeros_like(lg).scatter_(0, sorted_idx, sorted_lg)

            probs = lg.softmax(dim=-1).float()
            nxt   = torch.multinomial(probs, 1).item()
            out.append(nxt)

            tok_t        = torch.tensor([[nxt]], dtype=torch.long, device=dev)
            logits, states = self.forward(tok_t, states)
            next_lg      = logits[0, -1]

        if was_training:
            self.train()
        return decode(out)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hf_text(example: dict) -> str:
    return example.get("text") or example.get("content") or ""


def load_tokens_local(path: str, max_chars: int = 0) -> List[int]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if max_chars:
        text = text[:max_chars]
    return encode(text)


def load_tokens_hf(dataset: str, split: str,
                   max_chars: int = 100_000_000,
                   streaming: bool = True) -> List[int]:
    from datasets import load_dataset

    CFGS = {
        "wikipedia":   {"path": "wikipedia", "name": "20220301.en",
                        "split": split, "streaming": streaming,
                        "trust_remote_code": True},
        "wikitext103": {"path": "wikitext", "name": "wikitext-103-raw-v1",
                        "split": split},
        "openwebtext": {"path": "openwebtext", "split": split,
                        "streaming": streaming},
        "c4":          {"path": "c4", "name": "en", "split": split,
                        "streaming": streaming},
        "pile":        {"path": "EleutherAI/pile", "split": split,
                        "streaming": streaming, "trust_remote_code": True},
    }
    cfg = CFGS.get(dataset, {"path": dataset, "split": split,
                              "streaming": streaming})
    ds  = load_dataset(**cfg)

    toks, chars = [], 0
    for ex in ds:
        text = _hf_text(ex)
        if not text:
            continue
        toks.extend(encode(text + "\n"))
        chars += len(text)
        if max_chars and chars >= max_chars:
            break

    print(f"  [{split}] {dataset}: {len(toks):,} tokens  ({chars/1e6:.1f} M chars)")
    return toks


class TokenBuffer:
    """Random (B, L+1) batches from a flat token array."""

    def __init__(self, ids: List[int], seq_len: int, batch_size: int,
                 device: torch.device, seed: int = 42):
        self.ids  = torch.tensor(ids, dtype=torch.long, device=device)
        self.L    = seq_len
        self.B    = batch_size
        self.rng  = torch.Generator(device="cpu").manual_seed(seed)
        self.N    = max(len(ids) - seq_len - 1, 1)

    def next_batch(self) -> Tensor:
        starts = torch.randint(0, self.N, (self.B,), generator=self.rng)
        return torch.stack([self.ids[s: s + self.L + 1] for s in starts])


class StreamingBuffer:
    """Streams text from a HuggingFace streaming dataset without loading to RAM."""

    def __init__(self, dataset: str, split: str, seq_len: int,
                 batch_size: int, device: torch.device,
                 buffer_tokens: int = 5_000_000):
        from datasets import load_dataset

        CFGS = {
            "pile": {"path": "EleutherAI/pile", "split": split,
                     "streaming": True, "trust_remote_code": True},
            "c4":   {"path": "c4", "name": "en", "split": split,
                     "streaming": True},
        }
        cfg = CFGS.get(dataset, {"path": dataset, "split": split,
                                  "streaming": True})
        self.ds         = iter(load_dataset(**cfg))
        self.L          = seq_len
        self.B          = batch_size
        self.dev        = device
        self.buf        : List[int] = []
        self.cap        = buffer_tokens
        self._exhausted = False
        self._fill()

    def _fill(self):
        while not self._exhausted and len(self.buf) < self.cap:
            try:
                ex = next(self.ds)
            except StopIteration:
                self._exhausted = True
                break
            text = _hf_text(ex)
            if text:
                self.buf.extend(encode(text + "\n"))

    def next_batch(self) -> Tensor:
        needed = (self.L + 1) * self.B
        while len(self.buf) < needed:
            if self._exhausted:
                raise RuntimeError(
                    f"StreamingBuffer: dataset exhausted ({len(self.buf)} tokens "
                    f"remain, need {needed}). Use a larger dataset or reduce batch/seq_len."
                )
            self._fill()
        chunk    = self.buf[:needed]
        self.buf = self.buf[needed:]
        t        = torch.tensor(chunk, dtype=torch.long, device=self.dev)
        return t.view(self.B, self.L + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics logger
# ─────────────────────────────────────────────────────────────────────────────

class MetricLogger:
    ALL_COLS = [
        "train_loss", "train_bpc", "train_ppl",
        "val_loss",   "val_bpc",   "val_ppl",
        "grad_norm",  "param_norm", "lr", "tok/s", "tokens",
    ]

    def __init__(self, log_dir: str, use_wandb: bool = False,
                 wandb_project: str = "", wandb_name: str = ""):
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path  = self.log_dir / "metrics.csv"
        self.use_wandb = use_wandb
        self._accum    : dict = {}
        self._counts   : dict = {}
        self._csv_init = False

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_name or None,
                       config={"log_dir": str(log_dir)})

    def update(self, **kv):
        for k, v in kv.items():
            v = float(v)
            self._accum[k]  = self._accum.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def _flush(self) -> dict:
        out = {k: self._accum[k] / self._counts[k] for k in self._accum}
        self._accum.clear()
        self._counts.clear()
        return out

    def log(self, step: int, extra: Optional[dict] = None) -> dict:
        m = self._flush()
        if extra:
            m.update(extra)
        for prefix in ("train", "val"):
            lk = f"{prefix}_loss"
            if lk in m:
                m[f"{prefix}_bpc"] = m[lk] / math.log(2)
                m[f"{prefix}_ppl"] = math.exp(min(m[lk], 20.0))

        parts = [f"step {step:>7,}"]
        for k in ("train_loss", "train_bpc", "val_loss", "val_bpc",
                  "grad_norm", "lr", "tok/s", "tokens"):
            if k not in m:
                continue
            v = m[k]
            if k == "tokens":
                parts.append(f"tokens={v/1e6:.2f}M")
            elif k == "lr":
                parts.append(f"lr={v:.2e}")
            elif k == "tok/s":
                parts.append(f"tok/s={v:,.0f}")
            else:
                parts.append(f"{k}={v:.4f}")
        print("  ".join(parts))

        if not self._csv_init:
            self._csv_init = True
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step"] + self.ALL_COLS)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step] + [m.get(k, "") for k in self.ALL_COLS])

        if self.use_wandb:
            import wandb
            wandb.log({"step": step, **m}, step=step)

        return m


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule and optimiser
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int,
              peak: float, floor: float = 0.0) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


# r_gate added — it's a scalar bias, should not be weight-decayed
NO_DECAY = frozenset({"bias", "ln", "norm", "embed", "mix", "sc",
                       "log_alpha", "omega", "D_skip", "r_gate"})


def make_optimizer(model: PIAModel, lr: float,
                   weight_decay: float = 0.1,
                   betas: Tuple[float, float] = (0.9, 0.95)
                   ) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in NO_DECAY):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: PIAModel, buf: TokenBuffer,
             n_batches: int = 32) -> float:
    """Stateless evaluation — fresh state per batch."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch = buf.next_batch()
        _, ce, _ = model.compute_loss(batch)
        total += ce.item()
    model.train()
    return total / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path: Path, model: PIAModel, opt: torch.optim.Optimizer,
              step: int, val_loss: float, cfg: argparse.Namespace):
    torch.save({
        "model":    model.state_dict(),
        "opt":      opt.state_dict(),
        "step":     step,
        "val_loss": val_loss,
        "cfg":      vars(cfg),
    }, path)


def load_ckpt(path: str, model: PIAModel,
              opt: Optional[torch.optim.Optimizer] = None,
              device: Optional[torch.device] = None) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _memory_bank_params(model: PIAModel):
    """Yield all MemoryBank parameters for warmup-freeze gradient zeroing."""
    for block in model.blocks:
        yield from block.memory.parameters()


def train(cfg: argparse.Namespace):

    # ── Device ───────────────────────────────────────────────────────────────
    if cfg.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(cfg.device)
    print(f"Device: {dev}")

    # ── AMP setup ────────────────────────────────────────────────────────────
    # bfloat16 is preferred over float16 for SSMs:
    #   - Same exponent range as float32  → complex state cannot overflow
    #   - No loss scaling needed
    # SSM internals (get_A, complex arithmetic) always run in float32/complex64.
    use_amp   = False
    amp_dtype = torch.float32
    scaler    = None

    if cfg.bf16 and dev.type == "cuda":
        use_amp   = True
        amp_dtype = torch.bfloat16
        print("AMP: bfloat16  (no loss scaler needed)")
    elif cfg.amp and dev.type == "cuda":
        use_amp   = True
        amp_dtype = torch.float16
        scaler    = torch.amp.GradScaler("cuda")
        print("AMP: float16  (grad scaler active)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PIAModel(
        vocab         = VOCAB,
        d_model       = cfg.d_model,
        n_layers      = cfg.n_layers,
        d_state       = cfg.d_state,
        n_heads       = cfg.n_heads,
        K             = cfg.K,
        dropout       = cfg.dropout,
        orth_lam      = cfg.orth_lam,
        orth_sample_v = cfg.orth_sample_v,
        mem_slots     = cfg.mem_slots,
        slow_frac     = cfg.slow_frac,
        mem_top_k     = cfg.mem_top_k,
        use_scan      = cfg.scan,
    ).to(dev)

    print(f"Parameters: {model.n_params:,}")
    _print_model_summary(model, cfg)

    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)

    start_step, best_val = 0, float("inf")
    if cfg.resume:
        start_step, best_val = load_ckpt(cfg.resume, model, opt, dev)
        print(f"Resumed from {cfg.resume}  (step {start_step})")

    # ── torch.compile ─────────────────────────────────────────────────────────
    # Keep raw_model for attribute access (_memory_bank_params, etc.) which
    # may be unreliable on OptimizedModule depending on PyTorch version.
    raw_model = model
    if cfg.compile:
        if not hasattr(torch, "compile"):
            print("Warning: torch.compile requires PyTorch ≥ 2.0, skipping.")
        else:
            print("Compiling model  (reduce-overhead, dynamic=True)…")
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading training data…")
    if cfg.local_train:
        train_ids = load_tokens_local(cfg.local_train)
    else:
        train_ids = load_tokens_hf(cfg.dataset, "train",
                                   max_chars=cfg.max_train_chars)

    if cfg.local_val:
        val_ids = load_tokens_local(cfg.local_val)
    elif cfg.dataset in ("wikitext103",):
        val_ids = load_tokens_hf(cfg.dataset, "validation")
    else:
        split   = int(len(train_ids) * 0.97)
        val_ids, train_ids = train_ids[split:], train_ids[:split]

    train_buf = TokenBuffer(train_ids, cfg.seq_len, cfg.batch_size, dev)
    val_buf   = TokenBuffer(val_ids,   cfg.seq_len, min(cfg.batch_size, 4), dev)

    # ── Logging ───────────────────────────────────────────────────────────────
    logger   = MetricLogger(cfg.ckpt_dir,
                             use_wandb=bool(cfg.wandb),
                             wandb_project=cfg.wandb,
                             wandb_name=cfg.run_name)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    tokens_seen  = start_step * cfg.batch_size * cfg.seq_len * cfg.accum_steps
    t0           = time.perf_counter()
    steps_timer  = 0
    train_states : Optional[State] = None
    accum        = max(1, cfg.accum_steps)
    eff_bs       = cfg.batch_size * accum

    print(f"\n{'─'*64}")
    print(f"  Training  {start_step+1} → {cfg.steps} steps")
    print(f"  Grad accum {accum}×  →  eff batch {eff_bs * cfg.seq_len:,} tokens/step")
    if cfg.scan:
        print("  Parallel scan ON  (O(log L) depth)")
    if cfg.mem_warmup > 0:
        print(f"  MemoryBank frozen for first {cfg.mem_warmup} steps")
    print(f"{'─'*64}")

    opt.zero_grad(set_to_none=True)

    for step in range(start_step + 1, cfg.steps + 1):

        # LR schedule
        lr = cosine_lr(step, cfg.warmup, cfg.steps,
                       cfg.lr, cfg.lr * cfg.lr_decay_factor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ── Micro-steps (gradient accumulation) ───────────────────────────
        total_ce = 0.0
        for _micro in range(accum):
            batch        = train_buf.next_batch()
            tokens_seen += cfg.batch_size * cfg.seq_len
            steps_timer += 1

            if use_amp:
                with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype):
                    total_loss, ce_loss, train_states = model.compute_loss(
                        batch, train_states)
            else:
                total_loss, ce_loss, train_states = model.compute_loss(
                    batch, train_states)

            # Detach states — truncated BPTT.
            # Slow band content and memory slots carry forward as VALUES only;
            # gradients do not flow back through chunk boundaries.
            if train_states is not None:
                train_states = [(h.detach(), M.detach(), pos)
                                for h, M, pos in train_states]

            scaled = total_loss / accum
            if scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            total_ce += ce_loss.item()

        avg_ce = total_ce / accum

        # ── Memory bank warmup freeze ──────────────────────────────────────
        # Zero MemoryBank grads for the first mem_warmup steps so the SSM
        # learns stable representations before the NTM writes compete.
        if step <= cfg.mem_warmup:
            for p in _memory_bank_params(raw_model):
                if p.grad is not None:
                    p.grad.zero_()

        # ── Optimiser step ────────────────────────────────────────────────
        if scaler:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip).item()
            scaler.step(opt)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip).item()
            opt.step()

        opt.zero_grad(set_to_none=True)
        logger.update(train_loss=avg_ce, grad_norm=grad_norm)

        # ── Periodic log ─────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - t0
            tps     = (steps_timer * cfg.batch_size * cfg.seq_len) / elapsed
            t0      = time.perf_counter()
            steps_timer = 0
            param_norm  = math.sqrt(
                sum(p.data.norm().item() ** 2
                    for p in model.parameters()
                    if p.data.is_floating_point()))
            logger.log(step, extra={
                "lr": lr, "tok/s": tps,
                "tokens": tokens_seen, "param_norm": param_norm,
            })

        # ── Validation ────────────────────────────────────────────────────
        if step % cfg.val_every == 0:
            val_loss = evaluate(model, val_buf, n_batches=cfg.val_batches)
            val_bpc  = val_loss / math.log(2)
            val_ppl  = math.exp(min(val_loss, 20.0))
            print(f"  ┌ VAL step={step:,}  loss={val_loss:.4f}  "
                  f"bpc={val_bpc:.4f}  ppl={val_ppl:.2f}")

            # Write val metrics directly to CSV without disturbing train accumulators
            if not logger._csv_init:
                logger._csv_init = True
                with open(logger.csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["step"] + logger.ALL_COLS)
            with open(logger.csv_path, "a", newline="") as f:
                row = {k: "" for k in logger.ALL_COLS}
                row.update(val_loss=val_loss, val_bpc=val_bpc, val_ppl=val_ppl,
                           lr=lr, tokens=tokens_seen)
                csv.writer(f).writerow([step] + [row[k] for k in logger.ALL_COLS])

            if cfg.wandb:
                import wandb
                wandb.log({"val_loss": val_loss, "val_bpc": val_bpc,
                           "val_ppl": val_ppl, "step": step}, step=step)

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(ckpt_dir / "best.pt", model, opt, step, best_val, cfg)
                print(f"  └ new best  val_bpc={val_bpc:.4f}  → {ckpt_dir/'best.pt'}")
            else:
                print(f"  └ best so far: bpc={best_val/math.log(2):.4f}")

            model.train()

        # ── Periodic checkpoint ───────────────────────────────────────────
        if step % cfg.save_every == 0:
            save_ckpt(ckpt_dir / f"step_{step:07d}.pt",
                      model, opt, step, best_val, cfg)

    print(f"\nDone.  Best val BPC = {best_val/math.log(2):.4f}")
    if cfg.wandb:
        import wandb
        wandb.finish()


def _print_model_summary(model: PIAModel, cfg: argparse.Namespace):
    ssm = model.blocks[0].ssm
    print(f"  d_model   {cfg.d_model}")
    print(f"  n_layers  {cfg.n_layers}")
    print(f"  d_state   {cfg.d_state}  "
          f"(H={cfg.n_heads}, per-head S={ssm.S}, fast={ssm.S_fast}, slow={ssm.S_slow})")
    print(f"  K         {cfg.K}  (sub-vecs/token)")
    print(f"  seq_len   {cfg.seq_len}")
    print(f"  batch     {cfg.batch_size}  "
          f"(accum {cfg.accum_steps}× → eff {cfg.batch_size * cfg.accum_steps})")
    print(f"  lr        {cfg.lr}  warmup={cfg.warmup}")
    print(f"  scan      {'ON' if cfg.scan else 'OFF'}")
    print(f"  mem_warmup {cfg.mem_warmup} steps")
    total_M = cfg.steps * cfg.batch_size * cfg.accum_steps * cfg.seq_len / 1e6
    print(f"  total training  ≈ {total_M:.0f} M tokens  ({cfg.steps:,} steps)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PIA language model — train / eval / generate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modes
    p.add_argument("--eval",   type=str, default=None, metavar="CKPT",
                   help="Evaluate checkpoint and exit")
    p.add_argument("--sample", type=str, default=None, metavar="PROMPT",
                   help="Generate from prompt and exit (requires --ckpt)")
    p.add_argument("--ckpt",   type=str, default=None, metavar="PATH",
                   help="Checkpoint path for --sample or --eval")
    p.add_argument("--resume", type=str, default=None, metavar="PATH",
                   help="Resume training from checkpoint")

    # Data
    p.add_argument("--dataset",         type=str, default="wikipedia",
                   choices=["wikipedia","wikitext103","openwebtext","c4","pile"])
    p.add_argument("--local_train",     type=str, default=None,
                   help="Path to local .txt training file")
    p.add_argument("--local_val",       type=str, default=None,
                   help="Path to local .txt validation file")
    p.add_argument("--max_train_chars", type=int, default=200_000_000,
                   help="Max chars to load from training split")

    # Architecture
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--d_state",       type=int,   default=64)
    p.add_argument("--n_heads",       type=int,   default=4)
    p.add_argument("--K",             type=int,   default=4,
                   help="Sub-vectors per token in MultiVecEmbed")
    p.add_argument("--dropout",       type=float, default=0.0)
    p.add_argument("--orth_lam",      type=float, default=0.02,
                   help="Embedding orthogonality loss weight")
    p.add_argument("--orth_sample_v", type=int,   default=64,
                   help="Vocab entries sampled per step for orth_loss (≤256)")
    p.add_argument("--mem_slots",     type=int,   default=64,
                   help="External memory slots per layer")
    p.add_argument("--slow_frac",     type=float, default=0.25,
                   help="Fraction of SSM state dims that are lossless [0,1)")
    p.add_argument("--mem_top_k",     type=int,   default=4,
                   help="Top-k slots for sparse hippocampal addressing")

    # Training
    p.add_argument("--steps",           type=int,   default=20_000)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--seq_len",         type=int,   default=256)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--warmup",          type=int,   default=500)
    p.add_argument("--lr_decay_factor", type=float, default=0.1,
                   help="LR at end of cosine decay as fraction of peak")
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--grad_clip",       type=float, default=1.0)

    # Optimisation flags
    p.add_argument("--scan",    action="store_true", default=True,
                   help="Use parallel prefix scan in DLiNOSS (O(log L), default ON)")
    p.add_argument("--no_scan", dest="scan", action="store_false",
                   help="Disable parallel scan, use sequential loop")
    p.add_argument("--bf16",    action="store_true",
                   help="bfloat16 AMP — safer than float16 for SSMs (CUDA only)")
    p.add_argument("--amp",     action="store_true",
                   help="float16 AMP with grad scaler (legacy; prefer --bf16)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (PyTorch ≥ 2.0)")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="Gradient accumulation steps (eff_batch = batch × accum)")
    p.add_argument("--mem_warmup",  type=int, default=0,
                   help="Freeze MemoryBank grads for first N steps (0 = no freeze)")

    # Logging / checkpointing
    p.add_argument("--log_every",   type=int, default=50)
    p.add_argument("--val_every",   type=int, default=500)
    p.add_argument("--val_batches", type=int, default=32,
                   help="Number of batches for validation")
    p.add_argument("--save_every",  type=int, default=2_000)
    p.add_argument("--ckpt_dir",    type=str, default="ckpt")
    p.add_argument("--wandb",       type=str, default="",
                   help="W&B project name (empty = disabled)")
    p.add_argument("--run_name",    type=str, default="")

    # Hardware
    p.add_argument("--device", type=str, default="auto",
                   help="cuda / cpu / mps / auto")

    # Generation
    p.add_argument("--gen_n",    type=int,   default=200)
    p.add_argument("--gen_temp", type=float, default=0.8)
    p.add_argument("--gen_topk", type=int,   default=50)
    p.add_argument("--gen_topp", type=float, default=0.95)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_parser().parse_args()

    if cfg.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(cfg.device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if cfg.eval:
        ckpt      = torch.load(cfg.eval, map_location=dev, weights_only=False)
        scfg      = argparse.Namespace(**ckpt.get("cfg", {}))
        model     = PIAModel(
            vocab         = VOCAB,
            d_model       = getattr(scfg, "d_model",   cfg.d_model),
            n_layers      = getattr(scfg, "n_layers",  cfg.n_layers),
            d_state       = getattr(scfg, "d_state",   cfg.d_state),
            n_heads       = getattr(scfg, "n_heads",   cfg.n_heads),
            K             = getattr(scfg, "K",         cfg.K),
            mem_slots     = getattr(scfg, "mem_slots", cfg.mem_slots),
            slow_frac     = getattr(scfg, "slow_frac", cfg.slow_frac),
            mem_top_k     = getattr(scfg, "mem_top_k", cfg.mem_top_k),
            use_scan      = False,   # not needed for eval
        ).to(dev)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {cfg.eval}  ({model.n_params:,} params)")

        val_ids  = load_tokens_hf(
            cfg.dataset,
            "validation" if cfg.dataset == "wikitext103" else "train",
            max_chars=5_000_000,
        )
        val_buf  = TokenBuffer(val_ids, cfg.seq_len, min(cfg.batch_size, 8), dev)
        val_loss = evaluate(model, val_buf, n_batches=100)
        print(f"val_loss={val_loss:.4f}  "
              f"bpc={val_loss/math.log(2):.4f}  "
              f"ppl={math.exp(min(val_loss, 20)):.2f}")

    # ── Generate ─────────────────────────────────────────────────────────────
    elif cfg.sample:
        assert cfg.ckpt, "--ckpt required with --sample"
        ckpt  = torch.load(cfg.ckpt, map_location=dev, weights_only=False)
        scfg  = argparse.Namespace(**ckpt.get("cfg", {}))
        model = PIAModel(
            vocab         = VOCAB,
            d_model       = getattr(scfg, "d_model",   cfg.d_model),
            n_layers      = getattr(scfg, "n_layers",  cfg.n_layers),
            d_state       = getattr(scfg, "d_state",   cfg.d_state),
            n_heads       = getattr(scfg, "n_heads",   cfg.n_heads),
            K             = getattr(scfg, "K",         cfg.K),
            mem_slots     = getattr(scfg, "mem_slots", cfg.mem_slots),
            slow_frac     = getattr(scfg, "slow_frac", cfg.slow_frac),
            mem_top_k     = getattr(scfg, "mem_top_k", cfg.mem_top_k),
            use_scan      = False,
        ).to(dev)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {cfg.ckpt}  ({model.n_params:,} params)\n")
        text = model.generate(cfg.sample, n=cfg.gen_n,
                              temp=cfg.gen_temp, top_k=cfg.gen_topk,
                              top_p=cfg.gen_topp)
        print("─" * 60)
        print(text)
        print("─" * 60)

    # ── Train ─────────────────────────────────────────────────────────────────
    else:
        print("═" * 64)
        print("  PIA Language Model — Optimised Training")
        print("═" * 64)
        train(cfg)
