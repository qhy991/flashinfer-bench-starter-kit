"""Experimental Triton GDN decode for gdn_decode_qk4_v8_d128_k_last.

RTX 4090 dev-lane prototype. Not B200 performance evidence.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, BLOCK_K)
    qk_head = head // 2

    k_vals = tl.load(k + ((batch * 4 + qk_head) * 128 + offs_k)).to(tl.float32)
    q_vals = tl.load(q + ((batch * 4 + qk_head) * 128 + offs_k)).to(tl.float32)
    v_vals = tl.load(v + ((batch * 8 + head) * 128 + offs_v), mask=offs_v < 128, other=0.0).to(tl.float32)

    a_val = tl.load(a + batch * 8 + head).to(tl.float32)
    b_val = tl.load(b + batch * 8 + head).to(tl.float32)
    alog = tl.load(A_log + head).to(tl.float32)
    dbias = tl.load(dt_bias + head).to(tl.float32)
    x = a_val + dbias
    # softplus(x) = log(1 + exp(x)); x range in FIB blobs is safe for exp here.
    g = tl.exp(-tl.exp(alog) * tl.log(1.0 + tl.exp(x)))
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    st_ptrs = state + ((batch * 8 + head) * 128 + offs_v[:, None]) * 128 + offs_k[None, :]
    st = tl.load(st_ptrs, mask=offs_v[:, None] < 128, other=0.0).to(tl.float32)
    old_state = g * st
    old_v = tl.sum(old_state * k_vals[None, :], axis=1)
    delta = beta * (v_vals - old_v)
    ns = old_state + delta[:, None] * k_vals[None, :]
    out = scale * tl.sum(ns * q_vals[None, :], axis=1)

    tl.store(new_state + ((batch * 8 + head) * 128 + offs_v[:, None]) * 128 + offs_k[None, :], ns, mask=offs_v[:, None] < 128)
    tl.store(output + ((batch * 8 + head) * 128 + offs_v), out.to(tl.bfloat16), mask=offs_v < 128)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = int(q.shape[0])
    K = int(q.shape[-1])
    scale_value = scale
    if scale_value is None or scale_value == 0.0:
        scale_value = 1.0 / math.sqrt(K)
    if state is None:
        state = torch.zeros((B, 8, 128, 128), dtype=torch.float32, device=q.device)
    output_flat = torch.empty((B, 8, 128), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty_like(state, dtype=torch.float32)
    grid = (B, 8, triton.cdiv(128, 64))
    _gdn_decode_kernel[grid](q, k, v, state, A_log, a, dt_bias, b, output_flat, new_state, float(scale_value), BLOCK_V=64, BLOCK_K=128, num_warps=4)
    return output_flat.view(B, 1, 8, 128), new_state
