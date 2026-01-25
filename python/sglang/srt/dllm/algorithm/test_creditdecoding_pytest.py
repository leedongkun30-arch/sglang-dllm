# test_creditdecoding_pytest.py
"""
Pytest version of the standalone dummy-based tests.

Run:
  pytest -q
"""

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
import torch

# ----------------------------------------------------------------------
# Optional Triton
# ----------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


# ----------------------------------------------------------------------
# Torch reference implementations (the "original" logic pieces)
# ----------------------------------------------------------------------
@torch.no_grad()
def credit_update_top1_torch(idx: torch.Tensor, val: torch.Tensor,
                            top1_idx: torch.Tensor, top1_p: torch.Tensor,
                            gamma: float, alpha: float, eps: float):
    """
    idx: (B,T,K) int32/int64
    val: (B,T,K) fp16/fp32
    top1_idx: (B,T) int64/int32
    top1_p: (B,T) fp32
    """
    B, T, K = idx.shape
    val.mul_(gamma)

    inc = top1_p.clamp_min(eps).pow(alpha).to(val.dtype)  # (B,T)

    match = (idx == top1_idx.unsqueeze(-1))  # (B,T,K)
    has = match.any(dim=-1)                  # (B,T)
    match_slot = match.to(torch.int64).argmax(dim=-1)  # (B,T)

    empty = (idx < 0)
    has_empty = empty.any(dim=-1)
    empty_slot = empty.to(torch.int64).argmax(dim=-1)
    min_slot = val.argmin(dim=-1)

    ins_slot = torch.where(has_empty, empty_slot, min_slot)
    chosen = torch.where(has, match_slot, ins_slot)  # (B,T)

    rows_b = torch.arange(B, device=idx.device).view(B, 1).expand(B, T)
    rows_t = torch.arange(T, device=idx.device).view(1, T).expand(B, T)

    val[rows_b, rows_t, chosen] += inc

    do_insert = ~has
    idx[rows_b, rows_t, chosen] = torch.where(
        do_insert,
        top1_idx.to(idx.dtype),
        idx[rows_b, rows_t, chosen],
    )


@torch.no_grad()
def fused_top1_prob_torch(logits: torch.Tensor,
                          idx: torch.Tensor, val: torch.Tensor,
                          raw_top1_id: torch.Tensor,
                          raw_top1_logit: torch.Tensor,
                          logZ: torch.Tensor,
                          lam: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (B,T,V)
    idx: (B,T,K) int
    val: (B,T,K) float
    raw_top1_id/raw_top1_logit/logZ: (B,T)
    returns fused_id: (B,T), fused_p: (B,T)
    """
    valid = idx >= 0
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))

    lk = logits.gather(-1, safe_idx)  # (B,T,K)
    lk = lk.masked_fill(~valid, torch.finfo(lk.dtype).min)

    delta = lam * torch.log1p(val.to(torch.float32))
    delta = delta.masked_fill(~valid, 0.0)
    lk_fused = lk.to(torch.float32) + delta

    best_k_logits, best_k_pos = lk_fused.max(dim=-1)  # (B,T)
    best_k_id = safe_idx.gather(-1, best_k_pos.unsqueeze(-1)).squeeze(-1)  # (B,T)

    onehot = valid & (safe_idx == raw_top1_id.unsqueeze(-1))
    raw_delta = (delta * onehot.to(delta.dtype)).sum(dim=-1)
    raw_fused_logit = raw_top1_logit + raw_delta

    choose_k = best_k_logits > raw_fused_logit
    fused_id = torch.where(choose_k, best_k_id, raw_top1_id)
    fused_top1_logit = torch.where(choose_k, best_k_logits, raw_fused_logit)

    logZ_u = logZ.unsqueeze(-1)
    sum_pk = torch.exp(lk.to(torch.float32) - logZ_u).masked_fill(~valid, 0.0).sum(dim=-1)
    sum_pk_boost = torch.exp(lk_fused - logZ_u).masked_fill(~valid, 0.0).sum(dim=-1)

    corr = (1.0 - sum_pk + sum_pk_boost).clamp_min(eps)
    logZ_fused = logZ + torch.log(corr)

    fused_p = torch.exp(fused_top1_logit - logZ_fused)
    return fused_id, fused_p


# ----------------------------------------------------------------------
# Triton kernels (standalone, BT-flattened)
# ----------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def credit_update_top1_kernel_bt(
        IDX_PTR, VAL_PTR,
        TOP1_IDX_PTR, TOP1_P_PTR,
        gamma: tl.constexpr,
        alpha: tl.constexpr,
        eps: tl.constexpr,
        K: tl.constexpr,
        stride_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = pid * stride_k
        offs_k = tl.arange(0, K)

        idx = tl.load(IDX_PTR + base + offs_k, mask=True, other=-1).to(tl.int32)
        val = tl.load(VAL_PTR + base + offs_k, mask=True, other=0).to(tl.float32)

        val = val * gamma

        top1 = tl.load(TOP1_IDX_PTR + pid).to(tl.int32)
        p = tl.load(TOP1_P_PTR + pid).to(tl.float32)
        p = tl.maximum(p, eps)
        inc = tl.power(p, alpha)

        match = idx == top1
        has = tl.sum(match.to(tl.int32), axis=0) > 0

        match_i = tl.where(match, offs_k, 1_000_000)
        match_slot = tl.min(match_i, axis=0).to(tl.int32)

        empty = idx < 0
        has_empty = tl.sum(empty.to(tl.int32), axis=0) > 0
        empty_i = tl.where(empty, offs_k, 1_000_000)
        empty_slot = tl.min(empty_i, axis=0).to(tl.int32)

        min_val = tl.min(val, axis=0)
        is_min = val == min_val
        min_pos_i = tl.where(is_min, offs_k, 1_000_000)
        min_slot = tl.min(min_pos_i, axis=0).to(tl.int32)

        ins_slot = tl.where(has_empty, empty_slot, min_slot)
        chosen = tl.where(has, match_slot, ins_slot)

        is_chosen = offs_k == chosen
        val = tl.where(is_chosen, val + inc, val)

        do_insert = ~has
        idx = tl.where(is_chosen & do_insert, top1, idx)

        tl.store(VAL_PTR + base + offs_k, val.to(tl.float16))
        tl.store(IDX_PTR + base + offs_k, idx.to(tl.int32))

    @triton.jit
    def fused_top1_prob_kernel_bt(
        LOGITS_PTR,
        IDX_PTR, VAL_PTR,
        RAW_TOP1_ID_PTR,
        RAW_TOP1_LOGIT_PTR,
        LOGZ_PTR,
        OUT_ID_PTR,
        OUT_P_PTR,
        V: tl.constexpr,
        K: tl.constexpr,
        lam: tl.constexpr,
        eps: tl.constexpr,
        stride_logits_row: tl.constexpr,
        stride_k: tl.constexpr,
    ):
        pid = tl.program_id(0)

        base_k = pid * stride_k
        offs_k = tl.arange(0, K)

        idx = tl.load(IDX_PTR + base_k + offs_k, mask=True, other=-1).to(tl.int32)
        val = tl.load(VAL_PTR + base_k + offs_k, mask=True, other=0).to(tl.float32)
        valid = idx >= 0

        row_ptr = LOGITS_PTR + pid * stride_logits_row
        lk = tl.load(row_ptr + idx, mask=valid, other=-1e20).to(tl.float32)

        delta = lam * tl.log1p(val)
        delta = tl.where(valid, delta, 0.0)
        lk_fused = lk + delta

        best_k_logits = tl.max(lk_fused, axis=0)
        best_k_pos = tl.argmax(lk_fused, axis=0).to(tl.int32)
        best_k_id = tl.load(IDX_PTR + base_k + best_k_pos, mask=True, other=0).to(tl.int32)

        raw_id = tl.load(RAW_TOP1_ID_PTR + pid).to(tl.int32)
        raw_logit = tl.load(RAW_TOP1_LOGIT_PTR + pid).to(tl.float32)
        logZ = tl.load(LOGZ_PTR + pid).to(tl.float32)

        onehot = valid & (idx == raw_id)
        raw_delta = tl.sum(tl.where(onehot, delta, 0.0), axis=0)
        raw_fused_logit = raw_logit + raw_delta

        choose_k = best_k_logits > raw_fused_logit
        fused_id = tl.where(choose_k, best_k_id, raw_id)
        fused_logit = tl.where(choose_k, best_k_logits, raw_fused_logit)

        x = lk - logZ
        xb = lk_fused - logZ
        sum_pk = tl.sum(tl.where(valid, tl.exp(x), 0.0), axis=0)
        sum_pk_boost = tl.sum(tl.where(valid, tl.exp(xb), 0.0), axis=0)

        corr = tl.maximum(1.0 - sum_pk + sum_pk_boost, eps)
        logZ_fused = logZ + tl.log(corr)
        fused_p = tl.exp(fused_logit - logZ_fused)

        tl.store(OUT_ID_PTR + pid, fused_id)
        tl.store(OUT_P_PTR + pid, fused_p.to(tl.float32))


# ----------------------------------------------------------------------
# Dummy end-to-end runner
# ----------------------------------------------------------------------
@dataclass
class DummyForwardBatch:
    input_ids: torch.Tensor   # (BT,)
    batch_size: int


class DummyModelRunner:
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        self.W = torch.randn(vocab_size, vocab_size, generator=g, device=device, dtype=torch.float16)

    @torch.no_grad()
    def forward(self, forward_batch: DummyForwardBatch):
        BT = forward_batch.input_ids.numel()
        V = self.vocab_size

        ids = forward_batch.input_ids.clamp(0, V - 1).to(torch.long)
        logits = self.W[ids]  # (BT, V)

        pos = torch.arange(BT, device=logits.device, dtype=torch.float16)
        logits = logits + (pos.unsqueeze(1) * (1.0 / 4096.0))

        class Out:
            pass

        out = Out()
        out.logits_output = Out()
        out.logits_output.full_logits = logits
        out.can_run_graph = False
        return out


# ----------------------------------------------------------------------
# End-to-end: original-like vs optimized-like
# ----------------------------------------------------------------------
@torch.no_grad()
def run_original_like(runner: DummyModelRunner, fb: DummyForwardBatch,
                      mask_id: int, T: int, K: int,
                      threshold: float, gamma: float, lam: float, alpha: float, eps: float):
    B = fb.batch_size
    BT = B * T
    V = runner.vocab_size

    idx = torch.full((B, T, K), -1, device=fb.input_ids.device, dtype=torch.int32)
    val = torch.zeros((B, T, K), device=fb.input_ids.device, dtype=torch.float16)

    input_bt = fb.input_ids.view(B, T)
    start_list = (T - (input_bt == mask_id).sum(dim=1)).tolist()

    for _ in range(T):
        mask_bt = (input_bt == mask_id)
        if mask_bt.sum().item() == 0:
            break

        logits = runner.forward(fb).logits_output.full_logits.view(B, T, V)

        raw_id = torch.argmax(logits, dim=-1)
        raw_logit = logits.gather(-1, raw_id.unsqueeze(-1)).squeeze(-1).to(torch.float32)
        logZ = torch.logsumexp(logits.to(torch.float32), dim=-1)
        p_raw = torch.exp(raw_logit - logZ)

        credit_update_top1_torch(idx, val, raw_id.to(torch.int32), p_raw, gamma, alpha, eps)
        fused_id, fused_p = fused_top1_prob_torch(
            logits,
            idx.to(torch.int64),
            val.to(torch.float32),
            raw_id.to(torch.int64),
            raw_logit,
            logZ,
            lam,
            eps,
        )

        p = torch.maximum(p_raw, fused_p)
        x = torch.where(fused_p > p_raw, fused_id.to(raw_id.dtype), raw_id)

        confidence = torch.where(mask_bt, p, torch.full_like(p, -float("inf")))
        x_fill = torch.where(mask_bt, x, input_bt)

        transfer = confidence > threshold
        sel = torch.argmax(confidence, dim=1)
        transfer.scatter_(1, sel.view(B, 1), True)

        input_bt[transfer] = x_fill[transfer]

    return fb.input_ids.clone(), idx.clone(), val.clone(), [fb.input_ids.view(B, T)[i, start_list[i]:].clone() for i in range(B)]


@torch.no_grad()
def run_optimized_like(runner: DummyModelRunner, fb: DummyForwardBatch,
                       mask_id: int, T: int, K: int,
                       threshold: float, gamma: float, lam: float, alpha: float, eps: float):
    B = fb.batch_size
    BT = B * T
    V = runner.vocab_size
    device = fb.input_ids.device

    idx = torch.full((B, T, K), -1, device=device, dtype=torch.int32)
    val = torch.zeros((B, T, K), device=device, dtype=torch.float16)

    input_bt = fb.input_ids.view(B, T)
    start_list = (T - (input_bt == mask_id).sum(dim=1)).tolist()

    for _ in range(T):
        mask_bt = (input_bt == mask_id)
        if mask_bt.sum().item() == 0:
            break

        logits_btV = runner.forward(fb).logits_output.full_logits.view(BT, V)

        raw_id_bt = torch.argmax(logits_btV, dim=1)
        raw_logit_bt = logits_btV.gather(1, raw_id_bt.view(BT, 1)).squeeze(1).to(torch.float32)
        logZ_bt = torch.logsumexp(logits_btV.to(torch.float32), dim=1)
        p_raw_bt = torch.exp(raw_logit_bt - logZ_bt)

        idx_btK = idx.view(BT, K)
        val_btK = val.view(BT, K)

        use_triton = HAS_TRITON and device.type == "cuda"

        if use_triton:
            grid = (BT,)
            credit_update_top1_kernel_bt[grid](
                idx_btK,
                val_btK,
                raw_id_bt.to(torch.int32),
                p_raw_bt.to(torch.float32),
                gamma=gamma, alpha=alpha, eps=eps,
                K=K, stride_k=K,
                num_warps=1,
            )

            fused_id_bt = torch.empty((BT,), device=device, dtype=torch.int32)
            fused_p_bt = torch.empty((BT,), device=device, dtype=torch.float32)
            fused_top1_prob_kernel_bt[grid](
                logits_btV,
                idx_btK,
                val_btK,
                raw_id_bt.to(torch.int32),
                raw_logit_bt,
                logZ_bt,
                fused_id_bt,
                fused_p_bt,
                V=V, K=K, lam=lam, eps=eps,
                stride_logits_row=V, stride_k=K,
                num_warps=1,
            )
        else:
            # torch fallback
            credit_update_top1_torch(idx, val, raw_id_bt.view(B, T).to(torch.int32), p_raw_bt.view(B, T),
                                     gamma, alpha, eps)
            fid, fp = fused_top1_prob_torch(
                logits_btV.view(B, T, V),
                idx.to(torch.int64),
                val.to(torch.float32),
                raw_id_bt.view(B, T).to(torch.int64),
                raw_logit_bt.view(B, T),
                logZ_bt.view(B, T),
                lam,
                eps,
            )
            fused_id_bt = fid.reshape(BT).to(torch.int32)
            fused_p_bt = fp.reshape(BT).to(torch.float32)

        p = torch.maximum(p_raw_bt, fused_p_bt).view(B, T)
        x = torch.where(fused_p_bt > p_raw_bt, fused_id_bt.to(raw_id_bt.dtype), raw_id_bt).view(B, T)

        confidence = torch.where(mask_bt, p, torch.full_like(p, -float("inf")))
        x_fill = torch.where(mask_bt, x, input_bt)

        transfer = confidence > threshold
        sel = torch.argmax(confidence, dim=1)
        transfer.scatter_(1, sel.view(B, 1), True)
        input_bt[transfer] = x_fill[transfer]

    return fb.input_ids.clone(), idx.clone(), val.clone(), [fb.input_ids.view(B, T)[i, start_list[i]:].clone() for i in range(B)]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def assert_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-5, name="tensor"):
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = (a - b).abs().max().item()
        max_rel = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
        raise AssertionError(f"{name} not close: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")


# ----------------------------------------------------------------------
# Pytest fixtures / parameterization
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_piecewise_equivalence(device):
    B, T, K, V = 4, 8, 8, 256
    gamma, lam, alpha, eps = 0.65, 0.70, 0.50, 1e-6

    logits = torch.randn(B, T, V, device=device, dtype=torch.float16)
    raw_id = torch.argmax(logits, dim=-1)
    raw_logit = logits.gather(-1, raw_id.unsqueeze(-1)).squeeze(-1).to(torch.float32)
    logZ = torch.logsumexp(logits.to(torch.float32), dim=-1)
    p_raw = torch.exp(raw_logit - logZ)

    idx_ref = torch.full((B, T, K), -1, device=device, dtype=torch.int32)
    val_ref = torch.zeros((B, T, K), device=device, dtype=torch.float16)
    idx_opt = idx_ref.clone()
    val_opt = val_ref.clone()

    # reference update
    credit_update_top1_torch(idx_ref, val_ref, raw_id.to(torch.int32), p_raw, gamma, alpha, eps)

    # optimized update
    BT = B * T
    if HAS_TRITON and device.type == "cuda":
        grid = (BT,)
        credit_update_top1_kernel_bt[grid](
            idx_opt.view(BT, K),
            val_opt.view(BT, K),
            raw_id.reshape(BT).to(torch.int32),
            p_raw.reshape(BT).to(torch.float32),
            gamma=gamma, alpha=alpha, eps=eps,
            K=K, stride_k=K,
            num_warps=1,
        )
    else:
        credit_update_top1_torch(idx_opt, val_opt, raw_id.to(torch.int32), p_raw, gamma, alpha, eps)

    assert_close(val_ref.to(torch.float32), val_opt.to(torch.float32), rtol=1e-3, atol=1e-3, name="val(after update)")
    assert torch.equal(idx_ref, idx_opt), "idx(after update) mismatch"

    # reference fused
    fused_id_ref, fused_p_ref = fused_top1_prob_torch(
        logits,
        idx_ref.to(torch.int64),
        val_ref.to(torch.float32),
        raw_id.to(torch.int64),
        raw_logit,
        logZ,
        lam,
        eps,
    )

    # optimized fused
    if HAS_TRITON and device.type == "cuda":
        fused_id_opt_bt = torch.empty((BT,), device=device, dtype=torch.int32)
        fused_p_opt_bt = torch.empty((BT,), device=device, dtype=torch.float32)
        grid = (BT,)
        fused_top1_prob_kernel_bt[grid](
            logits.view(BT, V),
            idx_opt.view(BT, K),
            val_opt.view(BT, K),
            raw_id.reshape(BT).to(torch.int32),
            raw_logit.reshape(BT),
            logZ.reshape(BT),
            fused_id_opt_bt,
            fused_p_opt_bt,
            V=V, K=K, lam=lam, eps=eps,
            stride_logits_row=V, stride_k=K,
            num_warps=1,
        )
        fused_id_opt = fused_id_opt_bt.view(B, T).to(torch.int64)
        fused_p_opt = fused_p_opt_bt.view(B, T)
    else:
        fused_id_opt, fused_p_opt = fused_id_ref.clone(), fused_p_ref.clone()

    assert torch.equal(fused_id_ref, fused_id_opt), "fused_id mismatch"
    assert_close(fused_p_ref, fused_p_opt, rtol=2e-3, atol=2e-3, name="fused_p")


def test_end_to_end_equivalence(device):
    B, T, K, V = 4, 8, 8, 256
    mask_id = 999
    threshold = 0.95
    gamma, lam, alpha, eps = 0.65, 0.70, 0.50, 1e-6

    runner = DummyModelRunner(vocab_size=V, device=device)

    # deterministic input_ids with masks in last 3 positions
    input_ids = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    input_ids[:, -3:] = mask_id
    input_bt = input_ids.reshape(B * T).contiguous()

    fb1 = DummyForwardBatch(input_ids=input_bt.clone(), batch_size=B)
    fb2 = DummyForwardBatch(input_ids=input_bt.clone(), batch_size=B)

    final_ref, idx_ref, val_ref, next_ref = run_original_like(
        runner, fb1, mask_id, T, K, threshold, gamma, lam, alpha, eps
    )
    final_opt, idx_opt, val_opt, next_opt = run_optimized_like(
        runner, fb2, mask_id, T, K, threshold, gamma, lam, alpha, eps
    )

    assert torch.equal(final_ref, final_opt), "final input_ids mismatch"
    assert torch.equal(idx_ref, idx_opt), "idx(cache) mismatch"
    assert_close(val_ref.to(torch.float32), val_opt.to(torch.float32), rtol=1e-3, atol=1e-3, name="val(cache)")

    assert len(next_ref) == len(next_opt)
    for i, (a, b) in enumerate(zip(next_ref, next_opt)):
        assert torch.equal(a, b), f"next_list[{i}] mismatch"
