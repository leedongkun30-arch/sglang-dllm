"""
Unofficial implementation of CreditDecoding: Accelerating Parallel Decoding in
Diffusion Large Language Models with Trace Credits (https://arxiv.org/pdf/2510.06133)
"""

from typing import List, Tuple, Union

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


# =============================================================================
# Triton kernels
# =============================================================================
if _HAS_TRITON:

    @triton.jit
    def raw_top1_logz_kernel_bt(
        LOGITS_PTR,            # (BT, V)
        OUT_ID_PTR,            # (BT,) int32
        OUT_MAX_PTR,           # (BT,) fp32
        OUT_LOGZ_PTR,          # (BT,) fp32
        OUT_P_PTR,             # (BT,) fp32  (optional but we output)
        V: tl.constexpr,
        stride_logits_row,     # runtime scalar (elements)
        BLOCK_V: tl.constexpr, # e.g. 256/512/1024
    ):
        pid = tl.program_id(0)
        row_ptr = LOGITS_PTR + pid * stride_logits_row

        m = tl.full([], -1.0e20, tl.float32)  
        s = tl.full([], 0.0, tl.float32)      

        best_val = tl.full([], -1.0e20, tl.float32)
        best_idx = tl.full([], 0, tl.int32)

        for off in tl.static_range(0, V, BLOCK_V):
            offs = off + tl.arange(0, BLOCK_V)
            mask = offs < V

            x = tl.load(row_ptr + offs, mask=mask, other=-1.0e20).to(tl.float32)

            tile_max = tl.max(x, axis=0)
            tile_arg = tl.argmax(x, axis=0).to(tl.int32) + tl.full([], off, tl.int32)

            better = tile_max > best_val
            equal  = tile_max == best_val
            smaller = tile_arg < best_idx
            take = better | (equal & smaller)
            best_val = tl.where(take, tile_max, best_val)
            best_idx = tl.where(take, tile_arg, best_idx)

            ex = tl.exp(x - tile_max) * mask.to(tl.float32)
            tile_sum = tl.sum(ex, axis=0)

            new_m = tl.maximum(m, tile_max)
            s = s * tl.exp(m - new_m) + tile_sum * tl.exp(tile_max - new_m)
            m = new_m

        logZ = m + tl.log(s)
        p_raw = tl.exp(best_val - logZ)

        tl.store(OUT_ID_PTR + pid, best_idx)
        tl.store(OUT_MAX_PTR + pid, best_val)
        tl.store(OUT_LOGZ_PTR + pid, logZ)
        tl.store(OUT_P_PTR + pid, p_raw)


    @triton.jit
    def credit_update_top1_kernel_bt_k8(
        IDX_PTR, VAL_PTR,
        TOP1_IDX_PTR, TOP1_P_PTR,
        gamma,
        alpha,
        eps,
        stride_k,   # must be 8
    ):
        pid = tl.program_id(0)
        base = pid * stride_k

        idx0 = tl.load(IDX_PTR + base + 0, mask=True, other=-1).to(tl.int32)
        idx1 = tl.load(IDX_PTR + base + 1, mask=True, other=-1).to(tl.int32)
        idx2 = tl.load(IDX_PTR + base + 2, mask=True, other=-1).to(tl.int32)
        idx3 = tl.load(IDX_PTR + base + 3, mask=True, other=-1).to(tl.int32)

        val0 = tl.load(VAL_PTR + base + 0, mask=True, other=0.0).to(tl.float32) * gamma
        val1 = tl.load(VAL_PTR + base + 1, mask=True, other=0.0).to(tl.float32) * gamma
        val2 = tl.load(VAL_PTR + base + 2, mask=True, other=0.0).to(tl.float32) * gamma
        val3 = tl.load(VAL_PTR + base + 3, mask=True, other=0.0).to(tl.float32) * gamma

        top1 = tl.load(TOP1_IDX_PTR + pid).to(tl.int32)
        p = tl.load(TOP1_P_PTR + pid).to(tl.float32)
        p = tl.maximum(p, eps)
        inc = tl.exp(tl.log(p) * alpha)

        # match slot (first)
        has = (idx0 == top1)
        match_slot = tl.full([], 0, tl.int32)

        mk = (idx1 == top1); upd = mk & (~has); has = has | mk
        match_slot = match_slot * (1 - upd.to(tl.int32)) + tl.full([], 1, tl.int32) * upd.to(tl.int32)
        mk = (idx2 == top1); upd = mk & (~has); has = has | mk
        match_slot = match_slot * (1 - upd.to(tl.int32)) + tl.full([], 2, tl.int32) * upd.to(tl.int32)
        mk = (idx3 == top1); upd = mk & (~has); has = has | mk
        match_slot = match_slot * (1 - upd.to(tl.int32)) + tl.full([], 3, tl.int32) * upd.to(tl.int32)


        # empty slot (first)
        has_empty = (idx0 < 0)
        empty_slot = tl.full([], 0, tl.int32)

        ek = (idx1 < 0); upd = ek & (~has_empty); has_empty = has_empty | ek
        empty_slot = empty_slot * (1 - upd.to(tl.int32)) + tl.full([], 1, tl.int32) * upd.to(tl.int32)
        ek = (idx2 < 0); upd = ek & (~has_empty); has_empty = has_empty | ek
        empty_slot = empty_slot * (1 - upd.to(tl.int32)) + tl.full([], 2, tl.int32) * upd.to(tl.int32)
        ek = (idx3 < 0); upd = ek & (~has_empty); has_empty = has_empty | ek
        empty_slot = empty_slot * (1 - upd.to(tl.int32)) + tl.full([], 3, tl.int32) * upd.to(tl.int32)


        # min slot (first minimum)
        min_val = val0
        min_slot = tl.full([], 0, tl.int32)

        bm = (val1 < min_val); min_val = min_val * (1.0 - bm.to(tl.float32)) + val1 * bm.to(tl.float32)
        min_slot = min_slot * (1 - bm.to(tl.int32)) + tl.full([], 1, tl.int32) * bm.to(tl.int32)
        bm = (val2 < min_val); min_val = min_val * (1.0 - bm.to(tl.float32)) + val2 * bm.to(tl.float32)
        min_slot = min_slot * (1 - bm.to(tl.int32)) + tl.full([], 2, tl.int32) * bm.to(tl.int32)
        bm = (val3 < min_val); min_val = min_val * (1.0 - bm.to(tl.float32)) + val3 * bm.to(tl.float32)
        min_slot = min_slot * (1 - bm.to(tl.int32)) + tl.full([], 3, tl.int32) * bm.to(tl.int32)


        ins_slot = empty_slot * has_empty.to(tl.int32) + min_slot * (1 - has_empty.to(tl.int32))
        chosen = match_slot * has.to(tl.int32) + ins_slot * (1 - has.to(tl.int32))
        do_insert = ~has

        is0 = chosen == tl.full([], 0, tl.int32)
        is1 = chosen == tl.full([], 1, tl.int32)
        is2 = chosen == tl.full([], 2, tl.int32)
        is3 = chosen == tl.full([], 3, tl.int32)

        val0 = val0 + inc * is0.to(tl.float32)
        val1 = val1 + inc * is1.to(tl.float32)
        val2 = val2 + inc * is2.to(tl.float32)
        val3 = val3 + inc * is3.to(tl.float32)

        ins0 = (is0 & do_insert).to(tl.int32); idx0 = idx0 * (1 - ins0) + top1 * ins0
        ins1 = (is1 & do_insert).to(tl.int32); idx1 = idx1 * (1 - ins1) + top1 * ins1
        ins2 = (is2 & do_insert).to(tl.int32); idx2 = idx2 * (1 - ins2) + top1 * ins2
        ins3 = (is3 & do_insert).to(tl.int32); idx3 = idx3 * (1 - ins3) + top1 * ins3

        tl.store(VAL_PTR + base + 0, val0.to(tl.float16)); tl.store(IDX_PTR + base + 0, idx0)
        tl.store(VAL_PTR + base + 1, val1.to(tl.float16)); tl.store(IDX_PTR + base + 1, idx1)
        tl.store(VAL_PTR + base + 2, val2.to(tl.float16)); tl.store(IDX_PTR + base + 2, idx2)
        tl.store(VAL_PTR + base + 3, val3.to(tl.float16)); tl.store(IDX_PTR + base + 3, idx3)


    @triton.jit
    def fused_top1_prob_kernel_bt_k8(
        LOGITS_PTR,
        IDX_PTR, VAL_PTR,
        RAW_TOP1_ID_PTR,
        RAW_TOP1_LOGIT_PTR,
        LOGZ_PTR,
        OUT_ID_PTR,
        OUT_P_PTR,
        lam: tl.constexpr,
        eps: tl.constexpr,
        stride_logits_row: tl.constexpr,
        stride_k: tl.constexpr,   # must be 8
    ):
        pid = tl.program_id(0)
        base = pid * stride_k
        row_ptr = LOGITS_PTR + pid * stride_logits_row

        raw_id = tl.load(RAW_TOP1_ID_PTR + pid).to(tl.int32)
        raw_logit = tl.load(RAW_TOP1_LOGIT_PTR + pid).to(tl.float32)
        logZ = tl.load(LOGZ_PTR + pid).to(tl.float32)

        idx0 = tl.load(IDX_PTR + base + 0, mask=True, other=-1).to(tl.int32)
        idx1 = tl.load(IDX_PTR + base + 1, mask=True, other=-1).to(tl.int32)
        idx2 = tl.load(IDX_PTR + base + 2, mask=True, other=-1).to(tl.int32)
        idx3 = tl.load(IDX_PTR + base + 3, mask=True, other=-1).to(tl.int32)

        v0 = tl.load(VAL_PTR + base + 0, mask=True, other=0.0).to(tl.float32)
        v1 = tl.load(VAL_PTR + base + 1, mask=True, other=0.0).to(tl.float32)
        v2 = tl.load(VAL_PTR + base + 2, mask=True, other=0.0).to(tl.float32)
        v3 = tl.load(VAL_PTR + base + 3, mask=True, other=0.0).to(tl.float32)

        best_val = tl.full([], -1.0e20, tl.float32)
        best_id  = tl.full([], 0, tl.int32)
        sum_pk = tl.full([], 0.0, tl.float32)
        sum_pk_boost = tl.full([], 0.0, tl.float32)
        raw_delta = tl.full([], 0.0, tl.float32)

        valid = idx0 >= 0
        lk = tl.load(row_ptr + idx0, mask=valid, other=-1.0e20).to(tl.float32)
        delta = lam * tl.log(1.0 + v0)
        delta = delta * valid.to(tl.float32)
        lk_fused = lk + delta

        better = lk_fused > best_val  
        best_val = best_val * (1.0 - better.to(tl.float32)) + lk_fused * better.to(tl.float32)
        best_id  = best_id  * (1 - better.to(tl.int32)) + idx0 * better.to(tl.int32)

        raw_delta = raw_delta + delta * (valid & (idx0 == raw_id)).to(tl.float32)
        sum_pk = sum_pk + tl.exp(lk - logZ) * valid.to(tl.float32)
        sum_pk_boost = sum_pk_boost + tl.exp(lk_fused - logZ) * valid.to(tl.float32)

        
        valid = idx1 >= 0
        lk = tl.load(row_ptr + idx1, mask=valid, other=-1.0e20).to(tl.float32)
        delta = lam * tl.log(1.0 + v1)
        delta = delta * valid.to(tl.float32)
        lk_fused = lk + delta

        better = lk_fused > best_val  
        best_val = best_val * (1.0 - better.to(tl.float32)) + lk_fused * better.to(tl.float32)
        best_id  = best_id  * (1 - better.to(tl.int32)) + idx1 * better.to(tl.int32)

        raw_delta = raw_delta + delta * (valid & (idx1 == raw_id)).to(tl.float32)
        sum_pk = sum_pk + tl.exp(lk - logZ) * valid.to(tl.float32)
        sum_pk_boost = sum_pk_boost + tl.exp(lk_fused - logZ) * valid.to(tl.float32)

        
        valid = idx2 >= 0
        lk = tl.load(row_ptr + idx2, mask=valid, other=-1.0e20).to(tl.float32)
        delta = lam * tl.log(1.0 + v2)
        delta = delta * valid.to(tl.float32)
        lk_fused = lk + delta

        better = lk_fused > best_val  
        best_val = best_val * (1.0 - better.to(tl.float32)) + lk_fused * better.to(tl.float32)
        best_id  = best_id  * (1 - better.to(tl.int32)) + idx2 * better.to(tl.int32)

        raw_delta = raw_delta + delta * (valid & (idx2 == raw_id)).to(tl.float32)
        sum_pk = sum_pk + tl.exp(lk - logZ) * valid.to(tl.float32)
        sum_pk_boost = sum_pk_boost + tl.exp(lk_fused - logZ) * valid.to(tl.float32)

        
        valid = idx3 >= 0
        lk = tl.load(row_ptr + idx3, mask=valid, other=-1.0e20).to(tl.float32)
        delta = lam * tl.log(1.0 + v3)
        delta = delta * valid.to(tl.float32)
        lk_fused = lk + delta

        better = lk_fused > best_val  
        best_val = best_val * (1.0 - better.to(tl.float32)) + lk_fused * better.to(tl.float32)
        best_id  = best_id  * (1 - better.to(tl.int32)) + idx3 * better.to(tl.int32)

        raw_delta = raw_delta + delta * (valid & (idx3 == raw_id)).to(tl.float32)
        sum_pk = sum_pk + tl.exp(lk - logZ) * valid.to(tl.float32)
        sum_pk_boost = sum_pk_boost + tl.exp(lk_fused - logZ) * valid.to(tl.float32)

        raw_fused_logit = raw_logit + raw_delta
        choose_k = best_val > raw_fused_logit
        fused_id = best_id * choose_k.to(tl.int32) + raw_id * (1 - choose_k.to(tl.int32))
        fused_logit = best_val * choose_k.to(tl.float32) + raw_fused_logit * (1.0 - choose_k.to(tl.float32))

        corr = tl.maximum(1.0 - sum_pk + sum_pk_boost, eps)
        logZ_fused = logZ + tl.log(corr)
        fused_p = tl.exp(fused_logit - logZ_fused)

        tl.store(OUT_ID_PTR + pid, fused_id)
        tl.store(OUT_P_PTR + pid, fused_p.to(tl.float32))


    @triton.jit
    def apply_fill_block32_kernel_b(
        INPUT_PTR,             # (B*T,) int32
        RAW_ID_PTR,            # (B*T,) int32
        P_RAW_PTR,             # (B*T,) fp32
        FUSED_ID_PTR,          # (B*T,) int32
        FUSED_P_PTR,           # (B*T,) fp32
        mask_id: tl.constexpr,
        threshold,
        T: tl.constexpr,       # must be 32
    ):
        b = tl.program_id(0)
        offs = b * T + tl.arange(0, T)

        inp = tl.load(INPUT_PTR + offs).to(tl.int32)
        is_mask = inp == mask_id

        raw_id = tl.load(RAW_ID_PTR + offs).to(tl.int32)
        p_raw = tl.load(P_RAW_PTR + offs).to(tl.float32)
        fused_id = tl.load(FUSED_ID_PTR + offs).to(tl.int32)
        fused_p = tl.load(FUSED_P_PTR + offs).to(tl.float32)

        use_fused = fused_p > p_raw
        p = tl.maximum(p_raw, fused_p)
        x = raw_id * (1 - use_fused.to(tl.int32)) + fused_id * use_fused.to(tl.int32)

        m0 = is_mask.to(tl.int32)
        has_mask = tl.sum(m0, axis=0) > 0 

        neg_inf = tl.full([T], -1.0e20, tl.float32)
        conf = tl.where(is_mask, p, neg_inf)

        sel = tl.argmax(conf, axis=0)

        transfer = is_mask & (p > threshold)
        sel_mask = (tl.arange(0, T) == sel) & has_mask
        transfer = transfer | sel_mask

        out = tl.where(transfer, x, inp)
        tl.store(INPUT_PTR + offs, out.to(tl.int32))




# =============================================================================
# Torch fallbacks (CPU / no triton)
# =============================================================================
@torch.no_grad()
def _credit_update_top1_torch(idx_bt: torch.Tensor, val_bt: torch.Tensor,
                             top1_idx_bt: torch.Tensor, top1_p_bt: torch.Tensor,
                             gamma: float, alpha: float, eps: float):
    """
    idx_bt: (BT, K) int32/long
    val_bt: (BT, K) float16/float32
    top1_idx_bt: (BT,)
    top1_p_bt: (BT,)
    """
    BT, K = idx_bt.shape
    val_bt.mul_(gamma)

    inc = top1_p_bt.clamp_min(eps).pow(alpha).to(val_bt.dtype)

    # match slots
    match = (idx_bt == top1_idx_bt.view(BT, 1))
    has = match.any(dim=1)
    match_slot = match.to(torch.int64).argmax(dim=1)

    empty = (idx_bt < 0)
    has_empty = empty.any(dim=1)
    empty_slot = empty.to(torch.int64).argmax(dim=1)
    min_slot = val_bt.argmin(dim=1)

    ins_slot = torch.where(has_empty, empty_slot, min_slot)
    chosen = torch.where(has, match_slot, ins_slot)

    rows = torch.arange(BT, device=idx_bt.device)
    val_bt[rows, chosen] += inc

    do_insert = ~has
    idx_bt[rows, chosen] = torch.where(do_insert, top1_idx_bt.to(idx_bt.dtype), idx_bt[rows, chosen])


@torch.no_grad()
def _fused_top1_prob_torch(logits_btV: torch.Tensor, idx_btK: torch.Tensor, val_btK: torch.Tensor,
                          raw_top1_id_bt: torch.Tensor, raw_top1_logit_bt: torch.Tensor, logZ_bt: torch.Tensor,
                          lam: float, eps: float):
    """
    logits_btV: (BT, V)
    idx_btK: (BT, K)
    val_btK: (BT, K)
    returns: fused_id (BT,), fused_p (BT,)
    """
    BT, V = logits_btV.shape
    K = idx_btK.shape[1]

    valid = idx_btK >= 0
    safe_idx = torch.where(valid, idx_btK, torch.zeros_like(idx_btK))

    lk = logits_btV.gather(1, safe_idx)
    lk = lk.masked_fill(~valid, torch.finfo(lk.dtype).min)

    delta = lam * torch.log1p(val_btK.to(torch.float32))
    delta = delta.masked_fill(~valid, 0.0)
    lk_fused = lk.to(torch.float32) + delta

    best_k_logits, best_k_pos = lk_fused.max(dim=1)
    best_k_id = safe_idx.gather(1, best_k_pos.view(BT, 1)).squeeze(1)

    onehot = (safe_idx == raw_top1_id_bt.view(BT, 1)) & valid
    raw_delta = (delta * onehot.to(delta.dtype)).sum(dim=1)
    raw_fused_logit = raw_top1_logit_bt + raw_delta

    choose_k = best_k_logits > raw_fused_logit
    fused_id = torch.where(choose_k, best_k_id, raw_top1_id_bt)
    fused_logit = torch.where(choose_k, best_k_logits, raw_fused_logit)

    logZ_u = logZ_bt.view(BT, 1)
    sum_pk = torch.exp(lk.to(torch.float32) - logZ_u).masked_fill(~valid, 0.0).sum(dim=1)
    sum_pk_boost = torch.exp(lk_fused - logZ_u).masked_fill(~valid, 0.0).sum(dim=1)

    corr = (1.0 - sum_pk + sum_pk_boost).clamp_min(eps)
    logZ_fused = logZ_bt + torch.log(corr)

    fused_p = torch.exp(fused_logit - logZ_fused)
    return fused_id, fused_p


# =============================================================================
# Main algorithm
# =============================================================================
class CreditDecoding(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)

        algo_cfg = config.algorithm_config

        self.threshold = float(algo_cfg.get("threshold", 0.95))

        self.gamma = float(algo_cfg.get("credit_decay_gamma", 0.65))       # decay factor gamma
        self.lam = float(algo_cfg.get("credit_fusion_lambda", 0.70))       # fusion strength lambda
        self.alpha = float(algo_cfg.get("credit_prob_alpha", 0.50))        # concave exponent alpha (<1 boosts low p)

        self.credit_slots = int(algo_cfg.get("credit_slots", 4))

        self.eps = float(algo_cfg.get("credit_eps", 1e-6))

        self.cache_inited = False
        self.idx = None   
        self.val = None   

    def init_cache(self, device, batch_size: int):
        BT = batch_size * self.block_size
        if (not self.cache_inited) or (self.idx is None) or (self.idx.shape[0] != batch_size) or (self.idx.device != device):
            self.cache_inited = True
            self.idx = torch.full(
                (batch_size, self.block_size, self.credit_slots),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.val = torch.zeros(
                (batch_size, self.block_size, self.credit_slots),
                dtype=torch.float16,
                device=device,
            )

            self.raw_id    = torch.empty((BT,), device=device, dtype=torch.int32)
            self.raw_logit = torch.empty((BT,), device=device, dtype=torch.float32)
            self.logZ      = torch.empty((BT,), device=device, dtype=torch.float32)
            self.p_raw     = torch.empty((BT,), device=device, dtype=torch.float32)
        else:
            self.idx.fill_(-1)
            self.val.zero_()


    @torch.no_grad()
    def _credit_update_and_fuse_bt(
        self,
        logits_btV: torch.Tensor,          # (BT, V)
        raw_top1_id_bt: torch.Tensor,      # (BT,)
        raw_top1_logit_bt: torch.Tensor,   # (BT,)
        logZ_bt: torch.Tensor,             # (BT,)
        p_raw_bt: torch.Tensor,            # (BT,)
    ):
        B, T, K = self.idx.shape
        BT = B * T
        assert K == 4, "K==4 only for now"

        idx_btK = self.idx.view(BT, K)
        val_btK = self.val.view(BT, K)

        top1_idx_bt = raw_top1_id_bt.to(torch.int32)
        top1_p_bt = p_raw_bt.to(torch.float32)

        devtype = logits_btV.device.type
        use_triton = _HAS_TRITON and (devtype in ("cuda", "npu"))

        if use_triton:
            grid = (BT,)
            credit_update_top1_kernel_bt_k8[grid](
                idx_btK,
                val_btK,
                top1_idx_bt,
                top1_p_bt,
                gamma=self.gamma,
                alpha=self.alpha,
                eps=self.eps,
                stride_k=4,
                num_warps=1,
            )

            fused_top1_prob_kernel_bt_k8[grid](
                logits_btV,
                idx_btK,
                val_btK,
                top1_idx_bt,                         
                raw_top1_logit_bt.to(torch.float32),
                logZ_bt.to(torch.float32),
                self.fused_id,
                self.fused_p,
                lam=self.lam,
                eps=self.eps,
                stride_logits_row=logits_btV.shape[1],
                stride_k=4,
                num_warps=1,
            )
            return self.fused_id, self.fused_p

        # fallback (CPU/no triton)
        _credit_update_top1_torch(idx_btK, val_btK, top1_idx_bt, top1_p_bt, self.gamma, self.alpha, self.eps)
        fid, fp = _fused_top1_prob_torch(
            logits_btV,
            idx_btK.to(torch.int64),
            val_btK.to(torch.float32),
            top1_idx_bt.to(torch.int64),
            raw_top1_logit_bt.to(torch.float32),
            logZ_bt.to(torch.float32),
            self.lam,
            self.eps,
        )
        self.fused_id.copy_(fid.to(torch.int32))
        self.fused_p.copy_(fp.to(torch.float32))
        return self.fused_id, self.fused_p



    def run(self, model_runner: ModelRunner, forward_batch: ForwardBatch):
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        if forward_batch.input_ids.dtype != torch.int32:
            forward_batch.input_ids = forward_batch.input_ids.to(torch.int32)

        mask_index = (forward_batch.input_ids == self.mask_id)
        if not torch.any(mask_index):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        self.init_cache(device, batch_size)

        T = self.block_size
        BT = batch_size * T

        input_bt = forward_batch.input_ids.view(batch_size, T)

        for _ in range(T):
            mask_bt = (input_bt == self.mask_id)
            if not torch.any(mask_bt):
                break
            

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            logits_btV = logits_output.full_logits.view(BT, -1)  # (BT, V)

            
            raw_top1_id_bt = torch.argmax(logits_btV, dim=1)  # (BT,)
            raw_top1_logit_bt = logits_btV.gather(1, raw_top1_id_bt.view(BT, 1)).squeeze(1).to(torch.float32)
            logZ_bt = torch.logsumexp(logits_btV.to(torch.float32), dim=1)
            p_raw_bt = torch.exp(raw_top1_logit_bt - logZ_bt)

            fused_id_bt, fused_p_bt = self._credit_update_and_fuse_bt(
                logits_btV=logits_btV,
                raw_top1_id_bt=raw_top1_id_bt,
                raw_top1_logit_bt=raw_top1_logit_bt,
                logZ_bt=logZ_bt,
                p_raw_bt=p_raw_bt,
            )

            devtype = forward_batch.input_ids.device.type
            if _HAS_TRITON and devtype in ("cuda", "npu"):
                apply_fill_block32_kernel_b[(batch_size,)](
                    forward_batch.input_ids,              
                    raw_top1_id_bt.to(torch.int32),
                    p_raw_bt.to(torch.float32),
                    fused_id_bt,
                    fused_p_bt,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    T=32,
                    num_warps=1,
                )
            else:
                p_bt = torch.maximum(p_raw_bt, fused_p_bt)
                x_bt = torch.where(fused_p_bt > p_raw_bt, fused_id_bt.to(raw_top1_id_bt.dtype), raw_top1_id_bt)
                p_bt2 = p_bt.view(batch_size, T)
                x_bt2 = x_bt.view(batch_size, T)
                mask_bt = (input_bt == self.mask_id)
                confidence = torch.where(mask_bt, p_bt2, torch.full_like(p_bt2, -float("inf")))
                x_fill = torch.where(mask_bt, x_bt2, input_bt)
                transfer = confidence > self.threshold
                sel = torch.argmax(confidence, dim=1)
                transfer.scatter_(1, sel.view(batch_size, 1), True)
                input_bt[transfer] = x_fill[transfer]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = forward_batch.input_ids[mask_index]

        return logits_output, next_token_ids, can_run_cuda_graph



Algorithm = CreditDecoding
