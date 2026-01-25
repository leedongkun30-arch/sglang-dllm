"""
Unofficial implementation of CreditDecoding: Accelerating Parallel Decoding in
Diffusion Large Language Models with Trace Credits (https://arxiv.org/pdf/2510.06133)
"""

from typing import List, Tuple, Union

import numpy as np
import torch

# ---- Triton (optional) -------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ---- sglang imports ----------------------------------------------------------
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
    def credit_update_top1_kernel_bt(
        IDX_PTR,              # int32, [BT, K]
        VAL_PTR,              # fp16/bf16, [BT, K]
        TOP1_IDX_PTR,         # int32, [BT]
        TOP1_P_PTR,           # fp32/fp16, [BT]
        gamma: tl.constexpr,
        alpha: tl.constexpr,
        eps: tl.constexpr,
        K: tl.constexpr,
        stride_k: tl.constexpr,  # == K
    ):
        pid = tl.program_id(0)  # 0..BT-1
        base = pid * stride_k
        offs_k = tl.arange(0, K)

        idx = tl.load(IDX_PTR + base + offs_k, mask=True, other=-1).to(tl.int32)
        val = tl.load(VAL_PTR + base + offs_k, mask=True, other=0).to(tl.float32)

        # decay
        val = val * gamma

        top1 = tl.load(TOP1_IDX_PTR + pid).to(tl.int32)
        p = tl.load(TOP1_P_PTR + pid).to(tl.float32)
        p = tl.maximum(p, eps)
        inc = tl.power(p, alpha)  # fp32

        match = idx == top1
        has = tl.sum(match.to(tl.int32), axis=0) > 0

        # first match slot
        match_i = tl.where(match, offs_k, 1_000_000)
        match_slot = tl.min(match_i, axis=0).to(tl.int32)  # large if none

        empty = idx < 0
        has_empty = tl.sum(empty.to(tl.int32), axis=0) > 0
        empty_i = tl.where(empty, offs_k, 1_000_000)
        empty_slot = tl.min(empty_i, axis=0).to(tl.int32)

        # min slot by val (when no empty)
        min_val = tl.min(val, axis=0)
        # pick first position where val == min_val
        is_min = val == min_val
        min_pos_i = tl.where(is_min, offs_k, 1_000_000)
        min_slot = tl.min(min_pos_i, axis=0).to(tl.int32)

        ins_slot = tl.where(has_empty, empty_slot, min_slot)
        chosen = tl.where(has, match_slot, ins_slot)

        is_chosen = offs_k == chosen
        val = tl.where(is_chosen, val + inc, val)

        do_insert = ~has
        idx = tl.where(is_chosen & do_insert, top1, idx)

        # store
        # keep VAL as fp16 for bandwidth (you can switch to bf16 if you prefer)
        tl.store(VAL_PTR + base + offs_k, val.to(tl.float16))
        tl.store(IDX_PTR + base + offs_k, idx.to(tl.int32))

    @triton.jit
    def fused_top1_prob_kernel_bt(
        LOGITS_PTR,                 # fp16/bf16/fp32, [BT, V]
        IDX_PTR,                    # int32, [BT, K]
        VAL_PTR,                    # fp16/bf16, [BT, K]
        RAW_TOP1_ID_PTR,            # int32, [BT]
        RAW_TOP1_LOGIT_PTR,         # fp32, [BT]
        LOGZ_PTR,                   # fp32, [BT]
        OUT_ID_PTR,                 # int32, [BT]
        OUT_P_PTR,                  # fp32, [BT]
        V: tl.constexpr,
        K: tl.constexpr,
        lam: tl.constexpr,
        eps: tl.constexpr,
        stride_logits_row: tl.constexpr,  # == V
        stride_k: tl.constexpr,           # == K
    ):
        pid = tl.program_id(0)

        base_k = pid * stride_k
        offs_k = tl.arange(0, K)

        idx = tl.load(IDX_PTR + base_k + offs_k, mask=True, other=-1).to(tl.int32)
        val = tl.load(VAL_PTR + base_k + offs_k, mask=True, other=0).to(tl.float32)
        valid = idx >= 0

        row_ptr = LOGITS_PTR + pid * stride_logits_row
        lk = tl.load(row_ptr + idx, mask=valid, other=-1e20).to(tl.float32)

        # log1p(val) 호환: log(1 + val)
        delta = lam * tl.log(1.0 + val)
        delta = tl.where(valid, delta, 0.0)
        lk_fused = lk + delta

        # argmax 호환: max 값만 구하고, "첫 번째로 max와 같은 위치"를 선택
        best_k_logits = tl.max(lk_fused, axis=0)

        # best_k_pos: lk_fused == best_k_logits 인 곳 중 가장 작은 index
        is_best = lk_fused == best_k_logits
        best_pos_i = tl.where(is_best, offs_k, 1_000_000)
        best_k_pos = tl.min(best_pos_i, axis=0).to(tl.int32)

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

        self.credit_slots = int(algo_cfg.get("credit_slots", 8))

        # IMPORTANT: eps must be float (your original code had int(...), which makes 1e-6 -> 0)
        self.eps = float(algo_cfg.get("credit_eps", 1e-6))

        # cache
        self.cache_inited = False
        self.idx = None   # int32 [B, T, K]
        self.val = None   # fp16  [B, T, K]

    def init_cache(self, device, batch_size: int):
        if (not self.cache_inited) or (self.idx is None) or (self.idx.shape[0] != batch_size) or (self.idx.device != device):
            self.cache_inited = True
            self.idx = torch.full(
                (batch_size, self.block_size, self.credit_slots),
                -1,
                dtype=torch.int32,
                device=device,
            )
            # store val as fp16 to reduce bandwidth (internal ops use fp32)
            self.val = torch.zeros(
                (batch_size, self.block_size, self.credit_slots),
                dtype=torch.float16,
                device=device,
            )
        else:
            self.idx.fill_(-1)
            self.val.zero_()

    @torch.no_grad()
    def _credit_update_and_fuse_bt(
        self,
        logits_btV: torch.Tensor,          # (BT, V)
        raw_top1_id_bt: torch.Tensor,      # (BT,) int64/int32
        raw_top1_logit_bt: torch.Tensor,   # (BT,) fp32
        logZ_bt: torch.Tensor,             # (BT,) fp32
        p_raw_bt: torch.Tensor,            # (BT,) fp32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates credits with raw top1 info, then returns fused top1 id and fused probability.
        """
        B, T, K = self.idx.shape
        BT = B * T
        assert logits_btV.shape[0] == BT

        idx_btK = self.idx.view(BT, K)
        val_btK = self.val.view(BT, K)
        top1_idx_bt = raw_top1_id_bt.to(torch.int32)
        top1_p_bt = p_raw_bt.to(torch.float32)

        use_triton = (
            _HAS_TRITON
            and logits_btV.is_cuda
            and idx_btK.is_cuda
            and val_btK.is_cuda
        )

        # 1) credit update
        if use_triton:
            grid = (BT,)
            credit_update_top1_kernel_bt[grid](
                idx_btK,
                val_btK,
                top1_idx_bt,
                top1_p_bt,
                gamma=self.gamma,
                alpha=self.alpha,
                eps=self.eps,
                K=K,
                stride_k=K,
                num_warps=1,
            )
        else:
            _credit_update_top1_torch(idx_btK, val_btK, top1_idx_bt, top1_p_bt, self.gamma, self.alpha, self.eps)

        # 2) fused top1 + prob
        fused_id = torch.empty((BT,), device=logits_btV.device, dtype=torch.int32)
        fused_p = torch.empty((BT,), device=logits_btV.device, dtype=torch.float32)

        if use_triton:
            grid = (BT,)
            fused_top1_prob_kernel_bt[grid](
                logits_btV,
                idx_btK,
                val_btK,
                top1_idx_bt,
                raw_top1_logit_bt.to(torch.float32),
                logZ_bt.to(torch.float32),
                fused_id,
                fused_p,
                V=logits_btV.shape[1],
                K=K,
                lam=self.lam,
                eps=self.eps,
                stride_logits_row=logits_btV.shape[1],
                stride_k=K,
                num_warps=1,
            )
        else:
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
            fused_id.copy_(fid.to(torch.int32))
            fused_p.copy_(fp.to(torch.float32))

        return fused_id, fused_p

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:

        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # mask check
        mask_index = forward_batch.input_ids == self.mask_id
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        self.init_cache(device, batch_size)

        # reshape views
        T = self.block_size
        BT = batch_size * T

        # start_list (dynamic length)
        # forward_batch.input_ids shape: (BT,)
        input_bt = forward_batch.input_ids.view(batch_size, T)
        mask_bt = (input_bt == self.mask_id)
        # start = T - num_masks
        start_list = (T - mask_bt.sum(dim=1)).tolist()

        # iterative fill
        for _ in range(T):
            mask_bt = (input_bt == self.mask_id)
            if mask_bt.sum().item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            logits_btV = logits_output.full_logits.view(BT, -1)  # (BT, V)

            # raw top1 / logZ / p_raw
            raw_top1_id_bt = torch.argmax(logits_btV, dim=1)  # (BT,)
            raw_top1_logit_bt = logits_btV.gather(1, raw_top1_id_bt.view(BT, 1)).squeeze(1).to(torch.float32)
            logZ_bt = torch.logsumexp(logits_btV.to(torch.float32), dim=1)
            p_raw_bt = torch.exp(raw_top1_logit_bt - logZ_bt)

            # credit update + fused
            fused_id_bt, fused_p_bt = self._credit_update_and_fuse_bt(
                logits_btV=logits_btV,
                raw_top1_id_bt=raw_top1_id_bt,
                raw_top1_logit_bt=raw_top1_logit_bt,
                logZ_bt=logZ_bt,
                p_raw_bt=p_raw_bt,
            )

            # choose between raw and credit
            p_bt = torch.maximum(p_raw_bt, fused_p_bt)  # (BT,)
            x_bt = torch.where(fused_p_bt > p_raw_bt, fused_id_bt.to(raw_top1_id_bt.dtype), raw_top1_id_bt)

            # apply only to masked positions
            p_bt2 = p_bt.view(batch_size, T)
            x_bt2 = x_bt.view(batch_size, T)
            mask_bt = (input_bt == self.mask_id)

            confidence = torch.where(mask_bt, p_bt2, torch.full_like(p_bt2, -float("inf")))
            x_fill = torch.where(mask_bt, x_bt2, input_bt)

            # transfer_index: confidence > threshold, but ensure at least 1 position per block
            transfer = confidence > self.threshold
            sel = torch.argmax(confidence, dim=1)  # (B,)
            transfer.scatter_(1, sel.view(batch_size, 1), True)

            # in-place update input ids
            input_bt[transfer] = x_fill[transfer]

        # final forward
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        # next token ids list with dynamic lengths
        next_token_ids = forward_batch.input_ids.view(batch_size, -1)
        next_token_ids_list = [next_token_ids[i, start_list[i]:] for i in range(batch_size)]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = CreditDecoding
