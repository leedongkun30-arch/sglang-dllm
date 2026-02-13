"""
Unofficial implementation of CreditDecoding: Accelerating Parallel Decoding in
Diffusion Large Language Models with Trace Credits (https://arxiv.org/pdf/2510.06133)
"""

import math
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import logging

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_npu

_is_npu = is_npu()


_TRITON_OK = False
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False



def parallel_decoding_update_input_ids_vectorized(
    input_ids_1d: torch.Tensor,  # [B*blk]
    full_logits_2d: torch.Tensor,  # [B*blk, V]
    mask_id: int,
    block_size: int,
    threshold: float,
    finished: Optional[
        torch.Tensor
    ] = None,  # [B] bool, True means do NOT update this row
    force_at_least_one: bool = True,
) -> None:
    """
    Vectorized update for parallel decoding:
      - reshape to [B, blk] and update each row in-place
      - replaces per-batch Python loop + .item() sync points
      - optionally gates out finished rows

    Semantics (matches your old code):
      - Only positions where input_ids == mask_id are candidates for replacement
      - Replace with argmax token if its confidence > threshold
      - If a row has any mask positions but none pass threshold, force select 1 position (max confidence)
      - Rows with no mask positions are left unchanged (equivalent to `continue`)
      - If `finished` is provided, finished rows are always left unchanged
    """
    assert (
        input_ids_1d.dim() == 1
    ), f"input_ids_1d must be 1D, got {tuple(input_ids_1d.shape)}"
    assert (
        full_logits_2d.dim() == 2
    ), f"full_logits_2d must be 2D, got {tuple(full_logits_2d.shape)}"
    assert (
        input_ids_1d.shape[0] == full_logits_2d.shape[0]
    ), f"len mismatch: {input_ids_1d.shape[0]} vs {full_logits_2d.shape[0]}"
    assert (
        input_ids_1d.shape[0] % block_size == 0
    ), f"len(input_ids_1d) must be divisible by block_size, got {input_ids_1d.shape[0]}, block_size={block_size}"
    assert 0.0 < threshold <= 1.0, f"threshold should be in (0,1], got {threshold}"

    B = input_ids_1d.shape[0] // block_size
    blk = block_size
    V = full_logits_2d.shape[1]
    dev = input_ids_1d.device

    # Views (no copy)
    input_ids = input_ids_1d.view(B, blk)  # [B, blk]
    logits = full_logits_2d.view(B, blk, V)  # [B, blk, V]

    # Candidate mask
    mask = input_ids.eq(mask_id)  # [B, blk]
    if finished is not None:
        assert finished.shape == (
            B,
        ), f"finished must be shape [B]={B}, got {tuple(finished.shape)}"
        alive = (~finished).view(B, 1)  # [B,1]
        eff_mask = mask & alive
    else:
        eff_mask = mask

    # Argmax token id per position and its log-prob.
    # logp = log softmax(logits)[argmax] = max_logit - logsumexp(logits)
    max_logit, x = logits.max(dim=-1)  # [B, blk], [B, blk]
    lse = torch.logsumexp(logits, dim=-1)  # [B, blk]
    logp = max_logit - lse  # [B, blk]

    neg_inf = torch.full_like(logp, float("-inf"))
    confidence = torch.where(eff_mask, logp, neg_inf)  # [B, blk]

    log_thr = math.log(threshold)
    transfer = confidence > log_thr  # [B, blk] bool

    # Apply threshold updates first
    new_ids = torch.where(transfer, x, input_ids)  # [B, blk]

    # Force-at-least-one (graph-friendly): unconditional sparse per-row write, gated to no-op when not needed
    if force_at_least_one:
        row_has_any = eff_mask.any(dim=1)  # [B]
        hit_any = transfer.any(dim=1)  # [B]
        need_force = row_has_any & (~hit_any)  # [B]

        # Best idx per row (even if all -inf -> 0); gated by need_force
        best_idx = confidence.argmax(dim=1)  # [B] long

        # idx2: where to write per row:
        #   - need_force: best_idx
        #   - else: 0 (safe position)
        zero_idx = torch.zeros_like(best_idx)
        idx2 = torch.where(need_force, best_idx, zero_idx)  # [B]

        rows = torch.arange(B, device=dev)  # [B]

        # val2: what to write per row:
        #   - need_force: x[row, best_idx]
        #   - else: new_ids[row, 0] (no-op)
        force_val = x[rows, best_idx]  # [B]
        noop_val = new_ids[rows, zero_idx]  # [B] == new_ids[:,0]
        val2 = torch.where(need_force, force_val, noop_val)  # [B]

        # Single-point update per row (typically much cheaper than [B,blk] one-hot scatter_)
        new_ids[rows, idx2] = val2

    # In-place write-back (preserve storage / better for graph capture)
    input_ids.copy_(new_ids)



if _TRITON_OK:

    @triton.jit
    def k1_top1_and_update_bonus(
        logits_ptr,         
        mask_ptr,
        step_idx,          
        hist_ids_ptr,            
        hist_bonus_ptr,        
        # intermediates
        fused_id_ptr, confidence_ptr, transfer_ptr,
        # const
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
        alpha: tl.constexpr,
        gamma: tl.constexpr,
        log_eps: tl.constexpr,
        lam: tl.constexpr, log_thr: tl.constexpr,
        K: tl.constexpr = 32,
    ):
        pid = tl.program_id(0)
        row_start = pid * V

        lanes = tl.arange(0, K)
        base = pid * K

        # load + decay history bonuses
        hist_ids = tl.load(hist_ids_ptr + base + lanes).to(tl.int32)
        hist_bonus = tl.load(hist_bonus_ptr + base + lanes).to(tl.float32)
        hist_bonus = hist_bonus * gamma

        # stream top1 + logsumexp
        m = tl.full((), -float("inf"), tl.float32)
        s = tl.full((), 0.0, tl.float32)
        best_val = tl.full((), -float("inf"), tl.float32)
        best_idx = tl.full((), 0, tl.int32)

        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            maskv = cols < V
            x = tl.load(logits_ptr + row_start + cols, mask=maskv, other=-float("inf")).to(tl.float32)

            cmax = tl.max(x, axis=0)
            carg = tl.argmax(x, axis=0).to(tl.int32)
            cidx = carg + off

            take = cmax > best_val
            best_val = tl.where(take, cmax, best_val)
            best_idx = tl.where(take, cidx, best_idx)

            m_new = tl.maximum(m, cmax)
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m = m_new

        logZ = m + tl.log(s)
        raw_id = best_idx
        raw_logit = best_val
        raw_logp = tl.minimum(raw_logit - logZ, 0.0)

        # increment for current top1
        inc = tl.exp(alpha * tl.maximum(raw_logp, log_eps))


        hist_ids = tl.where(lanes == step_idx, raw_id, hist_ids)
        hist_bonus = tl.where(lanes == step_idx, inc, hist_bonus)

        # store states + intermediates
        tl.store(hist_ids_ptr + base + lanes, hist_ids)
        tl.store(hist_bonus_ptr + base + lanes, hist_bonus)


        best_val = tl.full((), -float("inf"), tl.float32)
        best_id  = tl.full((), 0, tl.int32)
        sum_delta = tl.full((), 0.0, tl.float32)
        
        logZ_raw = logZ

        for i in range(0, 32):
            mask_i = lanes == i

            kid_i = tl.sum(tl.where(mask_i, hist_ids, 0), axis=0).to(tl.int32)

            valid_i = (kid_i >= 0) & (kid_i < V)
            safe_i = tl.where(valid_i, kid_i, 0)

            # sum duplicate bonuses for same token id
            bonus_sum_i = tl.full((), 0.0, tl.float32)
            has_prev = tl.full((), 0, tl.int32)

            for j in range(0, 32):
                mask_j = lanes == j

                kid_j = tl.sum(tl.where(mask_j, hist_ids, 0), axis=0).to(tl.int32)
                hb_j = tl.sum(tl.where(mask_j, hist_bonus, 0), axis=0).to(tl.float32)
                valid_j = (kid_j >= 0) & (kid_j < V)

                b_j = lam * tl.log(1.0 + tl.maximum(hb_j, 0.0))
                same = valid_i & valid_j & (kid_j == kid_i)
                bonus_sum_i += tl.where(same, b_j, 0.0)

                kid_j = tl.load(hist_ids_ptr + base + j).to(tl.int32)
                valid_j = (kid_j >= 0) & (kid_j < V)
                prev_same = (j < i) & valid_i & valid_j & (kid_j == kid_i)
                has_prev += tl.where(prev_same, 1, 0)
            is_first = valid_i & (has_prev == 0)

            
            # original logit of candidate token
            logit_i = tl.load(logits_ptr + row_start + safe_i, mask=valid_i, other=-float("inf")).to(tl.float32)
            p_i = tl.where(valid_i, tl.exp(logit_i - logZ_raw), 0.0)

            fused_i = logit_i + bonus_sum_i

            take = valid_i & (fused_i > best_val)
            best_val = tl.where(take, fused_i, best_val)
            best_id  = tl.where(take, kid_i, best_id)

            sum_delta += tl.where(is_first, p_i * (tl.exp(bonus_sum_i) - 1.0), 0.0)

        
        logZ_new = logZ_raw + tl.log(1.0 + sum_delta)

        fused_id = best_id
        fused_logit = best_val
        fused_logp = tl.minimum(fused_logit - logZ_new, 0.0)

        # outputs
        active = tl.load(mask_ptr + pid).to(tl.int32) != 0
        neg_inf = tl.full((), -float("inf"), tl.float32)
        conf = tl.where(active, fused_logp, neg_inf)
        transfer = active & (conf > log_thr)

        tl.store(fused_id_ptr + pid, fused_id.to(tl.int32))
        tl.store(confidence_ptr + pid, conf)
        tl.store(transfer_ptr + pid, transfer.to(tl.int32))
        

class CreditState32:
    def __init__(self, BT: int, block_size: int, device: str = "cuda"):
        self.hist_ids = torch.full((BT, block_size), -1, dtype=torch.int32, device=device)
        self.hist_bonus = torch.zeros((BT, block_size), dtype=torch.float32, device=device)

    def reset(self):
        self.hist_ids.fill_(-1)
        self.hist_bonus.zero_()


class CreditDecoding(DllmAlgorithm):

    step_total = 0

    def __init__(self, config):
        super().__init__(config)
        algo_cfg = config.algorithm_config

        self.threshold = float(algo_cfg.get("threshold", 0.95))
        self.gamma = float(algo_cfg.get("credit_decay_gamma", 0.75))
        self.lam = float(algo_cfg.get("credit_fusion_lambda", 0.50)) 
        self.alpha = float(algo_cfg.get("credit_prob_alpha", 0.50))
        self.eps = float(algo_cfg.get("credit_eps", 1e-6))

        self.log_thr = math.log(self.threshold)
        self.log_eps = math.log(self.eps)


    @torch.no_grad()
    def parallel_decoding_streamed(
        self,
        model_runner,
        forward_batch,
        mask_index,
        skip_attn_backend_init: bool = False,
    ):
        out = model_runner.forward(forward_batch, skip_attn_backend_init, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        logits = logits_output.full_logits
        BT, V = logits.shape

        buf = getattr(self, "state", None)
        need_alloc = (buf is None)

        if need_alloc:
            self.state = CreditState32(BT, self.block_size)

            self._tmp_fused_id = torch.empty((BT,), device=logits.device, dtype=torch.int32)
            self._tmp_conf = torch.empty((BT,), device=logits.device, dtype=torch.float32)
            self._tmp_transfer = torch.empty((BT,), device=logits.device, dtype=torch.int8)
            self._tmp_new_i32 = torch.empty((BT,), device=logits.device, dtype=torch.int32)


        logits_c = logits.contiguous()
        mask_i32 = mask_index.to(torch.int32).contiguous()

        grid1 = (BT,)
        k1_top1_and_update_bonus[grid1](
            logits_c, mask_i32, self.step, self.state.hist_ids, self.state.hist_bonus,
            self._tmp_fused_id, self._tmp_conf, self._tmp_transfer,
            V=V, BLOCK_V=2048,
            alpha=self.alpha, gamma=self.gamma, log_eps=self.log_eps,
            lam=self.lam, log_thr=self.log_thr, K=self.block_size,
        )


        _, select_index_max = torch.topk(self._tmp_conf, k=1)
        self._tmp_transfer[select_index_max] = True

        x = torch.where(mask_index, self._tmp_fused_id, forward_batch.input_ids)
        new_i32 = torch.where(self._tmp_transfer.to(torch.bool), x, forward_batch.input_ids)

        forward_batch.input_ids = new_i32.to(forward_batch.input_ids.dtype)

        return logits_output, can_run_cuda_graph


    @torch.no_grad()
    def parallel_decoding(
        self,
        model_runner,
        forward_batch,
        mask_index,
        skip_attn_backend_init: bool = False,
    ):
        out = model_runner.forward(forward_batch, skip_attn_backend_init, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        logits = logits_output.full_logits
        BT, V = logits.shape

        raw_top1 = torch.argmax(logits, dim=-1).to(torch.int32) 
        raw_top1_logit = logits.gather(1, raw_top1.view(BT, 1)).squeeze(1).to(torch.float32) 
        logZ = torch.logsumexp(logits.to(torch.float32), dim=-1) 

        raw = raw_top1.to(torch.int32)


        hit0 = (raw == self.prev0) 
        hit1 = (raw == self.prev1) 
        hit = hit0 | hit1

        self.b0.mul_(self.gamma)
        self.b1.mul_(self.gamma)

        last = self.last_slot.to(torch.int32)
        evict = 1 - last

        
        log_p = raw_top1_logit - logZ
        log_p_clamped = torch.maximum(log_p, torch.full_like(log_p, self.log_eps))
        inc = torch.exp(self.alpha * log_p_clamped)  
        
        use0 = hit0 | (~hit & (evict == 0))
        use1 = hit1 | (~hit & (evict == 1)) 

        self.prev0 = torch.where(~hit & (evict == 0), raw, self.prev0)
        self.prev1 = torch.where(~hit & (evict == 1), raw, self.prev1)

        new_b0 = torch.where(hit0, self.b0 + inc, self.b0)
        new_b1 = torch.where(hit1, self.b1 + inc, self.b1)

        new_b0 = torch.where(~hit & (evict == 0), inc, new_b0)
        new_b1 = torch.where(~hit & (evict == 1), inc, new_b1)

        self.b0 = new_b0
        self.b1 = new_b1

        new_last = torch.where(hit0, torch.zeros_like(last),
                torch.where(hit1, torch.ones_like(last), evict))
        self.last_slot = new_last.to(torch.int8)

        self.bonus = torch.where(hit0 | (~hit & (evict == 0)), self.b0, self.b1)
        self.bonus.add_(inc)

        self.bonus_non = torch.where(hit0 | (~hit & (evict == 0)), self.b1, self.b0)

        raw_non_top1 = torch.where(hit0, self.prev1, torch.where(hit1, self.prev0, torch.full_like(self.prev0, -1)))
        raw_non_top1_logit = logits.gather(1, raw_non_top1.view(BT, 1)).squeeze(1).to(torch.float32)  


        delta = self.lam * torch.log1p(self.bonus)
        p = torch.exp(log_p)  
        pdel = p * delta 

        log_p_non = raw_non_top1_logit - logZ
        delta_non = self.lam * torch.log1p(self.bonus_non) 
        p_non = torch.exp(log_p_non)           
        pdel_non = p_non * delta_non           


        logZ_new = logZ + pdel + pdel_non
        score = (raw_top1_logit + delta) - logZ_new
        score_non = (raw_non_top1_logit + delta_non) - logZ_new  

        fused_id = torch.where(score>score_non, raw_top1, raw_non_top1)
        fused_score = torch.where(score>score_non, score, score_non)

        logits32 = logits.to(torch.float32)

        BT, V = logits32.shape
        rows = torch.arange(BT, device=logits32.device)

        delta_fused = torch.where(fused_id == raw_top1, delta, delta_non).to(torch.float32)

        logits_boost = logits32.clone()
        logits_boost[rows, fused_id.to(torch.long)] += delta_fused


        x = torch.argmax(logits_boost, dim=-1)
        p = torch.squeeze(
            torch.gather(
                F.softmax(logits_boost, dim=-1),
                dim=-1,
                index=torch.unsqueeze(x, -1),
            ),
            -1,
        )
        neg_inf = torch.full_like(p, float("-inf"))
        confidence = torch.where(mask_index, p, neg_inf)
        
        transfer_index = confidence > self.threshold

        _, select_index_max = torch.topk(confidence, k=1)
        transfer_index[select_index_max] = True
 
        new_input_ids = torch.where(transfer_index, x, forward_batch.input_ids)
        forward_batch.input_ids = new_input_ids

        return


    def _triton_ready(self) -> bool:
        if not _TRITON_OK:
            return False
        return True
    
    @torch.no_grad()
    def parallel_decoding_dispatch(
        self,
        model_runner,
        forward_batch,
        mask_index,
        skip_attn_backend_init: bool = False,
    ):
        if self._triton_ready():
            return self.parallel_decoding_streamed(
                model_runner, forward_batch, mask_index, skip_attn_backend_init
            )
        return self.parallel_decoding(
            model_runner, forward_batch, mask_index, skip_attn_backend_init
        )


    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        if _is_npu:
            skip_attn_backend_init = False

        for _ in range(self.block_size):
            self.step = _
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            if _is_npu:
                self.parallel_decoding_dispatch(
                    model_runner, forward_batch, mask_index, skip_attn_backend_init
                )
                skip_attn_backend_init = True
            else:
                self.parallel_decoding_dispatch(
                    model_runner, forward_batch, mask_index, skip_attn_backend_init
                )

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

        if getattr(self, "state", None):
            self.state.reset()

        if _is_npu:
            out = model_runner.forward(
                forward_batch, skip_attn_backend_init, pp_proxy_tensors=None
            )
        else:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph

Algorithm = CreditDecoding
