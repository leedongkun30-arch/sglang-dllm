"""
Unofficial implementation of CreditDecoding: Accelerating Parallel Decoding in
Diffusion Large Language Models with Trace Credits (https://arxiv.org/pdf/2510.06133)
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import logging

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

import torch_npu
import math

experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)

logger = logging.getLogger(__name__)


_TRITON_OK = False
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


if _TRITON_OK:

    # =========================================================
    # Kernel 1) raw top1 + raw logp -> history bonus update
    # =========================================================
    @triton.jit
    def k1_top1_and_update_bonus(
        logits_ptr,              # [BT, V]
        mask_ptr,
        step_idx,            # [BT] int32
        hist_ids_ptr,            # [BT, 32] int32, invalid=-1
        hist_bonus_ptr,          # [BT, 32] fp32
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
    def __init__(self, BT: int, device: str = "cuda"):
        self.hist_ids = torch.full((BT, 32), -1, dtype=torch.int32, device=device)
        self.hist_bonus = torch.zeros((BT, 32), dtype=torch.float32, device=device)

    def reset(self):
        self.hist_ids.fill_(-1)
        self.hist_bonus.zero_()


class CreditDecoding(DllmAlgorithm):

    step_total = 0

    def __init__(self, config):
        super().__init__(config)
        algo_cfg = config.algorithm_config

        self.threshold = float(algo_cfg.get("threshold", 0.95))
        self.gamma = float(algo_cfg.get("credit_decay_gamma", 0.65))
        self.lam = float(algo_cfg.get("credit_fusion_lambda", 0.70)) #0.70
        self.alpha = float(algo_cfg.get("credit_prob_alpha", 0.90))
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
            self.state = CreditState32(BT)

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
            lam=self.lam, log_thr=self.log_thr,
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

        raw_top1 = torch.argmax(logits, dim=-1).to(torch.int32)  # (BT,)
        raw_top1_logit = logits.gather(1, raw_top1.view(BT, 1)).squeeze(1).to(torch.float32)  # (BT,)
        logZ = torch.logsumexp(logits.to(torch.float32), dim=-1)  # (BT,)

        raw = raw_top1.to(torch.int32)


        hit0 = (raw == self.prev0)          # (BT,) bool
        hit1 = (raw == self.prev1)          # (BT,) bool
        hit = hit0 | hit1

        self.b0.mul_(self.gamma)
        self.b1.mul_(self.gamma)

        last = self.last_slot.to(torch.int32)
        evict = 1 - last                    # (BT,) int32

        
        log_p = raw_top1_logit - logZ                 # (BT,)
        log_p_clamped = torch.maximum(log_p, torch.full_like(log_p, self.log_eps))
        inc = torch.exp(self.alpha * log_p_clamped)   # (BT,)  
        
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

        self.bonus = torch.where(hit0 | (~hit & (evict == 0)), self.b0, self.b1)  # (BT,)
        self.bonus.add_(inc)

        self.bonus_non = torch.where(hit0 | (~hit & (evict == 0)), self.b1, self.b0)  # (BT,)

        raw_non_top1 = torch.where(hit0, self.prev1, torch.where(hit1, self.prev0, torch.full_like(self.prev0, -1)))
        raw_non_top1_logit = logits.gather(1, raw_non_top1.view(BT, 1)).squeeze(1).to(torch.float32)  # (BT,)


        delta = self.lam * torch.log1p(self.bonus)     # (BT,)
        p = torch.exp(log_p)              # (BT,)  BT exp only
        pdel = p * delta             # (BT,)  BT exp only

        log_p_non = raw_non_top1_logit - logZ
        delta_non = self.lam * torch.log1p(self.bonus_non)     # (BT,)
        p_non = torch.exp(log_p_non)              # (BT,)  BT exp only
        pdel_non = p_non * delta_non             # (BT,)  BT exp only


        logZ_new = logZ + pdel + pdel_non
        score = (raw_top1_logit + delta) - logZ_new   # log(p_boost)
        score_non = (raw_non_top1_logit + delta_non) - logZ_new   # log(p_boost)

        fused_id = torch.where(score>score_non, raw_top1, raw_non_top1)
        fused_score = torch.where(score>score_non, score, score_non)

        logits32 = logits.to(torch.float32)

        BT, V = logits32.shape
        rows = torch.arange(BT, device=logits32.device)

        # delta_fused: fused_id가 raw면 delta, non이면 delta_non
        delta_fused = torch.where(fused_id == raw_top1, delta, delta_non).to(torch.float32)

        # logits_boost: logits32 복사해서 fused_id 위치에만 delta 더함
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
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        mask_index = forward_batch.input_ids == self.mask_id
        start = len(forward_batch.input_ids) - torch.sum(mask_index).item()

        skip_attn_backend_init = False
        
        
        for _ in range(self.block_size):
            self.step = _
            mask_index = forward_batch.input_ids == self.mask_id
            if not torch.any(mask_index):
                break
            self.parallel_decoding_dispatch(
                model_runner, forward_batch, mask_index, skip_attn_backend_init
            )
            skip_attn_backend_init = True

        CreditDecoding.step_total += self.step
        self.state.reset()

        logger.info(f"paralle decoding: forward times = {CreditDecoding.step_total}")
        out = model_runner.forward(forward_batch, skip_attn_backend_init, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids = forward_batch.input_ids[start:]
        return logits_output, next_token_ids, can_run_cuda_graph

Algorithm = CreditDecoding

