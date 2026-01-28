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

    @triton.jit
    def stream_fuse_kernel(
        logits_ptr,                 # [BT, V]
        input_ids_ptr,              # [BT]
        mask_ptr,                   # [BT] bool
        prev0_ptr, prev1_ptr,       # [BT] int32
        b0_ptr, b1_ptr,             # [BT] fp32
        last_ptr,                   # [BT] int8
        # outputs
        fused_id_ptr,               # [BT] int32
        confidence_ptr,             # [BT] fp32
        transfer_ptr,               # [BT] int1/bool (we'll store int8)
        new_input_ids_i32_ptr,      # [BT] int32
        # const params
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
        alpha: tl.constexpr, gamma: tl.constexpr, lam: tl.constexpr, log_eps: tl.constexpr, log_thr: tl.constexpr,
    ):
        pid = tl.program_id(0)   # row id
        row_start = pid * V

        # ----- streaming argmax + logsumexp (online) -----
        m = tl.full((), -float("inf"), tl.float32)
        s = tl.full((), 0.0, tl.float32)

        best_val = tl.full((), -float("inf"), tl.float32)
        best_idx = tl.full((), 0, tl.int32)

        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            maskv = cols < V
            x = tl.load(logits_ptr + row_start + cols, mask=maskv, other=-float("inf")).to(tl.float32)

            cmax = tl.max(x, axis=0)

            # tie -> smaller index
            cidx = tl.argmax(x, axis=0).to(tl.int32) + off
            take = cmax > best_val
            best_val = tl.where(take, cmax, best_val)
            best_idx = tl.where(take, cidx, best_idx)

            m_new = tl.maximum(m, cmax)
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m = m_new

        logZ = m + tl.log(s)

        raw = best_idx
        raw_top1_logit = best_val

        # ----- load state -----
        prev0 = tl.load(prev0_ptr + pid).to(tl.int32)
        prev1 = tl.load(prev1_ptr + pid).to(tl.int32)
        b0 = tl.load(b0_ptr + pid).to(tl.float32) * gamma
        b1 = tl.load(b1_ptr + pid).to(tl.float32) * gamma
        last = tl.load(last_ptr + pid).to(tl.int32)

        hit0 = raw == prev0
        hit1 = raw == prev1
        hit = hit0 | hit1
        evict = 1 - last

        # ----- inc -----
        log_p = raw_top1_logit - logZ
        log_p_clamped = tl.maximum(log_p, log_eps)
        inc = tl.exp(alpha * log_p_clamped)

        # prev update (branchless)
        prev0 = tl.where((~hit) & (evict == 0), raw, prev0)
        prev1 = tl.where((~hit) & (evict == 1), raw, prev1)

        # b update
        new_b0 = tl.where(hit0, b0 + inc, b0)
        new_b1 = tl.where(hit1, b1 + inc, b1)
        new_b0 = tl.where((~hit) & (evict == 0), inc, new_b0)
        new_b1 = tl.where((~hit) & (evict == 1), inc, new_b1)
        b0, b1 = new_b0, new_b1

        # last_slot update
        last = tl.where(hit0, 0, tl.where(hit1, 1, evict))

        # bonus / bonus_non
        use0 = hit0 | ((~hit) & (evict == 0))   # select slot 0
        bonus = tl.where(use0, b0, b1) + inc
        bonus_non = tl.where(use0, b1, b0)

        raw_non = tl.where(hit0, prev1, tl.where(hit1, prev0, -1))

        raw_non_safe = tl.where(raw_non < 0, V - 1, raw_non)

        non_logit = tl.load(logits_ptr + row_start + raw_non_safe).to(tl.float32)

        delta = lam * tl.log(1.0+bonus)
        p = tl.exp(log_p)
        pdel = p * delta

        log_p_non = non_logit - logZ
        delta_non = lam * tl.log(1.0+bonus_non)
        p_non = tl.exp(log_p_non)
        pdel_non = p_non * delta_non

        logZ_new = logZ + pdel + pdel_non
        score = (raw_top1_logit + delta) - logZ_new
        score_non = (non_logit + delta_non) - logZ_new

        fused_id = tl.where(score > score_non, raw, raw_non)
        fused_score = tl.where(score > score_non, score, score_non)

        mask_index = tl.load(mask_ptr + pid).to(tl.int1)
        neg_inf = tl.full((), -float("inf"), tl.float32)
        confidence = tl.where(mask_index, fused_score, neg_inf)

        # transfer flag (threshold)
        transfer = confidence > log_thr

        # new_input_ids (int32)
        in_id_i32 = tl.load(input_ids_ptr + pid).to(tl.int32)
        x = tl.where(mask_index, fused_id, in_id_i32)
        new_id = tl.where(transfer, x, in_id_i32)

        # ----- store state + outputs -----
        tl.store(prev0_ptr + pid, prev0)
        tl.store(prev1_ptr + pid, prev1)
        tl.store(b0_ptr + pid, b0)
        tl.store(b1_ptr + pid, b1)
        tl.store(last_ptr + pid, last.to(tl.int8))

        tl.store(fused_id_ptr + pid, fused_id)
        tl.store(confidence_ptr + pid, confidence)
        tl.store(transfer_ptr + pid, transfer.to(tl.int8))
        tl.store(new_input_ids_i32_ptr + pid, new_id)

    @triton.jit
    def force_one_max_kernel(conf_ptr, transfer_ptr, BT: tl.constexpr):
        offs = tl.arange(0, BT)
        conf = tl.load(conf_ptr + offs).to(tl.float32)

        m = tl.max(conf, axis=0)
        is_max = conf == m
        big = tl.full((BT,), 2**30, tl.int32)
        idx = offs.to(tl.int32)
        idx_masked = tl.where(is_max, idx, big)
        midx = tl.min(idx_masked, axis=0)

        tl.store(transfer_ptr + midx, tl.full((), 1, tl.int8))

class CreditDecoding(DllmAlgorithm):

    def __init__(self, config):
        super().__init__(config)
        algo_cfg = config.algorithm_config

        self.threshold = float(algo_cfg.get("threshold", 0.95))
        self.gamma = float(algo_cfg.get("credit_decay_gamma", 0.65))
        self.lam = float(algo_cfg.get("credit_fusion_lambda", 0.70))
        self.alpha = float(algo_cfg.get("credit_prob_alpha", 0.50))
        self.eps = float(algo_cfg.get("credit_eps", 1e-6))

        self.log_thr = math.log(self.threshold)
        self.log_eps = math.log(self.eps)

        self.cache_inited = False
        self.prev_top1 = None      # (BT,) int32
        self.prev_top2 = None      # (BT,) int32
        self.bonus = None          # (BT,) float32
        self.bonus2 = None

        # ---- triton temp buffers ----
        self._tmp_inited = False
        self._tmp_top1 = None          # int32 (BT,)
        self._tmp_top1_logit = None    # float32 (BT,)
        self._tmp_logZ = None          # float32 (BT,)
        self._tmp_conf = None          # float32 (BT,)
        self._tmp_fused_id = None
        self._temp_fused_score = None

        self._tmp_tile_inited = False
        self._tmp_tile_max = None
        self._tmp_tile_sumexp = None
        self._tmp_tile_best_val = None
        self._tmp_tile_best_idx = None
        self._tmp_ntiles = None
        self._tmp_block_v = None

    def _ensure_tile_tmp(self, device, BT: int, V: int, BLOCK_V: int):
        NTILES = triton.cdiv(V, BLOCK_V)
        need = (not self._tmp_tile_inited) or (self._tmp_tile_max is None) or (self._tmp_ntiles != NTILES) \
            or (self._tmp_tile_max.numel() != BT * NTILES) or (self._tmp_tile_max.device != device)
        if need:
            self._tmp_tile_inited = True
            self._tmp_ntiles = NTILES
            self._tmp_block_v = BLOCK_V
            n = BT * NTILES
            self._tmp_tile_max = torch.empty((n,), device=device, dtype=torch.float32)
            self._tmp_tile_sumexp = torch.empty((n,), device=device, dtype=torch.float32)
            self._tmp_tile_best_val = torch.empty((n,), device=device, dtype=torch.float32)
            self._tmp_tile_best_idx = torch.empty((n,), device=device, dtype=torch.int32)
        return NTILES


    def _ensure_tmp(self, device, BT: int):
        if (not self._tmp_inited) or (self._tmp_top1 is None) or (self._tmp_top1.numel() != BT) or (self._tmp_top1.device != device):
            self._tmp_inited = True
            self._tmp_top1 = torch.empty((BT,), device=device, dtype=torch.int32)
            self._tmp_top1_logit = torch.empty((BT,), device=device, dtype=torch.float32)
            self._tmp_logZ = torch.empty((BT,), device=device, dtype=torch.float32)
            self._tmp_conf = torch.empty((BT,), device=device, dtype=torch.float32)
            self._tmp_fused_id = torch.empty((BT,), device=device, dtype=torch.int32)
            self._temp_fused_score = torch.empty((BT,), device=device, dtype=torch.int32)

    def init_cache(self, device, BT: int):
        if (not self.cache_inited) or (self.prev_top1 is None) or (self.prev_top1.numel() != BT) or (self.prev_top1.device != device):
            self.cache_inited = True
            self.prev0 = torch.full((BT,), -1, device=device, dtype=torch.int32)
            self.prev1 = torch.full((BT,), -1, device=device, dtype=torch.int32)
            self.b0 = torch.zeros((BT,), device=device, dtype=torch.float32)
            self.b1 = torch.zeros((BT,), device=device, dtype=torch.float32)
            self.last_slot = torch.zeros((BT,), device=device, dtype=torch.int8)  # 0/1
        else:
            self.prev0.fill_(-1)
            self.prev1.fill_(-1)
            self.b0.zero_()
            self.b1.zero_()
            self.last_slot.zero_()

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

        buf = getattr(self, "_tmp_fused_id", None)
        need_alloc = (buf is None) or (buf.numel() != BT) or (buf.device != logits.device)

        if need_alloc:
            self._tmp_fused_id = torch.empty((BT,), device=logits.device, dtype=torch.int32)
            self._tmp_conf = torch.empty((BT,), device=logits.device, dtype=torch.float32)
            self._tmp_transfer = torch.empty((BT,), device=logits.device, dtype=torch.int8)
            self._tmp_new_i32 = torch.empty((BT,), device=logits.device, dtype=torch.int32)

        BLOCK_V = 2048

        if not logits.is_contiguous():
            logits = logits.contiguous()

        stream_fuse_kernel[(BT,)](
            logits,
            forward_batch.input_ids,
            mask_index,
            self.prev0, self.prev1, self.b0, self.b1, self.last_slot,
            self._tmp_fused_id,
            self._tmp_conf,
            self._tmp_transfer,
            self._tmp_new_i32,
            V=V,
            BLOCK_V=BLOCK_V,
            alpha=self.alpha, gamma=self.gamma, lam=self.lam,
            log_eps=self.log_eps, log_thr=self.log_thr,
        )


        _, select_index_max = torch.topk(self._tmp_conf, k=1)
        self._tmp_transfer[select_index_max] = True

        transfer = self._tmp_transfer.to(torch.bool)
        input_i32 = forward_batch.input_ids.to(torch.int32)
        x = torch.where(mask_index, self._tmp_fused_id, input_i32)
        new_i32 = torch.where(transfer, x, input_i32)

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

        neg_inf = torch.full_like(fused_score, float("-inf"))
        confidence = torch.where(mask_index, fused_score, neg_inf)  # (BT,)

        transfer_index = confidence > self.log_thr

        _, select_index_max = torch.topk(confidence, k=1)
        transfer_index[select_index_max] = True

        x = torch.where(mask_index, fused_id, forward_batch.input_ids.to(torch.int32))
        new_input_ids = torch.where(transfer_index, x, forward_batch.input_ids.to(torch.int32))
        forward_batch.input_ids = new_input_ids

        return logits_output, can_run_cuda_graph


    def _triton_ready(self) -> bool:
        if not _TRITON_OK:
            return False
        if not hasattr(self, "parallel_decoding_streamed"):
            return False
        return ("stream_fuse_kernel" in globals()) 
    
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

        self.init_cache(forward_batch.input_ids.device, forward_batch.input_ids.shape[-1])
        
        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if not torch.any(mask_index):
                break
            self.parallel_decoding_dispatch(
                model_runner, forward_batch, mask_index, skip_attn_backend_init
            )
            skip_attn_backend_init = True

        out = model_runner.forward(forward_batch, skip_attn_backend_init, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids = forward_batch.input_ids[start:]
        return logits_output, next_token_ids, can_run_cuda_graph

Algorithm = CreditDecoding
