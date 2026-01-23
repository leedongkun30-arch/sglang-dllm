"""
Unofficial implementation of CreditDecoing: Accelerating Parallel Decoding in
Diffusion Large Language Models with Trace Credits (https://arxiv.org/pdf/2510.06133)
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class CreditDecoding(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)

        algo_cfg = config.algorithm_config
      
        self.threshold = config.algorithm_config.get("threshold", 0.95)

        self.gamma = config.algorithm_config.get("credit_decay_gamma", 0.65)      # decay factor gamma
        self.lam = config.algorithm_config.get("credit_fusion_lambda", 0.70)    # fusion strength lambda
        self.alpha = config.algorithm_config.get("credit_prob_alpha", 0.50)       # concave exponenet alpha, (<1 boosts low p)

        self.credit_slots = int(algo_cfg.get("credit_slots", 8))
        self.eps =  int(algo_cfg.get("credit_eps", 1e-6))

        self.cache_inited = False

    def init_cache(self, device, batch_size):
        if not self.cache_inited:
            self.cache_inited = True
            self.idx = torch.full((batch_size, self.block_size, self.credit_slots), -1, dtype=torch.long, device=device)
            self.val = torch.zeros((batch_size, self.block_size, self.credit_slots), dtype=torch.float32, device=device)
            self._rows_cache = torch.arange(self.block_size, device=device)
        else:
            self.idx.fill_(-1)
            self.val.fill_(0)
    
    @torch.no_grad()
    def _credit_update_top1_only(
      self,
      batch_id,
      top1_idx: torch.Tensor,
      top1_p: torch.Tensor,
    ) -> None:
        self.val[batch_id] *= self.gamma

        inc = (top1_p.clamp_min(self.eps).pow(self.alpha)).to(self.val[batch_id].dtype)

        match = (self.idx[batch_id] == top1_idx.unsqueeze(1))
        has = match.any(dim=1)
        match_slot = match.to(torch.int64).argmax(dim=1)

        empty = (self.idx[batch_id] < 0)
        has_empty = empty.any(dim=1)
        empty_slot = empty.to(torch.int64).argmax(dim=1)
        min_slot = self.val[batch_id].argmin(dim=1)

        ins_slot = torch.where(has_empty, empty_slot, min_slot)
        chosen_slot = torch.where(has, match_slot, ins_slot)

        rows = self._rows_cache
        self.val[batch_id][rows, chosen_slot] += inc

        do_insert = (~has)
        self.idx[batch_id][rows, chosen_slot] = torch.where(do_insert, top1_idx, self.idx[batch_id][rows, chosen_slot])

    @torch.no_grad()
    def _fused_top1_prob_sparse_from_raw(
        self,
        batch_id,
        logits: torch.Tensor,
        raw_top1_id: torch.Tensor,
        raw_top1_logit: torch.Tensor,
        logZ: torch.Tensor,
    ):
        valid = (self.idx[batch_id] >= 0)
        idx = torch.where(valid, self.idx[batch_id], 0)

        lk = logits.gather(1, idx)
        lk = lk.masked_fill(~valid, torch.finfo(lk.dtype).min)

        delta = self.lam * torch.log1p(self.val[batch_id])
        delta = delta.masked_fill(~valid, 0.0)

        lk_fused = lk + delta

        best_k_logits, best_k_idx = lk_fused.max(dim=1)
        best_k_id = idx.gather(1, best_k_idx.unsqueeze(1)).squeeze(1)

        onehot = (idx == raw_top1_id.unsqueeze(1)).squeeze(1)
        raw_delta = (delta * onehot.to(delta.dtype)).sum(dim=1)
        raw_top1_fused_logit = raw_top1_logit + raw_delta

        choose_k = best_k_logits > raw_top1_fused_logit
        fused_top1_id = torch.where(choose_k, best_k_id, raw_top1_id)
        fused_top1_logit = torch.where(choose_k, best_k_logits, raw_top1_fused_logit)

        logZ_u = logZ.unsqueeze(1)

        sum_pk = torch.exp(lk - logZ_u).masked_fill(~valid, 0.0).sum(dim=1)
        sum_pk_boost = torch.exp(lk_fused - logZ_u).masked_fill(~valid, 0.0).sum(dim=1)

        corr = (1.0 - sum_pk + sum_pk_boost).clamp_min(self.eps)
        logZ_fused = logZ + torch.log(corr)

        fused_p = torch.exp(fused_top1_logit - logZ_fused)

        return fused_top1_id, fused_p
      

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
        
        self.init_cache(forward_batch.input_ids.device, batch_size)
        
        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            
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

                raw_top1_full = torch.argmax(curr_logits, dim=-1)
                raw_top1_logit_full = curr_logits.gather(1, raw_top1_full.unsqueeze(1)).squeeze(1)
                logZ_full = torch.logsumexp(curr_logits, dim=-1)
                p_raw_full = torch.exp(raw_top1_logit_full - logZ_full)

                self._credit_update_top1_only(
                    batch_id=batch_id,
                    top1_idx=raw_top1_full,
                    top1_p=p_raw_full,
                )

                x_credit_full, p_credit_full = self._fused_top1_prob_sparse_from_raw(
                    batch_id=batch_id,
                    logits=curr_logits,
                    raw_top1_id=raw_top1_full,
                    raw_top1_logit=raw_top1_logit_full,
                    logZ=logZ_full,
                )

                p = torch.maximum(p_raw_full, p_credit_full)
                x = torch.maximum(p_credit_full > p_raw_full, x_credit_full, raw_top1_full)

                confidence = torch.where(block_mask_index, p, -np.inf)
                x = torch.where(block_mask_index, x, block_input_ids)

                transfer_index = confidence > self.threshold
                _, sel = torch.topk(confidence, k=1)
                transfer_index.scatter_(0, sel, True)

                block_input_ids[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_grap



Algorithm = CreditDecoding
