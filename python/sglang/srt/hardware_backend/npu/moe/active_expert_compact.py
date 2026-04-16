from __future__ import annotations

from typing import Optional, Tuple

import torch


def _compact_active_expert_bank(
    topk_ids: torch.Tensor,
    *expert_tensors: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, int, Tuple[Optional[torch.Tensor], ...], torch.Tensor]:
    """Compact the local expert bank to only the experts referenced by topk_ids.

    topk_ids are assumed to already be local expert ids (or -1 for invalid).
    Returns remapped ids, active_count, compacted tensors, and active_idx.
    """
    if len(expert_tensors) == 0 or expert_tensors[0] is None:
        raise ValueError("expert_tensors must include at least one non-None tensor")

    num_experts = expert_tensors[0].shape[0]
    active_idx = torch.unique(topk_ids[topk_ids >= 0].to(torch.int64), sorted=True)
    active_count = int(active_idx.numel())

    if active_count == 0 or active_count == num_experts:
        return topk_ids, num_experts, expert_tensors, active_idx

    remap = torch.full((num_experts,), -1, dtype=torch.int32, device=topk_ids.device)
    remap[active_idx] = torch.arange(active_count, dtype=torch.int32, device=topk_ids.device)

    safe_topk_ids = torch.where(topk_ids >= 0, topk_ids, torch.zeros_like(topk_ids))
    remapped_topk_ids = torch.where(
        topk_ids >= 0,
        remap[safe_topk_ids.to(torch.int64)],
        topk_ids,
    ).to(torch.int32)

    compacted = []
    for tensor in expert_tensors:
        if tensor is None:
            compacted.append(None)
        else:
            compacted.append(tensor.index_select(0, active_idx))

    return remapped_topk_ids, active_count, tuple(compacted), active_idx


def enable_npu_active_expert_compact() -> None:
    """Enable active local expert compaction for NPU FusedMoE paths.

    This monkeypatch targets the standard NPU FusedMoE launch path where
    local expert banks are otherwise passed to grouped_matmul uncompressed.
    """

    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.quantization import fused_moe_method_npu as npu_moe_mod
    from sglang.srt.layers.quantization import unquant as unquant_mod
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    if getattr(npu_moe_mod, "_sglang_active_expert_compact_enabled", False):
        return

    orig_npu_fused_experts = npu_moe_mod.npu_fused_experts
    orig_forward_npu = unquant_mod.UnquantizedFusedMoEMethod.forward_npu

    def patched_npu_fused_experts(
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        **kwargs,
    ):
        w13_offset = kwargs.get("w13_offset", None)
        w2_offset = kwargs.get("w2_offset", None)

        topk_ids, num_experts, compacted, _ = _compact_active_expert_bank(
            topk_ids,
            w13,
            w13_scale,
            w2,
            w2_scale,
            w13_offset,
            w2_offset,
        )
        w13, w13_scale, w2, w2_scale, w13_offset, w2_offset = compacted
        return orig_npu_fused_experts(
            hidden_states=hidden_states,
            w13=w13,
            w13_scale=w13_scale,
            w2=w2,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            w13_offset=w13_offset,
            w2_offset=w2_offset,
            use_wna16=kwargs.get("use_wna16", False),
        )

    def patched_forward_npu(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)

        if layer.w13_weight.shape[-1] == layer.hidden_size:
            w13 = layer.w13_weight.transpose(1, 2)
            w2 = layer.w2_weight.transpose(1, 2)
        else:
            w13 = layer.w13_weight
            w2 = layer.w2_weight

        topk_ids, num_experts, compacted, _ = _compact_active_expert_bank(topk_ids, w13, w2)
        w13, w2 = compacted

        row_idx_len = num_tokens * layer.top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
            .view(layer.top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w13],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        if self.moe_runner_config.activation == "silu":
            hidden_states = torch_npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul
            hidden_states = GeluAndMul()(hidden_states)

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
        return StandardCombineInput(hidden_states=final_hidden_states)

    npu_moe_mod.npu_fused_experts = patched_npu_fused_experts
    unquant_mod.UnquantizedFusedMoEMethod.forward_npu = patched_forward_npu
    npu_moe_mod._sglang_active_expert_compact_enabled = True
