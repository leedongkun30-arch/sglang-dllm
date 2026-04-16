from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


_COMPACT_EXPERT_WS: dict = {}
_COMPACT_TOPK_WS: dict = {}


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _in_capture_mode() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        return bool(get_is_capture_mode())
    except Exception:
        return False


def _get_static_ws(key: tuple, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ws = _COMPACT_EXPERT_WS.get(key)
    if ws is None or tuple(ws.shape) != tuple(shape) or ws.dtype != dtype or ws.device != device:
        ws = torch.empty(shape, dtype=dtype, device=device)
        _COMPACT_EXPERT_WS[key] = ws
    return ws


def _get_topk_ws(key: tuple, shape: tuple, dtype: torch.dtype, device: torch.device, fill_value: int = -1) -> torch.Tensor:
    ws = _COMPACT_TOPK_WS.get(key)
    if ws is None or tuple(ws.shape) != tuple(shape) or ws.dtype != dtype or ws.device != device:
        ws = torch.full(shape, fill_value=fill_value, dtype=dtype, device=device)
        _COMPACT_TOPK_WS[key] = ws
    return ws


def _compact_exact(topk_ids: torch.Tensor, *expert_tensors: Optional[torch.Tensor]):
    num_experts = expert_tensors[0].shape[0]
    active_idx = torch.unique(topk_ids[topk_ids >= 0].to(torch.int64), sorted=True)
    active_count = int(active_idx.numel())
    if active_count == 0 or active_count == num_experts:
        return topk_ids, num_experts, expert_tensors, active_idx

    remap = torch.full((num_experts,), -1, dtype=torch.int32, device=topk_ids.device)
    remap[active_idx] = torch.arange(active_count, dtype=torch.int32, device=topk_ids.device)
    safe_ids = torch.where(topk_ids >= 0, topk_ids, torch.zeros_like(topk_ids))
    remapped_topk_ids = torch.where(topk_ids >= 0, remap[safe_ids.to(torch.int64)], topk_ids).to(torch.int32)

    compacted = []
    for tensor in expert_tensors:
        if tensor is None:
            compacted.append(None)
        else:
            compacted.append(tensor.index_select(0, active_idx))
    return remapped_topk_ids, active_count, tuple(compacted), active_idx


def _compact_bucket_capture_safe(topk_ids: torch.Tensor, bucket_size: int, *expert_tensors: Optional[torch.Tensor]):
    base_tensor = next(t for t in expert_tensors if t is not None)
    num_experts = int(base_tensor.shape[0])
    bucket_size = min(bucket_size, num_experts)

    safe_ids = torch.where(topk_ids >= 0, topk_ids, torch.zeros_like(topk_ids)).to(torch.int64)
    active_mask = torch.zeros((num_experts,), dtype=torch.int32, device=topk_ids.device)
    active_mask.scatter_(0, safe_ids.reshape(-1), 1)

    base = torch.arange(num_experts, device=topk_ids.device, dtype=torch.int32)
    scores = active_mask * (num_experts + 1) - base
    padded_idx = torch.argsort(scores, descending=True)[:bucket_size].to(torch.int64)

    remap = torch.full((num_experts,), -1, dtype=torch.int32, device=topk_ids.device)
    remap[padded_idx] = torch.arange(bucket_size, dtype=torch.int32, device=topk_ids.device)

    remap_ws = _get_topk_ws(
        ("topk", tuple(topk_ids.shape), str(topk_ids.device), str(topk_ids.dtype), bucket_size),
        tuple(topk_ids.shape),
        torch.int32,
        topk_ids.device,
        fill_value=-1,
    )
    remap_ws.copy_(torch.where(topk_ids >= 0, remap[safe_ids], topk_ids.to(torch.int32)))

    compacted = []
    for idx, tensor in enumerate(expert_tensors):
        if tensor is None:
            compacted.append(None)
            continue
        out = _get_static_ws(
            ("expert", idx, tuple(tensor.shape[1:]), str(tensor.device), str(tensor.dtype), bucket_size),
            (bucket_size, *tensor.shape[1:]),
            tensor.dtype,
            tensor.device,
        )
        torch.index_select(tensor, 0, padded_idx, out=out)
        compacted.append(out)

    return remap_ws, bucket_size, tuple(compacted), padded_idx


def _compact_active_expert_bank(topk_ids: torch.Tensor, *expert_tensors: Optional[torch.Tensor]):
    bucket_size = _get_env_int("SGLANG_NPU_ACTIVE_EXPERT_BUCKET", 0)
    if _in_capture_mode() and bucket_size > 0:
        return _compact_bucket_capture_safe(topk_ids, bucket_size, *expert_tensors)
    return _compact_exact(topk_ids, *expert_tensors)


def enable_npu_active_expert_compact_capture_safe() -> None:
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.quantization import fused_moe_method_npu as npu_moe_mod
    from sglang.srt.layers.quantization import unquant as unquant_mod
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    if getattr(npu_moe_mod, "_sglang_active_expert_compact_capture_safe_enabled", False):
        return

    orig_npu_fused_experts = npu_moe_mod.npu_fused_experts

    def patched_npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, top_k, **kwargs):
        w13_offset = kwargs.get("w13_offset", None)
        w2_offset = kwargs.get("w2_offset", None)
        topk_ids, num_experts, compacted, _ = _compact_active_expert_bank(
            topk_ids, w13, w13_scale, w2, w2_scale, w13_offset, w2_offset
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
    npu_moe_mod._sglang_active_expert_compact_capture_safe_enabled = True
