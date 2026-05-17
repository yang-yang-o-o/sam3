# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


def _fa3_compute_dtype(device: torch.device) -> torch.dtype:
    """Hopper (sm90+) supports float8; Ampere/Ada (e.g. RTX 4090) need fp16/bf16."""
    if device.type != "cuda":
        return torch.bfloat16
    major = torch.cuda.get_device_properties(device).major
    if major >= 9:
        return torch.float8_e4m3fn
    return torch.bfloat16


def _fa3_unified_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.dtype:
    """FA3 requires q/k/v to share one dtype (autocast + RoPE can mix bf16 and fp32)."""
    dtypes = {q.dtype, k.dtype, v.dtype}
    if len(dtypes) == 1 and q.dtype in (torch.float16, torch.bfloat16):
        return q.dtype
    if torch.bfloat16 in dtypes:
        return torch.bfloat16
    if torch.float16 in dtypes:
        return torch.float16
    return _fa3_compute_dtype(q.device)


def flash_attn_func(q, k, v):
    out_dtype = q.dtype
    compute_dtype = _fa3_unified_dtype(q, k, v)
    if q.dtype != compute_dtype or k.dtype != compute_dtype or v.dtype != compute_dtype:
        q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
    return flash_attn_func_op(q, k, v).to(out_dtype)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    # output needs to be bfloat16, not float8!
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
