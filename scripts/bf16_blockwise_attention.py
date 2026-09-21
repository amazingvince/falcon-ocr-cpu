#!/usr/bin/env python3
"""Readable independent PyTorch oracle for observed Flex BF16 block arithmetic.

This does not call FlexAttention or its generated kernels. Tile constants come from
the saved compiled-source inventory, and sparse block ordering comes from the real
BlockMask. PyTorch CUDA matrix operations retain BF16 operands and FP32 outputs.
The oracle intentionally remains independent in reduction and exp2 implementation.
"""
import math
import types

import torch


def _blocks(mask, row, full, n, block_n=64):
    count = getattr(mask, "full_kv_num_blocks" if full else "kv_num_blocks")
    index = getattr(mask, "full_kv_indices" if full else "kv_indices")
    if count is None:
        return []
    size = int(count[0, 0, row])
    sparse_size = mask.BLOCK_SIZE[1]
    ids = index[0, 0, row, :size].tolist()
    return [(s * sparse_size + t * block_n, full) for s in ids
            for t in range(sparse_size // block_n)][:max(math.ceil(n / block_n), 1)]


def _update(q, k, v, allowed, blocks):
    heads, rows, dim = q.shape
    maximum = torch.full((heads, rows), -torch.inf, device=q.device)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros((heads, rows, dim), device=q.device)
    for start, full in blocks:
        stop = min(start + 64, k.shape[1])
        if stop <= start:
            continue
        kb, vb = k[:, start:stop], v[:, start:stop]
        # tl.dot(BF16, BF16) accumulates in FP32; never round QK to BF16.
        scores = torch.bmm(q, kb.transpose(1, 2), out_dtype=torch.float32) * 0.125
        if not full:
            scores = scores.masked_fill(~allowed[:, start:stop][None], -torch.inf)
        scores = scores * 1.44269504
        new_maximum = torch.maximum(maximum, scores.amax(-1))
        safe_maximum = torch.where(new_maximum == -torch.inf, 0, new_maximum)
        rescale = torch.exp2(maximum - safe_maximum)
        unnormalized = torch.exp2(scores - safe_maximum[..., None])
        denominator = denominator * rescale + unnormalized.sum(-1)
        accumulator = accumulator * rescale[..., None]
        accumulator = accumulator + torch.bmm(unnormalized.to(torch.bfloat16), vb, out_dtype=torch.float32)
        maximum = new_maximum
    return accumulator, denominator, maximum


def blockwise(q, k, v, *, block_mask, return_aux=None):
    del return_aux
    assert q.shape[0] == k.shape[0] == v.shape[0] == 1
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 64
    n, rows = k.shape[2], q.shape[2]
    qi = torch.arange(rows, device=q.device)[:, None]
    ki = torch.arange(n, device=q.device)[None, :]
    zero = torch.zeros((), dtype=torch.long, device=q.device)
    allowed = block_mask.mask_mod(zero, zero, qi, ki)
    output, lse = [], []
    if rows == 1:
        # Actual BF16 decode source: BLOCK_M=16, BLOCK_N=64, SPLIT_KV=16.
        partial = _blocks(block_mask, 0, False, n)
        full = _blocks(block_mask, 0, True, n)
        tile = math.ceil(math.ceil(n / 16) / 64)
        splits = []
        for split in range(16):
            lo, hi = split * tile, (split + 1) * tile
            flo, fhi = (15 - split) * tile, (16 - split) * tile
            splits.append(_update(q[0], k[0], v[0], allowed, partial[lo:hi] + full[flo:fhi]))
        acc, den, maximum = [torch.stack([x[i] for x in splits]) for i in range(3)]
        global_maximum = maximum.amax(0)
        scale = torch.exp2(maximum - global_maximum[None])
        combined_den = (den * scale).sum(0)
        combined_acc = (acc * scale[..., None]).sum(0)
        raw = (combined_acc / combined_den[..., None]).to(torch.bfloat16)
        natural_lse = (torch.log2(combined_den) + global_maximum) * 0.6931471805599453
        return raw[None], types.SimpleNamespace(lse=natural_lse[None])
    # Actual BF16 prefill source: BLOCK_M=128, BLOCK_N=64, partial then full.
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        row = start // block_mask.BLOCK_SIZE[0]
        blocks = _blocks(block_mask, row, False, n) + _blocks(block_mask, row, True, n)
        acc, den, maximum = _update(q[0, :, start:stop], k[0], v[0], allowed[start:stop], blocks)
        den = torch.where(den == 0, 1, den)
        output.append((acc / den[..., None]).to(torch.bfloat16))
        lse.append((maximum + torch.log2(den)) * 0.6931471805599453)
    return torch.cat(output, dim=1)[None], types.SimpleNamespace(lse=torch.cat(lse, dim=1)[None])
