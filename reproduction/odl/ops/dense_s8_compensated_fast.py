"""Execution-only S8 candidate, imported explicitly; no production dispatch.

Keep the initial BF16 round/FP32 contiguous conversion, both torch norms,
coefficients, polynomial kernels, and four subsequent BF16 stages unchanged.
The third compensated product fuses scale division and its BF16 store.
Optional packing reuses BF16 high/low operands across all MM output tiles.
"""
import math
import numbers

import torch
import triton

from odl.ops.bf16_compensated_mm_fast import mm
from odl.ops.dense_s8_compensated import _first_poly
from odl.ops.dense_s8_fast import _output_scale
from odl.ops.minimax_gram_funnel import _fixed_poly


@torch.no_grad()
def dense_s8_compensated_fast(x, coefficients, margin=1.35, multiplier=1.,
                              poly_block=1024, tuned=False, *, fuse_scale=True,
                              packed=False, packed_prefix=False):
    """Same shape/dtype/orientation as dense_s8_compensated.

    ``tuned=False`` uses the original tile set; True expands it. ``fuse_scale``
    can be disabled to compare explicit versus fused FP32 division. Initial
    conversion is deliberately unfused pending GPU bitwise verification.
    ``packed=True`` pre-splits operands for all three compensated MMs, sharing
    packed planes for x/x.mT and g/g. Norm then division stays in its original
    order; only the later scale division is fused into packing.
    ``packed_prefix=True`` implies packing and additionally emits the first
    polynomial's high/low planes directly, avoiding its FP32 staging buffer
    and separate packing launch. Both FP32 Gram matrices are preserved.
    No numerical checks/synchronization are added to the training hot path.
    """
    if not x.is_cuda or x.ndim != 3:
        raise ValueError('Requires a CUDA [batch,rows,cols] input')
    if len(coefficients) != 5 or any(len(abc) != 3 for abc in coefficients):
        raise ValueError('Requires five coefficient triples')
    if any(not isinstance(v, numbers.Real) or not math.isfinite(v)
           for abc in coefficients for v in abc):
        raise ValueError('Coefficients must be finite real scalars')
    if not isinstance(margin, numbers.Real) or not math.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be a finite positive scalar')
    if not isinstance(multiplier, numbers.Real) or not math.isfinite(multiplier):
        raise ValueError('multiplier must be a finite real scalar')
    if not isinstance(poly_block, int) or poly_block <= 0 or poly_block & (poly_block - 1):
        raise ValueError('poly_block must be a positive power of two')
    dtype = x.dtype
    trans = x.shape[-2] > x.shape[-1]
    if trans:
        x = x.mT
    x = x.bfloat16().float().contiguous()
    if not x.numel():
        out = torch.empty(x.shape, device=x.device, dtype=dtype)
        return out.mT if trans else out
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    g = mm(x, x.mT, tune=tuned, packed=packed or packed_prefix)
    g2 = mm(g, g, tune=tuned, packed=packed or packed_prefix)
    scale = (margin * g2.norm(dim=(-2, -1), keepdim=True).pow(.25)).clamp_min(1e-30)
    if not fuse_scale:
        x = x / scale
    r = x.shape[-2]
    grid = (x.shape[0], triton.cdiv(r * r, poly_block))
    if packed_prefix:
        from odl.ops.dense_s8_compensated_packed_prefix import _first_update
        x = _first_update(g, g2, scale, x, coefficients[0], poly_block,
                          tuned=tuned, fuse_scale=fuse_scale)
    else:
        p = torch.empty_like(g)
        _first_poly[grid](g, g2, scale, p, r, *coefficients[0], poly_block,
                          enable_fp_fusion=False)
        x = mm(p, x, output_dtype=torch.bfloat16,
               b_scale=scale if fuse_scale else None, tune=tuned, packed=packed)
    for abc in coefficients[1:]:
        g = x @ x.mT
        g2 = g @ g
        p = torch.empty_like(g)
        _fixed_poly[grid](g, g2, p, r, *abc, poly_block, enable_fp_fusion=False)
        x = p @ x
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    _output_scale[(triton.cdiv(x.numel(), 1024),)](
        x, out, x.numel(), multiplier, 1024, enable_fp_fusion=False)
    return out.mT if trans else out


# Same callable name as the source module for explicit import substitution.
dense_s8_compensated = dense_s8_compensated_fast
