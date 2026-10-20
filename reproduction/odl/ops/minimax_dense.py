"""Experimental dense transfer of the live-r funnel; coefficients NOT refit.

Direct dense stages avoid propagating Gram/transformation matrices, whose
extra square GEMMs are only attractive for sufficiently rectangular slabs.
No PI, slab merge, or persistent spectral state. Output scale remains .63.

execution='gram'/'auto' are diagnostic only: RTX 3060 profiling found a
2048x8192 saved momentum case with cosine 0.719 to the direct BF16 path.
Do not promote them to training defaults without fixing propagation drift.
"""
import math
import torch
from odl.ops.minimax_gram_funnel import (
    MARGIN, _constants, _normalization, _polynomial, triton,
)

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _reuse_gram_kernel(g, g2, scales, out, out2, N: tl.constexpr, BLOCK: tl.constexpr):
        b, tile = tl.program_id(0), tl.program_id(1)
        i = tile*BLOCK+tl.arange(0, BLOCK)
        s = tl.load(scales+b)
        v = tl.load(g+b*N+i, i<N, other=0).to(tl.float32)
        w = tl.load(g2+b*N+i, i<N, other=0).to(tl.float32)
        # Sequential divisions avoid overflowing inverse powers for zero input.
        v = tl.div_rn(tl.div_rn(v, s), s)
        for _ in tl.static_range(4):
            w = tl.div_rn(w, s)
        tl.store(out+b*N+i, v, i<N)
        tl.store(out2+b*N+i, w, i<N)


@torch.no_grad()
def dense_live_r(x, output_scale=.63, *, reference=False, reuse_gram=False, execution="direct", refresh_after=()):
    if x.ndim != 3 or x.shape[-2] > x.shape[-1]:
        raise ValueError("Expected oriented [batch, rows, cols], rows <= cols")
    if execution not in ("direct", "gram", "auto"):
        raise ValueError("execution must be direct, gram, or auto")
    refresh_after = tuple(refresh_after)
    if any(stage not in (1, 2, 3, 4) for stage in refresh_after):
        raise ValueError("refresh_after counts completed stages, between 1 and 4")
    # With first-Gram reuse: direct costs 10*r*r*c+5*r**3;
    # Gram recurrence costs 2*r*r*c+17*r**3 (multiply-add counted as one).
    # Conservative threshold above the theoretical c/r > 1.5 break-even.
    gram_execution = execution == "gram" or (execution == "auto" and x.shape[-1] >= 2*x.shape[-2])
    if gram_execution:
        reuse_gram = True
    original_dtype = x.dtype
    fused = x.is_cuda and triton is not None and not reference
    # Both CUDA paths use the same native BF16 GEMMs.
    x = x.to(torch.bfloat16).contiguous() if x.is_cuda else x.contiguous()
    dtype = x.dtype
    def normalize(v, sensor=None, power=2, factor=1., target=1., return_scale=False):
        if fused:
            return _normalization(v, sensor, power, factor, target, return_scale)
        sensor = v if sensor is None else sensor
        norm = sensor.float().norm(dim=(-2, -1), keepdim=True)
        if power == 8:
            norm = norm.pow(.25)
        scale = (factor*norm).clamp_min(1e-30)
        out = (v.float()*target/scale).to(dtype)
        return (out, scale) if return_scale else out
    x = normalize(x)
    g = x@x.transpose(-1, -2)
    g2 = g@g
    x, scale = normalize(x, g2, power=8, factor=MARGIN, return_scale=True)
    if reuse_gram:
        if fused:
            scaled_g, scaled_g2 = torch.empty_like(g), torch.empty_like(g2)
            count = g[0].numel()
            _reuse_gram_kernel[(g.shape[0], triton.cdiv(count, 512))](
                g, g2, scale, scaled_g, scaled_g2, count, 512,
                enable_fp_fusion=False)
            g, g2 = scaled_g, scaled_g2
        else:
            g = (g.float()/scale/scale).to(dtype)
            g2 = (g2.float()/scale/scale/scale/scale).to(dtype)
    anchors = _constants(x.device)
    eye = None if fused else torch.eye(x.shape[-2], device=x.device)
    base, transform = x, None
    for stage in range(5):
        if not gram_execution and (stage or not reuse_gram):
            g = x@x.transpose(-1, -2)
        if stage or not reuse_gram:
            g2 = g@g
        if fused:
            p = _polynomial(g, g2, stage, anchors)
        else:
            s4 = g.float().norm(dim=(-2, -1), keepdim=True).sqrt()
            s8 = g2.float().norm(dim=(-2, -1), keepdim=True).pow(.25)
            t = ((s8/s4.clamp_min(1e-30)-.85)/.15).clamp(0, 1)
            abc = anchors[0, stage]+t*(anchors[1, stage]-anchors[0, stage])
            a, b, c = (abc[..., j:j+1] for j in range(3))
            p = (a*eye+b*g.float()+c*g2.float()).to(dtype)
        if gram_execution:
            transform = p if transform is None else p@transform
            if stage+1 in refresh_after:
                # Re-anchor the recurrence to its actual BF16 materialization.
                # Do NOT normalize here: this changes execution, not the math.
                x = transform@base
                base, transform = x, None
                g = x@x.transpose(-1, -2)
            elif stage < 4:
                g = (p@g)@p.transpose(-1, -2)
        else:
            x = p@x
    if gram_execution:
        x = transform@base
    if output_scale is not None:
        x = normalize(x, target=output_scale*math.sqrt(x.shape[-2]))
    return x.to(original_dtype)
