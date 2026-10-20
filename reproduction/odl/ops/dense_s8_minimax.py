"""Dense fixed-polynomial S8 path. Coefficients supplied by offline Remez fit."""
import torch
from odl.ops.minimax_gram_funnel import _normalization, _fixed_poly, triton
from odl.ops.minimax_dense import _reuse_gram_kernel


@torch.no_grad()
def dense_s8_minimax_original(x, coefficients, margin=1.01, multiplier=1., reference=False):
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2)
    original_dtype = x.dtype
    x = x.contiguous()
    fused = x.is_cuda and not reference
    if fused:
        x = x.bfloat16()
        x = _normalization(x)
    else:
        x = (x.float()/x.float().norm(dim=(-2,-1), keepdim=True).clamp_min(1e-30)).to(x.dtype)
    g = x@x.transpose(-1,-2)
    g2 = g@g
    if fused:
        x, s = _normalization(x, sensor=g2, power=8, factor=margin, return_scale=True)
        a, b = torch.empty_like(g), torch.empty_like(g2)
        _reuse_gram_kernel[(g.shape[0], triton.cdiv(g[0].numel(),512))](
            g,g2,s,a,b,g[0].numel(),512,enable_fp_fusion=False)
        g,g2 = a,b
    else:
        s = (margin*g2.float().norm(dim=(-2,-1),keepdim=True).pow(.25)).clamp_min(1e-30)
        x = (x.float()/s).to(x.dtype)
        g = (g.float()/s/s).to(g.dtype)
        g2 = (g2.float()/s/s/s/s).to(g2.dtype)
    for stage, abc in enumerate(coefficients):
        if stage:
            g = x@x.transpose(-1,-2)
            g2 = g@g
        if fused:
            p = torch.empty_like(g)
            _fixed_poly[(g.shape[0],triton.cdiv(g.shape[1]**2,512))](
                g,g2,p,g.shape[1],*abc,512,enable_fp_fusion=False)
        else:
            eye = torch.eye(g.shape[-1], device=x.device)
            p = (abc[0]*eye+abc[1]*g.float()+abc[2]*g2.float()).to(g.dtype)
        x = p@x
    x = (x.float()*multiplier).to(original_dtype)
    return x.transpose(-1,-2) if transposed else x


def dense_s8_minimax(x, coefficients, margin=1.01, multiplier=1., reference=False):
    """Dense S8 with execution-only fusion for the five-stage CUDA path.

    Coefficients, normalization and rounding boundaries are unchanged.
    The explicit original remains available for reference and timing controls.
    """
    if reference or not x.is_cuda or len(coefficients) != 5:
        return dense_s8_minimax_original(x, coefficients, margin, multiplier, reference)
    from odl.ops.dense_s8_fast import dense_s8_fast
    return dense_s8_fast(x, coefficients, margin, multiplier)
