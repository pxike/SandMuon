"""Execution-only dense S8 NS5: unchanged arithmetic and BF16 boundaries."""
import torch
from odl.ops.dense_s8_minimax import dense_s8_minimax_original as _baseline
from odl.ops.minimax_gram_funnel import _normalization, _fixed_poly, triton

if triton is not None:
    import triton.language as tl
    from odl.ops.dense_s8_fused import _rescaled_fixed_poly

    @triton.jit
    def _output_scale(x, out, N: tl.constexpr, MULTIPLIER: tl.constexpr,
                      BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x+i, i<N, other=0).to(tl.float32)
        multiplier = tl.full((), MULTIPLIER, tl.float32)
        tl.store(out+i, value*multiplier, i<N)


@torch.no_grad()
def dense_s8_fast(x, coefficients, margin=1.35, multiplier=1., reference=False,
                  poly_block=1024, fused_output=True, fused_first=True,
                  compensated_first=False, direct_s8=False,
                  fused_tail_poly=False):
    """Fast ordinary-BF16 S8 execution.

    ``direct_s8`` uses the exact homogeneity of the *real-arithmetic* S8
    scale to omit the preliminary Frobenius normalization.  It is deliberately
    opt-in: BF16 rounding means this is not equivalent to the trained
    compensated recipe (or to the ordinary normalized path).  It retains the
    same five coefficients, margin and Schatten-8 sensor, and still has the
    same fifteen Tensor-Core products as Polar Express.

    ``fused_tail_poly`` writes ``aI + bG + cG^2`` from the four tail ``G@G``
    products directly.  This removes the materialized ``G^2`` and its
    elementwise polynomial kernel in those stages, but changes the rounding of
    ``G^2``.  It is likewise a speed-only arithmetic candidate.
    """
    if len(coefficients) != 5:
        raise ValueError('This execution path requires all five NS stages')
    if reference or not x.is_cuda or triton is None:
        return _baseline(x, coefficients, margin, multiplier, reference=True)
    if compensated_first:
        from odl.ops.dense_s8_compensated import dense_s8_compensated
        return dense_s8_compensated(x, coefficients, margin, multiplier, poly_block)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2)
    dtype = x.dtype
    x = x.contiguous().bfloat16()
    if not direct_s8:
        x = _normalization(x)
    g = x @ x.transpose(-1, -2)
    g2 = g @ g
    if direct_s8:
        # In exact arithmetic, normalizing by ||x||_F first cancels exactly:
        # (x/||x||_F) / (margin ||((xx^T)/||x||_F^2)^2||_F^(1/4))
        # = x / (margin ||(xx^T)^2||_F^(1/4)).  Keep the S8 sensor in FP32
        # without materializing a FP32 Gram-square copy.
        scale = (margin * torch.linalg.vector_norm(
            g2, dim=(-2, -1), dtype=torch.float32, keepdim=True).pow(.25)
        ).clamp_min(1e-30)
        x = (x.float()/scale).to(x.dtype)
    else:
        x, scale = _normalization(
            x, sensor=g2, power=8, factor=margin, return_scale=True)
    if not fused_first:
        from odl.ops.minimax_dense import _reuse_gram_kernel
        a, b = torch.empty_like(g), torch.empty_like(g2)
        _reuse_gram_kernel[(g.shape[0], triton.cdiv(g[0].numel(),512))](
            g,g2,scale,a,b,g[0].numel(),512,enable_fp_fusion=False)
        g,g2 = a,b
    for stage, abc in enumerate(coefficients):
        if stage:
            g = x @ x.transpose(-1, -2)
        grid = (g.shape[0], triton.cdiv(g.shape[1]**2, poly_block))
        if stage and fused_tail_poly:
            from odl.ops.dense_s8_tail_fused import gram_square_polynomial
            p = gram_square_polynomial(g, *abc)
        else:
            if stage:
                g2 = g @ g
            p = torch.empty_like(g)
            if stage == 0 and fused_first:
                _rescaled_fixed_poly[grid](g,g2,scale,p,g.shape[1],*abc,poly_block,
                                           enable_fp_fusion=False)
            else:
                _fixed_poly[grid](g,g2,p,g.shape[1],*abc,poly_block,enable_fp_fusion=False)
        x = p @ x
    if fused_output:
        out = torch.empty(x.shape, device=x.device, dtype=dtype)
        _output_scale[(triton.cdiv(x.numel(),1024),)](
            x,out,x.numel(),multiplier,1024,enable_fp_fusion=False)
        x = out
    else:
        x = (x.float()*multiplier).to(dtype)
    return x.transpose(-1, -2) if transposed else x


@torch.no_grad()
def dense_s8_direct_fast(x, coefficients, margin=1.35, multiplier=1., *,
                         fused_tail_poly=False):
    """Explicit direct-homogeneous, ordinary-BF16 S8 speed candidate.

    This convenience entrypoint exists so optimizer dispatch and benchmark
    metadata cannot accidentally describe it as the trained compensated S8.
    """
    return dense_s8_fast(
        x, coefficients, margin, multiplier, direct_s8=True,
        fused_tail_poly=fused_tail_poly)
