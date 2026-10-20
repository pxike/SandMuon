"""Experimental BF16 live-r minimax Gram funnel, never selected implicitly.

Coefficients: outputs/bf16_minimax_fused/bf16_minimax_slabs.json, fused_fp32
fit. Native BF16 GEMMs; FP32 moments/interpolation/polynomial, one BF16 store.
Default final norm is legacy .63*sqrt(min(full_rows, cols)), NOT stock norm
matching. Pass output_scale=None for raw fitted direction and parity checks.
"""
import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None

LO = ((4.811112927794569, -10.848687392184711, 7.351963555437045),
      (1.9420053200435188, -1.3097888040416241, .3536316783915939),
      (4.705493661365867, -10.302348581684575, 6.761202572595993),
      (1.9644648069746913, -1.3758548163997055, .3849307217469888),
      (1.888696973178235, -1.237844418127991, .3651808443655983))
HI = ((4.811112927794569, -10.848687392184711, 7.351963555437045),
      (1.963316181642894, -1.3703945810104552, .3872067036864561),
      (4.755315600989576, -10.621380404453689, 7.097605582546588),
      (1.9931854761617613, -1.395891465266848, .3888799331643735),
      (1.866480080273455, -1.2217792818958808, .3750691978293568))
MARGIN = 1.6
# Native-BF16 real-momentum fit, experiment_funnel_r_freedom.py (fixed).
# Explicit FP32-rounded triples; opt-in, not a replacement for live-r.
REFIT_FIXED = ((5.597528457641602, -11.102691650390625, 6.712371826171875),
               (1.8793039321899414, -1.4088671207427979, .3163202702999115),
               (5.454198837280273, -10.378914833068848, 4.8467488288879395),
               (2.2066409587860107, -1.7846355438232422, .44029438495635986),
               (1.7725071907043457, -1.460353136062622, .2879667282104492))
_CACHE = {}


def _constants(device):
    key = (device.type, device.index)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor((LO, HI), device=device, dtype=torch.float32)
    return _CACHE[key]


if triton is not None:
    @triton.jit
    def _partials(x, y, out, N: tl.constexpr, PARTS: tl.constexpr,
                  PAIR: tl.constexpr, BLOCK: tl.constexpr):
        b, p = tl.program_id(0), tl.program_id(1)
        i = p*BLOCK + tl.arange(0, BLOCK)
        a = tl.load(x+b*N+i, i<N, other=0).to(tl.float32)
        tl.store(out+(b*PARTS+p)*2, tl.sum(a*a, 0))
        if PAIR:
            v = tl.load(y+b*N+i, i<N, other=0).to(tl.float32)
            tl.store(out+(b*PARTS+p)*2+1, tl.sum(v*v, 0))

    @triton.jit
    def _norm(partials, scales, PARTS: tl.constexpr, POWER: tl.constexpr,
              FACTOR: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        p = tl.arange(0, BLOCK)
        ss = tl.sum(tl.load(partials+(b*PARTS+p)*2, p<PARTS, other=0), 0)
        n = tl.sqrt(ss)
        if POWER == 8:
            n = tl.sqrt(tl.sqrt(n))
        tl.store(scales+b, tl.maximum(FACTOR*n, 1.e-30))

    @triton.jit
    def _scale(x, scales, out, N: tl.constexpr, TARGET: tl.constexpr,
               BLOCK: tl.constexpr):
        b, p = tl.program_id(0), tl.program_id(1)
        i = p*BLOCK+tl.arange(0, BLOCK)
        v = tl.load(x+b*N+i, i<N, other=0).to(tl.float32)
        s = tl.load(scales+b)
        tl.store(out+b*N+i, tl.div_rn(v*TARGET, s), i<N)

    @triton.jit
    def _coeff(partials, anchors, coeff, PARTS: tl.constexpr,
               STAGE: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        p = tl.arange(0, BLOCK)
        ss4 = tl.sum(tl.load(partials+(b*PARTS+p)*2, p<PARTS, other=0), 0)
        ss8 = tl.sum(tl.load(partials+(b*PARTS+p)*2+1, p<PARTS, other=0), 0)
        s4 = tl.sqrt(tl.sqrt(ss4))
        s8 = tl.sqrt(tl.sqrt(tl.sqrt(ss8)))
        t = tl.minimum(1., tl.maximum(0., (tl.div_rn(s8, tl.maximum(s4, 1.e-30))-.85)/.15))
        for j in tl.static_range(3):
            lo = tl.load(anchors+STAGE*3+j)
            hi = tl.load(anchors+15+STAGE*3+j)
            tl.store(coeff+b*3+j, lo+t*(hi-lo))

    @triton.jit
    def _poly(g, g2, coeff, out, R: tl.constexpr, BLOCK: tl.constexpr):
        b, p = tl.program_id(0), tl.program_id(1)
        i = p*BLOCK+tl.arange(0, BLOCK)
        v = tl.load(g+b*R*R+i, i<R*R, other=0).to(tl.float32)
        w = tl.load(g2+b*R*R+i, i<R*R, other=0).to(tl.float32)
        a, bb, c = tl.load(coeff+b*3), tl.load(coeff+b*3+1), tl.load(coeff+b*3+2)
        z = a*(i//R == i%R).to(tl.float32) + bb*v + c*w
        tl.store(out+b*R*R+i, z, i<R*R)

    @triton.jit
    def _fixed_poly(g, g2, out, R: tl.constexpr, A: tl.constexpr,
                    B: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
        batch, part = tl.program_id(0), tl.program_id(1)
        i = part*BLOCK+tl.arange(0, BLOCK)
        v = tl.load(g+batch*R*R+i, i<R*R, other=0).to(tl.float32)
        w = tl.load(g2+batch*R*R+i, i<R*R, other=0).to(tl.float32)
        a = tl.full((), A, tl.float32)
        b = tl.full((), B, tl.float32)
        c = tl.full((), C, tl.float32)
        z = a*(i//R == i%R).to(tl.float32)+b*v+c*w
        tl.store(out+batch*R*R+i, z, i<R*R)


def _normalization(x, sensor=None, power=2, factor=1., target=1., return_scale=False):
    sensor = x if sensor is None else sensor
    batch, n = sensor.shape[0], sensor[0].numel()
    parts = triton.cdiv(n, 512)
    partials = torch.empty((batch, parts, 2), device=x.device, dtype=torch.float32)
    scale = torch.empty(batch, device=x.device, dtype=torch.float32)
    _partials[(batch, parts)](sensor, sensor, partials, n, parts, False, 512,
                              enable_fp_fusion=False)
    _norm[(batch,)](partials, scale, parts, power, factor, triton.next_power_of_2(parts),
                    enable_fp_fusion=False)
    out = torch.empty_like(x)
    count = x[0].numel()
    _scale[(batch, triton.cdiv(count, 512))](x, scale, out, count, target, 512,
                                           enable_fp_fusion=False)
    return (out, scale) if return_scale else out


def _polynomial(g, g2, stage, anchors):
    batch, rows = g.shape[:2]
    parts = triton.cdiv(rows*rows, 512)
    partials = torch.empty((batch, parts, 2), device=g.device, dtype=torch.float32)
    coeff = torch.empty((batch, 3), device=g.device, dtype=torch.float32)
    out = torch.empty_like(g)
    _partials[(batch, parts)](g, g2, partials, rows*rows, parts, True, 512,
                              enable_fp_fusion=False)
    _coeff[(batch,)](partials, anchors, coeff, parts, stage, triton.next_power_of_2(parts),
                     enable_fp_fusion=False)
    _poly[(batch, parts)](g, g2, coeff, out, rows, 512, enable_fp_fusion=False)
    return out


def _rescale_gram_pair(g, g2, scale, fused=False):
    """Reuse G/s² and G²/s⁴; exact algebra, distinct BF16 rounding."""
    if fused:
        from odl.ops.minimax_dense import _reuse_gram_kernel
        out, out2 = torch.empty_like(g), torch.empty_like(g2)
        count = g[0].numel()
        _reuse_gram_kernel[(g.shape[0], triton.cdiv(count, 512))](
            g, g2, scale, out, out2, count, 512, enable_fp_fusion=False)
        return out, out2
    scale = scale.reshape(-1, 1, 1)
    return ((g.float()/scale/scale).to(g.dtype),
            (g2.float()/scale/scale/scale/scale).to(g2.dtype))


@torch.no_grad()
def reference_minimax_gram_funnel(blocks, output_scale=.63, *, reuse_gram=False, fixed_coefficients=False):
    """Native dtype GEMM reference; no Triton. CPU tiny tests also supported."""
    if blocks.ndim != 3 or blocks.shape[0] % 4:
        raise ValueError("Expected [N*4, slab_rows, cols]")
    dtype = blocks.dtype
    n, rows, cols = blocks.shape[0]//4, blocks.shape[1], blocks.shape[2]
    x = blocks
    x = (x.float()/x.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)).to(dtype)
    g = x @ x.transpose(-1, -2)
    g2 = g@g
    s8 = g2.float().norm(dim=(-2, -1), keepdim=True).pow(.25)
    scale = (MARGIN*s8).clamp_min(1e-30)
    x = (x.float()/scale).to(dtype)
    if reuse_gram:
        g, g2 = _rescale_gram_pair(g, g2, scale)
    anchors = _constants(blocks.device)
    for stages in ((0, 1), (2, 3, 4)):
        if stages[0] == 2:
            x = (x.reshape(n*2, rows*2, cols).float()/math.sqrt(2)).to(dtype)
        base, transform = x, None
        reuse_first = reuse_gram and stages[0] == 0
        if not reuse_first:
            g = x @ x.transpose(-1, -2)
        eye = torch.eye(g.shape[-1], device=x.device, dtype=torch.float32)
        for index, stage in enumerate(stages):
            if index or not reuse_first:
                g2 = g@g
            s4 = g.float().norm(dim=(-2, -1), keepdim=True).sqrt()
            s8 = g2.float().norm(dim=(-2, -1), keepdim=True).pow(.25)
            t = ((s8/s4.clamp_min(1e-30)-.85)/.15).clamp(0, 1)
            if fixed_coefficients:
                t = torch.zeros_like(t)
            abc = anchors[0, stage].reshape(1, 1, 3)+t*(anchors[1, stage]-anchors[0, stage]).reshape(1, 1, 3)
            if fixed_coefficients and fixed_coefficients is not True:
                abc = torch.tensor(fixed_coefficients[stage], device=x.device,
                                   dtype=torch.float32).reshape(1, 1, 3)
            a, b, c = (abc[..., j:j+1] for j in range(3))
            p = (a*eye+b*g.float()+c*g2.float()).to(dtype)
            transform = p if transform is None else p@transform
            if index+1 < len(stages):
                g = (p@g)@p.transpose(-1, -2)
        x = transform@base
    if output_scale is not None:
        grouped = x.reshape(n, 4*rows, cols)
        norm = grouped.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        x = (grouped.float()*(output_scale*math.sqrt(min(4*rows, cols)))/norm).to(dtype).reshape_as(x)
    return x


@torch.no_grad()
def batched_minimax_gram_funnel(blocks, output_scale=.63, *, reuse_gram=False, fixed_coefficients=False):
    """Input q4 slabs; output q2 slabs. No host numerical checks in hot path."""
    if blocks.ndim != 3 or blocks.shape[0] % 4:
        raise ValueError("Expected [N*4, slab_rows, cols]")
    if not blocks.is_cuda or triton is None:
        return reference_minimax_gram_funnel(blocks, output_scale, reuse_gram=reuse_gram,
                                             fixed_coefficients=fixed_coefficients)
    if blocks.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16 or FP32")
    output_dtype = blocks.dtype
    blocks = blocks.to(torch.bfloat16).contiguous()
    n, rows, cols = blocks.shape[0]//4, blocks.shape[1], blocks.shape[2]
    x = _normalization(blocks)
    g = x@x.transpose(-1, -2)
    g2 = g@g
    x, scale = _normalization(x, sensor=g2, power=8, factor=MARGIN, return_scale=True)
    if reuse_gram:
        g, g2 = _rescale_gram_pair(g, g2, scale, fused=True)
    anchors = _constants(x.device)
    for stages in ((0, 1), (2, 3, 4)):
        if stages[0] == 2:
            x = (x.reshape(n*2, rows*2, cols)/math.sqrt(2))
        base, transform = x, None
        reuse_first = reuse_gram and stages[0] == 0
        if not reuse_first:
            g = x@x.transpose(-1, -2)
        for index, stage in enumerate(stages):
            if index or not reuse_first:
                g2 = g@g
            if fixed_coefficients:
                p = torch.empty_like(g)
                _fixed_poly[(g.shape[0], triton.cdiv(g.shape[1]**2, 512))](
                    g, g2, p, g.shape[1],
                    *(LO if fixed_coefficients is True else fixed_coefficients)[stage],
                    512, enable_fp_fusion=False)
            else:
                p = _polynomial(g, g2, stage, anchors)
            transform = p if transform is None else p@transform
            if index+1 < len(stages):
                g = (p@g)@p.transpose(-1, -2)
        x = transform@base
    if output_scale is not None:
        shaped = x.reshape(n, 4*rows, cols)
        x = _normalization(shaped, target=output_scale*math.sqrt(min(4*rows, cols))).reshape_as(x)
    return x.to(output_dtype)
