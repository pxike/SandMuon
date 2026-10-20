"""Optional first-polynomial/packing fusion for the execution-only candidate.

Internal helper: arguments have already been validated by the public wrapper.
Both FP32 Grams remain live for the original norm and polynomial arithmetic.
Compared with packed=True, omit the FP32 polynomial allocation, its write/read
round trip, and one packing launch. No claim of end-to-end speedup is implied.
"""
import torch
import triton
import triton.language as tl

from odl.ops.bf16_compensated_mm_fast import _mm_original, _mm_expanded, _pack


@triton.jit
def _first_poly_packed(g, g2, scales, hi, lo, R: tl.constexpr,
                       A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                       BLOCK: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(scales + batch)
    v = tl.load(g + batch * R * R + i, i < R * R, other=0).to(tl.float32)
    w = tl.load(g2 + batch * R * R + i, i < R * R, other=0).to(tl.float32)
    v = tl.div_rn(tl.div_rn(v, s), s)
    for _ in tl.static_range(4):
        w = tl.div_rn(w, s)
    # Identical FP32 expression to _first_poly; launch with fusion disabled.
    z = A * (i // R == i % R).to(tl.float32) + B * v + C * w
    zh = z.to(tl.bfloat16)
    zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
    tl.store(hi + batch * R * R + i, zh, i < R * R)
    tl.store(lo + batch * R * R + i, zl, i < R * R)


def _first_update(g, g2, scale, x, abc, poly_block, *, tuned, fuse_scale):
    """Return BF16 P @ (x/scale), or P @ x if already explicitly divided."""
    batch, r, n = x.shape
    storage = torch.empty((2, batch, r, r), device=x.device, dtype=torch.bfloat16)
    ph, pl = storage.unbind(0)
    out = torch.empty((batch, r, n), device=x.device, dtype=torch.bfloat16)
    kernel = _mm_expanded if tuned else _mm_original
    with torch.cuda.device(x.device):
        _first_poly_packed[(batch, triton.cdiv(r * r, poly_block))](
            g, g2, scale, ph, pl, r, *abc, poly_block, enable_fp_fusion=False)
        xh, xl = _pack(x, scale if fuse_scale else None)
        # Use exactly the existing packed MM specialization and tuner: same
        # strides, dtypes, K loop, dot order, and direct BF16 output store.
        kernel[lambda meta: (triton.cdiv(r, meta['BM']),
                             triton.cdiv(n, meta['BN']), batch)](
            ph, xh, out, xh, pl, xl, r, n, r,
            *ph.stride()[1:], *xh.stride()[1:], ph.stride(0), xh.stride(0),
            batch, False, 0, x.device.index, True, enable_fp_fusion=False)
    return out
