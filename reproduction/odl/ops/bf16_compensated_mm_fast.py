"""Opt-in compensated MM; the baseline module and its tuner are untouched.

Three BF16 products per K tile, FP32 accumulation, corrections before high*high.
Changing the selected tile (especially BK) can change FP32 rounding. ``tune``
selects expanded versus original tile candidates, not whether autotuning runs.
"""
import torch
import triton
import triton.language as tl


def _original_configs():
    return [triton.Config(dict(BM=m, BN=n, BK=k), num_warps=4)
            for m, n, k in ((32, 64, 32), (64, 64, 32),
                            (64, 64, 64), (32, 64, 64))]


@triton.jit
def _mm_body(A, B, C, SCALE, AL, BL,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             AS0: tl.constexpr, AS1: tl.constexpr,
             BS0: tl.constexpr, BS1: tl.constexpr,
             AST: tl.constexpr, BST: tl.constexpr, BATCH: tl.constexpr,
             SCALED: tl.constexpr, SS: tl.constexpr,
             DEVICE: tl.constexpr, PACKED: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    batch = tl.program_id(2)
    A = A + batch * AST
    B = B + batch * BST
    if PACKED:
        AL = AL + batch * AST
        BL = BL + batch * BST
    C = C + batch * M * N
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    if SCALED:
        scale = tl.load(SCALE + batch * SS).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        kk = base * BK + ki
        valid_a = (mi[:, None] < M) & (kk[None, :] < K)
        a_offsets = mi[:, None] * AS0 + kk[None, :] * AS1
        b_offsets = kk[:, None] * BS0 + ni[None, :] * BS1
        a = tl.load(A + a_offsets, valid_a, other=0)
        valid_b = (kk[:, None] < K) & (ni[None, :] < N)
        b = tl.load(B + b_offsets, valid_b, other=0)
        if PACKED:
            ah = a
            bh = b
            al = tl.load(AL + a_offsets, valid_a, other=0)
            bl = tl.load(BL + b_offsets, valid_b, other=0)
        else:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
            if SCALED:
                # Round division BEFORE splitting, like materializing B/scale.
                # Padding stays zero even for zero/NaN scale values.
                b = tl.where(valid_b, tl.div_rn(b, scale), 0.)
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bh, acc)
    tl.store(C + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


_KEY = ['M', 'N', 'K', 'AS0', 'AS1', 'BS0', 'BS1', 'AST', 'BST',
        'BATCH', 'SCALED', 'SS', 'DEVICE', 'PACKED']
# Independent autotuners/caches around our own JIT kernel; neither wraps nor
# mutates the baseline autotuner. Pointer dtypes also enter Triton's cache key.
_mm_original = triton.autotune(configs=_original_configs(), key=_KEY)(_mm_body)


def _prune_tiny_tiles(configs, named_args, **kwargs):
    # Masked oversized tiles add compilation/register cost for tiny diagnostics.
    # This does not exclude any candidate for the production 768+ dimensions.
    if min(named_args['M'], named_args['N']) < 128:
        return [c for c in configs if c.kwargs['BM'] <= 64 and c.kwargs['BN'] <= 64]
    m = max(64, triton.next_power_of_2(named_args['M']))
    n = max(64, triton.next_power_of_2(named_args['N']))
    return [c for c in configs if c.kwargs['BM'] <= m and c.kwargs['BN'] <= n]


_mm_expanded = triton.autotune(
    configs=_original_configs() + [
        triton.Config(dict(BM=m, BN=n, BK=k),
                      num_warps=4 if (m, n, k) == (128, 64, 32) else 8,
                      num_stages=2)
        for m, n in ((128, 64), (64, 128), (128, 128))
        for k in (32, 64)
    ], key=_KEY, prune_configs_by={'early_config_prune': _prune_tiny_tiles}
)(_mm_body)


@triton.jit
def _pack_kernel(X, HI, LO, SCALE, R: tl.constexpr, C: tl.constexpr,
                 S0: tl.constexpr, S1: tl.constexpr, ST: tl.constexpr,
                 SCALED: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = i < R * C
    v = tl.load(X + batch * ST + (i // C) * S0 + (i % C) * S1,
                valid, other=0).to(tl.float32)
    if SCALED:
        scale = tl.load(SCALE + batch * SS).to(tl.float32)
        v = tl.div_rn(v, scale)
    hi = v.to(tl.bfloat16)
    lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(HI + batch * R * C + i, hi, valid)
    tl.store(LO + batch * R * C + i, lo, valid)


def _pack(x, scale=None):
    # One allocation, two individually contiguous planes; no FP32 staging.
    storage = torch.empty((2, *x.shape), device=x.device, dtype=torch.bfloat16)
    hi, lo = storage.unbind(0)
    batch, rows, cols = x.shape
    _pack_kernel[(batch, triton.cdiv(rows * cols, 1024))](
        x, hi, lo, x if scale is None else scale, rows, cols,
        *x.stride()[1:], x.stride(0), scale is not None,
        0 if scale is None else scale.stride(0), 1024, enable_fp_fusion=False)
    return hi, lo


def _same_view(a, b):
    return (a.data_ptr() == b.data_ptr() and a.dtype == b.dtype
            and a.shape == b.shape and a.stride() == b.stride())


def mm(a, b, *, output_dtype=torch.float32, b_scale=None, tune=False,
       packed=False):
    """Return contiguous [batch,M,N]; never modify inputs.

    ``b_scale`` is FP32 on the input GPU, shaped [batch] or [batch,1,1],
    including strided/broadcast batch views. Division uses FP32 round-to-nearest.
    FP32/BF16 output changes only the final store. Both modes autotune on first
    use; ``tune=True`` adds six larger tiles. Triton's autotuner skips candidates
    that fail its resource checks. Requires NVIDIA CUDA; SM75 is allowed for diagnostics only.
    ``packed=True`` splits operands once outside MM (including B division),
    reusing A's packed planes when B is the same view or its matrix transpose.
    Packing adds launches and temporary storage; benchmark end-to-end. It keeps
    the K loop and dot order but can cause autotuning to select a different BK.
    """
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError('Requires compatible [batch,M,K] and [batch,K,N] inputs')
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError('Inputs must be on the same CUDA device')
    if a.dtype not in (torch.float32, torch.bfloat16) or b.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Inputs must be FP32 or BF16')
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('output_dtype must be FP32 or BF16')
    batch, m, k = a.shape
    n = b.shape[2]
    if b_scale is not None:
        if (b_scale.device != a.device or b_scale.dtype != torch.float32
                or tuple(b_scale.shape) not in ((batch,), (batch, 1, 1))):
            raise ValueError('b_scale must be FP32 [batch] or [batch,1,1] on the input device')
    c = torch.empty((batch, m, n), device=a.device, dtype=output_dtype)
    if not batch or not m or not n:
        return c
    if not k:
        return c.zero_()
    if torch.version.hip is not None:
        raise ValueError('Compensated MM requires NVIDIA CUDA; HIP is unsupported')
    kernel = _mm_expanded if tune else _mm_original
    with torch.cuda.device(a.device):
        if packed:
            ah, al = _pack(a)
            if b_scale is None and _same_view(a, b):
                bh, bl = ah, al
            elif b_scale is None and _same_view(a.mT, b):
                bh, bl = ah.mT, al.mT
            else:
                bh, bl = _pack(b, b_scale)
            a, b = ah, bh
            b_scale = None  # Already divided with FP32 rounding in _pack.
        else:
            al, bl = a, b  # Unused pointers in the unpacked specialization.
        kernel[lambda meta: (triton.cdiv(m, meta['BM']),
                             triton.cdiv(n, meta['BN']), batch)](
            a, b, c, b if b_scale is None else b_scale, al, bl, m, n, k,
            *a.stride()[1:], *b.stride()[1:], a.stride(0), b.stride(0), batch,
            b_scale is not None, 0 if b_scale is None else b_scale.stride(0),
            a.device.index, packed, enable_fp_fusion=False)
    return c
