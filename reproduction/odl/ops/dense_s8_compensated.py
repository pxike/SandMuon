"""Opt-in fast S8 with a BF16-product compensated first stage.
Same scalar coefficients as ordinary S8. FP32 accumulators and temporaries,
no FP32 GEMM. Four later stages retain the standard fast BF16 kernels.
"""
import torch
import triton
import triton.language as tl
from odl.ops.bf16_compensated_mm import mm
from odl.ops.minimax_gram_funnel import _normalization,_fixed_poly

@triton.jit
def _first_poly(g,g2,scales,out,R:tl.constexpr,A:tl.constexpr,B:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    batch,tile=tl.program_id(0),tl.program_id(1)
    i=tile*BLOCK+tl.arange(0,BLOCK)
    s=tl.load(scales+batch)
    v=tl.load(g+batch*R*R+i,i<R*R,other=0).to(tl.float32)
    w=tl.load(g2+batch*R*R+i,i<R*R,other=0).to(tl.float32)
    v=tl.div_rn(tl.div_rn(v,s),s)
    for _ in tl.static_range(4):w=tl.div_rn(w,s)
    # Preserve the first-stage correction: no BF16 store of rescaled Grams.
    z=A*(i//R==i%R).to(tl.float32)+B*v+C*w
    tl.store(out+batch*R*R+i,z,i<R*R)

@torch.no_grad()
def dense_s8_compensated(x,coefficients,margin=1.35,multiplier=1.,poly_block=1024):
    if not x.is_cuda or x.ndim!=3:raise ValueError('Requires a CUDA [batch,rows,cols] input')
    if len(coefficients)!=5:raise ValueError('Requires five coefficient triples')
    dtype=x.dtype;trans=x.shape[-2]>x.shape[-1]
    if trans:x=x.mT
    x=x.bfloat16().float().contiguous()
    x=x/x.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30)
    g=mm(x,x.mT);g2=mm(g,g)
    scale=(margin*g2.norm(dim=(-2,-1),keepdim=True).pow(.25)).clamp_min(1e-30)
    x=x/scale
    r=x.shape[-2];grid=(x.shape[0],triton.cdiv(r*r,poly_block))
    p=torch.empty_like(g)
    _first_poly[grid](g,g2,scale,p,r,*coefficients[0],poly_block,enable_fp_fusion=False)
    x=mm(p,x).bfloat16()
    for abc in coefficients[1:]:
        g=x@x.mT;g2=g@g;p=torch.empty_like(g)
        _fixed_poly[grid](g,g2,p,r,*abc,poly_block,enable_fp_fusion=False)
        x=p@x
    from odl.ops.dense_s8_fast import _output_scale
    out=torch.empty(x.shape,device=x.device,dtype=dtype)
    _output_scale[(triton.cdiv(x.numel(),1024),)](x,out,x.numel(),multiplier,1024,enable_fp_fusion=False)
    return out.mT if trans else out
