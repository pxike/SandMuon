"""Batched first-stage precision using BF16 Tensor Core products, never FP32 GEMMs.
Three BF16 products approximate a product of two FP32 operands (omit low*low).
FP32 is used for accumulation and temporary buffers, as in mixed precision.
"""
import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({'BM':32,'BN':64,'BK':32},num_warps=4),
    triton.Config({'BM':64,'BN':64,'BK':32},num_warps=4),
    triton.Config({'BM':64,'BN':64,'BK':64},num_warps=4),
    triton.Config({'BM':32,'BN':64,'BK':64},num_warps=4),
],key=['M','N','K','AS0','AS1','BS0','BS1','BATCH'])
@triton.jit
def _mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        AS0:tl.constexpr,AS1:tl.constexpr,BS0:tl.constexpr,BS1:tl.constexpr,
        AST:tl.constexpr,BST:tl.constexpr,BATCH:tl.constexpr,
        BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    batch=tl.program_id(2)
    A=A+batch*AST;B=B+batch*BST;C=C+batch*M*N
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for base in range(tl.cdiv(K,BK)):
        kk=base*BK+ki
        a=tl.load(A+mi[:,None]*AS0+kk[None,:]*AS1,(mi[:,None]<M)&(kk[None,:]<K),other=0).to(tl.float32)
        b=tl.load(B+kk[:,None]*BS0+ni[None,:]*BS1,(kk[:,None]<K)&(ni[None,:]<N),other=0).to(tl.float32)
        ah=a.to(tl.bfloat16);bh=b.to(tl.bfloat16)
        al=(a-ah.to(tl.float32)).to(tl.bfloat16);bl=(b-bh.to(tl.float32)).to(tl.bfloat16)
        # Accumulate small corrections before high*high.
        acc=tl.dot(ah,bl,acc)
        acc=tl.dot(al,bh,acc)
        acc=tl.dot(ah,bh,acc)
    tl.store(C+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))

def mm(a,b):
    assert a.ndim==b.ndim==3 and a.shape[0]==b.shape[0] and a.shape[2]==b.shape[1]
    batch,m,k=a.shape;n=b.shape[2]
    c=torch.empty((batch,m,n),device=a.device,dtype=torch.float32)
    _mm[lambda meta:(triton.cdiv(m,meta['BM']),triton.cdiv(n,meta['BN']),batch)](a,b,c,m,n,k,*a.stride()[1:],*b.stride()[1:],a.stride(0),b.stride(0),batch)
    return c

