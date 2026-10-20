#!/usr/bin/env python3
"""Diagnostic RTL: wide column folding in 32-wide online attention.

Handwritten Triton isolates the compiler transformation; this is not
standard Torch performance or R40-08 acceptance evidence.
"""
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
import torch
import triton
import triton.language as tl
from tnca import aot, device
from tnca.runtime import Session

@triton.jit
def probe(Q,K,V,O,S:tl.constexpr,D:tl.constexpr):
    d=tl.arange(0,D)
    r=tl.arange(0,16)
    q=tl.load(Q+d).to(tl.float32)
    kd=tl.make_tensor_descriptor(K,shape=[S,D],strides=[D,1],block_shape=[16,D])
    vd=tl.make_tensor_descriptor(V,shape=[S,D],strides=[D,1],block_shape=[16,D])
    m=tl.full((),-float('inf'),tl.float32)
    z=tl.full((),0,tl.float32)
    o=tl.full((D,),0,tl.float32)
    for start in range(triton.cdiv(S,16)):
        pos=start*16+r
        k=kd.load([start*16,0]).to(tl.float32)
        score=tl.sum(k*q[None,:],1)*(D**-0.5)
        score=tl.where(pos<S,score,-float('inf'))
        new_m=tl.maximum(m,tl.max(score,0))
        alpha=tl.exp(m-new_m)
        p=tl.exp(score-new_m)
        v=vd.load([start*16,0]).to(tl.float32)
        o=o*alpha+tl.sum(p[:,None]*v,0)
        z=z*alpha+tl.sum(p,0)
        m=new_m
    tl.store(O+d,o/z)

def main():
    with tempfile.TemporaryDirectory(prefix='r40-wide-columns-') as tmp:
        result = aot.compile_aot(probe, kernel_name='probe',
            signature={'Q': '*fp16', 'K': '*fp16', 'V': '*fp16', 'O': '*fp16'},
            constexprs={'S': 17, 'D': 32}, profile='core-vrf-sync-v5',
            output=str(Path(tmp) / 'probe.tnca'))
        device.prepare('core-vrf-sync-v5/hbm')
        torch.manual_seed(43)
        q=(torch.randn(32)*.1).half()
        k,v=[(torch.randn(17,32)*.1).half() for _ in range(2)]
        ref=torch.softmax(q.float()@k.float().T*(32**-.5),dim=-1)@v.float()
        with Session(profile='core-vrf-sync-v5/hbm',host_timeout=172800,deadline_cycles=2000000) as s:
            m=s.load(result.path)
            args={}
            for name,t in zip(('Q','K','V'),(q,k,v)):
                args[name]=s.allocate(t.numel()*2);s.upload(args[name],t.numpy().tobytes())
            args['O']=s.allocate(32*2);s.upload(args['O'],bytes(32*2))
            e=s.submit(m,args);s.wait(e);assert e.ok,e.error
            out=torch.frombuffer(bytearray(s.download(args['O'])),dtype=torch.float16).reshape(32)
            torch.testing.assert_close(out.float(),ref,atol=2e-3,rtol=2e-3)
            print('PASS wide-column online diagnostic; max_abs_error',float((out.float()-ref).abs().max()),flush=True)
            print('event',e,flush=True)


if __name__ == '__main__':
    main()
