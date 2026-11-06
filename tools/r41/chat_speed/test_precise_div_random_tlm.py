#!/usr/bin/env python3
"""Compare explicit Triton div_rn against CPU IEEE float32 on DP TLM."""
import json, sys, struct, os, tempfile
from pathlib import Path
root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm'), str(root/'tools/performance_harness/systemc')]
import triton
import triton.language as tl

@triton.jit
def divide(A, B, C, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+i); b=tl.load(B+i)
    tl.store(C+i, tl.div_rn(a,b))

def main():
    import torch, tlm_backend
    from tnca import aot
    from tnca.runtime import Session
    tlm_backend.register()
    torch.manual_seed(int(os.environ.get("DIV_SEED", "93041")))
    raw_a=torch.randint(-(2**31),2**31,(65536,),dtype=torch.int32)
    raw_b=torch.randint(-(2**31),2**31,(65536,),dtype=torch.int32)
    aa=raw_a.view(torch.float32); bb=raw_b.view(torch.float32)
    aa=torch.where(torch.isfinite(aa),aa,torch.ones_like(aa))
    bb=torch.where(torch.isfinite(bb)&(bb!=0),bb,torch.ones_like(bb))
    want=(aa/bb).view(torch.int32).tolist()
    result=aot.compile_aot(divide,signature={'A':'*fp32','B':'*fp32','C':'*fp32','BLOCK':'constexpr'},constexprs={'BLOCK':256},profile='core-vrf-sync-v5-x8-g2c4-dp',options={'index_opt':1},output=str(Path(tempfile.mkdtemp(prefix='r41-div-random-'))/'divide.tnca'))
    with Session(profile='core-vrf-sync-v5-x8-g2c4-dp',backend='systemc-tlm-grid',host_timeout=180) as sess:
        module=sess.load(result.path); x,y,z=[sess.allocate(262144) for _ in range(3)]
        sess.upload(x,aa.numpy().tobytes()); sess.upload(y,bb.numpy().tobytes())
        before=sess.dev.counters_snapshot(); event=sess.submit(module,{'A':x,'B':y,'C':z},grid=(256,1,1)); sess.wait(event);after=sess.dev.counters_snapshot()
        if not event.ok: raise RuntimeError(str(event.error))
        got=struct.unpack('<65536i',sess.download(z)); mismatches=[i for i,(a,b) in enumerate(zip(got,want)) if a!=b]
        ins=sum(v-before.get(k,0) for k,v in after.items() if k.startswith('core_') and k.endswith('_instructions'))
        print(json.dumps(dict(domain='DP TLM',count=len(got),bit_mismatches=len(mismatches),first_mismatches=mismatches[:16],cycles=event.device_cycles,instructions=ins,launches=1,loads=1)))
        for i in mismatches: print(json.dumps(dict(i=i,a=float(aa[i]),b=float(bb[i]),want=f'{want[i]&0xffffffff:08x}',got=f'{got[i]&0xffffffff:08x}')),flush=True)
        if mismatches: raise RuntimeError('IEEE reference mismatch')
if __name__=='__main__':main()
