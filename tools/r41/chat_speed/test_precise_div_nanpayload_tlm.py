#!/usr/bin/env python3
"""Compare explicit div_rn NaN bits against the locked GB10 ARM reference."""
import json, sys, struct, tempfile
from pathlib import Path
root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm'), str(root/'tools/performance_harness/systemc')]
import triton
import triton.language as tl

@triton.jit
def divide(A, B, C, BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    a=tl.load(A+i); b=tl.load(B+i)
    tl.store(C+i, tl.div_rn(a,b))

def main():
    import torch, tlm_backend
    from tnca import aot
    from tnca.runtime import Session
    tlm_backend.register()
    patterns=[0x7f800001,0xff800001,0x7fc12345,0xffc12345,0x7fffffff,0xffffffff,0x7f800000,0xff800000,0x00000000,0x80000000,0x3f800000,0xbf800000,0x00800000,0x80800000,0x00000001,0x80000001]
    nums=[v if v<2**31 else v-2**32 for v in patterns]
    bits=torch.tensor(nums,dtype=torch.int32)
    aa=bits.repeat_interleave(16).view(torch.float32)
    bb=bits.repeat(16).view(torch.float32)
    fixture = root/'evidence/r41/chat/speed/nan_reference_gb10_20261031.txt'
    words = [int(x, 16) for x in fixture.read_text().split()]
    if len(words) != 256:
        raise ValueError('locked NaN reference must contain 256 words')
    want = [x if x < 2**31 else x - 2**32 for x in words]
    result=aot.compile_aot(divide,signature={'A':'*fp32','B':'*fp32','C':'*fp32','BLOCK':'constexpr'},constexprs={'BLOCK':256},profile='core-vrf-sync-v5-x8-g2c4-dp',options={'index_opt':1},output=str(Path(tempfile.mkdtemp(prefix='r41-div-nan-'))/'divide.tnca'))
    with Session(profile='core-vrf-sync-v5-x8-g2c4-dp',backend='systemc-tlm-grid',host_timeout=180) as sess:
        module=sess.load(result.path); x,y,z=[sess.allocate(1024) for _ in range(3)]
        sess.upload(x,aa.numpy().tobytes()); sess.upload(y,bb.numpy().tobytes())
        before=sess.dev.counters_snapshot(); event=sess.submit(module,{'A':x,'B':y,'C':z}); sess.wait(event);after=sess.dev.counters_snapshot()
        if not event.ok: raise RuntimeError(str(event.error))
        got=struct.unpack('<256i',sess.download(z)); mismatches=[i for i,(a,b) in enumerate(zip(got,want)) if a!=b]
        ins=sum(v-before.get(k,0) for k,v in after.items() if k.startswith('core_') and k.endswith('_instructions'))
        print(json.dumps(dict(domain='DP TLM',count=len(got),bit_mismatches=len(mismatches),first_mismatches=mismatches[:16],cycles=event.device_cycles,instructions=ins,launches=1,loads=1)))
        for i in mismatches: print(json.dumps(dict(i=i,a=float(aa[i]),b=float(bb[i]),want=f'{want[i]&0xffffffff:08x}',got=f'{got[i]&0xffffffff:08x}')),flush=True)
        if mismatches: raise RuntimeError('IEEE reference mismatch')
if __name__=='__main__':main()
