#!/usr/bin/env python3
"""Run exact FP16 descriptor GEMM via the optimized TNCA-IR AOT path."""
import argparse,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(ROOT/'tools/performance_harness/systemc')]

def descriptor(base,extent,strides,block):
 data=bytearray(256);struct.pack_into('<Q',data,0,base)
 for i,v in enumerate(extent):struct.pack_into('<Q',data,8+8*i,v)
 for i,v in enumerate(strides):struct.pack_into('<q',data,48+8*i,v)
 for i,v in enumerate(block):struct.pack_into('<I',data,88+4*i,v)
 struct.pack_into('4B',data,108,len(extent),0,0,0);struct.pack_into('<QQ',data,112,7319,7)
 return bytes(data)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import tlm_backend
 from tnca import aot
 from tnca.runtime import Session
 from tnca_cc.gemm import plan_gemm,build_gemm
 tlm_backend.register();profile='core-vrf-sync-v5-x8-g2c4-dp';m=n=16;k=32
 abits=[0x3c00 if row%2==0 else 0xbc00 for row in range(m) for _ in range(k)]
 bbits=[0x3800]*(k*n);want=[0x41800000 if row%2==0 else 0xc1800000 for row in range(m) for _ in range(n)]
 rows=[]
 for opt in (0,1,2):
  with Session(profile=profile,backend='systemc-tlm-grid',host_timeout=300) as s:
   da,db,aa,bb,out=[s.allocate(sz) for sz in (256,256,m*k*2,k*n*2,m*n*4)]
   def program(b,target):
    plan=plan_gemm(target,m=m,n=n,k=k,dtype='f16')
    build_gemm(target,plan,name='ir_descriptor_gemm',a_desc_addr=da.addr,b_desc_addr=db.addr,out_addr=out.addr,builder=b);b.exit()
   art=aot.compile_aot(program,kernel_name='ir_descriptor_gemm',signature={},source_kind='tnca-ir',profile=profile,options={'opt_level':opt},output=str(a.out/f'opt{opt}.tnca'))
   mod=s.load(art.path);s.upload(aa,struct.pack('<%dH'%len(abits),*abits));s.upload(bb,struct.pack('<%dH'%len(bbits),*bbits))
   s.upload(da,descriptor(aa.addr,[m,k],[k*2,2],[m,16]));s.upload(db,descriptor(bb.addr,[k,n],[n*2,2],[16,n]))
   before=s.dev.counters_snapshot();ev=s.submit(mod,{});s.wait(ev);after=s.dev.counters_snapshot()
   if not ev.ok:raise RuntimeError(str(ev.error))
   got=struct.unpack('<256I',s.download(out));bad=sum(x!=y for x,y in zip(got,want));ins=sum(after[x]-before.get(x,0) for x in after if x.startswith('core_') and x.endswith('_instructions'))
   rows.append(dict(opt_level=opt,bit_mismatches=bad,elements=len(got),device_cycles=ev.device_cycles,instructions_per_mac=ins/(m*n*k),launches=1,code_loads_during_launch=0))
   if bad:raise RuntimeError(f'opt={opt}: {bad} bit mismatches')
 result=dict(schema='r41-tnca-ir-fp16-gemm/1',status='pass',domain='DP SystemC/TLM; not board',shape=[m,n,k],results=rows)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
