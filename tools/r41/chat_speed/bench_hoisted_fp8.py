#!/usr/bin/env python3
"""Compare official fused FP8 with exact separately quantised FP8 on DP TLM.

Diagnostic only: TNCA_IEEE_DIV=1 is necessary for the standalone v4 quantiser.
Each route runs in a fresh process and has a cold and a steady measured call.
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm'),
                str(ROOT/'tools/performance_harness'), str(ROOT/'tools/performance_harness/systemc')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--route',choices=('fused','hoisted'),required=True)
    p.add_argument('--m',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--zero-row',action='store_true')
    p.add_argument('--default-div',action='store_true');a=p.parse_args()
    if a.k%128: raise ValueError('this first diagnostic requires K divisible by 128')
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1', TNCA_TRITON='1',
      TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp', TNCA_BACKEND='systemc-tlm-grid',
      TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR=str(a.out/'dump'),TRITON_CACHE_DIR=str(a.out/'triton'),
      TORCHINDUCTOR_CACHE_DIR=str(a.out/'inductor'))
    if a.default_div:
        os.environ.pop('TNCA_IEEE_DIV', None)
    else:
        os.environ['TNCA_IEEE_DIV'] = '1'
    os.environ.setdefault('TNCA_DEADLINE','40000000000');os.environ.setdefault('TNCA_HOST_TIMEOUT','40000')
    import torch, triton, kernels, tlm_backend
    import tnca.torch_device as td
    import fp8_block_reference as ref
    from tnca.runtime import Session
    from tnca.triton_backend import driver
    tlm_backend.register();td.register();torch.set_num_threads(1);ref.set_dot_mode('mma.dp')
    torch.manual_seed(a.m*7919+a.n*31+a.k)
    A=torch.randn(a.m,a.k,dtype=torch.bfloat16)
    if a.zero_row:A[0].zero_()
    W=(torch.randn(a.n,a.k)*0.05).to(torch.float8_e4m3fn)
    Ws=torch.rand((a.n+127)//128,(a.k+127)//128)*0.01+0.001
    want=ref.block_dynamic(A,W,Ws,[128,128],torch.bfloat16,ieee=True)
    q,scale=ref.act_quant(A,128,ieee=True,clamp=True)
    ref2=ref.block_prequantised(q,W,scale,Ws,[128,128],torch.bfloat16)
    if not torch.equal(want.view(torch.int16),ref2.view(torch.int16)):
        raise RuntimeError('CPU hoisted algebra does not match fused reference bitwise')
    # The registered v4 snapshot; explicit revision also works fully offline.
    k4=kernels.get_kernel('kernels-community/finegrained-fp8',
                          revision='0e5764aa5c27d0f005117a92645156481540950a')
    k1=kernels.get_local_kernel(a.v1,trust_remote_code=True) if a.route=='hoisted' else None
    x,w,ws=A.to('tnca'),W.to('tnca'),Ws.to('tnca')
    sess=td.MEMORY.session(0)
    def compute():
        if a.route=='fused': return k4.matmul_2d(x,w,ws,[128,128],torch.bfloat16)
        qd,sd=k4.fp8_act_quant(x,128)
        return k1.w8a8_block_fp8_matmul(qd,w,sd,ws,[128,128],torch.bfloat16)
    loads=0;orig=Session.load
    def counted(self,*args,**kwargs):
        nonlocal loads
        result=orig(self,*args,**kwargs);loads+=1;return result
    Session.load=counted
    result=dict(schema='r41-chat-speed-hoist/1',domain='uncalibrated SystemC/TLM; not board',route=a.route,
       shape=[a.m,a.n,a.k],zero_row=a.zero_row,profile=os.environ['TNCA_PROFILE'],
       ieee_div_override='default' if a.default_div else 'TNCA_IEEE_DIV=1: standalone quantisation diagnostic',
       x8_pipeline=os.environ.get('TNCA_X8_PIPELINE','default'),
       stage_first=os.environ.get('TNCA_X8_STAGE_FIRST','default'),cpu_refs_bit_equal=True,calls=[])
    try:
        for iteration in range(2):
            sess.drain();before=sess.dev.counters_snapshot();oldloads=loads;ev0=driver.evicted;start=time.monotonic()
            with driver.launch_log() as log: y=compute()
            sess.drain();after=sess.dev.counters_snapshot();elapsed=time.monotonic()-start
            got=y.cpu();bad=int((got.contiguous().view(torch.int16)!=want.contiguous().view(torch.int16)).sum())
            keys=[k for k in after if k.startswith('core_') and k.endswith('_instructions')]
            instr=sum(after[k]-before.get(k,0) for k in keys)
            measured=[e for e in log if not e.get('skipped')]
            if any(e.get('device_cycles') is None for e in measured):
                raise RuntimeError('a device launch has no recorded cycles: '+repr(measured))
            row=dict(iteration=iteration,phase='cold' if iteration==0 else 'steady',wall_seconds=elapsed,
                bit_mismatches=bad,output_elements=got.numel(),macs=a.m*a.n*a.k,
                instructions=instr,instructions_per_mac=instr/(a.m*a.n*a.k),
                launches=len(measured),device_cycles=sum(e['device_cycles'] for e in measured),
                code_loads=loads-oldloads,code_evictions=driver.evicted-ev0,
                skipped_compile_launches=len(log)-len(measured),
                kernels=[dict(name=e['kernel'],cycles=e['device_cycles'],grid=list(e['grid'])) for e in measured])
            result['calls'].append(row);print(json.dumps(row),flush=True)
            (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            if bad:raise RuntimeError('FP8 output differs from exact reference')
    finally: Session.load=orig
    result['status']='pass';(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
