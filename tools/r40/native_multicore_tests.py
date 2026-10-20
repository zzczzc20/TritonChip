#!/usr/bin/env python3
"""Standard resident Torch graph on the real multicore mailbox RTL backend."""
import os
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(path))
os.environ.setdefault('TNCA_PROFILE','core-vrf-sync-v5')
os.environ.setdefault('TNCA_BACKEND','sim-multicore')
parser=argparse.ArgumentParser()
parser.add_argument('--cores',type=int,choices=(1,2,3,4,8),default=3)
args=parser.parse_args()
os.environ.setdefault('TNCA_BACKEND_ARGS',json.dumps({'core_count':args.cores}))
os.environ.setdefault('TNCA_RCTX_SPMD','0')
os.environ.setdefault('TNCA_HOST_TIMEOUT','1800')
import torch
import triton
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log

def main():
    active=int(os.environ.get('TNCA_ACTIVE_CORES',args.cores))
    assert 1<=active<=args.cores
    D.register();torch.manual_seed(40)
    x,y=torch.randn(4097),torch.randn(4097)
    dx,dy=x.to('tnca'),y.to('tnca')
    fn=lambda a,b:(a+b)*0.5
    compiled=torch.compile(fn,fullgraph=True,dynamic=False)
    warm=compiled(dx,dy)
    torch.testing.assert_close(warm.cpu(),fn(x,y));del warm
    session=D.MEMORY.session();before=len(session.dev.history)
    with launch_log() as launches:out=compiled(dx,dy)
    torch.testing.assert_close(out.cpu(),fn(x,y))
    actual=[record for record in session.dev.history[before:] if record['op']==11]
    assert launches and len(actual)==len(launches)
    assert all(record['upload_bytes']==record['download_bytes']==0 for record in launches)
    assert not any(record['op'] in (3,9) for record in session.dev.history[before:])
    assert all(record['grid'][0]*record['grid'][1]*record['grid'][2]>=args.cores for record in launches)
    assert session.physical_grid and session._physical_core_count==args.cores
    metrics=session.dev.physical_grid_metrics()
    programs=launches[-1]['grid'][0]*launches[-1]['grid'][1]*launches[-1]['grid'][2]
    assert metrics['peak_active']==active,metrics
    assert sum(row['issued'] for row in metrics['cores'])==programs,metrics
    assert all(row['issued']==row['completed'] and row['occupied_cycles']>0
               for row in metrics['cores'][:active]),metrics
    assert all(row['issued']==row['completed']==row['occupied_cycles']==0
               for row in metrics['cores'][active:]),metrics
    print('HARDWARE MULTICORE METRICS '+json.dumps(metrics,sort_keys=True),flush=True)
    print(f'NATIVE MULTICORE TORCH PASS cores={args.cores} active={active} elements=4097 kernels={len(launches)} whole_grid_launches={len(actual)} tensor_staging=0',flush=True)
if __name__=='__main__':main()
