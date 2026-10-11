#!/usr/bin/env python3
"""Standard native Torch GEMM graphs with logical tiles over MMA fragments."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--columns', type=int, choices=(16,32,64), default=32)
args = ap.parse_args()
os.environ['TNCA_GEMM_TILE'] = f'16,{args.columns},16'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(47)
    for name, fn in [('mm', lambda a,b,c: a @ b),
                     ('addmm', lambda a,b,c: torch.addmm(c,a,b))]:
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        for m,n,k in ((17,65,33),(33,31,65)):
            a,b,c = (torch.randn(m,k).half()*.1, torch.randn(k,n).half()*.1,
                     torch.randn(n).half()*.1)
            da,db,dc = (t.to('tnca') for t in (a,b,c))
            with launch_log() as log:
                out = compiled(da,db,dc)
            torch.testing.assert_close(out.cpu(),fn(a,b,c),rtol=2e-3,atol=2e-3)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            if args.columns > 16:
                plans = [artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance']
                         .get('register_blocks',{}) for e in log]
                assert any([16,args.columns,16] in p.get('gemm_tiles',[]) for p in plans), plans
                print(f'AUDIT logical GEMM tiles: {plans}',flush=True)
            print(f'PASS {name}, logical 16x{args.columns}x16, shape {m}x{n}x{k}',flush=True)
    print('r40-gemm-tiles-tests PASS (4/4)',flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
