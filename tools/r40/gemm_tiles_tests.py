#!/usr/bin/env python3
"""Standard native Torch GEMM graphs with logical tiles over MMA fragments."""
import argparse
import atexit
import hashlib
import os
from pathlib import Path
import sys
import shutil
import tempfile
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(path))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--columns', type=int, choices=(16,32,64), default=32)
ap.add_argument('--depth', type=int, choices=(16,32,64), default=16)
ap.add_argument('--dtype', choices=('float16', 'bfloat16'), default='float16')
args = ap.parse_args()
os.environ['TNCA_GEMM_TILE'] = f'16,{args.columns},{args.depth}'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log


def main():
    dump = tempfile.mkdtemp(prefix='r40-gemm-asm-')
    atexit.register(shutil.rmtree, dump)
    os.environ['TRITON_CACHE_DIR'] = os.path.join(dump, 'cache')
    os.environ['TNCA_KERNEL_DUMP'] = '1'
    os.environ['TNCA_DUMP_DIR'] = dump
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(47)
    for name, fn in [('mm', lambda a,b,c: a @ b),
                     ('addmm', lambda a,b,c: torch.addmm(c,a,b))]:
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        for m,n,k in ((17,65,33),(33,31,65)):
            dtype = getattr(torch, args.dtype)
            a,b,c = (torch.randn(m,k).to(dtype)*.1, torch.randn(k,n).to(dtype)*.1,
                     torch.randn(n).to(dtype)*.1)
            da,db,dc = (t.to('tnca') for t in (a,b,c))
            with launch_log() as log:
                out = compiled(da,db,dc)
            torch.testing.assert_close(out.cpu(),fn(a,b,c),rtol=2e-3,atol=2e-3)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            for entry in log:
                manifest, _ = artifact.load(entry['artifact_path'])
                prov = manifest['optional_metadata']['provenance']
                paths = list(Path(dump).rglob(entry['kernel'] + '.s'))
                assert len(paths) == 1, paths
                asm = paths[0].read_text()
                assert hashlib.sha256(asm.encode()).hexdigest() == prov['assembly_sha256']
                assert f'm.mma.async.k{args.depth}.' in asm, 'requested K was not encoded'
                print(f'AUDIT physical MMA K={args.depth}: {manifest["resources"]}', flush=True)
            if args.columns > 16 or args.depth > 16:
                plans = [artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance']
                         .get('register_blocks',{}) for e in log]
                assert any([16,args.columns,args.depth] in p.get('gemm_tiles',[]) for p in plans), plans
                print(f'AUDIT logical GEMM tiles: {plans}',flush=True)
            print(f'PASS {name}, logical 16x{args.columns}x{args.depth}, shape {m}x{n}x{k}',flush=True)
    print('r40-gemm-tiles-tests PASS (4/4)',flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
