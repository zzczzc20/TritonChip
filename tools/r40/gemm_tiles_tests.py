#!/usr/bin/env python3
"""Standard native Torch GEMM graphs with logical tiles over MMA fragments."""
import argparse
import atexit
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
ap.add_argument('--dump-dir', type=Path,
                help='retain compiler stages and artifacts in a fresh directory')
args = ap.parse_args()
os.environ['TNCA_GEMM_TILE'] = f'16,{args.columns},{args.depth}'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log
from tnca_isa import decode_fields, MAJ, MMA_K, MMA_DT


def main():
    if args.dump_dir is None:
        dump = tempfile.mkdtemp(prefix='r40-gemm-asm-')
        atexit.register(shutil.rmtree, dump)
    else:
        if args.dump_dir.exists() and any(args.dump_dir.iterdir()):
            ap.error('--dump-dir must be empty; preserve earlier attempts')
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        dump = str(args.dump_dir.resolve())
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
            # Compilation can execute native copy kernels while initializing
            # autotuner inputs. Check them outside the steady GEMM ledger.
            warm = compiled(da,db,dc)
            torch.testing.assert_close(warm.cpu(),fn(a,b,c),rtol=2e-3,atol=2e-3)
            del warm
            with launch_log() as log:
                out = compiled(da,db,dc)
            torch.testing.assert_close(out.cpu(),fn(a,b,c),rtol=2e-3,atol=2e-3)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            for entry in log:
                manifest, segments = artifact.load(entry['artifact_path'])
                prov = manifest['optional_metadata']['provenance']
                code = segments['text']
                fields = [decode_fields(int.from_bytes(code[i:i + 8], 'little'))
                          for i in range(0, len(code), 8)]
                commands = [f for f in fields if f['major'] == MAJ['MATRIX'] and f['func'] == 1]
                dtype_code = MMA_DT['bf16' if args.dtype == 'bfloat16' else 'f16']
                retiles = prov.get('mma_retiles', [])
                physical_depth = next((step for logical, step in retiles
                                       if logical == args.depth), args.depth)
                assert args.depth % physical_depth == 0
                assert len(commands) % (args.depth // physical_depth) == 0
                assert commands and all((f['mod'] & 3) == MMA_K[f'k{physical_depth}'] and
                    ((f['mod'] >> 2) & 3) == dtype_code for f in commands), commands
                if int(os.environ.get('TNCA_WINDOW_REUSE', '0')):
                    layout = prov['window_lifetime']
                    assert layout['reserved_registers'] <= layout['logical_registers']
                    definitions = [f for f in fields
                                   if f['major'] == MAJ['TDESC'] and f['func'] == 0x10]
                    actual = {(f['d'], f['b'], f['c']) for f in definitions}
                    assert actual == {(w['window'], w['start'], w['count'])
                                      for w in layout['layouts']}, actual
                    registers = {r for _, start, count in actual
                                 for r in range(start, start + count)}
                    assert len(registers) == layout['reserved_registers']
                    assert all(n >= len(registers) for n in
                               manifest['resources']['rf_private_per_role'].values())
                    print(f'AUDIT awaited window layout: {layout}', flush=True)
                print(f'AUDIT logical K={args.depth}, physical MMA K={physical_depth}, '
                      f'retiles={retiles}: {manifest["resources"]}', flush=True)
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
