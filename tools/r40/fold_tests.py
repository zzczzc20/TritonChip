#!/usr/bin/env python3
"""Standard Torch multi-output reductions across register chunks."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(p))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--width', type=int, choices=(16, 32, 64, 128), default=64)
ap.add_argument('--rows', type=int, default=1, choices=(1,2,4,8,16))
ap.add_argument('--index-bias', type=int, default=0)
args = ap.parse_args()
os.environ['TNCA_REDUCTION_ROWS'] = str(args.rows)
os.environ['TNCA_MAX_BLOCK'] = str(args.width)
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    def graph(t):
        values, indices = torch.max(t, -1)
        return values, indices + args.index_bias if args.index_bias else indices
    fn = torch.compile(graph, fullgraph=True)
    for n in (127, 17, 33, 65):
        h = torch.arange(max(3,args.rows+1)*n, dtype=torch.float32).reshape(-1,n) % 7
        h[0, n//2] = float('nan')
        x = h.to('tnca')
        with launch_log() as log:
            values, indices = fn(x)
        expected = torch.max(h, -1)
        torch.testing.assert_close(values.cpu(), expected.values, equal_nan=True)
        torch.testing.assert_close(indices.cpu(), expected.indices + args.index_bias)
        assert log and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in log)
        if n == 127 and args.width > 16:
            plans = [artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance']
                     .get('register_blocks', {}) for e in log]
            assert any([args.width, 'body-fold', 1 if args.rows > 1 else 0] in p.get('reductions', []) for p in plans), plans
            print(f'AUDIT {plans}', flush=True)
        print(f'PASS values/index fold, width {args.width}, tail {n}, NaN and first tie', flush=True)
    print('r40-fold-tests PASS (4/4)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
