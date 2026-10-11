#!/usr/bin/env python3
"""Standard native Torch reductions through register chunks."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--width', type=int, default=64, choices=(16, 32, 64, 128))
ap.add_argument('--rows', type=int, default=1, choices=(1, 2, 4, 8, 16))
ap.add_argument('--only', choices=('sum','amax','rmsnorm','softmax','layernorm'))
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
    torch.manual_seed(46)
    for name, fn in [('sum', lambda t: t.sum(-1)),
                     ('amax', lambda t: t.amax(-1)),
                     ('rmsnorm', lambda t: t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-5)),
                     ('softmax', lambda t: torch.softmax(t, -1)),
                     ('layernorm', lambda t: torch.nn.functional.layer_norm(t, (t.shape[-1],)))]:
        if args.only and name != args.only:
            continue
        compiled = torch.compile(fn, fullgraph=True)
        for n in (127, 17, 33, 65):
            h = torch.randn(max(3, args.rows + 1), n)
            x = h.to('tnca')
            with launch_log() as log:
                out = compiled(x)
            torch.testing.assert_close(out.cpu(), fn(h), rtol=3e-5, atol=3e-5)
            assert log and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in log)
            if n == 127 and args.rows > 1:
                shapes = [shape for e in log for shape in
                          artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance']
                          .get('reduction_tiles', [])]
                assert any(1 < shape[0] <= args.rows for shape in shapes), shapes
                print(f'AUDIT actual row tiles: {shapes}', flush=True)
            if n == 127 and args.width > 16:
                plans = [artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance']
                         .get('register_blocks', {}) for e in log]
                assert any(any(r[0] == args.width for r in p.get('reductions', [])) for p in plans), plans
                print(f'AUDIT {name} reductions: {plans}', flush=True)
            print(f'PASS {name}, width cap {args.width}, shape {tuple(h.shape)}', flush=True)
    print(f'r40-reduction-tests PASS ({4 if args.only else 20}/{4 if args.only else 20})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
