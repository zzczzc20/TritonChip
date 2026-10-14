#!/usr/bin/env python3
"""Audit real ring buffers in standard native Torch GEMM; no perf sign-off."""
import argparse
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--stages', type=int, choices=(2, 3), required=True)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]
os.environ.update(TNCA_RCTX_ROLES='2', TNCA_RCTX_STAGES=str(args.stages))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')

import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import artifact, device
from tnca.triton_backend.driver import launch_log


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(47)
    count = 0
    for name, fn in [('mm', lambda a,b,c: a@b),
                     ('addmm', lambda a,b,c: torch.addmm(c,a,b))]:
        graph = torch.compile(fn, fullgraph=True, dynamic=False)
        for m,n,k in ((16,16,48), (17,31,65)):
            a,b,c = [torch.randn(*shape,dtype=torch.float16)*.1
                     for shape in ((m,k),(k,n),(n,))]
            da,db,dc = [x.to('tnca') for x in (a,b,c)]
            warm = graph(da,db,dc)
            torch.testing.assert_close(warm.cpu(), fn(a,b,c), rtol=2e-3, atol=2e-3)
            del warm
            with launch_log() as log:
                out = graph(da,db,dc)
            torch.testing.assert_close(out.cpu(), fn(a,b,c), rtol=2e-3, atol=2e-3)
            assert log and all(e['upload_bytes']==e['download_bytes']==0 for e in log)
            rings = []
            for entry in log:
                manifest,_ = artifact.load(entry['artifact_path'])
                prov = manifest['optional_metadata']['provenance']
                ring = prov.get('rctx_roles_ring')
                if ring:
                    assert ring['stages'] == args.stages, ring
                    assert ring['ring_windows'] >= args.stages, ring
                    assert len(manifest['execution']['program_to_role']['0']) == 2
                    rings.append(ring)
            assert rings, [(artifact.load(e['artifact_path'])[0]['optional_metadata']['provenance'])
                           for e in log]
            count += 1
            print(f'PASS {name} {m}x{n}x{k}, stages={args.stages}, rings={rings}',flush=True)
    print(f'r40-native-pipeline PASS ({count}/{count})',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
