#!/usr/bin/env python3
"""Real native kernels: cold compiler copies and strict steady launch audit."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v2')
import torch
import triton
from tnca import device
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log
from archbench.platforms.tnca import _steady


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    h = torch.arange(64).reshape(4, 16).float()
    x = h.to('tnca')
    primary = torch.compile(lambda t: t + 1, fullgraph=True, dynamic=False)
    decoy = torch.compile(lambda t: t * 2, fullgraph=True, dynamic=False)
    with torch.no_grad(), launch_log() as cold:
        y = primary(x)
    torch.testing.assert_close(y.cpu(), h + 1)
    assert len(cold) > 1 and any(e['kernel'] == '_copy_kernel' for e in cold), cold
    print(f'PASS cold call retains {len(cold)} launches including compiler literal copies', flush=True)
    good = _steady(primary, (x,), 3, 1, launch_log, torch)
    assert all(c['launches'] == 1 and not c.get('problem') for c in good)
    torch.testing.assert_close(good[-1]['out'].cpu(), h + 1)
    print('PASS steady calls execute exactly one graph kernel; numerical reference matches', flush=True)
    with torch.no_grad():
        decoy(x)  # compile before the negative audit
    def extra(t):
        result = primary(t)
        decoy(t)  # an actual additional device kernel, not a synthetic log entry
        return result
    bad = _steady(extra, (x,), 2, 1, launch_log, torch)
    assert all(c['launches'] == 2 and c.get('problem') for c in bad)
    torch.testing.assert_close(bad[-1]['out'].cpu(), h + 1)
    print('PASS an extra physical kernel is refused despite correct graph output', flush=True)
    print('r40-cold-scope-tests PASS (3/3)', flush=True)


if __name__ == '__main__':
    main()
