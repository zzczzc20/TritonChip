#!/usr/bin/env python3
"""Standard Torch graph endpoints through the board MMIO RTL binding.

This is a correctness check, not protocol-2 performance measurement.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v2')
os.environ.setdefault('TNCA_BACKEND', 'fpga-u50-sim')
os.environ.setdefault('TNCA_BACKEND_ARGS', '{"device_clock":true}')
os.environ.setdefault('TNCA_HOST_TIMEOUT', '172800')
import torch
import triton  # noqa: F401; load the plugin before importing its driver
import tnca.torch_device as D
from tnca.device_timing import clock_span
from tnca.triton_backend.driver import launch_log


def main():
    if os.environ.get('TNCA_LAUNCH') == 'skip':
        raise RuntimeError('actual MMIO execution is required')
    D.register()
    torch.manual_seed(42)
    a, b = torch.randn(4, 16).half() * .1, torch.randn(16, 4).half() * .1
    da, db = a.to('tnca'), b.to('tnca')
    fn = lambda x, y: (x @ y).sum(-1)
    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    with torch.no_grad():
        warm = compiled(da, db)
        torch.testing.assert_close(warm.cpu(), fn(a, b), rtol=2e-3, atol=2e-3)
        del warm
        for mode in ('single', 'batched'):
            os.environ['TNCA_SUBMIT'] = mode
            with launch_log() as launches:
                out = compiled(da, db)
            torch.testing.assert_close(out.cpu(), fn(a, b), rtol=2e-3, atol=2e-3)
            assert len(launches) >= 2, 'the graph must exercise dependent kernels'
            assert all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
            span = clock_span([e['device_timing'] for e in launches])
            assert span is not None, 'the image/backend has no common clock endpoints'
            active = sum(e['device_cycles'] for e in launches)
            assert span['cycles'] > active, 'submission gaps were lost'
            print(f'PASS {mode}: {len(launches)} kernels, active={active}, '
                  f'graph_span={span["cycles"]}, epoch={span["epoch"]}', flush=True)
    print('r40-device-clock-native-tests PASS (2/2)', flush=True)


if __name__ == '__main__':
    main()
