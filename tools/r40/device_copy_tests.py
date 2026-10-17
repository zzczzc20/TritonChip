#!/usr/bin/env python3
"""Same-device copies preserve values and gaps without host tensor transfers."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm',
          ROOT / 'tools/r40'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log
from frozen_residency_tests import transfers


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    session = D.MEMORY.session()
    passed = 0

    def check(name, src, dst, whole, want):
        nonlocal passed
        previous_driver = triton.runtime.driver.active
        with transfers(session) as events, launch_log() as launches:
            dst.copy_(src)
        assert triton.runtime.driver.active is previous_driver, 'copy changed the active driver'
        assert not [e for e in events if e['tensor'] or e['staging']], events
        assert launches and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches)
        got = whole.cpu()
        assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)), name
        passed += 1
        print(f'PASS {name}: {len(launches)} native copy launch(es), tensor transfers=0', flush=True)

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                  torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool):
        host = torch.arange(-17, 18).to(dtype)
        source = host.to('tnca')
        dest = torch.empty_like(source)
        check(str(dtype), source, dest, dest, host)
    host = torch.arange(30).reshape(3, 10).float()
    source = host.to('tnca')
    dest = torch.full((3, 10), -7.).to('tnca')
    want = torch.full((3, 10), -7.)
    want[:, 2:8:2].copy_(host[:, 1:7:2])
    check('strided gaps', source[:, 1:7:2], dest[:, 2:8:2], dest, want)
    source = torch.tensor([[-3., 2., 5.]]).to('tnca')
    dest = torch.empty((3, 3), device='tnca')
    check('broadcast', source, dest, dest, torch.tensor([[-3., 2., 5.]]).expand(3, 3).clone())
    source = torch.arange(7).half().to('tnca')
    dest = torch.empty(7, dtype=torch.float32, device='tnca')
    check('dtype conversion', source, dest, dest, torch.arange(7).float())
    shared = torch.arange(40).to('tnca')
    want = torch.arange(40)
    want[24:32].copy_(want[4:12])
    check('disjoint storage aliases', shared[4:12], shared[24:32], shared, want)
    try:
        shared[1:9].copy_(shared[:8])
    except RuntimeError as exc:
        assert 'overlapping' in str(exc)
    else:
        raise AssertionError('overlapping copy was accepted')
    assert torch.equal(shared.cpu(), want), 'refusal changed the storage'
    resized = torch.arange(35).to('tnca')
    with transfers(session) as events:
        resized.resize_(70)
    assert not [e for e in events if e['tensor'] or e['staging']], events
    assert torch.equal(resized[:35].cpu(), torch.arange(35)), 'resize lost its prefix'
    passed += 1
    print('PASS storage resize: prefix preserved, tensor transfers=0', flush=True)
    print(f'r40-device-copy-tests PASS ({passed}/{passed}), overlap refused', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
