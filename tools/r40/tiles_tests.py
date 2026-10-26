#!/usr/bin/env python3
"""Standard Torch pointwise graphs through wide register-block lowering."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--width', type=int, choices=(16, 32, 64, 128), default=64)
args = ap.parse_args()
os.environ['TNCA_MAX_BLOCK'] = str(args.width)
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device, artifact
from tnca.triton_backend.driver import launch_log


def main():
    if os.environ.get('TNCA_BACKEND', 'sim-file') in ('sim-file', 'sim-pipe'):
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    fn = torch.compile(lambda x, y: torch.where(x > 0, x * y + 1, x - y), fullgraph=True)
    for n in (127, 17, 31, 33, 65):
        h = torch.linspace(-2, 2, n)
        w = torch.arange(n).float() * .125
        x, y = h.to('tnca'), w.to('tnca')
        with launch_log() as log:
            out = fn(x, y)
        torch.testing.assert_close(out.cpu(), torch.where(h > 0, h * w + 1, h - w))
        assert log and all(e['upload_bytes'] == e['download_bytes'] == 0 for e in log)
        if n == 127 and args.width > 16:
            manifests = [artifact.load(e['artifact_path'])[0] for e in log]
            assert any(args.width in m['optional_metadata']['provenance']
                       .get('register_blocks', {}).get('widths', []) for m in manifests), log
            for m in manifests:
                plan = m['optional_metadata']['provenance'].get('vector_spill', {})
                assert m['resources']['scratch_bytes'] >= plan.get('scratch_bytes', 0)
                print('AUDIT ' + json.dumps({'width': args.width, 'spill': plan,
                      'resources': m['resources']}, sort_keys=True), flush=True)
        print(f'PASS block {args.width}, tail {n}, masked select/load/store', flush=True)
    h = torch.arange(127, dtype=torch.int64)
    h = torch.where(h % 2 == 0, h + (1 << 35), -h - (1 << 35))
    w = torch.arange(127, dtype=torch.int64) + 1
    x, y = h.to('tnca'), w.to('tnca')
    with launch_log() as integer_log:
        out = fn(x, y)
    torch.testing.assert_close(out.cpu(), torch.where(h > 0, h * w + 1, h - w))
    for entry in integer_log:
        m = artifact.load(entry['artifact_path'])[0]
        plan = m['optional_metadata']['provenance'].get('vector_spill', {})
        assert m['resources']['scratch_bytes'] >= plan.get('scratch_bytes', 0)
        print('AUDIT ' + json.dumps({'width': args.width, 'dtype': 'int64', 'spill': plan,
              'resources': m['resources']}, sort_keys=True), flush=True)
    print(f'PASS block {args.width}, signed int64 values beyond 32 bits', flush=True)
    for dtype in (torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32):
        h = (torch.arange(127) % 16 - 8).to(dtype)
        w = (torch.arange(127) % 8 + 1).to(dtype)
        x, y = h.to('tnca'), w.to('tnca')
        with launch_log():
            out = fn(x, y)
        torch.testing.assert_close(out.cpu(), torch.where(h > 0, h * w + 1, h - w))
        print(f'PASS block {args.width}, {dtype}', flush=True)
    base = torch.arange(263).float() - 120
    db = base.to('tnca')
    view = torch.as_strided(db, (127,), (2,), 3)
    mutate = torch.compile(lambda t: t.add_(2), fullgraph=True)
    with launch_log():
        out = mutate(view)
    expected = base.clone()
    expected.as_strided((127,), (2,), 3).add_(2)
    torch.testing.assert_close(db.cpu(), expected)
    assert out.untyped_storage().data_ptr() == db.untyped_storage().data_ptr()
    print(f'PASS block {args.width}, strided offset alias and in-place stores', flush=True)
    print('r40-tiles-tests PASS (12/12)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
