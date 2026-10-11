#!/usr/bin/env python3
"""Native storage identity when recovery reuses the same physical address."""
import gc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device


def refused(fn):
    try:
        fn()
    except Exception as e:
        assert 'stale' in str(e) or 'epoch' in str(e), e
    else:
        raise AssertionError('stale storage/event accepted')


def main():
    if os.environ.get('TNCA_BACKEND', 'sim-file') != 'sim-file':
        raise RuntimeError('recovery ownership check uses an isolated RTL simulator')
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    h = torch.arange(16, dtype=torch.float32)
    graph = torch.compile(lambda t: t * 2 + 1, fullgraph=True)
    x = h.to('tnca')
    torch.testing.assert_close(graph(x).cpu(), h * 2 + 1)
    for _ in range(3):
        old, alias = x, x.view(4, 4)
        addr = old.untyped_storage().data_ptr()
        event = torch.tnca.Event()
        event.record()
        s = D.MEMORY.session()
        s.recover()
        x = h.to('tnca')
        assert x.untyped_storage().data_ptr() == addr, 'test did not force address reuse'
        refused(event.query)
        refused(old.cpu)
        refused(alias.cpu)
        refused(lambda: graph(old))
        refused(lambda: old.resize_(32))
        del old, alias
        gc.collect()
        assert D.MEMORY.buffers[addr].alive, 'old destructor retired new storage'
        torch.testing.assert_close(graph(x).cpu(), h * 2 + 1)
    print('PASS 3 recovery epochs with forced physical address reuse')
    print('PASS stale event, copy, view, launch and growing resize rejected')
    print('PASS old storage destruction preserves the new allocation at the same address')
    print('r40-recovery-tests PASS (3/3)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
