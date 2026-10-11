#!/usr/bin/env python3
"""Guarded replay captured from real compiled native Torch graphs."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
os.environ['TNCA_SUBMIT'] = 'batched'
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log, TNCALauncher


def refused(g, s):
    previous = s.latest_event
    try:
        g.replay()
    except RuntimeError as exc:
        assert 'guard' in str(exc) or 'invalid' in str(exc), exc
    else:
        raise AssertionError('invalid replay accepted')
    assert s.latest_event is previous, 'guard failure admitted a partial graph'


def main():
    if os.environ.get('TNCA_BACKEND', 'sim-file') != 'sim-file':
        raise RuntimeError('replay recovery check uses an isolated RTL simulator')
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(43)
    h = torch.randn(16, 16).half()
    hw = (torch.randn(16, 16) * .1).half()
    x, w = h.to('tnca'), hw.to('tnca')
    def fn(t, weight):
        return torch.softmax(torch.nn.functional.layer_norm(t, (16,)) @ weight, -1)
    compiled = torch.compile(fn, fullgraph=True)
    compiled(x, w).cpu()
    g = torch.tnca.Graph()
    with launch_log() as capture_log:
        with torch.tnca.graph(g):
            y = compiled(x, w)
    assert len(g.commands) >= 3
    s = D.MEMORY.session()
    live = D.MEMORY.live
    original = TNCALauncher.__call__
    def deny(*args, **kwargs):
        raise AssertionError('replay reentered Triton wrapper')
    TNCALauncher.__call__ = deny
    try:
        for i in range(3):
            updated = h + i * .125
            x.copy_(updated)
            with launch_log() as replay_log:
                event = g.replay()
            assert event.ok
            assert [e['artifact'] for e in replay_log] == [e['artifact'] for e in capture_log]
            assert all(e['replay'] and e['device_cycles'] > 0 and
                       e['upload_bytes'] == e['download_bytes'] == 0 for e in replay_log)
            torch.testing.assert_close(y.cpu(), fn(updated, hw), rtol=2e-3, atol=2e-3)
    finally:
        TNCALauncher.__call__ = original
    assert D.MEMORY.live == live
    print('PASS retained multi-kernel Torch graph replay, updated input, artifact and byte audit', flush=True)
    x.resize_(8, 32)
    refused(g, s)
    x.resize_(16, 16)
    refused(g, s)  # invalidation is permanent until recapture
    print('PASS shape/stride guard and permanent invalidation before admission', flush=True)
    g.reset()
    with torch.tnca.graph(g):
        y = compiled(x, w)
    replacement = h.to('tnca')
    x.set_(replacement.untyped_storage(), 0, (16, 16), (16, 1))
    refused(g, s)
    print('PASS changed storage alias rejected before admission', flush=True)
    g.reset()
    with torch.tnca.graph(g):
        y = compiled(x, w)
    s.recover()
    refused(g, s)
    print('PASS recovery epoch rejects whole graph without submission', flush=True)
    g.reset()
    print('r40-replay-tests PASS (4/4)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
