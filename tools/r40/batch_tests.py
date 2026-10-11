#!/usr/bin/env python3
"""Queued standard Torch graphs, events, storage retirement and bounded stress."""
import argparse
import gc
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calls', type=int, default=1000)
    args = ap.parse_args()
    if args.calls < 1:
        ap.error('--calls must be positive')
    if os.environ.get('TNCA_BACKEND', 'sim-file') in ('sim-file', 'sim-pipe'):
        device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    torch.manual_seed(42)
    h, w = torch.randn(16, 16).half(), (torch.randn(16, 16) * 0.1).half()
    x, dw = h.to('tnca'), w.to('tnca')
    def fn(t, weight):
        return torch.softmax(torch.nn.functional.layer_norm(t, (16,)) @ weight, -1)
    graph = torch.compile(fn, fullgraph=True)
    os.environ['TNCA_SUBMIT'] = 'single'
    with launch_log() as serial:
        y = graph(x, dw)
    torch.testing.assert_close(y.cpu(), fn(h, w), rtol=2e-3, atol=2e-3)
    assert len(serial) >= 3, f'chain fused to {len(serial)} kernels; no batched graph tested'
    s = D.MEMORY.session()
    s._stop.set()
    s._worker.join(timeout=10)
    assert not s._worker.is_alive()
    try:
        os.environ['TNCA_SUBMIT'] = 'batched'
        y = graph(x, dw)
        event = torch.tnca.Event()
        event.record()
        assert not event.query(), 'queued graph event reports completion while worker is held'
        retired = [b for b in s._buffers.values() if not b.alive and b.leases]
        assert retired, 'graph did not exercise free of an unfinished intermediate'
        probe = torch.empty(256, device='tnca')
        assert probe.data_ptr() not in {b.addr for b in retired}, 'unfinished storage recycled'
    finally:
        s._start_worker()
    event.synchronize()
    assert event.query()
    torch.tnca.Stream().wait_event(event)
    torch.testing.assert_close(y.cpu(), fn(h, w), rtol=2e-3, atol=2e-3)
    print('PASS queued multi-kernel graph, event visibility and deferred intermediate free', flush=True)
    t0 = time.monotonic()
    with launch_log() as batched:
        y = graph(x, dw)
    elapsed = time.monotonic() - t0
    assert [e['artifact'] for e in batched] == [e['artifact'] for e in serial]
    assert all(e['device_cycles'] > 0 and '_event' not in e for e in batched)
    assert all(e['upload_bytes'] == e['download_bytes'] == 0 for e in batched)
    assert sum(e['host_seconds']['run'] for e in batched) <= elapsed + 0.01
    print('PASS completed launch log, unchanged artifacts and zero kernel transfers', flush=True)
    first_stream, second_stream = torch.tnca.Stream(), torch.tnca.Stream()
    default = torch.tnca.current_stream()
    with torch.tnca.stream(first_stream):
        assert torch.tnca.current_stream() is first_stream
        y = graph(x, dw)
        boundary = first_stream.record_event()
    assert torch.tnca.current_stream() is default
    with torch.tnca.stream(second_stream):
        second_stream.wait_event(boundary)
        with launch_log() as other_stream:
            y = graph(x, dw)
    assert all(e['stream_id'] == second_stream.stream_id for e in other_stream), other_stream
    torch.testing.assert_close(y.cpu(), fn(h, w), rtol=2e-3, atol=2e-3)
    print('PASS stream contexts, recorded dependency, runtime stream identity and restoration', flush=True)
    fast = torch.compile(lambda t: t * 2 + 1, fullgraph=True)
    fast(x).cpu()  # compile before the allocator stress
    barrier = threading.Barrier(2)
    results, failures = [], []
    def concurrent(call, expected):
        try:
            with launch_log() as entries:
                barrier.wait(timeout=30)
                out = call()
            torch.testing.assert_close(out.cpu(), expected, rtol=2e-3, atol=2e-3)
            results.append([e['artifact'] for e in entries])
        except BaseException as exc:
            failures.append(exc)
    workers = [threading.Thread(target=concurrent, args=(lambda: graph(x, dw), fn(h, w))),
               threading.Thread(target=concurrent, args=(lambda: fast(x), h * 2 + 1))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    assert not any(worker.is_alive() for worker in workers), 'concurrent graph stalled'
    assert not failures, failures
    assert sorted(map(len, results)) == [1, len(serial)], results
    print('PASS concurrent graph execution and thread-local launch audit', flush=True)
    gc.collect()
    torch.tnca.synchronize()
    live = D.MEMORY.live
    for _ in range(args.calls):
        z = fast(x)
    torch.testing.assert_close(z.cpu(), h * 2 + 1)
    del z
    gc.collect()
    torch.tnca.synchronize()
    assert D.MEMORY.live == live, (D.MEMORY.live, live)
    assert not s.unfinished and all(not b.leases for b in s._buffers.values())
    print(f'PASS {args.calls} graph calls, bounded queue, final storage/lease audit', flush=True)
    print(f'AUDIT storage_live_bytes={D.MEMORY.live} storage_peak_bytes={D.MEMORY.peak} '
          f'live_buffers={sum(b.alive for b in s._buffers.values())} '
          f'leases={sum(b.leases for b in s._buffers.values())} unfinished={len(s.unfinished)}', flush=True)
    print('r40-batch-tests PASS (5/5)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
