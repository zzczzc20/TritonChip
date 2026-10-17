#!/usr/bin/env python3
"""Native asynchronous allocation waits for retired buffers under pressure."""
import gc
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ['TNCA_SUBMIT'] = 'batched'
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch
import triton  # noqa: F401
import tnca.torch_device as D
from tnca import device


def main():
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    h = torch.arange(16384, dtype=torch.float32)
    x = h.to('tnca')
    fn = torch.compile(lambda t: t * 2 + 1, fullgraph=True)
    fn(x).cpu()
    s = D.MEMORY.session()
    waits = []
    wait = s.wait
    def measured(event, *args, **kwargs):
        waits.append((event.job_id, len(s.unfinished)))
        return wait(event, *args, **kwargs)
    s.wait = measured
    for _ in range(12):
        y = fn(x)
    torch.testing.assert_close(y.cpu(), h * 2 + 1)
    assert waits, 'test never exercised allocation backpressure'
    from tnca.runtime import MAX_UNFINISHED
    assert all(depth < MAX_UNFINISHED for _job, depth in waits), waits
    assert D.MEMORY.peak < s._heap.limit - s._heap.base
    del y
    gc.collect()
    torch.tnca.synchronize()
    assert not s.unfinished and all(not b.leases for b in s._buffers.values())
    assert D.MEMORY.live == x.untyped_storage().nbytes()
    print(f'PASS 12 native calls under a {s._heap.limit-s._heap.base}-byte heap, correct output')
    print(f'PASS {len(waits)} allocation waits below queue capacity; storage peak {D.MEMORY.peak} B, final {D.MEMORY.live} B')
    print(f'AUDIT {s.memory_stats}')
    s.wait = wait
    print('r40-allocation-pressure-tests PASS (2/2)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
