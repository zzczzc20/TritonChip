#!/usr/bin/env python3
"""Deferred storage recycling while real RTL tasks retain buffer leases."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'tools/r2', ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
import runtime_tests as fixture
from tnca import device
from tnca.errors import TncaError


def main():
    device.prepare('core-sim')
    with fixture.session() as s:
        mod = s.load(fixture.artifact_path())
        x, y, out = [s.allocate(64) for _ in range(3)]
        s.upload(x, fixture.f32s(list(range(16))))
        s.upload(y, fixture.f32s([3] * 16))
        s._stop.set()
        s._worker.join(timeout=10)
        assert not s._worker.is_alive(), 'worker did not enter the controlled hold'
        args = dict(x_ptr=x, y_ptr=y, o_ptr=out, n=16, alpha=2.0)
        first = s.submit(mod, args)
        last = s.submit(mod, args, depends_on=[first])
        assert x.leases == 2
        s.retire(x)
        reserved = s.allocate(64)
        assert reserved.addr != x.addr, 'retired storage recycled with live leases'
        try:
            s.submit(mod, args)
        except TncaError:
            pass
        else:
            raise AssertionError('retired buffer accepted by a new task')
        s._start_worker()
        s.wait(last)
        assert first.ok and last.ok, (first, last)
        assert fixture.unpack(s.download(out)) == [2 * i + 3 for i in range(16)]
        recycled = s.allocate(64)
        assert recycled.addr == x.addr, 'storage not reclaimed after final lease'
        assert x.leases == 0
        print('PASS retiring a buffer preserves two queued RTL tasks and delays address reuse')
        print('PASS new use of retired storage is refused, final lease recycles its address')
        for b in (y, out, reserved, recycled):
            s.free(b)
    print('r40-lifetime-tests PASS (2/2)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
