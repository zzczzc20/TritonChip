#!/usr/bin/env python3
"""R39-00: the ablation tool and ArchBench's record keys, tested on records
built here (no simulator, no board).

    python3 tools/r39/ablate_tests.py

1. Identical sides: S_geo is exactly 1.0 and nothing regresses.
2. A planted 5 % slowdown on one point is reported as a regression and fails
   the gate; listed with --explained it no longer does.
3. A uniform 1.25x speedup gives S_geo = 1.25 and passes a 1.2 gate, and
   fails a 1.3 one.
4. Two runs of one profile that differ only in the Inductor configuration,
   the knobs or the device build get different report keys, so neither
   replaces the other (the merge that report.load used to do).
5. A side holding two configurations is refused.
6. A point that passes on one side only fails the gate.
"""
import copy
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'r39'))

from archbench import registry, report          # noqa: E402
from archbench.bounds import VERSION            # noqa: E402
import ablate                                   # noqa: E402

PLATFORM = dict(kind='tnca', name='tnca:core-vrf-sync-v1@fpga-u280', profile='core-vrf-sync-v1',
                backend='fpga-u280', opt_level=2,
                inductor=dict(max_block=16, tensor_descriptors=True),
                knobs={}, device_id='xclbin:0123456789abcdef')


def records(scale=1.0, platform=PLATFORM):
    out = []
    for k, p in enumerate(sorted(registry.points(tier=['tiny']), key=lambda p: p.id)):
        t = int((1000 + 37 * k) * scale)
        out.append(dict(point_id=p.id, tier=p.tier, family=p.family, kind=p.kind,
                        track=list(p.track), status='pass', platform=copy.deepcopy(platform),
                        time=dict(unit='cycles', device=t, timing='serial-grid'),
                        bound=dict(version=VERSION, value=10.0, unit='cycles', binding='mem_bytes'),
                        eta=10.0 / t,
                        counters=dict(perf_cycles=t, perf_issued=t // 4, perf_stall_total=t // 2,
                                      perf_x_hazard=t // 2, perf_x_frontend=t - t // 4 - t // 2)))
    return out


def write(tmp, name, recs):
    p = Path(tmp) / name
    p.write_text(''.join(json.dumps(r) + '\n' for r in recs))
    return str(p)


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            rc = ablate.main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
            print(e)
    return rc, buf.getvalue()


def main():
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f'  ({detail})' if detail and not cond else ''))
        ok &= bool(cond)

    with tempfile.TemporaryDirectory(prefix='r39-ablate-') as tmp:
        a = write(tmp, 'a.jsonl', records())
        out = Path(tmp) / 'o.json'

        rc, txt = run([a, a, '--gate', '--min-speedup', '1.0', '--json', str(out)])
        res = json.loads(out.read_text())
        check('identical sides: S_geo == 1.0', res['s_geo'] == 1.0, res['s_geo'])
        check('identical sides: no regression, gate passes', rc == 0 and not res['regress'], txt[-300:])

        slow = records()
        slow[3]['time']['device'] = int(slow[3]['time']['device'] * 1.05)
        b = write(tmp, 'slow.jsonl', slow)
        rc, txt = run([a, b, '--gate', '--json', str(out)])
        res = json.loads(out.read_text())
        check('planted 5 % slowdown is reported', res['regress'] == [slow[3]['point_id']], res['regress'])
        check('planted 5 % slowdown fails the gate', rc != 0)
        # S_geo is just under 1.0 here, so the speedup floor is relaxed to
        # isolate the --explained behaviour.
        rc, _ = run([a, b, '--gate', '--min-speedup', '0.99', '--explained', slow[3]['point_id']])
        check('an explained regression passes the gate', rc == 0)

        fast = write(tmp, 'fast.jsonl', records(scale=0.8))
        rc, txt = run([a, fast, '--gate', '--min-speedup', '1.2', '--json', str(out)])
        res = json.loads(out.read_text())
        check('uniform 1.25x: S_geo == 1.25', abs(res['s_geo'] - 1.25) < 0.01, res['s_geo'])
        check('uniform 1.25x passes a 1.2 gate', rc == 0)
        rc, _ = run([a, fast, '--gate', '--min-speedup', '1.3'])
        check('uniform 1.25x fails a 1.3 gate', rc != 0)

        for what, change in (('inductor', dict(inductor=dict(max_block=64, tensor_descriptors=True))),
                             ('knobs', dict(knobs={'TNCA_SCHEDULE': '0'})),
                             ('device_id', dict(device_id='xclbin:fedcba9876543210'))):
            plat = dict(PLATFORM, **change)
            other = write(tmp, f'other_{what}.jsonl', records(platform=plat))
            merged = report.load([a, other])
            keys = {report.platform_key(r) for r in merged}
            check(f'runs differing in {what} keep separate keys', len(keys) == 2 and len(merged) == 2 * len(records()),
                  f'{len(keys)} keys, {len(merged)} records')

        mixed = records()
        mixed[0]['platform']['knobs'] = {'TNCA_SCHEDULE': '0'}
        m = write(tmp, 'mixed.jsonl', mixed)
        rc, txt = run([m, a])
        check('a side with two configurations is refused', rc != 0 and 'one configuration' in txt, txt[-200:])

        half = records()
        half[5]['status'] = 'numeric-failure'
        h = write(tmp, 'half.jsonl', half)
        rc, txt = run([a, h, '--gate'])
        check('a point passing on one side only fails the gate', rc != 0 and 'one side only' in txt, txt[-300:])

    print(f"r39-ablate-tests       {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
