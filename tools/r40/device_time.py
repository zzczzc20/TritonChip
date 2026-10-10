#!/usr/bin/env python3
"""R40-01 acceptance 3: device time of the tnca device against B39.

    python3 tools/r40/device_time.py --native DIR [--evidence]

DIR holds u280_<tier>.jsonl from `archbench.run --set TNCA_TORCH_DEVICE=native`
(protocol 2) on B39's image. The comparison is the plan's experiment
R40-01.device-time: the frozen subset `existing` (tiny + small + medium), A =
B39 (spec/r40/baseline.json, wrapped), B = the native device, timing
device-cycles-serial-grid, threshold S_geo >= 1.0, through tools/r40/ablate.py
and its refusals.

On the on-chip-RAM image some points do not fit when their tensors stay
resident (capacity-exceeded); the protocol then refuses the comparison
(`missing`: a frozen point that did not pass is not left out), and the gate
is `not-measured` until R40-04 gives the device external memory (doc R40-01
acceptance 1: "HBM 就绪前先跑容量可容纳的 smoke，完整签收在 R40-04 后补齐").
What the points that did run show is recorded beside it as a diagnostic --
never as the gate, since a subset chosen after measuring is not a frozen one.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablate                # noqa: E402
import b39                   # noqa: E402
import evidence as EV        # noqa: E402

TIERS = ('tiny', 'small', 'medium')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args(argv)
    plan = json.loads((ROOT / 'spec' / 'r40' / 'plan.json').read_text())
    exp = plan['experiments']['R40-01.device-time']
    subset, _ = ablate.load_subset(ROOT / exp['subset'])
    native = {}
    for t in TIERS:
        p = args.native / f'u280_{t}.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    native[r['point_id']] = r
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / 'b39.jsonl'
        b39.main(['wrap', '--out', str(a)])
        bpath = Path(d) / 'native.jsonl'
        bpath.write_text(''.join(json.dumps(r) + '\n' for r in native.values()))
        try:
            res = ablate.compare(a, bpath, subset, exp['vary'], legacy_a=True, timing=exp['timing'],
                                 threshold=exp['threshold'], explained=tuple(exp.get('explained', ())))
            refused = None
        except ablate.Refused as e:
            res, refused = None, e
        # The diagnostic: the points that passed on both sides.
        ran = {pid: sha for pid, sha in subset.items()
               if native.get(pid, {}).get('status') == 'pass'}
        diag = ablate.compare(a, bpath, ran, exp['vary'], legacy_a=True, timing=exp['timing']) if ran else None
    statuses = {}
    for pid in subset:
        st = native.get(pid, {}).get('status', 'absent')
        statuses[st] = statuses.get(st, 0) + 1
    ident = sum(1 for r in (diag or {}).get('points', []) if r['a'] == r['b'])
    checks = [dict(name='every frozen point measured', ok=refused is None,
                   why=str(refused) if refused else '')]
    if res:
        checks += [dict(name=f"S_geo >= {exp['threshold']}", ok=res['s_geo'] >= exp['threshold'],
                        s_geo=res['s_geo']),
                   dict(name='no unexplained regression over 5 %', ok=not res['regress_5pct_unexplained'])]
        status = 'capability-pass' if all(c['ok'] for c in checks) else 'fail'
    else:
        only_capacity = set(statuses) <= {'pass', 'capacity-exceeded'}
        status = 'not-measured' if only_capacity and refused.kind == 'missing' else 'fail'
    detail = (f"points: {statuses}; " + (f"S_geo {res['s_geo']:.4f}" if res else f'refused ({refused.kind})')
              + (f"; diagnostic over the {len(ran)} that ran: S_geo {diag['s_geo']:.4f}, {ident} "
                 f"cycle-identical, worst {diag['worst']['point']} {diag['worst']['speedup']:.3f}"
                 if diag else ''))
    print(f'R40-01.device-time  {status}  {detail}')
    # Acceptance 1: every point through model.to("tnca") and torch.compile
    # on the device, outputs on it, no host fallback, nothing staged.
    dev_checks = []
    for pid in subset:
        r = native.get(pid)
        if r is None:
            dev_checks.append(dict(name=pid, ok=False, why='not run'))
            continue
        why = []
        if r['status'] != 'pass':
            why.append(r['status'])
        if r.get('fallbacks'):
            why.append(f"fallbacks {r['fallbacks']}")
        if (r['platform'].get('knobs') or {}).get('TNCA_TORCH_DEVICE') != 'native':
            why.append('not the native device')
        dev_checks.append(dict(name=pid, ok=not why, why='; '.join(why)))
    cap = [c for c in dev_checks if c['why'] == 'capacity-exceeded']
    other = [c for c in dev_checks if not c['ok'] and c['why'] != 'capacity-exceeded']
    dev_status = 'fail' if other else ('not-measured' if cap else 'capability-pass')
    dev_detail = (f'{sum(c["ok"] for c in dev_checks)}/{len(dev_checks)} pass on the device; '
                  f'{len(cap)} capacity-exceeded on this image (resident tensors in the on-chip heap)')
    print(f'R40-01.device  {dev_status}  {dev_detail}')
    if args.evidence:
        EV.record('R40-01.device', dev_status, dev_checks, detail=dev_detail)
        EV.record('R40-01.device-time', status, checks, detail=detail,
                  diagnostic=dict(points=len(ran), s_geo=diag and diag['s_geo'],
                                  identical=ident, worst=diag and diag['worst'],
                                  regress_2pct=diag and diag['regress_2pct']) if diag else None)
    return 0 if status in ('capability-pass', 'not-measured') else 1


if __name__ == '__main__':
    sys.exit(main())
