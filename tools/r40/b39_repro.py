#!/usr/bin/env python3
"""R40-00 acceptance 2: B39 reproduces, and its time boundaries can be checked.

    python3 tools/r40/b39_repro.py --b39 DIR --p2 DIR [--gates RUN.json] [--evidence]

doc/r40_tasks_and_acceptance.md R40-00 acceptance 2: "B39 全量数值与现有 gate
可复现；device、host、传输及 compile 的时间边界可从日志核对". Three runs are
judged against spec/r40/baseline.json:

  --b39 DIR   u280_<tier>.jsonl: the frozen software (an export of the B39
              commit, protocol 1) run again on the B39 image. Every point
              passes, and its device cycles equal the frozen record's, point
              by point -- except where an open item of baseline.json already
              says they may not (B4), which is reported, not hidden.
  --p2 DIR    u280_<tier>.jsonl: the R40-00 tree (protocol 2: a cold first
              call, warm-up, timed calls; provenance) on the same image.
              Its first call's device cycles equal B39's -- the new protocol
              measures the same thing -- and every record's time boundaries
              are consistent: per timed call, request >= upload + run +
              download; run >= device seconds; the first call (compile +
              run) >= a steady call; compile time = first call - steady
              call >= 0; and the samples, warm-up and provenance are present.
  --gates     tools/r40/b39_gates.py's run, checked against the frozen
              counts (spec/r40/b39/gates.json).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evidence as EV        # noqa: E402

TIERS = ('tiny', 'small', 'medium')
PROVENANCE = ('software_tree', 'artifacts', 'profile', 'capabilities', 'image', 'clock_hz',
              'launch_map')


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r['point_id']] = r          # a rerun replaces
    return out


def frozen():
    base = json.loads((ROOT / 'spec' / 'r40' / 'baseline.json').read_text())
    recs = {}
    for tier, meta in base['records'].items():
        recs.update(load(ROOT / meta['path']))
    allowed = {d['point'] for d in (base.get('b4') or {}).get('different', [])}
    return base, recs, allowed


def cycles(r):
    t = r.get('time') or {}
    fc = (t.get('first_call') or {}).get('device_cycles')
    return fc if fc is not None else t.get('device')


def judge_b39(d, base, F, allowed):
    checks = []
    got = {}
    for t in TIERS:
        p = Path(d) / f'u280_{t}.jsonl'
        if p.exists():
            got.update(load(p))
    for pid, f in sorted(F.items()):
        g = got.get(pid)
        if g is None:
            checks.append(dict(name=f'b39:{pid}', ok=False, why='not run'))
            continue
        ok_num = g['status'] == 'pass'
        same = cycles(g) == cycles(f)
        why = []
        if not ok_num:
            why.append(f"status {g['status']}")
        if not same:
            why.append(f'cycles {cycles(g)} vs frozen {cycles(f)}'
                       + (' (an open B4 item)' if pid in allowed else ''))
        checks.append(dict(name=f'b39:{pid}', ok=ok_num and (same or pid in allowed),
                           why='; '.join(why), cycles_equal=same))
    return checks


def judge_p2(d, F):
    checks = []
    got = {}
    for t in TIERS:
        p = Path(d) / f'u280_{t}.jsonl'
        if p.exists():
            got.update(load(p))
    for pid, f in sorted(F.items()):
        g = got.get(pid)
        if g is None:
            checks.append(dict(name=f'p2:{pid}', ok=False, why='not run'))
            continue
        why = []
        if g.get('schema') != 'archbench-torch-record/2':
            why.append(f"schema {g.get('schema')}")
        if g['status'] != 'pass':
            why.append(f"status {g['status']}")
        t, h = g.get('time') or {}, g.get('host') or {}
        tims = g.get('timings') or {}
        dev, lat = tims.get('device-cycles-serial-grid') or {}, tims.get('request-latency') or {}
        if not dev.get('samples') or not lat.get('samples'):
            why.append('no timed samples')
        if not t.get('warmup'):
            why.append('no warm-up')
        prov = g.get('provenance') or {}
        lack = [k for k in PROVENANCE if prov.get(k) in (None, '', [], {})]
        if lack:
            why.append(f'provenance lacks {lack}')
        st, clock = h.get('steady') or {}, t.get('clock_hz')
        if st and clock and dev.get('median'):
            dev_s = dev['median'] / clock
            if st['request_s'] + 1e-6 < st['upload_s'] + st['run_s'] + st['download_s']:
                why.append('request < upload + run + download')
            if st['run_s'] + 1e-6 < dev_s:
                why.append(f"host run {st['run_s']} s < device {dev_s:.6f} s")
            if (h.get('first_call_s') or 0) + 1e-3 < st['request_s']:
                why.append('first call faster than a steady call')
        else:
            why.append('no steady host times')
        same = cycles(g) == cycles(f)
        if not same:
            why.append(f'first-call cycles {cycles(g)} vs B39 {cycles(f)}')
        spread = (max(dev.get('samples') or [0]) - min(dev.get('samples') or [0]))
        checks.append(dict(name=f'p2:{pid}', ok=not why, why='; '.join(why), cycles_equal=same,
                           compile_s=round((h.get('first_call_s') or 0) - (st.get('request_s') or 0), 3),
                           device_cycle_spread=spread))
    return checks


def identity(p2_dir):
    """B39, wrapped, against the protocol-2 run of the same image through
    tools/r40/ablate.py: only the software tree may differ (and is
    declared); the comparison must be accepted with S_geo 1 and no
    significant gain -- the identity of acceptance 1, on real records."""
    import tempfile
    import ablate
    import b39
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / 'b39.jsonl'
        b39.main(['wrap', '--out', str(a)])
        b = Path(d) / 'p2.jsonl'
        b.write_text(''.join(Path(p2_dir, f'u280_{t}.jsonl').read_text() for t in TIERS
                             if Path(p2_dir, f'u280_{t}.jsonl').exists()))
        subset, _ = ablate.load_subset(ROOT / 'spec' / 'r40' / 'suites' / 'existing.json')
        try:
            res = ablate.compare(a, b, subset, ['platform.software_tree'], legacy_a=True,
                                 timing='device-cycles-serial-grid')
        except ablate.Refused as e:
            return dict(name='identity: B39 vs the R40-00 tree on the B39 image', ok=False,
                        why=f'refused: {e}')
    ok = abs(res['s_geo'] - 1) < 1e-9 and res['verdict'] == 'no-significant-gain'
    return dict(name='identity: B39 vs the R40-00 tree on the B39 image', ok=ok,
                why=f"S_geo {res['s_geo']:.6f}, CI {res['ci95']}, {res['verdict']}, worst "
                    f"{res['worst']['point']} {res['worst']['speedup']:.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b39', type=Path, required=True)
    ap.add_argument('--p2', type=Path, required=True)
    ap.add_argument('--gates', type=Path)
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args(argv)
    base, F, allowed = frozen()
    c1 = judge_b39(args.b39, base, F, allowed)
    c2 = judge_p2(args.p2, F)
    checks = c1 + c2
    checks.append(identity(args.p2))
    if args.gates:
        import b39_gates
        rc = b39_gates.check(args.gates, b39_gates.FROZEN, evidence=False)
        checks.append(dict(name='b39-gates', ok=rc == 0, why='' if rc == 0 else 'see b39_gates.py --check'))
    else:
        checks.append(dict(name='b39-gates', ok=False, why='no gate run given'))
    for c in checks:
        if not c['ok'] or not c.get('cycles_equal', True):
            print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']:<64s} {c.get('why', '')}")
    eq1 = sum(c.get('cycles_equal', False) for c in c1)
    eq2 = sum(c.get('cycles_equal', False) for c in c2)
    ok = all(c['ok'] for c in checks)
    detail = (f'B39 rerun: {sum(c["ok"] for c in c1)}/{len(c1)} pass, {eq1} cycle-identical; '
              f'protocol 2: {sum(c["ok"] for c in c2)}/{len(c2)} consistent, {eq2} with B39\'s cycles')
    print(f"r40-b39  {'PASS' if ok else 'FAIL'}  {detail}")
    if args.evidence:
        EV.record('R40-00.b39-reproducible', 'capability-pass' if ok else 'fail', checks, detail=detail)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
