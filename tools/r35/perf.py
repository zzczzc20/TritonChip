#!/usr/bin/env python3
"""
R35-PERF-01 -- doc/phase3_5_tasks_and_acceptance.md section 6.

Device cycles for the frozen workloads at each mode, with the numbers
checked against an independent oracle, on the hardware R3 v2 accepted.

Section 6.3 defines C as the cycles from a launch being accepted to
every result store committing and the resources draining, and it is
explicit that a host wall time or a model prediction may not stand in
for it. What is read here is the device's own cycle count for the
launch, through the same runtime a user has.

Condition 5 of section 6.3 is the one worth keeping in mind while
reading the output: a case where the static instruction count fell and
C did not is not a gain. This prints both, side by side, so that cannot
be quietly glossed.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
import r1lib as R                                          # noqa: E402

SPEC = json.load(open(os.path.join(ROOT, 'spec', 'r35_workloads.json')))
GAIN = SPEC['gain_thresholds']
PROFILE = SPEC['hardware_precondition']['profile']

MODES = {'B0': None, 'O0': 0, 'O1': 1, 'O2': 2}


def have_triton():
    try:
        __import__('triton')
        return True
    except Exception:                                      # noqa: BLE001
        return False


def _cycles_of(ev):
    return getattr(ev, 'device_cycles', None)


def run_one(wid, mode, tmp):
    """One workload at one mode, in a process of its own.

    Not in this one. Measured in-process, L06's baseline was 3241 cycles
    alone and 3221 when L05 had run first: state survives a Session and
    the second measurement starts from somewhere the first one left.
    Section 6.3 requires the memory, the reset state and the cache state
    to be identical between the runs being compared, and a difference of
    twenty cycles is larger than most of the gains being measured.

    A fresh interpreter was not enough on its own. The scratch directory
    has to be fresh too: measuring L06 twice into one directory gave 3221
    and then 3241, and into a new directory each time gave 3241 both
    times. So the child makes its own and this argument is ignored.

    It costs about a second per measurement and it is the difference
    between a comparison and a coincidence.
    """
    r = subprocess.run([sys.executable, os.path.abspath(__file__),
                        '--measure-one', wid, mode, tmp],
                       cwd=ROOT, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        return {'cycles': None, 'correct': False, 'words': None,
                'wall_s': None,
                'error': (r.stderr or r.stdout)[-300:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                                      # noqa: BLE001
        return {'cycles': None, 'correct': False, 'words': None,
                'wall_s': None,
                'error': f"unreadable result: {r.stdout[-200:]}"}


def measure_one(wid, mode, _tmp_ignored):
    """The measurement itself. Runs in the child, in its own directory."""
    tmp = tempfile.mkdtemp(prefix=f'r35-{wid}-{mode}-')
    import e2e_tests as E
    from tnca import aot
    from tnca.runtime import Session

    level = MODES[mode]
    opts = {} if level is None else {'opt_level': level}
    seen = []
    real_wait = Session.wait

    def wait(self, ev, *a, **kw):
        out = real_wait(self, ev, *a, **kw)
        seen.append(_cycles_of(ev))
        return out

    real_compile = aot.compile_aot
    words = {}

    def compile_at(*a, **kw):
        kw['options'] = dict(kw.get('options') or {}, **opts)
        r = real_compile(*a, **kw)
        words['n'] = len(r.asm['hex'].splitlines())
        return r

    Session.wait = wait
    aot.compile_aot = compile_at
    E.aot.compile_aot = compile_at
    started = time.time()
    try:
        case = next(c for c in E.INVENTORY['core']
                    if c['id'] == {'L01': 'C02', 'L04': 'C06',
                                   'L05': 'C03', 'L06': 'C05'}[wid])
        log = []
        if case['workload'] == 'gemm':
            ok, _vals = E.run_gemm(case, 'candidate', log, tmp)
        else:
            art = E.compile_case(case, tmp, E.profile_for(case, 'candidate'))
            ok, _vals = E.run_case(case, 'candidate', art, log)
    finally:
        Session.wait = real_wait
        aot.compile_aot = real_compile
        E.aot.compile_aot = real_compile
    return {'cycles': seen[-1] if seen else None, 'correct': bool(ok),
            'words': words.get('n'), 'wall_s': round(time.time() - started, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='*', default=['B0', 'O1'])
    ap.add_argument('--measure-one', nargs=3, metavar=('WID', 'MODE', 'TMP'),
                    help='internal: measure one case and print it as json')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    if args.measure_one:
        wid, mode, tmp = args.measure_one
        print(json.dumps(measure_one(wid, mode, tmp)))
        return 0

    print("R3.5: device cycles for the frozen workloads, by mode")
    print(f"  on {PROFILE}, the profile R3 v2 accepted")
    if not have_triton():
        print("  the locked Triton is not here; nothing can be compiled")
        return 1

    # Only the workloads that have a source today. L02 and L03 are frozen
    # in the spec and not yet built, and saying so is better than a table
    # with holes in it.
    HAVE = ['L01', 'L04', 'L05', 'L06']
    ids = [w for w in HAVE if not args.only or w in args.only]
    missing = [w['id'] for w in SPEC['workloads'] if w['id'] not in HAVE]

    tmp = tempfile.mkdtemp(prefix='r35-perf-')
    rows = []
    print()
    head = f"  {'case':<6s}"
    for m in args.modes:
        head += f" {m + ' cycles':>11s} {m + ' words':>9s}"
    print(head + f" {'saved':>7s} {'%':>7s}  correct")
    for wid in ids:
        row = {'id': wid, 'modes': {}}
        for m in args.modes:
            row['modes'][m] = run_one(wid, m, tmp)
        line = f"  {wid:<6s}"
        for m in args.modes:
            r = row['modes'][m]
            line += f" {r['cycles']:>11} {r['words']:>9}"
        base = row['modes'].get(args.modes[0], {}).get('cycles')
        last = row['modes'].get(args.modes[-1], {}).get('cycles')
        if base and last is not None:
            d = base - last
            row['saved'] = d
            row['relative'] = d / base
            # math.ceil, not int(): int() truncates, and truncating a
            # threshold lowers it. ceil(0.05 * 228) is 12, and int() of
            # the same expression is 11 -- an error of one cycle, in our
            # own favour, which is the direction section 6.3 forbids
            # moving a threshold in.
            need = max(GAIN['primary_absolute_cycles'],
                       math.ceil(GAIN['primary_relative'] * base))
            row['needed'] = need
            row['meets'] = d >= need
            line += f" {d:>7d} {100.0 * d / base:>6.1f}%"
        allok = all(r['correct'] for r in row['modes'].values())
        row['correct'] = allok
        line += f"  {'yes' if allok else 'NO'}"
        print(line, flush=True)
        rows.append(row)

    print()
    wins = [r for r in rows if r.get('meets')]
    for r in rows:
        if 'saved' not in r:
            continue
        verdict = 'meets the gate' if r['meets'] else \
            f"short of the {r['needed']}-cycle gate"
        print(f"    {r['id']}: {r['saved']:+d} cycles "
              f"({r['relative']:.1%}) -- {verdict}")
    print()
    print(f"  {len(wins)} workload(s) meet max("
          f"{GAIN['primary_absolute_cycles']} cycles, "
          f"{GAIN['primary_relative']:.0%}); section 6.3 wants "
          f"{GAIN['primary_workloads_required']}")
    if missing:
        print(f"  not measured, because they have no source yet: "
              f"{', '.join(missing)}")
    print("  section 6.3 condition 5: a case whose word count fell and "
          "whose cycles did not has not gained anything")

    ok = (len(wins) >= GAIN['primary_workloads_required'] and
          all(r['correct'] for r in rows) and not missing)
    detail = (f"{len(wins)}/{GAIN['primary_workloads_required']} workload(s) "
              f"over the gate; {len(missing)} not yet built")
    results = [{'id': 'R35-PERF-01', 'passed': bool(ok), 'detail': detail,
                'profile': PROFILE, 'rows': rows, 'unbuilt': missing}]
    if args.json:
        R.write_json(args.json, results)
    print()
    print(f"  r35-perf               {'PASS' if ok else 'FAIL':<6s} {detail}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
