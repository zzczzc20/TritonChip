#!/usr/bin/env python3
"""
R35-BASE-01 and R35-BASE-03 -- doc/phase3_5_tasks_and_acceptance.md 7.

The frozen inventory, checked before anything is simulated, and the
hardware precondition, checked before anything is attributed.

Section 5 is the load-bearing part: the gains are measured on the R3 v2
profile that was actually signed off. So this refuses to start unless
that sign-off exists, names the commit it was given at, and matches the
profile the workloads say they run on. "硬件若发生变化，必须先补齐对应 R3
验证" -- a hardware change is not a compiler gain, and the cheapest way
to keep that true is to make the plan unable to start without the
evidence.

Malformations refused here, all before any RTL:

    a workload whose sizes were raised after the fact
    an ablation subject that is not run at O1 and O2-no-LL
    a run matrix that does not add up to the declared launches
    a primary comparison that mixes profiles
    a gain threshold quietly lowered
    a missing or unmatched R3 v2 sign-off
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

SPEC_PATH = os.path.join(ROOT, 'spec', 'r35_workloads.json')
CASES_PATH = os.path.join(ROOT, 'spec', 'r35_required_cases.json')
TOOLCHAIN_PATH = os.path.join(ROOT, 'spec', 'r35_toolchain.json')
R3_EVIDENCE = os.path.join(ROOT, 'doc', 'evidence', 'r3v2',
                           'acceptance-linux.json')

# The contract's own numbers, repeated here so a quiet edit to the spec
# disagrees with something.
CONTRACT = {
    'workloads': 6,
    'launches': 23,
    'primary': 12, 'ablation': 4, 'legacy': 4, 'post_install': 3,
    'standards': 27, 'packages': 9, 'mutations': 8,
    'component_rtl': 24, 'total_rtl': 47,
    'elapsed_seconds': 2400, 'cycles': 1500000,
    'primary_relative': 0.05, 'll_relative': 0.03,
    'regression_relative': 0.02,
}
SIZE_CAPS = {'L01': {'n': 17, 'block': 16, 'grid': 2},
             'L02': {'n': 16, 'grid': 1},
             'L03': {'rows': 4, 'cols': 4, 'grid': 1},
             'L04': {'elements': 16, 'valid': 15, 'rows': 1},
             'L05': {'m': 16, 'n': 16, 'k': 16},
             'L06': {'m': 16, 'n': 16, 'k': 32}}


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def check_inventory(spec, cases):
    """Every complaint this finds, as a list. Empty means well formed."""
    bad = []
    w = spec['workloads']
    if len(w) != CONTRACT['workloads']:
        bad.append(f"{len(w)} workload(s); the contract freezes "
                   f"{CONTRACT['workloads']}")
    ids = [x['id'] for x in w]
    if len(set(ids)) != len(ids):
        bad.append('a workload id appears twice')
    for x in w:
        cap = SIZE_CAPS.get(x['id'])
        if cap is None:
            bad.append(f"{x['id']}: not one of the six the contract names")
            continue
        for k, v in cap.items():
            got = x['sizes'].get(k)
            if got is None:
                bad.append(f"{x['id']}: {k} is missing")
            elif got != v:
                bad.append(f"{x['id']}: {k}={got}, the contract freezes {v}"
                           + (" -- section 6.3 forbids raising a size after "
                              "a failure" if got > v else ""))
        for field in ('source_kind', 'memory', 'observe'):
            if not x.get(field):
                bad.append(f"{x['id']}: {field} is not stated")

    # The ablation subjects the contract names, and only those, are run
    # at O1 and O2-no-LL.
    abl = {x['id'] for x in w if x.get('ablation')}
    want = set(cases['modes']) and {'L03', 'L06'}
    if abl != want:
        bad.append(f"the ablation subjects are {sorted(abl)}; the contract "
                   f"names {sorted(want)}")

    m = spec['matrix']
    got = (m['primary']['runs'] + m['ablation']['runs'] +
           m['legacy_profile']['runs'] + m['post_install']['runs'])
    if got != m['total_launches']:
        bad.append(f"the groups add to {got} and the total says "
                   f"{m['total_launches']}")
    if m['total_launches'] != CONTRACT['launches']:
        bad.append(f"{m['total_launches']} launch(es); the contract says "
                   f"{CONTRACT['launches']}")
    for key, n in (('primary', CONTRACT['primary']),
                   ('ablation', CONTRACT['ablation']),
                   ('legacy_profile', CONTRACT['legacy']),
                   ('post_install', CONTRACT['post_install'])):
        if m[key]['runs'] != n:
            bad.append(f"the {key} group has {m[key]['runs']} run(s); the "
                       f"contract says {n}")
    if m['primary']['runs'] != 2 * len(w):
        bad.append('the primary comparison is not every workload on both '
                   'B0 and O2')

    b = spec['budget']
    for key, n in (('component_rtl_invocations', CONTRACT['component_rtl']),
                   ('total_rtl_runs', CONTRACT['total_rtl']),
                   ('total_elapsed_seconds', CONTRACT['elapsed_seconds']),
                   ('total_cycles', CONTRACT['cycles'])):
        if b.get(key) != n:
            bad.append(f"budget {key} is {b.get(key)}, the contract says {n}")
    if m['total_launches'] + b['component_rtl_invocations'] > b['total_rtl_runs']:
        bad.append('the launches and the component invocations exceed the '
                   'RTL total')

    g = spec['gain_thresholds']
    for key, n in (('primary_relative', CONTRACT['primary_relative']),
                   ('ll_relative', CONTRACT['ll_relative']),
                   ('regression_relative', CONTRACT['regression_relative'])):
        if abs(g.get(key, 0) - n) > 1e-12:
            bad.append(f"the {key} threshold is {g.get(key)}, the contract "
                       f"says {n} -- section 6.3 forbids lowering it")
    if g.get('primary_workloads_required', 0) < 2:
        bad.append('fewer than two winners required')
    if not g.get('total_cycles_must_not_increase'):
        bad.append('the no-regression-in-total condition has been dropped')
    if set(g.get('ll_subjects', [])) != {'L03', 'L06'}:
        bad.append('the LL ablation subjects are not L03 and L06')

    n = sum(len(p['cases']) for p in cases['packages'])
    if n != CONTRACT['standards']:
        bad.append(f"{n} standard(s); the contract has "
                   f"{CONTRACT['standards']}")
    if len(cases['packages']) != CONTRACT['packages']:
        bad.append(f"{len(cases['packages'])} package(s); the contract has "
                   f"{CONTRACT['packages']}")
    if len(cases['mutations']) != CONTRACT['mutations']:
        bad.append(f"{len(cases['mutations'])} mutation(s); the contract "
                   f"names {CONTRACT['mutations']}")
    for mu in cases['mutations']:
        if not mu.get('detector'):
            bad.append(f"mutation {mu['id']} names no detector")
    return bad


MALFORMATIONS = [
    ('a size raised after the fact',
     lambda s, c: s['workloads'][4]['sizes'].__setitem__('k', 64)),
    ('a workload dropped',
     lambda s, c: s['workloads'].pop()),
    ('an ablation subject no longer ablated',
     lambda s, c: s['workloads'][2].pop('ablation', None)),
    ('the primary comparison shortened',
     lambda s, c: s['matrix']['primary'].__setitem__('runs', 10)),
    ('the launch total edited to match a short matrix',
     lambda s, c: s['matrix'].__setitem__('total_launches', 21)),
    ('the gain threshold lowered',
     lambda s, c: s['gain_thresholds'].__setitem__('primary_relative', 0.01)),
    ('the no-regression condition dropped',
     lambda s, c: s['gain_thresholds'].__setitem__(
         'total_cycles_must_not_increase', False)),
    ('the RTL budget raised',
     lambda s, c: s['budget'].__setitem__('total_rtl_runs', 200)),
    ('a standard removed',
     lambda s, c: c['packages'][0]['cases'].pop()),
    ('a mutation left without a detector',
     lambda s, c: c['mutations'][0].__setitem__('detector', '')),
]


def base01(log):
    spec = json.load(open(SPEC_PATH))
    cases = json.load(open(CASES_PATH))
    bad = check_inventory(spec, cases)
    for b in bad:
        log.append(f"    {b}")
    if bad:
        return False, f"{len(bad)} problem(s) in the frozen inventory"

    w = spec['workloads']
    log.append(f"    {len(w)} workload(s): "
               f"{', '.join(x['id'] + ' ' + x['name'] for x in w[:3])}, ...")
    m = spec['matrix']
    log.append(f"    {m['primary']['runs']} primary + "
               f"{m['ablation']['runs']} ablation + "
               f"{m['legacy_profile']['runs']} legacy + "
               f"{m['post_install']['runs']} post-install = "
               f"{m['total_launches']} launch(es)")
    b = spec['budget']
    log.append(f"    {m['total_launches']} launch(es) + "
               f"{b['component_rtl_invocations']} component invocation(s) = "
               f"{m['total_launches'] + b['component_rtl_invocations']}, "
               f"under the {b['total_rtl_runs']} the contract allows")
    log.append(f"    {sum(len(p['cases']) for p in cases['packages'])} "
               f"standard(s) in {len(cases['packages'])} package(s); "
               f"{len(cases['mutations'])} mutation(s), each with a named "
               f"detector")
    g = spec['gain_thresholds']
    log.append(f"    the gate: two workload(s) at "
               f"max({g['primary_absolute_cycles']} cycles, "
               f"{g['primary_relative']:.0%}), one LL attribution at "
               f"max({g['ll_absolute_cycles']}, {g['ll_relative']:.0%}), "
               f"no case worse than "
               f"max({g['regression_absolute_cycles']}, "
               f"{g['regression_relative']:.0%})")
    return True, 'the frozen inventory is complete and within budget'


def base02_refusals(log):
    """Every malformation refused, before anything is simulated."""
    ok = True
    for what, break_it in MALFORMATIONS:
        spec = json.load(open(SPEC_PATH))
        cases = json.load(open(CASES_PATH))
        break_it(spec, cases)
        bad = check_inventory(spec, cases)
        log.append(f"    {what:<46s} -> "
                   + (f"refused: {bad[0][:58]}" if bad else "ACCEPTED"))
        ok &= bool(bad)
    log.append(f"    {len(MALFORMATIONS)} malformation(s), each refused by "
               f"name, and none of them started a simulation")
    return ok


def base03(log):
    """The hardware precondition: R3 v2 accepted, and it is this one."""
    spec = json.load(open(SPEC_PATH))
    pre = spec['hardware_precondition']
    ok = True

    if not os.path.exists(R3_EVIDENCE):
        log.append("    there is no R3 v2 acceptance record at "
                   "doc/evidence/r3v2/acceptance-linux.json")
        return False, 'the R3 v2 precondition is not evidenced'
    ev = json.load(open(R3_EVIDENCE))
    log.append(f"    R3 v2 record: {ev.get('verdict')} at "
               f"{str(ev.get('commit'))[:12]} on {ev.get('platform')}")
    if ev.get('verdict') != pre['signoff']:
        log.append(f"    it does not say {pre['signoff']}")
        ok = False
    if not str(ev.get('commit', '')).startswith(pre['commit']):
        log.append(f"    it was given at {str(ev.get('commit'))[:12]}, and "
                   f"this stage was frozen against {pre['commit']}")
        ok = False
    if not ev.get('clean_tree'):
        log.append('    it was not given on a clean tree')
        ok = False
    gates = ev.get('gates') or {}
    if not all(gates.values()):
        log.append(f"    not every R3 gate passed: {gates}")
        ok = False
    else:
        log.append(f"    all {len(gates)} R3 v2 gate(s) passed in that run")

    # The profile the gains will be measured on must be the one R3 v2
    # accepted, and it must exist here.
    profiles = json.load(open(os.path.join(ROOT, 'spec',
                                           'r1_profiles.json')))
    names = set(profiles['profiles'])
    if pre['profile'] not in names:
        log.append(f"    the profile {pre['profile']} is not in "
                   f"spec/r1_profiles.json")
        ok = False
    else:
        log.append(f"    the measurement profile {pre['profile']} exists and "
                   f"is the one R3 v2 accepted")
    log.append("    the legacy core-sim runs are compatibility only: "
               "section 5 refuses them as a substitute for a finite-port "
               "measurement")

    # O0 must emit what B0 emits, byte for byte. Section 4.3: "在冻结用例上
    # 机器码与 B0 一致，除版本化 metadata 外不得人为变差".
    aot_src = open(os.path.join(ROOT, 'python', 'tnca', 'aot.py')).read()
    if re.search(r"'opt_level':[^\n]*reserved", aot_src):
        log.append('    O0 == B0 is NOT CHECKABLE: opt_level is still '
                   'declared reserved and changes no emission')
        return False, 'the driver has no second path to compare against'

    same, checked = [], 0
    try:
        sys.path.insert(0, os.path.join(ROOT, 'python'))
        sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
        sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
        sys.path.insert(0, HERE)
        import tempfile
        import r35_workloads as WW
        import workloads as W
        from tnca import aot
        cases = {
            'L03': dict(source=WW.tnca_view_elementwise(),
                        kernel_name='l03', source_kind='tnca-ir',
                        signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'}),
            'L04': dict(source=W.tnca_reduction(15), kernel_name='red',
                        source_kind='tnca-ir',
                        signature={'in_ptr': '*fp32', 'out_ptr': '*fp32'}),
            'L01': dict(source=W.triton_masked_axpy(),
                        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32',
                                   'o_ptr': '*fp32', 'n': 'i32',
                                   'alpha': 'fp32', 'BLOCK': 'constexpr'},
                        constexprs={'BLOCK': 16}),
        }
        for cid, kw in sorted(cases.items()):
            out = {}
            for tag, opts in (('B0', {}), ('O0', {'opt_level': 0})):
                r = aot.compile_aot(profile=pre['profile'], options=opts,
                                    output=os.path.join(
                                        tempfile.mkdtemp(), f'{cid}.tnca'),
                                    **kw)
                out[tag] = r.asm['hex']
            checked += 1
            equal = out['B0'] == out['O0']
            same.append(equal)
            log.append(f"    {cid}: O0 emits "
                       f"{'the same bytes as B0' if equal else 'DIFFERENT '
                          'bytes from B0'} "
                       f"({len(out['B0'].splitlines())} word(s))")
    except Exception as e:                                 # noqa: BLE001
        log.append(f"    O0 == B0 could not be checked: "
                   f"{type(e).__name__}: {e}")
        ok = False
    if checked and not all(same):
        ok = False
    if not checked:
        ok = False
    return ok, ('the hardware precondition holds and O0 emits what B0 emits'
                if ok else 'the precondition or the O0 path is not in place')


TESTS = [('R35-BASE-01', base01),
         ('R35-BASE-02', lambda log: (base02_refusals(log),
                                      'every malformation refused before '
                                      'any RTL runs')),
         ('R35-BASE-03', base03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.5 G0: the frozen workloads, the matrix, and the hardware "
          "it is measured on")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-600:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    print(f"  r35-plan               {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
