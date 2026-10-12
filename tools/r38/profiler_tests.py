#!/usr/bin/env python3
"""S1-02 acceptance: device counters through the public runtime.

doc/r38_stage1_tasks_and_acceptance.md S1-02, tests 1-8.

    1 exclusive stalls sum to perf_stall_total, per program, exactly; and
      issued + stall_total + frontend == perf_cycles (R39-00)
    2 delta perf_issued == done_retired; 0 <= done + slack - delta perf_cycles,
      and the slack is attained (it is the real window offset, not a margin)
    3 stall_total and every overlapping count <= delta perf_cycles
    4 snapshots never decrease within a session
    5 all O2 cases pass with counters on, cycles identical to the baseline
    6 a device without op 7 reports unavailable, not zero, and still runs
    7 two RTL mutations of the counter mux are each caught by 1-3
    8 the generated include is current (R36 checks are run separately:
      make r36-rtl-tests r36-build-tests)

Benches are built from the working tree into sim/build/r38/, so this
measures the RTL in front of it rather than whatever the device cache holds.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402

OUT = ROOT / 'sim' / 'build' / 'r38' / 'profiler'
SPEC = json.loads((ROOT / 'spec' / 'r38' / 'counters.json').read_text())
SLACK = SPEC['identities']['run_window_slack']
EXCL = [c['name'] for c in SPEC['counters'] if c['group'] == 'stall_exclusive']
OVER = [c['name'] for c in SPEC['counters'] if c['group'] == 'stall_overlapping']
PROBE = ['mul2', 'softmax/n37', 'matmul_fp32_fp16_fp8/fp16', 'histogramming/partial']


def identity_problems(rows):
    """Tests 1-3 over every program of every passing row."""
    bad, attained = [], False
    for r in rows:
        if not r['ok']:
            continue
        c = r.get('counters') or {}
        if not c.get('available'):
            bad.append(f'{r["case"]}: counters unavailable')
            continue
        for i, p in enumerate(c['programs']):
            tag = f'{r["case"]}[{i}]'
            if sum(p[n] for n in EXCL) != p['perf_stall_total']:
                bad.append(f'{tag}: exclusive stalls sum to {sum(p[n] for n in EXCL)}, '
                           f'total says {p["perf_stall_total"]}')
            # R39-00: every running cycle issues, stalls or has nothing to
            # issue (perf_x_frontend), so the three partition the cycles.
            if 'perf_x_frontend' in p and (p['perf_issued'] + p['perf_stall_total']
                                           + p['perf_x_frontend'] != p['perf_cycles']):
                bad.append(f'{tag}: issued {p["perf_issued"]} + stalled {p["perf_stall_total"]} '
                           f'+ frontend {p["perf_x_frontend"]} != cycles {p["perf_cycles"]}')
            if p['perf_issued'] != p['done_retired']:
                bad.append(f'{tag}: issued {p["perf_issued"]} != retired {p["done_retired"]}')
            drain = p['done_cycles'] + SLACK - p['perf_cycles']
            if drain < 0:
                bad.append(f'{tag}: perf_cycles {p["perf_cycles"]} exceeds done_cycles '
                           f'{p["done_cycles"]} + slack {SLACK}')
            attained |= drain == 0
            for n in ['perf_stall_total'] + OVER:
                if p[n] > p['perf_cycles']:
                    bad.append(f'{tag}: {n}={p[n]} exceeds perf_cycles {p["perf_cycles"]}')
    return bad, attained


def monotonic_problems(rows):
    """Test 4: within one session, programs are read in order; each program's
    delta is non-negative by construction only if the counter never went
    backwards, since a wrap needs 2^64 events."""
    bad = []
    for r in rows:
        for i, p in enumerate((r.get('counters') or {}).get('programs', [])):
            for k, v in p.items():
                if v >= 1 << 63:
                    bad.append(f'{r["case"]}[{i}]: {k} went backwards')
    return bad


def baseline_cycles():
    frozen = ROOT / 'spec' / 'r38' / 'baseline_cycles.json'
    if frozen.exists():
        return json.loads(frozen.read_text())['cycles'], str(frozen.relative_to(ROOT))
    run = ROOT / 'sim' / 'build' / 'r38' / 'baseline_O2.json'
    if run.exists():
        return ({r['case']: r.get('cycles') for r in json.loads(run.read_text())['cases']},
                str(run.relative_to(ROOT)) + ' (not frozen)')
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='probe cases only (development)')
    ap.add_argument('--profiles', nargs='+', default=['core-sim', 'core-vrf-sync-v1'])
    ap.add_argument('--backend', help='a runtime backend other than the simulation bench '
                    '(e.g. fpga-u50-sim, or fpga-u50 on a board): runs tests 1-5 through it, '
                    'numerics only for 5; tests 6-7 are about the bench and are skipped')
    ap.add_argument('--backend-args', default='{}', help='JSON keyword arguments for --backend')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results, failed = {}, False

    def report(name, problems):
        nonlocal failed
        failed |= bool(problems)
        results[name] = problems
        print(f'{"FAIL" if problems else "PASS"} {name}' + ''.join(f'\n     {p}' for p in problems[:8]))

    gen = subprocess.run([sys.executable, str(HERE / 'gen_counters.py'), '--check'])
    report('8 generated include current', [] if gen.returncode == 0 else ['stale include'])

    base, base_src = baseline_cycles()
    for profile in args.profiles:
        only = PROBE if args.quick else None
        if args.backend:
            skw = dict(backend=args.backend, counters=True, **json.loads(args.backend_args))
        else:
            bench = H.build_bench(profile, OUT / f'bench-{profile}.vvp')
            skw = dict(bench=str(bench), counters=True)
        rows = H.run_cases(2, OUT / f'artifacts-{profile}', only=only, profile=profile,
                           session_kwargs=skw, progress=False)
        (OUT / f'rows-{profile}.json').write_text(json.dumps(rows, indent=1))
        bad, attained = identity_problems(rows)
        if not attained:
            bad.append(f'the slack {SLACK} is never attained: it is a margin, not the window offset')
        report(f'1-3 identities [{profile}]', bad)
        report(f'4 monotonic [{profile}]', monotonic_problems(rows))
        numeric = [f'{r["case"]}: {r["detail"]}' for r in rows if not r['ok']]
        if profile == 'core-sim' and not args.backend:
            if base is None:
                numeric.append('no S1-01 baseline to compare cycles with')
            else:
                numeric += [f'{r["case"]}: {r["cycles"]} cycles with counters, baseline {base.get(r["case"])}'
                            for r in rows if r['ok'] and r['cycles'] != base.get(r['case'])]
            if not args.quick and len(rows) != 149:
                numeric.append(f'{len(rows)} cases ran, expected 149')
        report(f'5 no perturbation [{profile}] vs '
               f'{base_src if profile == "core-sim" and not args.backend else "numerics only"}', numeric)

    if args.backend:
        (OUT / f'report-{args.backend}.json').write_text(json.dumps(results, indent=1))
        print(f'profiler [{args.backend}]: {"FAIL" if failed else "PASS"}'
              + (' (quick: NOT the acceptance run)' if args.quick else ''))
        return 1 if failed else 0

    # 6: the device as it was before op 7.
    old = OUT / 'old-device'
    rtl, tb = H.rtl_copy(old)
    for rel in ('rtl/tnca_shell.v', 'tb/device/tnca_device.v'):
        text = subprocess.run(['git', '-C', str(ROOT), 'show', f'3b5b28f:{rel}'],
                              capture_output=True, text=True, check=True).stdout
        (old / rel).write_text(text)
    bench = H.build_bench('core-sim', old / 'bench.vvp', rtl=rtl, tb=tb)
    rows = H.run_cases(2, old / 'artifacts', only=['mul2'], session_kwargs=dict(bench=str(bench), counters=True),
                       progress=False)
    r = rows[0]
    report('6 old device: unavailable, not zero',
           [] if r['ok'] and r['counters'] == {'available': False} else [f'{r}'])

    # 7: mutations of the generated mux.
    for label, old_line, new_line in (
        ('issued reads cycles', "5'd1: perf_rdata = perf_issued;", "5'd1: perf_rdata = perf_cycles;"),
        ('lsu stall stuck at zero', "5'd4: perf_rdata = perf_x_lsu;", "5'd4: perf_rdata = 64'd0;"),
    ):
        lab = OUT / ('mut-' + label.replace(' ', '-'))
        rtl, tb = H.rtl_copy(lab, [('rtl/common/tnca_perf_sel.vh', old_line, new_line)])
        bench = H.build_bench('core-sim', lab / 'bench.vvp', rtl=rtl, tb=tb)
        rows = H.run_cases(2, lab / 'artifacts', only=['mul2', 'matmul_fp32_fp16_fp8/fp16'],
                           session_kwargs=dict(bench=str(bench), counters=True), progress=False)
        caught, _ = identity_problems(rows)
        report(f'7 mutation caught: {label}', [] if caught else ['identities still hold'])

    (OUT / 'report.json').write_text(json.dumps(results, indent=1))
    print(f'profiler: {"FAIL" if failed else "PASS"}' + (' (quick: NOT the acceptance run)' if args.quick else ''))
    return 1 if failed or args.quick else 0


if __name__ == '__main__':
    sys.exit(main())
