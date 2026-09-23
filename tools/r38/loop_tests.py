#!/usr/bin/env python3
"""S1-07 acceptance: the transposed transfer as one three-layer change.

doc/r38_stage1_tasks_and_acceptance.md S1-07. The feature is ISA sec 9E
mod[5] in rtl/core/tnca_tensor.v (HAS_TENSOR_XPOSE), the capability
`tensor_transpose` on profile core-xpose-v1, and the compiler lowering of
`tl.trans` of a descriptor tile to one transposed transfer.

    1 off  : core-sim O2, 149/149, cycles and artifacts identical to the
             frozen baseline (the compiler emits nothing new without the
             capability; the RTL with HAS=0 behaves as before)
    2 on   : core-xpose-v1 O2, 149/149 numerically; the FP16 GEMM smoke point
             is faster, and the counters say why (fewer memory beats, fewer
             LSU stall cycles)
    3 compat: a core-xpose-v1 artifact that needs the capability is refused on
             core-sim as CAPABILITY_MISSING naming it; a core-sim artifact runs
             on core-xpose-v1 with the same cycles; a descriptor the engine
             cannot transfer traps DESCRIPTOR_FAULT instead of returning a
             wrong answer
    4 rollback: with the feature off, the engine's structural proxy equals the
             parent engine's (<= 0.5 % gates, <= 0.5 % flops)
    5 resource: the structural cost of the feature, recorded in the design
             change
    6 ISA   : make r38-xpose-rtl (224 lane assertions; ILLEGAL without the
             feature; DESCRIPTOR_FAULT for a 4-byte transposed transfer), and
             12 partial-tile FP16/BF16 GEMMs through the public path

O0/O1 at 447/447 on both profiles are run separately (`make r37` and
`tools/r37/aot_run.py --profile core-xpose-v1 --levels 0 1`) because they
take hours on this simulator; the status document records them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'flow' / 'synth'))
import harness as H                                         # noqa: E402

OUT = ROOT / 'sim' / 'build' / 'r38' / 'loop'
FROZEN = json.loads((ROOT / 'spec' / 'r38' / 'baseline_cycles.json').read_text())['cycles']
BASE_RUN = ROOT / 'sim' / 'build' / 'r38' / 'baseline_O2.json'
PARENT_COMMIT = '3b5b28f'
GEMM = 'matmul_fp32_fp16_fp8/fp16'
RESULTS = {}


def report(name, problems, info=None):
    RESULTS[name] = dict(problems=problems, info=info)
    print(f'{"FAIL" if problems else "PASS"} {name}' + (f'  {info}' if info else '')
          + ''.join(f'\n     {p}' for p in problems[:8]), flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-full', action='store_true', help='skip the two 149-case runs (development)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bench_off = H.build_bench('core-sim', OUT / 'bench-core-sim.vvp')
    bench_on = H.build_bench('core-xpose-v1', OUT / 'bench-core-xpose-v1.vvp')
    only = None if not args.skip_full else [GEMM, 'vector_add', 'softmax/n37']

    # 1 -----------------------------------------------------------------
    off = H.run_cases(2, OUT / 'art-off', only=only, profile='core-sim',
                      session_kwargs=dict(bench=str(bench_off), counters=True), progress=False)
    base_art = {r['case']: r.get('artifact_sha256') for r in json.loads(BASE_RUN.read_text())['cases']}
    probs = [f'{r["case"]}: {r["detail"]}' for r in off if not r['ok']]
    probs += [f'{r["case"]}: {r["cycles"]} cycles, baseline {FROZEN.get(r["case"])}'
              for r in off if r['ok'] and r['cycles'] != FROZEN.get(r['case'])]
    probs += [f'{r["case"]}: artifact differs from the baseline artifact'
              for r in off if r['ok'] and sha(ROOT / r['artifact']) != base_art.get(r['case'])]
    if not args.skip_full and len(off) != 149:
        probs.append(f'{len(off)} cases, expected 149')
    report('1 feature off: core-sim identical to the baseline', probs, f'{len(off)} cases')

    # 2 -----------------------------------------------------------------
    on = H.run_cases(2, OUT / 'art-on', only=only, profile='core-xpose-v1',
                     session_kwargs=dict(bench=str(bench_on), counters=True), progress=False)
    probs = [f'{r["case"]}: {r["detail"]}' for r in on if not r['ok']]
    if not args.skip_full and len(on) != 149:
        probs.append(f'{len(on)} cases, expected 149')
    g_off = next(r for r in off if r['case'] == GEMM)
    g_on = next(r for r in on if r['case'] == GEMM)
    t_off, t_on = g_off['counters']['total'], g_on['counters']['total']
    beats = lambda t: t['perf_mem_rd_beats'] + t['perf_mem_wr_beats']      # noqa: E731
    info = dict(cycles=(g_off['cycles'], g_on['cycles']), beats=(beats(t_off), beats(t_on)),
                x_lsu=(t_off['perf_x_lsu'], t_on['perf_x_lsu']),
                issued=(t_off['perf_issued'], t_on['perf_issued']))
    if not g_on['cycles'] < g_off['cycles']:
        probs.append(f'{GEMM}: not faster ({g_off["cycles"]} -> {g_on["cycles"]})')
    if not beats(t_on) < beats(t_off):
        probs.append('memory beats did not fall: the counters do not explain the gain')
    if not t_on['perf_x_lsu'] < t_off['perf_x_lsu']:
        probs.append('LSU stall cycles did not fall')
    speed = {r['case']: (FROZEN[r['case']], r['cycles']) for r in on
             if r['ok'] and r['cycles'] != FROZEN.get(r['case'])}
    report('2 feature on: core-xpose-v1 correct, GEMM faster for a counted reason', probs,
           dict(info, changed_cases=speed))

    # 3 -----------------------------------------------------------------
    from tnca.runtime import Session
    from tnca.errors import CapabilityError, TncaError
    probs = []
    xart = ROOT / g_on['artifact']
    try:
        with Session(profile='core-sim', bench=str(bench_off)) as s:
            s.load(str(xart))
        probs.append('a core-xpose-v1 artifact loaded on core-sim')
    except CapabilityError as e:
        if 'tensor_transpose' not in str(e):
            probs.append(f'refused, but without naming the capability: {e}')
    except TncaError as e:
        probs.append(f'refused with {e.code}, expected CAPABILITY_MISSING: {e}')
    # a core-sim artifact on the new hardware: same code, same cycles
    import copy
    cases = H.cases()
    c = copy.copy(cases[GEMM])
    c.buffers = copy.deepcopy(c.buffers)
    orig_run = H.aot_run.run
    try:
        _, det = orig_run(GEMM, c, 2, 'core-sim', OUT / 'art-cross',
                          session_kwargs=dict(bench=str(bench_on)), session_profile='core-xpose-v1')
        if det['cycles'] != FROZEN[GEMM]:
            probs.append(f'core-sim artifact on core-xpose-v1: {det["cycles"]} cycles, baseline {FROZEN[GEMM]}')
    except TypeError:
        probs.append('could not run a core-sim artifact under a core-xpose-v1 session')
    except Exception as e:                                # noqa: BLE001
        probs.append(f'core-sim artifact on core-xpose-v1 failed: {type(e).__name__}: {e}')
    # a descriptor the engine refuses: row stride 36 bytes (K = 18 FP16)
    import xpose_cases as X
    X.SHAPES = [(16, 16, 18)]
    bad = X.build(H.TILEBENCH)
    bad = {k: v for k, v in bad.items() if 'fp16' in k}
    r_on = H.run_points(bad, 2, OUT / 'art-bad-on', profile='core-xpose-v1',
                        session_kwargs=dict(bench=str(bench_on)), progress=False)[0]
    r_off = H.run_points(bad, 2, OUT / 'art-bad-off', profile='core-sim',
                         session_kwargs=dict(bench=str(bench_off)), progress=False)[0]
    if r_on['ok'] or 'code 0x04' not in r_on.get('detail', ''):
        probs.append(f'unaligned descriptor on core-xpose-v1 did not trap DESCRIPTOR_FAULT: {r_on.get("detail")}')
    if not r_off['ok']:
        probs.append(f'the same kernel failed on core-sim: {r_off["detail"]}')
    report('3 compatibility and refusal', probs,
           dict(unaligned_on=r_on.get('detail', '')[:90], unaligned_off='pass' if r_off['ok'] else 'fail'))

    # 4, 5 ----------------------------------------------------------------
    import synth
    parent_dir = OUT / 'parent-rtl' / 'core'
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / 'tnca_tensor.v').write_text(subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{PARENT_COMMIT}:rtl/core/tnca_tensor.v'],
        capture_output=True, text=True, check=True).stdout)
    cur = {x: synth.run_one(f'tnca_tensor@x{x}', ['core/tnca_tensor.v'], {'HAS_XPOSE': x}) for x in (0, 1)}
    rtl_dir = synth.RTL
    synth.RTL = str(OUT / 'parent-rtl')
    parent = synth.run_one('tnca_tensor@parent', ['core/tnca_tensor.v'], {})
    synth.RTL = rtl_dir
    rel = lambda a, b: (a - b) / b                                          # noqa: E731
    probs = []
    for k in ('gates', 'flops'):
        if abs(rel(cur[0][k], parent[k])) > 0.005:
            probs.append(f'feature off: {k} {cur[0][k]} vs parent {parent[k]}')
    proxy = {name: {k: r[k] for k in ('gates', 'flops', 'depth')}
             for name, r in (('parent', parent), ('off', cur[0]), ('on', cur[1]))}
    report('4 rollback: feature off is the parent engine (structural proxy)', probs, proxy)
    cost = {k: cur[1][k] - cur[0][k] for k in ('gates', 'flops', 'depth')}
    report('5 resource: cost of the feature (structural-proxy, tnca_tensor)', [], cost)

    # 6 -----------------------------------------------------------------
    r = subprocess.run(['make', '-s', 'r38-xpose-rtl'], cwd=ROOT, capture_output=True, text=True)
    probs = [] if r.returncode == 0 else [l for l in r.stdout.splitlines() if 'FAIL' in l or 'ERROR' in l] or ['make r38-xpose-rtl failed']
    X.SHAPES = [(16, 16, 24), (16, 24, 16), (24, 16, 16), (16, 40, 40), (40, 24, 48), (16, 16, 8)]
    edge = H.run_points(X.build(H.TILEBENCH), 2, OUT / 'art-edge', profile='core-xpose-v1',
                        session_kwargs=dict(bench=str(bench_on)), progress=False)
    probs += [f'{e["case"]}: {e["detail"]}' for e in edge if not e['ok']]
    report('6 ISA-level and partial-tile tests', probs, f'{sum(e["ok"] for e in edge)}/{len(edge)} edge GEMMs')

    (OUT / 'report.json').write_text(json.dumps(RESULTS, indent=1, default=str) + '\n')
    failed = [k for k, v in RESULTS.items() if v['problems']]
    print(f'loop: {"FAIL" if failed else "PASS"}' + (' (--skip-full: NOT the acceptance run)' if args.skip_full else ''))
    return 1 if failed or args.skip_full else 0


if __name__ == '__main__':
    sys.exit(main())
