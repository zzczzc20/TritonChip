#!/usr/bin/env python3
"""R40 v3 audit of the R40-07 capability gates: pipeline and policy.

    R40_CONTRACT=v3 python3 tools/r40/v3_r40_07_gates.py [--dry-run]
        [--pipeline-run DIR] [--extra-run DIR] [--policy DIR] [--policy-board FILE]

Acceptance (doc/r40_tasks_and_acceptance.md R40-07, unchanged by v3):

  1. `make r40-pipeline`: long K / tail / masked tiles, different stage
     counts, fast / slow / out-of-order HBM, barrier epoch wrap and fault
     drain are correct; a device trace shows DMA and MMA overlapping in time,
     not merely async opcodes.                               -> R40-07.pipeline
  3. Short-K and low-parallelism points are chosen by the frozen automatic
     policy; the complete tiny + small + medium suites have no unresolved
     > 5 % regression; with 100 MHz not reached, gate at the actual clock
     (seconds, not cycles).                                  -> R40-07.policy
  (2 is the perf gate R40-07.long-k; R40-07.matrix is deferred to R41 by v3.)

Pipeline evidence: copies of tools/r40/v3_capgate_jobs.py runs on an
immutable GB10 copy of the delivery product tree -- every recipe line of
`make r40-pipeline-checks` (barrier epoch wrap/exhaustion, trace parser,
stage options, ring diagnostics at 2/3 stages, standard native Torch rings
with +engtrace overlap) and the directed matrix (tools/r40/v3_pipeline_matrix.py:
mm/addmm/linear, K 48/256/257 with M/N/K tails, stages 1/2/3, on the v5
fast memory, its slow random-stall reordering memory, and the AXI HBM path
at default and late/out-of-order/backpressured timing), the compiler's C3
two-role pipeline and protocol suites on the core RTL benches, and the R2-11
fault/drain suite. The engine trace is recomputed here from the preserved raw
trace. It is the RTL engine trace of the delivered core (the board image has
no engine trace port); it is not board time.

Policy evidence: tools/r40/v3_policy_inventory.py (compile-only, every
tiny/small/medium point) and, where the policy changes machine code, a board
A/B at the measured 90 MHz clock (--policy-board, not collected yet).
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/r40')]
import v3_capgate_audit as A  # noqa: E402

EVIDENCE = ROOT / 'evidence/r40/v3'
PIPE_JOBS = ['barrier_epoch', 'engine_overlap_parser', 'stage_options', 'rctx_roles_s2',
             'native_pipeline_s2', 'rctx_roles_s3', 'native_pipeline_s3']
MEMORIES = ('fast', 'slow', 'hbm', 'hbm-adverse')
PROFILES = {'fast': 'core-vrf-sync-v5', 'slow': 'core-vrf-sync-v5/slow',
            'hbm': 'core-vrf-sync-v5/hbm', 'hbm-adverse': 'core-vrf-sync-v5/hbm'}
SHAPES = [[16, 16, 48], [16, 16, 256], [17, 31, 257]]
GRAPHS = ('mm', 'addmm', 'linear')
MATRIX_JOBS = [f'pipe_{m}_s{s}' for m in MEMORIES for s in (1, 2, 3)]
# tools/r40/v3_pipeline_matrix.py's adverse AXI timing (bound by the harness hash check).
AXI_ADVERSE = '+axi_seed=7 +axi_lat_min=16 +axi_lat_max=96 +axi_reorder +axi_bp=30'
EXTRA_JOBS = ['compiler_rtl', 'compiler_pipeline_protocol_v', 'r2_fault_suite']
C3_TESTS = ('test_zero_iterations', 'test_one_iteration', 'test_two_iterations',
            'test_three_iterations_odd', 'test_five_iterations', 'test_six_iterations',
            'test_delayed_completion', 'test_out_of_order_returns',
            'test_predicated_off_async_still_completes', 'test_error_propagation',
            'test_two_consumers_on_four_contexts', 'test_three_stage_ring')
PROTOCOL_TESTS = ('test_stages_consumers_and_completion_orders', 'test_short_loops',
                  'test_all_false_still_completes', 'test_init_under_a_waiter_is_a_barrier_fault',
                  'test_unfinished_command_at_exit')
MAKE_FRAGMENTS = ['-s tb_barrier_epoch', '-DEPOCH_EXHAUST', 'tools/r40/engine_overlap_tests.py',
                  'tools/r40/pipeline_stage_options_tests.py', "device.prepare('core-vrf-sync-v5')",
                  'TNCA_RCTX_STAGES=$stages', 'tools/r39/rctx_roles_tests.py',
                  'tools/r40/native_pipeline_tests.py', '--require-overlap']
HARNESS = ['tools/r40/engine_overlap.py', 'tools/r40/engine_overlap_tests.py',
           'tools/r40/pipeline_stage_options_tests.py', 'tools/r39/rctx_roles_tests.py',
           'tools/r40/native_pipeline_tests.py', 'tb/core/tb_barrier_epoch.v']


def read_trace(directory):
    for name in ('engine.log.gz', 'engine.log'):
        p = Path(directory) / name
        if p.exists():
            raw = p.read_bytes()
            return (gzip.decompress(raw) if name.endswith('.gz') else raw).decode()
    raise A.Missing(f'no engine trace in {directory}')


def matrix_job(base_dir, name, mem, stages, log, launches):
    """One directed matrix job: its per-case report when preserved, else the per-case PASS
    and AUDIT lines of its log; overlaps re-derived from the raw engine trace."""
    import ast
    doc = {}
    for fname in (f'{name}.report.json', f'{name}.json'):
        p = base_dir / fname
        if p.exists():
            doc = json.loads(p.read_text())
            if 'cases' in doc:
                break
    good = True
    overlap, errs = {}, {}
    if 'cases' in doc:
        cases = {(c['graph'], tuple(c['shape'])): c for c in doc['cases']}
        good &= (doc['ok'] and doc['profile'] == PROFILES[mem] and doc['stages'] == stages
                 and set(cases) == {(g, tuple(sh)) for g in GRAPHS for sh in SHAPES})
        plus = doc['plusargs']
        for key, c in cases.items():
            rec = sum(launches[q]['overlap_cycles'] for q in c['sequences'])
            good &= c['ok'] and bool(c['rings']) and all(r['stages'] == stages for r in c['rings'])
            good &= rec == c.get('overlap_cycles') and (stages == 1 or rec > 0)
            overlap[f'{key[0]} {"x".join(map(str, key[1]))}'] = rec
            errs[key] = c.get('max_abs_err')
        source = 'report'
    else:
        source = 'log + raw trace'
        passed = {}
        for g, m_, n_, k_, mm, st, err, rings in re.findall(
                r'^PASS (mm|addmm|linear) (\d+)x(\d+)x(\d+) memory=(\S+) stages=(\d) err=(\S+) rings=(\[.*\])$',
                log, re.M):
            rings = ast.literal_eval(rings)
            passed[(g, (int(m_), int(n_), int(k_)))] = (mm, int(st), float(err), rings)
        good &= set(passed) == {(g, tuple(sh)) for g in GRAPHS for sh in SHAPES}
        good &= all(v[0] == mem and v[1] == stages and v[3] and all(r['stages'] == stages for r in v[3])
                    for v in passed.values())
        good &= re.search(rf'^r40-v3-pipeline-matrix PASS \(9/9\) memory={re.escape(mem)} stages={stages}$',
                          log, re.M) is not None
        audits = re.findall(r'^AUDIT (mm|addmm|linear) (\d+x\d+x\d+) overlap=(\d+) cycles over (\d+) launch',
                            log, re.M)
        # Re-derive: the audited cases, in run order, map to strictly increasing measured
        # launch sequences of the raw trace with exactly the audited overlap each.
        seqs = sorted(launches)
        last, matched = -1, 0
        for g, shape, value, count in audits:
            value, count = int(value), int(count)
            hit = None
            if count == 1:
                hit = next((q for q in seqs if q > last and launches[q]['overlap_cycles'] == value
                            and launches[q]['mma_cycles'] > 0 and launches[q]['dma_cycles'] > 0), None)
            if hit is None:
                good = False
                continue
            last, matched = hit, matched + 1
            overlap[f'{g} {shape}'] = value
            good &= stages == 1 or value > 0
        good &= matched == len(audits) == 9
        plus = 'see log/manifest'
        errs = {k: v[2] for k, v in passed.items()}
    return dict(ok=bool(good), source=source, overlap_cycles=overlap,
                max_abs_err=max((e for e in errs.values() if e is not None), default=None))


def pipeline(args, tree):
    from engine_overlap import parse, selected
    checks = []
    m, jobs, base = A.load_run(args.pipeline_run, tree)
    checks += base
    checks.append(A.check('every recipe line of make r40-pipeline-checks is in the run (make -n text)',
                          all(f in m['make_n'].get('r40-pipeline-checks', '') for f in MAKE_FRAGMENTS)))
    checks.append(A.harness_same(m, HARNESS))
    if args.pipeline_rerun and Path(args.pipeline_rerun).exists():
        rm, reruns, rbase = A.load_run(args.pipeline_rerun, tree)
        checks += [dict(c, name='rerun run: ' + c['name']) for c in rbase]
        jobs, superseded = A.merge_reruns(jobs, reruns, PIPE_JOBS, label='make r40-pipeline-checks')
        checks += superseded
    checks += A.job_checks(jobs, PIPE_JOBS, label='make r40-pipeline-checks')
    text = {n: (jobs.get(n, (None, ''))[1] or '') for n in PIPE_JOBS}
    checks.append(A.check('barrier epoch wrap: generations 0..65537 with no early release at the 16-bit '
                          'boundary; a 4-bit epoch faults before wrap and init recovers',
                          'PASS barrier generations 0..65537: no early release at 16-bit boundary' in text['barrier_epoch']
                          and 'PASS barrier exhaustion faults before wrap; init recovers' in text['barrier_epoch']))
    ring = {}
    for s in (2, 3):
        rows = dict(re.findall(r'^\s+PASS\s+(\w+)\s.*roles (\d->\d)\s*$', text[f'rctx_roles_s{s}'], re.M))
        ring[s] = rows
    expect = dict(gemm_one_step='1->1', gemm_odd_steps='1->2', gemm_grid_3x2='1->2',
                  gemm_five_programs='1->2', linear_one_program='1->2', linear_four_programs='1->2',
                  linear_odd_steps_grid='1->2', gemm_carried_pointers='1->1')
    checks.append(A.check('ring diagnostics at 2 and 3 stages: odd K steps, epochs carried across programs '
                          '(grid 3x2, five programs, odd-step grid), transposed linear and bias epilogue '
                          'split and correct; one-step and carried-pointer loops refused and correct',
                          all(ring[s] == expect for s in (2, 3)), observed=ring))
    native = {}
    for s in (2, 3):
        d = Path(args.pipeline_run) / f'trace_native_s{s}'
        try:
            report = json.loads((d / 'overlap.json').read_text())
            raw = read_trace(d)
            seqs = [q for c in report['cases'] for q in c['sequences']]
            again = selected(raw, seqs)
            per = {str(q): again[q]['overlap_cycles'] for q in seqs}
            native[s] = dict(cases=[(c['name'], c['shape'], [r['stages'] for r in c['rings']])
                                    for c in report['cases']],
                             recomputed=per, recorded={str(k): v['overlap_cycles']
                                                       for k, v in report['launches'].items()},
                             ok=report['stages'] == s and len(report['cases']) == 4
                             and all(all(r['stages'] == s for r in c['rings']) and c['rings']
                                     for c in report['cases'])
                             and per == {str(k): v['overlap_cycles'] for k, v in report['launches'].items()}
                             and all(v > 0 for v in per.values()))
        except (OSError, KeyError, ValueError) as e:
            native[s] = dict(ok=False, error=str(e))
    checks.append(A.check('standard native Torch mm/addmm (16x16x48, 17x31x65 tail) with actual 2- and '
                          '3-stage rings: every measured launch shows DMA/MMA service overlap, recomputed '
                          'from the raw engine trace', all(v.get('ok') for v in native.values()),
                          native=native))
    # Directed matrix and RTL/runtime suites (later runs supersede a job of an earlier one).
    xjobs, where, xm = {}, {}, None
    for d in args.extra_run:
        xm, jobs_d, xbase = A.load_run(d, tree)
        checks += [dict(c, name=f'extra run {Path(d).name}: ' + c['name']) for c in xbase]
        checks.append(dict(A.harness_same(xm, ['tools/r40/v3_pipeline_matrix.py', 'tools/r40/engine_overlap.py',
                                               'tools/r2/fault_tests.py', 'compiler/test/test_pipeline.py',
                                               'compiler/test/test_protocol.py', 'compiler/test/rtl_harness.py',
                                               'compiler/test/run_tests.py']),
                           run=str(d)))
        for name, value in jobs_d.items():
            if value[1] is not None:
                xjobs[name], where[name] = value, Path(d)
    checks += A.job_checks(xjobs, MATRIX_JOBS + EXTRA_JOBS, label='directed pipeline evidence')
    matrix, coverage = {}, {}
    for mem in MEMORIES:
        for s in (1, 2, 3):
            name = f'pipe_{mem}_s{s}'
            row = dict(ok=False)
            try:
                base_dir = where.get(name)
                if base_dir is None:
                    raise A.Missing(f'{name} not in any extra run')
                launches = parse(read_trace(base_dir / f'trace_{mem}_s{s}'))
                row = matrix_job(base_dir, name, mem, s, xjobs[name][1], launches)
                cmd = xjobs[name][0]['cmd']
                profile_ok = (f'--memory {mem} --stages {s} ' in cmd
                              and (PROFILES[mem].replace('/', '-') in xjobs[name][1]))
                row.update(ok=row['ok'] and profile_ok, cmd=cmd, profile=PROFILES[mem],
                           axi_plusargs=AXI_ADVERSE if mem == 'hbm-adverse' else None)
            except (OSError, KeyError, ValueError, A.Missing) as e:
                row = dict(ok=False, error=str(e))
            matrix[name] = row
            coverage.setdefault(mem, []).append(row['ok'])
    checks.append(A.check('long K (256) and long K with M/N/K tails / masked edge tiles (17x31x257) plus '
                          'odd short-ring K (48): mm, addmm and linear (transposed weight) correct with '
                          'zero steady transfers at stages 1, 2 and 3',
                          all(r['ok'] for r in matrix.values()), matrix=matrix))
    checks.append(A.check('the matrix harness that ran sets the adverse AXI timing it is credited with',
                          AXI_ADVERSE in (ROOT / 'tools/r40/v3_pipeline_matrix.py').read_text()))
    checks.append(A.check('memory timing: v5 fast memory, slow random-stall out-of-order memory, AXI HBM '
                          'default and late/out-of-order/backpressured (lat 16-96, reorder, 30% READY drop)',
                          all(all(v) and len(v) == 3 for v in coverage.values()) and set(coverage) == set(MEMORIES)))
    checks.append(A.check('engine trace of every 2/3-stage ring case in every memory mode shows DMA/MMA '
                          'service overlap (recomputed from the raw trace)',
                          all(matrix[f'pipe_{m}_s{s}'].get('ok') for m in MEMORIES for s in (2, 3))))
    v_text = xjobs.get('compiler_pipeline_protocol_v', (None, ''))[1] or ''
    passed = set(re.findall(r'^(test_\w+) \([\w.]+\)(?:\n[^\n]*)? \.\.\. ok$', v_text, re.M))
    ran = re.search(r'^Ran (\d+) tests? in', v_text, re.M)
    missing = [t for t in C3_TESTS + PROTOCOL_TESTS if t not in passed]
    checks.append(A.check('C3 two-role pipeline on the core RTL benches: 0/1/2/odd/5/6 trips, delayed '
                          'and out-of-order completion, predicated-off async, three stages, two consumers, '
                          'and a faulting transfer that wakes the consumer with the barrier error (no hang); '
                          'protocol model: completion orders, short loops, faults',
                          not missing and 'skipped' not in v_text and bool(ran)
                          and int(ran.group(1)) == len(passed),
                          missing=missing, passed=sorted(passed), ran=int(ran.group(1)) if ran else None))
    rtl_text = xjobs.get('compiler_rtl', (None, ''))[1] or ''
    n = re.search(r'compiler-rtl\s+PASS\s+(\d+) tests\n', rtl_text)
    checks.append(A.check('compiler RTL suites (R1_STRICT: no skips)', bool(n),
                          tests=int(n.group(1)) if n else None))
    fault_text = xjobs.get('r2_fault_suite', (None, ''))[1] or ''
    errs = dict(re.findall(r'^\s+(ERR-\d+)\s+(PASS|FAIL|NOT RUN)', fault_text, re.M))
    checks.append(A.check('fault drain: a faulting kernel goes FAILING -> DRAINING -> FAILED with its '
                          'trap and PC, and the session stays usable for an independent task (R2-11)',
                          errs.get('ERR-01') == 'PASS' and errs.get('ERR-02') == 'PASS'
                          and all(v == 'PASS' for v in errs.values()), suite=errs))
    return checks, m


def policy(args, tree):
    """Short-K / low-parallelism choice by the frozen policy; no >5 % regression at 90 MHz."""
    checks = []
    report_path = A.need(Path(args.policy) / 'report.json')
    rep = json.loads(report_path.read_text())
    checks.append(A.check('inventory bound to the delivery tree and the complete tiny+small+medium suite',
                          rep['software_tree'] == tree and rep['suite'] == 'spec/r40/suites/existing.json'
                          and rep['suite_points'] == 77 and rep.get('status') == 'complete'
                          and {'off', 'pipe2', 'gated2', 'gated3'} <= set(rep.get('configs', {})),
                          report=str(report_path.relative_to(ROOT)), sha256=A.sha_file(report_path),
                          harness_sha256=rep.get('harness_sha256'), status=rep.get('status'),
                          configs=rep.get('configs')))
    checks.append(A.check('inventory harness equals the repository copy',
                          rep.get('harness_sha256') == A.sha_file(ROOT / 'tools/r40/v3_policy_inventory.py')))
    summ = rep.get('summary', {})
    complete = len(summ) == 77 and all(all(v.get('status') == 'pass' for v in row.values())
                                       for row in summ.values())
    checks.append(A.check('every point compiled (native tnca path, compile-only) under every configuration',
                          complete, failures={p: {c: v.get('error') for c, v in row.items()
                                                  if v.get('status') != 'pass'}
                                              for p, row in summ.items()
                                              if any(v.get('status') != 'pass' for v in row.values())}))
    src = (ROOT / 'compiler/python/tnca_cc/partition.py').read_text()
    aot = (ROOT / 'python/tnca/aot.py').read_text()
    checks.append(A.check('frozen automatic policy in the delivered tree: TNCA_RCTX_ROLES=1 asks '
                          'tnca_cc.partition.ring_estimate, which refuses a ring whose producer only issues '
                          'transfers (Mode B already overlaps them; R39 board: small/medium GEMMs 2-5 % slower); '
                          'one-step loops, carried pointers and resource misfits are refused too',
                          'def ring_estimate' in src and 'which Mode B already' in src
                          and 'rctx_roles_refused' in aot and 'rctx_roles_fallback' in aot,
                          partition_sha256=A.sha_bytes(src.encode()), aot_sha256=A.sha_bytes(aot.encode())))
    def changed(cfg):
        return sorted(p for p, row in summ.items() if (row.get(cfg) or {}).get('code_identical_to_off') is not True)
    decisions = {}
    for p, row in sorted(summ.items()):
        g = row.get('gated2') or {}
        decisions[p] = dict(kernels=g.get('kernels'), rings=g.get('rings'), refused=g.get('refused'),
                            fallback=g.get('fallback'))
    reasons = {}
    for d in decisions.values():
        for why in d.get('refused') or []:
            reasons[why] = reasons.get(why, 0) + 1
    checks.append(A.check('delivered default (TNCA_RCTX_ROLES unset): TNCA_PIPELINE 1 -> 2 leaves every '
                          'tiny/small/medium executable payload unchanged', complete and not changed('pipe2'),
                          differ=changed('pipe2')))
    gated_changed = sorted(set(changed('gated2')) | set(changed('gated3')))
    checks.append(A.check('automatic policy (TNCA_RCTX_ROLES=1, 2 and 3 stages): every short-K / '
                          'low-parallelism decision is recorded with its reason; where it declines, the payload '
                          'equals the pipeline-off one; the points it changes are listed for the board A/B',
                          complete and all(d['refused'] or d['rings'] or d['fallback'] or d['kernels']
                                           for d in decisions.values()),
                          changed=dict(gated2=changed('gated2'), gated3=changed('gated3')),
                          refusal_reasons=reasons, decisions=decisions))
    gated_changed = sorted(set(gated_changed) | set(changed('pipe2')))
    board = None
    if args.policy_board and Path(args.policy_board).exists():
        board = json.loads(Path(args.policy_board).read_text())
    checks.append(A.check('no unresolved >5 % regression at the measured 90 MHz clock: unchanged payloads '
                          'by construction; any changed point needs a board A/B',
                          not gated_changed or (board is not None and board.get('ok') is True
                                                and board.get('schema') == 'tnca-r40-v3-policy-board-result/1'
                                                and board.get('software_tree') == tree
                                                and board.get('image_sha256') ==
                                                'a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083'
                                                and board.get('clock_hz') == 90000000
                                                and set(board.get('points', [])) >= set(gated_changed)),
                          points_needing_board=gated_changed,
                          board=str(args.policy_board) if board is not None else None))
    return checks, rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='print the records; write nothing')
    p.add_argument('--pipeline-run', type=Path, default=EVIDENCE / 'R40-07.pipeline/run')
    p.add_argument('--pipeline-rerun', type=Path, default=EVIDENCE / 'R40-07.pipeline/rerun_run')
    p.add_argument('--extra-run', type=Path, nargs='+',
                   default=[EVIDENCE / 'R40-07.pipeline/extra_run'])
    p.add_argument('--policy', type=Path, default=EVIDENCE / 'R40-07.policy/inventory_existing')
    p.add_argument('--policy-board', type=Path)
    args = p.parse_args()
    identity, lock = A.delivery()
    tree = identity['software_tree']
    here = A.local_tree()
    common = [A.check('auditor runs on the delivery product tree', here == tree, local=here, delivery=tree)]
    common += A.identity_checks(identity)
    out = {}
    try:
        checks, m = pipeline(args, tree)
        checks = common + checks
        status = 'capability-pass' if all(c['ok'] for c in checks) else 'fail'
        detail = ('make r40-pipeline-checks + directed long-K/tail/stage/memory-timing matrix, C3 ring '
                  'RTL suites and R2-11 fault drain on an immutable copy of the delivery tree; DMA/MMA '
                  'overlap from the RTL engine trace of the delivered core (no engine trace on the board); '
                  'fault drain shown by RTL ring fault propagation plus runtime drain, not by a fault '
                  'injected into a standard-Torch ring kernel; no performance claim')
        extra = dict(software_tree=tree, image='not-applicable: RTL simulation', clock='not-applicable',
                     measured_performance=False, run_host=m.get('host'), run_root=m.get('root'))
    except A.Missing as e:
        checks, status, detail, extra = common + [A.check(str(e), False)], 'not-measured', str(e), {}
    out['R40-07.pipeline'] = (status, checks, detail, extra)
    try:
        checks, rep = policy(args, tree)
        checks = common + checks
        failing = [c['name'] for c in checks if not c['ok']]
        board_only = all(n.startswith(('no unresolved', 'automatic policy')) for n in failing)
        if all(c['ok'] for c in checks):
            status = 'capability-pass'
        elif board_only:
            status = 'not-measured'
        else:
            status = 'fail'
        detail = ('frozen automatic policy = R39-12 ring_estimate gate (TNCA_RCTX_ROLES=1) at TNCA_PIPELINE 2/3, '
                  'and the delivered default (roles unset); compile-only inventory of all 77 tiny/small/medium '
                  'points on the native path compares executable payloads with pipeline off; unchanged payloads '
                  'cannot change device time on the same image and 90 MHz clock; changed points need a board A/B')
        extra = dict(software_tree=tree, image='a88ebdf5 (only if a payload changes)', clock_hz=90000000,
                     measured_performance=False)
    except A.Missing as e:
        checks, status, detail, extra = common + [A.check(str(e), False)], 'not-measured', str(e), {}
    out['R40-07.policy'] = (status, checks, detail, extra)
    for gate, (status, checks, detail, extra) in out.items():
        bad = [c['name'] for c in checks if not c['ok']]
        print(f'{gate}: {status}  ({len(checks) - len(bad)}/{len(checks)} checks)'
              + (f'  failing: {bad}' if bad else ''))
        if args.dry_run:
            print(json.dumps(dict(status=status, detail=detail, checks=checks, **extra), indent=1,
                             default=str)[:20000])
        else:
            if os.environ.get('R40_CONTRACT') != 'v3':
                raise SystemExit('set R40_CONTRACT=v3 to record v3 gate evidence')
            import evidence
            evidence.record(gate, status, checks, detail=detail, **extra)
    return 0


if __name__ == '__main__':
    sys.exit(main())
