#!/usr/bin/env python3
"""R40 v3 audit of the R40-02 capability gates: zero-transfer and stress.

Judges preserved raw evidence (never the summary a producer wrote about
itself) and only on the v3 delivery identity resolved by
contract_v3.delivery(plan): the frozen image and 90 MHz clock, and the product
tree named by the frozen plan or its identity amendment. Board evidence must
come from a functional board queue on that identity. Pure RTL-simulation
evidence must come from an immutable copy of the delivery product tree whose
staged RTL equals its own rtl/ and tb/ sources (v3 change 2); the product
files that differ from the tree the image was built from are recorded.
Evidence measured on another product tree is not-measured, never a pass.
Performance gates R40-02.resident and R40-02.batched are out of scope.

    R40_CONTRACT=v3 python3 tools/r40/v3_r40_02_gates.py [--dry-run]
        [--board QUEUE_DIR] [--board-snapshot SOURCE_ROOT] [--model-report report.json[.gz]]
        [--sim RUN_DIR ...]
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'tools/r40')]
import contract_v3

PLAN = json.loads((ROOT / 'doc/r40_v3/plan.json').read_text())
DELIVERY = contract_v3.delivery(PLAN)
TREE = DELIVERY['software_tree']
IMAGE = DELIVERY['image_sha256']
CLOCK_HZ = DELIVERY['clock_hz']
BACKEND = DELIVERY['backend']
BACKEND_SHAPE = DELIVERY['backend_args_shape']
EVIDENCE = ROOT / 'evidence/r40/v3'
# Simulation steps each gate consumes (see tools/r40/v3_sim_chains.py).
STRESS_STEPS = ('batch-1000', 'replay-1000', 'recovery', 'recovery-multicore-dma', 'lifetime',
                'allocation-pressure', 'residency-controls', 'device-copy', 'io-concurrency', 'child-exec')
ZERO_SIM_STEPS = ('residency-controls', 'frozen-attention-tiny')
HELPERS = ('tools/r40/frozen_residency_tests.py', 'tools/r40/bucket_decode.py')
BOARD_HARNESS = ('frozen_residency_tests.py', 'native_model.py', 'bucket_decode.py')


class Missing(Exception):
    """Evidence absent or measured on another identity: not-measured, never a pass."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def need(path):
    path = Path(path)
    if not path.exists():
        raise Missing(f'missing evidence {path}')
    return path


def same_identity(tree, what):
    if tree != TREE:
        raise Missing(f'{what} was measured on {tree}; the delivery product tree is {TREE}')


def backend_args_ok(args):
    return (isinstance(args, dict) and {k: v for k, v in args.items() if k != 'xclbin'} == BACKEND_SHAPE and
            str(args.get('xclbin', '')).endswith('.xclbin'))


def local_tree():
    from archbench.provenance import software_tree
    return software_tree(refresh=True)


def amendment_record():
    """Product files that differ between the delivery tree and the tree the
    frozen image/plan named (empty without an identity amendment)."""
    if 'identity_amendment' not in DELIVERY:
        return dict(from_software_tree=TREE, files={})
    doc = json.loads((ROOT / DELIVERY['identity_amendment']).read_text())
    for path, digests in doc['files'].items():
        require(file_sha(ROOT / path) == digests['after_sha256'], f'{path} is not the amended delivery file')
    require(all(not p.startswith(('rtl/', 'tb/')) for p in doc['files']),
            'the amendment changes RTL; the delivered image no longer matches its sources')
    return dict(from_software_tree=doc['from_software_tree'], files=doc['files'])


# ---------------------------------------------------------------- simulation
def sim_state(directory):
    directory = need(directory)
    state = json.loads(need(directory / 'state.json').read_text())
    require(state['schema'] == 'tnca-r40-v3-sim-chains/1', 'unknown simulation run schema')
    same_identity(state.get('software_tree'), f'simulation run {directory.name}')
    if 'software_tree_after' not in state:
        raise Missing(f'simulation run {directory.name} still running')
    require(state['software_tree'] == state['software_tree_after'] == state['expect_tree'],
            'simulation product tree changed during the run')
    staged = state['staged_identity']
    require(set(staged) == {'rtl', 'tb/device', 'tb/common'} and
            all(v['source'] == v['staged'] for v in staged.values()),
            'simulated RTL is not byte-identical to the delivery tree sources')
    require(state['prepare']['exit_code'] == 0, 'simulator preparation failed')
    return state


def sim_run(directories, steps):
    """Audit preserved v3_sim_chains runs (oldest first); return (states, used, {step: log}).

    A step's evidence is its latest attempt. An earlier failed attempt is
    superseded only by an attempt whose test script changed (a repaired
    harness); rerunning an unchanged failing test cannot replace it. A step
    stopped by a signal (time budget) is incomplete: not-measured, never a pass.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    states = []
    for d in directories:
        try:
            states.append((Path(d), sim_state(d)))
        except Missing as exc:   # an unfinished run contributes nothing
            if 'still running' not in str(exc):
                raise
    if not states:
        raise Missing(f'no finished delivery-tree simulation run among {[str(d) for d in directories]}')
    chosen, history = {}, {}
    for directory, state in states:
        for chain in state['chains'].values():
            for rec in chain:
                rec = dict(rec, directory=directory, harness=state['harness_sha256'])
                history.setdefault(rec['name'], []).append(rec)
                chosen[rec['name']] = rec
    logs, used = {}, {}
    for name in steps:
        if name not in chosen or 'exit_code' not in chosen[name]:
            raise Missing(f'simulation step {name} not complete')
        rec = chosen[name]
        if rec['exit_code'] < 0:   # stopped by a signal (time budget): incomplete, never a pass
            raise Missing(f'simulation step {name} was terminated by signal {-rec["exit_code"]} before completion')
        script = rec['argv'][2]
        for f in (script, *HELPERS):
            if f in rec['harness']:
                require(file_sha(ROOT / f) == rec['harness'][f], f'{name}: {f} differs from the repository harness')
        for old in history[name][:-1]:
            if (old.get('exit_code') or 0) > 0:   # a real failure, not an unfinished/terminated attempt
                require(old['harness'].get(script) != rec['harness'][script],
                        f'{name}: an unchanged rerun cannot supersede its failed attempt')
        raw = need(rec['directory'] / rec['log']).read_bytes()
        require(sha(raw) == rec['log_sha256'], f'{name}: preserved log changed')
        require(rec['exit_code'] == 0, f'{name}: exited {rec["exit_code"]}')
        logs[name] = raw.decode()
        require(not re.search(r'^Traceback|^FAIL', logs[name], re.M), f'{name}: failure in log')
        used[name] = dict(run=rec['directory'].name, log=rec['log'], wall_s=rec['wall_s'],
                          script_sha256=rec['harness'][script],
                          superseded=[dict(run=o['directory'].name, exit_code=o.get('exit_code'),
                                           script_sha256=o['harness'].get(script))
                                      for o in history[name][:-1]])
    return [s for _, s in states], used, logs


def sim_identity(states):
    return dict(name='simulation identity: delivery product tree before/after, staged RTL/testbench '
                     'byte-identical to the tree, harness hashes equal the repository; product files '
                     'differing from the image source tree recorded', ok=True, software_tree=TREE,
                staged_identity=states[-1]['staged_identity'], runners=[s['runner_sha256'] for s in states],
                host=states[-1]['host'], image_source_delta=amendment_record())


def memory_audit(text, name):
    rows = [ast.literal_eval(m) for m in re.findall(r'^AUDIT (\{.*\})$', text, re.M)]
    require(rows, f'{name}: no allocator high-water audit')
    for row in rows:
        require({'heap_capacity_bytes', 'heap_allocated_bytes', 'heap_peak_bytes',
                 'code_allocated_bytes', 'code_peak_bytes'} <= set(row) and
                0 < row['heap_peak_bytes'] <= row['heap_capacity_bytes'] and
                0 <= row['heap_allocated_bytes'] <= row['heap_peak_bytes'],
                f'{name}: incomplete or inconsistent high-water ledger')
    return rows[-1]


def has(text, line, name, count=1):
    require(text.count(line) == count, f'{name}: expected {count}x {line!r}')


def stress_checks(directories):
    states, used, logs = sim_run(directories, STRESS_STEPS)
    checks = []
    b = logs['batch-1000']
    for line in ('PASS queued multi-kernel graph, event visibility and deferred intermediate free',
                 'PASS completed launch log, unchanged artifacts and zero kernel transfers',
                 'PASS stream contexts, recorded dependency, runtime stream identity and restoration',
                 'PASS concurrent graph execution and thread-local launch audit',
                 'PASS 1000 graph calls, bounded queue, final storage/lease audit',
                 'r40-batch-tests PASS (5/5)'):
        has(b, line, 'batch-1000')
    end = re.search(r'^AUDIT storage_live_bytes=(\d+) storage_peak_bytes=(\d+) live_buffers=(\d+) '
                    r'leases=(\d+) unfinished=(\d+)$', b, re.M)
    require(end and end.group(4) == '0' and end.group(5) == '0', 'batch-1000: leases/unfinished remain')
    checks.append(dict(name='1000 standard Torch graph calls: bounded queue, correct output, final live '
                            'storage unchanged, zero leases/unfinished; end-of-run storage and allocator '
                            'high water audited', ok=True, step=used['batch-1000'],
                       end_storage=dict(live_bytes=int(end.group(1)), peak_bytes=int(end.group(2)),
                                        live_buffers=int(end.group(3))),
                       memory_stats=memory_audit(b, 'batch-1000')))
    checks.append(dict(name='multi-stream events: recorded dependency, cross-stream wait, concurrent '
                            'threads, queued-event visibility while the worker is held', ok=True,
                       step=used['batch-1000']))
    r = logs['replay-1000']
    for line in ('PASS 1000 retained graph replays, bounded submission and storage/lease audit',
                 'PASS retained multi-kernel Torch graph replay, updated input, artifact and byte audit',
                 'PASS shape/stride guard and permanent invalidation before admission',
                 'PASS changed storage alias rejected before admission',
                 'PASS recovery epoch rejects whole graph without submission',
                 'r40-replay-tests PASS (5/5)'):
        has(r, line, 'replay-1000')
    checks.append(dict(name='1000 guarded replays; shape/stride, storage alias and recovery-epoch '
                            'invalidation refused before admission', ok=True, step=used['replay-1000'],
                       memory_stats=memory_audit(r, 'replay-1000')))
    for name, flag in (('recovery', 'False'), ('recovery-multicore-dma', 'True')):
        for line in ('PASS 3 recovery epochs with forced physical address reuse',
                     'PASS stale event, copy, view, launch and growing resize rejected',
                     'PASS old storage destruction preserves the new allocation at the same address',
                     f'r40-recovery-tests PASS (3/3) multicore_dma={flag}'):
            has(logs[name], line, name)
    checks.append(dict(name='stale handles after fault recovery (single-core RTL and the 3-core/3-port '
                            'HBM-DMA platform RTL): event, copy, view, launch, resize refused', ok=True,
                       steps=[used['recovery'], used['recovery-multicore-dma']]))
    has(logs['lifetime'], 'r40-lifetime-tests PASS (2/2)', 'lifetime')
    a = logs['allocation-pressure']
    has(a, 'r40-allocation-pressure-tests PASS (2/2)', 'allocation-pressure')
    checks.append(dict(name='asynchronous free/realloc: retired storage keeps queued leases and delays '
                            'address reuse; allocation waits for retirement under heap pressure', ok=True,
                       steps=[used['lifetime'], used['allocation-pressure']],
                       memory_stats=memory_audit(a, 'allocation-pressure')))
    res, copy = logs['residency-controls'], logs['device-copy']
    for line in ('PASS resident: strided offset aliases and in-place update',
                 'PASS per-kernel: strided offset aliases and in-place update',
                 'r40-residency-tests PASS (6/6)'):
        has(res, line, 'residency-controls')
    for line in ('PASS strided gaps:', 'PASS disjoint storage aliases:'):
        require(line in copy, f'device-copy: missing {line!r}')
    require(re.search(r'^r40-device-copy-tests PASS \((\d+)/\1\), overlap refused$', copy, re.M),
            'device-copy: incomplete')
    checks.append(dict(name='non-contiguous alias/offset: in-place strided aliases under both transfer '
                            'policies, strided/aliased device copies, overlapping copy refused', ok=True,
                       steps=[used['residency-controls'], used['device-copy']]))
    has(logs['io-concurrency'], 'r40-io-concurrency-tests PASS (2/2)', 'io-concurrency')
    require(re.search(r'^OK$', logs['child-exec'], re.M), 'child-exec: unittest not OK')
    checks.append(dict(name='transport payload concurrency and child launch lifecycle', ok=True,
                       steps=[used['io-concurrency'], used['child-exec']]))
    checks.append(sim_identity(states))
    return checks


# --------------------------------------------------------------------- board
def functional_queue(directory, snapshot):
    """A functional board queue on the delivery identity: (state, commands)."""
    directory = need(directory)
    state = json.loads(need(directory / 'state.json').read_text())
    same_identity(state.get('software_tree'), f'board queue {directory}')
    require(state['image_sha256'] == IMAGE and state['clock_hz'] == CLOCK_HZ and
            backend_args_ok(state['backend_args']), 'board queue image/clock/configuration is not the delivery')
    controller = need(directory / 'controller.py').read_text()
    for pattern in (rf"TNCA_BACKEND\W+{BACKEND}\b", r'TNCA_BACKEND_ARGS\W', r"TNCA_ACTIVE_CORES\W+3\b",
                    r"TNCA_TORCH_DEVICE\W+native\b", r'software_tree\(refresh=True\)\s*==\s*tree'):
        require(re.search(pattern, controller), f'board controller lacks {pattern!r}')
    snapshot = need(snapshot)
    for name in BOARD_HARNESS:
        candidates = [snapshot / 'tools/r40' / name, snapshot / name]
        path = next((c for c in candidates if c.exists()), None)
        require(path is not None and file_sha(path) == file_sha(ROOT / 'tools/r40' / name),
                f'board snapshot {name} differs from the repository harness')
    return state, dict(json.loads(need(directory / 'commands.json').read_text()))


def model_stage(directory, state, commands):
    """(completion record, argv) of the queue's board model stage.

    Either a top-level queue stage `frozen-original-model`, or the same stage
    run by the queue's tools/r40/conditional_delivery.py stage; that run's own
    record is preserved as <queue>/<stage name>/delivery.json and must be the
    complete, identity-bound, repository-harness run the queue command names.
    """
    done = {c['name']: c for c in state['completed']}
    if 'frozen-original-model' in done or 'frozen-original-model' in commands:
        if 'frozen-original-model' not in done:
            raise Missing('board model stage not complete')
        return done['frozen-original-model'], commands['frozen-original-model']
    outer = [n for n, argv in commands.items()
             if len(argv) > 2 and str(argv[2]).endswith('/tools/r40/conditional_delivery.py')]
    if not outer:
        raise Missing('board queue has no model stage')
    require(len(outer) == 1, f'board queue has several delivery stages {outer}')
    name = outer[0]
    argv = commands[name]
    if name not in done:
        raise Missing(f'board stage {name} not complete')
    require(done[name]['exit_code'] == 0, f'{name} exited {done[name]["exit_code"]}')
    require('--bucket' in argv and argv[argv.index('--image-sha256') + 1] == IMAGE and
            argv[argv.index('--software-tree-sha256') + 1] == TREE, f'{name}: delivery command changed')
    record = json.loads(need(Path(directory) / name / 'delivery.json').read_text())
    require(record['schema'] == 'tnca-conditional-delivery/1' and
            record['status'] == 'functional-stages-complete-review-required' and
            record['model_cache_policy'] == 'fixed-bucket', f'{name}: delivery run not complete')
    require(record['harness_sha256'] == file_sha(ROOT / 'tools/r40/conditional_delivery.py'),
            f'{name}: tools/r40/conditional_delivery.py differs from the repository harness')
    ident = record['identity']
    require(ident['software_tree'] == TREE and ident['image_sha256'] == IMAGE and ident['clock_hz'] == CLOCK_HZ,
            f'{name}: delivery run identity is not the delivery')
    stages = {s['name']: s['command'] for s in record['stages']}
    require(len(stages) == len(record['stages']) == len(record['completed']) and
            [c['name'] for c in record['completed']] == [s['name'] for s in record['stages']] and
            all(c['exit_code'] == 0 for c in record['completed']), f'{name}: a delivery stage failed or did not run')
    inner = stages.get('frozen-original-model')
    require(inner is not None and inner[inner.index('--out') + 1] ==
            argv[argv.index('--out') + 1].rstrip('/') + '/model', f'{name}: model stage absent or wrote elsewhere')
    return next(c for c in record['completed'] if c['name'] == 'frozen-original-model'), inner


def graph_checks(directory, snapshot):
    state, commands = functional_queue(directory, snapshot)
    frozen = json.loads((ROOT / 'spec/r40/suites/R40-02.graphs3.json').read_text())['points']
    require(len(frozen) == 24, 'frozen graphs3 subset changed')
    done = {c['name']: c for c in state['completed']}
    totals = dict(resident_calls=0, staging_calls=0, staging_bytes=0, kernels=[])
    for i, point in enumerate(frozen):
        name = f'graphs24-point{i:02d}'
        if name not in done:
            raise Missing(f'board graph stage {name} not complete')
        require(done[name]['exit_code'] == 0, f'{name} exited {done[name]["exit_code"]}')
        argv = commands[name]
        require(argv[2].endswith('/tools/r40/frozen_residency_tests.py') and
                argv[3:] == ['--point', point['id'], '--steps', '2', '--source-tree', TREE],
                f'{name}: graph command changed')
        log = need(Path(directory) / f'{name}.log').read_text()
        rows = re.findall(r'^PASS (\S+) (resident|per-kernel) step=([01]): '
                          r'(\d+) kernels, actual tensor bytes=(\{[^\n]+\})$', log, re.M)
        require([(p, s) for _, p, s, _, _ in rows] ==
                [('resident', '0'), ('resident', '1'), ('per-kernel', '0'), ('per-kernel', '1')],
                f'{name}: missing/duplicate two-policy two-step observation')
        for pid, policy, _, count, raw in rows:
            require(pid == point['id'] and int(count) >= 3, f'{name}: not the frozen >=3-kernel graph')
            counts = ast.literal_eval(raw)
            require(set(counts) == {'h2d', 'd2h'} and all(type(v) is int and v >= 0 for v in counts.values()),
                    f'{name}: invalid transfer ledger')
            if policy == 'resident':
                require(counts == dict(h2d=0, d2h=0), f'{name}: resident graph transferred tensors')
                totals['resident_calls'] += 1
            else:
                require(counts['h2d'] == counts['d2h'] > 0, f'{name}: staging positive control missing')
                totals['staging_calls'] += 1
                totals['staging_bytes'] += counts['h2d']
            totals['kernels'].append(int(count))
        require(log.count('r40-frozen-residency-tests PASS (4/4)') == 1 and
                not re.search(r'Traceback|^FAIL', log, re.M), f'{name}: did not finish its assertions')
    return [dict(name='all 24 frozen >=3-kernel graphs on the delivery board: resident steady calls move '
                      '0 tensor bytes; per-kernel staging positive control moves real bytes; explicit '
                      'output download observed', ok=True, queue=str(directory),
                 resident_calls=totals['resident_calls'], staging_calls=totals['staging_calls'],
                 staging_bytes=totals['staging_bytes'], kernels_min=min(totals['kernels']),
                 kernels_max=max(totals['kernels']))]


def model_report(path):
    """(report, digest record) from report.json or report.json.gz, optionally beside a manifest."""
    raw = need(path).read_bytes()
    if Path(path).suffix == '.gz':
        compressed, raw = raw, gzip.decompress(raw)
    manifest = Path(str(path).replace('.json.gz', '').replace('.json', '') + '_manifest.json')
    digest = dict(raw_sha256=sha(raw), raw_bytes=len(raw), path=str(path))
    if manifest.exists():
        m = json.loads(manifest.read_text())
        require(m['raw_sha256'] == digest['raw_sha256'] and m['raw_bytes'] == len(raw), 'model report changed')
        digest['source'] = m.get('source')
    return json.loads(raw), digest


def model_checks(report_path, queue, snapshot):
    import residency_acceptance
    state, commands = functional_queue(queue, snapshot)
    stage, argv = model_stage(queue, state, commands)
    require(stage['exit_code'] == 0, 'board model stage failed')
    require(argv[2].endswith('/tools/r40/native_model.py') and '--bucket' in argv and
            argv[argv.index('--image-sha256') + 1] == IMAGE and
            argv[argv.index('--software-tree-sha256') + 1] == TREE, 'model command changed')
    report, digest = model_report(report_path)
    same_identity(report['identity']['software_tree'], 'model report')
    residency_acceptance.model(report, TREE, IMAGE, dict(backend_args=state['backend_args'], clock_hz=CLOCK_HZ))
    return [dict(name='full frozen 4-layer model (prefill512 + 64 decode) on the delivery board: weights '
                      'uploaded once, zero steady tensor/staging transfers, explicit downloads only, '
                      'drained completion and host visibility', ok=True, report=digest,
                 steps=len(report['steps']), steady_launches=sum(len(s['launches']) for s in report['steps']),
                 initial_weight_upload_bytes=report['initial_weight_upload_bytes'])]


def zero_transfer_checks(board, snapshot, report, sim):
    checks = graph_checks(board, snapshot)
    checks += model_checks(report, board, snapshot)
    states, used, logs = sim_run(sim, ZERO_SIM_STEPS)
    has(logs['residency-controls'], 'r40-residency-tests PASS (6/6)', 'residency-controls')
    has(logs['frozen-attention-tiny'], 'r40-frozen-residency-tests PASS (4/4)', 'frozen-attention-tiny')
    checks.append(dict(name='make r40-residency transfer controls on delivery-tree RTL: resident 0 B, staging '
                            'round trip, artifact identity, offset aliases, unknown policy refused', ok=True,
                       steps=used))
    checks.append(sim_identity(states))
    return checks


# ---------------------------------------------------------------------- main
def default_sims():
    """Preserved delivery-tree simulation runs, oldest first by their recorded start."""
    runs = [d for d in (EVIDENCE / 'R40-02.stress/sim').glob('*') if (d / 'state.json').exists()]
    return sorted(runs, key=lambda d: json.loads((d / 'state.json').read_text())['started_utc'])


def judge(fn, *args):
    try:
        return 'capability-pass', fn(*args), ''
    except Missing as exc:
        return 'not-measured', [dict(name='complete delivery-identity evidence', ok=False, detail=str(exc))], str(exc)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError, SyntaxError) as exc:
        return 'fail', [dict(name='delivery-identity evidence', ok=False,
                             detail=f'{type(exc).__name__}: {exc}')], str(exc)


def emit(results, dry_run, auditor):
    tree = local_tree()
    for gate, (status, checks, detail) in results.items():
        if status == 'capability-pass' and tree != TREE:
            status, detail = 'fail', 'auditor product tree differs from the delivery tree'
            checks = checks + [dict(name='auditor runs on the delivery product tree', ok=False, detail=tree)]
        extra = dict(software_tree=TREE, image_sha256=IMAGE, clock_hz=CLOCK_HZ,
                     auditor_sha256=file_sha(auditor), auditor_tree=tree)
        if dry_run:
            print(json.dumps(dict(gate=gate, status=status, checks=checks, detail=detail, **extra),
                             indent=1, default=str))
        else:
            os.environ['R40_CONTRACT'] = 'v3'
            import evidence
            evidence.record(gate, status, checks, detail=detail, **extra)
        results[gate] = (status, checks, detail)
        print(f'{gate}: {status}' + (f' ({detail})' if status != 'capability-pass' else ''))
    return 0 if all(s == 'capability-pass' for s, _, _ in results.values()) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='print the records; write nothing')
    p.add_argument('--board', type=Path, default=EVIDENCE / 'R40-02.zero-transfer/board_queue',
                   help='functional board queue directory (state.json, controller.py, commands.json, '
                        'graphs24-pointNN.log; a tools/r40/conditional_delivery.py stage keeps its own '
                        'delivery.json under <stage name>/)')
    p.add_argument('--board-snapshot', type=Path, default=None,
                   help='source root (or harness copy) the queue ran; default <board>/snapshot_harness')
    p.add_argument('--model-report', type=Path, default=EVIDENCE / 'R40-08.decode/board_model/model_report.json.gz')
    p.add_argument('--sim', type=Path, nargs='+', default=None,
                   help='v3_sim_chains run directories, oldest first (default: evidence/r40/v3/R40-02.stress/sim/*)')
    a = p.parse_args()
    contract_v3.verify()
    sims = a.sim or default_sims()
    snapshot = a.board_snapshot or a.board / 'snapshot_harness'
    results = {}
    status, checks, why = judge(zero_transfer_checks, a.board, snapshot, a.model_report, sims)
    results['R40-02.zero-transfer'] = (status, checks, why or
        'acceptance 1: frozen chained graphs + full multi-step decode on the delivery board, controls on '
        'delivery-tree RTL; no latency claim')
    status, checks, why = judge(stress_checks, sims)
    results['R40-02.stress'] = (status, checks, why or
        'acceptance 3 on delivery-tree RTL simulation (single-core and 3-core platform); no latency claim')
    return emit(results, a.dry_run, __file__)


if __name__ == '__main__':
    raise SystemExit(main())
