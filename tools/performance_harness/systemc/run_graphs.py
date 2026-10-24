#!/usr/bin/env python3
"""Run ArchBench points end to end through the native SystemC model.

    run_graphs.py --source-root SRC --binary BUILD/tnca-chip --out FRESH_DIR \
        --suite SRC/spec/r40/suites/existing.json [--tier tiny] [--point ID ...] \
        [--knobs-from A_fix2/runs] [--backend systemc-tlm-grid|systemc-tlm|sim-file] [--jobs 4]

The standard path, unchanged: torch.compile -> Inductor -> Triton -> TNCA
artifact -> Session, launching into the model through a runtime backend
registered from outside the product tree (tlm_backend.py). Each point runs
in a fresh child process (one Session, one long-lived model process per
graph) through archbench.platforms.tnca.run, which compiles, launches,
compares with the PyTorch reference at the point's own tolerance and counts
host fallbacks. Records are labelled `model-prediction-uncalibrated`; with
--backend sim-file the same child runs the RTL simulator instead, for the
wall-time comparison only.

The delivered default configuration (B40-N) is reproduced from the compile
preflight's result.json knobs for each point when --knobs-from is given.

--dataset BOARD_DATASET.json[.gz] --split calibration --tree fe1b runs every
distinct (point, knobs) configuration of that software tree's board records
in the named split(s), with exactly the knobs the board record states and
nothing else set (a knob the board left unset keeps its default, e.g. the
image's three memory ports and eight outstanding lines). The source root must
be a checkout of that tree. predictions.py turns the results into
calibrate.py prediction rows.

Each results row records the knobs actually in effect: the child's own
record (platform.knobs, archbench's definition of a knob) is compared with
the environment the parent gave it, and a mismatch refuses the row
(status knob-mismatch). `knobs` holds the configuration knobs a board record
can match; `model_knobs` the model-harness keys (TNCA_TLM_*, DRAM path/map,
RTL cache); `requested_knobs` the suite/dataset/preflight configuration
before --env/--active-cores overrides.
"""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
from prediction_domain import timing_status, VALID_STATUSES

HARNESS = dict(CUDA_VISIBLE_DEVICES='', TNCA_PROFILE='core-vrf-sync-v5', TNCA_OPT_LEVEL='2', TNCA_TRITON='1',
               TNCA_KERNEL_DUMP='1', PYTHONDONTWRITEBYTECODE='1')
RUNTIME = dict(TNCA_TRANSFER_POLICY='per-kernel', TNCA_ACTIVE_CORES='1', TNCA_RTL_MEM_PORTS='1',
               TNCA_RTL_OUTSTANDING='1', TNCA_SUBMIT='single',
               # tools/r40/v3_campaign.py's watchdog, declared identically for every configuration
               TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='28800')
B40N = dict(TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16', TNCA_REDUCTION_ROWS='1', TNCA_BANK_ALLOC='0',
            TNCA_WINDOW_REUSE='0', TNCA_SDPA='decomposed', TNCA_PIPELINE='2', TNCA_TUNE='0')
# What the model process is and where it writes: no board record carries
# these, so they are kept apart from the configuration a prediction matches.
MODEL_ONLY_PREFIXES = ('TNCA_TLM_',)
MODEL_ONLY = frozenset({'TNCA_CACHE', 'TNCA_DRAM_PATH', 'TNCA_DRAM_MAP', 'TNCA_RAMULATOR2_CONFIG'})


def model_only(name):
    return name.startswith(MODEL_ONLY_PREFIXES) or name in MODEL_ONLY


def knobs_in_effect(rec, env):
    """(configuration knobs, model knobs, None) or (None, None, why not).

    The record's platform.knobs is what the child actually ran with; it must
    equal the TNCA_* environment the parent set, less the keys the child's
    archbench does not count as knobs (reported in the record).
    """
    recorded = (rec.get('platform') or {}).get('knobs')
    excluded = rec.get('harness_not_knobs')
    if not isinstance(recorded, dict) or not isinstance(excluded, list):
        return None, None, 'record lacks platform.knobs or the harness knob exclusions'
    effective = {k: v for k, v in env.items() if k.startswith('TNCA_') and k not in set(excluded)}
    if recorded != effective:
        wrong = sorted(k for k in set(recorded) | set(effective) if recorded.get(k) != effective.get(k))
        return None, None, ('record platform.knobs differ from the run environment: ' +
                            '; '.join(f'{k} record={recorded.get(k)!r} run={effective.get(k)!r}' for k in wrong))
    return ({k: v for k, v in sorted(recorded.items()) if not model_only(k)},
            {k: v for k, v in sorted(recorded.items()) if model_only(k)}, None)


def bounded_child(cmd, *, cwd, env, stderr, timeout):
    """A graph timeout must also terminate its compiler/model descendants."""
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=stderr, text=True, start_new_session=True)
    try:
        try:
            stdout, _ = process.communicate(timeout=timeout)
            return stdout, process.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, _ = process.communicate()
            return stdout, 'timeout'
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def complete_run(row):
    return row.get('status') == 'pass' and row.get('returncode') == 0 and not row.get('fallbacks')


def child(a):
    root = Path(a.source_root).resolve()
    sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm'), str(HERE)]
    from archbench import registry
    from archbench.platforms import tnca
    import tlm_backend
    if a.points_module:
        # Extra points defined outside the frozen suites (registered by import).
        import importlib.util
        spec = importlib.util.spec_from_file_location('run_graphs_extra_points', a.points_module)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    tlm_backend.register()
    for name in tlm_backend.NAMES:
        tnca.EVIDENCE[name] = tlm_backend.EVIDENCE
    point = registry.get(a.point[0])
    if os.environ.get('TNCA_TLM_UNTIMED_COLD') == '1':
        # The cold call (compile, cold setup, first run) runs untimed; the
        # model is timed from the first steady call, the measured one.
        original = tnca._steady

        def timed_steady(*args, **kwargs):
            for inst in tlm_backend.backend_class().instances:
                if inst.alive():
                    # Counters at the switch: subtracting them from the final
                    # snapshot leaves the measured (steady) calls alone.
                    (inst.dir/'counters_untimed_cold.json').write_text(
                        json.dumps(inst.counters_snapshot(), indent=1, sort_keys=True) + '\n')
                    inst.set_timing(True)
            return original(*args, **kwargs)
        tnca._steady = timed_steady
    t = time.monotonic()
    # Native storage: the cold call may clone device literals (archbench's
    # cold_setup), so the graph is measured on the steady call(s) after it,
    # exactly as the board campaign does.
    rec = tnca.run(point, profile=os.environ['TNCA_PROFILE'], opt_level=2, seed=0, warmup=a.warmup, timed=a.timed)
    rec['child_wall_s'] = round(time.monotonic() - t, 3)
    # The parent verifies platform.knobs against what it set; it needs this
    # archbench's own definition of which TNCA_* variables are not knobs.
    rec['harness_not_knobs'] = sorted(tnca.NOT_KNOBS)
    instances = []
    for inst in tlm_backend.backend_class().instances:
        info = dict(identity=getattr(inst, 'identity', None), model_dir=str(inst.dir), launches=len(inst.launches),
                    model_cycles=sum(l.get('cycles', 0) for l in inst.launches),
                    retired=sum(l.get('retired', 0) for l in inst.launches),
                    host_seconds={k: round(v, 6) for k, v in inst.host_seconds.items()},
                    statuses=sorted({l.get('status') for l in inst.launches}))
        if inst.alive():
            try:
                counters = inst.counters_snapshot()
                (inst.dir/'counters.json').write_text(json.dumps(counters, indent=1, sort_keys=True) + '\n')
                info['counters_file'] = str(inst.dir/'counters.json')
                flat = counters.get('counters', counters) if isinstance(counters, dict) else {}
                # Slow DRAM path: per-run totals of the per-command fixed point
                # (per command: dram_fixed_point.jsonl in the model directory).
                info['dram'] = {k: v for k, v in flat.items() if k.startswith('dram_')}
                info['timing_status'] = timing_status(flat, slow=inst.identity['dram_path'] == 'slow')
            except Exception as e:                           # noqa: BLE001
                info['counters_error'] = repr(e)
                info['timing_status'] = 'inconclusive-counter-read'
        (inst.dir/'launches.json').write_text(json.dumps(inst.launches, indent=1) + '\n')
        instances.append(info)
    rec['systemc_model'] = instances
    statuses = {i.get('timing_status', 'inconclusive-counter-read') for i in instances}
    rec['model_timing_status'] = ('inconclusive-fixed-point' if not statuses or statuses-VALID_STATUSES
                                  else 'uncalibrated-fixed-point-checked' if 'uncalibrated-fixed-point-checked' in statuses
                                  else 'uncalibrated')
    # Model graph makespan of the measured call: first launch start to last
    # launch end on the model clock, host gaps included (server host-time
    # model). The measured call is the last `launches` launches of the session.
    t = rec.get('time') or {}
    insts = tlm_backend.backend_class().instances
    n = t.get('launches')
    measured = 'steady' if a.timed else 'cold'
    if not a.timed and a.graph_launches:
        # No steady call: the graph is the cold call's last N launches (the
        # ones the generated wrapper makes); what precedes them is the native
        # path's cold setup (device-literal copies made while compiling).
        n = a.graph_launches
    if insts and n:
        ls = insts[-1].launches[-n:]
        per = (t.get('per_launch') or [])[-n:]
        if ls and all(l.get('status') == 0 for l in ls):
            rec['model_graph'] = dict(graph_cycles=ls[-1]['end_cycle'] - ls[0]['start_cycle'],
                                      device_cycles=sum(l['cycles'] for l in ls), measured_call=measured,
                                      launches=[dict(kernel=p.get('kernel'), cycles=l['cycles'])
                                                for p, l in zip(per or [{}] * len(ls), ls)])
    print('ARCHBENCH-RECORD ' + json.dumps(rec, default=str), flush=True)


def knobs_for(point, knobs_from):
    if not knobs_from:
        return dict(B40N)
    run = Path(knobs_from)/(point.replace('/', '__') + '__b40n')
    if not (run/'result.json').exists():
        return dict(B40N)
    knobs = json.loads((run/'result.json').read_text())['knobs']
    return {k: v for k, v in knobs.items() if v is not None}


def dataset_items(path, splits, tree, only_points):
    raw = Path(path).read_bytes()
    doc = json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)
    seen = {}
    for r in doc['rows']:
        if r['split'] not in splits or not (r.get('software_tree') or '').startswith(tree):
            continue
        if only_points and r['point'] not in only_points:
            continue
        knobs = {k: v for k, v in sorted((r.get('knobs') or {}).items()) if k != 'TNCA_OPERATING_CLOCK_EVIDENCE'}
        tag = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[:10]
        cost = sum(l.get('cycles') or 0 for l in r.get('launches') or [])
        key = f"{r['point']}@{tag}"
        seen[key] = (r['point'], knobs, max(cost, seen.get(key, (0, 0, 0))[2]), len(r.get('launches') or []))
    # Longest first, so a parallel campaign is not left waiting on one point.
    order = sorted(seen.items(), key=lambda kv: -kv[1][2])
    return [(key, point, knobs) for key, (point, knobs, _c, _n) in order], {k: v[3] for k, v in order}


def summarize(rec, wall):
    t = rec.get('time') or {}
    model = rec.get('systemc_model') or []
    corr = rec.get('correctness') or {}
    return dict(point=rec.get('point_id'), status=rec.get('status'), problems=[p[:400] for p in rec.get('problems') or []],
                fallbacks=rec.get('fallbacks'), launches=t.get('launches'), device_cycles=t.get('device'),
                first_call_device_cycles=(t.get('first_call') or {}).get('device_cycles'),
                steady_request_s=((rec.get('timings') or {}).get('request-latency') or {}).get('median'),
                timing=(rec.get('provenance') or {}).get('launch_map', {}).get('timing'),
                max_rel_err=corr.get('max_rel_err'), tol=corr.get('tol'), first_call_wall_s=(rec.get('host') or {}).get('wall_s'),
                child_wall_s=rec.get('child_wall_s'), process_wall_s=round(wall, 2),
                model_launch_host_s=sum(m['host_seconds'].get('launch', 0) for m in model) if model else None,
                model_transfer_host_s=sum(m['host_seconds'].get('upload', 0) + m['host_seconds'].get('download', 0)
                                          for m in model) if model else None,
                model_retired=sum(m['retired'] for m in model) if model else None,
                model_dirs=[m['model_dir'] for m in model], evidence=rec.get('evidence_level'),
                model_graph=rec.get('model_graph'),
                model_timing_status=rec.get('model_timing_status'),
                dram={k: sum(m.get('dram', {}).get(k, 0) for m in model) for k in
                      sorted({k for m in model for k in m.get('dram', {})})} if model else None,
                # The only problem is the cold call's extra setup launches
                # (outputs matched the reference: archbench lists a mismatch
                # as a second problem).
                cold_setup_only=(rec.get('status') == 'not-measured' and len(rec.get('problems') or []) == 1
                                 and 'the generated wrapper makes' in rec['problems'][0]))


def parent(a):
    root = Path(a.source_root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    # Work items: (key, point, knobs). A suite or --point names the delivered
    # configuration; --preflight-runs takes every (point, configuration) of a
    # compile preflight with that run's own knobs (e.g. the A/B experiments).
    items = [] if a.dataset else [(p, p, knobs_for(p, a.knobs_from)) for p in (a.point or [])]
    board_launches = {}
    if a.dataset:
        found, board_launches = dataset_items(a.dataset, set(a.split or ['calibration']), a.tree or '', set(a.point or []))
        items += [i for i in found if not a.key or i[0] in a.key]
    if a.suite:
        suite = json.loads(Path(a.suite).read_text())
        items += [(p['id'], p['id'], knobs_for(p['id'], a.knobs_from)) for p in suite['points']
                  if not a.tier or p['tier'] in a.tier]
    if a.preflight_runs:
        for run in sorted(Path(a.preflight_runs).iterdir()):
            if not (run/'result.json').exists():
                continue
            parts = run.name.split('__')
            point, label = '/'.join(parts[:3]), '__'.join(parts[3:])
            if a.tier and parts[1] not in a.tier:
                continue
            knobs = {k: v for k, v in json.loads((run/'result.json').read_text())['knobs'].items() if v is not None}
            items.append((f'{point}@{label}', point, knobs))
    points = [key for key, _p, _k in items]
    by_key = {key: (point, knobs) for key, point, knobs in items}
    results = {}

    def one(key):
        point, knobs = by_key[key]
        run_dir = out/'runs'/key.replace('/', '__').replace('@', '__')
        for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump', 'model'):
            (run_dir/d).mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))
               or k in ('TNCA_CACHE', 'TNCA_DRAM_PATH', 'TNCA_RAMULATOR2_CONFIG')}  # RTL simulator cache; DRAM path
        env.update(HARNESS)
        if a.dataset:   # the board record's knobs and nothing else
            env.update(TNCA_DEADLINE=RUNTIME['TNCA_DEADLINE'], TNCA_HOST_TIMEOUT=RUNTIME['TNCA_HOST_TIMEOUT'])
        else:
            env.update(RUNTIME)
        env.update(knobs)
        env.update(dict(kv.split('=', 1) for kv in a.env or []))
        env.update(TNCA_BACKEND=a.backend, TNCA_TORCH_DEVICE=a.torch_device,
                   PYTHONPATH=os.pathsep.join(str(x) for x in (root, root/'python', root/'compiler/python',
                                                              root/'tools/asm', HERE)),
                   TRITON_CACHE_DIR=str(run_dir/'triton_cache'), TRITON_HOME=str(run_dir/'triton_home'),
                   TORCHINDUCTOR_CACHE_DIR=str(run_dir/'inductor_cache'), TMPDIR=str(run_dir/'tmp'),
                   TNCA_DUMP_DIR=str(run_dir/'dump'), TNCA_TLM_BINARY=str(Path(a.binary).resolve()),
                   TNCA_TLM_OUT=str(run_dir/'model'), TNCA_TLM_MONITOR=a.monitor)
        if a.backend in ('sim-file', 'systemc-tlm'):
            env.pop('TNCA_ACTIVE_CORES', None)   # per-program backends have no physical grid
        elif a.active_cores and not a.dataset:
            env['TNCA_ACTIVE_CORES'] = a.active_cores
        if a.torch_extensions_dir:
            env['TORCH_EXTENSIONS_DIR'] = a.torch_extensions_dir
        if a.config:
            env['TNCA_TLM_CONFIG'] = str(Path(a.config).resolve())
        if a.memory_bytes:
            env['TNCA_TLM_MEMORY_BYTES'] = a.memory_bytes
        if a.dram_path:
            env['TNCA_DRAM_PATH'] = a.dram_path
        (run_dir/'env.json').write_text(json.dumps({k: v for k, v in env.items()
                                                    if k.startswith(('TNCA_', 'TRITON', 'TORCH'))}, indent=1))
        timed = a.timed
        if a.timed_auto and key in board_launches:
            # One launch: its graph makespan is the launch, and the cold call
            # (archbench's deterministic sample) measures it; a multi-launch
            # graph is measured on a steady call, as on the board.
            timed = 0 if board_launches[key] == 1 else 1
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--child', '--source-root', str(root),
               '--point', point, '--warmup', str(a.warmup), '--timed', str(timed)]
        if a.timed_auto and key in board_launches and timed == 0:
            cmd += ['--graph-launches', str(board_launches[key])]
        if a.points_module:
            cmd += ['--points-module', str(Path(a.points_module).resolve())]
        if a.untimed_cold and timed > 0:
            env['TNCA_TLM_UNTIMED_COLD'] = '1'
        t = time.monotonic()
        with (run_dir/'child.log').open('w') as log:
            stdout, code = bounded_child(cmd, cwd=run_dir, env=env, stderr=log, timeout=a.timeout)
        wall = time.monotonic() - t
        (run_dir/'stdout.log').write_text(stdout)
        rec = None
        for line in stdout.splitlines():
            if line.startswith('ARCHBENCH-RECORD '):
                rec = json.loads(line[len('ARCHBENCH-RECORD '):])
        config = model_knobs = None
        if rec is None:
            tail = (run_dir/'child.log').read_text(errors='replace')[-1500:]
            row = dict(point=point, status='harness-error', returncode=code, process_wall_s=round(wall, 2), tail=tail)
        else:
            (run_dir/'record.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
            row = summarize(rec, wall)
            row['returncode'] = code
            row['record'] = str(run_dir/'record.json')
            config, model_knobs, problem = knobs_in_effect(rec, env)
            if problem:
                # Never label a run with knobs it did not run: refuse the row.
                row.update(status='knob-mismatch', knob_problem=problem)
        row.update(key=key, requested_knobs=knobs, knobs=config, model_knobs=model_knobs,
                   knobs_verified=config is not None)
        results[key] = row
        print(json.dumps({k: row.get(k) for k in ('key', 'status', 'launches', 'device_cycles', 'max_rel_err',
                                                  'process_wall_s', 'model_launch_host_s')}), flush=True)
        (out/'results.json').write_text(json.dumps(dict(
            schema='tnca-systemc-graph-runs/1', backend=a.backend, binary=str(Path(a.binary).resolve()),
            source_root=str(root), points=[results[p] for p in points if p in results]), indent=1) + '\n')
        return row

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(one, points))
    complete = sum(1 for r in results.values() if complete_run(r))
    print('DONE', complete, '/', len(points), flush=True)
    return 0 if complete == len(points) and points else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--point', action='append')
    ap.add_argument('--suite')
    ap.add_argument('--tier', action='append')
    ap.add_argument('--binary')
    ap.add_argument('--config')
    ap.add_argument('--out')
    ap.add_argument('--knobs-from')
    ap.add_argument('--preflight-runs', help='run every (point, configuration) of a compile preflight runs/ directory')
    ap.add_argument('--backend', default='systemc-tlm-grid')
    ap.add_argument('--torch-device', default='native')
    ap.add_argument('--torch-extensions-dir')
    ap.add_argument('--monitor', default='aggregate')
    ap.add_argument('--memory-bytes')
    ap.add_argument('--active-cores', help='override the delivered TNCA_ACTIVE_CORES=1 core count (physical-grid backends)')
    ap.add_argument('--dataset', help='board timing dataset: run its configurations (see above)')
    ap.add_argument('--split', action='append', help='dataset split(s) to run (default calibration)')
    ap.add_argument('--key', action='append', help='with --dataset: only these POINT@KNOBHASH configurations')
    ap.add_argument('--tree', help='software tree prefix of the dataset records to run (the source root\'s tree)')
    ap.add_argument('--timed-auto', action='store_true',
                    help='with --dataset: no steady call for a one-launch board record, one otherwise')
    ap.add_argument('--untimed-cold', action='store_true',
                    help='run the cold call untimed when a steady call is measured (timeline memory only)')
    ap.add_argument('--env', action='append', help='KEY=VALUE set for every run after the knobs (e.g. TNCA_RTL_MEM_PORTS=3)')
    ap.add_argument('--dram-path', choices=('fast', 'slow'), help='TNCA_DRAM_PATH for the model (overrides dram_path)')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--timed', type=int, default=1)
    ap.add_argument('--graph-launches', type=int, help=argparse.SUPPRESS)
    ap.add_argument('--points-module', help='Python file registering extra ArchBench points (outside the frozen suites)')
    a = ap.parse_args()
    if a.child:
        child(a)
        return 0
    else:
        return parent(a)


if __name__ == '__main__':
    raise SystemExit(main())
