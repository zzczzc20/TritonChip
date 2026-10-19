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
"""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

HARNESS = dict(CUDA_VISIBLE_DEVICES='', TNCA_PROFILE='core-vrf-sync-v5', TNCA_OPT_LEVEL='2', TNCA_TRITON='1',
               TNCA_KERNEL_DUMP='1', PYTHONDONTWRITEBYTECODE='1')
RUNTIME = dict(TNCA_TRANSFER_POLICY='per-kernel', TNCA_ACTIVE_CORES='1', TNCA_RTL_MEM_PORTS='1',
               TNCA_RTL_OUTSTANDING='1', TNCA_SUBMIT='single',
               # tools/r40/v3_campaign.py's watchdog, declared identically for every configuration
               TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='28800')
B40N = dict(TNCA_MAX_BLOCK='16', TNCA_GEMM_TILE='16,16,16', TNCA_REDUCTION_ROWS='1', TNCA_BANK_ALLOC='0',
            TNCA_WINDOW_REUSE='0', TNCA_SDPA='decomposed', TNCA_PIPELINE='2', TNCA_TUNE='0')


def child(a):
    root = Path(a.source_root).resolve()
    sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm'), str(HERE)]
    from archbench import registry
    from archbench.platforms import tnca
    import tlm_backend
    tlm_backend.register()
    for name in tlm_backend.NAMES:
        tnca.EVIDENCE[name] = tlm_backend.EVIDENCE
    point = registry.get(a.point[0])
    t = time.monotonic()
    # Native storage: the cold call may clone device literals (archbench's
    # cold_setup), so the graph is measured on the steady call(s) after it,
    # exactly as the board campaign does.
    rec = tnca.run(point, profile=os.environ['TNCA_PROFILE'], opt_level=2, seed=0, warmup=a.warmup, timed=a.timed)
    rec['child_wall_s'] = round(time.monotonic() - t, 3)
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
            except Exception as e:                           # noqa: BLE001
                info['counters_error'] = repr(e)
        (inst.dir/'launches.json').write_text(json.dumps(inst.launches, indent=1) + '\n')
        instances.append(info)
    rec['systemc_model'] = instances
    print('ARCHBENCH-RECORD ' + json.dumps(rec, default=str), flush=True)


def knobs_for(point, knobs_from):
    if not knobs_from:
        return dict(B40N)
    run = Path(knobs_from)/(point.replace('/', '__') + '__b40n')
    if not (run/'result.json').exists():
        return dict(B40N)
    knobs = json.loads((run/'result.json').read_text())['knobs']
    return {k: v for k, v in knobs.items() if v is not None}


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
                model_dirs=[m['model_dir'] for m in model], evidence=rec.get('evidence_level'))


def parent(a):
    root = Path(a.source_root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    # Work items: (key, point, knobs). A suite or --point names the delivered
    # configuration; --preflight-runs takes every (point, configuration) of a
    # compile preflight with that run's own knobs (e.g. the A/B experiments).
    items = [(p, p, knobs_for(p, a.knobs_from)) for p in (a.point or [])]
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
               or k == 'TNCA_CACHE'}  # where `make device` / device.prepare put the RTL simulator
        env.update(HARNESS)
        env.update(RUNTIME)
        env.update(knobs)
        env.update(TNCA_BACKEND=a.backend, TNCA_TORCH_DEVICE=a.torch_device,
                   PYTHONPATH=os.pathsep.join(str(x) for x in (root, root/'python', root/'compiler/python',
                                                              root/'tools/asm', HERE)),
                   TRITON_CACHE_DIR=str(run_dir/'triton_cache'), TRITON_HOME=str(run_dir/'triton_home'),
                   TORCHINDUCTOR_CACHE_DIR=str(run_dir/'inductor_cache'), TMPDIR=str(run_dir/'tmp'),
                   TNCA_DUMP_DIR=str(run_dir/'dump'), TNCA_TLM_BINARY=str(Path(a.binary).resolve()),
                   TNCA_TLM_OUT=str(run_dir/'model'), TNCA_TLM_MONITOR=a.monitor)
        if a.backend in ('sim-file', 'systemc-tlm'):
            env.pop('TNCA_ACTIVE_CORES', None)   # per-program backends have no physical grid
        elif a.active_cores:
            env['TNCA_ACTIVE_CORES'] = a.active_cores
        if a.torch_extensions_dir:
            env['TORCH_EXTENSIONS_DIR'] = a.torch_extensions_dir
        if a.config:
            env['TNCA_TLM_CONFIG'] = str(Path(a.config).resolve())
        if a.memory_bytes:
            env['TNCA_TLM_MEMORY_BYTES'] = a.memory_bytes
        (run_dir/'env.json').write_text(json.dumps({k: v for k, v in env.items()
                                                    if k.startswith(('TNCA_', 'TRITON', 'TORCH'))}, indent=1))
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--child', '--source-root', str(root),
               '--point', point, '--warmup', str(a.warmup), '--timed', str(a.timed)]
        t = time.monotonic()
        with (run_dir/'child.log').open('w') as log:
            try:
                proc = subprocess.run(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=log,
                                      text=True, timeout=a.timeout)
                stdout, code = proc.stdout, proc.returncode
            except subprocess.TimeoutExpired as e:
                stdout, code = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), 'timeout'
        wall = time.monotonic() - t
        (run_dir/'stdout.log').write_text(stdout)
        rec = None
        for line in stdout.splitlines():
            if line.startswith('ARCHBENCH-RECORD '):
                rec = json.loads(line[len('ARCHBENCH-RECORD '):])
        if rec is None:
            tail = (run_dir/'child.log').read_text(errors='replace')[-1500:]
            row = dict(point=point, status='harness-error', returncode=code, process_wall_s=round(wall, 2), tail=tail)
        else:
            (run_dir/'record.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
            row = summarize(rec, wall)
            row['returncode'] = code
        row.update(key=key, knobs=knobs)
        results[key] = row
        print(json.dumps({k: row.get(k) for k in ('key', 'status', 'launches', 'device_cycles', 'max_rel_err',
                                                  'process_wall_s', 'model_launch_host_s')}), flush=True)
        (out/'results.json').write_text(json.dumps(dict(
            schema='tnca-systemc-graph-runs/1', backend=a.backend, binary=str(Path(a.binary).resolve()),
            source_root=str(root), points=[results[p] for p in points if p in results]), indent=1) + '\n')
        return row

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(one, points))
    print('DONE', sum(1 for r in results.values() if r.get('status') == 'pass'), '/', len(points), flush=True)


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
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--timed', type=int, default=1)
    a = ap.parse_args()
    if a.child:
        child(a)
    else:
        parent(a)


if __name__ == '__main__':
    main()
