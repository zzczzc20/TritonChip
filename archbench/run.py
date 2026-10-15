"""Run ArchBench points on a platform and append one JSON record per point.

    python -m archbench.run --platform tnca --tier tiny --jobs 8 --out results/tnca_tiny.jsonl
    python -m archbench.run --platform tnca --backend fpga-u50-sim --point model.mlp/tiny/fp16-t16-d64
    python -m archbench.run --platform gpu --tier tiny small large --out results/gpu.jsonl
    python -m archbench.run --list --tier tiny

TNCA points run each in its own subprocess (isolation: a compiler or device
failure in one point cannot poison the next, and simulator time parallelises
across cores). GPU points run one after another in this process: the GPU is
one device and timing must not share it.

Environment for TNCA (see python/tnca/triton_backend/driver.py):
TNCA_BACKEND, TNCA_BACKEND_ARGS. `--backend`/`--backend-args` set them.
CUDA is hidden from TNCA subprocesses, so the CUDA Triton driver stays
inactive and only the tnca driver serves Inductor's kernels.
"""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _one(args):
    """Child process: run a single point, print its record as JSON."""
    from archbench import registry
    p = registry.get(args.point[0])
    if args.platform == 'tnca':
        from archbench.platforms import tnca
        bundle = str(Path(args.bundle_dir) / p.id.replace('/', '__')) if args.bundle_dir else None
        rec = tnca.run(p, profile=args.profile, opt_level=args.opt_level, seed=args.seed,
                       counters=args.counters, warmup=args.warmup, timed=args.timed, bundle=bundle)
    else:
        from archbench.platforms import gpu
        rec = gpu.run(p, variant=args.variant, seed=args.seed, reps=args.reps)
    print('ARCHBENCH-RECORD ' + json.dumps(rec, default=str), flush=True)


def _platform(args):
    """The platform description of a record the parent writes itself (a
    worker that died before reporting), matching what the runners record."""
    if args.platform == 'tnca':
        return dict(kind='tnca', name=f'tnca:{args.profile}@{args.backend}', profile=args.profile,
                    backend=args.backend, opt_level=args.opt_level)
    return dict(kind='gpu', name='gpu:gb10', spec='gb10', variant=args.variant)


def _evidence(args):
    if args.platform == 'gpu':
        return 'measured-gpu'
    return 'measured-board' if args.backend in ('fpga-u50', 'fpga-u280', 'fpga-u280-multicore') else 'measured-rtl'


def _child_env(args):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'python'), env.get('PYTHONPATH', '')])
    if args.platform == 'tnca':
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['TNCA_TRITON'] = '1'
        env['TNCA_BACKEND'] = args.backend
        # A simulated launch runs ~500 target cycles per wall second, so a
        # model's larger kernels take hours of host time per launch.
        env.setdefault('TNCA_HOST_TIMEOUT', str(int(args.timeout)))
        env.setdefault('TNCA_DEADLINE', '200000000')
        if args.backend_args:
            env['TNCA_BACKEND_ARGS'] = args.backend_args
        if args.compile_only:
            env['TNCA_LAUNCH'] = 'skip'
        # R39: the standard path has tensor descriptors on (the default in
        # tnca.inductor.configure); this is the switch that turns them off.
        if args.no_tensor_descriptors:
            env['TNCA_TENSOR_DESCRIPTORS'] = '0'
    # Ablation knobs: KEY=VALUE set in every child's environment, and
    # recorded with every point (platform.knobs) so two runs that differ in
    # one are never mistaken for the same configuration.
    for kv in args.set or []:
        k, _, v = kv.partition('=')
        env[k] = v
    return env


def _spawn(pid, args, logdir):
    cmd = [sys.executable, '-m', 'archbench.run', '--child', '--platform', args.platform,
           '--point', pid, '--profile', args.profile, '--opt-level', str(args.opt_level),
           '--seed', str(args.seed), '--variant', args.variant]
    if args.counters:
        cmd.append('--counters')
    cmd += ['--warmup', str(args.warmup), '--timed', str(args.timed)]
    if args.bundle_dir:
        cmd += ['--bundle-dir', str(Path(args.bundle_dir).resolve())]
    if args.reps:
        cmd += ['--reps', str(args.reps)]
    t = time.monotonic()
    log = logdir / (pid.replace('/', '__') + '.log')
    with open(log, 'w') as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=_child_env(args), stdout=subprocess.PIPE,
                              stderr=fh, text=True, timeout=args.timeout)
    rec = None
    for line in proc.stdout.splitlines():
        if line.startswith('ARCHBENCH-RECORD '):
            rec = json.loads(line[len('ARCHBENCH-RECORD '):])
    if rec is None:
        from archbench import registry
        from archbench.platforms.common import base_record
        rec = base_record(registry.get(pid), _platform(args), _evidence(args))
        rec.update(status='runtime-fault',
                   problems=[f'child exited {proc.returncode} without a record; see {log}'])
    rec['host'] = dict(rec['host'] or {}, process_s=round(time.monotonic() - t, 1), log=str(log))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', choices=('tnca', 'gpu'), default='tnca')
    ap.add_argument('--tier', nargs='+', default=None)
    ap.add_argument('--kind', choices=('op', 'model'))
    ap.add_argument('--track', choices=('llm', 'general'))
    ap.add_argument('--family', nargs='+')
    ap.add_argument('--point', nargs='+', help='explicit point ids')
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--opt-level', type=int, default=2)
    ap.add_argument('--backend', default=os.environ.get('TNCA_BACKEND', 'sim-file'))
    ap.add_argument('--backend-args', default=os.environ.get('TNCA_BACKEND_ARGS'))
    ap.add_argument('--counters', action='store_true')
    ap.add_argument('--no-tensor-descriptors', action='store_true',
                    help='TNCA: Inductor without tensor descriptors (the standard path has them on)')
    ap.add_argument('--set', action='append', metavar='KEY=VALUE',
                    help='an environment knob for every point (repeatable), recorded with each point')
    ap.add_argument('--compile-only', action='store_true',
                    help='TNCA: compile every kernel, launch none (TNCA_LAUNCH=skip)')
    ap.add_argument('--variant', default='inductor-cudagraph')
    ap.add_argument('--reps', type=int)
    ap.add_argument('--warmup', type=int, default=0,
                    help='TNCA, protocol 2: untimed calls after the cold first call')
    ap.add_argument('--timed', type=int, default=0,
                    help='TNCA, protocol 2: timed calls, one sample each (0: the first call\'s device '
                         'cycles are the one, deterministic, sample -- the simulator)')
    ap.add_argument('--bundle-dir', type=Path,
                    help='TNCA, protocol 2: write each point\'s provenance bundle under this directory')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=6 * 3600)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:
        return _one(args)

    from archbench import registry
    if args.point:
        pts = [registry.get(p) for p in args.point]
    else:
        pts = registry.points(tier=args.tier, kind=args.kind, track=args.track, family=args.family)
    if args.list:
        for p in pts:
            d = p.demand
            print(f'{p.id:60s} macs={d.macs:.3g} bytes={d.read_bytes + d.write_bytes:.3g} '
                  f'track={",".join(p.track)}')
        print(f'{len(pts)} point(s)')
        return 0
    if not pts:
        print('no points selected')
        return 2

    out = args.out or ROOT / 'sim' / 'build' / 'archbench' / f'{args.platform}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    logdir = out.parent / (out.stem + '_logs')
    logdir.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs if args.platform == 'tnca' else 1
    t0 = time.monotonic()
    n_pass = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool, open(out, 'a') as fh:
        futs = {pool.submit(_spawn, p.id, args, logdir): p for p in pts}
        for f in cf.as_completed(futs):
            p = futs[f]
            try:
                rec = f.result()
            except Exception as e:                           # noqa: BLE001
                from archbench.platforms.common import base_record
                rec = base_record(p, _platform(args), _evidence(args))
                rec.update(status='timeout' if isinstance(e, subprocess.TimeoutExpired) else 'runtime-fault',
                           problems=[f'{type(e).__name__}: {e}'])
            fh.write(json.dumps(rec, default=str) + '\n')
            fh.flush()
            # Compile-only: a point passes the sweep when every kernel built.
            n_pass += rec['status'] == 'pass' or (args.compile_only and rec['status'] == 'not-measured')
            t = rec.get('time') or {}
            eta = rec.get('eta')
            print(f'{rec["status"]:18s} {p.id:58s} time={t.get("device")}{t.get("unit", "")} '
                  f'eta={eta if eta is None else f"{eta:.4g}"} '
                  f'err={(rec.get("correctness") or {}).get("max_rel_err")} '
                  f'[{(rec.get("host") or {}).get("process_s")}s]'
                  + ''.join(f'\n      {q[:300]}' for q in rec['problems'][:3]), flush=True)
    what = 'compile' if args.compile_only else 'pass'
    print(f'{n_pass}/{len(pts)} {what} in {time.monotonic() - t0:.0f}s -> {out}')
    return 0 if n_pass == len(pts) else 1


if __name__ == '__main__':
    sys.exit(main())
