#!/usr/bin/env python3
"""Record core traces from the RTL: one ArchBench point per child, one trace per point.

    record_rtl.py --python TORCH_PYTHON --out RUN_DIR \
        (--suite spec/r40/suites/existing.json --tier tiny | --point ID ...) [--jobs N]

The bench is the product's own multicore device (tb/device/tnca_multicore_device.v
through python/tnca/backends/multicore_file.py's file set) elaborated once
into RUN_DIR/bench/device.vvp with tnca_core_trace.v as a second top. Every
point then runs the standard path (archbench.run --platform tnca --backend
sim-multicore) with that bench, one active core, and TNCA_DEVICE_PLUSARGS
carrying +core_trace=RUN_DIR/traces/<point>.trace. Nothing under rtl/,
python/ or archbench/ is modified or bypassed: the recording is the ordinary
run with a monitor listening.

Runs where Icarus and the Torch/Triton environment are (the evaluation host),
from the source tree being recorded.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SYSTEMC = HERE.parent / 'systemc'
sys.path.insert(0, str(SYSTEMC))
# The model driver's environment, so both recordings compile and launch the
# same programs: HARNESS (profile, opt level, Triton), RUNTIME (transfer
# policy, one active core, one outlet, one outstanding line, single submit),
# and the delivered compile knobs (B40N).
from run_graphs import HARNESS, RUNTIME, knobs_for  # noqa: E402
PROFILE = HARNESS['TNCA_PROFILE']


def slug(point):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', point)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# The board-top stand-in's structure (python/tnca/backends/multicore_mmio_sim.py MulticoreMMIOLink):
# the U280 image's link and pipeline stages, maintenance and memory controls, three AXI outlets.
BOARD = dict(link_stages=2, noc_link_stages=2, dispatch_stages=2, home_stages=2, code_stages=2,
             maintenance=1, memory_controls=1, memory_ports=3, memory_bytes=1 << 20, board_top=True)


def build_bench(out, *, core_count, memory_bytes, memory_ports, traced_core, overrides, iverilog, bench='device'):
    """The runtime's own file set for the chosen bench plus the monitor top."""
    rtl, tb = ROOT / 'rtl', ROOT / 'tb'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'device.vvp'
    if bench == 'board':   # multicore_mmio_sim.MulticoreMMIOLink._build, board_top=True
        top = 'tnca_multicore_board_regs'
        args = [iverilog, '-g2012', '-I', str(rtl / 'common'), '-I', str(rtl / 'noc'),
                '-s', top, '-s', 'tnca_core_trace', '-DTNCA_TRACE_BOARD', f'-DTNCA_TRACE_CORES={core_count}',
                f'-DTNCA_TRACE_PORTS={BOARD["memory_ports"]}', '-o', str(target)]
        params = dict(N=core_count, PORTS=BOARD['memory_ports'], MEMORY_BYTES=BOARD['memory_bytes'],
                      STAGES=BOARD['link_stages'], NOC_PIPE=BOARD['noc_link_stages'], DISPATCH_PIPE=BOARD['dispatch_stages'],
                      HOME_PIPE=BOARD['home_stages'], CODE_PIPE=BOARD['code_stages'], MAINTENANCE=BOARD['maintenance'],
                      MEMORY_CONTROLS=BOARD['memory_controls'])
        args += [f'-P{top}.{k}={v}' for k, v in params.items()]
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            args += ['-y', str(rtl / library)]
        args += ['-y', str(rtl), '-y', str(rtl / 'fpga'), '-y', str(rtl / 'board')]
        args += [str(p) for p in sorted((rtl / 'common').glob('*.v'))]
        args += [str(tb / 'common/tnca_axi_line_model.v'), str(tb / f'fpga/{top}.v'), str(HERE / 'tnca_core_trace.v')]
    else:
        args = [iverilog, '-g2012', '-I', str(rtl / 'common'), '-I', str(rtl / 'noc'),
                '-s', 'tnca_multicore_device', '-s', 'tnca_core_trace',
                f'-DTNCA_TRACE_CORES={core_count}',
                f'-Ptnca_multicore_device.N={core_count}',
                f'-Ptnca_multicore_device.TOTAL_WORDS={memory_bytes // 8}',
                f'-Ptnca_multicore_device.PORTS={memory_ports}', '-o', str(target)]
        args += [f'-Ptnca_multicore_device.{k}={v}' for k, v in sorted(overrides.items())]
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            args += ['-y', str(rtl / library)]
        args += [str(p) for p in sorted((rtl / 'common').glob('*.v'))]
        args += [str(rtl / p) for p in ('tnca_top.v', 'tnca_grid_dispatch.v', 'tnca_grid_deadline.v', 'tnca_multicore_shell.v')]
        args += [str(tb / 'common/tnca_dram_model.v'), str(tb / 'device/tnca_multicore_device.v'),
                 str(HERE / 'tnca_core_trace.v')]
    (out / 'iverilog.cmd').write_text(' \\\n  '.join(args) + '\n')
    r = subprocess.run(args, capture_output=True, text=True)
    (out / 'iverilog.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f'bench elaboration failed, see {out / "iverilog.log"}')
    version = subprocess.run([iverilog, '-V'], capture_output=True, text=True).stdout.splitlines()[0]
    return dict(path=str(target), sha256=file_sha(target), compiler=iverilog, compiler_version=version,
                core_count=core_count, memory_bytes=memory_bytes, memory_ports=memory_ports,
                traced_core=traced_core, rtl_overrides=overrides, bench=bench,
                board=BOARD if bench == 'board' else None)


def build_described(out, *, device, memory_bytes):
    """The board-top stand-in of a device description (R41 on), elaborated by the product's own build
    (MulticoreMMIOLink._build: the image's parameters, striped HBM, the AXI width) with the monitor as an
    observation top -- the bench the agent harness's evaluator runs, plus a monitor that only listens."""
    sys.path.insert(0, str(ROOT / 'python'))
    from tnca import devices
    from tnca.backends.multicore_mmio_sim import MulticoreMMIOLink
    desc = devices.load(str(ROOT / device))
    cores, ports = desc['platform']['core_count'], desc['uncore']['MEM_PORTS']
    out.mkdir(parents=True, exist_ok=True)
    link = MulticoreMMIOLink(board_top=True, device=desc, memory_bytes=memory_bytes, workdir=str(out),
                             observe=dict(tops=['tnca_core_trace'],
                                          sources=[str((HERE / 'tnca_core_trace.v').relative_to(ROOT))],
                                          defines=['TNCA_TRACE_BOARD', f'TNCA_TRACE_CORES={cores}',
                                                   f'TNCA_TRACE_PORTS={ports}']))
    target = Path(link._build())
    version = subprocess.run([shutil.which('iverilog') or 'iverilog', '-V'], capture_output=True,
                             text=True).stdout.splitlines()[0]
    return dict(path=str(target), sha256=file_sha(target), compiler_version=version, core_count=cores,
                memory_bytes=memory_bytes, memory_ports=ports, bench='described', device=device,
                device_name=desc['name'], profile=desc['per_core_profile'])


# The agent harness's evaluator runs a described machine's point exactly so (tritonchip_agent_harness
# evaluator.py, the multicore path): the product's defaults for every compile knob, the backend in the
# environment, the scored profile on the command line, no warm-up or timed calls of its own.
def evaluator_env(*, device, memory_bytes, bench_path, active_cores, timeout, work):
    args = dict(board_top=True, device=str(ROOT / device), memory_bytes=memory_bytes, hbm_dma=True,
                harness=str(bench_path))
    return dict(TNCA_BACKEND='fpga-multicore-sim', TNCA_BACKEND_ARGS=json.dumps(args),
                TNCA_COMPILE_CACHE_DIR=str(work / 'tnca-inductor'), TNCA_ACTIVE_CORES=str(active_cores),
                TNCA_HOST_TIMEOUT=f'{timeout:g}')


def suite_points(path, tier):
    suite = json.loads(Path(path).read_text())
    # spec/r40 suites tag every point with its tier; archbench/suites/<tier>.json (R41) tag the suite.
    return [p['id'] for p in suite['points'] if not tier or p.get('tier', suite.get('tier')) in tier]


def run_point(point, *, python, bench, out, core_count, memory_ports, timeout, extra_env, warmup=0, timed=1,
              torch_device='native', active_cores=1, bench_kind='device', device=None, profile=None,
              memory_bytes=None):
    trace = out / 'traces' / f'{slug(point)}.trace'
    log = out / 'logs' / f'{slug(point)}.log'
    rec = out / 'records' / f'{slug(point)}.jsonl'
    for p in (trace.parent, log.parent, rec.parent):
        p.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
    work = out / 'work' / slug(point)
    if bench_kind == 'described':
        # The evaluator's environment, nothing of run_graphs.py's.
        env.update(evaluator_env(device=device, memory_bytes=memory_bytes, bench_path=bench,
                                 active_cores=active_cores, timeout=timeout, work=work))
        env.update(extra_env)
    else:
        env.update(HARNESS)
        env.update(RUNTIME)
        env.update(knobs_for(point, None))
        env.update(extra_env)
        env['TNCA_ACTIVE_CORES'] = str(active_cores)
        env['TNCA_TORCH_DEVICE'] = torch_device   # run_graphs.py --torch-device: native tensors, as the model runs
    utrace = out / 'traces' / f'{slug(point)}.uncore.trace'
    plusargs = [f'+core_trace={trace}'] + [f'+core_trace{c}={trace}.core{c}' for c in range(1, core_count)] + [f'+uncore_trace={utrace}']
    env['TNCA_DEVICE_PLUSARGS'] = ' '.join(plusargs)
    for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
        (work / d).mkdir(parents=True, exist_ok=True)
    env.update(TRITON_CACHE_DIR=str(work / 'triton_cache'), TRITON_HOME=str(work / 'triton_home'),
               TORCHINDUCTOR_CACHE_DIR=str(work / 'inductor_cache'), TMPDIR=str(work / 'tmp'),
               TNCA_DUMP_DIR=str(work / 'dump'))
    if bench_kind == 'described':
        # the evaluator's command line: the backend comes from the environment
        cmd = [python, '-m', 'archbench.run', '--platform', 'tnca', '--profile', profile,
               '--point', point, '--out', str(rec)]
    else:
        if bench_kind == 'board':
            backend = 'fpga-multicore-sim'
            backend_args = json.dumps(dict(harness=str(bench), core_count=core_count, **BOARD))
        else:
            backend = 'sim-multicore'
            backend_args = json.dumps(dict(bench=str(bench), core_count=core_count, memory_ports=memory_ports))
        cmd = [python, '-m', 'archbench.run', '--platform', 'tnca', '--profile', PROFILE,
               '--backend', backend, '--backend-args', backend_args, '--opt-level', '2', '--seed', '0',
               '--warmup', str(warmup), '--timed', str(timed),
               '--point', point, '--out', str(rec), '--timeout', str(timeout)]
    t0 = time.time()
    with log.open('w') as f:
        f.write(' '.join(cmd) + '\n')
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    # the monitor names core c>0's file <trace>.core<c>; keep the directory's naming (<slug>.core<c>.trace)
    for c in range(1, core_count):
        extra = Path(str(trace) + f'.core{c}')
        if extra.exists():   # keep the monitor's name (the model replays <trace>.core<c>) and the directory's
            shutil.copyfile(extra, out / 'traces' / f'{slug(point)}.core{c}.trace')
    status = dict(point=point, returncode=r.returncode, seconds=round(time.time() - t0, 1),
                  trace=str(trace) if trace.exists() else None, record=str(rec) if rec.exists() else None,
                  uncore=str(utrace) if utrace.exists() else None)
    if trace.exists():
        with trace.open() as f:
            last = ''
            for last in f:
                pass
        status['trace_end'] = last.strip()
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--python', required=True, help='the Torch/Triton interpreter with tnca installed')
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite')
    ap.add_argument('--tier', nargs='*', default=['tiny'])
    ap.add_argument('--point', action='append', default=[])
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=7200, help='host seconds per point (TNCA_HOST_TIMEOUT)')
    ap.add_argument('--core-count', type=int, default=3, help='cores elaborated; one is active')
    ap.add_argument('--memory-ports', type=int, default=1)
    ap.add_argument('--memory-bytes', type=int, default=8 * 1024 * 1024)
    ap.add_argument('--active-cores', type=int, default=1, help='TNCA_ACTIVE_CORES: cores the grid runs on')
    ap.add_argument('--bench', choices=['device', 'board'], default='device',
                    help='device: tb/device bench on sim-multicore (DRAM model); board: the board-top stand-in (fpga-multicore-sim, AXI outlets)')
    ap.add_argument('--iverilog', default=shutil.which('iverilog'))
    ap.add_argument('--set', action='append', default=[], help='KEY=VALUE in every child environment')
    ap.add_argument('--warmup', type=int, default=0, help='untimed calls after the cold call (as run_graphs.py)')
    ap.add_argument('--timed', type=int, default=1, help='timed calls (as run_graphs.py)')
    ap.add_argument('--torch-device', default='native', help='TNCA_TORCH_DEVICE (as run_graphs.py)')
    ap.add_argument('--reuse-bench', action='store_true', help='keep RUN_DIR/bench/device.vvp if present')
    ap.add_argument('--device', help='a device description (spec/devices/*.json, tree-relative): its board-top '
                                     'stand-in by the product\'s own build, run in the agent harness evaluator\'s '
                                     'environment (overrides --bench, --core-count, --memory-ports)')
    ap.add_argument('--profile', help='with --device: the scored profile (default: the description\'s per-core profile)')
    args = ap.parse_args()
    if args.device:
        args.bench = 'described'
        if args.memory_bytes == 8 * 1024 * 1024:
            args.memory_bytes = 16 << 20   # the evaluator's MULTICORE_MEMORY
    if not args.iverilog:
        sys.exit('iverilog not found')
    points = list(args.point) + (suite_points(args.suite, set(args.tier)) if args.suite else [])
    if not points:
        sys.exit('no points: give --suite or --point')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    extra_env = dict(kv.split('=', 1) for kv in args.set)
    overrides = {}
    for k, v in extra_env.items():
        if k.startswith('TNCA_RTL_') and k[9:] in ('VRF_BYPASS', 'FE_NOBUBBLE', 'VRF_PREFETCH', 'RCTX_SWITCH',
                                                   'MTX_ABUF', 'LSU_INFLIGHT', 'FPU_OVERLAP'):
            overrides[k[9:]] = v
    bench_dir = out / 'bench'
    if args.reuse_bench and (bench_dir / 'bench.json').exists() and Path(
            json.loads((bench_dir / 'bench.json').read_text())['path']).exists():
        bench = json.loads((bench_dir / 'bench.json').read_text())
    elif args.device:
        bench = build_described(bench_dir, device=args.device, memory_bytes=args.memory_bytes)
        (bench_dir / 'bench.json').write_text(json.dumps(bench, indent=1) + '\n')
    else:
        bench = build_bench(bench_dir, core_count=args.core_count, memory_bytes=args.memory_bytes,
                            memory_ports=args.memory_ports, traced_core=0,
                            overrides=overrides, iverilog=args.iverilog, bench=args.bench)
        (bench_dir / 'bench.json').write_text(json.dumps(bench, indent=1) + '\n')
    if args.device:
        args.core_count, args.memory_ports = bench['core_count'], bench['memory_ports']
        profile = args.profile or bench['profile']
    else:
        profile = PROFILE
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'rtl', 'tb', 'python', 'archbench', 'compiler'],
                           cwd=ROOT, capture_output=True, text=True)
    source = dict(root=str(ROOT), commit=head.stdout.strip() if head.returncode == 0 else None,
                  dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None)
    if args.device:
        environment = evaluator_env(device=args.device, memory_bytes=args.memory_bytes, bench_path=bench['path'],
                                    active_cores=args.active_cores, timeout=args.timeout, work=out / 'work' / '<point>')
        environment.update(extra_env)
        knobs, warmup, timed, torch_device = 'the product\'s defaults (the evaluator\'s)', None, None, None
    else:
        environment = dict(HARNESS, **RUNTIME)
        environment.update(extra_env)
        knobs, warmup, timed, torch_device = 'run_graphs.knobs_for (B40N)', args.warmup, args.timed, args.torch_device
    manifest = dict(schema='tnca-core-trace-run/2', source=source,
                    bench=bench, profile=profile, active_cores=args.active_cores, points=points, environment=environment,
                    compile_knobs=knobs, warmup=warmup, timed=timed, torch_device=torch_device,
                    python=args.python, started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_point, p, python=args.python, bench=bench['path'], out=out,
                               core_count=args.core_count, memory_ports=args.memory_ports,
                               timeout=args.timeout, extra_env=extra_env, warmup=args.warmup,
                               timed=args.timed, torch_device=args.torch_device,
                               active_cores=args.active_cores, bench_kind=args.bench, device=args.device,
                               profile=profile, memory_bytes=args.memory_bytes): p for p in points}
        for fut in cf.as_completed(futures):
            s = fut.result()
            results.append(s)
            print(f"{s['point']}: rc={s['returncode']} {s['seconds']}s {s.get('trace_end', 'no trace')}", flush=True)
    with (out / 'records.jsonl').open('w') as f:
        for s in results:
            if s['record']:
                f.write(Path(s['record']).read_text())
    manifest['results'] = sorted(results, key=lambda s: s['point'])
    manifest['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    failed = [s['point'] for s in results if s['returncode'] or not s['trace']]
    print(f'{len(results) - len(failed)} of {len(results)} points recorded' + (f'; failed: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
