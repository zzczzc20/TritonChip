#!/usr/bin/env python3
"""Record one test case of your own on the RTL and on the model, and compare them the evaluation's way.

    python /opt/tnca-skills/cycle-model-alignment/scripts/align_case.py CASE.py [--level uncore|core]
        [--out DIR] [--device DESC] [--config CFG] [--active-cores N] [--model-build DIR] [--bench-dir DIR]
        [--reuse-bench] [--data]

Run it from the repository checkout you work in (or set TNCA_ROOT). A case
file defines `make()` -> (torch.nn.Module, tuple of CPU input tensors), as the
test-case skill's cases do. The script then

  1. builds the model from your tree (`tools/performance_harness/systemc`,
     CMake + Ninja; the SystemC library is in the environment), incrementally
     in the --model-build directory;
  2. elaborates the machine's board-top bench from its device description
     (`spec/devices/*.json`, your tree's: `align/record_rtl.py`'s
     `build_described`, the evaluator's own build) with the trace monitor
     (`tools/performance_harness/align/tnca_core_trace.v`) as an observation
     top, in --bench-dir;
  3. runs the case on the RTL (`fpga-multicore-sim` on that bench, the
     evaluator's environment) with the trace switched on: DIR/rtl/case.trace,
     case.core<c>.trace, case.uncore.trace;
  4. extracts the replay input of the chosen level from the RTL's trace
     (`replay_extract.py`): the one boundary stream the model is given;
  5. runs the case on the model in replay, with its trace switched on:
     DIR/model/case.trace, case.core<c>.trace, case.uncore.trace;
  6. runs the gate (`tracediff.py gate`) over the two directories and prints
     its verdict and the first divergence of every stream: DIR/alignment.json.

Both sides compile and launch the same program under the same environment:
the evaluator's (the product's defaults for every compile knob, the scored
profile, the active cores you ask for), one call of the compiled module.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The machine the R41 line is measured on, and the memory the evaluator gives its bench (MULTICORE_MEMORY).
DEVICE = 'spec/devices/u280-3core-r41.json'
CONFIG = 'tools/performance_harness/systemc/configs/r41-board-cycle.cfg'
MEMORY_BYTES = 16 << 20
STREAMS = ('issue', 'request', 'response', 'length')


def root():
    r = Path(os.environ.get('TNCA_ROOT', os.getcwd())).resolve()
    if not (r / 'tools/performance_harness/align/tracediff.py').exists():
        sys.exit(f'{r} is not a TritonChip checkout (set TNCA_ROOT or run from the checkout)')
    return r


def build_model(r, build):
    """The model from your tree: configured once per build directory, then built incrementally every time, so
    a change to the model's source is always in the binary that runs."""
    binary = build / 'tnca-chip'
    build.mkdir(parents=True, exist_ok=True)
    cmds = [['cmake', '--build', str(build), '--target', 'tnca-chip']]
    if not (build / 'CMakeCache.txt').exists():
        cmds.insert(0, ['cmake', '-S', str(r / 'tools/performance_harness/systemc'), '-B', str(build), '-G', 'Ninja',
                        '-DCMAKE_BUILD_TYPE=Release'])
    for cmd in cmds:
        p = subprocess.run(cmd, capture_output=True, text=True)
        (build / 'build.log').open('a').write(' '.join(cmd) + '\n' + p.stdout + p.stderr)
        if p.returncode:
            sys.exit(f'the model did not build; see {build / "build.log"}')
    return binary


def build_bench(r, out, device, reuse):
    """The machine's board-top bench, from its device description, as the evaluator builds it, with the trace
    monitor as an observation top (align/record_rtl.py's build_described)."""
    stamp = out / 'built-from.json'
    if reuse and stamp.exists():
        return json.loads(stamp.read_text())
    if out.exists():
        shutil.rmtree(out)
    sys.path.insert(0, str(r / 'tools/performance_harness/align'))
    import record_rtl                                        # noqa: E402  your tree's recorder
    bench = record_rtl.build_described(out, device=device, memory_bytes=MEMORY_BYTES)
    stamp.write_text(json.dumps(bench, indent=1) + '\n')
    return bench


def side_env(r, work, active_cores, profile, timeout):
    """The evaluator's environment (the product's defaults for every compile knob): the profile, the active
    cores and the host timeout are all it sets; each side adds its backend."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
    env.update(TNCA_PROFILE=profile, TNCA_ACTIVE_CORES=str(active_cores), TNCA_HOST_TIMEOUT=f'{timeout:g}',
               TNCA_COMPILE_CACHE_DIR=str(work / 'tnca-inductor'), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='')
    for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
        (work / d).mkdir(parents=True, exist_ok=True)
    env.update(TRITON_CACHE_DIR=str(work / 'triton_cache'), TRITON_HOME=str(work / 'triton_home'),
               TORCHINDUCTOR_CACHE_DIR=str(work / 'inductor_cache'), TMPDIR=str(work / 'tmp'),
               TNCA_DUMP_DIR=str(work / 'dump'))
    env['PYTHONPATH'] = os.pathsep.join([str(r), str(r / 'python'), str(r / 'compiler/python'), str(r / 'tools/asm'),
                                         str(r / 'tools/performance_harness/systemc')]
                                        + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return env


def run_side(r, side, case, out, env, timeout):
    """One side in a fresh process: this script with --side, which compiles and runs the case once."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, '--side', side, str(case), '--out', str(out)]
    with (out / 'run.log').open('w') as log:
        log.write(' '.join(cmd) + '\n')
        log.flush()
        p = subprocess.run(cmd, cwd=r, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    result = out / 'result.json'
    return json.loads(result.read_text()) if result.exists() else dict(status='no-result', returncode=p.returncode)


def run_case(case, out):
    """Inside a side's process: compile the case with TNCA as the backend and run it once, every launch logged."""
    if os.environ.get('TNCA_BACKEND', '').startswith('systemc-tlm'):
        import tlm_backend                                   # noqa: E402  the model's backend, before torch
        tlm_backend.register()
    import torch
    import triton                                            # noqa: F401
    import tnca.inductor
    from tnca.triton_backend.driver import launch_log
    from torch._inductor.utils import run_and_get_code
    # A case may import the test-case skill's helpers (its scripts/, beside this skill), or files beside it.
    for extra in (Path(case).resolve().parent, Path(__file__).resolve().parents[2] / 'testcase-design' / 'scripts'):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(f'_case_{Path(case).stem}', case)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    module, inputs = mod.make()
    rec = dict(status='not-measured', problems=[], launches=None, device_cycles=None)
    tnca.inductor.configure()
    torch._dynamo.reset()
    try:
        with torch.no_grad(), launch_log() as log:
            got, code = run_and_get_code(torch.compile(module), *tuple(inputs))
    except Exception as exc:                                 # noqa: BLE001 -- the failure is the result
        rec['status'] = 'failed'
        rec['problems'].append(f'{type(exc).__name__}: {str(exc)[:800]}')
    else:
        rec['launches'] = len(log)
        rec['device_cycles'] = sum(e['device_cycles'] or 0 for e in log)
        rec['fallbacks'] = tnca.inductor.host_fallbacks(code)
        rec['status'] = 'pass' if log and not rec['fallbacks'] else 'unsupported'
        if rec['fallbacks']:
            rec['problems'].append(f'host fallback: {rec["fallbacks"]}')
    if os.environ.get('TNCA_BACKEND', '').startswith('systemc-tlm'):
        for inst in tlm_backend.backend_class().instances:   # the model processes end: their traces close
            try:
                inst.stop()
            except Exception:                                # noqa: BLE001
                pass
    (out / 'result.json').write_text(json.dumps(rec, indent=1) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('case', nargs='?', help='a case file defining make()')
    ap.add_argument('--level', choices=['core', 'uncore'], default='uncore')
    ap.add_argument('--out', help='where the traces and the verdict go (default align-<case>/ beside the case)')
    ap.add_argument('--device', default=DEVICE, help='the machine: a device description in your tree')
    ap.add_argument('--profile', help='the scored profile (default: the description\'s per-core profile)')
    ap.add_argument('--config', default=CONFIG, help='the model configuration (the CA model of the board-top bench)')
    ap.add_argument('--active-cores', type=int, default=3, help='the cores the grids run on (3: as the card runs)')
    ap.add_argument('--model-build', default='/tmp/tnca-model', help='the model\'s build directory (built incrementally)')
    ap.add_argument('--bench-dir', default='/tmp/tnca-trace-bench', help='the RTL bench\'s directory')
    ap.add_argument('--reuse-bench', action='store_true', help='keep the bench in --bench-dir (RTL unchanged since)')
    ap.add_argument('--timeout', type=float, default=7200, help='seconds per side')
    ap.add_argument('--data', action='store_true', help='compare write and read data as well')
    ap.add_argument('--side', choices=['rtl', 'model'], help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.side:
        run_case(a.case, Path(a.out))
        return 0
    if not a.case:
        ap.error('a case file is required')
    r = root()
    case = Path(a.case).resolve()
    out = Path(a.out).resolve() if a.out else case.parent / f'align-{case.stem}'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    binary = build_model(r, Path(a.model_build).resolve())
    bench = build_bench(r, Path(a.bench_dir).resolve(), a.device, a.reuse_bench)
    cores, profile = int(bench['core_count']), a.profile or bench['profile']
    align = r / 'tools/performance_harness/align'
    sys.path.insert(0, str(align))
    from replay_extract import extract                       # noqa: E402

    # 1. the RTL, with the monitor listening
    rtl = out / 'rtl'
    env = side_env(r, out / 'work-rtl', a.active_cores, profile, a.timeout)
    env['TNCA_BACKEND'] = 'fpga-multicore-sim'
    env['TNCA_BACKEND_ARGS'] = json.dumps(dict(board_top=True, device=str(r / a.device), memory_bytes=MEMORY_BYTES,
                                               hbm_dma=True, harness=str(bench['path'])))
    rtl.mkdir(parents=True)
    plusargs = [f'+core_trace={rtl / "case.trace"}'] + [f'+core_trace{c}={rtl / f"case.trace.core{c}"}' for c in range(1, cores)]
    plusargs.append(f'+uncore_trace={rtl / "case.uncore.trace"}')
    env['TNCA_DEVICE_PLUSARGS'] = ' '.join(plusargs)
    t0 = time.time()
    r_rtl = run_side(r, 'rtl', case, rtl, env, a.timeout + 600)
    for c in range(1, cores):                                # the monitor's <trace>.core<c>, kept by stem as well
        extra = rtl / f'case.trace.core{c}'
        if extra.exists():
            shutil.copyfile(extra, rtl / f'case.core{c}.trace')
    print(f'rtl:   {r_rtl.get("status")}  launches {r_rtl.get("launches")}  cycles {r_rtl.get("device_cycles")}  '
          f'({time.time() - t0:.0f} s)', flush=True)
    if r_rtl.get('status') != 'pass':
        print('\n'.join(r_rtl.get('problems', [])) or f'see {rtl / "run.log"}')
        return 2

    # 2. the replay input: the one stream the level replays, and nothing else of the recording
    replay = out / 'replay'
    if a.level == 'core':
        extract('core', rtl / 'case.trace', replay / 'case.replay')
        for c in range(1, cores):
            extra = rtl / f'case.core{c}.trace'
            if extra.exists():
                extract('core', extra, replay / f'case.replay.core{c}')
        replay_env = {'TNCA_MODEL_CORE_REPLAY': str(replay / 'case.replay')}
    else:
        extract('uncore', rtl / 'case.uncore.trace', replay / 'case.uncore.replay')
        replay_env = {'TNCA_MODEL_UNCORE_REPLAY': str(replay / 'case.uncore.replay')}

    # 3. the model, in replay, with its trace switched on
    model = out / 'model'
    env = side_env(r, out / 'work-model', a.active_cores, profile, a.timeout)
    env.update(TNCA_BACKEND='systemc-tlm-grid', TNCA_TLM_BINARY=str(binary),
               TNCA_TLM_CONFIG=str((r / a.config).resolve()), TNCA_TLM_OUT=str(model / 'tlm'),
               TNCA_MODEL_CORE_TRACE=str(model / 'case.trace'), TNCA_MODEL_CORE_TRACE_CORE='all', **replay_env)
    t0 = time.time()
    r_model = run_side(r, 'model', case, model, env, a.timeout + 600)
    print(f'model: {r_model.get("status")}  launches {r_model.get("launches")}  cycles {r_model.get("device_cycles")}  '
          f'({time.time() - t0:.0f} s)', flush=True)
    # the model names core c>0's file <trace>.core<c> and the uncore's <trace>.uncore: the gate pairs on stems
    for extra in sorted(model.glob('case.trace.*')):
        suffix = extra.name.split('.')[-1]
        if suffix.startswith('core') or suffix == 'uncore':
            extra.rename(model / f'case.{suffix}.trace')
    if r_model.get('status') != 'pass':
        print('\n'.join(r_model.get('problems', [])) or f'see {model / "run.log"}')

    for extra in rtl.glob('case.trace.core*'):
        extra.unlink()

    # 4. the gate
    verdict = out / 'alignment.json'
    cmd = [sys.executable, str(align / 'tracediff.py'), 'gate', '--a', str(rtl), '--b', str(model), '--out', str(verdict)]
    if a.data:
        cmd.append('--data')
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout.rstrip())
    if p.stderr.strip():
        print(p.stderr.rstrip(), file=sys.stderr)
    if verdict.exists():
        v = json.loads(verdict.read_text())
        for stem, pt in sorted(v.get('points', {}).items()):
            streams = pt.get('streams') or {}
            bad = {s: d for s, d in streams.items() if d != 'equal'}
            print(f"  {stem:28} {pt.get('status', '?'):9} launches {pt.get('launches_a')}/{pt.get('launches_b')}"
                  + (f"  first divergence: {json.dumps(bad)[:300]}" if bad else ''))
        print(f'verdict: {v.get("verdict")}  ({verdict})')
        return 0 if v.get('verdict') == 'aligned' else 1
    return 2


if __name__ == '__main__':
    sys.exit(main())
