#!/usr/bin/env python3
"""Record core traces from the model: run_graphs.py with the trace switched on, one trace per point.

    record_model.py --binary BUILD/tnca-chip --out RUN_DIR \
        (--suite spec/r40/suites/existing.json --tier tiny | --point ID ...) [--config CFG] [--jobs N]

The run is the model's own standard path (tools/performance_harness/systemc/
run_graphs.py: torch.compile -> Inductor -> Triton -> tnca -> Session ->
systemc-tlm-grid, one child and one model process per point) with
TNCA_MODEL_CORE_TRACE=1, so every model process writes core_trace.txt into
its run directory. Afterwards each point's trace is copied to
RUN_DIR/traces/<point>.trace, named as record_rtl.py names the RTL's, so
tracediff.py gate can pair them.

The same knob sets as run_graphs.py (HARNESS, RUNTIME, the point's compile
knobs) are the environment of both recordings, so both sides compile and
launch the same programs; record_rtl.py imports them from run_graphs.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SYSTEMC = HERE.parent / 'systemc'
sys.path.insert(0, str(HERE))
from record_rtl import slug  # noqa: E402


def collect(out):
    """results.json rows -> traces/<slug>.trace; returns {point: status}."""
    results = json.loads((out / 'results.json').read_text())
    traces = out / 'traces'
    traces.mkdir(exist_ok=True)
    status = {}
    for row in results['points']:
        point = row['key'].split('@')[0]
        dirs = [Path(d) for d in row.get('model_dirs') or []]
        found = [d / 'core_trace.txt' for d in dirs if (d / 'core_trace.txt').exists()]
        if not found:
            run_dir = out / 'runs' / row['key'].replace('/', '__').replace('@', '__')
            found = sorted((run_dir / 'model').glob('model-*/core_trace.txt'))
        if len(found) == 1:
            shutil.copyfile(found[0], traces / f'{slug(point)}.trace')
            status[point] = dict(status=row.get('status'), trace=str(found[0]), device_cycles=row.get('device_cycles'))
        else:
            status[point] = dict(status=row.get('status'), trace=None, model_dirs=[str(d) for d in dirs],
                                 found=[str(f) for f in found])
    (out / 'traces.json').write_text(json.dumps(status, indent=1, sort_keys=True) + '\n')
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', required=True, help='tnca-chip built with the trace emitter')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=str(SYSTEMC / 'configs/r40-board-m2.cfg'))
    ap.add_argument('--suite')
    ap.add_argument('--tier', nargs='*', default=['tiny'])
    ap.add_argument('--point', action='append', default=[])
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--python', default=sys.executable, help='the Torch/Triton interpreter')
    ap.add_argument('--env', action='append', default=[], help='KEY=VALUE for every run (run_graphs.py --env)')
    ap.add_argument('--collect-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if not args.collect_only:
        cmd = [args.python, str(SYSTEMC / 'run_graphs.py'), '--source-root', str(ROOT), '--binary', args.binary,
               '--config', args.config, '--out', str(out), '--jobs', str(args.jobs), '--timeout', str(args.timeout),
               '--env', 'TNCA_MODEL_CORE_TRACE=1']
        for kv in args.env:
            cmd += ['--env', kv]
        if args.suite:
            cmd += ['--suite', args.suite]
            for t in args.tier:
                cmd += ['--tier', t]
        for p in args.point:
            cmd += ['--point', p]
        print(' '.join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f'run_graphs.py exited {r.returncode}', file=sys.stderr)
    status = collect(out)
    missing = [p for p, s in status.items() if not s['trace']]
    print(f'{len(status) - len(missing)} of {len(status)} model traces collected' + (f'; missing: {missing}' if missing else ''))
    return 1 if missing or not status else 0


if __name__ == '__main__':
    sys.exit(main())
