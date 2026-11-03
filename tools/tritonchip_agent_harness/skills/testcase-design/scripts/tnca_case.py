#!/usr/bin/env python3
"""Measure a test case of your own the way the evaluation measures a point.

    python /opt/tnca-skills/testcase-design/scripts/tnca_case.py CASE.py [CASE2.py ...] [--profile P] [--opt-level N] [--json OUT]
    python /opt/tnca-skills/testcase-design/scripts/tnca_case.py /opt/tnca-skills/testcase-design/examples/*.py

Run it from the repository checkout you work in (or set TNCA_ROOT): the bound
reads that checkout's spec/ and compiler/targets. The device it measures is the
one your environment selects: this script sets TNCA_PROFILE and TNCA_OPT_LEVEL
and nothing else about the machine, so TNCA_BACKEND and its arguments are your
shell's, as the task text sets them.

A case file defines

    def make():                # -> (torch.nn.Module, tuple of CPU input tensors)
    MODEL = False              # True: a whole model, judged at the model tolerance
    TOL = None                 # a tolerance of your own instead of the protocol's
    FP_OPS = 0.0; SFU_OPS = 0.0   # demand the FLOP counter cannot see (see bound.demand_of)
    DEMAND = None              # a bound.Demand declared by hand, instead of the measured one

and this script does, for each case, what the evaluation does for a point:

  1. the reference: eager PyTorch in binary32 on the CPU, inputs and weights
     upcast exactly (protocol.reference);
  2. the run: torch.compile with TNCA as the backend, compiled once and run
     once, every launch logged;
  3. the verdict: the protocol's criterion at the protocol's tolerance
     (protocol.compare); any host fallback fails the case; a case that does not
     compile or run fails with the evaluation's status names;
  4. the time: the sum of the device cycles of every launch;
  5. the bound: demand measured from the case (bound.demand_of), L from the
     profile's spec_peak, eta = L / T, and which resource binds. A profile is
     one core, so L and eta are one core's, as the evaluation's own bound is.

What differs from the evaluation, and how to read it:

  * The cycles here are the ones the device reports to the driver. The score
    uses the platform's own count at the device boundary, which is a few
    cycles more per launch. Compare cases and versions by their RELATIVE
    cycles; do not expect the absolute number to equal a score.
  * The device you measure is the one your environment points at, built as
    the task text says. After a change to the RTL, a profile, a device
    description or a compiler target, build it again or you measure the old
    machine.
  * Each case is compiled afresh (torch._dynamo.reset), so the first call
    compiles and runs, as in the evaluation.

Importable too: `measure(module, inputs, profile=..., opt_level=...)` returns the
same dict the --json output holds.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import protocol  # noqa: E402
import bound as B  # noqa: E402

STATUSES = ('pass', 'numeric-error', 'compile-failure', 'runtime-fault', 'capacity-exceeded',
            'timeout', 'unsupported', 'not-measured')


def _env(profile, opt_level):
    os.environ['TNCA_PROFILE'] = profile
    os.environ['TNCA_OPT_LEVEL'] = str(opt_level)
    os.environ.setdefault('TNCA_HOST_TIMEOUT', '3600')


def measure(module, inputs, *, profile, opt_level=2, tol=None, model=False, demand=None,
            fp_ops=0.0, sfu_ops=0.0):
    """One case, measured the evaluation's way. Never raises for a failing case:
    the failure is the result (`status`, `problems`)."""
    _env(profile, opt_level)
    import torch
    import triton                                            # noqa: F401  (before the driver)
    import tnca.inductor
    from tnca.triton_backend.driver import launch_log
    from torch._inductor.utils import run_and_get_code

    dtype = protocol.dtype_of(module, inputs)
    if tol is None:
        if dtype is None:
            raise ValueError('the case has no floating-point input or parameter; pass tol=')
        tol = protocol.tolerance(dtype, model=model)
    rec = dict(schema='tnca-testcase/1', profile=profile, opt_level=opt_level, dtype=dtype, tol=tol,
               status='not-measured', problems=[], fallbacks=[], correctness=None, time=None,
               demand=None, bound=None, eta=None, host=None)

    ref = protocol.reference(module, inputs)
    try:
        d = demand or B.demand_of(module, inputs, fp_ops=fp_ops, sfu_ops=sfu_ops)
        rec['demand'] = d.as_dict()
        L, binding, terms = B.bound(d, profile)
        rec['bound'] = dict(cycles=L, binding=binding, terms=terms, unit='cycles',
                            scope='one core of the profile, as the evaluation\'s own bound')
    except Exception as exc:                                 # a bound is advice, not a verdict
        rec['problems'].append(f'no bound: {type(exc).__name__}: {exc}')

    tnca.inductor.configure()
    torch._dynamo.reset()
    t0 = time.monotonic()
    try:
        with torch.no_grad(), launch_log() as log:
            got, code = run_and_get_code(torch.compile(module), *inputs)
    except Exception as exc:
        rec['status'] = protocol.classify(exc)
        rec['problems'].append(f'{type(exc).__name__}: {str(exc)[:800]}')
        return rec
    wall = time.monotonic() - t0

    rec['fallbacks'] = tnca.inductor.host_fallbacks(code)
    cmp = protocol.compare(got, ref, tol)
    rec['correctness'] = cmp
    cycles = sum(e['device_cycles'] or 0 for e in log)
    rec['time'] = dict(device_cycles=cycles, launches=len(log),
                       per_launch=[dict(kernel=e['kernel'], grid=list(e['grid']), cycles=e['device_cycles'])
                                   for e in log],
                       note='the device\'s count as the driver sees it; the score counts at the device '
                            'boundary, a few cycles more per launch')
    rec['host'] = dict(wall_s=round(wall, 3),
                       upload_bytes=sum(e['upload_bytes'] for e in log),
                       download_bytes=sum(e['download_bytes'] for e in log))
    if not log:
        rec['problems'].append('no launch: nothing ran on the device')
    if rec['fallbacks']:
        rec['status'] = 'unsupported'
        rec['problems'].append(f'host fallback: {rec["fallbacks"]}')
    elif not cmp['ok']:
        rec['status'] = 'numeric-error'
        rec['problems'].append(cmp['detail'])
    elif not log:
        rec['status'] = 'unsupported'
    else:
        rec['status'] = 'pass'
    if rec['status'] == 'pass' and rec['bound'] and cycles:
        rec['eta'] = B.eta(rec['bound']['cycles'], cycles)
        if rec['eta'] > 1:
            rec['problems'].append(f'eta {rec["eta"]:.3g} > 1: the demand is overcounted or the count is wrong')
    return rec


def load_case(path):
    spec = importlib.util.spec_from_file_location(f'_case_{Path(path).stem}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'make'):
        raise SystemExit(f'{path}: defines no make()')
    return mod


def run_case(path, profile, opt_level):
    mod = load_case(path)
    module, inputs = mod.make()
    rec = measure(module, tuple(inputs), profile=profile, opt_level=opt_level,
                  tol=getattr(mod, 'TOL', None), model=bool(getattr(mod, 'MODEL', False)),
                  demand=getattr(mod, 'DEMAND', None),
                  fp_ops=float(getattr(mod, 'FP_OPS', 0.0)), sfu_ops=float(getattr(mod, 'SFU_OPS', 0.0)))
    rec['case'] = str(path)
    return rec


def summary(rec):
    s = f"{rec['status']:14} {Path(rec['case']).name:28}"
    if rec['time']:
        s += f" {rec['time']['device_cycles']:>9} cycles  {rec['time']['launches']} launch(es)"
    if rec['correctness'] and rec['correctness']['max_rel_err'] is not None:
        s += f"  err {rec['correctness']['max_rel_err']:.2e} (tol {rec['tol']:g})"
    if rec['eta'] is not None:
        s += f"  eta {rec['eta']:.3f} ({rec['bound']['binding']} binds, L {rec['bound']['cycles']:.0f} per core)"
    if rec['problems']:
        s += '\n' + ''.join(f"               {p}\n" for p in rec['problems']).rstrip('\n')
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cases', nargs='+', help='case files, each with make()')
    ap.add_argument('--profile', default=os.environ.get('TNCA_PROFILE', 'core-sim'))
    ap.add_argument('--opt-level', type=int, default=int(os.environ.get('TNCA_OPT_LEVEL', '2')))
    ap.add_argument('--json', type=Path, help='append one JSON record per case to this file')
    args = ap.parse_args(argv)
    bad = 0
    for path in args.cases:
        rec = run_case(path, args.profile, args.opt_level)
        print(summary(rec), flush=True)
        bad += rec['status'] != 'pass'
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, 'a') as f:
                f.write(json.dumps(rec, default=str) + '\n')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
