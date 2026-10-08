"""TNCA: torch.compile -> Inductor -> Triton -> tnca plugin -> Session.

The whole torch-level path, unchanged from what a user runs
(doc/r38_pytorch_p1plus.md): `tnca.inductor.configure()`, then
`torch.compile`. The device is whichever runtime backend the environment
names (python/tnca/triton_backend/driver.py):

    sim-file      the RTL simulator (default)            evidence measured-rtl
    fpga-u50-sim  the board's register protocol on RTL   evidence measured-rtl
    fpga-u50      the Alveo U50 board                    evidence measured-board

Time is the sum over every kernel launch of the launch's device cycles
(`serial-grid`: programs of a grid run one after another, so this is not a
makespan of a concurrent device). Host transfers -- every launch uploads its
argument storages and downloads them back in Stage 1 -- are reported beside
it and are not device time. The first call compiles and runs; its launches
are the measurement, since a TNCA launch is deterministic and the simulator
is too slow to repeat it for nothing.
"""
import os
import time

from ..bounds import VERSION, tnca_bound, tnca_peak
from .common import base_record, classify, compare, error_text, reference

EVIDENCE = {'sim-file': 'measured-rtl', 'sim-socket': 'measured-rtl',
            'fpga-u50-sim': 'measured-rtl', 'fpga-u50': 'measured-board',
            'fpga-u280': 'measured-board'}

# The clock a cycle count may be converted with, by backend. Only a board
# has one; a simulated cycle has no duration (Stage 1 rule).
BOARD_CLOCK_HZ = {'fpga-u50': 37.5e6,
                  # 150 MHz kernel clock / 4, measured: with the image loaded
                  # (2026-10-08) the shell's frequency counter
                  # (clock_freq_counter.m.*/clock_counter_freqs in sysfs on the
                  # management function) reads 150 MHz for the kernel clock,
                  # to its 1 MHz resolution.
                  'fpga-u280': 37.5e6}

# The device heap a launch's argument storages must fit: Session's default
# heap, 0x20000..0x80000, of the 512 KiB memory the simulator and the U50
# image both have (python/tnca/runtime.py, rtl/fpga/tnca_fpga_mem.v WORDS).
HEAP_BYTES = 0x80000 - 0x20000


def run(point, *, profile='core-sim', opt_level=2, seed=0, counters=False):
    os.environ['TNCA_PROFILE'] = profile
    os.environ['TNCA_OPT_LEVEL'] = str(opt_level)
    if counters:
        os.environ['TNCA_COUNTERS'] = '1'
    import torch
    # Triton first: importing it discovers its backends, which imports the
    # tnca driver. Importing the driver first would hand Triton a
    # half-initialised module with no DriverBase subclass in it yet.
    import triton                                            # noqa: F401
    import tnca.inductor
    from tnca.triton_backend.driver import backend_choice, launch_log
    from torch._inductor.utils import run_and_get_code

    config = tnca.inductor.configure()
    backend, backend_kwargs = backend_choice()
    plat = dict(kind='tnca', name=f'tnca:{profile}@{backend}', profile=profile, backend=backend,
                backend_args=backend_kwargs, opt_level=opt_level, inductor=config)
    rec = base_record(point, plat, EVIDENCE.get(backend, 'measured-rtl'))

    compile_only = os.environ.get('TNCA_LAUNCH') == 'skip'
    try:
        # A compile-only sweep compares nothing, so it computes no reference
        # (for the large tier that is a 1B-parameter model on the CPU).
        ref = None if compile_only else reference(point, seed)
    except Exception as e:                                   # noqa: BLE001
        rec.update(status='not-measured', problems=[f'reference failed: {error_text(e)}'])
        return rec

    torch._dynamo.reset()
    m, inputs = point.make(seed)
    m = m.eval()
    t0 = time.monotonic()
    log = []
    try:
        with torch.no_grad(), launch_log() as log:
            got, code = run_and_get_code(torch.compile(m), *inputs)
    except Exception as e:                                   # noqa: BLE001
        rec.update(status=classify(e), problems=[error_text(e)],
                   host=dict(wall_s=round(time.monotonic() - t0, 2), launches=len(log)))
        return rec
    wall = time.monotonic() - t0

    rec['fallbacks'] = tnca.inductor.host_fallbacks(code)
    rec['kernels'] = tnca.inductor.triton_kernels(code)
    # The launches the generated wrapper makes. Anything beyond them -- an
    # autotuner timing candidates by launching them -- is not the graph's
    # work, and a measurement that contains it is refused below.
    expected = _wrapper_launches(code)
    if not compile_only:
        cmp = compare(got, ref, point.tolerance)
    else:
        cmp = dict(ok=False, detail='not run', max_rel_err=None)
    rec['correctness'] = dict(max_rel_err=cmp['max_rel_err'], tol=point.tolerance, detail=cmp['detail'])

    if compile_only:
        # Compile-only: every kernel built, none ran. Not a measurement.
        biggest = max((e['upload_bytes'] for e in log), default=0)
        rec.update(status='not-measured', correctness=None,
                   problems=['compile-only (TNCA_LAUNCH=skip): all kernels compiled, none launched'],
                   host=dict(wall_s=round(wall, 2), launches=len(log), expected_launches=expected,
                             max_launch_bytes=biggest,
                             max_launch_kernel=max(log, key=lambda e: e['upload_bytes'])['kernel'] if log else None))
        if rec['fallbacks']:
            rec['status'] = 'unsupported'
            rec['problems'].append(f'host fallback: {rec["fallbacks"]}')
        if expected is not None and len(log) != expected:
            rec['status'] = 'runtime-fault'
            rec['problems'].append(f'{len(log)} launches for {expected} in the wrapper (autotuning?)')
        heap = HEAP_BYTES
        if biggest > heap:
            rec['status'] = 'capacity-exceeded'
            rec['problems'].append(f'the largest launch needs {biggest} bytes of arguments; '
                                   f'the device heap is {heap}')
        return rec

    cycles = sum(e['device_cycles'] or 0 for e in log)
    clock = BOARD_CLOCK_HZ.get(backend)
    rec['time'] = dict(unit='cycles', device=cycles or None, timing='serial-grid',
                       roi='launch_accept_to_core_idle', launches=len(log),
                       per_launch=[dict(kernel=e['kernel'], grid=e['grid'], cycles=e['device_cycles'],
                                        programs=len(e['program_cycles'])) for e in log],
                       clock_hz=clock, device_seconds=(cycles / clock) if (clock and cycles) else None)
    rec['host'] = dict(wall_s=round(wall, 2),
                       upload_bytes=sum(e['upload_bytes'] for e in log),
                       download_bytes=sum(e['download_bytes'] for e in log),
                       transfer_s=round(sum(e['host_seconds']['upload'] + e['host_seconds']['download']
                                            for e in log), 3))
    if counters:
        tot = {}
        for e in log:
            c = e.get('counters') or {}
            for k, v in (c.get('total') or {}).items():
                tot[k] = tot.get(k, 0) + v
        rec['counters'] = tot or None

    problems = []
    if not log:
        problems.append('no kernel was launched on the device')
    if expected is not None and len(log) != expected:
        problems.append(f'{len(log)} launches, but the generated wrapper makes {expected}: '
                        f'launches that are not the graph (autotuning?) would be counted as its time')
    if rec['fallbacks']:
        problems.append(f'host fallback: {rec["fallbacks"]}')
    if not cmp['ok']:
        problems.append(cmp['detail'])
    rec['problems'] = problems
    if rec['fallbacks'] or not log:
        rec['status'] = 'unsupported'
    elif expected is not None and len(log) != expected:
        rec['status'] = 'not-measured'
    elif not cmp['ok']:
        rec['status'] = 'numeric-error'
    else:
        rec['status'] = 'pass'

    L, binding, terms = tnca_bound(point, profile)
    rec['bound'] = dict(version=VERSION, unit='cycles', value=L, binding=binding, terms=terms,
                        peak=tnca_peak(profile)['rates'])
    if rec['status'] == 'pass' and cycles:
        rec['eta'] = L / cycles
        if rec['eta'] > 1:
            rec['problems'].append(f'AUDIT: eta {rec["eta"]:.4f} > 1 -- a measurement below its strict bound')
    return rec


def _wrapper_launches(code):
    """Kernel launches in Inductor's generated `call`, or None if the wrapper
    has a loop (then the count is not a static property of the text)."""
    import re
    total = 0
    for src in code:
        body = src.split('def benchmark_compiled_module', 1)[0]
        if 'def call(' in body:
            body = body.split('def call(', 1)[1]
        lines = [l for l in body.splitlines() if not l.lstrip().startswith('#')]
        if any(re.match(r'\s*(for|while)\b', l) for l in lines):
            return None
        total += sum(len(re.findall(r'\btriton_\w+\.run\(', l)) for l in lines)
    return total
