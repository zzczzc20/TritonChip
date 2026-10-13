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
it and are not device time.

Protocol 2 (archbench/PROTOCOL.md, R40-00) separates the cold first call --
compilation, any autotuning and the first run -- from the measurement:
`warmup` calls follow it, then `timed` calls, each one sample. The samples
are per call: `device-cycles-serial-grid` (the call's launches' device
cycles) and, when there are timed calls, `request-latency` (host monotonic
time around the call; on this CPU-tensor path it includes every launch's
upload and download). With no timed call -- the simulator, where a launch is
deterministic and too slow to repeat for nothing -- the first call's device
cycles are the one sample, marked deterministic; there is then no latency.
The record carries its provenance (archbench/provenance.py): the software
tree, every artifact's sha256, the profile and its capabilities, the image
and clock, the launch map, and with `bundle` the files of the chain.

With backend_args.device_clock=true on an image providing MMIO clock
endpoints, timed calls additionally report device-graph-makespan from first
accepted launch to last completion, including submission gaps. Missing
endpoints leave this measurement unavailable; serial-grid keeps its meaning.
"""
import os
import time

from .. import provenance as _prov
from ..bounds import VERSION, tnca_bound, tnca_peak
from .common import SCHEMA2, base_record, classify, compare, error_text, reference

EVIDENCE = {'sim-file': 'measured-rtl', 'sim-socket': 'measured-rtl',
            'fpga-u50-sim': 'measured-rtl', 'fpga-u50': 'measured-board',
            'fpga-u280': 'measured-board'}

# The clock a cycle count may be converted with, by backend. Only a board
# has one; a simulated cycle has no duration (Stage 1 rule).
# The device clock of the image LOADED on each board -- a property of the
# bitstream, not of this checkout. The U280 runs a 100 MHz undivided image
# since 2026-10-09 (rtl/board/tnca_u50_xo.v DEV_DIV = 1, kernel clock 100 MHz);
# the U50's was built with 4. Change an entry in the same step that loads a
# new image.
BOARD_CLOCK_HZ = {'fpga-u50': 37.5e6,
                  # 100 MHz kernel clock, undivided, measured: with the image
                  # loaded (2026-10-09 12:05 HKT) the shell's frequency counter
                  # (clock_freq_counter.m.*/clock_counter_freqs in sysfs on the
                  # management function) reads 100 MHz, to its 1 MHz
                  # resolution, and the device's own cycle count over
                  # host-timed spin kernels gives 100.01-100.07 MHz
                  # (flow/fpga/measure_clock.py). Earlier the same day: /2 of
                  # 150, 75 MHz; until R38's timing work, /4, 37.5 MHz.
                  'fpga-u280': 100e6}
# How each board's clock is set (a configuration field of protocol-2 records:
# two runs at different clock policies are different configurations).
BOARD_CLOCK_POLICY = {'fpga-u50': '150MHz/4', 'fpga-u280': '100MHz-undivided'}

# The device heap a launch's argument storages must fit: Session's default
# heap, 0x20000..0x80000, of the 512 KiB memory the simulator and the U50
# image both have (python/tnca/runtime.py, rtl/fpga/tnca_fpga_mem.v WORDS).
HEAP_BYTES = 0x80000 - 0x20000


# The TNCA_* environment variables that do NOT change what is compiled or how
# it runs: the transport, timeouts, and what the harness itself sets from its
# own arguments (which the record already holds). Every other TNCA_* variable
# is an ablation knob and is recorded (R39-00).
NOT_KNOBS = frozenset({'TNCA_BACKEND', 'TNCA_BACKEND_ARGS', 'TNCA_HOST_TIMEOUT',
                       'TNCA_DEADLINE', 'TNCA_LAUNCH', 'TNCA_PROFILE', 'TNCA_OPT_LEVEL',
                       'TNCA_COUNTERS', 'TNCA_TRITON', 'TNCA_DEVICE',
                       # the provenance bundle's kernel dump (doc/dumping_ir.md):
                       # it writes files, the artifact is byte-identical
                       'TNCA_KERNEL_DUMP', 'TNCA_DUMP_DIR'})


def knobs():
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith('TNCA_') and k not in NOT_KNOBS}


def device_id(profile, backend, backend_kwargs):
    """What the device actually was: the simulator build, named by its
    profile and RTL hashes, or the board image, named by its sha256. Two
    runs with the same profile name on different RTL must not be merged."""
    import hashlib
    try:
        xclbin = (backend_kwargs or {}).get('xclbin')
        if xclbin:
            h = hashlib.sha256()
            with open(xclbin, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            return f'xclbin:{h.hexdigest()[:16]}'
        from tnca import device
        return f'sim:{os.path.basename(os.path.dirname(device.expected_path(profile)))}'
    except Exception as e:                                   # noqa: BLE001
        return f'unknown:{type(e).__name__}'


def run(point, *, profile='core-sim', opt_level=2, seed=0, counters=False, warmup=0, timed=0,
        bundle=None):
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
    # R40-01: TNCA_TORCH_DEVICE=native puts the module and its inputs on the
    # `tnca` PyTorch device (tnca.torch_device) -- storage in device memory,
    # Inductor generating for device "tnca" -- and copies the outputs back
    # explicitly for the comparison. Unset (or `cpu`), the B39 path: CPU
    # tensors, every launch staging its arguments.
    native = os.environ.get('TNCA_TORCH_DEVICE', 'cpu') == 'native'
    if native:
        import tnca.torch_device
        tnca.torch_device.register()
        config = dict(config, device='tnca')
    backend, backend_kwargs = backend_choice()
    plat = dict(kind='tnca', name=f'tnca:{profile}@{backend}', profile=profile, backend=backend,
                backend_args=backend_kwargs, opt_level=opt_level, inductor=config,
                knobs=knobs(), device_id=device_id(profile, backend, backend_kwargs))
    # The configuration a protocol-2 comparison checks, on every record --
    # one that fails as much as one that passes (tools/r40/ablate.py `mixed`).
    plat.update(image=plat.get('device_id'), clock_policy=BOARD_CLOCK_POLICY.get(backend),
                software_tree=_prov.software_tree())
    rec = base_record(point, plat, EVIDENCE.get(backend, 'measured-rtl'))
    rec['schema'] = SCHEMA2

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
    if native:
        try:
            m = m.to('tnca')
            inputs = tuple(x.to('tnca') if isinstance(x, torch.Tensor) else x for x in inputs)
        except Exception as e:                               # noqa: BLE001
            rec.update(status=classify(e), problems=[f'to("tnca"): {error_text(e)}'])
            return rec
    compiled = torch.compile(m, fullgraph=native)
    if bundle:
        # Every kernel compiled here, none taken from an earlier cache, so the
        # dump (TTIR, TNCA IR, assembly) and Triton's own cache entries are
        # the chain of THIS record.
        os.environ['TNCA_KERNEL_DUMP'] = '1'
        os.environ['TNCA_DUMP_DIR'] = os.path.join(bundle, 'kernels')
        os.environ['TRITON_CACHE_DIR'] = os.path.join(bundle, 'triton_cache')
    graph = _prov.GraphCapture() if bundle else None
    t0 = time.monotonic()
    log = []
    try:
        with torch.no_grad(), launch_log() as log:
            if graph:
                with graph:
                    got, code = run_and_get_code(compiled, *inputs)
            else:
                got, code = run_and_get_code(compiled, *inputs)
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
        # Native storage uses the active Session's real heap. The frozen
        # CPU migration path retains its original on-chip staging limit.
        heap = (tnca.torch_device.MEMORY.session().memory_stats['heap_capacity_bytes']
                if native else HEAP_BYTES)
        if native:
            rec['host']['heap_capacity_bytes'] = heap
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

    # ---- protocol 2: the measurement proper, and where it came from -------
    rec['schema'] = SCHEMA2
    first = dict(wall_s=round(wall, 3), device_cycles=cycles or None)
    calls = []
    if rec['status'] == 'pass' and (warmup or timed):
        got = None               # the first call's outputs are not held across the timed calls
        try:
            calls = _steady(compiled, inputs, warmup + timed, expected, launch_log, torch)
        except Exception as e:                               # noqa: BLE001
            rec['status'] = classify(e)
            rec['problems'].append(f'steady calls: {error_text(e)}')
            calls = []
        bad = [c for c in calls if c.get('problem')]
        if bad:
            rec['status'] = 'not-measured'
            rec['problems'].append(f'steady call {bad[0]["index"]}: {bad[0]["problem"]}')
        elif not compare(calls[-1].pop('out'), ref, point.tolerance)['ok']:
            rec['status'] = 'numeric-error'
            rec['problems'].append('the last timed call does not match the reference')
        for c in calls:
            c.pop('out', None)
    steady = [c for c in calls[warmup:]] if calls else []
    rec['timings'] = _timings(cycles, steady, warmup, clock)
    rec['time'].update(kind='device-cycles-serial-grid',
                       samples=rec['timings']['device-cycles-serial-grid']['samples'],
                       deterministic=not steady, warmup=warmup, timed=len(steady),
                       first_call=first)
    if steady:
        rec['time']['device'] = rec['timings']['device-cycles-serial-grid']['median']
        rec['time']['device_seconds'] = rec['time']['device'] / clock if clock else None
    rec['host'].update(first_call_s=first['wall_s'],
                       steady=_host_steady(steady) if steady else None,
                       # the raw per-call breakdown, timed calls only
                       calls=[{k: round(c[k], 6) for k in ('request_s', 'upload_s', 'run_s',
                                                          'download_s')} for c in steady] or None)
    rec['provenance'] = _provenance(profile, backend, backend_kwargs, plat, log, clock)
    if bundle:
        rec['provenance']['bundle'] = _prov.write_bundle(
            bundle, point_id=point.id, graph_text=graph.text, wrapper_code=code,
            launches=log, dump_dir=None)
    return rec


def _steady(compiled, inputs, n, expected, launch_log, torch):
    """`n` calls after the first, each with its launches and host time."""
    out = []
    got = None
    for i in range(n):
        got = None               # free the last call's outputs before the next
        with torch.no_grad(), launch_log() as lg:
            t = time.monotonic()
            got = compiled(*inputs)
        # launch_log waits for queued TNCA events at graph completion. Keep
        # that wait inside request latency when submission is asynchronous.
        dt = time.monotonic() - t
        c = dict(index=i, request_s=dt, launches=len(lg),
                 device_cycles=sum(e['device_cycles'] or 0 for e in lg),
                 upload_s=sum(e['host_seconds']['upload'] for e in lg),
                 run_s=sum(e['host_seconds']['run'] for e in lg),
                 download_s=sum(e['host_seconds']['download'] for e in lg))
        from tnca.device_timing import clock_span
        c['device_timing'] = clock_span([e.get('device_timing') for e in lg])
        if expected is not None and len(lg) != expected:
            c['problem'] = f'{len(lg)} launches, the wrapper makes {expected}'
        elif not c['device_cycles']:
            c['problem'] = 'no device time'
        if i == n - 1:
            c['out'] = got
        out.append(c)
    return out


def _median(xs):
    s = sorted(xs)
    k = len(s)
    m = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
    return int(m) if isinstance(m, float) and m.is_integer() and all(isinstance(x, int) for x in s) else m


def _p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def _timings(first_cycles, steady, warmup, clock):
    """Every timing kind this run measured, each with its own samples."""
    out = {}
    if steady:
        cyc = [c['device_cycles'] for c in steady]
        out['device-cycles-serial-grid'] = dict(unit='cycles', samples=cyc, median=_median(cyc),
                                                p95=_p95(cyc), warmup=warmup, deterministic=False,
                                                clock_hz=clock)
        if all(c.get('device_timing') is not None for c in steady):
            spans = [c['device_timing']['cycles'] for c in steady]
            out['device-graph-makespan'] = dict(
                unit='cycles', samples=spans, median=_median(spans), p95=_p95(spans),
                warmup=warmup, clock_hz=clock, deterministic=False,
                boundaries='first accepted launch to last device completion',
                includes=['device execution', 'submission gaps', 'backpressure'],
                instrumentation='sequence-checked MMIO endpoints after each program')
        lat = [round(c['request_s'], 6) for c in steady]
        out['request-latency'] = dict(unit='s', samples=lat, median=_median(lat), p95=_p95(lat),
                                      warmup=warmup,
                                      includes=['per-launch upload', 'device', 'per-launch download',
                                                'host dispatch'])
    else:
        cyc = [first_cycles] if first_cycles else []
        out['device-cycles-serial-grid'] = dict(unit='cycles', samples=cyc,
                                                median=first_cycles or None, p95=first_cycles or None,
                                                warmup=0, deterministic=True, clock_hz=clock,
                                                note='the first call: device time excludes compilation')
    return out


def _host_steady(steady):
    return {k: round(_median([c[k] for c in steady]), 6)
            for k in ('request_s', 'upload_s', 'run_s', 'download_s')}


def _provenance(profile, backend, backend_kwargs, plat, log, clock):
    try:
        from tnca import device as _device
        from tnca.adapters import r1
        target, _info = r1.target_for(_device.base_profile(profile))
        caps = target.capabilities if isinstance(target.capabilities, dict) else dict(target.capabilities)
        capabilities = sorted(k for k, v in caps.items() if v)
    except Exception as e:                                   # noqa: BLE001
        capabilities = None
        plat.setdefault('problems', []).append(f'capabilities: {e}')
    arts = sorted({e.get('artifact') for e in log if e.get('artifact')})
    seen, launches = set(), []
    for e in log:
        launches.append(dict(kernel=e['kernel'], grid=list(e['grid']),
                             programs=len(e.get('program_cycles') or []), artifact=e.get('artifact')))
        seen.add(e['kernel'])
    return dict(software_tree=_prov.software_tree(), git=_prov.git_commit(),
                artifacts=arts, profile=profile, capabilities=capabilities,
                image=plat.get('device_id'), clock_hz=clock, backend=backend,
                launch_map=dict(timing='serial-grid', cores=[0], launches=launches))


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
