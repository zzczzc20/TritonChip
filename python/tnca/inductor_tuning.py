"""Bounded compilation support for the public TNCA Inductor selector.

Candidate compilation stays outside the live device process. Device-event
measurement and selection must still validate the common Session identity;
success here is only compiler evidence, never a performance verdict.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

from .autotune import CandidateFailed, CandidateTimedOut, run_measurement

_BUDGETS = {}
SELECTIONS = []


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _software_sha():
    import tnca_cc
    digest = hashlib.sha256()
    for root in (Path(__file__).parent, Path(tnca_cc.__file__).parent):
        for path in sorted(root.rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0')
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _hardware_identity(session):
    from .backend import needs_hardware
    from .operating_clock import from_environment
    if not needs_hardware(session.backend_name):
        raise ValueError('no qualified physical device clock; model wall time cannot select configurations')
    if getattr(session.dev, 'bounded_launch_limits', False) is not True:
        raise ValueError('backend cannot bound candidate launches')
    if getattr(session.dev, 'device_clock', False) is not True:
        raise ValueError('backend common-clock endpoints are disabled')
    path = getattr(getattr(session.dev, 'link', None), 'xclbin', None)
    if not path:
        raise ValueError('loaded physical image identity unavailable')
    image = _file_sha(path)
    clock = from_environment(image)
    if not clock:
        raise ValueError('image-bound operating clock evidence required for tuning')
    window = getattr(session.dev, 'axi_window', None)
    if not isinstance(window, dict):
        raise ValueError('actual device memory map unavailable')
    return dict(image=image, clock=clock, memory_map=window,
                capabilities=session.dev.capabilities())


def choose(tuner, args, *, stream, kwargs):
    """Public CachingAutotuner runtime selection, disabled unless TNCA_TUNE=1.

    Compile and rank a bounded set of standard Inductor retiles. Only
    qualified physical common-clock endpoints can rank them. Unsupported
    execution domains, aliasing, mutations and capture retain the baseline.
    A device fault is propagated; live tensors never silently change epochs.
    """
    import uuid
    import torch
    import triton
    from .autotune import GraphBudget, PersistentCache, DeviceSamples, select, measure_native
    from .torch_device import MEMORY
    from .torch_device.graph import active_capture
    from .errors import DeviceLost, SessionPoisoned
    from .inductor import _code_bytes, WIDE_CODE_LIMIT
    if not tuner.launchers:
        tuner.precompile()
    baseline = getattr(tuner, '_tnca_baseline_result', tuner.compile_results[0])
    tuner._tnca_baseline_result = baseline
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    signature = tuple((tuple(a.shape), tuple(a.stride()), str(a.dtype),
                       a.untyped_storage().data_ptr(), a.storage_offset())
                      if isinstance(a, torch.Tensor) else
                      ((type(a).__name__, repr(a)) if type(a) in (int, float, bool, type(None))
                       else ('unsupported', type(a).__module__, type(a).__qualname__)) for a in args)
    session = MEMORY.session()
    signature = (session.id, session.epoch, signature)
    if getattr(tuner, '_tnca_selected_signature', None) == signature:
        return
    # Restore the conservative launch before considering a new argument set.
    tuner.compile_results = [baseline]
    tuner.launchers = [baseline.make_launcher()]
    tuner._cached_launcher = None
    row = dict(schema='tnca-inductor-selection/1', kernel=tuner.fn.__name__,
               status='conservative', baseline_config=dict(baseline.config.kwargs),
               baseline_compile_seconds=getattr(tuner, '_tnca_baseline_compile_seconds', None))
    SELECTIONS.append(row)
    tuner._tnca_selected_signature = signature
    if (active_capture() is not None or not tensors or
            any(a.device.type != 'tnca' for a in tensors) or
            any(not isinstance(a, torch.Tensor) and type(a) not in (int, float, bool, type(None)) for a in args) or
            len({a.untyped_storage().data_ptr() for a in tensors}) != len(tensors) or
            tuner.mutated_arg_names or tuner.inductor_meta.get('atomic_add_found')):
        row['reason'] = 'capture, mutation, aliasing or unsupported argument metadata'
        return
    seconds = float(os.environ.get('TNCA_TUNE_SECONDS', '600'))
    maximum = int(os.environ.get('TNCA_TUNE_CANDIDATES', '4'))
    # GraphBudget enforces the contract's hard 600s/16-candidate limits.
    graph = (os.getpid(), str(tuner.compile_id))
    if tuner.compile_id is None:
        row['reason'] = 'graph compile identity unavailable'
        return
    budget = _BUDGETS.setdefault(graph, GraphBudget(max_seconds=seconds, max_candidates=maximum))
    try:
        hardware = _hardware_identity(session)
    except (ValueError, OSError) as error:
        row['reason'] = str(error)
        return
    root = Path(os.environ.get('TNCA_TUNE_DIR', '~/.cache/tnca/tune')).expanduser()
    directory = root/'runs'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    row['evidence_directory'] = str(directory)
    identity = dict(kernel_source=_file_sha(tuner.filename),
        shape=[list(a.shape) for a in tensors], stride=[list(a.stride()) for a in tensors],
        dtype=[str(a.dtype) for a in tensors], scalar_arguments=[repr(a) for a in args if not isinstance(a, torch.Tensor)],
        placements=[dict(address=a.untyped_storage().data_ptr(), offset=a.storage_offset()) for a in tensors],
        software_tree=_software_sha(), compiler=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
        profile=session.profile, capabilities=hardware['capabilities'], rtl=dict(image_sha256=hardware['image']),
        memory_map=hardware['memory_map'], image=hardware['image'], clock_hz=hardware['clock']['clock_hz'],
        clock_evidence=hardware['clock']['operating_clock_evidence_sha256'])
    results = list(getattr(tuner, '_tnca_compiled_candidates', [baseline]))
    reasons = []
    configs = getattr(tuner, '_tnca_tuning_configs', [])
    for config in configs:
        if len(results) >= maximum or budget.time_left() <= 0:
            break
        if any(config.kwargs == old.config.kwargs for old in results) or getattr(config, 'pre_hook', None):
            continue
        try:
            result = compile_candidate(tuner, config, budget=budget,
                                       directory=directory/f'compile-{len(reasons)}')
            if _code_bytes(result) > WIDE_CODE_LIMIT:
                raise CandidateFailed('candidate exceeds code admission limit')
            results.append(result)
            reasons.append(dict(config=config.kwargs, status='compiled'))
        except (CandidateFailed, CandidateTimedOut) as error:
            reasons.append(dict(config=config.kwargs, status='refused', reason=str(error)))
    row['compilation'] = reasons
    tuner._tnca_compiled_candidates = results
    if len(results) < 2:
        row['reason'] = 'no alternative legal candidate; baseline retained'
        row['tuning_seconds'] = budget.clock()-budget.started
        (directory/'selection.json').write_text(json.dumps(row, indent=2)+'\n')
        return
    candidates = [dict(knobs=dict(result.config.kwargs),
                       artifact_sha256=hashlib.sha256(result.kernel.asm['tnca']).hexdigest(),
                       resources={'code_bytes': _code_bytes(result)}) for result in results]
    def measure(candidate):
        index = candidates.index(candidate)
        if budget.time_left() <= 0:
            raise CandidateTimedOut('graph budget exhausted before device measurement')
        launcher = results[index].make_launcher()
        session.drain(timeout=budget.time_left())
        samples = []
        for sample in range(3):
            remaining = budget.time_left()
            if remaining <= 0:
                raise CandidateTimedOut('graph budget exhausted between independent samples')
            with session.measurement_scope(wall_seconds=remaining, clock_hz=identity['clock_hz']):
                samples.append(measure_native(lambda: launcher(*args, stream=stream, **kwargs),
                                              warmup=1 if sample == 0 else 0))
        return DeviceSamples(samples)
    try:
        selected = select(identity, candidates, capacities={'code_bytes': WIDE_CODE_LIMIT},
                          capabilities=hardware['capabilities'], budget=budget,
                          measure=measure, cache=PersistentCache(root/'cache'),
                          conservative=candidates[0])
        if selected['budget_exceeded']:
            row['reason'] = 'graph tuning budget exhausted; baseline retained'
        else:
            result = results[candidates.index(selected['selected'])]
            tuner.compile_results = [result]
            tuner.launchers = [result.make_launcher()]
            tuner._cached_launcher = None
            row.update(status='selected', selection=selected)
    except (DeviceLost, SessionPoisoned):
        row.update(status='device-lost', reason='explicit recovery and recreation of input handles required')
        raise
    except ValueError as error:
        row['reason'] = str(error)
    finally:
        row['tuning_seconds'] = budget.clock()-budget.started
        (directory/'selection.json').write_text(json.dumps(row, indent=2)+'\n')


def compile_candidate(tuner, config, *, budget, directory):
    """Compile one legal-retile candidate under the graph's remaining budget.

    A killed worker cannot race the parent's resident tensor allocator: it
    never receives tensor handles or opens a Session. Preserve its request,
    compiler log and result for diagnosing refusals and deadline failures.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(tuner.filename).read_text()
    request, output = directory/'request.pickle', directory/'result.pickle'
    request.write_bytes(pickle.dumps((source, tuner.fn.__name__, config),
                                    protocol=pickle.HIGHEST_PROTOCOL))
    remaining = budget.time_left()
    if remaining <= 0:
        budget.exceeded = True
        raise CandidateTimedOut('graph deadline reached before candidate compilation')
    command = [sys.executable, '-m', 'tnca.inductor_tuning_worker', str(request), str(output)]
    started = time.monotonic()
    # A user-registered backend need not be imported inside compiler workers.
    # Resolve its compile partition in the parent, never infer a different
    # partition from the worker's smaller backend registry.
    from .aot import _physical_chunks_default
    env = dict(os.environ, TNCA_PHYSICAL_CHUNKS=str(_physical_chunks_default()))
    record = dict(schema='tnca-tuning-compile/1', status='running',
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  config=dict(config.kwargs), command=command, timeout_seconds=remaining,
                  physical_chunks=int(env['TNCA_PHYSICAL_CHUNKS']))
    def decode(path):
        result = pickle.loads(path.read_bytes())
        if result.config != config and result.config.__dict__ != config.__dict__:
            raise CandidateFailed('candidate compiler returned a different configuration')
        result.kernel.src.fn = tuner.fn
        return result
    try:
        result = run_measurement(command, budget=budget, result_path=output,
                                 log_path=directory/'compiler.log', env=env,
                                 read_result=decode, worker_record=record)
        record.update(status='compiled', returncode=0,
                      result_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        return result
    except Exception as error:
        record['status'] = 'timed-out' if isinstance(error, CandidateTimedOut) else 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['host_compile_seconds'] = time.monotonic()-started
        (directory/'compile.json').write_text(json.dumps(record, indent=2)+'\n')
