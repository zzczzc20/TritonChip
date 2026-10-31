"""Guarded replay of commands captured from standard native Torch launches.

Capture retains every tensor storage, including intermediates. Replay checks
all guards before admitting any command; it uses the runtime FIFO and actual
completion events. CPU numerical work and staging cannot be captured.
"""
import contextlib
import copy
import threading
import time

_local = threading.local()


def active_capture():
    return getattr(_local, 'capture', None)


def _signature(value):
    import torch
    from . import storage_address
    if isinstance(value, torch.Tensor):
        if value.device.type != 'tnca':
            raise RuntimeError('tnca graph capture requires native device tensors')
        return (tuple(value.shape), tuple(value.stride()), str(value.dtype),
                str(value.device), value.storage_offset(), storage_address(value))
    if type(value).__name__ == 'TensorDescriptor' and hasattr(value, 'base'):
        return (_signature(value.base), tuple(value.shape), tuple(value.strides),
                tuple(value.block_shape), getattr(value, 'padding', 'zero'),
                getattr(value, 'round_f32_to_tf32', False))
    if isinstance(value, (int, float, bool)):
        return (type(value).__name__, value)
    raise RuntimeError(f'tnca graph capture cannot guard {type(value).__name__}')


class Graph:
    def __init__(self):
        self.commands = []
        self.ready = False
        self.invalid_reason = None

    def _append(self, command, entry):
        session, module, bound, grid, values = command
        signatures = tuple(_signature(v) for v in values)
        if self.commands and self.commands[0][0] is not session:
            raise RuntimeError('one captured graph must use one device session')
        template = {k: v for k, v in entry.items() if not k.startswith('_')}
        self.commands.append((session, module, dict(bound), grid, values,
                              signatures, session.epoch, template))

    def reset(self):
        if active_capture() is self:
            raise RuntimeError('cannot reset a graph during capture')
        self.commands.clear()
        self.ready = False
        self.invalid_reason = None

    def locality_plan(self):
        """Inspect conservative storage/alias/lifetime policy without device work."""
        from tnca.locality import plan_commands
        if not self.ready or self.invalid_reason:
            raise RuntimeError('locality requires a valid captured graph')
        # Tensor guards are checked independently of runtime handle guards.
        for session,module,bound,grid,values,signatures,epoch,template in self.commands:
            if tuple(_signature(v) for v in values)!=signatures:
                raise RuntimeError('locality graph shape, stride or storage changed')
        return plan_commands(self.commands)

    def replay(self, *, locality=False):
        from tnca.runtime import MAX_UNFINISHED
        from tnca.triton_backend.driver import _active_logs, TNCALaunchError
        from . import _DeviceModule
        if not self.ready or self.invalid_reason:
            raise RuntimeError(f'tnca graph is invalid: {self.invalid_reason or "not captured"}')
        if active_capture() is not None:
            raise RuntimeError('cannot replay during capture')
        # Validate the entire graph before the first admission, so a changed
        # alias/shape/epoch never causes a partly replayed graph.
        try:
            for session, module, bound, grid, values, signatures, epoch, template in self.commands:
                if session.closed or session.poisoned or session.epoch != epoch or not module.alive:
                    raise RuntimeError('stale graph session, module or recovery epoch')
                if tuple(_signature(v) for v in values) != signatures:
                    raise RuntimeError('graph shape, stride or storage alias changed')
                session._resolve(module, bound)
        except Exception as exc:
            self.invalid_reason = str(exc)
            raise RuntimeError(f'tnca graph guard failed: {exc}') from exc
        if locality:
            from tnca.locality import replay_with_locality
            return replay_with_locality(self, lambda: self.replay())
        last = None
        for session, module, bound, grid, values, signatures, epoch, template in self.commands:
            while len(session.unfinished) >= MAX_UNFINISHED:
                oldest = session.unfinished[0]
                session.wait(oldest)
                if not oldest.ok:
                    raise TNCALaunchError(f'replay predecessor failed: {oldest.error}')
            previous = session.latest_event
            start = time.perf_counter()
            last = session.submit(module, bound, grid=grid,
                                  depends_on=[previous] if previous else [])
            last.stream_id = _DeviceModule.current_stream().stream_id
            entry = copy.deepcopy(template)
            entry.update(replay=True, stream_id=last.stream_id,
                         device_cycles=None, program_cycles=[], counters=None,
                         host_seconds=dict(upload=round(time.perf_counter() - start, 6),
                                           run=0., download=0.),
                         _event=last, _session=session)
            for log in _active_logs():
                log.append(entry)
        return last


@contextlib.contextmanager
def graph(captured):
    from . import MEMORY
    if not isinstance(captured, Graph):
        raise TypeError('tnca.graph expects a tnca.Graph')
    if active_capture() is not None:
        raise RuntimeError('nested tnca graph capture is unsupported')
    MEMORY.synchronize()
    captured.reset()
    _local.capture = captured
    try:
        yield captured
        MEMORY.synchronize()
        if not captured.commands:
            raise RuntimeError('capture contains no native TNCA commands')
        captured.ready = True
    except BaseException:
        captured.commands.clear()
        captured.ready = False
        raise
    finally:
        _local.capture = None
