"""TNCA as a Triton backend: the launch half.

A compiled kernel is a `.tnca` artifact (see compiler.py), and launching it
is `tnca.runtime.Session` -- the one runtime every other path uses. What this
file adds is the translation Triton needs: its launcher receives the Python
arguments of a `kernel[grid](...)` call, and a TNCA device does not share
memory with the host.

Tensor residency, Stage 1: arguments are host (CPU) torch tensors. For each
distinct storage among the arguments the launcher uploads the storage to a
device buffer, passes (buffer, storage_offset) for every tensor viewing it,
launches the whole grid, waits, and downloads the storage back. Storages are
de-duplicated so two views of one tensor stay one buffer on the device, and a
region the kernel does not write comes back unchanged because it was
uploaded. This is an explicit host round trip per launch; keeping tensors on
the device is the PrivateUse1 device (roadmap P2) and not this.

One Session per process, profile and backend, opened on first launch. The
backend is the runtime's (tnca.backend registry), chosen by environment so
that the same torch program reaches any of them unchanged:

    TNCA_BACKEND       sim-file (default) | sim-socket | fpga-u50-sim | fpga-u50
    TNCA_BACKEND_ARGS  JSON object of keyword arguments the backend needs and
                       the runtime cannot know, e.g. {"xclbin": "/path/x.xclbin"}
                       for the board (Session passes them through untouched)
    TNCA_DEADLINE      device cycles one launch may take (default 5e6)
    TNCA_HOST_TIMEOUT  wall seconds the host waits for one launch (default 600;
                       a simulator runs ~500 target cycles per second)
    TNCA_LAUNCH=skip   compile every kernel and run none (coverage sweeps);
                       the launch log marks each entry `skipped`
    TNCA_GUARD_BYTES, TNCA_GUARD_CHECK
                       debugging: poisoned margins around every buffer, to
                       catch out-of-bounds writes and reads (_guards)

`launch_log()` records every launch made while it is open -- kernel, grid,
device cycles, per-program cycles, counters, bytes moved between host and
device -- which is how a caller measures a whole torch graph rather than
one kernel.
"""
import contextlib
import json
import os
import tempfile
import threading
import time

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _torch():
    import torch
    return torch


def backend_choice():
    """(backend name, keyword arguments) from TNCA_BACKEND / TNCA_BACKEND_ARGS."""
    name = os.environ.get('TNCA_BACKEND') or 'sim-file'
    raw = os.environ.get('TNCA_BACKEND_ARGS') or '{}'
    try:
        kwargs = json.loads(raw)
    except ValueError as e:
        raise TNCALaunchError(f'TNCA_BACKEND_ARGS is not JSON: {e}') from None
    if not isinstance(kwargs, dict):
        raise TNCALaunchError('TNCA_BACKEND_ARGS must be a JSON object')
    return name, kwargs


class _Sessions:
    lock = threading.Lock()
    open = {}

    @classmethod
    def get(cls, profile):
        from tnca.runtime import Session
        backend, kwargs = backend_choice()
        key = (profile, backend, json.dumps(kwargs, sort_keys=True),
               os.environ.get('TNCA_ACTIVE_CORES'))
        with cls.lock:
            s = cls.open.get(key)
            if s is None or s.closed or s.poisoned:
                s = Session(profile=profile, deadline_cycles=int(os.environ.get('TNCA_DEADLINE', '5000000')),
                            host_timeout=float(os.environ.get('TNCA_HOST_TIMEOUT', '600')),
                            counters=os.environ.get('TNCA_COUNTERS') == '1',
                            backend=backend, **kwargs)
                cls.open[key] = s
            return s

    @classmethod
    def close_all(cls):
        """Close every session at interpreter exit, so each device -- a
        simulator, a board link -- is shut down rather than abandoned."""
        with cls.lock:
            for s in cls.open.values():
                try:
                    if not s.closed:
                        s.close()
                except Exception:                        # noqa: BLE001
                    pass
            cls.open.clear()


import atexit as _atexit                                 # noqa: E402
_atexit.register(_Sessions.close_all)


_log_local = threading.local()


def _active_logs():
    if not hasattr(_log_local, "logs"):
        _log_local.logs = []
    return _log_local.logs


@contextlib.contextmanager
def launch_log(*, isolated=False):
    """Collect one dict per launch made while open (nested logs all see it)."""
    log = []
    previous = _active_logs() if isolated else None
    if isolated:
        _log_local.logs = []
    _active_logs().append(log)
    try:
        yield log
    finally:
        # Restore the outer collection before waiting; async entries already
        # belong to this log and completion does not issue new launches.
        assert _active_logs().pop() is log
        if previous is not None:
            _log_local.logs = previous
        # An asynchronous launch log is measured through completion, never
        # just submission. Nested contexts share each entry's final values.
        for entry in log:
            ev = entry.pop('_event', None)
            if ev is not None:
                session = entry.pop('_session')
                session.wait(ev)
                if not ev.ok:
                    raise TNCALaunchError(f'{entry["kernel"]}: {ev.error}')
                entry['device_cycles'] = ev.device_cycles
                entry['program_cycles'] = list(ev.program_cycles or [])
                entry['device_timing'] = ev.device_timing
                entry['counters'] = ev.counters
                stamps = dict(ev.history)
                entry['host_seconds']['run'] = round(
                    stamps.get('SUCCEEDED', 0) - stamps.get('RUNNING', 0), 6)


class TNCALaunchError(RuntimeError):
    pass


_TORCH_TO_TNCA = {'float32': 'fp32', 'float16': 'fp16', 'bfloat16': 'bf16', 'float64': 'fp64',
                  'int8': 'i8', 'uint8': 'u8', 'int16': 'i16', 'uint16': 'u16',
                  'int32': 'i32', 'uint32': 'u32', 'int64': 'i64', 'uint64': 'u64',
                  'float8_e4m3fn': 'e4m3', 'float8_e5m2': 'e5m2'}


class TNCALauncher:
    """`driver.active.launcher_cls(src, metadata)`: one per compiled kernel."""

    last = None          # the launcher that ran most recently (tests read its cycles)

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata
        self.profile = getattr(metadata, 'tnca_profile', 'core-sim')
        self.params = list(getattr(metadata, 'tnca_params', []))
        # The positions Triton passes and the kernel does not take: constexprs,
        # and arguments the JIT specialised into the code.
        sig = getattr(src, 'signature', {}) or {}
        constants = getattr(src, 'constants', {}) or {}
        names = list(sig) if isinstance(sig, dict) else []
        const_names = set()
        for k in constants:
            if isinstance(k, tuple):
                if len(k) == 1 and isinstance(k[0], int) and k[0] < len(names):
                    const_names.add(names[k[0]])
            elif isinstance(k, int) and k < len(names):
                const_names.add(names[k])
            else:
                const_names.add(k)
        self.keep = [i for i, n in enumerate(names)
                     if sig[n] != 'constexpr' and n not in const_names]
        self.names = names
        self.last_event = None
        self.last_cycles = None

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        torch = _torch()
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        if os.environ.get('TNCA_LAUNCH') == 'skip':
            # Compile-only sweeps (TNCA_LAUNCH=skip): every kernel of a graph
            # is compiled, none is run, and the outputs are whatever the host
            # buffers held. The log says so, so no caller can take the
            # result for a measurement.
            # What the launch WOULD move: one buffer per distinct storage,
            # exactly as below -- the capacity question a board asks.
            storages = {}
            for v in args:
                if isinstance(v, torch.Tensor):
                    st = v.untyped_storage()
                    storages[st.data_ptr()] = st.nbytes()
                elif type(v).__name__ == 'TensorDescriptor' and hasattr(v, 'base'):
                    st = v.base.untyped_storage()
                    storages[st.data_ptr()] = st.nbytes()
            entry = dict(kernel=self.metadata.name, grid=(gridX, gridY, gridZ), profile=self.profile,
                         backend=None, device_cycles=None, program_cycles=[], counters=None,
                         host_seconds=dict(upload=0.0, run=0.0, download=0.0),
                         upload_bytes=sum(storages.values()), download_bytes=0,
                         buffers=len(storages), skipped=True)
            for log in _active_logs():
                log.append(entry)
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)
            return
        # Bind by NAME: the artifact's parameters are the kernel's own names,
        # minus constexprs and anything the JIT specialised away.
        by_name = dict(zip(self.names, args)) if self.names else {}
        import re as _re
        if self.params and all(_re.fullmatch(r'arg\d+', p) for p in self.params) \
                and not any(p in by_name for p in self.params):
            # A TTIR printed without argument names: bind by position over
            # the parameters that survived (constexprs and specialised
            # constants removed), which is the order the function takes.
            kept = [args[i] for i in self.keep]
            if len(kept) != len(self.params):
                raise TNCALaunchError(
                    f'{self.metadata.name}: the kernel takes {len(self.params)} arguments '
                    f'and the launch supplied {len(kept)} after removing constants')
            by_name = dict(zip(self.params, kept))
        missing = [p for p in self.params if p not in by_name]
        if missing:
            raise TNCALaunchError(f'{self.metadata.name}: no launch value for {missing}')
        values = [by_name[p] for p in self.params]
        session, module = _load(_Sessions.get(self.profile), function)
        guard, poisons = _guards()
        grid = (max(gridX, 1), max(gridY, 1), max(gridZ, 1))
        submit_policy = os.environ.get('TNCA_SUBMIT', 'single')
        if submit_policy not in ('single', 'batched'):
            raise TNCALaunchError(f'unsupported TNCA_SUBMIT {submit_policy!r}')
        asynchronous = submit_policy == 'batched'
        runs = [self._launch_once(session, module, values, grid, guard, poison,
                                  asynchronous=asynchronous)
                for poison in poisons]
        event, outputs, moved, (t_start, t_submit, t_done) = runs[-1]
        event.stream_id = getattr(stream, 'stream_id', stream)
        if len(runs) > 1:
            differ = sorted({name for k, (name, _raw, data) in outputs.items()
                             if runs[0][1][k][2] != data})
            if differ:
                raise TNCALaunchError(
                    f'{self.metadata.name}: reads memory outside its buffers -- with two '
                    f'different poisons around every buffer (TNCA_GUARD_CHECK) the launch '
                    f'wrote different results into {differ}')
        for _name, raw, data in outputs.values():
            if data:
                raw.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        from tnca.torch_device.graph import active_capture
        capture = active_capture()
        if _active_logs() or capture is not None:
            entry = dict(kernel=self.metadata.name, grid=grid, profile=self.profile,
                         backend=session.backend_name,
                         stream_id=event.stream_id,
                         artifact=_artifact_sha256(function), artifact_path=function,
                         device_cycles=event.device_cycles,
                         program_cycles=list(event.program_cycles or []),
                         device_timing=event.device_timing,
                         counters=event.counters,
                         host_seconds=dict(upload=round(t_submit - t_start, 6),
                                           run=round(t_done - t_submit, 6),
                                           download=round(time.perf_counter() - t_done, 6)),
                         **moved)
            if asynchronous:
                entry.update(_event=event, _session=session)
            for log in _active_logs():
                log.append(entry)
            if capture is not None:
                capture._append(event._replay_command, entry)
                del event._replay_command
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)

    def _launch_once(self, session, module, values, grid, guard, poison, *, asynchronous=False):
        """Upload every distinct storage, run the grid, download them.

        Returns (event, {storage: (argument name, host bytes view, device
        bytes)}, bytes moved, (start, submit, done) times). Nothing is
        copied back into the host tensors here; the caller does that, once.

        With `guard` > 0 every buffer is allocated `guard` bytes larger on
        each side, the margins filled with `poison`; a margin that comes back
        changed is a write out of bounds and fails the launch.
        """
        torch = _torch()
        buffers, bound = {}, {}
        native_staged = {}
        transfer_policy = os.environ.get('TNCA_TRANSFER_POLICY', 'resident')
        if transfer_policy not in ('resident', 'per-kernel'):
            raise TNCALaunchError(f'unknown TNCA_TRANSFER_POLICY {transfer_policy!r}')
        moved = {'upload_bytes': 0, 'download_bytes': 0}
        t_start = time.perf_counter()

        devices = {a.device.type for a in values if isinstance(a, torch.Tensor)} | \
            {a.base.device.type for a in values if type(a).__name__ == 'TensorDescriptor'
             and hasattr(a, 'base')}
        if len(devices) > 1:
            raise TNCALaunchError(f'{self.metadata.name}: tensors on {sorted(devices)}; a kernel\'s '
                                  'tensors are all on the tnca device or all on the host')
        if asynchronous and (devices - {'tnca'} or guard or transfer_policy != 'resident'):
            raise TNCALaunchError('batched submission requires native resident tensors without staging guards')

        def device_ref(name, v):
            if v.device.type == 'tnca':
                # R40-01: the tensor lives in device memory already. Its own
                # buffer and byte offset are the argument; nothing is copied.
                if guard:
                    raise TNCALaunchError(f'{name}: TNCA_GUARD_* stages host tensors; '
                                          'a tnca tensor is not staged')
                from tnca.torch_device import MEMORY, storage_address
                buf, base_off = MEMORY.locate(storage_address(v))
                if buf.session_id != session.id or buf.epoch != session.epoch:
                    raise TNCALaunchError(f'{name}: its storage belongs to another session or epoch')
                off = base_off + v.storage_offset() * v.element_size()
                if transfer_policy == 'per-kernel':
                    # The native-device staging control uses identical
                    # compiled code. A distinct storage makes one round trip
                    # through host bytes into a temporary device buffer; all
                    # aliases bind to that same buffer with their own offset.
                    if buf.id not in native_staged:
                        shadow = session.allocate(buf.size, name=f'staging:{name}')
                        native_staged[buf.id] = (buf, shadow)
                        data = session.download(buf)
                        session.upload(shadow, data)
                        moved['download_bytes'] += len(data)
                        moved['upload_bytes'] += len(data)
                    buf = native_staged[buf.id][1]
                else:
                    moved['resident_bytes'] = moved.get('resident_bytes', 0) + v.untyped_storage().nbytes()
                return (buf, off) if off else buf
            if v.device.type != 'cpu':
                raise TNCALaunchError(f'{name}: tensor on {v.device}; TNCA takes host or tnca tensors')
            st = v.untyped_storage()
            key = st.data_ptr()
            if key not in buffers:
                raw = torch.empty(0, dtype=torch.uint8).set_(st)
                n = st.nbytes()
                if guard:
                    # One upload of margin, data, poison up to the next 64
                    # bytes, margin: the device takes 8-byte-aligned writes.
                    body = -(-n // 64) * 64
                    handle = session.allocate(body + 2 * guard, name=name)
                    session.upload(handle, poison * guard + raw.numpy().tobytes()
                                   + poison * (body - n + guard))
                    moved['upload_bytes'] += n
                else:
                    handle = session.allocate(max(n, 8), name=name)
                    if n:
                        session.upload(handle, raw.numpy().tobytes())
                        moved['upload_bytes'] += n
                buffers[key] = (handle, raw, name)
            handle = buffers[key][0]
            off = guard + v.storage_offset() * v.element_size()
            return (handle, off) if off else handle

        try:
            for name, v in zip(self.params, values):
                if isinstance(v, torch.Tensor):
                    bound[name] = device_ref(name, v)
                elif type(v).__name__ == 'TensorDescriptor' and hasattr(v, 'block_shape'):
                    # triton.tools.tensor_descriptor.TensorDescriptor -> the
                    # runtime's, which packs the engine's TensorMap.
                    from tnca.runtime import TensorDescriptor as TD
                    dt = _TORCH_TO_TNCA.get(str(v.base.dtype).replace('torch.', ''))
                    if dt is None:
                        raise TNCALaunchError(f'{name}: descriptor dtype {v.base.dtype} is not supported')
                    # On NVIDIA this makes the TMA round FP32 to TF32 on the
                    # way in. TNCA's transfer engine moves bits and its FP32
                    # dot is exact, so honouring the flag would need a
                    # rounding step nothing here performs -- and ignoring it
                    # would return different numbers under the same name.
                    if getattr(v, 'round_f32_to_tf32', False):
                        raise TNCALaunchError(f'{name}: round_f32_to_tf32 is not supported on TNCA '
                                              '(FP32 tensors are transferred and multiplied exactly)')
                    bound[name] = TD(device_ref(name, v.base), list(v.shape), list(v.strides),
                                     list(v.block_shape), dt,
                                     padding=getattr(v, 'padding', 'zero'))
                elif isinstance(v, bool):
                    bound[name] = int(v)
                elif isinstance(v, (int, float)):
                    bound[name] = v
                elif v is None:
                    raise TNCALaunchError(f'{name}: None argument')
                else:
                    raise TNCALaunchError(f'{name}: unsupported argument type {type(v).__name__}')
            from tnca.torch_device.graph import active_capture
            capture = active_capture()
            if capture is not None and (devices != {'tnca'} or guard or transfer_policy != 'resident'):
                raise TNCALaunchError('graph capture requires native resident tensors without guards')
            t_submit = time.perf_counter()
            if asynchronous:
                from tnca.runtime import MAX_UNFINISHED
                while len(session.unfinished) >= MAX_UNFINISHED:
                    oldest = session.unfinished[0]
                    session.wait(oldest)
                    if not oldest.ok:
                        raise TNCALaunchError(f'queued predecessor failed: {oldest.error}')
            previous = session.latest_event if asynchronous else None
            event = session.submit(module, bound, grid=grid,
                                   depends_on=[previous] if previous is not None else [])
            if capture is not None:
                event._replay_command = (session, module, bound, grid, tuple(values))
            if asynchronous:
                self.last_event = event
                self.last_cycles = None
                TNCALauncher.last = self
                return event, {}, moved, (t_start, t_submit, time.perf_counter())
            session.wait(event)
            t_done = time.perf_counter()
            self.last_event = event
            self.last_cycles = event.device_cycles
            TNCALauncher.last = self
            if event.state != 'SUCCEEDED':
                raise TNCALaunchError(f'{self.metadata.name}: {event.error}')
            outputs = {}
            for original, shadow in native_staged.values():
                data = session.download(shadow)
                session.upload(original, data)
                moved['download_bytes'] += len(data)
                moved['upload_bytes'] += len(data)
            for key, (handle, raw, name) in buffers.items():
                n = raw.numel()
                if guard:
                    whole = bytes(session.download(handle))
                    if whole[:guard] != poison * guard or \
                            whole[guard + n:] != poison * (len(whole) - guard - n):
                        raise TNCALaunchError(
                            f'{self.metadata.name}: wrote outside {name!r} (a TNCA_GUARD_BYTES '
                            f'margin of {guard} bytes came back changed)')
                    data = whole[guard:guard + n]
                else:
                    data = session.download(handle)[:n] if n else b''
                outputs[key] = (name, raw, bytes(data))
                moved['download_bytes'] += n
            return event, outputs, moved, (t_start, t_submit, t_done)
        finally:
            for _original, shadow in native_staged.values():
                try:
                    session.free(shadow)
                except Exception:                        # noqa: BLE001
                    pass
            for handle, _raw, _name in buffers.values():
                try:
                    session.free(handle)
                except Exception:                        # noqa: BLE001
                    pass

_loaded = {}
_artifact_hashes = {}
evicted = 0                  # modules unloaded to make code room (resident sessions)


def _artifact_sha256(path):
    """The sha256 of the .tnca a launch ran, for the record's provenance
    (archbench/PROTOCOL.md, protocol 2): what was measured, named by content."""
    h = _artifact_hashes.get(path)
    if h is None and isinstance(path, str) and os.path.exists(path):
        import hashlib
        with open(path, 'rb') as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        _artifact_hashes[path] = h
    return h


def _load(session, path):
    """The module for `path` in this session, loading it if needed.

    Reclaim least-recently-used modules after their launches drain. Preserve
    the device session and clock epoch for both CPU migration and resident
    tensors; replacing a session mid-graph invalidates common-clock spans.
    """
    from tnca.errors import BufferError as _BE
    key = (session.id, session.epoch, path)
    m = _loaded.pop(key, None)
    if m is not None:
        _loaded[key] = m             # most recently used last (eviction order)
        return session, m
    try:
        m = session.load(path)
    except _BE as e:
        if 'overlaps the device heap' not in str(e):
            raise
        session.drain()
        for old_key in list(_loaded):
            if old_key[0] != session.id or old_key[1] != session.epoch:
                continue
            session.unload(_loaded.pop(old_key))
            global evicted
            evicted += 1
            try:
                m = session.load(path)
                break
            except _BE as again:
                if 'overlaps the device heap' not in str(again):
                    raise
        else:
            raise TNCALaunchError(
                'the code region cannot hold this kernel even with every other module '
                'unloaded (capacity-exceeded: code region)') from e
    _loaded[key] = m
    return session, m


def _guards():
    """(guard bytes, poisons) for a launch -- debugging knobs, off by default.

    TNCA_GUARD_BYTES=N  every buffer gets an N-byte margin on each side,
                        filled with 0xFF, and a margin that comes back
                        changed fails the launch (a write out of bounds).
    TNCA_GUARD_CHECK=1  every launch runs twice, margins 0xFF and then 0x00
                        (4096 bytes unless TNCA_GUARD_BYTES says otherwise),
                        and fails if the results differ: the kernel read
                        memory outside its buffers. Without it such a read
                        returns whatever an earlier launch left there, so a
                        wrong answer comes and goes with the run's history.
    """
    raw = os.environ.get('TNCA_GUARD_BYTES')
    check = os.environ.get('TNCA_GUARD_CHECK') == '1'
    guard = int(raw) if raw else (4096 if check else 0)
    if guard % 64:
        raise TNCALaunchError(f'TNCA_GUARD_BYTES={guard}: must be a multiple of 64 (buffer alignment)')
    if check:
        return guard, (b'\xff', b'\x00')
    return guard, ((b'\xff',) if guard else (None,))

class TNCAUtils:
    def __init__(self):
        self._dir = tempfile.mkdtemp(prefix='tnca-triton-')

    def load_binary(self, name, kernel, shared, device):
        path = os.path.join(self._dir, f'{name}-{abs(hash(kernel)):x}.tnca')
        if not os.path.exists(path):
            with open(path, 'wb') as fh:
                fh.write(kernel)
        # (module, function, n_regs, n_spills, n_max_threads)
        return path, path, 0, 0, self.get_device_properties(device)["max_threads_per_block"]

    def unload_module(self, module):
        pass

    def get_device_properties(self, device):
        from tnca.torch_device.inductor_device import properties
        p = properties(device)
        # TNCA has addressable global scratch, not CUDA shared memory.
        return {'max_shared_mem': 0, 'multiprocessor_count': p.multi_processor_count,
                'max_num_regs': p.regs_per_multiprocessor, 'warpSize': p.warp_size,
                'sm_clock_rate': 0, 'mem_clock_rate': 0, 'mem_bus_width': 64,
                'max_threads_per_block': p.max_threads_per_block}


class TNCADriver(DriverBase):
    """Active unless TNCA_TRITON=0, or a CUDA device is present (so installing
    tnca next to a GPU does not take over its Triton)."""

    def __init__(self):
        self.utils = TNCAUtils()
        self.launcher_cls = TNCALauncher

    @classmethod
    def is_active(cls):
        flag = os.environ.get('TNCA_TRITON')
        if flag is not None:
            return flag == '1'
        try:
            import torch
            return not torch.cuda.is_available()
        except Exception:                                # noqa: BLE001
            return True

    def map_python_to_cpp_type(self, ty):
        return {'i32': 'int32_t', 'i64': 'int64_t', 'fp32': 'float'}.get(ty, 'void*')

    def get_current_target(self):
        return GPUTarget('tnca', 0, 1)

    def get_active_torch_device(self):
        return _torch().device('cpu')

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass

    def get_current_stream(self, device=None):
        torch = _torch()
        if hasattr(torch, "tnca"):
            return torch.tnca.current_stream().stream_id
        return 0

    def get_device_interface(self):
        return _torch().cpu

    def get_benchmarker(self):
        def bench(fn, *a, **k):
            import time
            t = time.perf_counter()
            fn()
            return (time.perf_counter() - t) * 1e3
        return bench

    def get_empty_cache_for_benchmark(self):
        torch = _torch()
        return torch.empty(1, dtype=torch.uint8)

    def clear_cache(self, cache):
        cache.zero_()
