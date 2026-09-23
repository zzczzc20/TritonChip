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

One Session per process and profile, opened on first launch.
"""
import os
import tempfile
import threading

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _torch():
    import torch
    return torch


class _Sessions:
    lock = threading.Lock()
    open = {}

    @classmethod
    def get(cls, profile):
        from tnca.runtime import Session
        with cls.lock:
            s = cls.open.get(profile)
            if s is None or s.closed or s.poisoned:
                s = Session(profile=profile, deadline_cycles=int(os.environ.get('TNCA_DEADLINE', '5000000')),
                            counters=os.environ.get('TNCA_COUNTERS') == '1')
                cls.open[profile] = s
            return s


class TNCALaunchError(RuntimeError):
    pass


_TORCH_TO_TNCA = {'float32': 'fp32', 'float16': 'fp16', 'bfloat16': 'bf16', 'int32': 'i32',
                  'int8': 'i8', 'float8_e4m3fn': 'e4m3', 'float8_e5m2': 'e5m2'}


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
        buffers, bound = {}, {}

        def device_ref(name, v):
            if v.device.type != 'cpu':
                raise TNCALaunchError(f'{name}: tensor on {v.device}; TNCA stage 1 takes host tensors')
            st = v.untyped_storage()
            key = st.data_ptr()
            if key not in buffers:
                raw = torch.empty(0, dtype=torch.uint8).set_(st)
                handle = session.allocate(max(st.nbytes(), 8), name=name)
                if st.nbytes():
                    session.upload(handle, raw.numpy().tobytes())
                buffers[key] = (handle, raw)
            handle = buffers[key][0]
            off = v.storage_offset() * v.element_size()
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
                    bound[name] = TD(device_ref(name, v.base), list(v.shape), list(v.strides),
                                     list(v.block_shape), dt)
                elif isinstance(v, bool):
                    bound[name] = int(v)
                elif isinstance(v, (int, float)):
                    bound[name] = v
                elif v is None:
                    raise TNCALaunchError(f'{name}: None argument')
                else:
                    raise TNCALaunchError(f'{name}: unsupported argument type {type(v).__name__}')
            event = session.submit(module, bound, grid=(max(gridX, 1), max(gridY, 1), max(gridZ, 1)))
            session.wait(event)
            self.last_event = event
            self.last_cycles = event.device_cycles
            TNCALauncher.last = self
            if event.state != 'SUCCEEDED':
                raise TNCALaunchError(f'{self.metadata.name}: {event.error}')
            for handle, raw in buffers.values():
                n = raw.numel()
                if n:
                    data = session.download(handle)[:n]
                    raw.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        finally:
            for handle, _raw in buffers.values():
                try:
                    session.free(handle)
                except Exception:                        # noqa: BLE001
                    pass
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


_loaded = {}


def _load(session, path):
    """The module for `path` in this session, loading it if needed.

    A process-wide session accumulates code: the runtime places each module
    and never unloads one, and the code region below the heap is finite. When
    it is full the session is replaced (a fresh device, empty code region)
    and the module loaded there; buffers never outlive a launch, so nothing
    is lost.
    """
    from tnca.errors import BufferError as _BE
    key = (id(session), session.epoch, path)
    m = _loaded.get(key)
    if m is not None:
        return session, m
    try:
        m = session.load(path)
    except _BE as e:
        if 'overlaps the device heap' not in str(e):
            raise
        profile = session.profile
        session.close()
        session = _Sessions.get(profile)
        m = session.load(path)
        key = (id(session), session.epoch, path)
    _loaded[key] = m
    return session, m


class TNCAUtils:
    def __init__(self):
        self._dir = tempfile.mkdtemp(prefix='tnca-triton-')

    def load_binary(self, name, kernel, shared, device):
        path = os.path.join(self._dir, f'{name}-{abs(hash(kernel)):x}.tnca')
        if not os.path.exists(path):
            with open(path, 'wb') as fh:
                fh.write(kernel)
        # (module, function, n_regs, n_spills, n_max_threads)
        return path, path, 0, 0, 1 << 20

    def unload_module(self, module):
        pass

    def get_device_properties(self, device):
        return {'max_shared_mem': 1 << 30, 'multiprocessor_count': 1, 'max_num_regs': 1 << 16,
                'warpSize': 1, 'sm_clock_rate': 0, 'mem_clock_rate': 0, 'mem_bus_width': 64,
                'max_threads_per_block': 1 << 20}


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
