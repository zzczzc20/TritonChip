"""The `tnca` PyTorch device: PrivateUse1, storage in TNCA device memory.

    import tnca.torch_device
    tnca.torch_device.register()            # once per process
    x = torch.randn(4, 8).to('tnca')         # bytes now live in device memory
    y = x[1:, 2:]                            # a view: same storage, no copy
    y.cpu()                                  # explicit copy back

doc/r40_tasks_and_acceptance.md R40-01, the device half. The storage is real:
csrc/tnca_device.cpp registers an Allocator whose DataPtr is a DEVICE
address, from a buffer this module allocates in the TNCA runtime's heap
(tnca.runtime.Session.allocate), and the metadata operators that build and
view tensors of it -- empty, empty_strided, as_strided, view, _reshape_alias,
set_, resize_ -- the same code the CPU uses, because none reads the data.
What moves or reads data is here, through the Session:

  copies     host <-> device and device <-> device (`_copy_from`,
             `_copy_from_and_resize`), any strides and dtypes: the span of
             the storage a strided tensor covers is staged on the host, so a
             view copies exactly the elements it names (R40-01 correctness;
             device-side copies and bulk DMA are R40-02's)
  fill_, zero_, item()

Everything else -- every computation -- is torch.compile through Inductor
and the tnca Triton backend; an eager operator with no tnca kernel is
refused by the dispatcher (NotImplementedError), never run on the host.

One Session per device, the one the Triton launcher uses (driver._Sessions),
so a kernel's argument is the tensor's own buffer and offset. `torch.tnca`
is the device module: is_available, device_count, current_device,
set_device, synchronize, Stream, Event. Batched launches retain storage
until completion; stream identities share the physical device FIFO and
events report actual queue completion.
"""
import atexit
import contextlib
import itertools
import os
import threading
from pathlib import Path

WORD = 8                     # tnca.backend: every transfer starts word-aligned
# Heap kept free for launches: Session.submit places every launch's argument
# block and per-context scratch in the same heap the storages come from
# (about 13 KB a launch, doc/r39_tasks_and_acceptance.md R39-13). Without a
# reserve, resident tensors could fill the heap and a launch would fail
# mid-graph; with it, running out is a tensor allocation that fails cleanly
# (torch.OutOfMemoryError, capacity-exceeded).
LAUNCH_RESERVE = int(os.environ.get('TNCA_LAUNCH_RESERVE', 32 * 1024))
_lock = threading.RLock()
_registered = False
_ext = None
NAME = 'tnca'


# ---- device memory ----------------------------------------------------------
class DeviceMemory:
    """Device address -> runtime Buffer, for every live storage."""

    def __init__(self):
        self.buffers = {}            # address -> Buffer
        self.allocations = {}        # unique token -> Buffer, across recovery
        self.owners = {}             # token -> allocating Session; destruction never opens a device
        self.next_token = 0
        self.peak = 0
        self.live = 0

    def session(self, index=0):
        # Triton first: importing it discovers its backends, which imports
        # the tnca driver (archbench/platforms/tnca.py says the same).
        import triton                                        # noqa: F401
        from tnca.triton_backend.driver import _Sessions
        if index != 0:
            raise RuntimeError(f'tnca:{index}: one device per process in R40-01')
        profile = os.environ.get('TNCA_PROFILE', 'core-sim')
        return _Sessions.get(profile)

    def alloc(self, nbytes, index=0):
        s = self.session(index)
        # Whole words: a transfer starts on an 8-byte boundary, and staging
        # rounds a span out to words, which must stay inside the storage.
        nbytes = (int(nbytes) + WORD - 1) // WORD * WORD
        while True:
            with s._lock:
                heap = s._heap
                free = sum(size for _addr, size in heap.free_list) + (heap.limit - heap.next)
                if free - nbytes >= LAUNCH_RESERVE:
                    buf = s.allocate(nbytes, name=f'storage@{len(self.buffers)}')
                    break
                pending = s.unfinished
                if not pending:
                    from tnca.errors import BufferError as _BE
                    raise _BE(f'out of device memory: a {nbytes}-byte storage would leave '
                              f'{free - nbytes} of the heap\'s free bytes, less than the '
                              f'{LAUNCH_RESERVE} kept for launches')
                oldest = pending[0]
            # A retired temporary can still be owned by queued work. Wait
            # outside the admission lock, then retry with its leases released.
            s.wait(oldest)
            if not oldest.ok:
                raise oldest.error
        self.buffers[buf.addr] = buf
        self.next_token += 1
        token = self.next_token
        self.allocations[token] = buf
        self.owners[token] = s
        self.live += buf.size
        self.peak = max(self.peak, self.live)
        return buf.addr, token

    def free(self, addr, token, index=0):
        buf = self.allocations.pop(token, None)
        owner = self.owners.pop(token, None)
        if buf is None:
            return
        if self.buffers.get(addr) is buf:
            self.buffers.pop(addr)
        self.live -= buf.size
        try:
            if owner is not None:
                owner.retire(buf)
        except Exception:                                    # noqa: BLE001
            # A dead or recovered session has no buffer to free; the storage
            # went with it.
            pass

    def validate(self, token):
        buf = self.allocations.get(token)
        s = self.session()
        if buf is None or not buf.alive or buf.session_id != s.id or buf.epoch != s.epoch:
            raise RuntimeError('tnca: storage belongs to a stale session or recovery epoch')

    def locate(self, addr):
        """(Buffer, byte offset) of a device address inside a live storage."""
        for base, buf in self.buffers.items():
            if base <= addr < base + max(buf.size, 1):
                return buf, addr - base
        raise RuntimeError(f'tnca: address {addr:#x} is in no live storage')

    def write(self, addr, data):
        """Bytes to the device at any address: the words at either end that
        the data only partly covers are read first and written back whole,
        so no byte outside [addr, addr + len) changes."""
        self.synchronize()
        data = bytes(data)
        a0 = addr - addr % WORD
        a1 = -(-(addr + len(data)) // WORD) * WORD
        if a0 != addr or a1 != addr + len(data):
            word = bytearray(self.read(a0, a1 - a0))
            word[addr - a0:addr - a0 + len(data)] = data
            data = bytes(word)
        buf, off = self.locate(a0)
        self.session().upload(buf, data, off)

    def read(self, addr, nbytes):
        """Bytes from the device at any address (transfers are word-aligned)."""
        self.synchronize()
        a0 = addr - addr % WORD
        buf, off = self.locate(a0)
        data = self.session().download(buf, nbytes + (addr - a0), off)
        return data[addr - a0:]

    def copy(self, dst, src, nbytes):
        """Device to device when an allocator grows a storage."""
        from .copy import copy_storage
        copy_storage(self, dst, src, nbytes)

    def synchronize(self):
        s = self.session()
        last = s.latest_event
        s.drain()
        if last is not None and not last.ok:
            raise last.error


MEMORY = DeviceMemory()


def storage_address(t):
    """The device address of a tnca tensor's storage (its base)."""
    return _ext.storage_address(t)


# ---- staging: a strided tensor's span ---------------------------------------
def _span(t):
    """(first byte address, byte length, lowest element offset) of the storage
    elements `t` names -- the bytes a copy must stage."""
    lo = hi = 0
    for size, stride in zip(t.shape, t.stride()):
        ext = (size - 1) * stride
        if ext < 0:
            lo += ext
        else:
            hi += ext
    item = t.element_size()
    base = storage_address(t) + (t.storage_offset() + lo) * item
    return base, (hi - lo + 1) * item, lo


def _staged(t, data, lo):
    """(flat, view): a CPU byte image of the span and `t`'s layout over it."""
    import torch
    flat = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return flat, flat.view(t.dtype).as_strided(t.shape, t.stride(), -lo)


def to_host(t):
    """A CPU tensor with `t`'s values (same shape and dtype)."""
    import torch
    if t.numel() == 0:
        return torch.empty(t.shape, dtype=t.dtype)
    base, n, lo = _span(t)
    return _staged(t, MEMORY.read(base, n), lo)[1].clone()


def from_host(dst, src):
    """dst[...] = src: dst on the device, src a CPU tensor broadcastable to it."""
    import torch
    if dst.numel() == 0:
        return
    base, n, lo = _span(dst)
    def write_dense(data):
        # A whole storage copy may define its allocator's word padding.
        # Those bytes lie beyond Storage.nbytes(), so no valid alias names
        # them. Partial storage copies still preserve neighboring bytes.
        if dst.storage_offset() == 0 and n == dst.untyped_storage().nbytes():
            data += bytes(-len(data) % WORD)
        MEMORY.write(base, data)
    if dst.is_contiguous() and n == dst.numel() * dst.element_size():
        host = torch.empty(dst.shape, dtype=dst.dtype)
        host.copy_(src)
        write_dense(host.view(-1).view(torch.uint8).numpy().tobytes())
        return
    dense, extent = True, 1
    for stride, size in sorted((stride, size) for size, stride in
                               zip(dst.shape, dst.stride()) if size > 1):
        if stride != extent:
            dense = False
            break
        extent *= size
    if dense:
        # A dense permutation covers every byte of the span. Reading first
        # needlessly observes uninitialized board RAM on a fresh allocation.
        flat = torch.empty(n, dtype=torch.uint8)
        view = flat.view(dst.dtype).as_strided(dst.shape, dst.stride(), -lo)
        view.copy_(src)
        write_dense(flat.numpy().tobytes())
        return
    # A view with gaps: the bytes between its elements belong to other
    # tensors and must come back unchanged, so the span is read first.
    flat, view = _staged(dst, MEMORY.read(base, n), lo)
    view.copy_(src)
    MEMORY.write(base, flat.numpy().tobytes())


# ---- operators --------------------------------------------------------------
def _register_ops(lib):
    import torch

    def _refuse(t):
        if t.is_complex():
            raise TypeError(f'tnca: {t.dtype} tensors are not supported (no complex arithmetic '
                            'on the device); keep them on the host')

    def _copy_from(self, dst, non_blocking=False):
        _refuse(self)
        _refuse(dst)
        if self.device.type == NAME and dst.device.type == NAME:
            from .copy import copy_tensor
            copy_tensor(self, dst)
        elif self.device.type == NAME:
            dst.copy_(to_host(self))
        elif dst.device.type == NAME:
            from_host(dst, self)
        else:
            raise RuntimeError(f'tnca _copy_from: {self.device} -> {dst.device}')
        return dst

    def _copy_from_and_resize(self, dst):
        dst.resize_(self.shape)
        return _copy_from(self, dst)

    def fill_scalar(self, value):
        host = torch.empty(self.shape, dtype=self.dtype)
        host.fill_(value)
        from_host(self, host)
        return self

    def zero_(self):
        return fill_scalar(self, 0)

    def local_scalar_dense(self):
        return to_host(self).item()

    lib.impl('_copy_from', _copy_from, 'PrivateUse1')
    lib.impl('_copy_from_and_resize', _copy_from_and_resize, 'PrivateUse1')
    lib.impl('fill_.Scalar', fill_scalar, 'PrivateUse1')
    lib.impl('zero_', zero_, 'PrivateUse1')
    lib.impl('_local_scalar_dense', local_scalar_dense, 'PrivateUse1')


# ---- the device module (torch.tnca) -------------------------------------------
_stream_local = threading.local()
_stream_ids = itertools.count(1)
_streams = {}


class _Stream:
    """Streams share the single device's in-order queue in this profile."""

    def __init__(self, device=None, priority=0, *, _id=None):
        import torch
        self.device = torch.device(NAME, 0) if device is None else torch.device(device)
        if self.device.type != NAME or self.device.index not in (None, 0):
            raise RuntimeError(f'unsupported tnca stream device {self.device}')
        self.stream_id = next(_stream_ids) if _id is None else _id
        _streams[self.stream_id] = self

    def synchronize(self):
        MEMORY.synchronize()

    def query(self):
        return not MEMORY.session().unfinished

    def wait_event(self, event):
        event.synchronize()

    def wait_stream(self, stream):
        stream.synchronize()

    def record_event(self, event=None):
        event = event or _Event()
        event.record(self)
        return event


class _Event:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.recorded = False
        self.events = ()
        self.session = None
        self.epoch = None

    def record(self, stream=None):
        self.recorded = True
        self.session = MEMORY.session()
        self.epoch = self.session.epoch
        latest = self.session.latest_event
        self.events = (latest,) if latest is not None else ()

    def query(self):
        self._check_epoch()
        return all(ev.terminal for ev in self.events)

    def synchronize(self):
        self._check_epoch()
        for ev in self.events:
            self.session.wait(ev)
            if not ev.ok:
                raise ev.error

    def wait(self, stream=None):
        self.synchronize()

    def _check_epoch(self):
        if self.session is not None and self.session.epoch != self.epoch:
            raise RuntimeError('tnca event belongs to a recovered session epoch')


from .graph import Graph, graph


class _DeviceModule:
    Graph = Graph
    graph = staticmethod(graph)
    Stream = _Stream
    Event = _Event

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def set_device(device):
        import torch
        idx = torch.device(device).index if not isinstance(device, int) else device
        if idx not in (None, 0):
            raise RuntimeError(f'tnca:{idx} does not exist')

    @staticmethod
    def synchronize(device=None):
        MEMORY.synchronize()

    @staticmethod
    def current_stream(device=None):
        return getattr(_stream_local, 'current', None) or _DeviceModule.default_stream(device)

    @staticmethod
    def default_stream(device=None):
        return _streams.get(0) or _Stream(device, _id=0)

    @staticmethod
    def set_stream(stream):
        if not isinstance(stream, _Stream):
            raise TypeError('set_stream expects a tnca Stream')
        _stream_local.current = stream

    @staticmethod
    @contextlib.contextmanager
    def stream(stream):
        previous = _DeviceModule.current_stream()
        if stream is not None:
            _DeviceModule.set_stream(stream)
        try:
            yield stream
        finally:
            _DeviceModule.set_stream(previous)

    @staticmethod
    def _set_stream_by_id(stream_id, device_index, device_type):
        if device_index != 0 or stream_id not in _streams:
            raise RuntimeError('unknown tnca stream or device')
        _DeviceModule.set_stream(_streams[stream_id])

    @staticmethod
    def get_device_name(device=None):
        return 'TNCA ' + os.environ.get('TNCA_PROFILE', 'core-sim')

    @staticmethod
    def is_initialized():
        return _registered

    @staticmethod
    def _is_in_bad_fork():
        return False

    @staticmethod
    def manual_seed_all(seed):
        pass

    @staticmethod
    def memory_allocated(device=None):
        return MEMORY.live

    @staticmethod
    def max_memory_allocated(device=None):
        return MEMORY.peak


def _build():
    from torch.utils.cpp_extension import load
    src = Path(__file__).resolve().parent / 'csrc' / 'tnca_device.cpp'
    return load(name='tnca_device_ext', sources=[str(src)], verbose=False,
                extra_cflags=['-O2'])


def register():
    """Make `tnca` a PyTorch device in this process (idempotent)."""
    global _registered, _ext
    with _lock:
        if _registered:
            return
        import torch
        torch.utils.rename_privateuse1_backend(NAME)
        _ext = _build()
        _ext.set_memory_callbacks(MEMORY.alloc, MEMORY.free, MEMORY.copy, MEMORY.validate)
        _ext.set_device_count(1)
        atexit.register(_ext.shutdown)
        torch._register_device_module(NAME, _DeviceModule())
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

        class _Hooks(torch._C._acc.PrivateUse1Hooks):
            def is_available(self):
                return True

            def has_primary_context(self, dev_id):
                return True

            def is_built(self):
                return True

        global _hooks
        _hooks = _Hooks()
        torch._C._acc.register_python_privateuseone_hook(_hooks)
        global _lib
        _lib = torch.library.Library('aten', 'IMPL')
        _register_ops(_lib)
        _registered = True
        # torch.compile on tnca tensors: Inductor's Triton code generation for
        # the device, with the TNCA heuristics, templates and descriptor rules.
        import tnca.inductor
        tnca.inductor.configure()
        from .inductor_device import register_inductor
        register_inductor()


_hooks = None
_lib = None
