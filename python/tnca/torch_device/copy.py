"""Same-device tensor copies execute on TNCA, including strided copyback."""
import threading
import tempfile
import weakref
import os
from pathlib import Path

import triton
import triton.language as tl

_lock = threading.RLock()
_storage_modules = weakref.WeakKeyDictionary()


@triton.jit
def _copy_storage_kernel(SRC, DST, N, BLOCK: tl.constexpr):
    stride = tl.num_programs(0) * BLOCK
    for block in range(tl.cdiv(N, stride)):
        index = (block * tl.num_programs(0) + tl.program_id(0)) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(SRC + index, index < N, other=0)
        tl.store(DST + index, value, index < N)


# Largest span one copy launch covers. The kernels index in i32 (the count,
# and offsets from each argument's own address); a longer copy is a sequence
# of launches whose arguments are 64-bit buffer offsets.
MAX_SPAN = (1 << 31) - (1 << 20)


def _try_stream_copy(memory, dst, src, nbytes):
    from tnca.backends.hbm_stream import NATIVE_COPY_MIN_BYTES, PAGE_GROUP
    if nbytes<NATIVE_COPY_MIN_BYTES or (src|dst|nbytes)&63 or (src-dst)%PAGE_GROUP:
        return False
    from .graph import active_capture
    if active_capture() is not None or os.environ.get('TNCA_LAUNCH') == 'skip':
        return False
    from tnca.triton_backend.driver import _bench_local
    if getattr(_bench_local, 'compile_only', False):
        return False
    session = memory.session()
    if not getattr(session.dev, 'stream_available', False):
        return False
    memory.synchronize()
    source, source_offset = memory.locate(src)
    destination, destination_offset = memory.locate(dst)
    return session.try_stream_copy(source,destination,nbytes,
        source_offset=source_offset,destination_offset=destination_offset)


def copy_storage(memory, dst, src, nbytes):
    """Allocator resize copy: opaque bytes stay on the device."""
    if not nbytes or dst == src:
        return
    if nbytes < 0:
        raise RuntimeError('tnca resize copy requires a positive byte count')
    if _try_stream_copy(memory,dst,src,nbytes):
        return
    from tnca import aot, device
    memory.synchronize()
    session = memory.session()
    source, source_offset = memory.locate(src)
    destination, destination_offset = memory.locate(dst)
    with _lock:
        cached = _storage_modules.get(session)
        if cached is None or cached[0] != session.epoch:
            with tempfile.TemporaryDirectory(prefix='tnca-storage-copy-') as temporary:
                path = Path(temporary) / 'copy.tnca'
                aot.compile_aot(_copy_storage_kernel,
                    signature={'SRC': '*u8', 'DST': '*u8', 'N': 'i32', 'BLOCK': 'constexpr'},
                    constexprs={'BLOCK': 128}, profile=device.base_profile(session.profile),
                    output=str(path))
                module = session.load(path)
            _storage_modules[session] = (session.epoch, module)
        else:
            module = cached[1]
        for done in range(0, nbytes, MAX_SPAN):
            n = min(MAX_SPAN, nbytes - done)
            event = session.submit(module, {'SRC': (source, source_offset + done),
                'DST': (destination, destination_offset + done), 'N': n},
                grid=(min(triton.cdiv(n, 128), 1024), 1, 1))
            session.wait(event)
            if not event.ok:
                raise event.error


@triton.jit
def _copy_kernel(SRC, DST, COUNT: tl.constexpr, SHAPE: tl.constexpr,
                 SRC_STRIDES: tl.constexpr, DST_STRIDES: tl.constexpr,
                 BYTES: tl.constexpr, RAW: tl.constexpr, CONTIGUOUS: tl.constexpr,
                 BLOCK: tl.constexpr, PROGRAMS: tl.constexpr, FP8_SOURCE: tl.constexpr = False):
    pid = tl.program_id(0)
    for block in range(tl.cdiv(COUNT, BLOCK * PROGRAMS)):
        index = (block * PROGRAMS + pid) * BLOCK + tl.arange(0, BLOCK)
        mask = index < COUNT
        if RAW:
            element = index // BYTES
            byte = index % BYTES
        else:
            element = index
        if CONTIGUOUS:
            src_offset = element
            dst_offset = element
        else:
            remaining = element
            src_offset = tl.full((BLOCK,), 0, tl.int32)
            dst_offset = tl.full((BLOCK,), 0, tl.int32)
            for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                coordinate = remaining % SHAPE[axis]
                remaining = remaining // SHAPE[axis]
                src_offset += coordinate * SRC_STRIDES[axis]
                dst_offset += coordinate * DST_STRIDES[axis]
        if RAW:
            src_bytes = SRC.to(tl.pointer_type(tl.uint8))
            dst_bytes = DST.to(tl.pointer_type(tl.uint8))
            value = tl.load(src_bytes + src_offset * BYTES + byte, mask, other=0)
            tl.store(dst_bytes + dst_offset * BYTES + byte, value, mask)
        elif FP8_SOURCE:
            # Triton has no integer -> FP8 cast for `other=0`. A masked-off
            # lane is never stored, so the load needs no `other` at all.
            value = tl.load(SRC + src_offset, mask)
            tl.store(DST + dst_offset, value.to(DST.dtype.element_ty), mask)
        else:
            value = tl.load(SRC + src_offset, mask, other=0)
            tl.store(DST + dst_offset, value.to(DST.dtype.element_ty), mask)


def copy_tensor(src, dst):
    import torch
    from . import storage_address, _span
    from tnca.triton_backend.driver import TNCADriver

    shape = tuple(dst.shape)
    if src.ndim > dst.ndim:
        raise RuntimeError('tnca copy: source rank exceeds destination rank')
    source_shape = (1,) * (dst.ndim - src.ndim) + tuple(src.shape)
    source_strides = (0,) * (dst.ndim - src.ndim) + tuple(src.stride())
    if any(s != 1 and s != d for s, d in zip(source_shape, shape)):
        raise RuntimeError('tnca copy: source is not broadcastable to destination')
    source_strides = tuple(0 if n == 1 else s for n, s in zip(source_shape, source_strides))
    if dst.numel() == 0:
        return dst
    if any(n > 1 and s == 0 for n, s in zip(shape, dst.stride())):
        raise RuntimeError('tnca copy: destination has overlapping elements')
    if storage_address(src) == storage_address(dst):
        if (src.data_ptr() == dst.data_ptr() and src.dtype == dst.dtype and
                tuple(src.shape) == shape and tuple(src.stride()) == tuple(dst.stride())):
            return dst
        sb, sn, _ = _span(src)
        db, dn, _ = _span(dst)
        if max(sb, db) < min(sb + sn, db + dn):
            raise RuntimeError('tnca copy: overlapping source and destination views')
        # A storage can own two disjoint views. An independent intermediate
        # also satisfies the runtime's conservative whole-buffer alias ABI.
        temporary = torch.empty(shape, device=src.device, dtype=src.dtype)
        copy_tensor(src, temporary)
        return copy_tensor(temporary, dst)
    raw = src.dtype == dst.dtype
    count = dst.numel() * (dst.element_size() if raw else 1)
    contiguous = src.is_contiguous() and dst.is_contiguous() and tuple(src.shape) == shape
    if raw and contiguous:
        from . import MEMORY
        if _try_stream_copy(MEMORY,dst.data_ptr(),src.data_ptr(),count):
            return dst
    source_extent = src.element_size() * (1 + sum((n - 1) * s
                                                  for s, n in zip(source_strides, shape)))
    destination_extent = dst.element_size() * (1 + sum((n - 1) * s
                                                       for s, n in zip(dst.stride(), shape)))
    if max(count, source_extent, destination_extent) > MAX_SPAN:
        # Halve the dimension spanning the most bytes until each launch's
        # count and spans fit its i32 indexing; each part is a view whose
        # 64-bit storage offset the launcher passes as its buffer offset.
        source = src.expand(shape) if tuple(src.shape) != shape else src
        dim = max((d for d, n in enumerate(shape) if n > 1),
                  key=lambda d: (shape[d] - 1) * max(source.stride(d), dst.stride(d)))
        half = shape[dim] // 2
        for part in (slice(0, half), slice(half, shape[dim])):
            index = (slice(None),) * dim + (part,)
            copy_tensor(source[index], dst[index])
        return dst
    contiguous = src.is_contiguous() and dst.is_contiguous() and tuple(src.shape) == shape
    block = 128 if contiguous else 16
    programs = min(triton.cdiv(count, block), 1024)
    with _lock:
        previous = triton.runtime.driver.active
        try:
            triton.runtime.driver.set_active(TNCADriver())
            _copy_kernel[(programs,)](
                src, dst, count, shape, source_strides, tuple(dst.stride()),
                dst.element_size(), raw,
                contiguous, block, programs, num_warps=1,
                **({'FP8_SOURCE': True} if not raw and src.dtype in
                   (torch.float8_e4m3fn, torch.float8_e5m2) else {}))
        finally:
            triton.runtime.driver.set_active(previous)
    return dst
