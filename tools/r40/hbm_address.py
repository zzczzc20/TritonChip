"""Address probes for an AXI window; no device or compiler dependencies."""

PC_BYTES = 256 << 20
PROBE_BYTES = 512


def topology_probe_offsets(window):
    """Disjoint probes at real allocated PC boundaries, with no PC-size guess.

    A boundary probe spans its two physical PCs; edge probes also reach the
    beginning/end of the exposed BO. The caller reserves argument/control space
    separately so it can execute while the complete data window is held.
    """
    base,size=window['base'],window['size']
    allocation=window.get('allocation') or {}
    banks=allocation.get('physical_banks') or []
    if allocation.get('mapping_error') or not allocation.get('topology_sha256') or not banks:
        raise ValueError('certified physical HBM topology is required')
    if base%64 or size%64 or size<PROBE_BYTES:
        raise ValueError('window must be aligned and large enough for a probe')
    cursor=base
    offsets={0,size-PROBE_BYTES}
    for bank in banks:
        lo=bank['allocated_base'];n=bank['allocated_size']
        if lo!=cursor or n<PROBE_BYTES or lo%64 or n%64:
            raise ValueError('physical PC allocations must cover the window without gaps')
        if lo>base:offsets.add(lo-base-PROBE_BYTES//2)
        cursor+=n
    if cursor!=base+size or len({b['pc'] for b in banks})!=len(banks):
        raise ValueError('physical PC allocations do not uniquely cover the window')
    result=sorted(offsets)
    if any(b-a<PROBE_BYTES for a,b in zip(result,result[1:])):
        raise ValueError('partial PC is too small for disjoint edge/boundary probes')
    return result


def probe_offsets(base, size):
    """Probe every physical PC intersecting the window, including the last.

    Small simulator windows use 32 evenly spaced probes instead. Offsets
    are 64-byte aligned so allocator padding does not move a probe's PC.
    A partial PC with less than a probe's worth of memory is rejected.
    """
    if base < 0 or size < PROBE_BYTES or base % 64 or size % 64:
        raise ValueError('window must be nonnegative, 64-byte aligned and at least 512 bytes')
    if size < 2 * PC_BYTES:
        count = min(32, size // PROBE_BYTES)
        step = (size // count) // 64 * 64
        return [i * step for i in range(count)]
    offsets = []
    end = base + size
    for pc in range(base // PC_BYTES, (end - 1) // PC_BYTES + 1):
        lo, hi = max(base, pc * PC_BYTES), min(end, (pc + 1) * PC_BYTES)
        if hi - lo < PROBE_BYTES:
            raise ValueError('partial PC cannot hold a 512-byte probe')
        offsets.append(lo - base + min(4096, (hi - lo - PROBE_BYTES) // 64 * 64))
    return offsets
