"""Address probes for an AXI window; no device or compiler dependencies."""

PC_BYTES = 256 << 20
PROBE_BYTES = 512


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
