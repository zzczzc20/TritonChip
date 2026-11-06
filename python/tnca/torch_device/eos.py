"""Latch EOS entirely on the device for the optional stop guard."""
import threading

import triton
import triton.language as tl

_lock = threading.RLock()


@triton.jit
def _latch_eos(TOKEN, FLAG, EOS: tl.constexpr):
    token = tl.load(TOKEN)
    stopped = tl.load(FLAG)
    for index in tl.static_range(len(EOS)):
        stopped = stopped | (token == EOS[index]).to(tl.int64)
    tl.store(FLAG, stopped)


def latch_eos(token, flag, eos):
    """Enqueue one sticky flag update after device-side token selection."""
    if not eos:
        raise ValueError('at least one EOS token is required')
    from tnca.triton_backend.driver import TNCADriver
    with _lock:
        previous = triton.runtime.driver.active
        try:
            triton.runtime.driver.set_active(TNCADriver())
            _latch_eos[(1,)](token, flag, tuple(sorted(eos)), num_warps=1)
        finally:
            triton.runtime.driver.set_active(previous)
