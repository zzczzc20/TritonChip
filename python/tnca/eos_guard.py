"""Device-only guard for skipping speculative work after an EOS token.

Each physical program receives a normal ABI argument block. The runtime
reserves a 256-byte header immediately before each core's block containing
the real entry PC and a pointer to a shared, device-resident stop flag. The
guard exits when that flag is set and otherwise jumps to the original entry.
No generated kernel, tensor ABI or host read of the flag is needed.
"""
import struct
from functools import lru_cache


HEADER_BYTES = 256
TARGET_OFFSET = HEADER_BYTES - 16
FLAG_OFFSET = HEADER_BYTES - 8
# The busy path retires at least one branch per iteration. The 15x margin
# makes an unknown or unexpectedly short counter fail closed during setup.
PROBE_ITERATIONS = 30000
PROBE_BUSY_MIN_CYCLES = 30000
PROBE_STOP_MAX_CYCLES = 2048


@lru_cache(maxsize=1)
def guard_code():
    # Only s60/s61 are clobbered before kernel entry. Kernels initialize
    # their scalar registers and read arg_base from its special register;
    # s0 remains the hardware zero register. Relative branches in the
    # original kernel are unaffected because its text is never rewritten.
    from tnca_asm import Assembler, build_image
    source = '''
        s.rdspecial s60, arg_base
        s.ld.d s61, [s60, #-8]
        s.ld.d s61, [s61, #0]
        br.nz s61, stopped
        s.ld.d s61, [s60, #-16]
        br.r s61
    stopped:
        exit
    '''
    words = build_image(Assembler().assemble(source))
    return struct.pack('<' + 'Q' * len(words), *words)


@lru_cache(maxsize=1)
def stop_probe_code():
    """A known long path; the stop guard's short path skips it completely."""
    from tnca_asm import Assembler, build_image
    source = f'''
        .li s60, #{PROBE_ITERATIONS}
    spin:
        s.addi s60, s60, #-1
        br.nz s60, spin
        exit
    '''
    words = build_image(Assembler().assemble(source))
    return struct.pack('<' + 'Q' * len(words), *words)


def stop_probe_indicates_stop(cycles):
    """Unknown counters fail closed and continue the ordinary device loop."""
    return type(cycles) is int and 0 < cycles < PROBE_STOP_MAX_CYCLES


def argument_table(block, stride, cores, target, flag):
    """Insert guard headers into an already relocated physical ABI table."""
    if (not isinstance(stride, int) or stride <= HEADER_BYTES or stride % 256
            or not isinstance(cores, int) or cores <= 0 or len(block) != stride * cores
            or any(not isinstance(value, int) or not 0 <= value < 1 << 64 or value % 8
                   for value in (target, flag))):
        raise ValueError('invalid guarded physical argument table')
    result = bytearray(len(block))
    payload = stride - HEADER_BYTES
    for core in range(cores):
        base = core * stride
        struct.pack_into('<QQ', result, base + TARGET_OFFSET, target, flag)
        result[base + HEADER_BYTES:base + stride] = block[base:base + payload]
    return result
