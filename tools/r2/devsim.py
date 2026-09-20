#!/usr/bin/env python3
"""
A device runner for the R2 tests -- verification only.

This is NOT the product runtime. R2-08 delivers that: a persistent session,
buffers, a queue and events. This is what the verification side uses until
then and afterwards to cross-check it: load an artifact through the product
loader, place its code and an argument block in a memory image, run one
program on a bench, and read the result back.

The contract allows exactly this shape -- "verification -- calls the public
interfaces above, plus its own oracles" -- and draws one hard line, MEM-05:
the PRODUCT download path may never read a testbench-level array. This module
does read the bench's memory dump, and that is why it lives under tools/ and
is named for what it is. When the runtime's transport exists, the E2E cases
move to it and this stays as the independent check.
"""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))

from tnca import abi, artifact                       # noqa: E402
from tnca.errors import TncaError                    # noqa: E402

WORD = 8
IMAGE_WORDS = 16384
BENCH = {
    'core-sim': 'sim/build/tb_core.vvp',
    'core-4r': 'sim/build/tb_core4.vvp',
    'system-2x2': 'sim/build/tb_top.vvp',
}


class Memory:
    """A flat device image, built by placing byte blobs at addresses."""

    def __init__(self, words=IMAGE_WORDS):
        self.mem = bytearray(words * WORD)

    def put(self, addr, blob):
        if addr % WORD:
            raise ValueError(f"address {addr:#x} is not 8-byte aligned")
        if addr + len(blob) > len(self.mem):
            raise ValueError(
                f"{len(blob)} bytes at {addr:#x} runs past the {len(self.mem)}"
                f"-byte image")
        self.mem[addr:addr + len(blob)] = blob
        return addr

    def put_f32(self, addr, values):
        return self.put(addr, b''.join(struct.pack('<f', v) for v in values))

    def put_i32(self, addr, values):
        return self.put(addr, b''.join(struct.pack('<i', v) for v in values))

    def write_hex(self, path):
        with open(path, 'w') as fh:
            for i in range(0, len(self.mem), WORD):
                fh.write(f"{int.from_bytes(self.mem[i:i + WORD], 'little'):016x}\n")


def read_f32(words, count):
    """Two f32 per 64-bit word, little-endian."""
    out = []
    for w in words:
        out.append(struct.unpack('<f', struct.pack('<I', w & 0xFFFFFFFF))[0])
        out.append(struct.unpack('<f', struct.pack('<I', (w >> 32) & 0xFFFFFFFF))[0])
    return out[:count]


class Run:
    def __init__(self, ok, stdout, dump_words):
        self.ok = ok
        self.stdout = stdout
        self.words = dump_words

    def f32(self, count):
        return read_f32(self.words, count)


def run_artifact(path, args, *, memory, arg_base, dump_at, dump_words,
                 profile='core-sim', max_cycles=200000, extra=()):
    """Load an artifact through the product loader and execute it once.

    `args` maps parameter name -> value, with pointers already resolved to
    device addresses. The argument block is encoded by `tnca.abi` -- the
    product's encoder, not a copy of it -- so a disagreement between what the
    compiler laid out and what the kernel reads shows up here.
    """
    man, segments = artifact.load(path)
    params = [(p['name'], p['type'])
              for p in man['parameters']['signature']]
    block = abi.encode_block(params, args)

    memory.put(0, segments['text'])
    memory.put(arg_base, block)

    bench = os.path.join(ROOT, BENCH[profile])
    if not os.path.exists(bench):
        raise TncaError(f"{BENCH[profile]} is not built",
                        code='DEVICE_LOST',
                        remedy=f"run `make core` (or core4/top) first")

    tmp = tempfile.mkdtemp(prefix='r2-dev-')
    hexf = os.path.join(tmp, 'image.hex')
    dumpf = os.path.join(tmp, 'dump.txt')
    memory.write_hex(hexf)

    argv = ['vvp', bench, f'+hex={hexf}', f'+maxcycles={max_cycles}',
            f'+argbase={arg_base}', f'+dump={dumpf}',
            f'+dumpfirst={dump_at // WORD}', f'+dumpcount={dump_words}',
            *extra]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    ok = 'PASS' in r.stdout and 'FAIL' not in r.stdout
    words = []
    if os.path.exists(dumpf):
        words = [int(line, 16) for line in open(dumpf) if line.strip()]
    return Run(ok, r.stdout, words)
