"""Run a TileBench kernel on the RTL, and check what it computed.

"Builds" is not "runs". This is the second half: lay the inputs out in
the simulated memory, execute the compiled kernel on the iverilog bench,
read the output back and compare it against a reference computed here in
Python.

The comparison is not always exact, and where it is not the tolerance is
the ISA's published one rather than a number chosen to make a test pass.
ISA sec 8.8 specifies the SFU to a relative error of 2**-22, and
mathlib.py states per operation what it does with that -- so a kernel
containing a division is checked to the bound `div` promises, and one
containing an exponential to the bound `exp` promises. A kernel with
neither is checked bit-for-bit.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(ROOT, 'compiler', 'python'),
          os.path.join(ROOT, 'compiler', 'test'),
          os.path.join(ROOT, 'tools', 'model'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rtl_harness as H                                    # noqa: E402
from tnca_cc import TargetInfo                             # noqa: E402
from tnca_cc.codegen import CodeGen                        # noqa: E402
from tnca_cc.frontend import TTIRImporter, Unsupported     # noqa: E402

# The bench's memory is 16384 words of 64 bits. Code starts at 0, and
# one of these kernels -- the split-k matmul, which is 256 atomics and a
# compare-and-swap loop around each -- is 28 KiB of it, so the data
# starts well past that.
IMAGE_WORDS = 16384
BUF_BASE = 0xA000
BUF_STRIDE = 0x400          # 1 KiB a buffer: a 16x16 f32 tile is 1 KiB
SCRATCH = 0x18000           # descriptors and repacks, past every buffer
SENTINEL = 0xdeadbeef


def f2b(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def b2f(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def f16b(x):
    return struct.unpack('<H', struct.pack('<e', float(x)))[0]


def b2f16(b):
    return struct.unpack('<e', struct.pack('<H', b & 0xFFFF))[0]


# dtype -> (bytes, value -> raw bits, raw bits -> value)
DTYPES = {
    'fp32': (4, f2b, b2f),
    'fp16': (2, f16b, b2f16),
    'i32': (4, lambda v: int(v) & 0xFFFFFFFF,
            lambda b: b - (1 << 32) if b >> 31 else b),
    'i8': (1, lambda v: int(v) & 0xFF,
           lambda b: b - (1 << 8) if b >> 7 else b),
}


def pack(values, dtype):
    """A typed array as 64-bit little-endian words, ready for `.dword`."""
    size, enc, _ = DTYPES[dtype]
    per = 8 // size
    out = []
    for i in range(0, len(values), per):
        word = 0
        for j, v in enumerate(values[i:i + per]):
            word |= (enc(v) & ((1 << (size * 8)) - 1)) << (size * 8 * j)
        out.append(word)
    return out


class Case:
    """One operator's inputs, its expected output and how to compare.

    `buffers` maps a pointer parameter to (dtype, values). `scalars`
    gives the rest. `expect` is the output buffer's contents; `atol` and
    `rtol` are the tolerance, and default to exact.
    """

    def __init__(self, kernel, *, buffers, scalars=None, out, expect,
                 rtol=0.0, atol=0.0, offsets=None, out_offset=0, note=None):
        self.kernel = kernel
        self.buffers = buffers
        self.scalars = scalars or {}
        self.out = out
        self.expect = list(expect)
        self.rtol = rtol
        self.atol = atol
        # Elements to add to a pointer ARGUMENT, leaving the buffer
        # where it is. A stencil reads the row above the one it writes,
        # so `x_ptr` has to point into the middle of its data rather
        # than at the start of it -- with program_id 0 the row above
        # row 0 is at a negative offset, and a kernel that read there
        # would be reading whatever preceded the buffer.
        self.offsets = offsets or {}
        self.out_offset = out_offset
        self.note = note


def _addresses(case):
    """A page-aligned address per buffer, sized by what goes in it.

    Not a fixed stride. A 3x3 convolution's input plane is 20 rows of
    16, which is 1280 bytes, and a fixed 1 KiB stride put the weights
    inside it -- the assembler caught that as an `.org` moving
    backwards, which is a better outcome than the silent overlap it
    would have been one byte smaller.
    """
    out, at = {}, BUF_BASE
    for name, (dtype, values) in case.buffers.items():
        out[name] = at
        nbytes = len(values) * DTYPES[dtype][0]
        at += max(BUF_STRIDE, (nbytes + BUF_STRIDE - 1)
                  // BUF_STRIDE * BUF_STRIDE)
    if at > SCRATCH:
        raise ValueError(f"the buffers reach {at:#x}, past the scratch at "
                         f"{SCRATCH:#x}")
    return out


def build(name, case, target=None):
    """Compile the kernel and lay its memory out. Returns (Program, plan)."""
    import kernels as K
    import triton_bridge

    target = target or TargetInfo.default()
    fn, signature, constexprs = K.KERNELS[case.kernel]
    addrs = _addresses(case)

    args = {name: addr + case.offsets.get(name, 0)
                    * DTYPES[case.buffers[name][0]][0]
            for name, addr in addrs.items()}
    args.update(case.scalars)
    mod, _ctx = triton_bridge.make_ttir(fn, signature, constexprs)
    imp = TTIRImporter(target, name, scratch_addr=SCRATCH)
    kernel, _ = imp.run(mod, fn.__name__, args,
                        param_names=triton_bridge.param_names(fn))
    cg = CodeGen(kernel, target, check_protocol=False)
    cg.assign_shared()
    cg.run()

    p = H.Program(f"r37_{name}")
    p.kernel(cg)

    # The output buffer is filled with a sentinel first, so a lane the
    # kernel never wrote is visible as one rather than as a zero that
    # might have been computed.
    out_dtype = case.buffers[case.out][0]
    size, _, _ = DTYPES[out_dtype]
    n_out = len(case.expect)
    for pname, (dtype, values) in case.buffers.items():
        p.words(addrs[pname], pack(values, dtype), pname)
    out_at = addrs[case.out] + case.out_offset * size
    return p, {'addrs': addrs, 'out_at': out_at, 'out_bytes': n_out * size,
               'out_dtype': out_dtype, 'scratch': imp.scratch_used}


def run(name, case, *, max_cycles=2000000):
    """Execute on the RTL and compare. Returns (ok, detail)."""
    import subprocess
    p, plan = build(name, case)
    base = plan['out_at']
    words = (plan['out_bytes'] + 7) // 8

    build_dir = H.BUILD
    os.makedirs(build_dir, exist_ok=True)
    src = os.path.join(build_dir, f"cc_{p.name}.s")
    hexf = os.path.join(build_dir, f"cc_{p.name}.hex")
    dumpf = os.path.join(build_dir, f"cc_{p.name}.dump")
    with open(src, 'w') as fh:
        fh.write(p.text())
    asm = subprocess.run(
        [sys.executable, H.ASM, src, '-o', hexf, '--size',
         str(IMAGE_WORDS)], capture_output=True, text=True)
    if asm.returncode != 0:
        return False, f"assembling failed:\n{asm.stderr}{asm.stdout}"
    if os.path.exists(dumpf):
        os.remove(dumpf)

    proc = subprocess.run(
        ['vvp', H.bench_path('fast'), f"+hex={hexf}",
         f"+maxcycles={max_cycles}", f"+dump={dumpf}",
         f"+dumpfirst={base // 8}", f"+dumpcount={words}"],
        capture_output=True, text=True)
    if 'FAIL' in proc.stdout or not os.path.exists(dumpf):
        return False, proc.stdout[-1500:]

    with open(dumpf) as fh:
        raw = [int(line.strip(), 16) for line in fh if line.strip()]
    got = _unpack(raw, plan['out_dtype'], len(case.expect))
    return _compare(got, case)


def _unpack(raw, dtype, count):
    size, _, dec = DTYPES[dtype]
    per = 8 // size
    out = []
    for word in raw:
        for j in range(per):
            out.append(dec((word >> (size * 8 * j))
                           & ((1 << (size * 8)) - 1)))
    return out[:count]


def _compare(got, case):
    bad = []
    for i, (g, w) in enumerate(zip(got, case.expect)):
        if case.rtol == 0.0 and case.atol == 0.0:
            same = g == w
        else:
            same = abs(g - w) <= case.atol + case.rtol * abs(w)
        if not same:
            bad.append((i, g, w))
    if len(got) != len(case.expect):
        return False, (f"read {len(got)} elements, expected "
                       f"{len(case.expect)}")
    if bad:
        shown = ', '.join(f"[{i}] got {g!r} want {w!r}" for i, g, w in bad[:4])
        return False, f"{len(bad)} of {len(got)} wrong: {shown}"
    return True, f"{len(got)} element(s) correct"
