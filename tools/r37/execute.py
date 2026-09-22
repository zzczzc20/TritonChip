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
import math
import re
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

# Expanded control flow and row-wise tile lowering need more code space
# than the small input tensors. Keep code, buffers and scratch disjoint.
IMAGE_WORDS = 65536
BUF_BASE = 0x20000
BUF_STRIDE = 0x400          # 1 KiB a buffer: a 16x16 f32 tile is 1 KiB
SCRATCH = 0x30000           # descriptors and repacks, past every buffer
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
    'i64': (8, lambda v: int(v) & ((1 << 64) - 1),
            lambda b: b - (1 << 64) if b >> 63 else b),
    'i8': (1, lambda v: int(v) & 0xFF,
           lambda b: b - (1 << 8) if b >> 7 else b),
}


# Use the existing exact format oracle only for host-side packing. Expected
# dot products in edge_cases.py are computed independently from the inputs.
def _fp8_encode(fmt, value):
    from fractions import Fraction
    import floatfmt
    return floatfmt.encode(fmt, abs(Fraction(value)), int(value < 0))


def _fp8_decode(fmt, bits):
    import floatfmt
    value = floatfmt.decode(fmt, bits)
    if value == 'nan':
        return math.nan
    if isinstance(value, tuple):
        return -math.inf if value[1] else math.inf
    return float(value)


for _dtype in ('e4m3', 'e5m2'):
    DTYPES[_dtype] = (1, lambda v, fmt=_dtype: _fp8_encode(fmt, v),
                     lambda b, fmt=_dtype: _fp8_decode(fmt, b))


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
                 rtol=0.0, atol=0.0, offsets=None, out_offset=0, note=None,
                 extra_outputs=None, preserve_output=False, program_id=(0, 0, 0),
                 signature=None, constexprs=None, kernel_spec=None, descriptors=None,
                 feeds=None):
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
        # Additional (buffer, element offset, expected values) outputs use
        # exact comparison, e.g. the expert index returned alongside a weight.
        self.extra_outputs = extra_outputs or []
        self.preserve_output = preserve_output  # atomic read/modify/write
        self.program_id = program_id
        self.kernel_spec = kernel_spec
        self.descriptors = descriptors or {}
        self.feeds = dict(feeds or {})  # buffer -> (preceding case name, buffer)
        self.signature = signature or {}
        self.constexprs = constexprs or {}


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
        nbytes = len(values) * DTYPES[dtype][0] + 8
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
    fn, signature, constexprs = case.kernel_spec or K.KERNELS[case.kernel]
    signature = dict(signature, **case.signature)
    constexprs = dict(constexprs, **case.constexprs)
    addrs = _addresses(case)

    args = {name: addr + case.offsets.get(name, 0)
                    * DTYPES[case.buffers[name][0]][0]
            for name, addr in addrs.items()}
    args = {name: value for name, value in args.items() if name in signature}
    args.update(case.scalars)
    descriptors = []
    for i, (desc_name, desc) in enumerate(case.descriptors.items()):
        from tnca_cc.tiles import tensormap_words
        addr = SCRATCH + i * 256
        # i32/FP8 descriptors use software row accesses. Their storage
        # metadata shares the hardware encoding with an equal-width type;
        # the original TTIR descriptor retains the actual arithmetic type.
        elem = {'fp16': 'f16', 'fp32': 'f32', 'i32': 'f32',
                'i8': 'i8', 'e4m3': 'i8', 'e5m2': 'i8'}[case.buffers[desc['buffer']][0]]
        size = DTYPES[case.buffers[desc['buffer']][0]][0]
        words = tensormap_words(base=addrs[desc['buffer']], extents=desc['shape'],
                                byte_strides=[s * size for s in desc['strides']],
                                block_shape=desc['block'], dtype=elem)
        descriptors.append((addr, words, desc_name))
        args[desc_name] = {'address': addr, 'shape': desc['shape'], 'strides': desc['strides']}
    mod, _ctx = triton_bridge.make_ttir(fn, signature, constexprs)
    imp = TTIRImporter(target, name, scratch_addr=SCRATCH + len(descriptors) * 256)
    kernel, _ = imp.run(mod, fn.__name__, args,
                        param_names=triton_bridge.param_names(fn))
    if imp.scratch_addr + imp.scratch_used > IMAGE_WORDS * 8:
        raise ValueError('compiler scratch exceeds the simulation memory')
    cg = CodeGen(kernel, target, check_protocol=False)
    cg.assign_shared()
    cg.run()

    p = H.Program(f"r37_{name}")
    p.kernel(cg)

    # Dump all buffers, including eight guard bytes after each one. Check
    # inputs and output tails bit-for-bit, as well as the numerical outputs.
    outputs = [(case.out, case.out_offset, case.expect, case.rtol, case.atol)]
    outputs += [(buf, offset, list(expect), 0.0, 0.0)
                for buf, offset, expect in case.extra_outputs]
    initial = {}
    regions = []
    for pname, (dtype, values) in case.buffers.items():
        data = bytearray().join(w.to_bytes(8, 'little') for w in pack(values, dtype))
        nbytes = len(values) * DTYPES[dtype][0]
        data = data[:nbytes] + bytearray(b'\xa5' * 8)
        for buf, offset, expect, rtol, atol in outputs:
            if buf != pname:
                continue
            size = DTYPES[dtype][0]
            lo, hi = offset * size, (offset + len(expect)) * size
            if lo < 0 or hi > nbytes:
                raise ValueError(f"output {buf}[{offset}:] exceeds its buffer")
            regions.append((addrs[buf] + lo, dtype, expect, rtol, atol, buf))
            if not case.preserve_output:
                # Poison even outputs expected to be zero. A missing store
                # must never pass merely because the input image held zero.
                data[lo:hi] = bytes([0xCD]) * (hi - lo)
        initial.update((addrs[pname] + i, v) for i, v in enumerate(data))
        data += bytes((-len(data)) % 8)
        p.words(addrs[pname], [int.from_bytes(data[i:i + 8], 'little')
                              for i in range(0, len(data), 8)], pname)
    for addr, dtype, expect, *_ in regions:
        for i in range(len(expect) * DTYPES[dtype][0]):
            initial.pop(addr + i)
    for addr, words, label in descriptors:
        p.words(addr, words, label)
    base = min(addrs.values())
    end = max(initial) + 1
    return p, {'dump_base': base, 'dump_words': (end - base + 7) // 8,
               'unchanged': initial, 'outputs': regions,
               'scratch': imp.scratch_used}


def run(name, case, *, max_cycles=2000000, bench='fast', capture=None):
    """Execute on the RTL and compare. Returns (ok, detail)."""
    import subprocess
    p, plan = build(name, case)
    base = plan['dump_base']
    words = plan['dump_words']

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
        ['vvp', H.bench_path(bench), f"+hex={hexf}",
         f"+maxcycles={max_cycles}", f"+dump={dumpf}",
         f"+dumpfirst={base // 8}", f"+dumpcount={words}"]
        + [f"+pid_{axis}={value}"
           for axis, value in zip('xyz', case.program_id)],
        capture_output=True, text=True)
    if (proc.returncode != 0 or 'FAIL' in proc.stdout
            or not re.search(r'^PASS$', proc.stdout, re.MULTILINE)
            or not os.path.exists(dumpf)):
        return False, (proc.stdout + proc.stderr)[-1500:]

    with open(dumpf) as fh:
        raw = [int(line.strip(), 16) for line in fh if line.strip()]
    if len(raw) != words:
        return False, f"read {len(raw)} words, expected {words}"
    data = b''.join(w.to_bytes(8, 'little') for w in raw)
    for addr, value in plan['unchanged'].items():
        if data[addr - base] != value:
            return False, f"input, output tail or guard overwritten at {addr:#x}"
    count = 0
    for addr, dtype, expect, rtol, atol, buf in plan['outputs']:
        size, _, dec = DTYPES[dtype]
        start = addr - base
        got = [dec(int.from_bytes(data[i:i + size], 'little'))
               for i in range(start, start + size * len(expect), size)]
        ok, detail = _compare_values(got, expect, rtol, atol)
        if not ok:
            return False, f"{buf}: {detail}"
        count += len(expect)
    if capture is not None:
        for buf, addr in _addresses(case).items():
            dtype, values = case.buffers[buf]
            size, _, dec = DTYPES[dtype]
            start = addr - base
            capture[buf] = (dtype, [dec(int.from_bytes(data[i:i+size], 'little'))
                                   for i in range(start, start+size*len(values), size)])
    return True, f"{count} element(s) correct; inputs, tails and guards intact"


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
    return _compare_values(got, case.expect, case.rtol, case.atol)


def _compare_values(got, expect, rtol=0.0, atol=0.0):
    bad = []
    for i, (g, w) in enumerate(zip(got, expect)):
        # Equal infinities are valid; a finite result must never pass against
        # infinity because both sides of the tolerance inequality are inf.
        if math.isnan(w):
            same = math.isnan(g)
        elif not math.isfinite(g) or not math.isfinite(w):
            same = g == w
        else:
            same = abs(g - w) <= atol + rtol * abs(w)
        if not same:
            bad.append((i, g, w))
    if len(got) != len(expect):
        return False, f"read {len(got)} elements, expected {len(expect)}"
    if bad:
        shown = ', '.join(f"[{i}] got {g!r} want {w!r}" for i, g, w in bad[:4])
        return False, f"{len(bad)} of {len(got)} wrong: {shown}"
    return True, f"{len(got)} element(s) correct"
