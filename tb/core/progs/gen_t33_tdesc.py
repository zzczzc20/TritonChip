#!/usr/bin/env python3
"""
Generate the rank 1-5 tensor-transfer tests (ISA sec 9D-9E, R38 tensor
descriptors aligned with the TMA contract Triton exposes on NVIDIA GPUs).

  t33_tdesc_load.s    t.ld.async at ranks 1-5 and element sizes of 1, 2, 4
                      and 8 bytes: coordinates that are negative, misaligned
                      and past the end in every dimension, bases and strides
                      off 8-byte boundaries, zero and NaN
                      padding, a box entirely outside the tensor, and padding
                      mode 2 on a box that is inside. Every lane of the window
                      is checked, and the lanes after the box must be
                      untouched.
  t34_tdesc_store.s   t.st.async at the same ranks: every word of a region
                      around the tensor is checked afterwards, so a byte
                      written outside the tensor or outside the box, or a byte
                      inside both that was not written, is a failure.
  t35_tdesc_f*.s      one program per refusal, each DESCRIPTOR_FAULT
                      (+expectfault=04): rank 6, padding mode 2 with an
                      out-of-bounds element, NaN padding of an integer
                      tensor, coordinates past S63, a block dimension over
                      256, and the reserved padding mode 1.

The engine moves bytes and never interprets them, so the tensors hold a
byte pattern and the expected results are computed from the definition:

  box element (i0..ir-1) <-> tensor element (c0+i0, ..., cr-1+ir-1),
  in bounds when 0 <= c_d + i_d < extent_d for every d;
  the window holds the box flat and row-major;
  a load fills an out-of-bounds element (zero, or the dtype's quiet NaN);
  a store does not write it.
"""
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

DTYPE = {'f16': (0, 2), 'bf16': (1, 2), 'f32': (2, 4), 'i8': (3, 1), 'i16': (4, 2),
         'i32': (5, 4), 'i64': (6, 8), 'e4m3': (7, 1), 'e5m2': (8, 1), 'f64': (9, 8)}
NAN = {'f16': 0x7E00, 'bf16': 0x7FC0, 'f32': 0x7FC00000, 'f64': 0x7FF8000000000000,
       'e4m3': 0x7F, 'e5m2': 0x7F}
SUFFIX = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}
PAD_ZERO, PAD_CLAMP, PAD_FAULT, PAD_NAN = 0, 1, 2, 3

WIN_BASE = 32          # the window is v32..v63
DESC = 0xC000          # TensorMaps, 256 bytes apart (code below, up to 48 KiB)
DATA = 0xD000          # tensors and window sources


def pattern(addr):
    return (addr * 37 + (addr >> 8) * 11 + 5) & 0xFF


def tensormap(base, extents, strides, block, dtype, pad):
    rank = len(extents)
    w = [0] * 32
    w[0] = base
    for i, e in enumerate(extents):
        w[1 + i] = e
    for i, s in enumerate(strides):
        w[6 + i] = s & (2**64 - 1)
    for i, b in enumerate(block):
        word, half = divmod(i, 2)
        w[11 + word] |= b << (32 * half)
    w[13] |= (rank << 32) | (DTYPE[dtype][0] << 40) | (pad << 48)
    return w


class Mem:
    """The data the program starts with, by byte address."""

    def __init__(self):
        self.b = {}

    def fill_pattern(self, lo, hi):
        for a in range(lo, hi):
            self.b[a] = pattern(a)

    def put_words(self, addr, words):
        for i, x in enumerate(words):
            for k in range(8):
                self.b[addr + 8 * i + k] = (x >> (8 * k)) & 0xFF

    def get(self, a):
        return self.b.get(a, 0)

    def emit(self, w):
        if not self.b:
            return
        words = sorted({a // 8 for a in self.b})
        run = []
        for wd in words:
            if run and wd != run[-1] + 1:
                self._emit_run(w, run)
                run = []
            run.append(wd)
        self._emit_run(w, run)

    def _emit_run(self, w, run):
        w(f".org {run[0] * 8:#x}")
        for wd in run:
            x = 0
            for k in range(8):
                x |= self.get(wd * 8 + k) << (8 * k)
            w(f"    .dword 0x{x:016x}")


def box_elements(block):
    """Box indices in window (flat row-major) order."""
    idx = [[]]
    for b in block:
        idx = [i + [j] for i in idx for j in range(b)]
    return idx


def tensor_span(base, extents, strides, esz):
    hi = base + sum((e - 1) * s for e, s in zip(extents, strides)) + esz
    return base, hi


class Case:
    def __init__(self, name, dtype, extents, strides_el, block, coords, pad=PAD_ZERO,
                 base_off=0):
        self.name, self.dtype = name, dtype
        self.esz = DTYPE[dtype][1]
        self.extents, self.block, self.coords, self.pad = extents, block, coords, pad
        self.strides = [s * self.esz for s in strides_el]
        self.base_off = base_off
        self.rank = len(extents)
        self.box_bytes = 1
        for b in block:
            self.box_bytes *= b
        self.box_bytes *= self.esz
        self.nregs = (self.box_bytes + 63) // 64

    def element(self, mem, idx):
        g = [c + i for c, i in zip(self.coords, idx)]
        if all(0 <= x < e for x, e in zip(g, self.extents)):
            a = self.base + sum(x * s for x, s in zip(g, self.strides))
            return [mem.get(a + k) for k in range(self.esz)], a
        if self.pad == PAD_NAN:
            v = NAN[self.dtype]
        else:
            v = 0
        return [(v >> (8 * k)) & 0xFF for k in range(self.esz)], None


def load_program(cases):
    out, mem, n = [], Mem(), 0
    w = out.append
    w("// t33_tdesc_load -- generated by gen_t33_tdesc.py; do not edit by hand.")
    w("// ISA sec 9E: rank 1-5 t.ld.async, every window lane checked.")
    w(".org 0")
    w("start:")
    w("    mb.init   b0")
    cursor = DATA
    for ci, c in enumerate(cases):
        c.base = cursor + c.base_off
        lo, hi = tensor_span(c.base, c.extents, c.strides, c.esz)
        # Pattern over the tensor's whole span plus a word either side, so a
        # read that strays is reading something recognisable.
        mem.fill_pattern(lo - 16, hi + 16)
        cursor = (hi + 64 + 63) // 64 * 64
        d = DESC + 0x100 * ci
        mem.put_words(d, tensormap(c.base, c.extents, c.strides, c.block, c.dtype, c.pad))
        w(f"    // ---- {c.name}: rank {c.rank} {c.dtype} extents {c.extents} "
          f"block {c.block} at {c.coords} pad {c.pad} ----")
        w(f"    .li       s1, #{d}")
        w(f"    td.load   d1, [s1]")
        regs = c.nregs + 1 if c.nregs < 32 else c.nregs
        w(f"    w.def     w1, #{WIN_BASE}, #{c.nregs}")
        for v in range(regs):
            w(f"    v.movi    v{WIN_BASE + v}, #-1")
        w("    mb.arm    s3, b0, #0, #1")
        for k, x in enumerate(c.coords):
            w(f"    .li       s{10 + k}, #{x}")
        w(f"    t.ld.async.{SUFFIX[c.esz]} w1, d1, s10, s3")
        w("    mb.wait   s3")
        win = []
        for idx in box_elements(c.block):
            win += c.element(mem, idx)[0]
        win += [0xFF] * (regs * 64 - len(win))       # untouched past the box
        for li in range(len(win) // 4):
            val = int.from_bytes(bytes(win[4 * li:4 * li + 4]), 'little')
            n += 1
            w(f"    .assert_lane v{WIN_BASE + li // 16}, #{li % 16}, #{val:#010x}, #{n}")
        w("    w.release w1")
        w("    td.release d1")
    w("    exit")
    w("")
    mem.emit(w)
    return out, n


def store_program(cases):
    out, mem, n = [], Mem(), 0
    w = out.append
    w("// t34_tdesc_store -- generated by gen_t33_tdesc.py; do not edit by hand.")
    w("// ISA sec 9E: rank 1-5 t.st.async; every word around the tensor checked.")
    w(".org 0")
    w("start:")
    w("    mb.init   b0")
    cursor = DATA
    SENT = 0xA5
    for ci, c in enumerate(cases):
        c.base = cursor + c.base_off
        lo, hi = tensor_span(c.base, c.extents, c.strides, c.esz)
        rlo, rhi = (lo - 16) // 8 * 8, (hi + 16 + 7) // 8 * 8
        for a in range(rlo, rhi):
            mem.b[a] = SENT                          # the region starts as sentinel
        cursor = (rhi + 64 + 63) // 64 * 64
        src = cursor                                 # what the window is loaded from
        srcb = [pattern(a) for a in range(src, src + c.nregs * 64)]
        for k, x in enumerate(srcb):
            mem.b[src + k] = x
        cursor = (src + c.nregs * 64 + 64 + 63) // 64 * 64
        d = DESC + 0x100 * ci
        mem.put_words(d, tensormap(c.base, c.extents, c.strides, c.block, c.dtype, c.pad))
        # expected memory after the store
        exp = {a: SENT for a in range(rlo, rhi)}
        for e, idx in enumerate(box_elements(c.block)):
            _, a = c.element(mem, idx)
            if a is None:
                continue
            for k in range(c.esz):
                exp[a + k] = srcb[e * c.esz + k]
        w(f"    // ---- {c.name}: rank {c.rank} {c.dtype} extents {c.extents} "
          f"block {c.block} at {c.coords} ----")
        w(f"    .li       s20, #{src}")
        for v in range(c.nregs):
            w(f"    v.ld.w    v{WIN_BASE + v}, [s20, #{64 * v}]")
        w(f"    .li       s1, #{d}")
        w(f"    td.load   d1, [s1]")
        w(f"    w.def     w1, #{WIN_BASE}, #{c.nregs}")
        w("    mb.arm    s3, b0, #0, #1")
        for k, x in enumerate(c.coords):
            w(f"    .li       s{10 + k}, #{x}")
        w(f"    t.st.async.{SUFFIX[c.esz]} w1, d1, s10, s3")
        w("    mb.wait   s3")
        w(f"    .li       s21, #{rlo}")
        for a in range(rlo, rhi, 8):
            val = int.from_bytes(bytes(exp[a + k] for k in range(8)), 'little')
            n += 1
            w(f"    s.ld.d    s5, [s21, #{a - rlo}]")
            w(f"    .assert_eq s5, #{val:#018x}, #{n}")
        w("    w.release w1")
        w("    td.release d1")
    w("    exit")
    w("")
    mem.emit(w)
    return out, n


def fault_program(name, why, c, sb=10):
    out, mem = [], Mem()
    w = out.append
    w(f"// {name} -- generated by gen_t33_tdesc.py; do not edit by hand.")
    w(f"// {why}")
    w(".org 0")
    w("start:")
    c.base = DATA
    lo, hi = tensor_span(c.base, c.extents, c.strides, c.esz)
    mem.fill_pattern(lo, hi)
    mem.put_words(DESC, tensormap(c.base, c.extents, c.strides, c.block, c.dtype, c.pad))
    w(f"    .li       s1, #{DESC}")
    w("    td.load   d1, [s1]")
    w(f"    w.def     w1, #{WIN_BASE}, #{max(1, min(c.nregs, 32))}")
    w("    mb.init   b0")
    w("    mb.arm    s3, b0, #0, #1")
    for k, x in enumerate(c.coords):
        if sb + k <= 63:
            w(f"    .li       s{sb + k}, #{x}")
    w(f"    t.ld.async.{SUFFIX[c.esz]} w1, d1, s{sb}, s3")
    w("    mb.wait   s3")
    w("    trap      #99                     // must be unreachable")
    w("")
    mem.emit(w)
    return out


LOADS = [
    Case('rank1 f32 inside', 'f32', [40], [1], [16], [8]),
    Case('rank1 f16 past the end', 'f16', [37], [1], [32], [20]),
    Case('rank1 f16 negative, misaligned', 'f16', [40], [1], [16], [-5]),
    Case('rank1 i8 misaligned', 'i8', [64], [1], [16], [3]),
    Case('rank2 f32 edges', 'f32', [10, 13], [16, 1], [4, 8], [7, 9]),
    Case('rank2 f32 entirely outside', 'f32', [10, 13], [16, 1], [2, 8], [-4, 2]),
    Case('rank3 f16', 'f16', [3, 5, 24], [5 * 24, 24, 1], [2, 4, 16], [1, 3, -3]),
    Case('rank4 f32', 'f32', [2, 3, 4, 6], [3 * 4 * 8, 4 * 8, 8, 1], [2, 2, 2, 4], [0, 2, 3, 4]),
    Case('rank5 i16', 'i16', [2, 2, 2, 3, 10], [2 * 2 * 3 * 12, 2 * 3 * 12, 3 * 12, 12, 1],
         [1, 2, 2, 2, 8], [1, 0, 1, 2, 5]),
    Case('rank5 f16 NaN', 'f16', [2, 2, 2, 3, 10], [2 * 2 * 3 * 12, 2 * 3 * 12, 3 * 12, 12, 1],
         [2, 1, 2, 2, 8], [-1, 1, 1, 2, -3], pad=PAD_NAN),
    Case('rank3 f64 NaN', 'f64', [2, 3, 5], [3 * 6, 6, 1], [2, 2, 4], [1, -1, 3], pad=PAD_NAN),
    Case('rank2 e4m3 NaN, 1-byte misaligned', 'e4m3', [3, 40], [48, 1], [2, 16], [0, -7],
         pad=PAD_NAN),
    Case('rank2 bf16 fault mode, inside', 'bf16', [6, 20], [24, 1], [4, 8], [1, 4],
         pad=PAD_FAULT),
    Case('rank2 f16 strided rows', 'f16', [5, 16], [40, 1], [4, 16], [2, -1]),
    # Base and strides off 8-byte boundaries: every row starts at a
    # different offset within its word.
    Case('rank2 f16 misaligned base and stride', 'f16', [5, 20], [13, 1], [4, 16], [1, 2],
         base_off=2),
    Case('rank3 i8 odd strides', 'i8', [3, 3, 20], [67, 21, 1], [2, 2, 16], [1, -1, 3],
         base_off=5),
]

STORES = [
    Case('rank1 f32 inside', 'f32', [40], [1], [16], [8]),
    Case('rank1 f16 negative, misaligned', 'f16', [40], [1], [32], [-3]),
    Case('rank2 f32 edges, misaligned', 'f32', [10, 13], [16, 1], [4, 8], [7, 9]),
    Case('rank3 i8 misaligned', 'i8', [3, 4, 30], [4 * 40, 40, 1], [2, 2, 16], [2, 3, 21]),
    Case('rank4 f16', 'f16', [2, 3, 3, 12], [3 * 3 * 16, 3 * 16, 16, 1], [2, 2, 2, 8],
         [-1, 1, 2, 6]),
    Case('rank5 f32', 'f32', [2, 2, 2, 2, 6], [2 * 2 * 2 * 8, 2 * 2 * 8, 2 * 8, 8, 1],
         [2, 2, 1, 2, 4], [0, 1, 1, -1, 3]),
    Case('rank2 f64 negative row', 'f64', [3, 5], [6, 1], [2, 4], [-1, 2]),
    Case('rank2 f16 misaligned base and stride', 'f16', [5, 20], [13, 1], [4, 16], [1, 2],
         base_off=2),
    Case('rank3 i8 odd strides', 'i8', [3, 3, 20], [67, 21, 1], [2, 2, 16], [1, -1, 3],
         base_off=5),
]


def faults():
    yield ('t35_tdesc_frank', 'Rank 6 is above TENSOR_MAX_RANK.',
           Case('rank6', 'f32', [2, 2, 2, 2, 2, 8], [128, 64, 32, 16, 8, 1],
                [1, 1, 1, 1, 1, 8], [0, 0, 0, 0, 0, 0]), 10)
    yield ('t35_tdesc_fpad2', 'Padding mode 2: one element out of bounds faults.',
           Case('pad2', 'f32', [4, 10], [16, 1], [2, 8], [3, 0], pad=PAD_FAULT), 10)
    yield ('t35_tdesc_fnanint', 'NaN padding of an integer tensor is refused.',
           Case('naninit', 'i32', [4, 10], [16, 1], [2, 8], [0, 0], pad=PAD_NAN), 10)
    yield ('t35_tdesc_fsb', 'Rank 5 from S61: the coordinates would run past S63.',
           Case('sb', 'f32', [1, 1, 1, 2, 8], [128, 64, 32, 16, 1], [1, 1, 1, 1, 8],
                [0, 0, 0, 0, 0]), 61)
    yield ('t35_tdesc_fblock', 'A block dimension of 512 is over the 256 limit.',
           Case('block', 'i8', [1, 1024], [1024, 1], [1, 512], [0, 0]), 10)
    yield ('t35_tdesc_fclamp', 'Padding mode 1 (clamp) is reserved.',
           Case('clamp', 'f32', [4, 10], [16, 1], [2, 8], [0, 0], pad=PAD_CLAMP), 10)


FAULT_NAMES = [f[0] for f in faults()]


def main():
    prog, n = load_program(LOADS)
    open(os.path.join(HERE, 't33_tdesc_load.s'), 'w').write('\n'.join(prog) + '\n')
    sprog, m = store_program(STORES)
    open(os.path.join(HERE, 't34_tdesc_store.s'), 'w').write('\n'.join(sprog) + '\n')
    for name, why, c, sb in faults():
        open(os.path.join(HERE, f'{name}.s'), 'w').write(
            '\n'.join(fault_program(name, why, c, sb)) + '\n')
    print(f"wrote t33_tdesc_load.s ({n} lane checks), t34_tdesc_store.s ({m} word checks), "
          f"{len(FAULT_NAMES)} fault programs")


if __name__ == '__main__':
    main()
