#!/usr/bin/env python3
"""
The whole flow, end to end: Triton source -> the four-tile machine -> numbers.

Everything else in this suite stops somewhere short of that. The compiler
tests check the IR. `test_triton_e2e` compiles a real @triton.jit kernel and
runs it, but on one core against a behavioural memory, and it checks the
result with assertions the same program carries. This runs the compiler output
on the **whole machine** -- four cores, four NUCA slices, the 2x2 mesh, DRAM --
as a grid, and then reads the answers back out of the machine and compares them
against a reference computed here in Python.

The distinction matters. A program that checks itself proves the arithmetic.
A checker that reads memory afterwards proves the *result reached memory*, and
in this machine that means it crossed a mesh, was allocated in some slice's
data array, possibly evicted a line, and stayed correct while three other cores
were doing the same thing. None of that is exercised by a self-checking
program on one core.

WHAT MAKES IT A GRID

`tl.program_id(0)` lowers to the `pid.x` special register, and the four-tile
system gives each tile its own. So the same image is a four-block grid: tile t
covers elements [t*BLOCK, (t+1)*BLOCK). Nothing in the compiler or the program
knows how many tiles there are -- the machine tells each core who it is, which
is what a grid launch is.

WHAT IS DELIBERATELY NOT MEASURED

Coverage. This is one kernel shape, one block size, one grid. RTL simulation of
the four-tile system runs at a few thousand cycles a second and a wide sweep
would take hours; the sweeps live in the unit benches, where they are cheap and
exhaustive. What this asserts is that the path from Triton source to a verified
number through the real machine is closed.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

import rtl_harness as H  # noqa: E402
import triton_bridge  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.codegen import CodeGen  # noqa: E402
from tnca_cc.frontend import TTIRImporter  # noqa: E402

TILES = 4
BLOCK = 16
N = TILES * BLOCK

# Data lives below 0x8000 so the bench's DRAM preload covers it.
X, Y, O = 0x4000, 0x4400, 0x4800
SENTINEL = 0xdeadbeef

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'BLOCK': 'constexpr'}


def f2b(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def b2f(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xffffffff))[0]


def xv(i):
    return i * 0.5 - 3.0


def yv(i):
    return i * 0.25 + 1.0


@unittest.skipUnless(triton_bridge.available(), 'triton not importable')
class TestTritonOnTheWholeMachine(unittest.TestCase):
    def setUp(self):
        # The system profile, not the default. They differ: the four-tile
        # machine reports four cores and no SFU, because rtl/tile/tnca_tile.v
        # elaborates its cores with HAS_SFU=0. Compiling against the
        # single-core target and running on the system is exactly the mismatch
        # the audit's F03 describes.
        self.target = TargetInfo.load('tnca-sim-2x2')
        if not H.bench_available('system'):
            self.skipTest("sim/build/tb_top.vvp not built; run `make top` first")

    def _image(self, fn, name):
        """Triton source -> TTIR -> tnca_cc -> assembly -> one image.

        `name` is the @triton.jit function's own name, because that is what it
        is called in the module the frontend walks.
        """
        mod, _ctx = triton_bridge.make_ttir(fn, SIG, {'BLOCK': BLOCK})
        kernel, _ = TTIRImporter(self.target, name).run(
            mod, name, {'x_ptr': X, 'y_ptr': Y, 'o_ptr': O, 'n': N})
        cg = CodeGen(kernel, self.target, check_protocol=False)
        cg.assign_shared()
        cg.run()

        p = H.Program(f"sys_{name}")
        p.kernel(cg)
        p.words(X, H.pack32([f2b(xv(i)) for i in range(N)]), 'xs')
        p.words(Y, H.pack32([f2b(yv(i)) for i in range(N)]), 'ys')
        p.words(O, H.pack32([SENTINEL] * N), 'os')
        return p

    def _check(self, name, fn, expect):
        p = self._image(fn, name)
        ok, words, out = H.run_system(p, dump_base=O, dump_words=N // 2)
        self.assertTrue(ok, f"the machine did not finish cleanly:\n{out}")
        self.assertEqual(len(words), N // 2,
                         f"dump did not come back:\n{out}")

        # Two 32-bit results per dumped 64-bit word, little-endian.
        got = []
        for wd in words:
            got.append(wd & 0xffffffff)
            got.append((wd >> 32) & 0xffffffff)

        untouched = [i for i, g in enumerate(got) if g == SENTINEL]
        self.assertEqual(untouched, [],
                         f"elements {untouched[:8]} were never written -- "
                         f"a block of the grid did not run")

        for i in range(N):
            with self.subTest(element=i):
                self.assertEqual(
                    got[i], f2b(expect(i)),
                    f"element {i}: got {b2f(got[i])!r}, "
                    f"want {expect(i)!r}")
        return out

    def test_add_kernel_as_a_four_block_grid(self):
        """The closure: Triton source, four cores, a number checked in Python.

        Every element must be right, and none may still hold the sentinel --
        which is what would remain if `pid.x` were not per tile and all four
        cores had computed the same block.
        """
        import kernels_triton as K
        out = self._check('add_kernel', K.add_kernel, lambda i: xv(i) + yv(i))
        # The mesh has to have been used: with a 64-byte line interleave the
        # four blocks do not all home locally, and a run where nothing left
        # its tile would not have tested the machine this test is named after.
        self.assertIn('remote_home=', out)
        remote = int(out.split('remote_home=')[1].split()[0])
        self.assertGreater(remote, 0, "no request ever crossed the mesh")

    def test_mul_add_kernel_as_a_four_block_grid(self):
        """A longer value graph, so the walk is a walk and not a match."""
        import kernels_triton as K
        self._check('mul_add_kernel', K.mul_add_kernel,
                    lambda i: xv(i) * yv(i) + xv(i))


if __name__ == '__main__':
    unittest.main(verbosity=2)
