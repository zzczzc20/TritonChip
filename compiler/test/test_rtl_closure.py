#!/usr/bin/env python3
"""
Compiled output, run on the RTL (compiler_overview.md sec 16.2, 第三层:
"编译产物 RTL 差分").

sec 15 C1 sets the bar: 不修改输出汇编，不人工补 barrier，直接运行编译产物.
So this takes what lower_elementwise() emits, appends only data and assertions
-- never an edit to the emitted instructions -- assembles it with the existing
tools/asm/tnca_asm.py, and runs it on the Icarus model of the core.

The reference it is checked against is tb/core/progs/t16_vecadd.s, written by
hand first. sec 11.1 keeps that comparison available precisely so a compiler
bug and a machine encoding bug can be told apart.

Skips rather than fails when iverilog or the built bench is absent, so the
compiler tests still run on a machine with no simulator.
"""
import os
import re
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

from fp32 import f2b  # noqa: E402
from tnca_cc import TargetInfo  # noqa: E402
from tnca_cc.lower import ElementwiseKernel, lower_elementwise  # noqa: E402

BENCH = os.path.join(ROOT, 'sim', 'build', 'tb_core.vvp')
ASM = os.path.join(ROOT, 'tools', 'asm', 'tnca_asm.py')
OUT = os.path.join(ROOT, 'sim', 'build', 'compiled_vecadd')

BLOCK = 16
N = 40
NUM_PROGRAMS = (N + BLOCK - 1) // BLOCK
PADDED = NUM_PROGRAMS * BLOCK
X_BASE, Y_BASE, O_BASE = 0x4000, 0x4400, 0x4800
SENTINEL = 0xdeadbeef


def xval(i):
    return float(i) * 0.5


def yval(i):
    return float(i) * 0.25 + 1.0


def build_kernel():
    return ElementwiseKernel(
        name='add_kernel', inputs=['x', 'y'], output='out',
        block=BLOCK, op='add', dtype='f32', masked=True, other=0,
        grid_loop=True, num_programs=NUM_PROGRAMS, n_elements=N,
        arg_addresses={'x': X_BASE, 'y': Y_BASE, 'out': O_BASE})


def pack32(values):
    words = []
    for k in range(0, len(values), 2):
        hi = values[k + 1] if k + 1 < len(values) else 0
        words.append((hi << 32) | values[k])
    return words


def full_program(target):
    """The emitted kernel, plus assertions and data. No edits to the kernel."""
    b, alloc = lower_elementwise(build_kernel(), target)

    lines = ['// Emitted by tnca_cc.lower; assertions and data appended, the',
             '// instructions untouched (compiler_overview.md sec 15 C1).',
             '.org 0', 'start:']
    lines.append(b.text().rstrip('\n'))

    # Checks use the assembler's pseudo-ops, which live outside the kernel.
    n = 1
    lines.append(f"    .li       s20, #{O_BASE}")
    for i in range(N):
        lines.append(f"    s.ld.w    s21, [s20, #{i * 4}]")
        lines.append(f"    .assert_eq s21, #{f2b(xval(i) + yval(i)):#010x}, #{n}")
        n += 1
    # The masked tail must not have been stored.
    for i in range(N, PADDED):
        lines.append(f"    s.ld.w    s21, [s20, #{i * 4}]")
        lines.append(f"    .assert_eq s21, #{SENTINEL:#010x}, #{n}")
        n += 1
    lines.append("    exit")
    lines.append("")

    def emit_words(addr, words, label):
        lines.append(f".org {addr:#x}")
        lines.append(f"{label}:")
        for word in words:
            lines.append(f"    .dword 0x{word & ((1 << 64) - 1):016x}")
        lines.append("")

    emit_words(X_BASE, pack32([f2b(xval(i)) for i in range(PADDED)]), 'xs')
    emit_words(Y_BASE, pack32([f2b(yval(i)) for i in range(PADDED)]), 'ys')
    emit_words(O_BASE, pack32([SENTINEL] * PADDED), 'os')

    return '\n'.join(lines) + '\n', alloc


class TestCompiledVectorAdd(unittest.TestCase):
    def setUp(self):
        self.target = TargetInfo.default()

    def test_emitted_assembly_matches_the_hand_written_reference(self):
        """sec 11.1: compare against hand-written assembly, instruction by
        instruction. Register numbering may differ; the sequence may not."""
        b, _ = lower_elementwise(build_kernel(), self.target)
        def is_instruction(line):
            line = line.split('//')[0].strip()
            return bool(line) and not line.endswith(':')

        emitted = [l for l in b.text().splitlines() if is_instruction(l)]

        ref_path = os.path.join(ROOT, 'tb', 'core', 'progs', 't16_vecadd.s')
        if not os.path.exists(ref_path):
            self.skipTest('reference program not generated')
        ref = []
        with open(ref_path) as fh:
            ref_lines = fh.readlines()
        for line in ref_lines:
            stripped = line.split('//')[0].strip()
            if (not is_instruction(line) or stripped.startswith('.org')
                    or stripped.startswith('.dword')):
                continue
            ref.append(line)
        # The reference continues into its assertion epilogue. Cut at the end
        # of the kernel body -- the loop branch -- rather than at len(emitted),
        # which would silently absorb a trailing instruction the lowering had
        # dropped.
        end = next(i for i, l in enumerate(ref) if 'br.nz' in l and 'loop' in l)
        ref = ref[:end + 1]
        self.assertEqual(len(emitted), len(ref),
                         'the lowering emitted a different number of '
                         'instructions than the hand-written reference')

        def shape(line):
            """Opcode and operands, with comments and register numbers erased.

            Register numbering is an allocator decision and may differ; the
            instruction sequence may not.
            """
            line = line.split('//')[0].strip()
            line = re.sub(r'\b([svpabdw])\d+\b', r'\1?', line)
            return re.sub(r'\s+', ' ', line)

        self.assertEqual([shape(x) for x in emitted], [shape(x) for x in ref])

    def test_fits_the_target(self):
        _, alloc = lower_elementwise(build_kernel(), self.target)
        self.assertLessEqual(alloc.peak['v'], self.target.vregs_per_rctx)
        self.assertLessEqual(alloc.peak['s'], self.target.sregs_per_rctx)

    def test_runs_on_the_rtl(self):
        """The whole point of C1: run the compiler's output unmodified."""
        if not os.path.exists(BENCH):
            self.skipTest(f"{BENCH} not built; run `make core` first")
        if not shutil.which('vvp'):
            self.skipTest('iverilog/vvp not installed')

        text, _ = full_program(self.target)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        src = OUT + '.s'
        with open(src, 'w') as fh:
            fh.write(text)

        asm = subprocess.run(
            [sys.executable, ASM, src, '-o', OUT + '.hex', '--size', '16384'],
            capture_output=True, text=True)
        self.assertEqual(asm.returncode, 0,
                         f"assembling the emitted kernel failed:\n{asm.stderr}")

        run = subprocess.run(
            ['vvp', BENCH, f"+hex={OUT}.hex", '+maxcycles=100000'],
            capture_output=True, text=True)
        self.assertIn('PASS', run.stdout,
                      f"compiled kernel failed on the RTL:\n{run.stdout}")
        self.assertNotIn('FAIL', run.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
