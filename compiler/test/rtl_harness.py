"""
Run compiled kernels on the RTL (compiler_overview.md sec 16.2, third layer).

sec 15 C1: 不修改输出汇编，不人工补 barrier，直接运行编译产物. So the harness
appends data and assertions around the emitted kernel and never edits an
instruction inside it. It reuses the existing assembler and the existing
bench, which is what sec 16.2 asks for: 复用现有 assembler、现有 exact FP
oracle、原有 NUCA/mesh 测试环境.
"""
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

BENCH = os.path.join(ROOT, 'sim', 'build', 'tb_core.vvp')
ASM = os.path.join(ROOT, 'tools', 'asm', 'tnca_asm.py')
BUILD = os.path.join(ROOT, 'sim', 'build')

from tnca_cc.codegen import CodeGen  # noqa: E402


def available():
    return os.path.exists(BENCH) and shutil.which('vvp') is not None


def why_unavailable():
    if not os.path.exists(BENCH):
        return f"{BENCH} not built; run `make core` first"
    return 'iverilog/vvp not installed'


def compile_kernel(kernel, target, *, check_protocol=True, role=0):
    cg = CodeGen(kernel, target, check_protocol=check_protocol)
    cg.assign_shared()
    cg.run(role)
    return cg


def pack32(values):
    """Two 32-bit words per 64-bit doubleword, little-endian."""
    out = []
    for k in range(0, len(values), 2):
        hi = values[k + 1] if k + 1 < len(values) else 0
        out.append((hi << 32) | (values[k] & 0xffffffff))
    return out


def pack16(values):
    """Four 16-bit halves per doubleword."""
    out = []
    for k in range(0, len(values), 4):
        word = 0
        for j in range(4):
            if k + j < len(values):
                word |= (values[k + j] & 0xffff) << (16 * j)
        out.append(word)
    return out


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


class Program:
    """An emitted kernel plus the data and checks around it."""

    def __init__(self, name):
        self.name = name
        self.body = []
        self.checks = []
        self.data = []
        self._ident = 0

    def kernel(self, cg):
        self.body.append(cg.b.text().rstrip('\n'))
        return self

    def raw(self, line):
        self.body.append(line)
        return self

    def next_ident(self):
        self._ident += 1
        return self._ident

    def check_word(self, base, offset, expect):
        n = self.next_ident()
        self.checks.append(f"    s.ld.w    s21, [s20, #{offset}]")
        self.checks.append(f"    .assert_eq s21, #{expect & 0xffffffff:#010x},"
                           f" #{n}")
        return self

    def check_base(self, addr):
        self.checks.append(f"    .li       s20, #{addr}")
        return self

    def words(self, addr, values, label):
        self.data.append(f".org {addr:#x}")
        self.data.append(f"{label}:")
        for w in values:
            self.data.append(f"    .dword 0x{w & ((1 << 64) - 1):016x}")
        self.data.append("")
        return self

    def text(self):
        lines = [f"// {self.name} -- emitted by tnca_cc; data and assertions",
                 "// appended, the instructions untouched (sec 15 C1).",
                 ".org 0", "start:"]
        lines += self.body
        lines += self.checks
        lines.append("    exit")
        lines.append("")
        lines += self.data
        return '\n'.join(lines) + '\n'


def run(program, *, extra_args=(), max_cycles=200000):
    """Assemble and run. Returns (ok, stdout)."""
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, f"cc_{program.name}.s")
    hexf = os.path.join(BUILD, f"cc_{program.name}.hex")
    with open(src, 'w') as fh:
        fh.write(program.text())

    asm = subprocess.run(
        [sys.executable, ASM, src, '-o', hexf, '--size', '16384'],
        capture_output=True, text=True)
    if asm.returncode != 0:
        return False, f"assembling failed:\n{asm.stderr}{asm.stdout}"

    proc = subprocess.run(
        ['vvp', BENCH, f"+hex={hexf}", f"+maxcycles={max_cycles}",
         *extra_args],
        capture_output=True, text=True)
    ok = 'PASS' in proc.stdout and 'FAIL' not in proc.stdout
    return ok, proc.stdout
