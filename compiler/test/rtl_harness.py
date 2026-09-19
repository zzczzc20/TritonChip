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
# The memory model is an elaboration parameter, so each behaviour is its own
# bench. sec 16.2 asks for 随机 memory latency/back-pressure over the
# compiler's own output, which means running it against these too.
BENCHES = {
    'fast': 'tb_core.vvp',
    'slow': 'tb_core_slow.vvp',     # MEM_LATENCY=7, RANDOM_STALL=1
    'ooo': 'tb_core_ooo.vvp',       # responses return out of order
    # Four Role Contexts, half the logical vector registers each. sec 15 C4
    # defers the P + M0 + M1 + E check until such a profile exists.
    'rctx4': 'tb_core4.vvp',
}
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


def bench_path(which='fast'):
    return os.path.join(BUILD, BENCHES[which])


def bench_available(which):
    return os.path.exists(bench_path(which)) and shutil.which('vvp')


def run(program, *, extra_args=(), max_cycles=200000, bench='fast'):
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
        ['vvp', bench_path(bench), f"+hex={hexf}",
         f"+maxcycles={max_cycles}", *extra_args],
        capture_output=True, text=True)
    ok = 'PASS' in proc.stdout and 'FAIL' not in proc.stdout
    return ok, proc.stdout


def multirole_program(name, kernel, target, *, check_protocol=False,
                      check_role=None):
    """One image holding every role, dispatched on role_id.

    Every Role Context starts at PC 0 and branches on its own index -- the
    layout tb/core/progs/t13_share.s uses. Whole-program resources are set up
    by role 0 only; the other roles wait for the flag it writes, because
    ISA sec 9A.6 makes mb.init under a sleeping waiter a BARRIER_FAULT, so the
    startup wait cannot itself be on a barrier (sec 13.3).

    `check_role` (default: the last role) has its trailing `exit` removed and
    is emitted last, so the assertions appended by Program.text() run as the
    continuation of that role. Leaving every role to exit put the assertions
    after the final exit, where nothing reached them -- and a test whose
    checks never execute passes for the wrong reason.
    """
    from tnca_cc.codegen import CodeGen

    if check_role is None:
        check_role = len(kernel.roles) - 1
    order = [r for r in range(len(kernel.roles)) if r != check_role]
    order.append(check_role)

    n_roles = len(kernel.roles)
    p = Program(name)
    p.raw("    s.rdspecial s60, role_id")
    # A machine may have more Role Contexts than the kernel uses, and the
    # bench starts every one of them. Park the extras: without this they fall
    # through into role 0 and the program runs several producers against the
    # same barriers, which advances each epoch once per stray context and
    # desynchronises the whole channel. ISA sec 5.1 reports NUM_ROLES as the
    # active count for the launch, so a kernel can always tell.
    p.raw(f"    .li       s59, #{n_roles}")
    p.raw("    s.cmp.geu s58, s60, s59")
    p.raw("    br.nz     s58, park")
    for r in range(1, n_roles):
        p.raw(f"    .li       s59, #{r}")
        p.raw(f"    s.cmp.eq  s58, s60, s59")
        p.raw(f"    br.nz     s58, role{r}")
    p.raw("    br        role0")
    p.raw("")
    p.raw("park:")
    p.raw("    exit")
    p.raw("")

    peaks = []
    for r in order:
        cg = CodeGen(kernel, target, check_protocol=check_protocol)
        cg.assign_shared()
        cg.run(r)
        text = cg.b.text().rstrip('\n')
        if r == check_role:
            # Drop the final exit so control reaches the assertions.
            lines = text.splitlines()
            while lines and lines[-1].strip() in ('exit', ''):
                lines.pop()
            text = '\n'.join(lines)
        p.raw(f"role{r}:")
        p.raw(text)
        p.raw("")
        peaks.append(cg.alloc.peak)
    return p, peaks
