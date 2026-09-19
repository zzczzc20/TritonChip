"""
Lowering a restricted elementwise kernel to TNCA assembly.

This is the narrow first slice of compiler_overview.md sec 15 C1: vector add
and masked elementwise, through the bring-up path of sec 11.1. It takes a
small description of a Triton elementwise kernel rather than TTIR, because the
frontend half and the lowering half fail in different ways and sec 11.1 exists
to keep them separable -- 编译器问题和机器码编码问题容易分离.

What it deliberately does not do, per sec 11.1: no general machine optimiser,
no clever scheduling. What it does do is refuse rather than approximate.
tb/core/progs/t16_vecadd.s is the hand-written reference this is checked
against, instruction for instruction.
"""
from dataclasses import dataclass, field

from .asm import AsmBuilder, RegisterAllocator
from .diagnostics import Diagnostic


@dataclass
class Tensor:
    """One block-shaped operand living in vector registers."""
    name: str
    dtype: str = 'f32'
    reg: int = None


@dataclass
class ElementwiseKernel:
    """A Triton elementwise kernel, as much of one as C1's first case needs.

    Deliberately not a general IR. sec 15 C1 lists vector add, masked
    elementwise, integer and FP32 reduction and simple control flow; this
    covers the first two, and the shape is meant to be replaced by TTIR
    rather than grown.
    """
    name: str
    inputs: list                     # pointer argument names
    output: str
    block: int
    op: str = 'add'                  # 'add' | 'mul' | 'sub'
    dtype: str = 'f32'
    masked: bool = True
    other: int = 0                   # the value masked-off lanes take
    grid_loop: bool = False          # walk every program id in one context
    num_programs: int = None
    arg_addresses: dict = field(default_factory=dict)
    n_elements: int = None


_FP_OPS = {'add': 'vf.add', 'sub': 'vf.sub', 'mul': 'vf.mul'}
_INT_OPS = {'add': 'v.add', 'sub': 'v.sub', 'mul': 'v.mul'}


def lower_elementwise(kernel, target, builder=None):
    """Emit TNCA assembly for one elementwise kernel."""
    if kernel.block % target.vlen:
        raise Diagnostic(
            f"BLOCK={kernel.block} is not a multiple of VLEN={target.vlen}",
            because='a block is laid out across whole vector registers; a '
                    'partial register would need a lane mask on every '
                    'operation, not just the boundary ones',
            remedy=f"use a multiple of {target.vlen}",
            source='compiler_overview.md sec 12')
    if kernel.block != target.vlen:
        # sec 12: a tensor wider than one register splits across several,
        # keeping the original index mapping. Worth doing; not done yet, and
        # saying so beats emitting something that addresses only the first.
        raise Diagnostic(
            f"BLOCK={kernel.block} spans "
            f"{kernel.block // target.vlen} vector registers",
            because='multi-register blocks are not lowered yet',
            remedy=f"use BLOCK={target.vlen} for now",
            source='compiler_overview.md sec 12')

    if kernel.dtype == 'f32':
        target.require('vector_fp32', f"emit a {kernel.dtype} {kernel.op}")
        opcode = _FP_OPS.get(kernel.op)
    else:
        opcode = _INT_OPS.get(kernel.op)
    if opcode is None:
        raise Diagnostic(f"no lowering for elementwise op {kernel.op!r}")

    b = builder or AsmBuilder(target, kernel.name)
    alloc = RegisterAllocator(target)

    # ---- kernel arguments -------------------------------------------------
    s_n = alloc.scalar()
    b.comment(f"{kernel.name}: {kernel.op} over BLOCK={kernel.block}, "
              f"{'masked' if kernel.masked else 'unmasked'}")
    b.directive(f".li       s{s_n}, #{kernel.n_elements}"
                f"              // n")

    ptr_regs = {}
    for arg in kernel.inputs + [kernel.output]:
        r = alloc.scalar()
        ptr_regs[arg] = r
        b.directive(f".li       s{r}, #{kernel.arg_addresses[arg]}"
                    f"              // {arg}")

    s_pid = alloc.scalar()
    b.directive(f".li       s{s_pid}, #0                     // pid")
    b.blank()

    if kernel.grid_loop:
        b.label('loop')

    # ---- offs = pid * BLOCK + arange(0, BLOCK) ---------------------------
    b.comment("offs = pid * BLOCK + arange(0, BLOCK)")
    s_block = alloc.scalar()
    b.directive(f".li       s{s_block}, #{kernel.block}")
    s_base = alloc.scalar()
    b.emit('s.mul', f"s{s_base}", f"s{s_pid}", f"s{s_block}",
           comment='pid * BLOCK')
    v_iota = alloc.vector()
    b.emit('v.iota', f"v{v_iota}", comment='arange(0, BLOCK)')
    v_pidb = alloc.vector()
    b.emit('v.splat', f"v{v_pidb}", f"s{s_base}")
    v_offs = alloc.vector()
    b.emit('v.add', f"v{v_offs}", f"v{v_iota}", f"v{v_pidb}", comment='offs')
    b.blank()

    # ---- mask = offs < n --------------------------------------------------
    p_mask = None
    if kernel.masked:
        b.comment("mask = offs < n")
        v_n = alloc.vector()
        b.emit('v.splat', f"v{v_n}", f"s{s_n}")
        p_mask = alloc.predicate()
        b.emit('v.cmp.lt', f"p{p_mask}", f"v{v_offs}", f"v{v_n}")
        b.blank()

    # ---- addresses --------------------------------------------------------
    # The offsets are contiguous, so a plain vector load reaches them: ISA
    # sec 9.1 gives lane i the address base + i*sz. Emitting a gather would
    # pay for per-lane addressing that buys nothing, and sec 12 wants a
    # uniform base with a bounded offset recognised rather than missed.
    b.comment("The offsets are contiguous, so a plain vector load reaches\n"
              "them: ISA sec 9.1 gives lane i the address base + i*sz. No\n"
              "gather is needed, and emitting one would cost per-lane\n"
              "addressing for nothing.")
    s_byte = alloc.scalar()
    b.emit('s.shli', f"s{s_byte}", f"s{s_base}", '#2',
           comment='byte offset = pid*BLOCK*4')
    addr_regs = {}
    for arg in kernel.inputs + [kernel.output]:
        r = alloc.scalar()
        addr_regs[arg] = r
        b.emit('s.add', f"s{r}", f"s{ptr_regs[arg]}", f"s{s_byte}",
               comment=f"&{arg}[pid*BLOCK]")
    b.blank()

    # ---- load -------------------------------------------------------------
    if kernel.masked:
        # sec 12: `other=` needs an explicit passthrough. A predicated load
        # leaves masked-off lanes holding whatever the register had, so the
        # value has to be put there first -- relying on a stale register is
        # how `other=0.0` silently becomes `other=garbage`.
        b.comment("other=0.0 is a value, not an absence (sec 12). The\n"
                  "masked-off lanes keep what the register already held, so\n"
                  "it has to be set first.")
    in_regs = []
    for arg in kernel.inputs:
        v = alloc.vector()
        in_regs.append(v)
        if kernel.masked:
            b.emit('v.movi', f"v{v}", f"#{kernel.other}")
    for arg, v in zip(kernel.inputs, in_regs):
        b.emit('v.ld.w', f"v{v}", f"[s{addr_regs[arg]}, #0]",
               pred=f"p{p_mask}" if kernel.masked else None)
    b.blank()

    # ---- compute and store ------------------------------------------------
    v_out = alloc.vector()
    acc = in_regs[0]
    for v in in_regs[1:]:
        b.emit(opcode, f"v{v_out}", f"v{acc}", f"v{v}")
        acc = v_out
    b.emit('v.st.w', f"[s{addr_regs[kernel.output]}, #0]", f"v{v_out}",
           pred=f"p{p_mask}" if kernel.masked else None)
    b.blank()

    # ---- grid loop --------------------------------------------------------
    if kernel.grid_loop:
        s_np = alloc.scalar()
        s_more = alloc.scalar()
        b.emit('s.addi', f"s{s_pid}", f"s{s_pid}", '#1')
        b.directive(f".li       s{s_np}, #{kernel.num_programs}")
        b.emit('s.cmp.lt', f"s{s_more}", f"s{s_pid}", f"s{s_np}")
        b.emit('br.nz', f"s{s_more}", 'loop')
        b.blank()

    return b, alloc
