"""
GEMM planning and lowering: milestone C2.

compiler_overview.md sec 9 is the reason this is a module of its own rather
than a few more cases in kernels.py. Its point:

    Transfer 写 flat row-major，MMA 读 flat row-major，因此不需要 repack --
    对完全匹配的一块原生 fragment 成立，但不能自动推广到任意 GEMM tile.

A 32x64 B tile loaded row-major puts 64*sizeof(dtype) between the rows of any
32x16 column sub-block, where the MMA fragment interpretation wants
16*sizeof(dtype). Moving the fragment base cannot close that gap. sec 9 lists
four ways out and recommends the first for now: split the tile into separate
transfers, each forming a compact fragment, rather than changing the MMA ISA.

That choice is not local. It changes the DMA command count, the barrier
operation count, the number of windows, the descriptor block shapes, the RF
live ranges and the issue overhead -- which is sec 9's conclusion:

    布局选择、通信协议和流水资源预算必须联合规划，不能作为三个互不相关的
    pass.

So one planner decides all of them together, and the numbers it produces are
what the barrier is armed with.

sec 9 also rules out a dynamic register base: W, V and A are static fields in
the encoding, so `slot = i % 2` cannot become a computed vector-register base.
The plan therefore unrolls by stage, giving every stage static numbering.
"""
from dataclasses import dataclass, field

from .diagnostics import Diagnostic, ResourceError
from .footprints import fragment_registers
from .ir import Builder, make_kernel

_DTYPE_BITS = {'f16': 16, 'bf16': 16}


@dataclass
class Transfer:
    """One asynchronous transfer producing one compact fragment."""
    operand: str            # 'A' or 'B'
    window: int             # index into the plan's windows
    descriptor: int
    coord0: int
    coord1: int
    k_step: int


@dataclass
class GemmPlan:
    """Layout, protocol and resources, decided together (sec 9)."""
    m: int
    n: int
    k: int
    dtype: str
    frag_m: int
    frag_n: int
    frag_k: int

    k_steps: int = 0
    k_padded: int = 0
    m_tail: bool = False
    n_tail: bool = False
    k_tail: bool = False
    transfers: list = field(default_factory=list)
    mma_count: int = 0
    window_registers: int = 0
    num_windows: int = 0
    num_descriptors: int = 2
    num_barriers: int = 2
    split_reason: str = ''

    @property
    def dma_operations(self):
        """What the barrier must be armed for.

        sec 8.2: the operation count is taken after the commands expand. This
        is that expanded number, not the count of source-level transfers.
        """
        return len(self.transfers)

    def describe(self):
        lines = [
            f"GEMM {self.m}x{self.n}x{self.k} {self.dtype}",
            f"  MMA fragment:        {self.frag_m}x{self.frag_n}x{self.frag_k}",
            f"  k steps:             {self.k_steps}",
            f"  DMA commands:        {self.dma_operations}",
            f"  MMA commands:        {self.mma_count}",
            f"  windows:             {self.num_windows} "
            f"({self.window_registers} vector registers each)",
            f"  descriptors:         {self.num_descriptors}",
            f"  barriers:            {self.num_barriers}",
        ]
        if self.split_reason:
            lines.append(f"  split:               {self.split_reason}")
        tails = [n for n, t in (('M', self.m_tail), ('N', self.n_tail),
                                ('K', self.k_tail)) if t]
        if tails:
            lines.append(f"  tail:                {', '.join(tails)} "
                         f"(padded by the descriptor, ISA sec 9E.1)")
        return '\n'.join(lines)


def check_descriptor_geometry(target, *, base, row_stride_bytes,
                              elem_bytes, block_cols, what):
    """Refuse a descriptor the transfer engine cannot execute (ISA sec 9E.2).

    The engine places whole 8-byte units into window registers through the
    existing lane write enables, so every row must both be a whole number of
    units and start on one. A 13-element FP16 row is 26 bytes and satisfies
    neither; before this check the engine returned data shifted by two bytes
    per row and raised nothing.
    """
    if base % 8:
        raise Diagnostic(
            f"the {what} base {base:#x} is not 8-byte aligned",
            because='ISA sec 9E.2: the first unit of every row would land '
                    'shifted in the window',
            remedy='align the allocation to 8 bytes',
            source='ISA sec 9E.2')
    if row_stride_bytes % 8:
        raise Diagnostic(
            f"the {what} row stride is {row_stride_bytes} bytes, not a "
            "multiple of 8",
            because='ISA sec 9E.2: rows after the first would land shifted, '
                    'even from an aligned base',
            remedy=(f"pad the leading dimension to a multiple of "
                    f"{8 // elem_bytes} elements, or fall back to an LSU "
                    "loop"),
            source='ISA sec 9E.2; compiler_overview.md sec 12')
    if (block_cols * elem_bytes) % 8:
        raise Diagnostic(
            f"the {what} block row is {block_cols * elem_bytes} bytes, not a "
            "multiple of 8",
            because='ISA sec 9E.2: a returned unit would straddle two window '
                    'registers',
            source='ISA sec 9E.2')
    return True


def plan_gemm(target, *, m, n, k, dtype='f16'):
    """Decide the transfer, fragment and resource plan for one output tile."""
    if dtype not in _DTYPE_BITS:
        raise Diagnostic(
            f"no GEMM lowering for {dtype!r}",
            because=f"the matrix unit takes {sorted(_DTYPE_BITS)}",
            source='ISA sec 9C')
    cap = 'matrix_bf16' if dtype == 'bf16' else 'matrix_f16'
    target.require(cap, f"emit m.mma.async with {dtype} inputs")

    fm, fn, fk = target.mma_shape
    if m > fm or n > fn:
        raise Diagnostic(
            f"an output tile of {m}x{n} exceeds the {fm}x{fn} the matrix unit "
            "produces",
            because='an accumulator holds one fragment-shaped tile; a larger '
                    'output needs several accumulators and an outer loop, '
                    'which is not planned yet',
            remedy=f"use at most {fm}x{fn} per accumulator",
            source='compiler_overview.md sec 9')

    bits = _DTYPE_BITS[dtype]
    plan = GemmPlan(m=m, n=n, k=k, dtype=dtype, frag_m=fm, frag_n=fn,
                    frag_k=fk)
    # sec 15 C2's N/M/K tail. The matrix unit takes whole fragments, so a
    # short K is not truncated -- it is padded. The descriptor's extents stop
    # at the real K, and padding mode 0 fills what is past them with zero
    # (ISA sec 9E.1 step 7, which fills rather than masking the access). The
    # arithmetic survives exactly: RNE_f32(exact(0)*exact(b) + C) is C, so
    # the padded k steps contribute nothing at all rather than something
    # small. A masked fragment would be the alternative, and the matrix unit
    # does not take one.
    plan.k_steps = (k + fk - 1) // fk
    plan.k_padded = plan.k_steps * fk
    plan.m_tail = m < fm
    plan.n_tail = n < fn
    plan.k_tail = k % fk != 0
    plan.window_registers = fragment_registers(fm, fk, bits)

    # sec 9, option 1: one transfer per fragment. A single transfer covering
    # the whole K extent would put k*sizeof(dtype) between fragment rows where
    # the MMA interpretation wants frag_k*sizeof(dtype), and no change of base
    # closes that.
    if plan.k_steps > 1:
        plan.split_reason = (
            f"K={k} spans {plan.k_steps} fragments; one transfer per fragment "
            f"so each lands compact (sec 9 option 1)")

    # A and B each need a window. They are separate because the two fragments
    # are read simultaneously by one MMA command.
    plan.num_windows = 2
    for step in range(plan.k_steps):
        plan.transfers.append(Transfer('A', window=0, descriptor=0,
                                       coord0=0, coord1=step * fk,
                                       k_step=step))
        plan.transfers.append(Transfer('B', window=1, descriptor=1,
                                       coord0=step * fk, coord1=0,
                                       k_step=step))
    plan.mma_count = plan.k_steps

    target.check_budget(barriers=plan.num_barriers,
                        windows=plan.num_windows,
                        descriptors=plan.num_descriptors,
                        accumulators=1)
    need = plan.num_windows * plan.window_registers
    if need > target.vregs_per_rctx:
        raise ResourceError(
            f"the fragments need {need} vector registers, target has "
            f"{target.vregs_per_rctx}",
            because='a window is a contiguous register range with no storage '
                    'of its own (ISA sec 9D.2)',
            remedy='a smaller K per step',
            source='compiler_overview.md sec 10.1')
    return plan


def build_gemm(target, plan, *, name='gemm', a_desc_addr, b_desc_addr,
               out_addr, epilogue=True):
    """Emit the IR for one output tile.

    The k loop is unrolled. sec 9: W, V and A are static fields in the
    encoding, so a stage index cannot become a computed register base; the
    first version unrolls by stage so every number is static.
    """
    k = make_kernel(name)
    b = Builder(k, k.role(0).region)

    b.comment(plan.describe())

    wa = b.window(plan.window_registers, name='wa')
    wb = b.window(plan.window_registers, name='wb')
    s_a = b.const(a_desc_addr)
    s_b = b.const(b_desc_addr)
    da = b.descriptor(s_a, name='da')
    db = b.descriptor(s_b, name='db')

    bar_fill = b.barrier(name='fill')
    bar_mma = b.barrier(name='mma')
    acc = b.accumulator(name='acc')
    b.acc_zero(acc)

    windows = {0: wa, 1: wb}
    descs = {0: da, 1: db}

    for step in range(plan.k_steps):
        step_transfers = [t for t in plan.transfers if t.k_step == step]
        b.comment(f"k step {step}: {len(step_transfers)} transfer(s), "
                  f"armed for exactly that many operations (sec 8.2)")
        tok = b.barrier_arm(bar_fill, arrivals=0, ops=len(step_transfers))
        for t in step_transfers:
            c0 = b.const(t.coord0)
            c1 = b.const(t.coord1)
            b.dma_load(windows[t.window], descs[t.descriptor], [c0, c1], tok,
                       dtype=plan.dtype)
        b.barrier_wait(tok)

        mtok = b.barrier_arm(bar_mma, arrivals=0, ops=1)
        b.mma_async(acc, wa, wb, mtok, dtype=plan.dtype)
        b.barrier_wait(mtok)

    if epilogue:
        # sec 15 C2's accumulator epilogue: the result leaves the accumulator
        # a row at a time, through the vector unit, and is stored. It cannot
        # be read before the MMA's completion (ISA sec 9C.3), which the wait
        # above provides.
        b.comment("accumulator epilogue: one row at a time through the "
                  "vector unit")
        s_out = b.const(out_addr)
        row_bytes = plan.n * 4

        # The accumulator is always fragment-shaped, so an N tail means the
        # columns past n hold products of padding. They are exactly zero
        # rather than garbage, but they are still not part of the result, so
        # the store is masked rather than trusted.
        mask = None
        if plan.n_tail:
            s_n = b.const(plan.n)
            mask = b.vcmp('lt', b.arange(), b.splat(s_n))
            b.comment(f"N tail: store only the first {plan.n} columns")

        for row in range(plan.m):
            s_row = b.const(row)
            v = b.acc_read(acc, s_row)
            addr = (s_out if row == 0
                    else b.sbini('add', s_out, row * row_bytes,
                                 comment=f"&out[{row}]"))
            b.store(addr, v, mask=mask)

    return k, b
