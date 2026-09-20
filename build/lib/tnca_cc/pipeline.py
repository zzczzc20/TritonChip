"""
Explicit two-role pipeline: milestone C3.

compiler_overview.md sec 15 calls this 第一阶段最重要的里程碑, and sec 15 C3
says the role plan is given by the test or a compile option rather than
derived -- 不急于自动推导. Automatic partitioning is C4; this is the protocol
it will later generate.

The structure is the RF channel of sec 7.4, not a shared-memory Aref: TNCA has
no scratchpad for an Aref to live in, so a channel is a set of vector-register
windows plus the barriers and ownership discipline that move one from the
producer to the consumer.

Three facts shape the generated code:

  sec 9     W, V and A are static fields in the encoding, so `slot = i % 2`
            cannot become a computed register base. The loop is unrolled by
            stage, giving every stage static numbering.

  sec 7.3   the scalar file is private per RCTX, so the consumer cannot read
            the producer's barrier token out of a register. It derives the
            generation from the iteration number (ISA sec 9A.2 makes a token
            {epoch, barrier_id}, which is constructible).

  sec 9A.1  three completions must not be confused. A transfer completing is
            not the consumer having finished reading it, so ready and free are
            separate barriers -- the most common way to get a double-buffered
            pipeline wrong is to implement only load-ready.
"""
from dataclasses import dataclass, field

from .diagnostics import Diagnostic
from .ir import Builder, make_kernel


@dataclass
class PipelinePlan:
    """A two-role, N-stage pipeline over one channel."""
    stages: int
    iterations: int
    transfers_per_stage: int = 1
    window_registers: int = 8
    dtype: str = 'f16'
    consumers: int = 1

    # Filled in by plan_pipeline.
    barriers: int = 0
    windows: int = 0
    rejected: list = field(default_factory=list)

    def slot(self, i):
        return i % self.stages

    def generation(self, i):
        """Which use of this slot iteration `i` is.

        The g-th use of slot s happens at iteration i = g*stages + s, and the
        first mb.arm of a barrier makes epoch 1, so the generation is g+1.
        """
        return i // self.stages + 1


def plan_pipeline(target, *, stages=2, iterations, transfers_per_stage=1,
                  window_registers=8, dtype='f16', consumers=1):
    if stages < 1:
        raise Diagnostic('a pipeline needs at least one stage')
    if consumers < 1:
        raise Diagnostic('a channel needs at least one consumer')
    plan = PipelinePlan(stages=stages, iterations=iterations,
                        transfers_per_stage=transfers_per_stage,
                        window_registers=window_registers, dtype=dtype,
                        consumers=consumers)
    # One ready and one free barrier per slot. They cannot share: a transfer
    # completing and a consumer finishing are different events (ISA sec 9A.1).
    plan.barriers = 2 * stages
    plan.windows = stages
    target.check_budget(barriers=plan.barriers, windows=plan.windows,
                        rctx=1 + consumers)
    if plan.windows * window_registers > target.vregs_per_rctx:
        raise Diagnostic(
            f"{plan.windows} windows of {window_registers} registers exceed "
            f"{target.vregs_per_rctx}",
            because='a window is a contiguous register range with no storage '
                    'of its own',
            remedy='fewer stages',
            source='compiler_overview.md sec 10.1')
    return plan


def build_pipeline(target, plan, *, name='pipeline', desc_addr,
                   coords_for=None, consume=None, epilogue=None,
                   predicate_off=()):
    """Emit a producer role and a consumer role over one RF channel.

    `coords_for(i)` gives the (coord0, coord1) of iteration i.
    `consume(b, i, window)` emits the consumer's body for iteration i.
    `epilogue(b)` runs in the consumer after the loop.
    `predicate_off` names iterations whose transfer is predicated off. ISA
    sec 9A.3 has such a command still complete once, so the barrier's
    operation count does not change -- which is the property that lets a
    static count survive predication at all.
    """
    coords_for = coords_for or (lambda i: (0, 0))
    k = make_kernel(name, num_roles=1 + plan.consumers)

    # ---- whole-program resources, fixed first (sec 10.1) ------------------
    shared = Builder(k, k.role(0).region)
    windows = [shared.window(plan.window_registers, name=f"w{s}")
               for s in range(plan.stages)]
    ready = [shared.barrier(name=f"ready{s}") for s in range(plan.stages)]
    free = [shared.barrier(name=f"free{s}") for s in range(plan.stages)]

    # ---- producer ---------------------------------------------------------
    b = shared
    b.comment(f"producer: {plan.stages}-stage channel, "
              f"{plan.iterations} iteration(s), unrolled by stage because "
              f"W and V are static encoding fields (sec 9)")
    s_desc = b.const(desc_addr)
    desc = b.descriptor(s_desc, name='d0')

    # No startup handshake. Barriers are quiescent at kernel entry (ISA
    # sec 9A.7), so the consumer can sleep on `ready` generation 1 before the
    # producer has run a single instruction. The memory-flag handshake this
    # replaced was not merely redundant: the consumer's polling starved the
    # producer's transfer engine of the memory port, and under a slow memory
    # model the flag could never be written at all.

    for i in range(plan.iterations):
        s = plan.slot(i)
        g = plan.generation(i)
        b.comment(f"iteration {i}: slot {s}, generation {g}")
        if i >= plan.stages:
            # Every consumer has to have finished with this slot's previous
            # contents. sec 7.4: 只有两个 consumer 都释放，producer 才能覆盖
            # 该 slot. Waiting on `ready` instead would be the classic
            # double-buffering bug of ISA sec 9A.1.
            b.comment(f"wait until all {plan.consumers} consumer(s) released "
                      f"slot {s} (generation {g - 1})")
            b.barrier_wait(b.barrier_token(free[s], g - 1))
        # The producer arms `free` too, and arms it BEFORE `ready`. A consumer
        # cannot arrive on a phase that is not armed, and it only arrives
        # after waiting on `ready`, so this order is what makes the arrival
        # count safe with more than one consumer -- having each consumer arm
        # it would advance the epoch once per consumer.
        b.barrier_arm(free[s], arrivals=plan.consumers, ops=0)
        tok = b.barrier_arm(ready[s], arrivals=0,
                            ops=plan.transfers_per_stage)
        c0, c1 = coords_for(i)
        mask = None
        if i in predicate_off:
            # An all-false lane mask. The command is issued, executes nothing
            # and still signals one completion (ISA sec 9A.3), so the count
            # armed above is unchanged.
            b.comment(f"iteration {i} is predicated off; it still completes "
                      "once (ISA sec 9A.3)")
            zero = b.const(0)
            mask = b.vcmp('lt', b.arange(), b.splat(zero))
        for _t in range(plan.transfers_per_stage):
            b.dma_load(windows[s], desc, [b.const(c0), b.const(c1)], tok,
                       dtype=plan.dtype, mask=mask)
    b.exit()

    # ---- consumer(s) ------------------------------------------------------
    for c in range(plan.consumers):
        cb = Builder(k, k.role(1 + c).region)
        cb.comment(f"consumer {c}: derives each generation from the iteration "
                   "number, because the scalar file is private per RCTX "
                   "(sec 7.3)")
        for i in range(plan.iterations):
            s = plan.slot(i)
            g = plan.generation(i)
            cb.comment(f"iteration {i}: slot {s}, generation {g}")
            cb.barrier_wait(cb.barrier_token(ready[s], g))
            if consume:
                consume(cb, i, windows[s])
            # Release: one arrival out of the `consumers` the producer armed
            # for. The slot returns to the producer only when the last of them
            # arrives (sec 7.4).
            cb.barrier_arrive(free[s])
        if epilogue:
            epilogue(cb)
        cb.exit()

    return k, shared
