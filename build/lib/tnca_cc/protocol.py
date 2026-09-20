"""
A small ownership/event model of the TNCA pipeline protocol.

compiler_overview.md sec 16.2 asks for exactly this and warns against the
alternative: "只建立小型 ownership/event 模型，不另写一整套 Triton 解释器."
So nothing here executes arithmetic. It tracks who owns which registers, which
asynchronous commands are still reading them, and when a barrier releases a
waiter -- because that is where the protocol can be wrong.

The hard question sec 17 poses is the one being modelled:

    who owns which stretch of RF after partitioning; when an asynchronous
    command stops reading it; how two consumers jointly release it; when a
    physical register may be reused.

Model, deliberately:

  * Registers are owned ranges, not values. A read of an unpublished range is
    a protocol error whatever bits it would have returned.
  * An asynchronous command holds a *lease* on its source range until its
    source-release event, and on its destination until its completion. Those
    are separate events even where v0.1 hardware fuses them, so the model can
    say what a future separate source completion would buy.
  * Barriers count arrivals and operations separately (sec 13.1), and an
    operation is one complete asynchronous command (ISA sec 9B).
  * Predication off does not skip a completion (sec 8.3); an all-false stage
    must still count.
"""
import itertools
from dataclasses import dataclass, field
from enum import Enum


class ProtocolError(Exception):
    """An invariant the generated code must never be able to violate."""

    def __init__(self, message, *, trace=None):
        self.message = message
        self.trace = list(trace or [])
        super().__init__(message)

    def __str__(self):
        if not self.trace:
            return self.message
        steps = '\n'.join(f"    {i:3d}. {s}" for i, s in enumerate(self.trace))
        return f"{self.message}\n  trace:\n{steps}"


class BufferState(Enum):
    FREE = 'free'              # no owner; may be acquired by a producer
    PRODUCING = 'producing'    # a producer owns it; fill in flight or pending
    FULL = 'full'              # published; consumers may acquire
    CONSUMING = 'consuming'    # one or more consumers are reading


@dataclass
class Lease:
    """An asynchronous command's hold on a register range.

    Two events, not one. `dest_done` is when the destination registers carry
    the result; `src_done` is when the engine has stopped reading the sources.
    ISA v0.1 raises a single completion covering both, which is the
    conservative case -- so the model defaults to it and can be asked what an
    earlier source release would allow.
    """
    cmd_id: int
    role: str
    buffer: int
    kind: str                  # 'dma' | 'mma'
    barrier: int
    src_done: bool = False
    dest_done: bool = False

    @property
    def retired(self):
        return self.src_done and self.dest_done


@dataclass
class Barrier:
    """An mbarrier.

    sec 13.1: arrivals and operations are counted separately, because they are
    different things -- an arrival is a Role Context saying "I am here", an
    operation is an asynchronous command saying "I have finished". A single
    counter cannot distinguish a producer that arrived early from a transfer
    that completed.
    """
    index: int
    armed: bool = False
    expect_arrivals: int = 0
    expect_ops: int = 0
    arrivals: int = 0
    ops: int = 0
    generation: int = 0
    waiters: list = field(default_factory=list)

    @property
    def satisfied(self):
        return (self.armed
                and self.arrivals >= self.expect_arrivals
                and self.ops >= self.expect_ops)


@dataclass
class Buffer:
    """One pipeline stage's RF window."""
    index: int
    state: BufferState = BufferState.FREE
    owner: str = None
    readers: set = field(default_factory=set)
    generation: int = 0        # bumped on each refill, to catch stale reads
    published_generation: int = -1


class ProtocolModel:
    """The state a generated pipeline must keep consistent."""

    def __init__(self, *, depth, consumers, num_barriers=32,
                 separate_source_completion=False):
        if depth < 1:
            raise ValueError("a pipeline needs at least one stage")
        self.buffers = {i: Buffer(i) for i in range(depth)}
        self.barriers = {i: Barrier(i) for i in range(num_barriers)}
        self.consumers = list(consumers)
        self.leases = {}
        self.separate_source_completion = separate_source_completion
        self.trace = []
        self._next_cmd = 0
        self.completed_reads = 0

    # ---- helpers ----------------------------------------------------------
    def _log(self, what):
        self.trace.append(what)

    def _fail(self, message):
        raise ProtocolError(message, trace=self.trace)

    # ---- barrier operations ----------------------------------------------
    def barrier_init(self, b):
        bar = self.barriers[b]
        if bar.waiters:
            # sec 13.3 / ISA sec 9A.6: re-initialising under a sleeping waiter
            # would strand it forever, so the hardware raises BARRIER_FAULT.
            self._fail(f"mb.init on barrier {b} with {len(bar.waiters)} waiter(s) "
                       "asleep on it -- BARRIER_FAULT (ISA sec 9A.6)")
        self._log(f"mb.init b{b}")
        self.barriers[b] = Barrier(index=b, generation=bar.generation)

    def barrier_arm(self, b, *, arrivals=0, ops=0):
        bar = self.barriers[b]
        if bar.armed and not bar.satisfied:
            self._fail(f"mb.arm on barrier {b} while the previous phase is "
                       "still outstanding")
        bar.armed = True
        bar.expect_arrivals = arrivals
        bar.expect_ops = ops
        bar.arrivals = 0
        bar.ops = 0
        self._log(f"mb.arm b{b} arrivals={arrivals} ops={ops}")

    def barrier_arrive(self, b, role):
        bar = self.barriers[b]
        if not bar.armed:
            self._fail(f"{role} arrived at barrier {b}, which is not armed")
        bar.arrivals += 1
        self._log(f"{role}: mb.arrive b{b} ({bar.arrivals}/{bar.expect_arrivals})")
        self._maybe_release(bar)

    def barrier_wait(self, b, role):
        """Sleep until the phase completes. Returns True if it already had."""
        bar = self.barriers[b]
        if bar.satisfied:
            self._log(f"{role}: mb.wait b{b} -- already satisfied")
            self._advance(bar)
            return True
        if not bar.armed:
            self._fail(f"{role} waits on barrier {b}, which is not armed -- "
                       "it would sleep forever")
        bar.waiters.append(role)
        self._log(f"{role}: mb.wait b{b} -- sleeping")
        return False

    def _maybe_release(self, bar):
        if bar.satisfied and bar.waiters:
            woken = list(bar.waiters)
            bar.waiters.clear()
            self._log(f"  barrier {bar.index} releases {woken}")
            self._advance(bar)

    def _advance(self, bar):
        bar.generation += 1
        bar.armed = False

    # ---- producer side ----------------------------------------------------
    def producer_acquire(self, buf, role):
        b = self.buffers[buf]
        if b.state is not BufferState.FREE:
            self._fail(f"{role} acquires buffer {buf} for writing while it is "
                       f"{b.state.value}"
                       + (f", read by {sorted(b.readers)}" if b.readers else ""))
        if any(l.buffer == buf and not l.retired for l in self.leases.values()):
            self._fail(f"{role} acquires buffer {buf} while an asynchronous "
                       "command still holds a lease on it")
        b.state = BufferState.PRODUCING
        b.owner = role
        b.generation += 1
        self._log(f"{role}: acquire buffer {buf} (gen {b.generation})")

    def issue_async(self, buf, role, barrier, kind='dma'):
        """Start an asynchronous command writing `buf`, counted on `barrier`.

        sec 8.2: the operation count is what the command expands to, not the
        number of source-level operations. One call here is one counted
        command, so a caller that expands a transfer into several must issue
        several.
        """
        b = self.buffers[buf]
        if b.state is not BufferState.PRODUCING or b.owner != role:
            self._fail(f"{role} issues a {kind} into buffer {buf}, which it "
                       f"does not own (state {b.state.value}, owner {b.owner})")
        bar = self.barriers[barrier]
        if not bar.armed:
            self._fail(f"{role} issues a {kind} counted on barrier {barrier}, "
                       "which is not armed -- the completion would be lost")
        cmd = self._next_cmd
        self._next_cmd += 1
        self.leases[cmd] = Lease(cmd_id=cmd, role=role, buffer=buf,
                                 kind=kind, barrier=barrier)
        self._log(f"{role}: {kind} #{cmd} -> buffer {buf}, counts on b{barrier}")
        return cmd

    def complete(self, cmd):
        """The engine finishes command `cmd`."""
        lease = self.leases[cmd]
        if lease.retired:
            self._fail(f"command #{cmd} completed twice")
        lease.src_done = True
        lease.dest_done = True
        bar = self.barriers[lease.barrier]
        bar.ops += 1
        self._log(f"  {lease.kind} #{cmd} complete "
                  f"({bar.ops}/{bar.expect_ops} ops on b{lease.barrier})")
        if bar.ops > bar.expect_ops:
            self._fail(f"barrier {lease.barrier} counted {bar.ops} operations "
                       f"but was armed for {bar.expect_ops} -- sec 8.2 requires "
                       "the count to be taken after command expansion")
        self._maybe_release(bar)

    def source_release(self, cmd):
        """The engine stops reading `cmd`'s sources.

        On v0.1 hardware this cannot happen before completion, so the model
        refuses it unless the target advertises a separate source completion.
        """
        if not self.separate_source_completion:
            self._fail(f"command #{cmd} has no separate source-release event; "
                       "ISA v0.1 raises one completion covering source and "
                       "destination, so the scheduler must take that bound")
        self.leases[cmd].src_done = True

    def publish(self, buf, role, barrier):
        """Hand the buffer to the consumers."""
        b = self.buffers[buf]
        if b.state is not BufferState.PRODUCING or b.owner != role:
            self._fail(f"{role} publishes buffer {buf} it does not own")
        live = [c for c, l in self.leases.items()
                if l.buffer == buf and not l.dest_done]
        if live:
            self._fail(f"{role} publishes buffer {buf} with command(s) "
                       f"{live} still writing it -- sec 8.1 says publish is "
                       "not a mechanical mb.arm; the fill must have completed")
        b.state = BufferState.FULL
        b.owner = None
        b.published_generation = b.generation
        b.readers = set()
        self._log(f"{role}: publish buffer {buf} (gen {b.generation})")
        self.barrier_arrive(barrier, role)

    # ---- consumer side ----------------------------------------------------
    def consumer_acquire(self, buf, role):
        b = self.buffers[buf]
        if b.state not in (BufferState.FULL, BufferState.CONSUMING):
            self._fail(f"{role} reads buffer {buf} in state {b.state.value} -- "
                       "the data it would see was never published")
        if role in b.readers:
            self._fail(f"{role} acquires buffer {buf} twice without releasing")
        if b.published_generation != b.generation:
            self._fail(f"{role} reads buffer {buf} at generation "
                       f"{b.generation} but only {b.published_generation} was "
                       "published -- a stale read")
        b.state = BufferState.CONSUMING
        b.readers.add(role)
        self.completed_reads += 1
        self._log(f"{role}: consume buffer {buf} (readers {sorted(b.readers)})")

    def consumer_release(self, buf, role, barrier):
        """Release a reader's claim.

        The buffer returns to FREE only when the last consumer releases: with
        two consumers, whichever finishes first must not free it. sec 17 names
        this as one of the genuinely hard parts.
        """
        b = self.buffers[buf]
        if role not in b.readers:
            self._fail(f"{role} releases buffer {buf} it is not reading")
        b.readers.discard(role)
        if not b.readers:
            b.state = BufferState.FREE
            self._log(f"{role}: release buffer {buf} -- last reader, now free")
        else:
            self._log(f"{role}: release buffer {buf} -- "
                      f"{sorted(b.readers)} still reading")
        self.barrier_arrive(barrier, role)

    # ---- final check ------------------------------------------------------
    def check_quiescent(self):
        # Outstanding commands first: a buffer stuck in PRODUCING is usually
        # just the symptom of a command nobody waited for, and naming the
        # cause is more useful than naming the symptom.
        live = [c for c, l in self.leases.items() if not l.retired]
        if live:
            self._fail(f"commands {live} never completed; their register "
                       "ranges can never be safely reused")
        for b in self.buffers.values():
            if b.state is not BufferState.FREE:
                self._fail(f"buffer {b.index} left {b.state.value} at exit "
                           f"(readers {sorted(b.readers)})")
        for bar in self.barriers.values():
            if bar.waiters:
                self._fail(f"barrier {bar.index} still has {bar.waiters} "
                           "asleep -- the kernel cannot terminate")
        return True


def completion_orders(n, limit=None):
    """Every order the engines could retire n outstanding commands in.

    sec 16.2 asks for "不同完成顺序" explicitly: asynchronous commands do not
    complete in issue order, and a protocol that only works when they do is
    broken in a way no single-trace test finds.
    """
    perms = itertools.permutations(range(n))
    if limit is None:
        return list(perms)
    return list(itertools.islice(perms, limit))
