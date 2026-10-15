"""
The low-level TNCA IR of the bring-up path.

compiler_overview.md sec 11.1:

    TNCA low-level IR -> conservative scheduling and restricted static
    allocation -> TNCA assembly -> tools/asm/tnca_asm.py -> RTL

This is that IR. It is deliberately below TTIR and above assembly: registers
are still virtual, but everything the machine distinguishes is already
explicit -- which register file a value lives in, which engine executes an
operation, and where each asynchronous command's completion is counted.

The design follows sec 5: value semantics stay generic (a vector add is a
vector add), and what this IR adds is resources and asynchronous execution.
Completion events are values rather than annotations, because sec 5.4 is
explicit that ownership written only as an annotation is invisible to
everything that reorders code.

The MLIR dialect in compiler/include mirrors this. They are two encodings of
the same model, and the Python one exists because sec 11.1 wants a path to RTL
that does not wait for the C++ half.
"""
from dataclasses import dataclass, field
from typing import Optional

from .diagnostics import Diagnostic

# Which register file a value lives in.
SCALAR = 'scalar'
VECTOR = 'vector'
PRED = 'pred'
ACC = 'acc'
WINDOW = 'window'
DESC = 'desc'
BARRIER = 'barrier'
EVENT = 'event'
TOKEN = 'token'


@dataclass(eq=False)
class Value:
    """An SSA value. `reg` is filled in by allocation, not by construction."""
    name: str
    kind: str
    dtype: str = 'i32'
    reg: Optional[int] = None
    length: int = 1          # windows: how many vector registers

    def __repr__(self):
        at = f"@{self.kind[0]}{self.reg}" if self.reg is not None else ''
        return f"%{self.name}:{self.kind}{at}"


@dataclass
class Op:
    """One operation. `kind` names it; operands and results are Values."""
    kind: str
    results: list = field(default_factory=list)
    operands: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def __repr__(self):
        res = ', '.join(repr(r) for r in self.results)
        ops = ', '.join(repr(o) for o in self.operands)
        at = ' '.join(f"{k}={v}" for k, v in self.attrs.items())
        head = f"{res} = " if res else ''
        return f"{head}{self.kind}({ops}) {at}".rstrip()


@dataclass
class Region:
    """A straight-line block, or a loop body."""
    ops: list = field(default_factory=list)
    label: Optional[str] = None


@dataclass
class Role:
    """One Role Context's program.

    sec 4.2 and frozen decision 2: the role count is an execution-resource
    budget, not a layout dimension. A role is a separate instruction stream
    sharing the SIMD, matrix, transfer and NUCA resources -- not a warp.
    """
    index: int
    name: str
    region: Region = field(default_factory=Region)


@dataclass
class Kernel:
    """A whole program: shared resources, then one region per role."""
    name: str
    roles: list = field(default_factory=list)
    # Whole-program resources, fixed before any role-local allocation
    # (sec 10.1). Populated by the planner.
    barriers: list = field(default_factory=list)
    accumulators: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    descriptors: list = field(default_factory=list)
    args: dict = field(default_factory=dict)

    def role(self, index):
        for r in self.roles:
            if r.index == index:
                return r
        raise Diagnostic(f"kernel {self.name!r} has no role {index}")

    def __str__(self):
        return format_kernel(self)


class Builder:
    """Construct IR. One builder per role region."""

    def __init__(self, kernel, region):
        self.kernel = kernel
        self.region = region
        self._n = 0

    # ---- value creation ---------------------------------------------------
    def _value(self, kind, dtype='i32', length=1, hint=None):
        self._n += 1
        name = f"{hint or kind[:3]}{self._n}"
        return Value(name=name, kind=kind, dtype=dtype, length=length)

    def _emit(self, kind, results=(), operands=(), **attrs):
        op = Op(kind=kind, results=list(results), operands=list(operands),
                attrs=attrs)
        self.region.ops.append(op)
        return op

    # ---- scalars ----------------------------------------------------------
    def const(self, value, dtype='i64'):
        """A scalar constant. Takes a NUMBER, not a Value.

        Handing it a Value produced `.li sX, #%c2:scalar@s2` -- the repr of
        an SSA value where an immediate belongs -- and the error surfaced in
        the assembler, three layers from the mistake. A builder that is given
        the wrong kind of thing should say so.
        """
        if isinstance(value, Value):
            raise Diagnostic(
                f"const() takes a number and was given {value!r}",
                because='a Value is already a register; there is nothing to '
                        'materialise',
                remedy='pass the Value where a Value is wanted, or a Python '
                       'int here',
                source='compiler_overview.md sec 12')
        v = self._value(SCALAR, dtype, hint='c')
        self._emit('const', [v], [], value=value)
        return v

    def program_id(self, dim=0):
        v = self._value(SCALAR, 'i32', hint='pid')
        self._emit('program_id', [v], [], dim=dim)
        return v

    def num_programs(self, dim=0):
        """The grid extent on one axis.

        The twin of program_id: ISA sec 5.1 gives both, pid.{x,y,z} and
        npid.{x,y,z}, and a kernel that can ask which program it is can
        ask how many there are.
        """
        v = self._value(SCALAR, 'i32', hint='npid')
        self._emit('num_programs', [v], [], dim=dim)
        return v

    def special(self, name):
        v = self._value(SCALAR, 'i64', hint='sp')
        self._emit('special', [v], [], name=name)
        return v

    def sbin(self, op, a, b, comment=None):
        v = self._value(SCALAR, a.dtype, hint='s')
        self._emit('sbin', [v], [a, b], op=op, comment=comment)
        return v

    def sbini(self, op, a, imm, comment=None):
        if not -32768 <= imm <= 32767:
            return self.sbin(op, a, self.const(imm), comment=comment)
        v = self._value(SCALAR, a.dtype, hint='s')
        self._emit('sbini', [v], [a], op=op, imm=imm, comment=comment)
        return v

    def scmp(self, cc, a, b):
        v = self._value(SCALAR, 'i32', hint='c')
        self._emit('scmp', [v], [a, b], cc=cc)
        return v

    # ---- vectors ----------------------------------------------------------
    def arange(self):
        v = self._value(VECTOR, 'i32', hint='ar')
        self._emit('arange', [v], [])
        return v

    def splat(self, s, dtype='i32'):
        # `v.splat` reads a SCALAR register. Handing it a vector used to
        # emit `v.splat vd, v7` -- a scalar register number that happened
        # to be a vector's -- and the kernel quietly read the wrong
        # register rather than failing.
        if s.kind != SCALAR:
            raise Diagnostic(
                f"v.splat broadcasts a scalar register; {s.name} is a "
                f"{s.kind}")
        v = self._value(VECTOR, dtype, hint='sp')
        self._emit('splat', [v], [s])
        return v

    def vmovi(self, imm, dtype='i32'):
        v = self._value(VECTOR, dtype, hint='k')
        self._emit('vmovi', [v], [], imm=imm)
        return v

    def view(self, v, bases, out_dims, out_sizes, *, what=''):
        """Read `v` under a different layout.

        doc/phase3_5_tasks_and_acceptance.md section 3. The layout is
        given the way upstream LinearLayout takes it -- a list of
        (input dimension, bases) and the output dimensions it maps to --
        because the compiler hands it straight to upstream rather than
        interpreting it here.

        What this op MEANS is a reindex: lane i of the result holds the
        element that the layout maps i to. What it COSTS depends on
        whether the compiler can show the reindex never has to happen,
        and section 3.2 makes that a classification with three answers:
        a compile-time rename, a real sequence of instructions, or a
        refusal. This op is the question; the layout pass is the answer.
        """
        out = self._value(VECTOR, v.dtype, hint='w')
        self._emit('view', [out], [v], bases=bases, out_dims=out_dims,
                   out_sizes=out_sizes, what=what)
        return out

    def vbin(self, op, a, b, dtype=None, comment=None, fmm=None):
        """A lane-local binary.

        `fmm` is the ISA sec 8.3 min/max NaN mode and is meaningful only
        for `min` and `max`. It is not a tuning knob: MLIR has two
        distinct maxima, `arith.maxnumf` (NaN operand ignored) and
        `arith.maximumf` (NaN operand propagates), and they are `.ieee`
        and `.prop` respectively. Defaulting both to `.ieee` -- which is
        what happened before this argument existed -- silently computed
        the wrong one of the two.
        """
        v = self._value(VECTOR, dtype or a.dtype, hint='v')
        self._emit('vbin', [v], [a, b], op=op, comment=comment, fmm=fmm)
        return v

    def vun(self, op, a, dtype=None, comment=None):
        """A lane-local unary.

        `neg` and `abs` are single ISA instructions (0x14, 0x15) that act
        on the sign bit alone. Synthesising them out of `vbin` would not
        be the same function: `0 - x` is +0.0 for x = +0.0, where negation
        is -0.0, and it does not preserve a NaN payload the way ISA sec
        0x14 promises.
        """
        v = self._value(VECTOR, dtype or a.dtype, hint='v')
        self._emit('vun', [v], [a], op=op, comment=comment)
        return v

    def vfma(self, a, b, c, dtype=None, comment=None):
        """`a*b + c` with a SINGLE rounding (ISA 0x06).

        Single rounding is the whole reason this exists as its own
        builder call: the argument reduction in the math lowerings gets
        its extra precision from the product NOT being rounded before
        the add, and a multiply followed by an add is a different
        function that happens to agree most of the time.
        """
        v = self._value(VECTOR, dtype or a.dtype, hint='v')
        self._emit('vfma', [v], [a, b, c], comment=comment)
        return v

    def vfms(self, a, b, c, dtype=None, comment=None):
        """`a*b - c` with a single rounding (ISA 0x07)."""
        v = self._value(VECTOR, dtype or a.dtype, hint='v')
        self._emit('vfms', [v], [a, b, c], comment=comment)
        return v

    def vcvt(self, src, dst, a, comment=None):
        """A numeric conversion (ISA sec 8.4).

        `src` and `dst` are the ISA's own format names, in the ISA's own
        order: `vcvt('s32', 'f32', x)` is `vf.cvt.s32.f32`, widening a
        signed integer to binary32.
        """
        v = self._value(VECTOR, dst, hint='c')
        self._emit('vcvt', [v], [a], src=src, dst=dst, comment=comment)
        return v

    def vextract(self, v, lane, dtype=None):
        """One lane of a vector, into a scalar register (ISA 0x08).

        `Sd = zext(Va[lane])`, so the scalar holds the lane's 32 bits --
        which for a float is its pattern, and that is what a scalar
        register holds for a float anyway.
        """
        out = self._value(SCALAR, dtype or v.dtype, hint='e')
        self._emit('vextract', [out], [v], lane=lane)
        return out

    def vinsert(self, v, lane, scalar):
        """One scalar into one lane, the others preserved (ISA 0x09).

        In place, like `acc_write`: the instruction reads its
        destination as well as writing it, so the value has to be one
        object across the sixteen inserts that fill a register rather
        than sixteen values the allocator is free to separate.
        """
        self._emit('vinsert', [], [v, scalar], lane=lane)

    def vshuffle(self, kind, v, amount, dtype=None):
        """A lane permutation (ISA sec 0x13 SHUF).

        `xor` gives lane i the value from lane `i ^ amount`, which is
        the compare-exchange partner every bitonic stage asks for.
        """
        out = self._value(VECTOR, dtype or v.dtype, hint='sh')
        self._emit('vshuffle', [out], [v, amount], op=kind)
        return out

    def vcmp(self, cc, a, b):
        p = self._value(PRED, 'i1', hint='m')
        self._emit('vcmp', [p], [a, b], cc=cc)
        return p

    def pbin(self, op, a, b):
        """Predicate logic (ISA sec 0x13).

        `m & (v == b)` is two comparisons and an AND, and the AND is on
        i1 lanes. Routing it through the integer vector unit would have
        needed the predicates in vector registers, which is not where a
        comparison leaves them.
        """
        out = self._value(PRED, 'i1', hint='m')
        self._emit('pbin', [out], [a, b], op=op)
        return out

    def pun(self, op, a):
        out = self._value(PRED, 'i1', hint='m')
        self._emit('pun', [out], [a], op=op)
        return out

    def vselect(self, p, a, b):
        if getattr(p, 'kind', None) != PRED:
            # v.sel takes its condition in a predicate register; a vector
            # here would be read as the predicate register with its number.
            raise Diagnostic(f"vselect on a {getattr(p, 'kind', None)} condition {p!r}; "
                             f"it takes a predicate")
        v = self._value(VECTOR, a.dtype, hint='v')
        self._emit('vselect', [v], [p, a, b])
        return v

    # ---- memory -----------------------------------------------------------
    def load(self, base, *, mask=None, other=0, dtype='f32', size='w'):
        """A contiguous vector load.

        ISA sec 9.1 gives lane i the address base + i*sz, so a contiguous
        block needs no gather. sec 12 wants `other=` materialised explicitly:
        a predicated load leaves masked-off lanes holding whatever was there.
        """
        v = self._value(VECTOR, dtype, hint='ld')
        self._emit('load', [v], [base], mask=mask, other=other, size=size)
        return v

    def store(self, base, value, *, mask=None, size='w', comment=None):
        self._emit('store', [], [base, value], mask=mask, size=size,
                   comment=comment)

    def gather(self, base, index, *, mask=None, other=0, dtype='f32',
               size='w'):
        """Per-lane addressing.

        ISA sec 9.1 scales the index by the access size, so `index` is in
        elements, not bytes. A lowering that computes byte offsets is wrong
        for every lane but lane 0.
        """
        v = self._value(VECTOR, dtype, hint='g')
        self._emit('gather', [v], [base, index], mask=mask, other=other,
                   size=size)
        return v

    def scatter(self, base, index, value, *, mask=None, size='w'):
        self._emit('scatter', [], [base, index, value], mask=mask, size=size)

    def sload(self, base, offset=0, size='d', comment=None, dtype='i64'):
        v = self._value(SCALAR, dtype, hint='m')
        self._emit('sload', [v], [base], offset=offset, size=size,
                   comment=comment)
        return v

    def sstore(self, base, value, offset=0, size='d'):
        # `s.st` names its data register in field `d` and the assembler
        # prints it as `sN`. Handing it a vector register printed that
        # vector's INDEX as a scalar one, so a reduction in v0 stored
        # s0 -- which ISA sec 3 defines to read as zero. Every affected
        # kernel wrote zeros and none of them failed to compile.
        for v, what in ((base, 'address'), (value, 'value')):
            if getattr(v, 'kind', SCALAR) != SCALAR:
                raise TypeError(
                    f"s.st takes scalar registers; its {what} is a "
                    f"{v.kind} ({v.name})")
        self._emit('sstore', [], [base, value], offset=offset, size=size)

    # ---- atomics (ISA sec 9.2) -------------------------------------------
    def atomic(self, op, addr, value, *, size='w', order='relaxed',
               scope='device', dtype='i32'):
        """A read-modify-write at the home slice of the address.

        Returns the value that was there. ISA sec 9.2 executes the whole
        operation inside one step of the slice arbiter, which is what
        makes it indivisible without a lock.
        """
        out = self._value(SCALAR, dtype, hint='a')
        self._emit('atomic', [out], [addr, value], op=op, size=size,
                   order=order, scope=scope)
        return out

    def fence(self, *, order='acq_rel', scope='device'):
        """Wait until every access this core has issued is complete.

        ISA sec 9.2: the LSU waits for a response to a store as well as
        to a load, so draining the LSU IS the fence at core and device
        scope. Asynchronous transfers are deliberately outside it --
        they complete against an mbarrier.
        """
        self._emit('fence', [], [], order=order, scope=scope)

    def atomic_cas(self, addr, expect, desired, *, size='w', order='relaxed',
                   scope='device', dtype='i32'):
        out = self._value(SCALAR, dtype, hint='a')
        self._emit('atomic_cas', [out], [addr, expect, desired], size=size,
                   order=order, scope=scope)
        return out

    def reduce(self, op, v, *, mask=None, dtype=None, broadcast=False):
        """A whole-vector reduction.

        ISA sec 8.7 fixes the order as a balanced binary tree over lane index
        so the result is bit-reproducible, and makes inactive lanes contribute
        the operation's identity -- so a masked reduction needs no pre-select.

        `broadcast` selects sec 8.7's `mod[0] = 1` form, which writes the
        result to every lane of a vector register instead of to a scalar
        one. That is the form a float reduction wants: there is no scalar
        floating-point unit (sec 7 has no `s.fadd`), so a reduced float
        has to end up in a vector register anyway, and the broadcast form
        puts it there without a separate splat.
        """
        out = self._value(VECTOR if broadcast else SCALAR,
                          dtype or v.dtype, hint='r')
        self._emit('reduce', [out], [v], op=op, mask=mask,
                   broadcast=broadcast)
        return out

    # ---- control ----------------------------------------------------------
    def loop(self, counter, limit, body_label='loop'):
        """A counted loop. Returns a context manager over the body."""
        return _Loop(self, counter, limit, body_label)

    def label(self, name):
        self._emit('label', [], [], name=name)

    def branch_nz(self, cond, target):
        self._emit('branch_nz', [], [cond], target=target)

    def exit(self):
        self._emit('exit', [], [])

    def comment(self, text):
        self._emit('comment', [], [], text=text)

    def proton_record(self, name, *, start=True):
        """Effectful Proton scope marker; requires explicit profiling lowering."""
        if not isinstance(name, str) or not name or type(start) is not bool:
            raise ValueError('proton.record needs a nonempty name and boolean start')
        self._emit('proton.record', name=name, isStart=start)

    # ---- resources (sec 10.1: whole-program, fixed before role-local) -----
    def window(self, length, name=None):
        w = Value(name=name or f"w{len(self.kernel.windows)}", kind=WINDOW,
                  length=length)
        self.kernel.windows.append(w)
        self._emit('window_def', [w], [], length=length)
        return w

    def window_reg(self, window, offset=0, dtype='i32'):
        """Name one vector register inside a window.

        ISA sec 9D.2: a window is register aliasing, not a scratchpad, so its
        contents are read as ordinary vector registers. This is the read side
        of the zero-copy handoff sec 7.3 puts above every other option for
        tensors -- and it only works because the two contexts' logical names
        map to the same physical registers.
        """
        if offset >= window.length:
            raise Diagnostic(
                f"register {offset} is past the end of a {window.length}-"
                "register window",
                because='ISA sec 9D.2 makes the window the bounds check',
                source='ISA sec 9D.2')
        v = Value(name=f"{window.name}[{offset}]", kind=VECTOR, dtype=dtype)
        self._emit('window_reg', [v], [window], offset=offset)
        return v

    def window_put(self, window, offset, value):
        """Place a vector value into a window register.

        The write half of the zero-copy handoff: the producer moves its result
        into the range both roles name, and the consumer reads it there. One
        `v.mov`, not a trip through memory -- sec 7.3 puts a shared RF window
        above every other way of moving a tensor between roles, and TNCA has
        no scratchpad for the alternatives anyway.
        """
        self._emit('window_put', [], [window, value], offset=offset)

    def descriptor(self, addr, name=None):
        d = Value(name=name or f"d{len(self.kernel.descriptors)}", kind=DESC)
        self.kernel.descriptors.append(d)
        self._emit('desc_snapshot', [d], [addr])
        return d

    def accumulator(self, name=None):
        a = Value(name=name or f"a{len(self.kernel.accumulators)}", kind=ACC)
        self.kernel.accumulators.append(a)
        return a

    def barrier(self, name=None, init=False):
        """Allocate a barrier.

        `init=False` by default, and that is the interesting part. There is no
        race-free way for one role to wait until another has run `mb.init`:
        waiting means sleeping on the barrier, and ISA sec 9A.6 makes an
        `mb.init` under a sleeping waiter a BARRIER_FAULT. Any handshake built
        to avoid that has to run outside the barrier mechanism -- polling a
        memory flag, which is what this compiler did until it deadlocked: the
        consumer's polling starved the producer's transfer engine of the
        memory port under a slow memory model, so the flag it was waiting for
        could never be written.

        The answer is that barriers are quiescent at kernel entry (ISA sec
        9A.7), so no role has to initialise them and no role has to wait for
        another to do it. `mb.init` remains for re-initialising a barrier
        within a kernel, where the program already knows nobody is waiting.
        """
        b = Value(name=name or f"b{len(self.kernel.barriers)}", kind=BARRIER)
        self.kernel.barriers.append(b)
        if init:
            self._emit('barrier_init', [], [b])
        return b

    # ---- asynchronous -----------------------------------------------------
    def barrier_arm(self, barrier, *, arrivals=0, ops=0):
        """Arm a phase. Returns the token every command counts against.

        sec 8.2: the operation count is what the commands expand to, not the
        number of source-level transfers, so the caller passes the expanded
        count and the lowering checks it against what it actually emits.
        """
        t = self._value(TOKEN, 'i64', hint='t')
        self._emit('barrier_arm', [t], [barrier], arrivals=arrivals, ops=ops)
        return t

    def barrier_token(self, barrier, epoch):
        """Construct a token for a generation this role did not arm.

        ISA sec 9A.2: token = {epoch[47:0], barrier_id[15:0]}. The scalar file
        is private per RCTX (sec 7.3), so a consumer cannot read the
        producer's token out of a register -- it derives the generation from
        the iteration number instead, which is what makes a static pipeline
        possible at all.
        """
        t = self._value(TOKEN, 'i64', hint='tk')
        self._emit('barrier_token', [t], [barrier], epoch=epoch)
        return t

    def barrier_token_dyn(self, barrier, epoch_value):
        """A token whose generation is computed at run time.

        sec 7.3: 对于静态 pipeline，epoch 通常可以由各 role 根据迭代编号独立
        计算；不必真的跨角色搬运每个 token. That is what makes a *rolled* role
        loop possible at all -- the consumer cannot read the producer's token
        out of a register, because the scalar file is per-RCTX private, but it
        can compute the same generation from its own trip counter.

        ISA sec 9A.2: token = {epoch[47:0], barrier_id[15:0]}.
        """
        t = self._value(TOKEN, 'i64', hint='tk')
        self._emit('barrier_token_dyn', [t], [barrier, epoch_value])
        return t

    def barrier_wait(self, token):
        self._emit('barrier_wait', [], [token])

    def barrier_arrive(self, barrier):
        """Signal one software arrival. ISA sec 9A.3: mb.arrive names the
        barrier, not a token -- it releases against whatever generation is
        currently armed."""
        self._emit('barrier_arrive', [], [barrier])

    def dma_load(self, window, desc, coords, token, *, dtype='f16',
                 mask=None, transpose=False):
        """An asynchronous tile load into a window.

        The coordinates must live in consecutive scalar registers (ISA
        sec 9E); the lowering enforces that rather than hoping the allocator
        happened to place them so.

        `transpose` is sec 9E's `mod[5]`, which asks the engine to write
        the tile transposed. It is how `tl.trans` costs nothing: the
        transpose happens while the data is already in flight, instead
        of as a permutation of registers the ISA has no instruction for.
        """
        e = self._value(EVENT, 'i1', hint='e')
        self._emit('dma_load', [e], [window, desc, token] + list(coords),
                   dtype=dtype, rank=len(coords), mask=mask,
                   transpose=transpose)
        return e

    def dma_store(self, window, desc, coords, token, *, dtype='f16',
                  mask=None):
        """An asynchronous tile store out of a window.

        The mirror image of dma_load, and the reason sec 13.4 lists three
        implicit footprints rather than two: the engine reads the source
        window LATE -- after the issuing instruction has retired -- so the
        window stays live until the completion event, which is exactly the
        range a liveness analysis built from the operand fields ends early.
        """
        e = self._value(EVENT, 'i1', hint='e')
        self._emit('dma_store', [e], [window, desc, token] + list(coords),
                   dtype=dtype, rank=len(coords), mask=mask)
        return e

    def acc_zero(self, acc):
        self._emit('acc_zero', [], [acc])

    def mma_async(self, acc, a_window, b_window, token, *, dtype='f16',
                  k='k16'):
        e = self._value(EVENT, 'i1', hint='e')
        self._emit('mma_async', [e], [acc, a_window, b_window, token],
                   dtype=dtype, k=k)
        return e

    def acc_write(self, acc, row, value):
        """One row of a vector value into an accumulator (ISA 0x01).

        The write half of sec 9B. It is what lets a value computed in
        the vector unit be an MMA's accumulator input, and sec 5.4 names
        FlashAttention's rescale of O as the reason the round trip must
        not go through memory.
        """
        self._emit('acc_write', [], [acc, row, value])

    def acc_read(self, acc, row):
        v = self._value(VECTOR, 'f32', hint='ac')
        self._emit('acc_read', [v], [acc, row])
        return v

    # ---- assertions, for tests --------------------------------------------
    def assert_eq(self, s, expect, ident):
        self._emit('assert_eq', [], [s], expect=expect, ident=ident)

    def assert_lane(self, v, lane, expect, ident):
        self._emit('assert_lane', [], [v], lane=lane, expect=expect,
                   ident=ident)


class _Loop:
    def __init__(self, builder, counter, limit, label):
        self.b = builder
        self.counter = counter
        self.limit = limit
        self.label = label

    def __enter__(self):
        self.b.label(self.label)
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        b = self.b
        nxt = b.sbini('add', self.counter, 1)
        b._emit('assign', [], [self.counter, nxt])
        # An int becomes a constant; a Value is already one. `b.const` used
        # to accept a Value and format its repr as an immediate, so
        # `b.loop(i, b.const(4000))` assembled to
        #     .li  s3, #%c2:scalar@s2
        # and failed in the assembler rather than in the builder that had
        # been handed the wrong kind of thing.
        lim = self.limit if isinstance(self.limit, Value) else b.const(self.limit)
        more = b.scmp('lt', self.counter, lim)
        b.branch_nz(more, self.label)
        return False


def make_kernel(name, num_roles=1):
    k = Kernel(name=name)
    for i in range(num_roles):
        k.roles.append(Role(index=i, name=f"role{i}"))
    return k

# ---------------------------------------------------------------------------
# The textual form
#
# An IR you cannot read is an IR you cannot review, so the kernel prints
# itself -- the same arrangement LLVM and MLIR use, and for the same reason:
# the printer lives next to the data structure and cannot drift from it.
#
# The form is deliberately close to the in-memory structure rather than
# pretty: `%name:kind@reg` is exactly what Value.__repr__ produces, so a
# dump and a debugger agree about what a value is called. Registers appear
# only after allocation, which is how a reader can tell a pre-allocation
# dump from a post-allocation one without being told.
# ---------------------------------------------------------------------------

def format_op(op, indent='    '):
    """One operation, one line."""
    if op.kind == 'proton.record':
        import json
        phase = 'start' if op.attrs['isStart'] else 'end'
        return f"{indent}proton.record {phase} {json.dumps(op.attrs['name'])}"
    if op.kind == 'label':
        return f"  ^{op.attrs.get('name', '?')}:"
    if op.kind == 'comment':
        return f"{indent}// {op.attrs.get('text', '')}"
    res = ', '.join(repr(r) for r in op.results)
    ops = ', '.join(repr(o) for o in op.operands)
    attrs = ' '.join(f"{k}={v!r}" for k, v in sorted(op.attrs.items())
                     if k != 'comment')
    line = f"{indent}{res} = " if res else indent
    line += f"tnca.{op.kind}({ops})"
    if attrs:
        line += f"  {{{attrs}}}"
    note = op.attrs.get('comment')
    if note:
        line += f"   // {note}"
    return line


def format_region(region, indent='    '):
    return [format_op(op, indent) for op in region.ops]


def format_kernel(kernel):
    """The whole kernel as text.

    Resources first, because they are fixed before any role-local
    allocation (sec 10.1) and reading the roles without them is reading a
    program without its declarations.
    """
    out = [f"tnca.kernel @{kernel.name} {{"]

    if kernel.args:
        out.append('  // arguments, bound through ARG_BASE')
        for name, info in kernel.args.items():
            out.append(f"  arg {name} = {info!r}")

    resources = [('barrier', kernel.barriers),
                 ('accumulator', kernel.accumulators),
                 ('window', kernel.windows),
                 ('descriptor', kernel.descriptors)]
    declared = [(what, items) for what, items in resources if items]
    if declared:
        out.append('  // whole-program resources')
        for what, items in declared:
            out.append(f"  {what}s: {len(items)}  {items!r}")
    elif not kernel.args:
        out.append('  // no shared resources, no bound arguments')

    for role in kernel.roles:
        out.append('')
        out.append(f"  role {role.index} @{role.name} {{")
        out.extend(format_region(role.region))
        out.append('  }')
    out.append('}')
    return '\n'.join(out) + '\n'
