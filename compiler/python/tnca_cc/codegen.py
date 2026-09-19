"""
TNCA IR -> assembly.

The second half of the bring-up path of compiler_overview.md sec 11.1. There
is no general machine optimiser here, on purpose (这一阶段不建设复杂的通用
寄存器优化器); allocation is a linear scan over last uses, and scheduling is
program order.

What is not simplified is ownership. Three things run alongside the walk:

  * an OwnershipChecker (sec 13.4), holding the *implicit* register ranges an
    asynchronous command reads -- a whole MMA fragment, a transfer's
    consecutive coordinates, an asynchronous store's source window -- so a
    later write into one of them is caught here rather than becoming a wrong
    number on silicon;

  * a ProtocolModel (sec 16.2), stepped by the same operations the assembly
    performs, so the emitted barrier arming and waiting is checked against the
    model the protocol tests enumerate;

  * a check that a transfer's coordinates really landed in consecutive scalar
    registers (ISA sec 9E), rather than hoping the allocator happened to.

sec 11.1: 资源不够可以先明确报错。Nothing here approximates.
"""
from .asm import ASM_SCRATCH_SREGS, AsmBuilder, RegisterAllocator
from .diagnostics import Diagnostic, ResourceError
from .footprints import (OwnershipChecker, RegRange, fragment_registers)
from .ir import (ACC, BARRIER, DESC, EVENT, PRED, SCALAR, TOKEN, VECTOR,
                 WINDOW)
from .protocol import ProtocolModel

# IR op -> mnemonic, for the plain arithmetic cases.
_SBIN = {'add': 's.add', 'sub': 's.sub', 'mul': 's.mul', 'and': 's.and',
         'or': 's.or', 'xor': 's.xor', 'shl': 's.shl', 'lshr': 's.lshr',
         'ashr': 's.ashr', 'min': 's.min.s', 'max': 's.max.s'}
_SBINI = {'add': 's.addi', 'sub': 's.subi', 'mul': 's.muli',
          'and': 's.andi', 'or': 's.ori', 'xor': 's.xori',
          'shl': 's.shli', 'lshr': 's.lshri', 'ashr': 's.ashri'}
_VBIN_INT = {'add': 'v.add', 'sub': 'v.sub', 'mul': 'v.mul'}
_VBIN_FP = {'add': 'vf.add', 'sub': 'vf.sub', 'mul': 'vf.mul',
            'min': 'vf.min', 'max': 'vf.max'}
_REDUCE_INT = {'add': 'v.reduce.add', 'max': 'v.reduce.max.s',
               'min': 'v.reduce.min.s', 'and': 'v.reduce.and',
               'or': 'v.reduce.or', 'xor': 'v.reduce.xor'}
_REDUCE_FP = {'add': 'vf.reduce.add', 'max': 'vf.reduce.max',
              'min': 'vf.reduce.min'}


def _is_fp(dtype):
    return dtype in ('f32', 'f16', 'bf16')


class CodeGen:
    """Lower one role's region to assembly."""

    def __init__(self, kernel, target, builder=None, *, check_protocol=True):
        self.kernel = kernel
        self.target = target
        self.b = builder or AsmBuilder(target, kernel.name)
        self.alloc = RegisterAllocator(target)
        self.own = OwnershipChecker()
        self.protocol = ProtocolModel(
            depth=max(1, len(kernel.windows)),
            consumers=[r.name for r in kernel.roles[1:]] or ['consumer0'],
            num_barriers=target.num_barriers,
            separate_source_completion=target.has_separate_source_completion,
        ) if check_protocol else None

        self._reg = {}          # Value -> physical index
        self._last_use = {}     # Value -> op position
        self._events = {}       # Value(EVENT) -> command id in the model
        self._token_barrier = {}  # Value(TOKEN) -> barrier index
        self._armed_ops = {}    # barrier index -> operations promised
        self._issued_ops = {}   # barrier index -> operations emitted
        self._mma_pending = {}  # barrier index -> fragment holds to release

    # ---- whole-program resources (sec 10.1) -------------------------------
    def assign_shared(self):
        """Fix barriers, accumulators, windows and descriptors first.

        sec 10.1 splits allocation in two and puts these first because they
        are shared across roles; assigning them after role-local temporaries
        would let one role's scratch sit inside another's window.
        """
        k, t = self.kernel, self.target
        t.check_budget(barriers=len(k.barriers),
                       accumulators=len(k.accumulators),
                       windows=len(k.windows),
                       descriptors=len(k.descriptors),
                       rctx=len(k.roles))
        for i, b in enumerate(k.barriers):
            b.reg = i
        for i, a in enumerate(k.accumulators):
            a.reg = i
        for i, d in enumerate(k.descriptors):
            d.reg = i

        # Windows are contiguous vector ranges, packed from the top of the
        # file downwards so role-local temporaries get the low registers.
        cursor = t.vregs_per_rctx
        for w in k.windows:
            cursor -= w.length
            if cursor < 0:
                raise ResourceError(
                    f"windows need more than {t.vregs_per_rctx} vector "
                    "registers",
                    because='a window is a contiguous register range with no '
                            'storage of its own (ISA sec 9D.2)',
                    remedy='fewer pipeline stages, or a smaller tile',
                    source='compiler_overview.md sec 10.1')
            w.reg = cursor
            self.alloc.reserve_vector(cursor, w.length)

    # ---- register handling ------------------------------------------------
    def _compute_last_uses(self, ops):
        """Last use of each value, accounting for loop back edges.

        A linear scan is wrong here. A value defined before a loop and used
        inside it is live again on the next iteration, so freeing it at its
        textually last use lets the allocator hand the register to something
        else and the loop reads the wrong number on its second trip. That is a
        wrong answer rather than a slow one, and it is invisible in a program
        that happens to run one iteration.
        """
        defined_at = {}
        for i, op in enumerate(ops):
            for v in op.results:
                defined_at.setdefault(id(v), i)

        def note(v, i):
            if v is not None and getattr(v, 'kind', None):
                self._last_use[id(v)] = max(self._last_use.get(id(v), -1), i)

        for i, op in enumerate(ops):
            for v in op.operands:
                note(v, i)
            note(op.attrs.get('mask'), i)

        # Loops: a label, and a later branch back to it.
        labels = {op.attrs['name']: i
                  for i, op in enumerate(ops) if op.kind == 'label'}
        loops = []
        for i, op in enumerate(ops):
            if op.kind == 'branch_nz' and op.attrs.get('target') in labels:
                loops.append((labels[op.attrs['target']], i))

        # Innermost first, so an outer loop's extension covers the inner one.
        for start, end in sorted(loops, key=lambda p: p[1] - p[0]):
            for key, last in list(self._last_use.items()):
                born = defined_at.get(key, -1)
                if born < start and start <= last <= end:
                    self._last_use[key] = end

    def _alloc(self, v):
        if v.reg is not None:
            return v.reg
        if v.kind == SCALAR:
            v.reg = self.alloc.scalar()
        elif v.kind == VECTOR:
            v.reg = self.alloc.vector()
        elif v.kind == PRED:
            v.reg = self.alloc.predicate()
        elif v.kind in (EVENT, TOKEN):
            v.reg = self.alloc.scalar()      # a barrier token is a scalar
        else:
            raise Diagnostic(f"cannot allocate a {v.kind} here")
        return v.reg

    def _free_dead(self, position, ops):
        for op in ops[:position + 1]:
            for v in op.results:
                if v.kind in (VECTOR, SCALAR, PRED) and v.reg is not None \
                        and self._last_use.get(id(v)) == position:
                    kindc = {VECTOR: 'v', SCALAR: 's', PRED: 'p'}[v.kind]
                    try:
                        self.alloc.free(kindc, v.reg)
                    except Diagnostic:
                        pass                  # part of a reserved window
                    v.reg = None

    @staticmethod
    def _n(v, prefix):
        return f"{prefix}{v.reg}"

    def _pred(self, mask):
        return f"p{mask.reg}" if mask is not None else None

    # ---- the walk ---------------------------------------------------------
    def run(self, role_index=0):
        role = self.kernel.role(role_index)
        ops = role.region.ops
        self._compute_last_uses(ops)
        for i, op in enumerate(ops):
            getattr(self, f"_op_{op.kind}", self._unhandled)(op)
            self._free_dead(i, ops)
        self._finish()
        return self.b

    def _unhandled(self, op):
        raise Diagnostic(
            f"no lowering for IR operation {op.kind!r}",
            remedy='add one, or raise earlier where the construct is still '
                   'recognisable',
            source='compiler_overview.md sec 11.1')

    def _finish(self):
        """Every asynchronous command must have been awaited (sec 10.2)."""
        for bar, promised in self._armed_ops.items():
            issued = self._issued_ops.get(bar, 0)
            if issued != promised:
                raise Diagnostic(
                    f"barrier b{bar} was armed for {promised} operation(s) "
                    f"but {issued} were issued",
                    because='sec 8.2 takes the operation count after the '
                            'commands expand, so an arm that does not match '
                            'what was emitted either hangs or releases early',
                    remedy='arm with the expanded count',
                    source='compiler_overview.md sec 8.2')
        if self.protocol:
            live = [c for c, l in self.protocol.leases.items() if not l.retired]
            if live:
                raise Diagnostic(
                    f"commands {live} were never awaited",
                    because='an asynchronous lifetime ends at a completion '
                            'event; without one the registers are owned for '
                            'ever',
                    source='compiler_overview.md sec 10.2')

    # ---- scalar ops -------------------------------------------------------
    def _op_comment(self, op):
        self.b.comment(op.attrs['text'])

    def _op_label(self, op):
        self.b.label(op.attrs['name'])

    def _op_const(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.directive(f".li       s{v.reg}, #{op.attrs['value']}")

    def _op_program_id(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.emit('s.rdspecial', f"s{v.reg}",
                    f"pid.{'xyz'[op.attrs['dim']]}")

    def _op_special(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.emit('s.rdspecial', f"s{v.reg}", op.attrs['name'])

    def _op_sbin(self, op):
        a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        self.b.emit(_SBIN[op.attrs['op']], f"s{v.reg}", f"s{a.reg}",
                    f"s{b.reg}", comment=op.attrs.get('comment'))

    def _op_sbini(self, op):
        a = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        self.b.emit(_SBINI[op.attrs['op']], f"s{v.reg}", f"s{a.reg}",
                    f"#{op.attrs['imm']}", comment=op.attrs.get('comment'))

    def _op_scmp(self, op):
        a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        self.b.emit(f"s.cmp.{op.attrs['cc']}", f"s{v.reg}", f"s{a.reg}",
                    f"s{b.reg}")

    def _op_assign(self, op):
        dst, src = op.operands
        self.b.emit('s.mov', f"s{dst.reg}", f"s{src.reg}")

    # ---- vector ops -------------------------------------------------------
    def _op_arange(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.emit('v.iota', f"v{v.reg}", comment='arange(0, VLEN)')

    def _op_splat(self, op):
        s = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        self.b.emit('v.splat', f"v{v.reg}", f"s{s.reg}")

    def _op_vmovi(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.emit('v.movi', f"v{v.reg}", f"#{op.attrs['imm']}")

    def _op_vbin(self, op):
        a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        table = _VBIN_FP if _is_fp(v.dtype) else _VBIN_INT
        if _is_fp(v.dtype):
            self.target.require('vector_fp32',
                                f"emit a {v.dtype} {op.attrs['op']}")
        mnem = table.get(op.attrs['op'])
        if mnem is None:
            raise Diagnostic(
                f"no {v.dtype} lowering for {op.attrs['op']!r}",
                source='compiler_overview.md sec 12')
        self.b.emit(mnem, f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    comment=op.attrs.get('comment'))

    def _op_vcmp(self, op):
        a, b = op.operands
        p = op.results[0]
        self._alloc(p)
        mnem = 'vf.cmp' if _is_fp(a.dtype) else 'v.cmp'
        self.b.emit(f"{mnem}.{op.attrs['cc']}", f"p{p.reg}", f"v{a.reg}",
                    f"v{b.reg}")

    def _op_vselect(self, op):
        p, a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        self.b.emit('v.sel', f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    pred=f"p{p.reg}")

    # ---- memory -----------------------------------------------------------
    def _op_load(self, op):
        base = op.operands[0]
        v = op.results[0]
        mask = op.attrs.get('mask')
        self._alloc(v)
        if mask is not None:
            # sec 12: `other=` is a value. A predicated load leaves inactive
            # lanes holding what the register had, so put the value there.
            self.b.emit('v.movi', f"v{v.reg}", f"#{op.attrs['other']}",
                        comment='other=, materialised (sec 12)')
        self.b.emit(f"v.ld.{op.attrs['size']}", f"v{v.reg}",
                    f"[s{base.reg}, #0]", pred=self._pred(mask))

    def _op_store(self, op):
        base, value = op.operands
        self.b.emit(f"v.st.{op.attrs['size']}", f"[s{base.reg}, #0]",
                    f"v{value.reg}", pred=self._pred(op.attrs.get('mask')))

    def _op_gather(self, op):
        base, index = op.operands
        v = op.results[0]
        mask = op.attrs.get('mask')
        self.target.require('gather_scatter', 'emit v.ld.gather')
        self._alloc(v)
        if mask is not None:
            self.b.emit('v.movi', f"v{v.reg}", f"#{op.attrs['other']}",
                        comment='other=, materialised (sec 12)')
        # ISA sec 9.1 scales the index by the access size: these are element
        # indices, not byte offsets.
        self.b.emit(f"v.ld.gather.{op.attrs['size']}", f"v{v.reg}",
                    f"[s{base.reg}, v{index.reg}]", pred=self._pred(mask),
                    comment='index is in elements (ISA sec 9.1)')

    def _op_scatter(self, op):
        base, index, value = op.operands
        self.target.require('gather_scatter', 'emit v.st.scatter')
        self.b.emit(f"v.st.scatter.{op.attrs['size']}",
                    f"[s{base.reg}, v{index.reg}]", f"v{value.reg}",
                    pred=self._pred(op.attrs.get('mask')))

    def _op_reduce(self, op):
        v = op.operands[0]
        s = op.results[0]
        self._alloc(s)
        table = _REDUCE_FP if _is_fp(v.dtype) else _REDUCE_INT
        mnem = table.get(op.attrs['op'])
        if mnem is None:
            raise Diagnostic(f"no {v.dtype} reduction {op.attrs['op']!r}")
        # ISA sec 8.7: inactive lanes contribute the identity, so a masked
        # reduction needs no pre-select.
        self.b.emit(mnem, f"s{s.reg}", f"v{v.reg}",
                    pred=self._pred(op.attrs.get('mask')))

    # ---- control ----------------------------------------------------------
    def _op_branch_nz(self, op):
        c = op.operands[0]
        self.b.emit('br.nz', f"s{c.reg}", op.attrs['target'])

    def _op_exit(self, op):
        self.b.emit('exit')

    # ---- assertions -------------------------------------------------------
    def _op_assert_eq(self, op):
        s = op.operands[0]
        self.b.directive(f".assert_eq s{s.reg}, #{op.attrs['expect']:#x}, "
                         f"#{op.attrs['ident']}")

    def _op_assert_lane(self, op):
        v = op.operands[0]
        self.b.directive(f".assert_lane v{v.reg}, #{op.attrs['lane']}, "
                         f"#{op.attrs['expect']:#x}, #{op.attrs['ident']}")

    # ---- resources --------------------------------------------------------
    def _op_window_def(self, op):
        w = op.results[0]
        if w.reg is None:
            raise Diagnostic(
                f"window {w.name} has no register range",
                because='whole-program resources are fixed before the walk '
                        '(sec 10.1); call assign_shared() first')
        self.b.emit('w.def', f"w{self.kernel.windows.index(w)}",
                    f"#{w.reg}", f"#{w.length}",
                    comment=f"v{w.reg}..v{w.reg + w.length - 1}")

    def _op_desc_snapshot(self, op):
        addr = op.operands[0]
        d = op.results[0]
        self.target.require('tensor_dma', 'snapshot a TensorMap')
        self.b.emit('td.load', f"d{d.reg}", f"[s{addr.reg}]",
                    comment='a versioned snapshot, not a pointer '
                            '(ISA sec 9D.1)')

    def _op_barrier_init(self, op):
        b = op.operands[0]
        self.target.require('mbarrier', 'initialise an mbarrier')
        self.b.emit('mb.init', f"b{b.reg}")
        if self.protocol:
            self.protocol.barrier_init(b.reg)

    def _op_barrier_arm(self, op):
        bar = op.operands[0]
        tok = op.results[0]
        self._alloc(tok)
        arrivals, ops_n = op.attrs['arrivals'], op.attrs['ops']
        self.b.emit('mb.arm', f"s{tok.reg}", f"b{bar.reg}",
                    f"#{arrivals}", f"#{ops_n}",
                    comment=f"{arrivals} arrival(s), {ops_n} operation(s)")
        self._token_barrier[id(tok)] = bar.reg
        self._armed_ops[bar.reg] = ops_n
        self._issued_ops[bar.reg] = 0
        if self.protocol:
            self.protocol.barrier_arm(bar.reg, arrivals=arrivals, ops=ops_n)

    def _op_barrier_wait(self, op):
        tok = op.operands[0]
        self.b.emit('mb.wait', f"s{tok.reg}")
        bar = self._token_barrier.get(id(tok))
        if bar is None:
            return
        # Waiting retires every command counted on this phase: that is what a
        # completion event means (sec 10.2), and it is the only point at which
        # the fragments and coordinate registers become reusable.
        if self.protocol:
            for cmd, lease in list(self.protocol.leases.items()):
                if lease.barrier == bar and not lease.retired:
                    self.protocol.complete(cmd)
                    self.own.release(cmd)
        for cmd in self._mma_pending.pop(bar, []):
            self.own.release(cmd)

    def _op_barrier_arrive(self, op):
        tok = op.operands[0]
        self.b.emit('mb.arrive', f"s{tok.reg}")

    # ---- asynchronous commands --------------------------------------------
    def _count_operation(self, bar):
        self._issued_ops[bar] = self._issued_ops.get(bar, 0) + 1
        promised = self._armed_ops.get(bar)
        if promised is not None and self._issued_ops[bar] > promised:
            raise Diagnostic(
                f"barrier b{bar} was armed for {promised} operation(s) but a "
                f"{self._issued_ops[bar]}th was issued",
                because='sec 8.2: the count is taken after the commands '
                        'expand, so arming for fewer than are emitted '
                        'releases the waiter early',
                source='compiler_overview.md sec 8.2')

    def _op_dma_load(self, op):
        window, desc, token = op.operands[0], op.operands[1], op.operands[2]
        coords = op.operands[3:]
        self.target.require('tensor_dma', 'emit t.ld.async')
        if len(coords) > self.target.dma_max_rank:
            raise Diagnostic(
                f"a rank-{len(coords)} transfer on a target whose engine "
                f"executes rank {self.target.dma_max_rank}",
                because='ISA sec 9E: a higher rank raises DESCRIPTOR_FAULT '
                        'rather than being approximated',
                remedy='flatten the descriptor if it is provably contiguous, '
                       'or fall back to an LSU loop',
                source='compiler_overview.md sec 12')

        # ISA sec 9E: consecutive scalar registers starting at Sb. Place them
        # explicitly; the instruction names only the first, so a scattered
        # allocation would read whatever sat next to it.
        run = self.alloc.scalar_run(len(coords))
        for c, reg in zip(coords, run):
            self.b.emit('s.mov', f"s{reg}", f"s{c.reg}",
                        comment='coordinates must be consecutive (ISA sec 9E)')

        bar = self._token_barrier[id(token)]
        self._count_operation(bar)
        suffix = {'f16': 'h', 'bf16': 'h', 'f32': 'w'}[op.attrs['dtype']]
        widx = self.kernel.windows.index(window)
        self.b.emit(f"t.ld.async.{suffix}", f"w{widx}", f"d{desc.reg}",
                    f"s{run[0]}", f"s{token.reg}")

        if self.protocol:
            from .protocol import BufferState
            if self.protocol.buffers[widx].state is BufferState.FREE:
                self.protocol.producer_acquire(widx, 'producer')
            cmd = self.protocol.issue_async(widx, 'producer', bar, kind='dma')
            self._events[id(op.results[0])] = cmd
            # sec 13.4: the engine reads the coordinate registers and writes
            # the whole window, neither of which the encoding names in full.
            self.own.hold(cmd, [RegRange('s', run[0], len(run)),
                                RegRange('v', window.reg, window.length)],
                          't.ld.async')

        # The coordinate copies are dead once the command has them, but the
        # engine keeps reading the registers, so the ownership hold above --
        # not this free -- is what protects them.
        for reg in run:
            self.alloc.free('s', reg)

    def _op_acc_zero(self, op):
        a = op.operands[0]
        self.b.emit('m.zero', f"a{a.reg}")

    def _op_mma_async(self, op):
        acc, aw, bw, token = op.operands
        cap = 'matrix_bf16' if op.attrs['dtype'] == 'bf16' else 'matrix_f16'
        self.target.require(cap, f"emit m.mma.async with "
                                 f"{op.attrs['dtype']} inputs")
        bar = self._token_barrier[id(token)]
        self._count_operation(bar)
        self.b.emit(f"m.mma.async.{op.attrs['k']}.{op.attrs['dtype']}",
                    f"a{acc.reg}", f"v{aw.reg}", f"v{bw.reg}",
                    f"s{token.reg}")
        # sec 13.4 / ISA sec 9C.3: the matrix unit reads both fragments in
        # full, for many cycles after this instruction retires. Hold the whole
        # range rather than the one register the encoding names.
        m, n, k = self.target.mma_shape
        bits = 16 if op.attrs['dtype'] in ('f16', 'bf16') else 32
        want_a = fragment_registers(m, k, bits)
        want_b = fragment_registers(k, n, bits)
        for w, want, which in ((aw, want_a, 'A'), (bw, want_b, 'B')):
            if w.length < want:
                raise Diagnostic(
                    f"the {which} window is {w.length} registers, but a "
                    f"{m}x{n}x{k} {op.attrs['dtype']} fragment occupies "
                    f"{want}",
                    because='ISA sec 9D.2 makes the window the transfer '
                            "engine's bounds check; a short one faults rather "
                            'than reading a neighbour',
                    source='compiler_overview.md sec 9')

        # The matrix unit is not modelled as a channel buffer -- it reads two
        # windows and writes an accumulator -- so the hold is tracked here
        # rather than through the channel protocol.
        cmd = f"mma{len(self._events)}"
        self.own.hold(cmd, [RegRange('v', aw.reg, want_a),
                            RegRange('v', bw.reg, want_b)], 'm.mma.async')
        self._events[id(op.results[0])] = cmd
        self._mma_pending.setdefault(bar, []).append(cmd)

    def _op_acc_read(self, op):
        acc, row = op.operands
        v = op.results[0]
        self._alloc(v)
        self.b.emit('a.vread', f"v{v.reg}", f"a{acc.reg}", f"s{row.reg}")
