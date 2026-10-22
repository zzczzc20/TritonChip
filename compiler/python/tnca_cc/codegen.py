"""
TNCA IR -> assembly.

The second half of the bring-up path of compiler_overview.md sec 11.1.
The default allocator is a linear scan over last uses. An opt-in bank
preference uses the profile's declared bank geometry; it changes temporary
register placement while preserving the same lifetime and ownership checks.

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
from .asm import AsmBuilder, RegisterAllocator
from .diagnostics import Diagnostic, ResourceError
from .footprints import (OwnershipChecker, RegRange, fragment_registers,
                         mma_shape_of)
from .ir import EVENT, PRED, SCALAR, TOKEN, VECTOR
from .protocol import ProtocolModel

# IR op -> mnemonic, for the plain arithmetic cases.
_SBIN = {'add': 's.add', 'sub': 's.sub', 'mul': 's.mul', 'and': 's.and',
         'or': 's.or', 'xor': 's.xor', 'shl': 's.shl', 'lshr': 's.lshr',
         'ashr': 's.ashr', 'min': 's.min.s', 'max': 's.max.s'}
_SBINI = {'add': 's.addi', 'sub': 's.subi', 'mul': 's.muli',
          'and': 's.andi', 'or': 's.ori', 'xor': 's.xori',
          'shl': 's.shli', 'lshr': 's.lshri', 'ashr': 's.ashri'}
_VBIN_INT = {'add': 'v.add', 'sub': 'v.sub', 'mul': 'v.mul',
             'and': 'v.and', 'or': 'v.or', 'xor': 'v.xor',
             'andn': 'v.andn', 'shl': 'v.shl', 'lshr': 'v.lshr',
             'ashr': 'v.ashr', 'min': 'v.min.s', 'max': 'v.max.s',
             'minu': 'v.min.u', 'maxu': 'v.max.u'}
_VBIN_FP = {'add': 'vf.add', 'sub': 'vf.sub', 'mul': 'vf.mul',
            'min': 'vf.min', 'max': 'vf.max'}
# Lane-local unaries. Float only: the ISA spells integer negation as a
# subtract from zero, which is exact, while float negation is a sign-bit
# flip that a subtract does not reproduce for zero or for NaN.
_VUN_FP = {'neg': 'vf.neg', 'abs': 'vf.abs'}

# ISA sec 0x13. Predicates have their own logic unit; an `and` of two
# comparisons belongs here and not in the integer vector unit.
_PBIN = {'and': 'p.and', 'or': 'p.or', 'xor': 'p.xor', 'andn': 'p.andn'}
_PUN = {'not': 'p.not', 'mov': 'p.mov'}

# ISA sec 8.8. Approximate by contract, not by accident: each is
# specified to <= 2**-22 relative error, which is roughly four units in
# the last place of binary32. Nothing here rounds correctly, so the
# lowerings in mathlib.py refine these rather than returning them.
_VUN_SFU = {'exp2': 'v.exp2.approx', 'rcp': 'v.rcp.approx',
            'rsqrt': 'v.rsqrt.approx', 'log2': 'v.log2.approx'}
# ISA sec 9.2. Integer only, and 4 or 8 bytes only: mod[2:0] accepts
# nothing narrower, because the memory's word is 8 bytes and a sub-word
# read-modify-write would need a path nothing else in the design has.
_ATOMIC = {'add': 'atom.add', 'and': 'atom.and', 'or': 'atom.or',
           'xor': 'atom.xor', 'xchg': 'atom.xchg', 'min': 'atom.min.s',
           'max': 'atom.max.s', 'minu': 'atom.min.u', 'maxu': 'atom.max.u'}

_REDUCE_INT = {'add': 'v.reduce.add', 'max': 'v.reduce.max.s',
               'min': 'v.reduce.min.s', 'and': 'v.reduce.and',
               'or': 'v.reduce.or', 'xor': 'v.reduce.xor',
               'maxu': 'v.reduce.max.u', 'minu': 'v.reduce.min.u'}
_REDUCE_FP = {'add': 'vf.reduce.add', 'max': 'vf.reduce.max',
              'min': 'vf.reduce.min'}


def _is_fp(dtype):
    return dtype in ('f32', 'f16', 'bf16')


class CodeGen:
    """Lower one role's region to assembly."""

    def __init__(self, kernel, target, builder=None, *, check_protocol=True,
                 opt_level=0, bank_alloc=False, window_reuse=False):
        # Phase 3.5 section 4.3. At 0 this is B0: every lowering decision
        # below is the one the stage was frozen with, so O0 and B0 emit
        # the same bytes. From 1 up, a lowering may choose a cheaper form
        # for the same meaning.
        self.opt_level = int(opt_level or 0)
        self.window_reuse = bool(window_reuse)
        self.kernel = kernel
        self.target = target
        self.b = builder or AsmBuilder(target, kernel.name)
        self.alloc = RegisterAllocator(target)
        self._bank_reads = {}
        if bank_alloc:
            from .bankcost import geometry
            self.bank_banks = geometry(target)
        else:
            self.bank_banks = 0
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
        counts = dict(barriers=len(k.barriers), accumulators=len(k.accumulators),
                      descriptors=len(k.descriptors))
        limits = dict(barriers=t.num_barriers, accumulators=t.num_accumulators,
                      descriptors=t.num_descriptors)
        # A class that does not fit one slot per value is numbered by live
        # range instead (resource_lifetime); one that fits keeps the fixed
        # numbering below, as it always had.
        renumbered = set()
        if any(counts[c] > limits[c] for c in counts):
            from .resource_lifetime import color
            for cls, kind in (('barriers', 'barrier'), ('accumulators', 'accumulator'),
                              ('descriptors', 'descriptor')):
                if counts[cls] > limits[cls]:
                    used = color(k, kind, limits[cls])
                    if used is not None:
                        counts[cls] = used
                        renumbered.add(cls)
        windows = len(k.windows)
        if windows > t.num_windows:
            from .resource_lifetime import color
            used = color(k, 'window', t.num_windows)
            if used is not None:
                windows = used
                renumbered.add('windows')
        t.check_budget(barriers=counts['barriers'],
                       accumulators=counts['accumulators'],
                       windows=windows,
                       descriptors=counts['descriptors'],
                       rctx=len(k.roles))
        if 'barriers' not in renumbered:
            for i, b in enumerate(k.barriers):
                b.reg = i
        if 'accumulators' not in renumbered:
            for i, a in enumerate(k.accumulators):
                a.reg = i
        if 'descriptors' not in renumbered:
            for i, d in enumerate(k.descriptors):
                d.reg = i
        self.renumbered = renumbered

        if self.window_reuse:
            from .window_lifetime import assign
            self._compute_last_uses(k.role(0).region.ops)
            count = assign(k, t.vregs_per_rctx, self._last_use)
            if count:
                self.alloc.reserve_vector(t.vregs_per_rctx - count, count)
            self.window_registers = count
            return

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
                self._last_use.setdefault(id(v), i)

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
            cost = None
            if self.bank_banks:
                neighbors = self._bank_reads.get(id(v), ())
                def cost(candidate):
                    # Only currently mapped co-read values inform this choice.
                    return sum(weight for other, weight in neighbors
                               if other.reg is not None and
                               other.reg % self.bank_banks == candidate % self.bank_banks)
            v.reg = self.alloc.vector(cost, self.bank_banks)
        elif v.kind == PRED:
            v.reg = self.alloc.predicate()
        elif v.kind in (EVENT, TOKEN):
            v.reg = self.alloc.scalar()      # a barrier token is a scalar
        else:
            raise Diagnostic(f"cannot allocate a {v.kind} here")
        return v.reg

    # ------------------------------------------------------------------
    # Every emitted instruction goes through here.
    #
    # The audit's F01 is a write that landed inside an outstanding command's
    # read set: `_op_window_put` emitted `v.mov` straight into a fragment the
    # matrix unit was still reading. Fixing that one lowering would have left
    # every other write unchecked -- an ordinary `vf.add`, a load, a gather
    # and a reduction all write vector registers too -- so the check lives
    # where the instruction is emitted rather than in the lowerings.
    #
    # Which operand is a destination comes from the assembler's ISA table,
    # which is generated from the same schema as the decoder. A new
    # instruction therefore cannot be added without its destination being
    # known here, and a lowering cannot forget to ask.
    # ------------------------------------------------------------------
    _DEST_KINDS = {'vd': 'v', 'sd': 's', 'pd': 'p', 'ad': 'a', 'rdst': None}

    @staticmethod
    def _isa_table():
        if CodeGen.__dict__.get('_isa_cache') is None:
            import os as _os, sys as _sys
            root = _os.path.abspath(_os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                '..', '..', '..'))
            _sys.path.insert(0, _os.path.join(root, 'tools', 'asm'))
            import tnca_isa
            CodeGen._isa_cache = tnca_isa.ISA
        return CodeGen._isa_cache

    @classmethod
    def _destination_kind(cls, mnem):
        """The register class this mnemonic writes, or None."""
        table = cls._isa_table()
        best = None
        for key in table:
            if mnem == key or mnem.startswith(key + '.'):
                if best is None or len(key) > len(best):
                    best = key
        if best is None:
            return None
        kinds = table[best][2]
        if not kinds:
            return None
        return cls._DEST_KINDS.get(kinds[0], None)

    _REG_PREFIXES = ('v', 's', 'p', 'a')

    def _emit(self, mnem, *operands, **kw):
        kind = self._destination_kind(mnem)
        sources = operands
        if kind is not None and operands:
            dst = operands[0]
            if isinstance(dst, str) and dst[:1] == kind and dst[1:].isdigit():
                self.own.check_write(kind, int(dst[1:]),
                                     by=f"{mnem} {dst}")
            sources = operands[1:]
        # Reads are checked too, and for the mirror-image reason: a range an
        # engine is still WRITING -- a DMA's target window, an accumulator
        # mid-MMA -- holds no defined value yet, so reading it is the error
        # there. Checking only writes would catch the fragment case and miss
        # this one entirely.
        for src in sources:
            if isinstance(src, str) and src[:1] in self._REG_PREFIXES \
                    and src[1:].isdigit():
                self.own.check_read(src[0], int(src[1:]),
                                    by=f"{mnem} reading {src}")
        return self.b.emit(mnem, *operands, **kw)

    def _free_dead(self, position, ops):
        for op in ops[:position + 1]:
            if op.kind == 'window_reg':
                continue    # an alias, not an allocation
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
        if mask is not None and getattr(mask, 'kind', None) != PRED:
            raise Diagnostic(f"a {mask.kind} {mask!r} used as an execution mask; "
                             f"masks are predicate registers")
        return f"p{mask.reg}" if mask is not None else None

    # ---- the walk ---------------------------------------------------------
    def run(self, role_index=0):
        self._role_index = role_index
        role = self.kernel.role(role_index)
        ops = role.region.ops
        self._compute_last_uses(ops)
        if self.bank_banks:
            edges = {}
            for op in ops:
                reads = {id(v): v for v in op.operands
                         if getattr(v, 'kind', None) == VECTOR}
                for key, value in reads.items():
                    neighbors = edges.setdefault(key, {})
                    for other_key, other in reads.items():
                        if other_key != key:
                            old = neighbors.get(other_key, (other, 0))
                            neighbors[other_key] = (other, old[1] + 1)
            self._bank_reads = {key: tuple(values.values()) for key, values in edges.items()}
        for i, op in enumerate(ops):
            getattr(self, f"_op_{op.kind}", self._unhandled)(op)
            self._free_dead(i, ops)
        self._finish()
        return self.b

    def _awaited_elsewhere(self):
        """Does any OTHER role of this kernel wait on a completion event?"""
        if len(self.kernel.roles) < 2:
            return False
        for r in self.kernel.roles:
            if r is self.kernel.role(self._role_index):
                continue
            if any(op.kind == 'barrier_wait' for op in r.region.ops):
                return True
        return False

    def _unhandled(self, op):
        raise Diagnostic(
            f"no lowering for IR operation {op.kind!r}",
            remedy='add one, or raise earlier where the construct is still '
                   'recognisable',
            source='compiler_overview.md sec 11.1')

    def _finish(self):
        """Every asynchronous command must have been awaited (sec 10.2)."""
        # A command whose range is still held when the region ends was never
        # waited for. The arm/issue counts below catch a mismatched *count*;
        # this catches the case where the counts agree and the wait is simply
        # absent, which is the one that leaves the engine reading registers
        # the program has gone on to reuse.
        left = self.own.outstanding()
        if left and self._awaited_elsewhere():
            # A producer role legitimately ends with transfers outstanding:
            # in a two-role pipeline the CONSUMER waits on them. What must
            # never happen is a kernel where nobody waits at all, and that is
            # what this distinguishes. The cross-role obligation -- which role
            # waits, on which generation -- is the channel protocol's, and
            # check_protocol=True is where it is verified.
            left = {}
        if left:
            first = sorted(left)[0]
            desc, ranges = left[first]
            raise Diagnostic(
                f"{len(left)} asynchronous command(s) still outstanding when "
                f"the region ends; #{first} ({desc}) holds {', '.join(ranges)}",
                because='ISA sec 9C.3 and sec 13.4: the engine reads and '
                        'writes those registers after the issuing instruction '
                        'retires, so leaving without waiting hands them to '
                        'whatever runs next',
                remedy='wait on the command\'s completion event before the '
                       'region ends',
                source='compiler_overview.md sec 10.2')
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

    def _op_proton_timestamp(self, op):
        # Effectful: clocks must never be CSE'd or treated as PURE specials.
        v = op.results[0]
        self._alloc(v)
        self._emit('s.rdspecial', f's{v.reg}', 'clock')

    def _op_label(self, op):
        self.b.label(op.attrs['name'])

    def _op_const(self, op):
        v = op.results[0]
        self._alloc(v)
        self.b.directive(self._materialise(v.reg, op.attrs['value']))

    def _materialise(self, reg, value):
        """One scalar constant into a register.

        `.li` is four words whatever the value: movz and three movk. That
        is deliberate on the assembler's side -- a pseudo-op has to be
        exactly sized before it is expanded, or a branch computed over it
        lands in the wrong place. It is not the assembler's business to
        guess, so the compiler is what should stop asking for four words
        when it needs one.

        A value that fits in the low sixteen bits is exactly what movz
        writes: it sets those bits and zeroes the rest. Above O0 that is
        what we ask for. The GEMM result drain spends four words per row
        on a row number between 0 and 15, sixteen times.
        """
        v = int(value)
        if self.opt_level >= 1 and 0 <= v < (1 << 16):
            return f"s.movz       s{reg}, #{v}"
        return f".li       s{reg}, #{v}"

    def _op_program_id(self, op):
        v = op.results[0]
        self._alloc(v)
        self._emit('s.rdspecial', f"s{v.reg}",
                    f"pid.{'xyz'[op.attrs['dim']]}")

    def _op_num_programs(self, op):
        v = op.results[0]
        self._alloc(v)
        self._emit('s.rdspecial', f"s{v.reg}",
                   f"npid.{'xyz'[op.attrs['dim']]}")

    def _op_special(self, op):
        v = op.results[0]
        self._alloc(v)
        self._emit('s.rdspecial', f"s{v.reg}", op.attrs['name'])

    def _op_sbin(self, op):
        a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        self._emit(_SBIN[op.attrs['op']], f"s{v.reg}", f"s{a.reg}",
                    f"s{b.reg}", comment=op.attrs.get('comment'))

    def _op_sbini(self, op):
        a = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        self._emit(_SBINI[op.attrs['op']], f"s{v.reg}", f"s{a.reg}",
                    f"#{op.attrs['imm']}", comment=op.attrs.get('comment'))

    def _op_scmp(self, op):
        a, b = op.operands
        v = op.results[0]
        self._alloc(v)
        self._emit(f"s.cmp.{op.attrs['cc']}", f"s{v.reg}", f"s{a.reg}",
                    f"s{b.reg}")

    def _op_assign(self, op):
        dst, src = op.operands
        bank = {VECTOR: 'v', PRED: 'p'}.get(dst.kind, 's')
        if src.kind != dst.kind:
            raise Diagnostic(f"assign of a {src.kind} into a {dst.kind} {dst!r}")
        self._emit(f'{bank}.mov', f'{bank}{dst.reg}', f'{bank}{src.reg}')

    # ---- vector ops -------------------------------------------------------
    def _op_arange(self, op):
        v = op.results[0]
        self._alloc(v)
        self._emit('v.iota', f"v{v.reg}", comment='arange(0, VLEN)')

    def _op_splat(self, op):
        s = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        self._emit('v.splat', f"v{v.reg}", f"s{s.reg}")

    def _op_vmovi(self, op):
        """A constant into every lane.

        ISA 0x40: `v.movi Vd, #imm16` sign-extends sixteen bits, which is
        not enough for a 32-bit value -- and every floating-point literal
        is one. `x * alpha + 1.0` produced `v.movi v0, #1065353216`, the
        bit pattern of 1.0f, and the assembler refused it three layers
        below the kernel that asked.

        A value that does not fit is materialised in a scalar register
        and broadcast. That costs an instruction or two more than the
        immediate form and is what the ISA provides; inventing a wider
        immediate field would not make one exist.
        """
        v = op.results[0]
        self._alloc(v)
        self._broadcast_imm(v.reg, int(op.attrs['imm']))

    def _broadcast_imm(self, vreg, imm, comment=None):
        """A 32-bit constant into every lane of `vreg`.

        Every caller that puts a literal in a vector register goes
        through here. The load and the gather used to emit their `other=`
        `v.movi` themselves, so a masked load of a float -- `other=-inf`,
        which is what every masked `tl.max` uses -- reached the assembler
        as `v.movi #4286578688` and was refused, while the identical
        constant arriving through `vmovi` was materialised correctly.
        """
        if -(1 << 15) <= imm < (1 << 15):
            self._emit('v.movi', f"v{vreg}", f"#{imm}", comment=comment)
            return
        s = self.alloc.scalar()
        self.b.directive(self._materialise(s, imm))
        self._emit('v.splat', f"v{vreg}", f"s{s}",
                    comment=comment or 'a 32-bit constant does not fit '
                                       'v.movi (ISA 0x40)')
        self.alloc.free('s', s)

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
        # ISA sec 8.3: `.ieee` ignores a NaN operand, `.prop` propagates
        # it. The two spellings are two different functions, so the
        # suffix is written out rather than left to the assembler's
        # default whenever the frontend knew which one it meant.
        fmm = op.attrs.get('fmm')
        if fmm is not None:
            if mnem not in ('vf.min', 'vf.max'):
                raise Diagnostic(
                    f"a min/max NaN mode on {op.attrs['op']!r}, which has "
                    f"none", source='doc/isa/TNCA-ISA-v0.1.md sec 8.3')
            mnem = f"{mnem}.{fmm}"
        self._emit(mnem, f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    comment=op.attrs.get('comment'))

    def _op_vun(self, op):
        a = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        sfu = _VUN_SFU.get(op.attrs['op'])
        if sfu is not None:
            self.target.require('sfu', f"emit {op.attrs['op']}")
            self._emit(sfu, f"v{v.reg}", f"v{a.reg}",
                        comment=op.attrs.get('comment'))
            return
        if not _is_fp(v.dtype):
            raise Diagnostic(
                f"no {v.dtype} lowering for {op.attrs['op']!r}: the ISA "
                f"has no integer {op.attrs['op']}",
                source='doc/isa/opcode-map.md')
        self.target.require('vector_fp32',
                            f"emit a {v.dtype} {op.attrs['op']}")
        mnem = _VUN_FP.get(op.attrs['op'])
        if mnem is None:
            raise Diagnostic(f"no {v.dtype} lowering for {op.attrs['op']!r}",
                             source='compiler_overview.md sec 12')
        self._emit(mnem, f"v{v.reg}", f"v{a.reg}",
                    comment=op.attrs.get('comment'))

    def _op_vfma(self, op):
        a, b, c = op.operands
        v = op.results[0]
        self._alloc(v)
        self.target.require('vector_fp32', f"emit a {v.dtype} fma")
        self._emit('vf.fma', f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    f"v{c.reg}", comment=op.attrs.get('comment'))

    def _op_vfms(self, op):
        a, b, c = op.operands
        v = op.results[0]
        self._alloc(v)
        self.target.require('vector_fp32', f"emit a {v.dtype} fms")
        self._emit('vf.fms', f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    f"v{c.reg}", comment=op.attrs.get('comment'))

    def _op_vcvt(self, op):
        a = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        src, dst = op.attrs['src'], op.attrs['dst']
        aliases = {'f8E4M3FN': 'e4m3', 'f8E5M2': 'e5m2'}
        src, dst = aliases.get(src, src), aliases.get(dst, dst)
        # sec 8.4 gives int<->float their own opcodes and float->float a
        # shared one; the assembler spells the first two out.
        self._emit(f"vf.cvt.{src}.{dst}", f"v{v.reg}", f"v{a.reg}",
                    comment=op.attrs.get('comment'))

    def _op_vextract(self, op):
        v = op.operands[0]
        out = op.results[0]
        self._alloc(out)
        self._emit('v.extract', f"s{out.reg}", f"v{v.reg}",
                    f"#{op.attrs['lane']}")

    def _op_vinsert(self, op):
        v, scalar = op.operands
        self._emit('v.insert', f"v{v.reg}", f"s{scalar.reg}",
                    f"#{op.attrs['lane']}")

    def _op_vshuffle(self, op):
        v, amount = op.operands
        out = op.results[0]
        self._alloc(out)
        self._emit(f"v.shuffle.{op.attrs['op']}", f"v{out.reg}",
                    f"v{v.reg}", f"s{amount.reg}")

    def _op_vcmp(self, op):
        a = op.operands[0]
        p = op.results[0]
        self._alloc(p)
        mnem = 'vf.cmp' if _is_fp(a.dtype) else 'v.cmp'
        # ISA sec 0x38: `v.cmpi Pd, Va, #imm16` compares against a signed
        # 16-bit immediate. When the optimiser has proved the right side
        # is a broadcast constant it leaves the value in `imm`, and the
        # splat and the constant that fed it are then dead.
        if 'imm' in op.attrs:
            self._emit(f"{mnem}i.{op.attrs['cc']}", f"p{p.reg}",
                        f"v{a.reg}", f"#{op.attrs['imm']}")
            return
        b = op.operands[1]
        self._emit(f"{mnem}.{op.attrs['cc']}", f"p{p.reg}", f"v{a.reg}",
                    f"v{b.reg}")

    def _op_pbin(self, op):
        a, b = op.operands
        p = op.results[0]
        self._alloc(p)
        mnem = _PBIN.get(op.attrs['op'])
        if mnem is None:
            raise Diagnostic(f"no predicate lowering for {op.attrs['op']!r}",
                             source='doc/isa/opcode-map.md sec 0x13')
        self._check_pred_operands(op, a, b)
        self._emit(mnem, f"p{p.reg}", f"p{a.reg}", f"p{b.reg}")

    @staticmethod
    def _check_pred_operands(op, *values):
        """A predicate instruction names predicate registers. A vector
        operand here would be printed as the predicate register with the
        same number -- whatever that happens to hold -- which is a wrong
        answer that only an allocation change reveals (R39-01 found
        `xmask ^ true` reading `true` that way)."""
        for v in values:
            if getattr(v, 'kind', None) != PRED:
                raise Diagnostic(f"{op.kind} {op.attrs.get('op')!r} on a {v.kind} operand "
                                 f"{v!r}; it takes predicates")

    def _op_pun(self, op):
        a = op.operands[0]
        p = op.results[0]
        self._alloc(p)
        self._check_pred_operands(op, a)
        self._emit(_PUN[op.attrs['op']], f"p{p.reg}", f"p{a.reg}")

    def _op_pred_to_scalar(self, op):
        value, = op.operands
        result, = op.results
        self._check_pred_operands(op, value)
        self._alloc(result)
        self._emit('p.ballot', f's{result.reg}', f'p{value.reg}')

    def _op_pred_from_scalar(self, op):
        value, = op.operands
        result, = op.results
        self._alloc(result)
        self._emit('p.frombits', f'p{result.reg}', f's{value.reg}')

    def _op_vselect(self, op):
        p, a, b = op.operands
        v = op.results[0]
        self._check_pred_operands(op, p)
        self._alloc(v)
        self._emit('v.sel', f"v{v.reg}", f"v{a.reg}", f"v{b.reg}",
                    pred=f"p{p.reg}")

    # ---- memory -----------------------------------------------------------
    def _op_load(self, op):
        base = op.operands[0]
        v = op.results[0]
        mask = op.attrs.get('mask')
        self._alloc(v)
        if mask is not None and not op.attrs.get('no_other'):
            # sec 12: `other=` is a value. A predicated load leaves inactive
            # lanes holding what the register had, so put the value there.
            self._broadcast_imm(v.reg, int(op.attrs['other']),
                                comment='other=, materialised (sec 12)')
        self._emit(f"v.ld.{op.attrs['size']}", f"v{v.reg}",
                    f"[s{base.reg}, #0]", pred=self._pred(mask))

    def _op_store(self, op):
        base, value = op.operands
        self._emit(f"v.st.{op.attrs['size']}", f"[s{base.reg}, #0]",
                    f"v{value.reg}", pred=self._pred(op.attrs.get('mask')))

    def _op_view(self, op):
        """A layout view that has to be realised, as instructions.

        `v.permute Vd, Va, Vb` (ISA 0x03 of the lane group) reads
        `Vd[i] = Va[Vb[i]]`, so this needs Vb: lane i holding the
        position the layout sends i to.

        A LinearLayout is linear over GF(2), so that vector is the XOR of
        the bases the set bits of the lane index select. Per bit: take
        it, spread it to a full mask, AND the mask with that basis, XOR
        it in. The basis values come from upstream, computed by
        `index_vector_recipe` -- this only emits them.

        Twenty-four instructions for a sixteen-lane permutation, and
        every one of them is on the page. Section 3.2 asks for a
        conversion's real cost to be checkable against the emitted code,
        and this is what that costs when nothing proves it unnecessary.
        """
        src = op.operands[0]
        out = op.results[0]
        basis = op.attrs.get('perm_basis')
        if basis is None:
            raise Diagnostic(
                "a layout view reached code generation without a plan",
                because='the layout pass classifies every view as a '
                        'rename, a real conversion or a refusal, and '
                        'only the middle one is emitted here',
                remedy='run the layout pass, or compile at O0 where '
                       'views are not accepted',
                source='phase3_5_tasks_and_acceptance.md sec 3.2')
        self._alloc(out)
        idx = self.alloc.vector()
        zero = self.alloc.vector()
        bit = self.alloc.vector()
        self._emit('v.iota', f"v{bit}",
                    comment=f"materialise {op.attrs.get('what') or 'a view'}")
        self._emit('v.movi', f"v{idx}", '#0')
        self._emit('v.movi', f"v{zero}", '#0')
        lane = self.alloc.vector()
        for k, b in enumerate(basis):
            if b == 0:
                continue
            self._emit('v.lshri', f"v{lane}", f"v{bit}", f"#{k}")
            self._emit('v.andi', f"v{lane}", f"v{lane}", '#1')
            self._emit('v.sub', f"v{lane}", f"v{zero}", f"v{lane}",
                        comment='0 or all ones')
            self._emit('v.andi', f"v{lane}", f"v{lane}", f"#{b}")
            self._emit('v.xor', f"v{idx}", f"v{idx}", f"v{lane}")
        self._emit('v.permute', f"v{out.reg}", f"v{src.reg}", f"v{idx}")
        for r in (idx, zero, bit, lane):
            self.alloc.free('v', r)

    def _op_gather(self, op):
        base, index = op.operands
        v = op.results[0]
        mask = op.attrs.get('mask')
        self.target.require('gather_scatter', 'emit v.ld.gather')
        self._alloc(v)
        if mask is not None:
            self._broadcast_imm(v.reg, int(op.attrs['other']),
                                comment='other=, materialised (sec 12)')
        # ISA sec 9.1 scales the index by the access size: these are element
        # indices, not byte offsets.
        self._emit(f"v.ld.gather.{op.attrs['size']}", f"v{v.reg}",
                    f"[s{base.reg}, v{index.reg}]", pred=self._pred(mask),
                    comment='index is in elements (ISA sec 9.1)')

    def _op_scatter(self, op):
        base, index, value = op.operands
        self.target.require('gather_scatter', 'emit v.st.scatter')
        self._emit(f"v.st.scatter.{op.attrs['size']}",
                    f"[s{base.reg}, v{index.reg}]", f"v{value.reg}",
                    pred=self._pred(op.attrs.get('mask')))

    def _op_sload(self, op):
        base = op.operands[0]
        v = op.results[0]
        self._alloc(v)
        self._emit(f"s.ld.{op.attrs['size']}", f"s{v.reg}",
                    f"[s{base.reg}, #{op.attrs['offset']}]",
                    comment=op.attrs.get('comment'))

    def _op_sstore(self, op):
        base, value = op.operands
        self._emit(f"s.st.{op.attrs['size']}",
                    f"[s{base.reg}, #{op.attrs['offset']}]", f"s{value.reg}",
                    comment=op.attrs.get('comment'))

    def _op_trap(self, op):
        self._emit('trap', f"#{op.attrs['code']}")

    def _op_fence(self, op):
        # sec 9.2: `fence` takes no access size -- mod[2:0] is the size
        # field and a fence has no access, so the decoder refuses a
        # non-zero one rather than ignoring it.
        self.target.require('fence', 'emit a fence')
        out = 'fence'
        if op.attrs.get('order', 'acq_rel') != 'relaxed':
            out += f".{op.attrs['order']}"
        if op.attrs.get('scope', 'device') != 'core':
            out += f".{op.attrs['scope']}"
        self._emit(out)

    def _op_atomic(self, op):
        addr, value = op.operands
        out = op.results[0]
        self._alloc(out)
        self.target.require(self._atomic_capability(op), 'emit an atomic')
        mnem = _ATOMIC.get(op.attrs['op'])
        if mnem is None:
            raise Diagnostic(
                f"no atomic {op.attrs['op']!r}",
                because='ISA sec 9.2 allocates add, and, or, xor, xchg, min '
                        'and max, and a compare-and-swap for the rest',
                source='doc/isa/TNCA-ISA-v0.1.md sec 9.2')
        self._emit(f"{mnem}{self._atomic_suffix(op)}", f"s{out.reg}",
                    f"[s{addr.reg}]", f"s{value.reg}")

    def _op_atomic_cas(self, op):
        addr, expect, desired = op.operands
        out = op.results[0]
        self._alloc(out)
        self.target.require(self._atomic_capability(op), 'emit an atomic')
        self._emit(f"atom.cas{self._atomic_suffix(op)}", f"s{out.reg}",
                    f"[s{addr.reg}]", f"s{expect.reg}", f"s{desired.reg}")

    @staticmethod
    def _atomic_capability(op):
        # sec 9.2: system scope is only legal when SYSTEM_ATOMICS is set,
        # and raises ILLEGAL_INSTRUCTION otherwise rather than silently
        # degrading to device scope.
        return ('atomics_system' if op.attrs.get('scope') == 'system'
                else 'atomics_device')

    @staticmethod
    def _atomic_suffix(op):
        out = f".{op.attrs.get('size', 'w')}"
        order = op.attrs.get('order', 'relaxed')
        scope = op.attrs.get('scope', 'device')
        if order != 'relaxed':
            out += f".{order}"
        if scope != 'core':
            out += f".{scope}"
        return out

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
        if op.attrs.get('broadcast'):
            self._emit(f"{mnem}.b", f"v{s.reg}", f"v{v.reg}",
                        pred=self._pred(op.attrs.get('mask')))
            return
        self._emit(mnem, f"s{s.reg}", f"v{v.reg}",
                    pred=self._pred(op.attrs.get('mask')))

    # ---- control ----------------------------------------------------------
    def _op_branch_nz(self, op):
        c = op.operands[0]
        self._emit('br.nz', f"s{c.reg}", op.attrs['target'])

    def _op_exit(self, op):
        self._emit('exit')

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
    def _window_id(self, w):
        """The window's id: its index, or the slot resource_lifetime gave it
        when the program names more windows than the target has."""
        wid = getattr(w, 'wid', None)
        return self.kernel.windows.index(w) if wid is None else wid

    def _op_window_def(self, op):
        w = op.results[0]
        if w.reg is None:
            raise Diagnostic(
                f"window {w.name} has no register range",
                because='whole-program resources are fixed before the walk '
                        '(sec 10.1); call assign_shared() first')
        self._emit('w.def', f"w{self._window_id(w)}",
                    f"#{w.reg}", f"#{w.length}",
                    comment=f"v{w.reg}..v{w.reg + w.length - 1}")

    def _op_window_reg(self, op):
        window = op.operands[0]
        v = op.results[0]
        if window.reg is None:
            raise Diagnostic(f"window {window.name} has no register range")
        # Aliases a window register rather than allocating: the whole point is
        # that no copy happens (sec 7.3).
        v.reg = window.reg + op.attrs['offset']

    def _op_window_put(self, op):
        window, value = op.operands
        if window.reg is None:
            raise Diagnostic(f"window {window.name} has no register range")
        dst = window.reg + op.attrs['offset']
        self._emit('v.mov', f"v{dst}", f"v{value.reg}",
                    comment='hand off through the shared window (sec 7.3)')

    def _op_desc_snapshot(self, op):
        addr = op.operands[0]
        d = op.results[0]
        self.target.require('tensor_dma', 'snapshot a TensorMap')
        self._emit('td.load', f"d{d.reg}", f"[s{addr.reg}]",
                    comment='a versioned snapshot, not a pointer '
                            '(ISA sec 9D.1)')

    def _op_barrier_init(self, op):
        b = op.operands[0]
        self.target.require('mbarrier', 'initialise an mbarrier')
        self._emit('mb.init', f"b{b.reg}")
        if self.protocol:
            self.protocol.barrier_init(b.reg)

    def _op_barrier_arm(self, op):
        bar = op.operands[0]
        tok = op.results[0]
        self._alloc(tok)
        arrivals, ops_n = op.attrs['arrivals'], op.attrs['ops']
        self._emit('mb.arm', f"s{tok.reg}", f"b{bar.reg}",
                    f"#{arrivals}", f"#{ops_n}",
                    comment=f"{arrivals} arrival(s), {ops_n} operation(s)")
        self._token_barrier[id(tok)] = bar.reg
        self._armed_ops[bar.reg] = ops_n
        self._issued_ops[bar.reg] = 0
        if self.protocol:
            self.protocol.barrier_arm(bar.reg, arrivals=arrivals, ops=ops_n)

    def _op_barrier_wait(self, op):
        tok = op.operands[0]
        self._emit('mb.wait', f"s{tok.reg}")
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
        # Every command counted on this phase, protocol model or not. The
        # release used to walk the protocol's leases only, so with
        # check_protocol=False a transfer's window was held forever and the
        # region could never end cleanly -- the debug switch changed what the
        # safety checks concluded, which is exactly what it must not do.
        for cmd in self._mma_pending.pop(bar, []):
            self.own.release(cmd)
        # R39-12: a producer waiting for the consumer to release a ring stage
        # knows the transfers it filled the stage with are complete -- the
        # consumer waited for them before it read the windows -- so the
        # windows are the producer's to fill again.
        for other in op.attrs.get('retires', ()):
            for cmd in self._mma_pending.pop(other.reg, []):
                self.own.release(cmd)

    def _op_barrier_token(self, op):
        bar = op.operands[0]
        tok = op.results[0]
        self._alloc(tok)
        epoch = op.attrs['epoch']
        # ISA sec 9A.2: token = {epoch[47:0], barrier_id[15:0]}.
        value = (epoch << 16) | bar.reg
        self.b.directive(
            f".li       s{tok.reg}, #{value}"
            f"          // b{bar.reg} generation {epoch} (ISA sec 9A.2)")
        self._token_barrier[id(tok)] = bar.reg

    def _op_barrier_token_dyn(self, op):
        bar, epoch = op.operands
        tok = op.results[0]
        self._alloc(tok)
        # token = {epoch[47:0], barrier_id[15:0]} (ISA sec 9A.2).
        self._emit('s.shli', f"s{tok.reg}", f"s{epoch.reg}", '#16',
                    comment='epoch into the token (ISA sec 9A.2)')
        self._emit('s.ori', f"s{tok.reg}", f"s{tok.reg}", f"#{bar.reg}",
                    comment=f"barrier b{bar.reg}")
        self._token_barrier[id(tok)] = bar.reg

    def _op_barrier_arrive(self, op):
        bar = op.operands[0]
        self._emit('mb.arrive', f"b{bar.reg}")

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
        self._lower_transfer(op, store=False)

    def _op_dma_store(self, op):
        self._lower_transfer(op, store=True)

    def _lower_transfer(self, op, *, store):
        window, desc, token = op.operands[0], op.operands[1], op.operands[2]
        coords = op.operands[3:]
        mnem = 't.st.async' if store else 't.ld.async'
        self.target.require('tensor_dma', f'emit {mnem}')
        if store:
            # ISA sec 9E func 0x08. The engine reads the window through its
            # own VRF port; a target without it still refuses here rather
            # than trapping at run time.
            self.target.require('tensor_store', 'emit t.st.async')
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
            self._emit('s.mov', f"s{reg}", f"s{c.reg}",
                        comment='coordinates must be consecutive (ISA sec 9E)')

        bar = self._token_barrier[id(token)]
        self._count_operation(bar)
        suffix = {'f16': 'h', 'bf16': 'h', 'i16': 'h', 'f32': 'w', 'i32': 'w',
                  'i8': 'b', 'f8E4M3FN': 'b', 'f8E5M2': 'b'}[op.attrs['dtype']]
        widx = self._window_id(window)
        # ISA sec 9A.3: a fully predicated-off asynchronous command still
        # completes once, so the static operation count stays correct under
        # predication. The count above is therefore unconditional.
        mask = op.attrs.get('mask')
        if op.attrs.get('transpose'):
            # sec 9E mod[5]. The engine executes it only where the core
            # was built with HAS_TENSOR_XPOSE (R38 S1-07), and anywhere
            # else raises ILLEGAL_INSTRUCTION. So asking for it on a
            # target without the capability is a compile error, which is
            # the one outcome that is neither a wrong answer nor a
            # runtime trap.
            self.target.require('tensor_transpose',
                                'emit a transposed t.ld.async')
            suffix += '.t'
        self._emit(f"{mnem}.{suffix}", f"w{widx}", f"d{desc.reg}",
                    f"s{run[0]}", f"s{token.reg}",
                    pred=self._pred(mask),
                    comment=('predicated off still completes once '
                             '(ISA sec 9A.3)') if mask is not None else None)

        # sec 13.4: the engine reads the coordinate registers and writes the
        # whole window, neither of which the encoding names in full.
        #
        # Taken unconditionally. This used to sit inside `if self.protocol`,
        # so `check_protocol=False` -- a debugging convenience -- silently
        # turned off emission safety as well as the channel model. The
        # contract forbids that: a supported compilation path must run the
        # unbypassable checks whatever the debug switches say.
        if self.protocol:
            from .protocol import BufferState
            if self.protocol.buffers[widx].state is BufferState.FREE:
                self.protocol.producer_acquire(widx, 'producer')
            cmd = self.protocol.issue_async(widx, 'producer', bar, kind='dma')
        else:
            cmd = f"dma{len(self._events)}"
        self._events[id(op.results[0])] = cmd

        # THE COORDINATE CONTRACT, checked against the RTL rather than assumed.
        #
        # The contract (R1-06) requires this to be settled one way or the
        # other: if the engine snapshots the coordinates at issue, prove it
        # and release them; if it reads them later, hold them until the
        # completion event. It snapshots.
        #
        #   rtl/core/tnca_core.v reads Sb and Sb+1 in the execute stage and
        #   passes them as cmd_c0/cmd_c1;
        #   rtl/core/tnca_tensor.v:357 latches `t_c0 <= cmd_c0` on cmd_valid,
        #   and every later use -- the row base at :448, the bounds at :232
        #   and :528 -- reads t_c0/t_c1, never the register file again.
        #
        # So the scalar run is dead the cycle after issue, and holding it
        # until completion would be a false dependency: it made `mb.arm`
        # writing an adjacent scalar a compile error in a program that is
        # correct. What must NOT be released early is the window, below.
        # For a load the window is what the engine WRITES, so reading it
        # before the transfer completes is the error -- not writing it.
        #
        # Direction decides the mode, and the mode decides which program is
        # wrong. A load WRITES the window, so reading it early is the error;
        # a store READS it -- late, after this instruction has retired
        # (sec 13.4's third implicit footprint) -- so OVERWRITING it early is
        # the error, and reading it meanwhile is perfectly legal.
        self.own.hold(cmd, [RegRange('v', window.reg, window.length)],
                      mnem, mode='read' if store else 'write')
        self._mma_pending.setdefault(bar, []).append(cmd)

        # Dead once the command has them; see the coordinate contract above.
        for reg in run:
            self.alloc.free('s', reg)

    def _op_acc_zero(self, op):
        a = op.operands[0]
        self._emit('m.zero', f"a{a.reg}")

    def _op_mma_async(self, op):
        acc, aw, bw, token = op.operands
        from .gemm import _DTYPE_CAP
        dtype = op.attrs['dtype']
        if dtype not in _DTYPE_CAP:
            raise Diagnostic(
                f"a matrix multiply with {dtype} inputs",
                because="ISA sec 9C's mod[3:2] names FP16, BF16 and the two "
                        "FP8 encodings as the input formats, and reserves "
                        "the rest; there is no binary32 MMA",
                remedy='convert the operands, which is what tl.dot on a '
                       'narrower type does anyway',
                source='doc/isa/TNCA-ISA-v0.1.md sec 9C')
        self.target.require(_DTYPE_CAP[dtype],
                            f"emit m.mma.async with {dtype} inputs")
        bar = self._token_barrier[id(token)]
        self._count_operation(bar)
        self._emit(f"m.mma.async.{op.attrs['k']}.{op.attrs['dtype']}",
                    f"a{acc.reg}", f"v{aw.reg}", f"v{bw.reg}",
                    f"s{token.reg}")
        # sec 13.4 / ISA sec 9C.3: the matrix unit reads both fragments in
        # full, for many cycles after this instruction retires. Hold the whole
        # range rather than the one register the encoding names.
        # From the instruction, not from the target's default shape. This is
        # the audit's F01: the emit above already says K64 and e4m3, and
        # sizing the footprint from target.mma_shape sized it for K16 FP16.
        tm, tn = self.target.mma_shape[0], self.target.mma_shape[1]
        fm, fn, _k, bits = mma_shape_of(op.attrs, m=tm, n=tn)
        m, n, k = fm, fn, _k
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
        # ISA sec 9C.3: the destination accumulator may not be read by the
        # vector unit until the command completes. Holding it as a *written*
        # range is what makes a.vread before the wait a compile error rather
        # than a race whose symptom is a plausible wrong number.
        self.own.hold(cmd, [RegRange('a', acc.reg, 1)], 'm.mma.async',
                      mode='write')
        self._events[id(op.results[0])] = cmd
        self._mma_pending.setdefault(bar, []).append(cmd)

    def _op_acc_write(self, op):
        acc, row, value = op.operands
        self.own.check_write('a', acc.reg, by=f"a.vwrite a{acc.reg}")
        self._emit('a.vwrite', f"a{acc.reg}", f"s{row.reg}",
                    f"v{value.reg}")

    def _op_acc_read(self, op):
        acc, row = op.operands
        self.own.check_read('a', acc.reg, by=f"a.vread a{acc.reg}")
        v = op.results[0]
        self._alloc(v)
        self._emit('a.vread', f"v{v.reg}", f"a{acc.reg}", f"s{row.reg}")


def emit_roles(kernel, target, *, check_protocol=False, opt_level=0, bank_alloc=False,
               window_reuse=False):
    """A kernel of several roles as one program (R39-12, Mode A).

    Every context starts at the same PC (a launch has one); each reads
    ROLE_ID and branches to its role's code, role = ROLE_ID mod the number
    of roles -- a Role Context pair (RPAIR) is contexts 2k and 2k+1, so the
    even one runs role 0 and the odd one role 1. Labels are the roles' own
    already (tnca_cc.partition renames them).

    A pair shares its vector registers, so the roles' temporaries must not
    meet: the windows are at the top of the file (assign_shared), role 0's
    temporaries just below them, and role 1's from v0 up. Role 0 is
    compiled once to measure how many it takes. Scalars and predicates are
    per context and need no such care.
    """
    import copy
    if window_reuse:
        raise ResourceError('window lifetime reuse requires a single IR role')
    if len(kernel.roles) != 2:
        raise Diagnostic(f"{len(kernel.roles)} roles; this emits two")
    trial = copy.deepcopy(kernel)
    cg = CodeGen(trial, target, check_protocol=check_protocol, opt_level=opt_level,
                 bank_alloc=bank_alloc)
    cg.assign_shared()
    cg.run(0)
    need = cg.alloc.max_vector + 1
    top = min([w.reg for w in trial.windows], default=target.vregs_per_rctx)
    if need > top:
        raise ResourceError(f"role 0 needs {need} vector registers below the windows at v{top}")
    texts, gens = [], []
    for r, (lo, hi) in ((0, (top - need, top)), (1, (0, top - need))):
        g = CodeGen(kernel, target, check_protocol=check_protocol, opt_level=opt_level,
                    bank_alloc=bank_alloc)
        g.assign_shared()
        g.alloc.limit_vectors(lo, hi)
        g.run(r)
        texts.append(g.b.text())
        gens.append(g)
        for v in _values_of(kernel.role(r)):
            if v.kind in (VECTOR, SCALAR, PRED, EVENT, TOKEN):
                v.reg = None
    head = AsmBuilder(target, kernel.name)
    head.comment(f"R39-12: {kernel.roles[0].name} on even contexts, "
                 f"{kernel.roles[1].name} on odd ones (ROLE_ID mod 2)")
    head.emit('s.rdspecial', 's61', 'role_id')
    head.emit('s.andi', 's61', 's61', '#1')
    head.emit('br.nz', 's61', '__tnca_role1')
    text = head.text() + texts[0] + '__tnca_role1:\n' + texts[1]
    return text, gens[1]


def _values_of(role):
    for op in role.region.ops:
        yield from op.results
