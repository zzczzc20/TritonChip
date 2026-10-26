"""
TNCA assembly emission.

compiler_overview.md sec 11.1 makes this the bring-up path:

    TNCA low-level IR -> conservative scheduling and restricted static
    allocation -> TNCA assembly -> tools/asm/tnca_asm.py -> RTL

and is specific about what not to build at this stage: 这一阶段不建设复杂的
通用机器优化器. So there is no scheduler here, and the allocator is a plain
linear scan. What there *is*, is a refusal to paper over a shortage --
资源不够可以先明确报错，不能为了快速"支持"而偷偷破坏异步生命周期.

The mnemonics come from the ISA schema, so an instruction that does not exist
fails here rather than at the assembler, with the source construct still in
hand.
"""
from .diagnostics import Diagnostic, ResourceError
from .isa import ISA

# ISA sec 3: S0 reads as zero and discards writes, so it is never allocated.
FIRST_SREG = 1
# The assembler's .assert_eq / .assert_lane pseudo-ops use these as scratch.
ASM_SCRATCH_SREGS = {61, 62, 63}
FIRST_PREG = 1


class RegisterAllocator:
    """A linear-scan allocator over one Role Context's registers.

    sec 10.1 splits allocation in two: whole-program resources (barriers,
    accumulators, RF windows, descriptors) are fixed first, across all roles,
    and only then are role-local S/V/P temporaries assigned. This handles the
    second half; the first is the planner's job, and `reserve_vector` exists
    so the planner's decisions can be pinned here before anything else moves.
    """

    def __init__(self, target):
        self.target = target
        self._s_free = [i for i in range(FIRST_SREG, target.sregs_per_rctx)
                        if i not in ASM_SCRATCH_SREGS]
        self._v_free = list(range(target.vregs_per_rctx))
        self._p_free = list(range(FIRST_PREG, target.pregs_per_rctx))
        self._reserved_v = set()
        self.peak = {'s': 0, 'v': 0, 'p': 0}
        self.max_vector = -1          # the highest vector register handed out
        self._live = {'s': set(), 'v': set(), 'p': set()}

    # ---- allocation -------------------------------------------------------
    def _take(self, kind, pool, name):
        if not pool:
            raise ResourceError(
                f"out of {name} registers while compiling",
                because=(f"the target gives one Role Context "
                         f"{self._capacity(kind)} of them, and there is no "
                         "scratchpad to spill into"),
                remedy='a smaller tile, or fewer live values',
                source='compiler_overview.md sec 10.1, 11.1')
        idx = pool.pop(0)
        if kind == 'v':
            self.max_vector = max(self.max_vector, idx)
        self._live[kind].add(idx)
        self.peak[kind] = max(self.peak[kind], len(self._live[kind]))
        return idx

    def _capacity(self, kind):
        return {'s': self.target.sregs_per_rctx,
                'v': self.target.vregs_per_rctx,
                'p': self.target.pregs_per_rctx}[kind]

    def scalar(self):
        return self._take('s', self._s_free, 'scalar')

    def vector(self, cost=None, banks=0):
        if cost is not None and self._v_free:
            # Bound the search to the first bank-sized group of free names.
            # Windows and role slices have already been removed from this pool.
            if banks < 1:
                raise Diagnostic('bank allocation requires a positive bank count')
            candidate = min(self._v_free[:banks], key=lambda i: (cost(i), i))
            self._v_free.remove(candidate)
            self._v_free.insert(0, candidate)
        return self._take('v', self._v_free, 'vector')

    def predicate(self):
        return self._take('p', self._p_free, 'predicate')

    def scalar_run(self, count):
        """A run of `count` consecutive scalar registers.

        ISA sec 9E takes a transfer's coordinates from consecutive scalar
        registers starting at Sb, and the instruction names only the first.
        A scan that hands out whatever is free cannot satisfy that, so the run
        is found explicitly -- and s63 is excluded as a base, because the next
        index would be s0, which reads as zero and discards writes, so the
        second coordinate could never be set.
        """
        if count < 1:
            raise Diagnostic('a coordinate run needs at least one register')
        free = set(self._s_free)
        for first in sorted(free):
            if first + count - 1 >= self.target.sregs_per_rctx:
                break
            run = list(range(first, first + count))
            if all(i in free for i in run):
                for i in run:
                    self._s_free.remove(i)
                self._live['s'].update(run)
                self.peak['s'] = max(self.peak['s'], len(self._live['s']))
                return run
        raise ResourceError(
            f"no run of {count} consecutive scalar registers is free",
            because='a transfer reads its coordinates from consecutive '
                    'registers (ISA sec 9E), so they cannot be scattered',
            remedy='fewer live scalars across the transfer',
            source='compiler_overview.md sec 10.1, 13.4')

    def limit_vectors(self, lo, hi):
        """Hand out only v[lo, hi): two roles of a Role Context pair share
        the file (R39-12), and each keeps to its own range."""
        self._v_free = [i for i in self._v_free if lo <= i < hi]

    def reserve_vector(self, first, count):
        """Pin a vector range the planner has already assigned.

        An RF window is a contiguous range (ISA sec 9D.2), so it cannot be
        assembled out of whatever the scan happens to have free.
        """
        wanted = list(range(first, first + count))
        missing = [i for i in wanted if i not in self._v_free]
        if missing:
            raise ResourceError(
                f"cannot reserve v{first}..v{first + count - 1}: "
                f"{missing} already allocated",
                because='a window must be a contiguous register range',
                remedy='reserve whole-program resources before role-local '
                       'temporaries (sec 10.1)',
                source='compiler_overview.md sec 10.1')
        for i in wanted:
            self._v_free.remove(i)
            self._reserved_v.add(i)
        self._live['v'].update(wanted)
        self.peak['v'] = max(self.peak['v'], len(self._live['v']))
        return first

    # ---- release ----------------------------------------------------------
    def free(self, kind, idx):
        if idx in self._reserved_v and kind == 'v':
            raise Diagnostic(
                f"v{idx} is part of a reserved window and is not a temporary")
        self._live[kind].discard(idx)
        pool = {'s': self._s_free, 'v': self._v_free, 'p': self._p_free}[kind]
        if idx not in pool:
            pool.append(idx)
            pool.sort()

    @property
    def private_vector_bytes(self):
        return self.peak['v'] * self.target.vlen * 4


class AsmBuilder:
    """Accumulate TNCA assembly lines.

    Checks each mnemonic against the ISA schema as it goes, so a lowering
    that invents an instruction fails with the construct still in hand rather
    than as an assembler error on a generated file.
    """

    def __init__(self, target, name='kernel'):
        self.target = target
        self.name = name
        self._lines = []
        self._labels = set()
        self.instruction_count = 0

    # ---- raw output -------------------------------------------------------
    def comment(self, text):
        for line in text.split('\n'):
            self._lines.append(f"    // {line}")
        return self

    def blank(self):
        self._lines.append("")
        return self

    def directive(self, text):
        self._lines.append(f"    {text}")
        return self

    def label(self, name):
        if name in self._labels:
            raise Diagnostic(f"duplicate label {name!r}")
        self._labels.add(name)
        self._lines.append(f"{name}:")
        return self

    # ---- instructions -----------------------------------------------------
    def emit(self, mnemonic, *operands, pred=None, comment=None):
        """One instruction, checked against the ISA.

        `pred` is a predicate register name, optionally negated as '!p1'.
        sec 12 wants a masked operation to be predicated rather than turned
        into a select after the fact: a masked load that executes and is then
        discarded has already performed the access.
        """
        base = mnemonic.split('.')[0] if mnemonic.startswith('.') else mnemonic
        if not self._known(base):
            raise Diagnostic(
                f"no TNCA instruction {mnemonic!r}",
                because='checked against isa/tnca-isa-v0.1.json, the single '
                        'source of sec 11.3',
                remedy='use an encoding the ISA defines, or add it there first',
                source='compiler_overview.md sec 11.3')

        text = mnemonic
        if operands:
            text = f"{mnemonic:<12s} " + ', '.join(str(o) for o in operands)
        if pred:
            text = f"@{pred} {text}"
        line = f"    {text}"
        if comment:
            line = f"{line:<48s} // {comment}"
        self._lines.append(line)
        self.instruction_count += 1
        return self

    @staticmethod
    def _known(mnemonic):
        if mnemonic in ISA:
            return True
        # Suffixed forms: 'v.ld.w' is 'v.ld' with a size suffix, 's.cmp.lt' is
        # 's.cmp' with a condition code. Strip trailing suffixes until an ISA
        # entry matches.
        parts = mnemonic.split('.')
        for cut in range(len(parts) - 1, 0, -1):
            if '.'.join(parts[:cut]) in ISA:
                return True
        # Assembler pseudo-ops (sec 11.1 reuses the existing assembler).
        return mnemonic in ('.li', '.assert_eq', '.assert_lane', '.dword',
                            '.word', '.org', '.space', '.align', '.float')

    def require(self, capability, what):
        """Refuse an encoding the target cannot execute (sec 12)."""
        return self.target.require(capability, what)

    # ---- output -----------------------------------------------------------
    def text(self):
        return '\n'.join(self._lines) + '\n'

    def __str__(self):
        return self.text()
