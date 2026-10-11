"""DCO: the compiler half of Dynamic Cache Orchestration (doc/DCO.pdf, sec IV).

The NUCA slices decide what to keep. What they cannot see is the dataflow:
how many times each tile of each tensor will be read before the kernel is
done with it. The compiler can, and this pass tells them, through the Tensor
Management Unit (rtl/nuca/tnca_tmu.v):

  nAcc     per tensor, how many times each of its tiles is read. A tile whose
           reads reach nAcc is dead, and the slices evict dead lines first.
  bypass   nAcc = 1: every tile is read once, so caching it can only evict
           something that will be read again. The slices do not allocate it.
  tile     the granule the TMU counts each tensor's tiles in: an aligned
           2^lsb-byte chunk, per tensor, as the paper's TMU keeps a tile size
           per tensor.

The paper computes nAcc as the product of the trip counts of the loops --
the grid among them -- that do not appear in the tile's index map. This pass
gets the same number by a more literal route: it RUNS the kernel's address
arithmetic, every program of the grid, every loop iteration, with nothing but
integers, and counts how many times each tile origin is loaded. On the
loop nests the paper's rule covers the two agree; where they would not (a
grid axis split with `pid // n` and `pid % n`, a loop strided by the program
id), counting is still right and the product would not be.

It refuses rather than guesses. A loop bound it cannot evaluate, a load whose
address depends on loaded data, a branch on something it does not know, or a
trace longer than its budget, and the kernel is compiled without
registrations -- which is the baseline, never a wrong answer. Everything the
TMU does with what it is told is a hint about where a line lives; nothing it
does can change what a load returns.

What reaches the machine is a handful of scalar stores at the top of the
kernel, before its first load:

  TMU+0x040  TILE    D_LSB, the default granule
  TMU+0x100  BASE    the tensor's address          } once per tensor; the
  TMU+0x108  SIZE    its extent in bytes           } write to DESC commits
  TMU+0x110  DESC    nAcc | bypass << 32 | id << 36 | tile lsb << 40

Every program of the grid makes them. Registration is by base address, so the
second registration of a tensor updates the first, and the TMU keeps one
staging line per requester, so programs registering at the same time do not
mix their fields.
"""
from dataclasses import dataclass, field

# The TMU's register window (rtl/nuca/tnca_tmu.v; tnca_top's TMU_BASE).
TMU_BASE = 0x7FFF0000
REG_TILE = 0x040
REG_STAGE = 0x100

# Interpretation budget, in operations evaluated across the whole grid.
BUDGET = 4_000_000

# A tile smaller than a line still lives in lines; one smaller than that
# would make the TMU count words of a line the slice evicts as a whole.
MIN_D_LSB = 6
MAX_D_LSB = 20


class Refused(Exception):
    """The pass cannot count this kernel exactly; it registers nothing."""


@dataclass(frozen=True)
class Ptr:
    arg: int          # the function argument the pointer derives from
    addr: int         # byte address of the element this run tracks
    esize: int        # bytes per element


@dataclass(frozen=True)
class Desc:
    arg: int
    base: int
    shape: tuple
    strides: tuple    # in elements, outermost first
    esize: int
    block: tuple


@dataclass
class TensorUse:
    """One registered tensor: what the TMU is told about it."""
    arg: int                  # function argument index
    name: str
    base: int                 # compile-time address (the literal)
    size: int                 # bytes, from the extent the trace touched
    nacc: int                 # reads of each tile (the most any tile sees)
    tiles: int                # distinct tiles read
    loads: int                # tile loads, in total
    run: int                  # contiguous bytes per tile row (or whole tile)
    bypass: bool = False
    operand: int = 0
    lsb: int = MIN_D_LSB      # the tile granule the TMU counts in, log2 bytes


@dataclass
class Plan:
    tensors: list = field(default_factory=list)
    d_lsb: int = MIN_D_LSB
    note: str = ''


def _granule(run):
    """The largest power of two in one contiguous run, at least a line.

    A chunk no larger than the run is read whole by every read of its tile,
    so its last word is read exactly as often as the tile is."""
    return max(MIN_D_LSB, min(MAX_D_LSB, run.bit_length() - 1))


_ESIZE = {'f64': 8, 'i64': 8, 'u64': 8, 'f32': 4, 'i32': 4, 'u32': 4,
          'f16': 2, 'bf16': 2, 'i16': 2, 'u16': 2, 'i8': 1, 'u8': 1, 'i1': 1,
          'f8E4M3FN': 1, 'f8E5M2': 1}


def _esize(elem):
    return _ESIZE.get(elem, 4 if not elem.startswith('f8') else 1)


def _cdiv(a, b):
    return -((-a) // b)


def _trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


_INT_BIN = {
    'arith.addi': lambda a, b: a + b,
    'arith.subi': lambda a, b: a - b,
    'arith.muli': lambda a, b: a * b,
    'arith.divsi': _trunc_div,
    'arith.divui': lambda a, b: a // b,
    'arith.remsi': lambda a, b: a - b * _trunc_div(a, b),
    'arith.remui': lambda a, b: a % b,
    'arith.ceildivsi': _cdiv,
    'arith.floordivsi': lambda a, b: a // b,
    'arith.minsi': min, 'arith.maxsi': max,
    'arith.minui': min, 'arith.maxui': max,
    'arith.andi': lambda a, b: a & b,
    'arith.ori': lambda a, b: a | b,
    'arith.xori': lambda a, b: a ^ b,
    'arith.shli': lambda a, b: a << b,
    'arith.shrsi': lambda a, b: a >> b,
    'arith.shrui': lambda a, b: a >> b,
}

_CMP = {
    'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b,
    'slt': lambda a, b: a < b, 'sle': lambda a, b: a <= b,
    'sgt': lambda a, b: a > b, 'sge': lambda a, b: a >= b,
    'ult': lambda a, b: a < b, 'ule': lambda a, b: a <= b,
    'ugt': lambda a, b: a > b, 'uge': lambda a, b: a >= b,
}
_CMPI_PRED = ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge']

# Value-preserving for an address: the element this run tracks is the same.
_PASS = {'tt.splat', 'tt.broadcast', 'tt.expand_dims', 'tt.reshape',
         'ttg.convert_layout', 'arith.extsi', 'arith.extui', 'arith.trunci',
         'arith.index_cast', 'arith.index_castui', 'tt.int_to_ptr',
         'tt.ptr_to_int'}

# Whatever these produce is not an address and cannot become one.
_OPAQUE_PREFIX = ('arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf',
                  'arith.negf', 'arith.maxnumf', 'arith.minnumf',
                  'arith.maximumf', 'arith.minimumf', 'arith.truncf',
                  'arith.extf', 'arith.sitofp', 'arith.uitofp', 'arith.fptosi',
                  'arith.fptoui', 'arith.cmpf', 'arith.bitcast', 'math.',
                  'tt.dot', 'tt.reduce', 'tt.scan', 'tt.trans', 'tt.join',
                  'tt.split', 'tt.fp_to_fp', 'tt.precise', 'tt.extern',
                  'tt.histogram', 'tt.gather', 'tt.cat', 'tt.clampf',
                  'tt.mulhiui', 'tt.bitcast')


class _Run:
    """One interpretation: one program of the grid, one element (lo or hi)."""

    def __init__(self, an, pid, mode):
        self.an = an
        self.pid = pid
        self.mode = mode          # 'lo': first element, 'hi': last
        self.v = {}
        self.events = []          # (op id, Ptr or (Desc, coords))

    def val(self, value):
        vid = value.id()
        if vid in self.v:
            return self.v[vid]
        if vid in self.an.consts:
            return self.an.consts[vid]
        return None

    def block(self, ops):
        an = self.an
        for op in ops:
            an.spent += 1
            if an.spent > BUDGET:
                raise Refused(f'the address trace exceeds {BUDGET} operations')
            name = op.get_name()
            if name in ('scf.yield', 'tt.return'):
                return [self.val(op.get_operand(i))
                        for i in range(op.get_num_operands())]
            self.op(op, name)
        return []

    def op(self, op, name):
        an = self.an
        res = op.get_result(0).id() if op.get_num_results() else None
        if name == 'arith.constant':
            if res not in an.consts:
                from .frontend import TTIRImporter
                k = TTIRImporter._constant_value(op)
                self.v[res] = k if isinstance(k, int) and not isinstance(k, bool) else None
            return
        if name == 'tt.get_program_id':
            self.v[res] = self.pid[op.get_int_attr('axis') or 0]
            return
        if name == 'tt.get_num_programs':
            self.v[res] = an.grid[op.get_int_attr('axis') or 0]
            return
        if name == 'tt.make_range':
            start = op.get_int_attr('start') or 0
            end = op.get_int_attr('end') or 0
            self.v[res] = start if self.mode == 'lo' else end - 1
            return
        if name in _PASS:
            self.v[res] = self.val(op.get_operand(0))
            return
        if name in _INT_BIN:
            a, b = self.val(op.get_operand(0)), self.val(op.get_operand(1))
            if isinstance(a, int) and isinstance(b, int):
                try:
                    self.v[res] = _INT_BIN[name](a, b)
                except ZeroDivisionError:
                    self.v[res] = None
            else:
                self.v[res] = None
            return
        if name == 'arith.cmpi':
            a, b = self.val(op.get_operand(0)), self.val(op.get_operand(1))
            pred = an.cmpi_pred(op)
            self.v[res] = (int(_CMP[pred](a, b))
                           if pred and isinstance(a, int) and isinstance(b, int)
                           else None)
            return
        if name == 'arith.select':
            c = self.val(op.get_operand(0))
            a, b = self.val(op.get_operand(1)), self.val(op.get_operand(2))
            self.v[res] = (a if c else b) if isinstance(c, int) else (a if a == b else None)
            return
        if name == 'tt.addptr':
            p, o = self.val(op.get_operand(0)), self.val(op.get_operand(1))
            self.v[res] = (Ptr(p.arg, p.addr + o * p.esize, p.esize)
                           if isinstance(p, Ptr) and isinstance(o, int) else None)
            return
        if name == 'tt.load':
            p = self.val(op.get_operand(0))
            if isinstance(p, Ptr):
                self.events.append((op, p))
            elif p is None and an.pointer_operand(op):
                raise Refused('a load whose address the pass cannot compute')
            self.v[res] = None
            return
        if name == 'tt.make_tensor_descriptor':
            self.v[res] = an.make_desc(op, self)
            return
        if name in ('tt.descriptor_load', 'tt.descriptor_gather'):
            d = self.val(op.get_operand(0))
            coords = [self.val(op.get_operand(i)) for i in range(1, op.get_num_operands())]
            if not isinstance(d, Desc) or any(not isinstance(c, int) for c in coords):
                raise Refused('a descriptor load the pass cannot place')
            if name == 'tt.descriptor_load':
                self.events.append((op, (d, tuple(coords))))
            self.v[res] = None
            return
        if name == 'scf.for':
            self.for_(op)
            return
        if name == 'scf.if':
            c = self.val(op.get_operand(0))
            if not isinstance(c, int):
                raise Refused('a branch on a value the pass does not know')
            out = []
            if c or op.get_num_regions() > 1:
                body = an.body(op.get_region(0 if c else 1))
                out = self.block(body) if body else []
            for i in range(op.get_num_results()):
                self.v[op.get_result(i).id()] = out[i] if i < len(out) else None
            return
        if name in ('scf.while', 'cf.br', 'cf.cond_br'):
            raise Refused(f'{name}: a trip count the pass cannot know in advance')
        # Stores, atomics, barriers, float arithmetic, dots: nothing an
        # address is made of. Their results are opaque.
        for i in range(op.get_num_results()):
            self.v[op.get_result(i).id()] = None

    def for_(self, op):
        an = self.an
        lo, hi, step = (self.val(op.get_operand(i)) for i in range(3))
        if not all(isinstance(x, int) for x in (lo, hi, step)) or step <= 0:
            raise Refused('a loop whose bounds the pass cannot evaluate')
        region = op.get_region(0)
        body = an.body(region)
        blk = body[0].get_block() if body else None
        n_carry = op.get_num_results()
        carried = [self.val(op.get_operand(3 + i)) for i in range(n_carry)]
        iv = lo
        while iv < hi:
            if blk is not None:
                self.v[blk.get_argument(0).id()] = iv
                for i in range(n_carry):
                    self.v[blk.get_argument(i + 1).id()] = carried[i]
            out = self.block(body)
            carried = [out[i] if i < len(out) else None for i in range(n_carry)]
            iv += step
        for i in range(n_carry):
            self.v[op.get_result(i).id()] = carried[i]


class _Analysis:
    def __init__(self, importer, fn, ops, bodies, arg_values, grid):
        self.imp = importer
        self.fn = fn
        self.ops = ops
        self.bodies = bodies
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.spent = 0
        self.consts = {k: v for k, v in importer._const.items() if isinstance(v, int)}
        self.args = {}            # MLIR arg id -> its value in a run
        self.arg_info = {}        # arg index -> (name, elem)
        from .frontend import _parse_type, _pointee
        self._parse_type, self._pointee = _parse_type, _pointee
        names = importer.arg_names
        for i in range(fn.get_num_args()):
            a = fn.args(i)
            ty = str(a.get_type())
            name = names[i] if i < len(names) else f'arg{i}'
            lit = arg_values.get(name)
            pe = _pointee(ty)
            if pe is not None:
                self.args[a.id()] = Ptr(i, int(lit), _esize(pe)) if isinstance(lit, int) else None
                self.arg_info[i] = (name, pe)
            elif ty.startswith('i') and isinstance(lit, int):
                self.args[a.id()] = int(lit)
            elif ty.startswith('!tt.tensordesc<'):
                raise Refused('a descriptor passed in from the host')
            else:
                self.args[a.id()] = None
        self.consts.update({k: v for k, v in self.args.items() if v is not None})
        self._cmpi_cache = {}

    def body(self, region):
        return self.bodies.get(region.id(), [])

    def pointer_operand(self, op):
        return self._pointee(self._parse_type(str(op.get_operand(0).get_type()))[1]) is not None

    def cmpi_pred(self, op):
        p = op.get_int_attr('predicate')
        if p is not None and 0 <= p < len(_CMPI_PRED):
            return _CMPI_PRED[p]
        return None

    def make_desc(self, op, run):
        base = run.val(op.get_operand(0))
        shape, elem = self._parse_type(
            str(op.get_result(0).get_type()).replace('!tt.tensordesc<', 'tensor<'))
        block = tuple(shape) if isinstance(shape, tuple) else (shape or 1,)
        rank = len(block)
        dims = [run.val(op.get_operand(1 + i)) for i in range(rank)]
        strides = [run.val(op.get_operand(1 + rank + i)) for i in range(rank)]
        if not isinstance(base, Ptr) or any(not isinstance(x, int) for x in dims + strides):
            raise Refused('a tensor descriptor the pass cannot evaluate')
        return Desc(base.arg, base.addr, tuple(dims), tuple(strides), _esize(elem), block)

    def trace(self, pid, mode):
        run = _Run(self, pid, mode)
        run.v.update(self.args)
        run.block(self.ops)
        return run.events


def _origin(ev):
    """(arg, byte address of the tile's first element) for one load event."""
    op, what = ev
    if isinstance(what, Ptr):
        return what.arg, what.addr
    d, coords = what
    off = sum(c * s for c, s in zip(coords, d.strides))
    return d.arg, d.base + off * d.esize


def _run_bytes(ev_lo, ev_hi, parse_type):
    """Contiguous bytes of one tile: all of it, or one row of it."""
    op, lo = ev_lo
    _, hi = ev_hi
    if isinstance(lo, Ptr):
        shape, elem = parse_type(str(op.get_result(0).get_type()))
        dims = shape if isinstance(shape, tuple) else (shape or 1,)
        n = 1
        for x in dims:
            n *= x
        span = hi.addr - lo.addr + lo.esize
        if span == n * lo.esize:
            return span, span
        return dims[-1] * lo.esize, span
    d, coords = lo
    inner = d.block[-1] * d.esize
    n = d.esize
    for x in d.block:
        n *= x
    if len(d.block) >= 2 and d.block[-1] == d.shape[-1] and d.strides[-2] == d.shape[-1]:
        return n, n
    span = n if len(d.block) == 1 else (d.block[-2] - 1) * d.strides[-2] * d.esize + inner
    return inner, span


def plan(importer, fn, ops, bodies, arg_values, grid=(1, 1, 1)):
    """Count every tile load of the kernel over the whole grid.

    Returns a Plan, or raises Refused with the reason.
    """
    an = _Analysis(importer, fn, ops, bodies, arg_values, grid)
    g = an.grid
    if g[0] * g[1] * g[2] > 4096:
        raise Refused(f'a grid of {g[0] * g[1] * g[2]} programs is beyond the budget')
    counts = {}      # arg -> {origin: reads}
    extent = {}      # arg -> furthest byte touched
    loads = {}
    for z in range(g[2]):
        for y in range(g[1]):
            for x in range(g[0]):
                for ev in an.trace((x, y, z), 'lo'):
                    arg, org = _origin(ev)
                    c = counts.setdefault(arg, {})
                    c[org] = c.get(org, 0) + 1
                    loads[arg] = loads.get(arg, 0) + 1
    # Geometry from program 0: the same loads, tracking the LAST element.
    first = an.trace((0, 0, 0), 'lo')
    last = an.trace((0, 0, 0), 'hi')
    if len(first) != len(last):
        raise Refused('the first- and last-element traces disagree')
    runs = {}
    span_of = {}
    for a, b in zip(first, last):
        arg, _ = _origin(a)
        run, span = _run_bytes(a, b, an._parse_type)
        runs[arg] = min(runs.get(arg, run), run)
        span_of[arg] = max(span_of.get(arg, 0), span)

    p = Plan()
    for i, (arg, c) in enumerate(sorted(counts.items())):
        name, _ = an.arg_info.get(arg, (f'arg{arg}', None))
        base = an.args[fn.args(arg).id()].addr
        top = max(c) + span_of.get(arg, 0)
        size = (top - base + 63) // 64 * 64
        nacc = max(c.values())
        run = runs.get(arg, 64)
        # Bypass only what is read once AT LINE GRANULARITY. A tile whose rows
        # are narrower than a line shares each line with its neighbour --
        # X[m, kb*16:(kb+1)*16] in f16 is half a line -- so a tile read once
        # is still a line read twice, and bypassing it would fetch the line
        # twice. Dead-block prediction still sees it die after its last word.
        whole_lines = run % 64 == 0 and all(o % 64 == 0 for o in c)
        p.tensors.append(TensorUse(arg=arg, name=name, base=base, size=size,
                                   nacc=nacc, tiles=len(c), loads=loads[arg],
                                   run=run, bypass=(nacc == 1 and whole_lines),
                                   operand=i & 15, lsb=_granule(run)))
    reused = [t.run for t in p.tensors if t.nacc > 1]
    if reused:
        p.d_lsb = _granule(min(reused))
    p.note = f'{an.spent} operations traced over {g[0] * g[1] * g[2]} program(s)'
    return p


def emit(importer, plan_, at):
    """Splice the registration stores into the kernel at op index `at`."""
    b = importer.b
    region = b.region
    tail = region.ops[at:]
    region.ops = region.ops[:at]
    tmu = b.const(TMU_BASE)
    b.comment(f'DCO: register {len(plan_.tensors)} tensor(s) with the TMU '
              f'(doc/DCO.pdf sec IV)')
    b.sstore(tmu, b.const(plan_.d_lsb), offset=REG_TILE)
    for t in plan_.tensors:
        base = importer._args[t.arg][1]
        desc = ((t.nacc & 0xFFFFFFFF) | (int(t.bypass) << 32)
                | ((t.operand & 15) << 36) | ((t.lsb & 63) << 40))
        b.comment(f'{t.name}: nAcc={t.nacc}{" bypass" if t.bypass else ""} '
                  f'tiles={t.tiles} size={t.size} granule=2^{t.lsb}')
        b.sstore(tmu, base, offset=REG_STAGE)
        b.sstore(tmu, b.const(t.size), offset=REG_STAGE + 8)
        b.sstore(tmu, b.const(desc), offset=REG_STAGE + 16)
    region.ops.extend(tail)
