"""Tiled element-wise lowering: a 1-D program as a loop over register tiles.

Option `tiled_pointwise` (TNCA_TILED_POINTWISE), default 0 (off):

  0  off: the program is lowered as before.
  1  a 1-D element-wise program of block W runs as a loop over tiles of
     TILE elements. Every contiguous access whose legality is shown below
     moves a whole tile through the tensor transfer engine (`t.ld.async`
     into a register window, `t.st.async` out of one). When the windows fit,
     an iteration handles two tiles A and B with separate windows: B's loads
     are issued as soon as A's have arrived and move while A is computed, and
     A's stores move while B is computed (software pipelining inside the
     iteration; every command is waited before the iteration ends, as the
     code generator's ownership check requires). Every other access keeps
     its vector load or store.
  2  the same loop with every access through the vector load/store unit.
  3  (R41 G4, for the line-wide multi-command tensor engine) the tensor-engine
     accesses of mode 1, streamed: the program's tiles are unrolled and each
     transfer has a ring of R windows. The loads of the first R tiles are
     issued at once; tile i waits only for its own loads (and, before its
     store window is refilled, for the store of tile i-R), and the loads of
     tile i+R are issued as soon as tile i has been computed. Up to R tiles
     of every load stream are in flight, without a vector register held per
     line by the load/store unit. Stores of the last R tiles are waited at
     the end of the program.

Which programs (generic, from the TTIR alone; nothing is recognised by name,
shape or size):
  * one straight-line block: no control flow, no region;
  * every tensor is 1-D, of one width W, and W is at least two tiles;
  * operations are element-wise (arith, math, the element-wise tt ops),
    pointer arithmetic, loads and stores; no reduction, scan, dot, gather
    of another lane, reshape, atomic or descriptor;
  * every lane index is `splat(s) + arange(0, W)` with s a scalar, and that
    splat and arange are used only there -- so tile t of the program sees
    lanes [t*TILE, (t+1)*TILE) of every value;
  * no load follows a store, and a pointer argument that is both loaded and
    stored is accessed through one index (the original program reads its
    whole block before writing; with one index per argument, reading and
    writing tile by tile gives the same result).

A tensor-engine access (mode 1) additionally needs:
  * a 4-byte element (the window layout equals the 16-lane register layout
    only for 32-bit elements; ISA sec 9C.2), and a target with tensor_dma
    (and tensor_store for a store);
  * the pointer `splat(arg) + (splat(s) + arange)`, with `arg` a pointer
    argument (element-aligned, as every Triton pointer is). The map's base is
    `arg` rounded down to 8 bytes and the coordinates start the skipped
    element later (ISA sec 9E.2 wants an 8-byte-aligned base and allows any
    coordinate);
  * no mask, or the mask `index < n` with n a scalar argument or constant,
    which becomes the descriptor's extent: the engine pads a load and
    clips a store outside it (ISA sec 9E), as the mask does;
  * a load without an `other` value.

This is a lowering of the same program: the numerics, the instructions'
semantics and the data hierarchy (register file, NUCA, DRAM) are unchanged.
"""
import re

_EW_TT = {'tt.addptr', 'tt.load', 'tt.store', 'tt.bitcast', 'tt.fp_to_fp',
          'tt.extern_elementwise', 'tt.precise_sqrt', 'tt.precise_divf',
          'tt.clampf', 'tt.mulhiui', 'tt.splat', 'tt.make_range'}
_SCALAR_OK = {'tt.get_program_id', 'tt.get_num_programs', 'llvm.intr.assume'}
_SKIP = {'tt.func', 'builtin.module', 'tt.return'}
EXTENT_UNMASKED = (1 << 31) - 1    # i32 lane indices never reach it
WINDOW_BUDGET = 48                 # of a context's 64 vector registers
LSU_GROUP_BUDGET = 56              # registers the in-flight load groups may hold
STREAM_BUDGET = 48                 # window registers of a mode-3 ring
_LONG_OPS = {'arith.divf', 'tt.precise_divf', 'tt.precise_sqrt', 'tt.extern_elementwise',
             'math.exp', 'math.exp2', 'math.log', 'math.log2', 'math.sqrt', 'math.rsqrt',
             'math.sin', 'math.cos', 'math.erf', 'math.log1p', 'math.tanh'}


def _shape(value):
    m = re.match(r'^tensor<((?:\d+x)+)', str(value.get_type()))
    return tuple(int(n) for n in m.group(1).split('x') if n) if m else ()


def _elem_of_ptr(value):
    m = re.search(r'!tt\.ptr<(\w+)>', str(value.get_type()))
    return m.group(1) if m else None


class Access:
    def __init__(self, op, kind, arg, elem, base_splat, n):
        self.op, self.kind, self.arg, self.elem = op, kind, arg, elem
        self.base_splat = base_splat     # the splat(s) op of the lane index
        self.n = n                       # extent: TTIR scalar value, int, or None
        self.windows = []
        self.desc = None


class Plan:
    def __init__(self, width, tile, scalars, tensor_ops, bases, ranges, accesses, double):
        self.width, self.tile = width, tile
        self.scalars, self.tensor_ops = scalars, tensor_ops
        self.bases, self.ranges = bases, ranges
        self.accesses = accesses          # op id -> Access (mode 1 only)
        self.double = double

    def summary(self):
        return dict(width=self.width, tile=self.tile, double_buffer=self.double,
                    groups=getattr(self, 'groups', 1), ring=getattr(self, 'ring', 0),
                    accesses=sorted(a.kind for a in self.accesses.values()))


def widest(ops):
    """The largest dimension of any tensor in `ops`."""
    w = 0
    for o in ops:
        for v in [o.get_operand(i) for i in range(o.get_num_operands())] + \
                 [o.get_result(i) for i in range(o.get_num_results())]:
            w = max([w] + list(_shape(v)))
    return w


def _no(imp, where):
    """Refuse, and keep the reason (its position in plan()) for provenance."""
    imp.tiled_pointwise_refused = where
    return None


def _zero_other(imp, defs, value):
    """True when a load's `other` is a dense +0 (what the engine pads with).
    The literal comes from the printed module (frontend._float_constants)."""
    o = defs.get(value.id())
    if o is None or o.get_name() != 'arith.constant':
        return False
    lit = (getattr(imp, '_printed_consts', None) or {}).get(value.id())
    m = re.fullmatch(r'dense<([^>]+)>', lit or '')
    if not m or m.group(1).strip().startswith('-'):
        return False
    try:
        return float(m.group(1)) == 0.0
    except ValueError:
        return m.group(1).strip() == 'false'


def plan(imp, ops, module_text, func_name):
    """A Plan, or None when the program is not a tileable 1-D element-wise one.

    Mode 3 applies to programs whose every access can use the tensor engine;
    any other tileable program takes mode 2's lowering instead."""
    if getattr(imp, 'tiled_pointwise', 0) == 3:
        p = _plan(imp, ops, module_text, func_name)
        if p is not None:
            return p
        imp.tiled_pointwise = 2
        try:
            return _plan(imp, ops, module_text, func_name)
        finally:
            imp.tiled_pointwise = 3
    return _plan(imp, ops, module_text, func_name)


def _plan(imp, ops, module_text, func_name):
    mode = getattr(imp, 'tiled_pointwise', 0)
    vlen = imp.target.vlen
    if not mode or not imp.index_opt:
        return _no(imp, 1)
    defs, users = {}, {}
    for o in ops:
        for i in range(o.get_num_results()):
            defs[o.get_result(i).id()] = o
        for i in range(o.get_num_operands()):
            users.setdefault(o.get_operand(i).id(), []).append((o, i))
    width = None
    scalars, tensor_ops, stored = [], [], False
    loads, stores = [], []
    for o in ops:
        name = o.get_name()
        if name in _SKIP:
            continue
        if name.startswith(('cf.', 'scf.')):
            return _no(imp, 2)
        shapes = [_shape(o.get_operand(i)) for i in range(o.get_num_operands())] + \
                 [_shape(o.get_result(i)) for i in range(o.get_num_results())]
        tensors = [s for s in shapes if s]
        if not tensors:
            if name in _SCALAR_OK or name.startswith(('arith.', 'math.')):
                scalars.append(o)
                continue
            return _no(imp, 3)
        if any(len(s) != 1 for s in tensors):
            return _no(imp, 4)
        if width is None:
            width = tensors[0][0]
        if any(s[0] != width for s in tensors):
            return _no(imp, 5)
        if name not in _EW_TT and not name.startswith(('arith.', 'math.')):
            return _no(imp, 6)
        if name == 'tt.make_range':
            if (o.get_int_attr('start') or 0) != 0 or o.get_int_attr('end') != width:
                return _no(imp, 7)
        if name == 'tt.store':
            stored = True
            stores.append(o)
        if name == 'tt.load':
            if stored:
                return _no(imp, 8)
            loads.append(o)
        tensor_ops.append(o)
    tile_min = 2 * vlen
    if width is None or width % tile_min or width < 2 * tile_min:
        return _no(imp, 9)
    # Every arange is the lane index of `splat(s) + arange`, and that splat
    # serves only such sums: then tile t is lanes [t*TILE, (t+1)*TILE).
    ranges = [o for o in tensor_ops if o.get_name() == 'tt.make_range']
    bases = {}
    if not ranges:
        return _no(imp, 10)
    for r in ranges:
        for u, _i in users.get(r.get_result(0).id(), []):
            if u.get_name() != 'arith.addi':
                return _no(imp, 11)
            other = u.get_operand(1 - _i)
            so = defs.get(other.id())
            if so is None or so.get_name() != 'tt.splat' or _shape(so.get_operand(0)):
                return _no(imp, 12)
            bases[id(so)] = so
    for so in bases.values():
        for u, _i in users.get(so.get_result(0).id(), []):
            if u.get_name() != 'arith.addi':
                return _no(imp, 13)
            o2 = defs.get(u.get_operand(1 - _i).id())
            if o2 is None or o2.get_name() != 'tt.make_range':
                return _no(imp, 14)

    def index_of(ptr):
        """(argument name, element, base splat op, offset value) of `splat(arg) + index`."""
        po = defs.get(ptr.id())
        if po is None or po.get_name() != 'tt.addptr':
            return _no(imp, 15)
        so, off = defs.get(po.get_operand(0).id()), po.get_operand(1)
        if so is None or so.get_name() != 'tt.splat':
            return _no(imp, 16)
        src = so.get_operand(0)
        if src.id() in defs or src.id() not in imp._ptr_base or imp._ptr_chain.get(src.id()):
            return _no(imp, 17)
        ao = defs.get(off.id())
        if ao is None or ao.get_name() != 'arith.addi':
            return _no(imp, 18)
        a, b = (defs.get(ao.get_operand(i).id()) for i in (0, 1))
        base = a if a is not None and id(a) in bases else b if b is not None and id(b) in bases else None
        if base is None:
            return _no(imp, 19)
        name, elem = imp._ptr_base[src.id()]
        return name, elem, base, off

    # One index per argument that is both read and written.
    def arg_offsets(group):
        out = {}
        for o in group:
            ptr = o.get_operand(0)
            po = defs.get(ptr.id())
            chain = []
            while po is not None and po.get_name() == 'tt.addptr':
                chain.append(po.get_operand(1).id())
                po = defs.get(po.get_operand(0).id())
            if po is None or po.get_name() != 'tt.splat':
                return _no(imp, 20)
            arg = po.get_operand(0).id()
            out.setdefault(arg, set()).add(tuple(chain))
        return out
    lo, so_ = arg_offsets(loads), arg_offsets(stores)
    if lo is None or so_ is None:
        return _no(imp, 21)
    for arg in set(lo) & set(so_):
        if len(lo[arg] | so_[arg]) != 1:
            return _no(imp, 22)

    accesses = {}
    if mode in (1, 3) and imp.target.has('tensor_dma'):
        for o in loads + stores:
            store = o.get_name() == 'tt.store'
            if store and not imp.target.has('tensor_store'):
                continue
            idx = index_of(o.get_operand(0))
            if idx is None:
                continue
            name, elem, base, off = idx
            if imp._ELEM_BYTES.get(elem) != 4:
                continue
            nmask = 2 if store else 1
            n = None
            if o.get_num_operands() > nmask:
                mo = defs.get(o.get_operand(nmask).id())
                if (mo is None or mo.get_name() != 'arith.cmpi' or mo.get_int_attr('predicate') != 2
                        or mo.get_operand(0).id() != off.id()):
                    continue
                no = defs.get(mo.get_operand(1).id())
                if no is None or no.get_name() != 'tt.splat':
                    continue
                n = no.get_operand(0)
                k = imp._scalar_const(n.id())
                if k is not None:
                    n = int(k)
                elif n.id() in defs:
                    continue        # computed in the body: not available before the loop
            if not store and o.get_num_operands() > 2 and not _zero_other(imp, defs, o.get_operand(2)):
                continue
            accesses[id(o)] = Access(o, 'store' if store else 'load', name, elem, base, n)
    # Contiguous accesses that stay on the vector load/store unit: one tile
    # base per argument, each register chunk at an immediate offset
    # (`v.ld`/`v.st [Sa, #imm16]`, ISA 0x10) instead of its own address sum.
    for o in loads + stores:
        if id(o) in accesses:
            continue
        store = o.get_name() == 'tt.store'
        idx = index_of(o.get_operand(0))
        if idx is None or (not store and o.get_num_operands() > 2
                           and not _zero_other(imp, defs, o.get_operand(2))):
            continue
        name, elem, base, off = idx
        if imp._ELEM_BYTES.get(elem) not in (1, 2, 4, 8):
            continue
        accesses[id(o)] = Access(o, 'vstore' if store else 'vload', name, elem, base, None)
    nl = sum(1 for a in accesses.values() if a.kind == 'load')
    ns = sum(1 for a in accesses.values() if a.kind == 'store')
    # Window registers: two tiles per iteration, each transfer with a window
    # per tile, when they fit (the second tile's transfers overlap the first
    # tile's arithmetic); else one tile per iteration.
    tile_regs, double = 8, False
    if nl + ns:
        for t in (16, 8, 4, 2):
            if 2 * (nl + ns) * t <= WINDOW_BUDGET and width % (2 * t * vlen) == 0:
                tile_regs, double = t, True
                break
        else:
            for t in (16, 8, 4, 2, 1):
                if (nl + ns) * t <= WINDOW_BUDGET and width % (t * vlen) == 0 and width >= 2 * t * vlen:
                    tile_regs = t
                    break
            else:
                return _no(imp, 30)
    tile_regs = min(tile_regs, width // (2 * vlen))
    # Without tensor-engine transfers the vector loads of later tiles are
    # issued before the arithmetic and stores of earlier ones (`groups` tiles
    # per iteration), so their memory latency overlaps: as many groups as the
    # registers hold (each tile keeps one register chunk per load plus one).
    groups = 2 if double else 1
    if not (nl + ns) and getattr(imp, 'tiled_groups_cap', 0) != 1:
        nvl = sum(1 for a in accesses.values() if a.kind == 'vload')
        # a masked access also holds its lane index and predicate per chunk
        masked = any(a.op.get_num_operands() > (2 if a.kind == 'vstore' else 1)
                     for a in accesses.values() if a.kind in ('vload', 'vstore'))
        # a computed result needs its own register; a plain copy stores the
        # loaded register itself
        computes = any(o.get_name().startswith(('arith.', 'math.')) or
                       o.get_name() in ('tt.fp_to_fp', 'tt.extern_elementwise', 'tt.bitcast',
                                        'tt.precise_sqrt', 'tt.precise_divf', 'tt.clampf')
                       for o in tensor_ops if id(o) not in bases and o.get_name() != 'tt.make_range'
                       and not (o.get_name() == 'arith.addi' and any(
                           defs.get(o.get_operand(i).id()) in ranges for i in (0, 1)))
                       and not o.get_name() == 'arith.constant' and not o.get_name() == 'arith.cmpi')
        per_chunk = nvl + (1 if computes else 0) + (2 if masked else 0)
        for g in (4, 2):
            if nvl and g * tile_regs * per_chunk <= LSU_GROUP_BUDGET and width % (g * tile_regs * vlen) == 0:
                groups = g
                break
    ring = 0
    if mode == 3:
        if nl + ns != len(loads) + len(stores):
            return _no(imp, 31)          # mode 3 needs every load and store on the tensor engine
        # An operation on the special-function unit or a division keeps the core
        # busy for many cycles per tile; mode 2's lane groups overlap that with
        # memory, mode 3's tile ring does not (model, g4/model_c4/), so such a
        # program takes mode 2's lowering.
        if any(o.get_name() in _LONG_OPS for o in tensor_ops):
            return _no(imp, 33)
        best = None
        for t in (16, 8, 4, 2, 1):
            nt = width // (t * vlen)
            if width % (t * vlen) or nt < 2:
                continue
            for r in (4, 3, 2):
                if r <= nt and r * (nl + ns) * t <= STREAM_BUDGET and 2 * r <= imp.target.num_barriers:
                    if best is None or r * t > best[0] * best[1] or (r * t == best[0] * best[1] and t > best[1]):
                        best = (r, t)
                    break
        if best is None:
            return _no(imp, 32)
        ring, tile_regs = best
    p = Plan(width, tile_regs * vlen, scalars, tensor_ops, bases, ranges, accesses, double)
    p.groups = groups
    p.ring = ring
    return p


class _TValue:
    """A TTIR value narrowed to one tile (of lane group `group` of an iteration)."""
    def __init__(self, value, tile, group=0):
        self.value, self._tile, self._group = value, tile, group

    def id(self):
        if not _shape(self.value):
            return self.value.id()
        return ('tile-split', self.value.id()) if not self._group else ('tile-split', self.value.id(), self._group)

    def get_type(self):
        text = str(self.value.get_type())
        return re.sub(r'^tensor<\d+x', f'tensor<{self._tile}x', text) if _shape(self.value) else self.value.get_type()

    def __getattr__(self, name):
        return getattr(self.value, name)


class _TOp:
    def __init__(self, op, tile, group=0):
        self.op, self._tile, self._group = op, tile, group

    def get_operand(self, i):
        return _TValue(self.op.get_operand(i), self._tile, self._group)

    def get_result(self, i):
        return _TValue(self.op.get_result(i), self._tile, self._group)

    def get_int_attr(self, name):
        if self.op.get_name() == 'tt.make_range' and name == 'end':
            return self._tile
        return self.op.get_int_attr(name)

    def __getattr__(self, name):
        return getattr(self.op, name)

    def __str__(self):
        return str(self.op)


def _chunk_ids(value, tile, vlen, group=0):
    from .register_blocks import _ValueView
    tv = _TValue(value, tile, group)
    if tile == vlen:
        return [tv.id()]
    return [_ValueView(tv, c, vlen).id() for c in range(tile // vlen)]


def run(imp, p, dispatch):
    """Emit the program as the loop over tiles that `p` describes."""
    if getattr(p, 'ring', 0):
        return run_stream(imp, p, dispatch)
    b = imp.b
    vlen = imp.target.vlen
    tile, width = p.tile, p.width
    nchunk = tile // vlen
    b.comment(f'tiled element-wise program: {width} lanes as {width // tile} tiles of {tile} '
              f'({len(p.accesses)} tensor-engine transfers, '
              f'{"two" if p.double else "one"} load buffer(s))')
    # Scalars first: none of them depends on a tensor in this program.
    for o in p.scalars:
        dispatch(o)
    tlist = sorted((a for a in p.accesses.values() if a.kind in ('load', 'store')),
                   key=lambda a: imp._op_pos.get(id(a.op), 0))
    loads = [a for a in tlist if a.kind == 'load']
    stores = [a for a in tlist if a.kind == 'store']
    nbuf = 2 if p.double else 1
    esize = 4
    for a in tlist:
        # ISA sec 9E.2 wants an 8-byte-aligned base; a 4-byte element may
        # start 4 bytes past one. The map's base is rounded down and the
        # coordinates move up by the element it skipped: the same bytes.
        arg = imp._arg_value(a.arg)
        low = b.sbini('and', arg, 7)
        base = b.sbin('sub', arg, low)
        a.delta = b.sbini('lshr', low, 2)
        if a.n is None:
            extent = EXTENT_UNMASKED
        elif isinstance(a.n, int):
            extent = b.sbini('add', a.delta, a.n)
        else:
            extent = b.sbin('add', a.delta, imp._value(a.n))
        a.desc = imp._build_descriptor(base=base, elem=a.elem, extents=[extent],
                                       byte_strides=[esize], block_shape=[tile],
                                       what=f'one {tile}-element tile of {a.arg}')
        a.windows = [b.window(nchunk, name=f'{a.kind}_{a.arg}_{i}') for i in range(nbuf)]
    # The lane-index base of each tile: s + t0.
    base_scalar = {bid: imp._value(so.get_operand(0)) for bid, so in p.bases.items()}
    for a in tlist:
        a.origin = b.sbin('add', base_scalar[id(a.base_splat)], a.delta)
    t0 = imp._copy_value(b.const(0))
    lbar = [b.barrier(name=f'tile_load{i}') for i in range(nbuf)] if loads else []
    sbar = [b.barrier(name=f'tile_store{i}') for i in range(nbuf)] if stores else []

    def coord(a, at):
        return b.sbin('add', a.origin, at)

    def issue_loads(buf, at):
        tok = b.barrier_arm(lbar[buf], arrivals=0, ops=len(loads))
        for a in loads:
            b.dma_load(a.windows[buf], a.desc.value, [coord(a, at)], tok, dtype=a.elem)
        return tok

    caches = ('_scalar_cache', '_fconst_cache', '_byte_offsets', '_aff_cache')
    saved = {n: dict(getattr(imp, n)) for n in caches}
    start, end = imp._control_label('tiles'), imp._control_label('endtiles')
    tile_ops = [o for o in p.tensor_ops if o.get_name() != 'tt.make_range' and id(o) not in p.bases]
    tte = bool(loads or stores)
    groups = nbuf if tte else getattr(p, 'groups', 1)
    # Use and definition maps, and printed literals, for the tile views.
    for g in range(groups):
        for o in p.tensor_ops:
            for i in range(o.get_num_results()):
                r = o.get_result(i)
                tid = _TValue(r, tile, g).id()
                imp._value_def[tid] = _TOp(o, tile, g)
                imp._value_users[tid] = [(_TOp(u, tile, g), k) for u, k in imp._value_users.get(r.id(), [])]
                printed = getattr(imp, '_printed_consts', None)
                if printed and r.id() in printed:
                    printed[tid] = printed[r.id()]
    # The order of one iteration. With tensor-engine transfers: group after
    # group (their transfers overlap inside emit_ops). Without: every group's
    # ops up to its last load first, then the rest of every group, so later
    # groups' loads are in flight while earlier groups compute and store.
    last_load = max([i for i, o in enumerate(tile_ops)
                     if getattr(p.accesses.get(id(o)), 'kind', None) == 'vload'] + [-1])
    if tte or groups == 1:
        order = [(g, tile_ops) for g in range(groups)]
    else:
        order = ([(g, tile_ops[:last_load + 1]) for g in range(groups)] +
                 [(g, tile_ops[last_load + 1:]) for g in range(groups)])
    b.label(start)
    stores_pending = []
    try:
        lane0 = {}
        for g in range(groups):
            b.comment(f'tile {g} of the iteration: lanes t0 + {g * tile}')
            tg = t0 if g == 0 else b.sbini('add', t0, g * tile)
            lane0[g] = tg
            # The lane index of this tile.
            for bid, so in p.bases.items():
                sb = b.sbin('add', base_scalar[bid], tg)
                vec = b.splat(sb, dtype='i32')
                for cid in _chunk_ids(so.get_result(0), tile, vlen, g):
                    imp._v[cid] = vec
                    imp._uscalar[cid] = sb
                    imp._offsets[cid] = ('uniform', cid)
            for r in p.ranges:
                for c, cid in enumerate(_chunk_ids(r.get_result(0), tile, vlen, g)):
                    v = b.arange()
                    if c:
                        v = b.vbin('add', v, b.vmovi(c * vlen))
                    imp._v[cid] = v
                    imp._offsets[cid] = ('arange', None) if c == 0 else ('contiguous', cid)
                    if c:
                        imp._uscalar[cid] = imp._const_scalar(c * vlen)
                        imp._nonneg.add(cid)
                        imp._multiple[cid] = c * vlen
        rowbase = {}
        waited = set()

        def tile_base(a, g):
            # &arg[s + t0 + g*TILE], once per argument and tile
            key = (a.arg, id(a.base_splat), g)
            if key not in rowbase:
                esz = imp._ELEM_BYTES[a.elem]
                sb = imp._uscalar[_chunk_ids(a.base_splat.get_result(0), tile, vlen, g)[0]]
                off = sb if esz == 1 else b.sbini('shl', sb, esz.bit_length() - 1)
                rowbase[key] = b.sbin('add', imp._arg_value(a.arg), off)
            return rowbase[key]

        for g, ops_now in order:
            for o in ops_now:
                a = p.accesses.get(id(o))
                if a is None:
                    dispatch(_TOp(o, tile, g))
                    continue
                if a.kind in ('vload', 'vstore'):
                    from .frontend import _load_suffix, _SIZE_SUFFIX
                    esz = imp._ELEM_BYTES[a.elem]
                    rb = tile_base(a, g)
                    nmask = 2 if a.kind == 'vstore' else 1
                    for c, cid in enumerate(_chunk_ids(o.get_result(0) if a.kind == 'vload'
                                                       else o.get_operand(1), tile, vlen, g)):
                        mask = None
                        if o.get_num_operands() > nmask:
                            mask = imp._as_pred(imp._value(_chunk_value(o.get_operand(nmask), tile, vlen, c, g)))
                        if a.kind == 'vload':
                            imp._v[cid] = b.load(
                                rb, mask=mask, other=0, dtype=a.elem,
                                size=_load_suffix(_SIZE_SUFFIX[esz], a.elem), offset=c * vlen * esz)
                        else:
                            val = imp._value(_chunk_value(o.get_operand(1), tile, vlen, c, g))
                            b.store(rb, val, mask=mask, size=_SIZE_SUFFIX[esz], offset=c * vlen * esz)
                    continue
                if a.kind == 'load':
                    if g not in waited:
                        # Two tiles per iteration (p.double): tile A's loads,
                        # then tile B's issued at once, so they move while A
                        # is computed. Every command is waited inside the
                        # iteration (compiler_overview.md sec 10.2).
                        if g == 0:
                            tok_a = issue_loads(0, t0)
                            b.barrier_wait(tok_a)
                            if p.double:
                                tok_b = issue_loads(1, lane0[1])
                        else:
                            b.barrier_wait(tok_b)
                        waited.add(g)
                    for c, cid in enumerate(_chunk_ids(o.get_result(0), tile, vlen, g)):
                        imp._v[cid] = b.window_reg(a.windows[g], c, dtype=a.elem)
                else:
                    value = o.get_operand(1)
                    w = a.windows[g]
                    for c, cid in enumerate(_chunk_ids(value, tile, vlen, g)):
                        held = imp._v.get(cid)
                        if held is None:
                            held = imp._value(_chunk_value(value, tile, vlen, c, g))
                        b.window_put(w, c, held)
                    tok = b.barrier_arm(sbar[g], arrivals=0, ops=1)
                    b.dma_store(w, a.desc.value, [coord(a, lane0[g])], tok, dtype=a.elem)
                    stores_pending.append(tok)
        # Tile A's stores overlapped tile B's arithmetic; all finish before
        # the windows are filled again.
        for tok in stores_pending:
            b.barrier_wait(tok)
        stores_pending.clear()
        nt = b.sbini('add', t0, groups * tile)
        b._emit('assign', [], [t0, nt])
        b.branch_nz(b.scmp('lt', t0, b.const(width)), start)
        b.label(end)
    finally:
        for n, cache in saved.items():
            setattr(imp, n, cache)


def _chunk_value(value, tile, vlen, c, group=0):
    from .register_blocks import _ValueView
    tv = _TValue(value, tile, group)
    return tv if tile == vlen else _ValueView(tv, c, vlen)


def run_stream(imp, p, dispatch):
    """Mode 3: every tile unrolled, a ring of `p.ring` windows per transfer."""
    b = imp.b
    vlen = imp.target.vlen
    tile, width, ring = p.tile, p.width, p.ring
    nchunk, ntiles = tile // vlen, width // tile
    b.comment(f'streamed tiled element-wise program: {width} lanes as {ntiles} tiles of {tile}, '
              f'a ring of {ring} windows per transfer ({len(p.accesses)} tensor-engine transfers)')
    for o in p.scalars:
        dispatch(o)
    tlist = sorted(p.accesses.values(), key=lambda a: imp._op_pos.get(id(a.op), 0))
    loads = [a for a in tlist if a.kind == 'load']
    stores = [a for a in tlist if a.kind == 'store']
    esize = 4
    for a in tlist:
        arg = imp._arg_value(a.arg)
        low = b.sbini('and', arg, 7)
        base = b.sbin('sub', arg, low)
        a.delta = b.sbini('lshr', low, 2)
        if a.n is None:
            extent = EXTENT_UNMASKED
        elif isinstance(a.n, int):
            extent = b.sbini('add', a.delta, a.n)
        else:
            extent = b.sbin('add', a.delta, imp._value(a.n))
        a.desc = imp._build_descriptor(base=base, elem=a.elem, extents=[extent],
                                       byte_strides=[esize], block_shape=[tile],
                                       what=f'one {tile}-element tile of {a.arg}')
        a.windows = [b.window(nchunk, name=f'{a.kind}_{a.arg}_{i}') for i in range(ring)]
    base_scalar = {bid: imp._value(so.get_operand(0)) for bid, so in p.bases.items()}
    for a in tlist:
        a.origin = b.sbin('add', base_scalar[id(a.base_splat)], a.delta)
    lbar = [b.barrier(name=f'stream_load{i}') for i in range(ring)] if loads else []
    sbar = [b.barrier(name=f'stream_store{i}') for i in range(ring)] if stores else []
    tile_ops = [o for o in p.tensor_ops if o.get_name() != 'tt.make_range' and id(o) not in p.bases]
    for g in range(ntiles):
        for o in p.tensor_ops:
            for i in range(o.get_num_results()):
                r = o.get_result(i)
                tid = _TValue(r, tile, g).id()
                imp._value_def[tid] = _TOp(o, tile, g)
                imp._value_users[tid] = [(_TOp(u, tile, g), k) for u, k in imp._value_users.get(r.id(), [])]
                printed = getattr(imp, '_printed_consts', None)
                if printed and r.id() in printed:
                    printed[tid] = printed[r.id()]
    caches = ('_scalar_cache', '_fconst_cache', '_byte_offsets', '_aff_cache')
    saved = {n: dict(getattr(imp, n)) for n in caches}
    load_tok, store_tok = {}, {}

    def issue_loads(i):
        s = i % ring
        tok = b.barrier_arm(lbar[s], arrivals=0, ops=len(loads))
        for a in loads:
            b.dma_load(a.windows[s], a.desc.value, [b.sbini('add', a.origin, i * tile)], tok, dtype=a.elem)
        load_tok[i] = tok

    try:
        if loads:
            for i in range(min(ring, ntiles)):
                issue_loads(i)
        for g in range(ntiles):
            s = g % ring
            b.comment(f'tile {g} (ring slot {s}): lanes {g * tile}..{(g + 1) * tile - 1}')
            for bid, so in p.bases.items():
                sb = b.sbini('add', base_scalar[bid], g * tile)
                vec = b.splat(sb, dtype='i32')
                for cid in _chunk_ids(so.get_result(0), tile, vlen, g):
                    imp._v[cid] = vec
                    imp._uscalar[cid] = sb
                    imp._offsets[cid] = ('uniform', cid)
            for r in p.ranges:
                for c, cid in enumerate(_chunk_ids(r.get_result(0), tile, vlen, g)):
                    v = b.arange()
                    if c:
                        v = b.vbin('add', v, b.vmovi(c * vlen))
                    imp._v[cid] = v
                    imp._offsets[cid] = ('arange', None) if c == 0 else ('contiguous', cid)
                    if c:
                        imp._uscalar[cid] = imp._const_scalar(c * vlen)
                        imp._nonneg.add(cid)
                        imp._multiple[cid] = c * vlen
            if loads:
                b.barrier_wait(load_tok.pop(g))
            store_waited = False
            for o in tile_ops:
                a = p.accesses.get(id(o))
                if a is None:
                    dispatch(_TOp(o, tile, g))
                    continue
                if a.kind == 'load':
                    for c, cid in enumerate(_chunk_ids(o.get_result(0), tile, vlen, g)):
                        imp._v[cid] = b.window_reg(a.windows[s], c, dtype=a.elem)
                    continue
                # a store: its window slot was last used by tile g - ring
                if not store_waited and g - ring in store_tok:
                    b.barrier_wait(store_tok.pop(g - ring))
                    store_waited = True
                value = o.get_operand(1)
                w = a.windows[s]
                for c, cid in enumerate(_chunk_ids(value, tile, vlen, g)):
                    held = imp._v.get(cid)
                    if held is None:
                        held = imp._value(_chunk_value(value, tile, vlen, c, g))
                    b.window_put(w, c, held)
                if g not in store_tok:
                    store_tok[g] = b.barrier_arm(sbar[s], arrivals=0, ops=len(stores))
                b.dma_store(w, a.desc.value, [b.sbini('add', a.origin, g * tile)], store_tok[g], dtype=a.elem)
            # this tile's load windows have been read: refill them with tile g + ring
            if loads and g + ring < ntiles:
                issue_loads(g + ring)
        for g in sorted(store_tok):
            b.barrier_wait(store_tok[g])
        store_tok.clear()
    finally:
        for n, cache in saved.items():
            setattr(imp, n, cache)
