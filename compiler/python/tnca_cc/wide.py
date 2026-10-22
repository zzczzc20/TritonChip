"""Exact 64-bit integer tensor operations over pairs of 32-bit vectors.

Scalar ISA operations retain all 64 bits. Extract/insert packs each logical
lane into a low/high pair, so radix counts never truncate at bit 32.

Two block shapes are held this way: a 1-D block of at most VLEN elements,
and a 2-D block of ONE row, `(1, n)` with n <= VLEN -- what Inductor's
reductions produce with one row per program (tnca.inductor), where an
embedding's token id is `[1, 1]` and the row of addresses it selects is
`[1, n]`. Element i of either is lane i. A non-64-bit value one of these
operations produces in a 2-D block is handed to the rest of the importer
as a one-row tile, which is how every 2-D value there is held.

A 64-bit offset meets a pointer only through a CHECKED narrowing: device
addresses are far below 2**31 elements, so the offset becomes the 32-bit
gather index ISA sec 9.1 takes, and a lane whose high word is not the sign
of its low word traps (TRAP_USER, NARROW_TRAP_CODE) rather than wrapping
into an address that happens to exist.

That premise holds wherever Inductor indexes in 32 bits, which it does
exactly when every numel and storage fits int32. A kernel it indexes in 64
bits (`index64`: a storage over 2**31 - 1 elements, e.g. a uint8 or FP8
storage over 2 GiB) addresses past it, so there a 1-D pointer's 64-bit
offset is not narrowed. When the offset is a lane-invariant 64-bit value
plus a 32-bit lane term (`xoffset + arange`, plus constants: the affine
`_form` tracked through the index arithmetic), the invariant part joins the
scalar address and the lane term stays the access's lane offset. Otherwise
every lane's address is formed in 64-bit scalar registers and the access
is made one lane at a time (`_lane64`).
"""

#: TRAP_USER code for a 64-bit offset that does not fit a 32-bit gather index.
NARROW_TRAP_CODE = 0xA5564


def _count(shape, vlen):
    """Elements of a block this file can hold, or None."""
    if isinstance(shape, int):
        return shape
    if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == 1 and shape[1] <= vlen:
        return shape[1]
    return None


def _vector(imp, src, typ):
    """A non-64-bit value of a held shape, as one vector (lane i = element i)."""
    from .frontend import _parse_type
    dims, _ = _parse_type(str(src.get_type()))
    v = imp._row_getter(src, 1, typ)(0) if isinstance(dims, tuple) else imp._value(src)
    if getattr(v, 'kind', None) == 'pred':
        # A mask is a predicate register, which has no lanes to extract:
        # as an integer it is 0 or 1 per lane (`x.to(tl.int64)` on a mask,
        # Inductor's "any finite score" in a softmax).
        v = imp.b.vselect(v, imp.b.vmovi(1), imp.b.vmovi(0))
    return v


def _publish(imp, dst, value, shape, elem):
    """Give a non-64-bit result to the rest of the importer in its form.

    A one-element 2-D result is made lane-uniform, as every one-column row
    in the importer is (see the 2-D gather in frontend._op_tt_load):
    tt.broadcast stretches such a row by naming it, not by copying lane 0.
    """
    from .tiles import Tile
    b = imp.b
    if shape == (1, 1):
        if value.kind == 'pred':
            lane = b.vextract(b.vselect(value, b.vmovi(1), b.vmovi(0)), 0)
            value = b.vcmp('ne', b.splat(lane), b.vmovi(0))
        else:
            value = b.splat(b.vextract(value, 0), dtype=elem)
    if isinstance(shape, tuple):
        if elem == 'i1':
            imp._tiles[dst.id()] = Tile('lazy', lambda r, v=value: v, 1, shape[1], 'i1')
        else:
            imp._tiles[dst.id()] = Tile('rows', [value], 1, shape[1], elem)
    else:
        imp._v[dst.id()] = value


def lower(imp, op):
    if _lower_rows(imp, op):
        return True
    from .frontend import _parse_type, Unsupported
    b = imp.b
    vlen = imp.target.vlen
    name = op.get_name()
    operands = [op.get_operand(i) for i in range(op.get_num_operands())]
    lane64 = getattr(imp, '_lane64', None)
    if lane64 and operands and operands[0].id() in lane64:
        if name == 'tt.addptr':
            # Another offset onto per-lane 64-bit offsets: summed per lane.
            base, lo, hi, n = lane64[operands[0].id()]
            plo, phi = b.vmovi(0), b.vmovi(0)
            for i in range(n):
                old = b.sbin('or', b.vextract(lo, i), b.sbini('shl', b.vextract(hi, i), 32))
                value = b.sbin('add', old, _lane_value(imp, operands[1], i))
                b.vinsert(plo, i, value)
                b.vinsert(phi, i, b.sbini('lshr', value, 32))
            lane64[op.get_result(0).id()] = (base, plo, phi, n)
            return True
        if name not in ('tt.load', 'tt.store'):
            raise Unsupported(f'{name} through a pointer with per-lane 64-bit offsets')
        _lane64_access(imp, op, operands)
        return True
    if name == 'tt.store':
        shape, elem = _parse_type(str(operands[1].get_type()))
        if elem == 'i64' and _count(shape, vlen) is not None:
            _store(imp, operands)
            return True
        return False
    if not op.get_num_results():
        return False
    dst = op.get_result(0)
    shape, elem = _parse_type(str(dst.get_type()))
    count = _count(shape, vlen)
    wide_result = count is not None and elem == 'i64'
    has_wide = any(x.id() in imp._wide for x in operands)
    if not wide_result and not has_wide:
        return False
    if name == 'arith.constant':
        return False  # constants remain lazy until read by lane()
    if wide_result:
        _track_form(imp, name, operands, dst, shape, count)

    def lane(src, i, signed=False):
        dims, typ = _parse_type(str(src.get_type()))
        if dims is None:
            return imp._value(src)
        if src.id() in imp._wide:
            lo, hi = imp._wide[src.id()]
            return b.sbin('or', b.vextract(lo, i), b.sbini('shl', b.vextract(hi, i), 32))
        pending = imp._pending_const.get(src.id())
        if pending:
            return b.const(int(pending[1]))
        value = b.vextract(_vector(imp, src, typ), i)
        if signed:
            value = b.sbini('ashr', b.sbini('shl', value, 32), 32)
        return value

    def output(fn, n):
        lo, hi = b.vmovi(0), b.vmovi(0)
        for i in range(n):
            value = fn(i)
            b.vinsert(lo, i, value)
            b.vinsert(hi, i, b.sbini('lshr', value, 32))
        imp._wide[dst.id()] = (lo, hi)

    if name == 'tt.load' and wide_result:
        _load(imp, operands, shape, dst)
        return True
    if name == 'tt.addptr' and has_wide and not wide_result:
        offs = operands[1]
        if offs.id() in imp._wide:
            oshape, _ = _parse_type(str(offs.get_type()))
            if index64(imp) and _rebase(imp, dst, operands[0], offs, oshape):
                return True
            lo, hi = imp._wide[offs.id()]
            _check_narrow(imp, lo, hi, _count(oshape, vlen))
            _publish(imp, offs, lo, oshape, 'i32')
        return False
    if name == 'arith.cmpi' and has_wide:
        from .frontend import _CMP_I
        if count is None:
            raise Unsupported(f'a 64-bit compare over a {shape} block')
        cc = _CMP_I[op.get_int_attr('predicate')]
        flags = b.vmovi(0)
        for i in range(count):
            b.vinsert(flags, i, b.scmp(cc, lane(operands[0], i, True), lane(operands[1], i, True)))
        _publish(imp, dst, b.vcmp('ne', flags, b.vmovi(0)), shape, 'i1')
        return True
    if name == 'arith.select' and wide_result:
        dims, _ = _parse_type(str(operands[0].get_type()))
        cond = imp._value(operands[0]) if dims is None else _vector(imp, operands[0], 'i1')
        if getattr(cond, 'kind', None) == 'pred':
            cond = b.vselect(cond, b.vmovi(1), b.vmovi(0))

        def pick(i):
            c = b.vextract(cond, i) if getattr(cond, 'kind', None) == 'vector' else cond
            yes, no = lane(operands[1], i, True), lane(operands[2], i, True)
            # no + c * (yes - no): branch-free, exact in 64-bit scalars.
            return b.sbin('add', no, b.sbin('mul', c, b.sbin('sub', yes, no)))
        output(pick, count)
        return True
    if name == 'tt.broadcast' and wide_result:
        sdims, _ = _parse_type(str(operands[0].get_type()))
        scount = _count(sdims, vlen)
        if scount == count:
            imp._wide[dst.id()] = imp._wide[operands[0].id()]
        elif scount == 1:
            output(lambda i: lane(operands[0], 0), count)
        else:
            raise Unsupported(f'a 64-bit broadcast from {sdims} to {shape}')
        return True
    if name in ('tt.reshape', 'tt.bitcast', 'tt.expand_dims') and wide_result and has_wide:
        imp._wide[dst.id()] = imp._wide[operands[0].id()]
    elif name in ('arith.extui', 'arith.extsi') and wide_result:
        output(lambda i: lane(operands[0], i, name == 'arith.extsi'), count)
    elif name == 'tt.splat' and wide_result:
        output(lambda i: lane(operands[0], 0), count)
    elif name == 'arith.trunci' and has_wide:
        _publish(imp, dst, imp._wide[operands[0].id()][0], shape, elem)
    elif name in ('tt.reduce', 'tt.scan'):
        _, body = imp._region_block(op)
        arithmetic = [x for x in body if x.get_name() not in ('tt.reduce.return','tt.scan.return')]
        if len(arithmetic) != 1 or arithmetic[0].get_name() != 'arith.addi':
            raise Unsupported('64-bit tensor reduction/scan requires integer addition')
        sdims, _ = _parse_type(str(operands[0].get_type()))
        n = _count(sdims, vlen)
        if n is None or (isinstance(sdims, tuple) and op.get_int_attr('axis') != 1):
            raise Unsupported(f'a 64-bit {name} over a {sdims} block')
        accum = b.const(0)
        lo, hi = b.vmovi(0), b.vmovi(0)
        for i in range(n):
            accum = b.sbin('add', accum, lane(operands[0], i))
            if name == 'tt.scan':
                b.vinsert(lo, i, accum)
                b.vinsert(hi, i, b.sbini('lshr', accum, 32))
        if name == 'tt.scan':
            imp._wide[dst.id()] = (lo, hi)
        elif count is not None:
            # A one-row block reduced along its row is a one-element block,
            # held like any other 64-bit block.
            output(lambda i: accum, count)
        else:
            imp._v[dst.id()] = accum
    elif name in _DIVISION and wide_result:
        signed, remainder = _DIVISION[name]
        d = _positive_constant(imp, operands[1])
        if d is None:
            raise Unsupported(f'64-bit {name} by a divisor that is not a positive constant')
        output(lambda i: _divide(b, lane(operands[0], i), d, signed, remainder), count)
    else:
        kinds={'arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
               'arith.andi':'and','arith.ori':'or','arith.xori':'xor',
               'arith.shli':'shl','arith.shrui':'lshr','arith.shrsi':'ashr',
               'arith.maxsi':'max','arith.minsi':'min'}
        if name not in kinds or not wide_result:
            raise Unsupported(f'64-bit tensor operation {name}')
        output(lambda i: b.sbin(kinds[name], lane(operands[0],i), lane(operands[1],i)), count)
    return True


# ---------------------------------------------------------------------------
# Exact 64-bit division by a constant
# ---------------------------------------------------------------------------

# arith op -> (signed, remainder). Triton's division truncates toward zero
# and its remainder takes the dividend's sign, as arith.divsi/remsi do.
_DIVISION = {'arith.divsi': (True, False), 'arith.divui': (False, False),
             'arith.remsi': (True, True), 'arith.remui': (False, True)}


def _positive_constant(imp, value):
    k = imp._scalar_const(value.id())
    if k is None:
        pending = imp._pending_const.get(value.id())
        k = int(pending[1]) if pending and pending[0] in ('scalar', 'vector') else None
    return k if k is not None and 0 < k < (1 << 63) else None


def _udiv_const(b, n, d):
    """floor(n / d) for an unsigned 64-bit scalar n and a constant
    1 <= d < 2**63, exact for every n: a shift for a power of two, else
    Granlund and Montgomery's multiply by a 64-bit reciprocal (s.mulh.u)."""
    if d & (d - 1) == 0:
        return n if d == 1 else b.sbini('lshr', n, d.bit_length() - 1)
    l = (d - 1).bit_length()                          # 2**(l-1) < d < 2**l
    m = ((1 << 64) * ((1 << l) - d)) // d + 1         # 1 <= m < 2**64
    t1 = b.sbin('mulhu', n, b.const(m))
    t = b.sbin('add', t1, b.sbini('lshr', b.sbin('sub', n, t1), 1))
    return b.sbini('lshr', t, l - 1)


def _divide(b, n, d, signed, remainder):
    if not signed:
        q = _udiv_const(b, n, d)
        return b.sbin('sub', n, b.sbini('mul', q, d)) if remainder else q
    sign = b.sbini('ashr', n, 63)                     # all ones when n < 0
    magnitude = b.sbin('sub', b.sbin('xor', n, sign), sign)
    q = _udiv_const(b, magnitude, d)
    q = b.sbin('sub', b.sbin('xor', q, sign), sign)   # truncated toward zero
    return b.sbin('sub', n, b.sbini('mul', q, d)) if remainder else q


# ---------------------------------------------------------------------------
# 64-bit indexing
# ---------------------------------------------------------------------------

def index64(imp):
    """Whether Inductor indexes this kernel in 64 bits.

    Its int64 index code converts the program id, `tl.program_id(0).
    to(tl.int64)` (pointwise and reduction kernels' xoffset, the templates'
    INDEX_DTYPE), and Inductor emits it only when a numel or a storage size
    exceeds 2**31 - 1 elements (SIMDScheduling.can_use_32bit_indexing).
    """
    cached = getattr(imp, '_index64', None)
    if cached is None:
        from .frontend import _parse_type
        pids = {o.get_result(0).id() for o in imp._all_ops
                if o.get_name() == 'tt.get_program_id'}
        cached = any(o.get_name() == 'arith.extsi' and o.get_operand(0).id() in pids
                     and _parse_type(str(o.get_result(0).get_type()))[1] == 'i64'
                     for o in imp._all_ops)
        imp._index64 = cached
    return cached


class _Term:
    """A synthetic element offset on a pointer's chain (frontend._address)."""
    _count = 0

    def __init__(self):
        _Term._count += 1
        self._id = ('addr64-term', _Term._count)

    def id(self):
        return self._id


def _form(imp, v):
    """(lane-invariant part, lane term) of a 1-D 64-bit integer value, or None.

    The value is `invariant + sext(lane term)` exactly. The invariant part
    is an int or a 64-bit scalar register; the lane term is None or a 32-bit
    TTIR vector value the frontend already holds (a make_range, or any i32
    vector, which a gather index sign-extends as extsi does)."""
    from .frontend import _parse_type
    forms = getattr(imp, '_wide_form', {})
    if v.id() in forms:
        return forms[v.id()]
    pending = imp._pending_const.get(v.id())
    if pending is not None and pending[0] in ('scalar', 'vector') and \
            isinstance(pending[1], int):
        return int(pending[1]), None
    dims, elem = _parse_type(str(v.get_type()))
    if dims is None and elem == 'i64':
        return imp._value(v), None
    return None


def _uop(imp, op, a, c):
    b = imp.b
    if isinstance(a, int) and isinstance(c, int):
        return {'add': a + c, 'sub': a - c, 'mul': a * c}[op]
    if op == 'mul':
        if isinstance(a, int):
            a, c = c, a
        return a if c == 1 else b.sbini('mul', a, c)
    if isinstance(c, int):
        if c == 0:
            return a
        return b.sbini('add', a, c if op == 'add' else -c)
    if isinstance(a, int):
        a = b.const(a)
    return b.sbin(op, a, c)


def _track_form(imp, name, operands, dst, shape, count):
    """Record `_form` of a 1-D 64-bit result in an index64 kernel."""
    from .frontend import _parse_type
    if not isinstance(shape, int) or not index64(imp):
        return
    f = None
    if name in ('arith.extsi', 'arith.extui'):
        src = operands[0]
        sdims, selem = _parse_type(str(src.get_type()))
        kind = imp._offsets.get(src.id(), (None, None))[0]
        if sdims == shape and selem == 'i32' and (name == 'arith.extsi' or kind == 'arange'):
            f = (0, src)
    elif name == 'tt.splat':
        src = operands[0]
        if _parse_type(str(src.get_type()))[0] is None:
            f = _form(imp, src)
            f = f if f is not None and f[1] is None else None
    elif name in ('arith.addi', 'arith.subi', 'arith.muli'):
        fa, fb = _form(imp, operands[0]), _form(imp, operands[1])
        if fa is not None and fb is not None:
            if name == 'arith.addi' and (fa[1] is None or fb[1] is None):
                f = (_uop(imp, 'add', fa[0], fb[0]), fa[1] if fa[1] is not None else fb[1])
            elif name == 'arith.subi' and fb[1] is None:
                f = (_uop(imp, 'sub', fa[0], fb[0]), fa[1])
            elif name == 'arith.muli':
                for x, y in ((fa, fb), (fb, fa)):
                    if isinstance(y[0], int) and y[1] is None:
                        if x[1] is None:
                            f = (_uop(imp, 'mul', x[0], y[0]), None)
                        elif y[0] == 1:
                            f = x
                        break
    elif name in ('tt.broadcast', 'tt.expand_dims', 'tt.reshape'):
        sdims, _ = _parse_type(str(operands[0].get_type()))
        if sdims == shape:
            f = _form(imp, operands[0])
    if f is not None:
        if not hasattr(imp, '_wide_form'):
            imp._wide_form = {}
        imp._wide_form[dst.id()] = f


def _rebase(imp, dst, base, offs, oshape):
    """`base + offs` for a 64-bit offset in an index64 kernel, without
    narrowing it to 32 bits. True if handled."""
    if isinstance(oshape, tuple):
        return len(oshape) == 2 and oshape[0] == 1 and \
            _rebase_tile(imp, dst, base, [imp._wide[offs.id()]], oshape[1])
    if not isinstance(oshape, int) or base.id() in imp._ptr2d or base.id() not in imp._ptr_base:
        return False
    f = getattr(imp, '_wide_form', {}).get(offs.id())
    if f is not None:
        invariant, lane = f
        chain = list(imp._ptr_chain.get(base.id(), []))
        if not (isinstance(invariant, int) and invariant == 0):
            term = _Term()
            if isinstance(invariant, int):
                imp._pending_const[term.id()] = ('scalar', invariant, 'i64')
            else:
                imp._v[term.id()] = invariant
            chain.append(term)
        if lane is not None:
            chain.append(lane)
        imp._ptr_base[dst.id()] = imp._ptr_base[base.id()]
        imp._ptr_chain[dst.id()] = chain
        imp._offsets[dst.id()] = ('ptr', None)
        return True
    lo, hi = imp._wide[offs.id()]
    if not hasattr(imp, '_lane64'):
        imp._lane64 = {}
    imp._lane64[dst.id()] = (base, lo, hi, oshape)
    return True


def _rebase_tile(imp, dst, base, pairs, cols):
    """A 2-D pointer plus a tile of 64-bit offsets (one (lo, hi) pair per
    row) in an index64 kernel. The offset of element [0, 0] joins the base
    address in 64 bits; each element's offset from it is the 32-bit gather
    index the tile is addressed by, checked to fit (a tile spanning 2**31
    elements traps, NARROW_TRAP_CODE). True if handled."""
    from .frontend import _DTYPE_BYTES
    from .tiles import Affine2D, Tile
    spec = imp._ptr2d.get(base.id())
    if spec is None:
        return False
    name, elem, acc = spec
    esize = _DTYPE_BYTES.get(elem)
    if esize is None:
        return False
    b = imp.b
    l0, h0 = b.vextract(pairs[0][0], 0), b.vextract(pairs[0][1], 0)
    anchor = b.sbin('or', l0, b.sbini('shl', h0, 32))
    first_lo, first_hi = b.splat(l0), b.splat(h0)
    rows = []
    for lo, hi in pairs:
        dlo = b.vbin('sub', lo, first_lo, dtype='i32')
        borrow = b.vselect(b.vcmp('ltu', lo, first_lo), b.vmovi(1), b.vmovi(0))
        dhi = b.vbin('sub', b.vbin('sub', hi, first_hi, dtype='i32'), borrow, dtype='i32')
        _check_narrow(imp, dlo, dhi, cols)
        rows.append(dlo)
    if esize > 1:
        anchor = b.sbini('shl', anchor, esize.bit_length() - 1)
    at = b.sbin('add', imp._arg_value(name), anchor, comment='a 64-bit tile origin')
    label = imp._control_label('ptr_rebased')
    imp._args[len(imp._args)] = (label, at)
    residual = Tile('rows', tuple(rows), len(rows), cols, 'i32')
    if isinstance(acc, Affine2D) and not (acc.row_stride or acc.col_stride or acc.const
                                          or acc.dynamic):
        combined = residual
    else:
        left, right = imp._offset_row(acc), imp._offset_row(residual)
        combined = imp._materialise(lambda r: b.vbin('add', left(r), right(r), dtype='i32'),
                                    len(rows), cols, 'i32')
    imp._ptr2d[dst.id()] = (label, elem, combined)
    return True


def _lane_value(imp, src, i):
    """Lane i of a 1-D integer value as an exact 64-bit scalar (sign-extended
    when the value is narrower)."""
    from .frontend import _parse_type
    b = imp.b
    dims, typ = _parse_type(str(src.get_type()))
    if dims is None:
        v = imp._value(src)
        return v if typ == 'i64' else b.sbini('ashr', b.sbini('shl', v, 32), 32)
    if src.id() in imp._wide:
        lo, hi = imp._wide[src.id()]
        return b.sbin('or', b.vextract(lo, i), b.sbini('shl', b.vextract(hi, i), 32))
    pending = imp._pending_const.get(src.id())
    if pending:
        return b.const(int(pending[1]))
    v = b.vextract(_vector(imp, src, typ), i)
    return b.sbini('ashr', b.sbini('shl', v, 32), 32)


def _lane64_access(imp, op, operands):
    """A load or store through a pointer whose lanes carry independent 64-bit
    offsets: each lane's address is base + offset * size in scalar
    registers, and the access is one single-lane gather or scatter per lane,
    under the access's own mask."""
    from .frontend import _parse_type, _DTYPE_BYTES, _SIZE_SUFFIX, _load_suffix, Unsupported
    b = imp.b
    base, lo, hi, count = imp._lane64[operands[0].id()]
    addr, _size, elem, kind, _index = imp._address(base)
    if kind != 'broadcast':
        raise Unsupported('per-lane 64-bit offsets on a pointer that already varies by lane')
    esize = _DTYPE_BYTES.get(elem)
    if esize is None:
        raise Unsupported(f'a per-lane 64-bit access of {elem}')
    load = op.get_name() == 'tt.load'
    mask_value = (operands[1] if len(operands) > 1 else None) if load else \
        (operands[2] if len(operands) > 2 else None)
    mask = imp._shape_mask(count, imp._value(mask_value) if mask_value is not None else None)
    wide = (_parse_type(str(op.get_result(0).get_type()))[1] if load else
            _parse_type(str(operands[1].get_type()))[1]) == 'i64'
    if wide:
        sizes = [('w', 0), ('w', 4)]
    else:
        sizes = [(_load_suffix(_SIZE_SUFFIX[esize], elem) if load else _SIZE_SUFFIX[esize], 0)]
    if load:
        other = imp._const.get(operands[2].id(), 0) if len(operands) > 2 else 0
        others = [other & 0xFFFFFFFF, (other >> 32) & 0xFFFFFFFF] if wide else [other]
        dtype = 'i32' if wide else elem
        acc = [None] * len(sizes)
    else:
        value = operands[1]
        if wide:
            if value.id() in imp._wide:
                parts = list(imp._wide[value.id()])
            else:
                pending = imp._pending_const.get(value.id())
                if not pending:
                    raise Unsupported('a per-lane 64-bit store of a value this does not hold')
                c = int(pending[1])
                parts = [b.vmovi(c & 0xFFFFFFFF), b.vmovi((c >> 32) & 0xFFFFFFFF)]
        else:
            parts = [imp._value(value)]
    zero = b.vmovi(0)
    for i in range(count):
        off = b.sbin('or', b.vextract(lo, i), b.sbini('shl', b.vextract(hi, i), 32))
        if esize > 1:
            off = b.sbini('shl', off, esize.bit_length() - 1)
        at = b.sbin('add', addr, off, comment=f'lane {i}: a 64-bit address')
        sel = b.vcmp('eq', b.arange(), b.splat(b.const(i)))
        active = sel if mask is None else b.pbin('and', mask, sel)
        for k, (size, disp) in enumerate(sizes):
            where = at if not disp else b.sbini('add', at, disp)
            if load:
                g = b.gather(where, zero, mask=active, other=others[k], dtype=dtype, size=size)
                acc[k] = g if acc[k] is None else b.vselect(sel, g, acc[k])
            else:
                b.scatter(where, zero, parts[k], mask=active, size=size)
    if load:
        rid = op.get_result(0).id()
        if wide:
            imp._wide[rid] = (acc[0], acc[1])
        else:
            imp._v[rid] = acc[0]


def _check_narrow(imp, lo, hi, count):
    """Trap unless every one of the first `count` lanes' (hi, lo) is lo
    sign-extended."""
    b = imp.b
    sign = b.vbin('ashr', lo, b.vmovi(31))
    bad = b.vselect(b.vcmp('ne', hi, sign), b.vmovi(1), b.vmovi(0))
    any_bad = b.reduce('or', bad, dtype='i32', mask=imp._shape_mask(count or imp.target.vlen, None))
    ok = imp._control_label('narrow_ok')
    b.branch_nz(b.scmp('eq', any_bad, b.const(0)), ok)
    b._emit('trap', [], [], code=NARROW_TRAP_CODE)
    b.label(ok)


def _words(imp, ptr, mask_value, shape):
    """(base address, 32-bit word index vector, mask) for an access of
    8-byte elements: element i is the word pair (2i, 2i+1)."""
    from .frontend import _parse_type
    b = imp.b
    count = _count(shape, imp.target.vlen)
    spec = imp._ptr2d.get(ptr.id())
    if spec is not None:
        # A one-row 2-D pointer: its element offsets are row 0 of its index.
        name, _elem, aff = spec
        addr = imp._arg_value(name)
        idx = imp._offset_row(aff)(0)
        mask = None
        if mask_value is not None:
            mdims, _ = _parse_type(str(mask_value.get_type()))
            mask = (imp._row_getter(mask_value, 1, 'i1')(0) if isinstance(mdims, tuple)
                    else imp._value(mask_value))
    else:
        addr, _size, _elem, kind, index = imp._address(ptr)
        mask = imp._value(mask_value) if mask_value is not None else None
        if kind == 'broadcast':
            idx = b.vmovi(0)
        elif kind == 'contiguous':
            idx = b.arange()
        else:
            idx = index
    mask = imp._shape_mask(count, mask)
    imp.target.require('gather_scatter', 'a 64-bit element as two words')
    return addr, b.vbin('shl', idx, b.vmovi(1)), mask


def _load(imp, operands, shape, dst):
    b = imp.b
    mask_value = operands[1] if len(operands) > 1 else None
    addr, widx, mask = _words(imp, operands[0], mask_value, shape)
    other = imp._const.get(operands[2].id(), 0) if len(operands) > 2 else 0
    lo = b.gather(addr, widx, mask=mask, other=other & 0xFFFFFFFF, dtype='i32', size='w')
    hi = b.gather(b.sbini('add', addr, 4), widx, mask=mask,
                  other=(other >> 32) & 0xFFFFFFFF, dtype='i32', size='w')
    imp._wide[dst.id()] = (lo, hi)


def _store(imp, operands):
    from .frontend import _parse_type
    b = imp.b
    shape, _ = _parse_type(str(operands[1].get_type()))
    mask_value = operands[2] if len(operands) > 2 else None
    addr, widx, mask = _words(imp, operands[0], mask_value, shape)
    val = operands[1]
    if val.id() in imp._wide:
        lo, hi = imp._wide[val.id()]
    else:
        pending = imp._pending_const.get(val.id())
        if not pending:
            raise ValueError(f'no 64-bit value for {val.get_type()}')
        c = int(pending[1])
        lo, hi = b.vmovi(c & 0xFFFFFFFF), b.vmovi((c >> 32) & 0xFFFFFFFF)
    b.scatter(addr, widx, lo, mask=mask, size='w')
    b.scatter(b.sbini('add', addr, 4), widx, hi, mask=mask, size='w')


class _RowValue:
    def __init__(self, value, row, shape, type_override=None):
        self.value, self.row, self.shape = value, row, shape
        self.type_override = type_override

    def id(self):
        return ('wide-row', self.value.id(), self.row)

    def get_type(self):
        import re
        if self.type_override is not None:
            return self.type_override
        return re.sub(r'^tensor<\d+x\d+x', f'tensor<1x{self.shape[1]}x', str(self.value.get_type()))

    def __getattr__(self, name):
        return getattr(self.value, name)


class _RowOp:
    def __init__(self, op, values):
        self.op, self.values = op, values

    def get_operand(self, i):
        return self.values[self.op.get_operand(i).id()]

    def get_result(self, i):
        return self.values[self.op.get_result(i).id()]

    def __getattr__(self, name):
        return getattr(self.op, name)


def _lower_rows(imp, op):
    """Exact low/high pairs per row for bounded 2-D integer arithmetic.

    Narrowing of address offsets still checks every high word before the
    ordinary tile pointer lowering consumes the low-word rows.
    """
    from .frontend import _parse_type, Unsupported
    from .tiles import Tile
    if getattr(imp, '_in_wide_row', False):
        return False
    values = ([op.get_operand(i) for i in range(op.get_num_operands())] +
              [op.get_result(i) for i in range(op.get_num_results())])
    wide_tiles = getattr(imp, '_wide_tiles', {})
    shapes = [(v, *_parse_type(str(v.get_type()))) for v in values]
    if not any((isinstance(shape, tuple) and shape[0] > 1 and elem == 'i64') or
               v.id() in wide_tiles for v, shape, elem in shapes):
        return False
    if any(len(shape) != 2 for _, shape, _ in shapes if isinstance(shape, tuple)):
        raise Unsupported('multi-row 64-bit lowering requires rank 2')
    rows = max(shape[0] for _, shape, _ in shapes if isinstance(shape, tuple))
    if rows > imp.target.vlen:
        raise Unsupported('64-bit tile row count exceeds VLEN')
    name = op.get_name()
    if name == 'arith.constant':
        return False
    if name in ('tt.reshape', 'tt.bitcast'):
        source_shape, _ = _parse_type(str(op.get_operand(0).get_type()))
        result_shape, _ = _parse_type(str(op.get_result(0).get_type()))
        if source_shape != result_shape:
            raise Unsupported('multi-row 64-bit reshape requires an explicit lane mapping')
    if name in ('tt.reduce', 'tt.scan'):
        raise Unsupported(f'multi-row 64-bit {name} requires row memory/fold state')
    if name == 'tt.addptr':
        offs = op.get_operand(1)
        shape, _ = _parse_type(str(offs.get_type()))
        if offs.id() not in wide_tiles:
            return False
        if index64(imp) and _rebase_tile(imp, op.get_result(0), op.get_operand(0),
                                         wide_tiles[offs.id()], shape[1]):
            return True
        lows = []
        for lo, hi in wide_tiles[offs.id()]:
            _check_narrow(imp, lo, hi, shape[1])
            lows.append(lo)
        imp._tiles[offs.id()] = Tile('rows', tuple(lows), rows, shape[1], 'i32')
        return False
    outputs = [[] for _ in range(op.get_num_results())]
    imp._in_wide_row = True
    try:
        for row in range(rows):
            mapped = {}
            for v, shape, elem in shapes:
                if not isinstance(shape, tuple):
                    mapped[v.id()] = v
                    continue
                use_row = row if shape[0] > 1 else 0
                view = _RowValue(v, use_row, shape)
                mapped[v.id()] = view
                if v.id() in imp._ptr2d:
                    base, pointee, offset = imp._ptr2d[v.id()]
                    index = imp._offset_row(offset)(use_row)
                    imp._ptr2d[view.id()] = (base, pointee,
                        Tile('lazy', lambda r, index=index: index, 1, shape[1], 'i32'))
                if v.id() in imp._ptr_base:
                    imp._ptr_base[view.id()] = imp._ptr_base[v.id()]
                if v.id() in wide_tiles:
                    imp._wide[view.id()] = wide_tiles[v.id()][use_row]
                elif v.id() in imp._wide:
                    imp._wide[view.id()] = imp._wide[v.id()]
                elif v.id() in imp._pending_const:
                    imp._pending_const[view.id()] = imp._pending_const[v.id()]
                elif (v.id() in imp._tiles or v.id() in imp._v or
                      v.id() in imp._affine):
                    get = imp._row_getter(v, shape[0], elem)
                    val = get(use_row)
                    imp._tiles[view.id()] = Tile('lazy', lambda r, val=val: val, 1, shape[1], elem)
            # Expanding a row-index vector into a column selects its lane.
            if name == 'tt.expand_dims' and op.get_int_attr('axis') == 1:
                src = op.get_operand(0)
                if src.id() in imp._wide:
                    lo, hi = imp._wide[src.id()]
                    view = _RowValue(src, row, (rows, 1), type_override='tensor<1xi64>')
                    imp._wide[view.id()] = tuple(imp.b.splat(imp.b.vextract(x, row)) for x in (lo, hi))
                    mapped[src.id()] = view
            view_op = _RowOp(op, mapped)
            if not lower(imp, view_op):
                raise Unsupported(f'multi-row 64-bit operation {name}')
            for i in range(op.get_num_results()):
                dst = view_op.get_result(i)
                shape, elem = _parse_type(str(dst.get_type()))
                if elem == 'i64':
                    outputs[i].append(imp._wide[dst.id()])
                else:
                    outputs[i].append(imp._row_getter(dst, 1, elem)(0))
    finally:
        imp._in_wide_row = False
    for i, result in enumerate(outputs):
        dst = op.get_result(i)
        shape, elem = _parse_type(str(dst.get_type()))
        if elem == 'i64':
            wide_tiles[dst.id()] = result
        else:
            imp._tiles[dst.id()] = Tile('rows', tuple(result), rows, shape[1], elem)
    imp._wide_tiles = wide_tiles
    return True
