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
    from .frontend import _parse_type, Unsupported
    b = imp.b
    vlen = imp.target.vlen
    name = op.get_name()
    operands = [op.get_operand(i) for i in range(op.get_num_operands())]
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
    else:
        kinds={'arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
               'arith.andi':'and','arith.ori':'or','arith.xori':'xor',
               'arith.shli':'shl','arith.shrui':'lshr','arith.shrsi':'ashr'}
        if name not in kinds or not wide_result:
            raise Unsupported(f'64-bit tensor operation {name}')
        output(lambda i: b.sbin(kinds[name], lane(operands[0],i), lane(operands[1],i)), count)
    return True


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
