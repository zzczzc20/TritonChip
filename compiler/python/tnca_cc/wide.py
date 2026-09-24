"""Exact 64-bit integer tensor operations over pairs of 32-bit vectors.

Scalar ISA operations retain all 64 bits. Extract/insert packs each logical
lane into a low/high pair, so radix counts never truncate at bit 32.
"""
def lower(imp, op):
    from .frontend import _parse_type, Unsupported
    b = imp.b
    name = op.get_name()
    operands = [op.get_operand(i) for i in range(op.get_num_operands())]
    if not op.get_num_results():
        return False
    dst = op.get_result(0)
    shape, elem = _parse_type(str(dst.get_type()))
    wide_result = isinstance(shape, int) and elem == 'i64'
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
        value = b.vextract(imp._value(src), i)
        if signed:
            value = b.sbini('ashr', b.sbini('shl', value, 32), 32)
        return value

    def output(fn, count):
        lo, hi = b.vmovi(0), b.vmovi(0)
        for i in range(count):
            value = fn(i)
            b.vinsert(lo, i, value)
            b.vinsert(hi, i, b.sbini('lshr', value, 32))
        imp._wide[dst.id()] = (lo, hi)

    if name in ('tt.reshape', 'tt.bitcast') and wide_result and has_wide:
        imp._wide[dst.id()] = imp._wide[operands[0].id()]
    elif name in ('arith.extui', 'arith.extsi') and wide_result:
        output(lambda i: lane(operands[0], i, name == 'arith.extsi'), shape)
    elif name == 'tt.splat' and wide_result:
        output(lambda i: lane(operands[0], 0), shape)
    elif name == 'arith.trunci' and has_wide:
        imp._v[dst.id()] = imp._wide[operands[0].id()][0]
    elif name in ('tt.reduce', 'tt.scan'):
        _, body = imp._region_block(op)
        arithmetic = [x for x in body if x.get_name() not in ('tt.reduce.return','tt.scan.return')]
        if len(arithmetic) != 1 or arithmetic[0].get_name() != 'arith.addi':
            raise Unsupported('64-bit tensor reduction/scan requires integer addition')
        count, _ = _parse_type(str(operands[0].get_type()))
        accum = b.const(0)
        lo, hi = b.vmovi(0), b.vmovi(0)
        for i in range(count):
            accum = b.sbin('add', accum, lane(operands[0], i))
            if name == 'tt.scan':
                b.vinsert(lo, i, accum)
                b.vinsert(hi, i, b.sbini('lshr', accum, 32))
        if name == 'tt.scan':
            imp._wide[dst.id()] = (lo, hi)
        else:
            imp._v[dst.id()] = accum
    else:
        kinds={'arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
               'arith.andi':'and','arith.ori':'or','arith.xori':'xor',
               'arith.shli':'shl','arith.shrui':'lshr','arith.shrsi':'ashr'}
        if name not in kinds or not wide_result:
            raise Unsupported(f'64-bit tensor operation {name}')
        output(lambda i: b.sbin(kinds[name], lane(operands[0],i), lane(operands[1],i)), shape)
    return True
