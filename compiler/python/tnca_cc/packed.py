"""Small multidimensional tensors packed in one physical vector.

A reshape retains row-major element order. Broadcasts and axis reductions
explicitly permute lanes; shape metadata is never discarded for those ops.
This is also how Triton's inlined topk sorting network reaches the ISA.
"""
import math
from .ir import PRED


def shape(value):
    from .frontend import _parse_type
    dims, elem = _parse_type(str(value.get_type()))
    return ((dims,) if isinstance(dims, int) else dims), elem


def coords(index, dims):
    result = []
    for dim in reversed(dims):
        result.append(index % dim)
        index //= dim
    return list(reversed(result))


def index(at, dims):
    result = 0
    for a, dim in zip(at, dims):
        result = result * dim + a
    return result


def value(imp, src):
    if src.id() in imp._v:
        return imp._v[src.id()]
    pending = imp._pending_const.get(src.id())
    if pending:
        v = imp.b.vmovi(pending[1], dtype=pending[2])
        imp._v[src.id()] = v
        return v
    raise ValueError(f'no packed value for {src.get_type()}')


def permute(imp, src, indices, dtype):
    b = imp.b
    if src.kind == PRED:
        src = b.vselect(src, b.vmovi(1), b.vmovi(0))
    # Constant permutation via extract/insert needs no memory ordering.
    dst = b.vmovi(0, dtype=dtype)
    for lane, origin in enumerate(indices):
        b.vinsert(dst, lane, b.vextract(src, origin))
    return b.vcmp('ne', dst, b.vmovi(0)) if dtype == 'i1' else dst


def lower(imp, op):
    name = op.get_name()
    operands = [op.get_operand(i) for i in range(op.get_num_operands())]
    if not any(v.id() in imp._packed for v in operands):
        return False
    if name in ('tt.store', 'tt.return', 'scf.yield'):
        return False
    if not op.get_num_results():
        return False
    res = op.get_result(0)
    dims, elem = shape(res)
    if dims is None or math.prod(dims) > imp.target.vlen:
        return False
    b = imp.b
    xs = [value(imp, v) for v in operands]
    count = math.prod(dims)
    from .frontend import _INT_BIN, _FP_BIN, _CMP_I, _CMP_F
    binary = dict(_INT_BIN, **_FP_BIN)
    binary.update({'arith.maxnumf': 'max', 'arith.minnumf': 'min',
                   'arith.xori': 'xor', 'arith.shrsi': 'ashr', 'arith.shrui': 'lshr'})
    if name in binary:
        result = (b.pbin(binary[name], *xs) if elem == 'i1' else
                  b.vbin(binary[name], *xs, dtype=elem))
    elif name in ('tt.reshape', 'tt.expand_dims', 'tt.bitcast'):
        result = xs[0]
    elif name == 'tt.broadcast':
        srcdims, _ = shape(operands[0])
        indices = [index([a if d != 1 else 0 for a, d in zip(coords(i, dims), srcdims)], srcdims)
                   for i in range(count)]
        result = permute(imp, xs[0], indices, elem)
    elif name == 'tt.reduce':
        srcdims, _ = shape(operands[0])
        axis = op.get_int_attr('axis')
        kind = imp._combinator(op)
        result = None
        for j in range(srcdims[axis]):
            indices = []
            for i in range(count):
                at = coords(i, dims)
                at.insert(axis, j)
                indices.append(index(at, srcdims))
            part = permute(imp, xs[0], indices, elem)
            result = part if result is None else b.vbin(kind, result, part, dtype=elem)
    elif name in ('arith.cmpi', 'arith.cmpf'):
        table = _CMP_I if name == 'arith.cmpi' else _CMP_F
        result = b.vcmp(table[op.get_int_attr('predicate')], *xs)
    elif name == 'arith.select':
        result = b.vselect(*xs)
    elif name in ('arith.extui', 'arith.extsi'):
        result = b.vselect(xs[0], b.vmovi(1), b.vmovi(0)) if xs[0].kind == PRED else xs[0]
    else:
        return False
    imp._v[res.id()] = result
    imp._packed.add(res.id())
    return True
