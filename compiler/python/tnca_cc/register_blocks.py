"""Split wide 1-D and one-row TTIR values into VLEN register chunks.

Views preserve scalar SSA identities and give every vector chunk its own
identity and type. Existing typed lowering handles each chunk, including
masked memory accesses and checked i64 addressing. This is a shared
compiler transformation; the incoming standard-path TTIR stays unchanged.
"""
import re


def _width(value):
    match = re.fullmatch(r'tensor<(?:1x)?(\d+)x([^\d].*)>', str(value.get_type()))
    return int(match.group(1)) if match else None


class _ValueView:
    def __init__(self, value, chunk, vlen):
        self.value, self.chunk, self.vlen = value, chunk, vlen

    def id(self):
        width = _width(self.value)
        return ('register-chunk', self.value.id(), self.chunk) if width and width > self.vlen else self.value.id()

    def get_type(self):
        text = str(self.value.get_type())
        width = _width(self.value)
        if width and width > self.vlen:
            prefix = 'tensor<1x' if text.startswith(f'tensor<1x{width}x') else 'tensor<'
            return re.sub(r'^tensor<(?:1x)?\d+x', f'{prefix}{self.vlen}x', text)
        return self.value.get_type()

    def __getattr__(self, name):
        return getattr(self.value, name)


class _BlockView:
    def __init__(self, block, chunk, vlen):
        self.block, self.chunk, self.vlen = block, chunk, vlen

    def get_argument(self, i):
        return _ValueView(self.block.get_argument(i), self.chunk, self.vlen)

    def __getattr__(self, name):
        return getattr(self.block, name)


class _OpView:
    _register_chunk = True
    def __init__(self, op, chunk, vlen):
        self.op, self.chunk, self.vlen = op, chunk, vlen

    def get_operand(self, i):
        return _ValueView(self.op.get_operand(i), self.chunk, self.vlen)

    def get_result(self, i):
        return _ValueView(self.op.get_result(i), self.chunk, self.vlen)

    def get_block(self):
        return _BlockView(self.op.get_block(), self.chunk, self.vlen)

    def get_int_attr(self, name):
        value = self.op.get_int_attr(name)
        if (self.op.get_name() == 'tt.make_range' and name in ('start', 'end') and
                (_width(self.op.get_result(0)) or 0) > self.vlen):
            start = self.op.get_int_attr('start') or 0
            return start + self.chunk * self.vlen + (self.vlen if name == 'end' else 0)
        return value

    def __getattr__(self, name):
        return getattr(self.op, name)

    def __str__(self):
        return str(self.op)


# Side effects and control flow require explicit semantics. Unrecognised
# operations fail rather than accidentally executing a region once per chunk.
_ALLOWED = {'tt.make_range', 'tt.splat', 'tt.addptr', 'tt.load', 'tt.store',
            'tt.broadcast', 'tt.bitcast', 'tt.fp_to_fp', 'tt.clamp', 'tt.reduce',
            'tt.expand_dims', 'tt.reshape'}


def lower(imp, op):
    if getattr(imp, '_in_register_chunk', False):
        return False
    values = ([op.get_operand(i) for i in range(op.get_num_operands())] +
              [op.get_result(i) for i in range(op.get_num_results())])
    widths = {_width(v) for v in values if _width(v) and _width(v) > imp.target.vlen}
    if not widths:
        return False
    from .frontend import Unsupported
    if len(widths) != 1:
        raise Unsupported('register-block operation with mismatched widths')
    width = widths.pop()
    if width % imp.target.vlen or width > 128:
        raise Unsupported(f'register block width {width}; expected a VLEN multiple up to 128')
    widths_seen = getattr(imp, '_register_block_widths', set())
    widths_seen.add(width)
    imp._register_block_widths = widths_seen
    name = op.get_name()
    if name == 'scf.for':
        return _loop(imp, op, width)
    if name == 'scf.if':
        return _if(imp, op, width)
    if name not in _ALLOWED and not name.startswith(('arith.', 'math.')):
        raise Unsupported(f'wide register block operation {name}')
    if name == 'tt.reduce':
        return _reduce(imp, op, width)
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            view = _OpView(op, chunk, imp.target.vlen)
            # Decimal/hex float literals are indexed during the original
            # module walk; the virtual SSA identities retain that literal.
            for i in range(op.get_num_results()):
                original = op.get_result(i).id()
                if original in (imp._printed_consts or {}):
                    imp._printed_consts[view.get_result(i).id()] = imp._printed_consts[original]
            pure_predicate = (width // imp.target.vlen >= max(1, imp.target.pregs_per_rctx // 8) and
                              name.startswith('arith.') and
                              name != 'arith.constant' and op.get_num_results() == 1 and
                              not str(op.get_result(0).get_type()).startswith('tensor<1x') and
                              str(op.get_result(0).get_type()).endswith('xi1>'))
            if pure_predicate:
                # Recompute pure lane masks at each use. Original operand
                # SSA values remain live; memory accesses keep their order.
                # Reserve predicate capacity for temporary comparisons in
                # folds as well as the block boundary and select masks.
                remat = getattr(imp, '_register_predicates', {})
                remat[view.get_result(0).id()] = view
                imp._register_predicates = remat
            else:
                imp._dispatch(view)
    finally:
        imp._in_register_chunk = False
    return True


def _reduce(imp, op, width):
    from .frontend import Unsupported, _parse_type
    from .ir import SCALAR
    from .tiles import Tile
    if op.get_num_operands() != 1 or op.get_num_results() != 1:
        return _reduce_general(imp, op, width)
    shape, elem = _parse_type(str(op.get_operand(0).get_type()))
    if elem in ('i64', 'u64'):
        raise Unsupported('wide int64 reduction requires a cross-chunk carry fold')
    if op.get_int_attr('axis') != (1 if isinstance(shape, tuple) else 0):
        raise Unsupported('wide reduction axis must be the contiguous lane axis')
    kind = imp._combinator(op)
    seen = getattr(imp, '_register_reductions', set())
    seen.add((width, kind, op.get_int_attr('axis')))
    imp._register_reductions = seen
    rid = op.get_result(0).id()
    partials = []
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            imp._dispatch(_OpView(op, chunk, imp.target.vlen))
            if rid in imp._tiles:
                tile = imp._tiles[rid]
                partials.append(imp._tile_row_getter(tile, elem)(0))
            else:
                partials.append(imp._v[rid])
    finally:
        imp._in_register_chunk = False
    scalar = partials[0].kind == SCALAR
    vectors = [imp.b.splat(value, dtype=elem) if value.kind == SCALAR else value
               for value in partials]
    # A balanced tree retains Triton's reduction association across chunks.
    while len(vectors) > 1:
        vectors = [imp.b.vbin(kind[:3] if kind.endswith('_prop') else kind,
                             vectors[i], vectors[i + 1], dtype=elem,
                             fmm='prop' if kind.endswith('_prop') else None)
                   for i in range(0, len(vectors) - 1, 2)] + (vectors[-1:] if len(vectors) % 2 else [])
    combined = imp.b.vextract(vectors[0], 0, dtype=elem) if scalar else vectors[0]
    if isinstance(shape, tuple):
        imp._tiles[rid] = Tile('col', (combined,), 1, 1, elem)
        imp._v.pop(rid, None)
    else:
        imp._v[rid] = combined
    return True


class _ExpandedBlock:
    def __init__(self, block, layout, vlen):
        self.block, self.layout, self.vlen = block, layout, vlen

    def get_argument(self, i):
        if i == 0:
            return self.block.get_argument(0)
        original, chunk = self.layout[i - 1]
        return _ValueView(self.block.get_argument(original + 1), chunk, self.vlen)

    def get_num_arguments(self):
        return 1 + len(self.layout)

    def __getattr__(self, name):
        return getattr(self.block, name)


class _LoopBodyView:
    def __init__(self, op, layout, vlen):
        self.op, self.layout, self.vlen = op, layout, vlen

    def get_block(self):
        return _ExpandedBlock(self.op.get_block(), self.layout, self.vlen)

    def get_num_operands(self):
        return len(self.layout) if self.op.get_name() == 'scf.yield' else self.op.get_num_operands()

    def get_operand(self, i):
        if self.op.get_name() == 'scf.yield':
            original, chunk = self.layout[i]
            return _ValueView(self.op.get_operand(original), chunk, self.vlen)
        return self.op.get_operand(i)

    def __getattr__(self, name):
        return getattr(self.op, name)


class _ExpandedLoop:
    def __init__(self, op, layout, vlen):
        self.op, self.layout, self.vlen = op, layout, vlen

    def get_num_results(self):
        return len(self.layout)

    def get_num_operands(self):
        return 3 + len(self.layout)

    def get_result(self, i):
        original, chunk = self.layout[i]
        return _ValueView(self.op.get_result(original), chunk, self.vlen)

    def get_operand(self, i):
        if i < 3:
            return self.op.get_operand(i)
        original, chunk = self.layout[i - 3]
        return _ValueView(self.op.get_operand(original + 3), chunk, self.vlen)

    def __getattr__(self, name):
        return getattr(self.op, name)


def _loop(imp, op, width):
    # Expand carried state, not the control-flow loop. Each body operation
    # still runs in original order across its register chunks. This retains
    # load/store and alias ordering between iterations, including softmax's
    # writes of the realised exp intermediate.
    layout = []
    for i in range(op.get_num_results()):
        count = (_width(op.get_result(i)) or 0) // imp.target.vlen
        count = count if count > 1 else 1
        layout.extend((i, chunk) for chunk in range(count))
    rid = op.get_region(0).id()
    body = imp._bodies.get(rid, [])
    imp._bodies[rid] = [_LoopBodyView(inner, layout, imp.target.vlen) for inner in body]
    try:
        imp._op_scf_for(_ExpandedLoop(op, layout, imp.target.vlen))
    finally:
        imp._bodies[rid] = body
    return True


def _if(imp, op, width):
    regions = {op.get_region(i).id(): imp._bodies.get(op.get_region(i).id(), [])
               for i in range(op.get_num_regions())}
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            for rid, body in regions.items():
                views = [_OpView(inner, chunk, imp.target.vlen) for inner in body]
                imp._bodies[rid] = views
                for view, inner in zip(views, body):
                    for i in range(inner.get_num_results()):
                        original = inner.get_result(i).id()
                        if original in (imp._printed_consts or {}):
                            imp._printed_consts[view.get_result(i).id()] = imp._printed_consts[original]
            imp._op_scf_if(_OpView(op, chunk, imp.target.vlen))
    finally:
        imp._bodies.update(regions)
        imp._in_register_chunk = False
    return True


def _reduce_general(imp, op, width):
    from .frontend import Unsupported, _parse_type
    if op.get_num_operands() != op.get_num_results():
        raise Unsupported('wide fold has mismatched inputs and outputs')
    for i in range(op.get_num_operands()):
        shape, elem = _parse_type(str(op.get_operand(i).get_type()))
        if elem in ('i64', 'u64'):
            raise Unsupported('wide multi-output fold requires explicit 64-bit carry state')
        if op.get_int_attr('axis') != (1 if isinstance(shape, tuple) else 0):
            raise Unsupported('wide fold axis must be the lane axis')
    partials = []
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            view = _OpView(op, chunk, imp.target.vlen)
            imp._fold(view, keep_partials=False)
            partials.append([imp._v[op.get_result(i).id()] for i in range(op.get_num_results())])
    finally:
        imp._in_register_chunk = False
    while len(partials) > 1:
        partials = [imp._eval_region(op, partials[i] + partials[i+1])
                    for i in range(0, len(partials)-1, 2)] + (partials[-1:] if len(partials)%2 else [])
    for i, value in enumerate(partials[0]):
        imp._v[op.get_result(i).id()] = value
    seen = getattr(imp, '_register_reductions', set())
    seen.add((width, 'body-fold', op.get_int_attr('axis')))
    imp._register_reductions = seen
    return True
