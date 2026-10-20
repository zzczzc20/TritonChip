"""Split wide 1-D and bounded-row TTIR values into VLEN register chunks.

Views preserve scalar SSA identities and give every vector chunk its own
identity and type. Existing typed lowering handles each chunk, including
masked memory accesses and checked i64 addressing. This is a shared
compiler transformation; the incoming standard-path TTIR stays unchanged.
"""
import re


def _width(value):
    match = re.fullmatch(r'tensor<(?:\d+x)?(\d+)x([^\d].*)>', str(value.get_type()))
    return int(match.group(1)) if match else None


def _rows(value):
    match = re.match(r'^tensor<(\d+)x\d+x', str(value.get_type()))
    return int(match.group(1)) if match else 1


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
            return re.sub(r'\d+x([^\d].*)>$', f'{self.vlen}x\\1>', text)
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

    def value_users(self, imp, value):
        """Project the original use graph into this independent column chunk.

        The module walk indexes real SSA values. Virtual chunk ids must use
        the same projected users, otherwise a loop's dot/add stops matching
        its resident accumulator and is lowered to row-by-row ACC traffic.
        The full original use count is retained, including external users:
        in-place accumulation must never hide a live use of the old value.
        """
        original = value.value if isinstance(value, _ValueView) else value
        return [(_OpView(user, self.chunk, self.vlen), operand)
                for user, operand in imp._value_users.get(original.id(), [])]

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


class _MatrixOpView(_OpView):
    def get_operand(self, i):
        # Column fragments share the complete K-wide left operand. Its
        # contiguous dimension is K, whereas the result's dimension is N.
        return self.op.get_operand(0) if i == 0 else super().get_operand(i)


class _RowValueView(_ValueView):
    def __init__(self, value, row, chunk, vlen):
        super().__init__(value, chunk, vlen)
        self.row = row

    def id(self):
        if _rows(self.value) > self.vlen:
            col = self.chunk if (_width(self.value) or 0) > self.vlen else 0
            return ('register-row-chunk', self.value.id(), self.row, col)
        return super().id()

    def get_type(self):
        text = str(super().get_type())
        if _rows(self.value) > self.vlen:
            text = re.sub(r'^tensor<\d+x', f'tensor<{self.vlen}x', text)
        return text


class _RowOpView(_OpView):
    def __init__(self, op, row, chunk, vlen):
        super().__init__(op, chunk, vlen)
        self.row = row

    def get_operand(self, i):
        value = self.op.get_operand(i)
        if self.op.get_name() == 'tt.expand_dims' and self.op.get_int_attr('axis') == 1:
            return _ValueView(value, self.row, self.vlen)
        return _RowValueView(value, self.row, self.chunk, self.vlen)

    def get_result(self, i):
        return _RowValueView(self.op.get_result(i), self.row, self.chunk, self.vlen)


def _native_matrix(imp, op):
    if op.get_name() != 'tt.dot':
        return None
    from .frontend import _parse_type
    a, ae = _parse_type(str(op.get_operand(0).get_type()))
    b, be = _parse_type(str(op.get_operand(1).get_type()))
    if not isinstance(a, tuple) or not isinstance(b, tuple) or len(a) != 2 or len(b) != 2:
        return None
    if (ae != be or ae not in ('f16', 'bf16') or a[0] > imp.target.vlen or
            a[1] != b[0] or a[1] not in (32, 64) or
            f'k{a[1]}' not in imp.target.mma['k_steps']):
        return None
    return a, b


def _matrix_descriptor(imp, op):
    if op.get_name() != 'tt.descriptor_load':
        return None
    from . import descriptors
    users = descriptors.users_of(imp, op.get_result(0).id())
    if not users or any(_native_matrix(imp, u) is None or k not in (0, 1)
                        for u, k in users):
        return None
    # Only an actual transfer-engine window can carry more than one VLEN
    # of K. A host map or an arithmetic consumer retains the existing path.
    d = imp._descs[op.get_operand(0).id()]
    if not descriptors.dma_ok(imp, d, op.get_result(0).id(), d.block[0], d.block[1]):
        return None
    return {k for _u, k in users}


# Side effects and control flow require explicit semantics. Unrecognised
# operations fail rather than accidentally executing a region once per chunk.
_ALLOWED = {'tt.make_range', 'tt.splat', 'tt.addptr', 'tt.load', 'tt.store',
            'tt.broadcast', 'tt.bitcast', 'tt.fp_to_fp', 'tt.clamp', 'tt.reduce',
            'tt.expand_dims', 'tt.reshape', 'tt.dot'}


def lower(imp, op):
    if getattr(imp, '_in_register_chunk', False):
        return False
    matrix = _native_matrix(imp, op)
    matrix_descriptor = _matrix_descriptor(imp, op)
    if matrix_descriptor == {0}:
        # A's wide contiguous dimension is read directly by K32/K64 MMA,
        # not by a vector ALU. Preserve its packed, snapshot window.
        return False
    values = ([op.get_operand(i) for i in range(op.get_num_operands())] +
              [op.get_result(i) for i in range(op.get_num_results())])
    widths = {_width(v) for v in values if _width(v) and _width(v) > imp.target.vlen}
    row_sizes = {_rows(v) for v in values if _rows(v) > imp.target.vlen}
    if row_sizes and op.get_name() == 'tt.expand_dims' and op.get_int_attr('axis') == 1:
        # The vector's lane chunks become rows, not a second independent
        # column axis in the [K, 1] result.
        widths = set()
    # DMA matrix windows preserve all K rows; pointer and arithmetic tiles
    # instead keep independent row chunks until their MMA repack.
    if matrix or matrix_descriptor == {1}:
        row_sizes = set()
    if matrix:
        a, b = matrix
        tiles = getattr(imp, '_logical_gemm_tiles', set())
        tiles.add((a[0], b[1], a[1]))
        imp._logical_gemm_tiles = tiles
        widths_seen = getattr(imp, '_register_block_widths', set())
        widths_seen.add(a[1])
        imp._register_block_widths = widths_seen
        widths = {b[1]} if b[1] > imp.target.vlen else set()
    if not widths and not row_sizes:
        return False
    for value in values:
        dims = re.match(r'^tensor<((?:\d+x)+)', str(value.get_type()))
        if dims:
            shape = [int(n) for n in dims.group(1).split('x') if n]
            matrix_rows = matrix is not None or matrix_descriptor == {1} or bool(row_sizes)
            if len(shape) > 2 or (len(shape) == 2 and shape[0] > imp.target.vlen and not matrix_rows):
                from .frontend import Unsupported
                raise Unsupported('register block requires at most VLEN rows and rank <= 2')
    from .frontend import Unsupported
    if len(widths) > 1 or len(row_sizes) > 1:
        raise Unsupported('register-block operation with mismatched widths')
    width = next(iter(widths), imp.target.vlen)
    rows = next(iter(row_sizes), imp.target.vlen)
    if rows % imp.target.vlen or rows > 64:
        raise Unsupported('register block rows must be a VLEN multiple up to 64')
    if width % imp.target.vlen or width > 128:
        raise Unsupported(f'register block width {width}; expected a VLEN multiple up to 128')
    widths_seen = getattr(imp, '_register_block_widths', set())
    widths_seen.add(width)
    imp._register_block_widths = widths_seen
    name = op.get_name()
    if name == 'scf.for':
        if row_sizes:
            raise Unsupported('register block loop with wide row-carried state')
        return _loop(imp, op, width)
    if name == 'scf.if':
        return _if(imp, op, width)
    if name in ('tt.descriptor_load', 'tt.descriptor_store', 'tt.descriptor_reduce'):
        if row_sizes:
            raise Unsupported('wide descriptor rows require matrix consumers')
        return _descriptor(imp, op, width)
    if name == 'tt.dot':
        from .frontend import _parse_type
        a, _ = _parse_type(str(op.get_operand(0).get_type()))
        b, _ = _parse_type(str(op.get_operand(1).get_type()))
        tiles = getattr(imp, '_logical_gemm_tiles', set())
        tiles.add((a[0], b[1], a[1]))
        imp._logical_gemm_tiles = tiles
    if name not in _ALLOWED and not name.startswith(('arith.', 'math.')):
        raise Unsupported(f'wide register block operation {name}')
    if name == 'tt.reduce':
        return _reduce(imp, op, width)
    fragment_cache = getattr(imp, '_register_mma_fragment_cache', None)
    if name == 'tt.dot':
        # A is shared by the column fragments of this one logical dot.
        # Limit reuse to this operation so it cannot cross a control-flow
        # boundary or reuse a window after another iteration overwrites it.
        imp._register_mma_fragment_cache = {}
    imp._in_register_chunk = True
    try:
        for part in range((width // imp.target.vlen) * (rows // imp.target.vlen)):
            row, chunk = divmod(part, width // imp.target.vlen)
            view = (_RowOpView(op, row, chunk, imp.target.vlen) if row_sizes else
                    (_MatrixOpView if matrix else _OpView)(op, chunk, imp.target.vlen))
            # Decimal/hex float literals are indexed during the original
            # module walk; the virtual SSA identities retain that literal.
            for i in range(op.get_num_results()):
                original = op.get_result(i).id()
                if original in (imp._printed_consts or {}):
                    imp._printed_consts[view.get_result(i).id()] = imp._printed_consts[original]
            pure_predicate = (width // imp.target.vlen >= max(1, imp.target.pregs_per_rctx // 8) and
                              name.startswith('arith.') and
                              name != 'arith.constant' and op.get_num_results() == 1 and
                              re.fullmatch(r'tensor<\d+xi1>', str(op.get_result(0).get_type())) is not None and
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
        imp._register_mma_fragment_cache = fragment_cache
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
    axis = op.get_int_attr('axis')
    if isinstance(shape, tuple) and axis == 0:
        # Each output lane combines the same column across rows. Columns
        # in different VLEN chunks are independent, so retain a distinct
        # result SSA identity per chunk rather than folding them together.
        imp._in_register_chunk = True
        try:
            for chunk in range(width // imp.target.vlen):
                imp._dispatch(_OpView(op, chunk, imp.target.vlen))
        finally:
            imp._in_register_chunk = False
        return True
    if axis != (1 if isinstance(shape, tuple) else 0):
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
                partials.append([imp._tile_row_getter(tile, elem)(r) for r in range(tile.rows)])
            else:
                partials.append([imp._v[rid]])
    finally:
        imp._in_register_chunk = False
    rows = len(partials[0])
    combined_rows = []
    for row in range(rows):
        values = [p[row] for p in partials]
        scalar = values[0].kind == SCALAR
        vectors = [imp.b.splat(v, dtype=elem) if v.kind == SCALAR else v for v in values]
        while len(vectors) > 1:
            vectors = [imp.b.vbin(kind[:3] if kind.endswith('_prop') else kind,
                                 vectors[i], vectors[i + 1], dtype=elem,
                                 fmm='prop' if kind.endswith('_prop') else None)
                       for i in range(0, len(vectors) - 1, 2)] + (vectors[-1:] if len(vectors) % 2 else [])
        combined_rows.append(imp.b.vextract(vectors[0], 0, dtype=elem) if scalar else vectors[0])
    if isinstance(shape, tuple):
        imp._tiles[rid] = Tile('col', tuple(combined_rows), rows, 1, elem)
        imp._v.pop(rid, None)
    else:
        imp._v[rid] = combined_rows[0]
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
    rows = shape[0] if isinstance(shape, tuple) else 1
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            view = _OpView(op, chunk, imp.target.vlen)
            imp._fold(view, keep_partials=False)
            part = []
            for row in range(rows):
                part.append([imp._row_getter(op.get_result(i), rows,
                             _parse_type(str(op.get_result(i).get_type()))[1])(row)
                             for i in range(op.get_num_results())])
            partials.append(part)
    finally:
        imp._in_register_chunk = False
    while len(partials) > 1:
        partials = [[imp._eval_region(op, partials[i][row] + partials[i+1][row])
                     for row in range(rows)]
                    for i in range(0, len(partials)-1, 2)] + (partials[-1:] if len(partials)%2 else [])
    from .tiles import Tile
    for i in range(op.get_num_results()):
        dst = op.get_result(i)
        if isinstance(shape, tuple):
            _, elem = _parse_type(str(dst.get_type()))
            imp._tiles[dst.id()] = Tile('col', tuple(p[i] for p in partials[0]), rows, 1, elem)
            imp._v.pop(dst.id(), None)
        else:
            imp._v[dst.id()] = partials[0][0][i]
    seen = getattr(imp, '_register_reductions', set())
    seen.add((width, 'body-fold', op.get_int_attr('axis')))
    imp._register_reductions = seen
    return True


class _CoordinateView:
    def __init__(self, value, chunk):
        self.value, self.chunk = value, chunk

    def id(self):
        return ('descriptor-chunk-coordinate', self.value.id(), self.chunk)

    def __getattr__(self, name):
        return getattr(self.value, name)


class _DescriptorOpView(_OpView):
    def __init__(self, op, chunk, vlen, coordinate, coordinate_index):
        super().__init__(op, chunk, vlen)
        self.coordinate = coordinate
        self.coordinate_index = coordinate_index

    def get_operand(self, i):
        if i == self.coordinate_index:
            return self.coordinate
        return super().get_operand(i)


def _descriptor(imp, op, width):
    from .frontend import Unsupported
    from . import descriptors
    import copy
    key = op.get_operand(0).id()
    original = imp._descs[key]
    if original.block[-1] != width:
        raise Unsupported('register block and descriptor box widths differ')
    if op.get_name() == 'tt.descriptor_load':
        coordinate = op.get_operand(op.get_num_operands() - 1)
    else:
        _, coords = descriptors._store_operands(op)
        coordinate = coords[-1]
    # Several dimensions may use the same scalar SSA value (notably zero).
    # Shift the last coordinate operand, rather than every use of that value.
    coordinate_index = max(i for i in range(1, op.get_num_operands())
                           if op.get_operand(i).id() == coordinate.id())
    block = list(original.block)
    block[-1] = imp.target.vlen
    if original.host:
        # Preserve runtime padding, extents and strides from the original
        # host map. The LSU slices the box; it cannot assume DMA alignment.
        fragment = copy.copy(original)
        fragment.block = block
    else:
        fragment = imp._build_descriptor(base=original.base, elem=original.elem,
            extents=descriptors.extents_of(imp, original),
            byte_strides=original.byte_strides, block_shape=block,
            what='register-block descriptor fragment', pad=original.pad, check=False)
    # One fragment map per operation, within the same control region: a
    # map first materialised in a branch must not be reused outside it.
    imp._descs[key] = fragment
    imp._in_register_chunk = True
    try:
        base_coordinate = imp._value(coordinate)
        for chunk in range(width // imp.target.vlen):
            shifted = _CoordinateView(coordinate, chunk)
            imp._v[shifted.id()] = (base_coordinate if chunk == 0 else
                imp.b.sbini('add', base_coordinate, chunk * imp.target.vlen))
            imp._dispatch(_DescriptorOpView(op, chunk, imp.target.vlen, shifted,
                                           coordinate_index))
    finally:
        imp._descs[key] = original
        imp._in_register_chunk = False
    return True
