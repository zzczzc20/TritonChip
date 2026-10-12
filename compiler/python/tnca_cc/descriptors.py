"""Tensor descriptors of rank 1 to 5: `tl.make_tensor_descriptor`, host
`TensorDescriptor` arguments, and every operation Triton defines on them.

The contract is the one Triton states for TMA descriptors on NVIDIA GPUs
(`triton.language.make_tensor_descriptor`, `triton.tools.tensor_descriptor`):

  rank 1-5, dimension 0 outermost, the last dimension contiguous;
  a box of `block_shape` elements at signed coordinates; an element is in
  bounds when 0 <= coordinate + index < extent in every dimension;
  a load fills an out-of-bounds element with zero, or NaN under
  padding_option="nan"; a store, an atomic reduction or a scatter does not
  touch it.

The machine has two ways to honour it, and this file chooses per operation:

  the transfer engine   `t.ld.async` / `t.st.async` (ISA sec 9E), rank 1-5,
                        the box flat and row-major in an RF window. Used when
                        the result is a tile a window can hold and be read
                        from: an MMA operand, a 32-bit tile one row per
                        register, or a value that only goes back to memory
                        through another descriptor.
  the LSU               one masked contiguous load or store per box row, or
                        one masked gather/scatter for a small N-D box that
                        fits one register. Exact for any geometry, including
                        the ones the engine refuses (a host descriptor
                        whose base or stride is not 8-byte aligned).

Both give the same answer. The engine is preferred where it applies, because
one command replaces a load per row.

Values: a box of at most VLEN elements is ONE register -- a 1-D vector when
only its last dimension is wider than one, otherwise a packed N-D value in
row-major lane order (packed.py). A larger box is a tile of
prod(block[:-1]) rows by block[-1] columns.
"""
import math

from .ir import VECTOR
from .tiles import Tile, DTYPE_CODE, PAD_ZERO, PAD_NAN, TENSORMAP_BYTES

# ISA sec 9D.3 padding codes, from Triton's PADDING_OPTION (PAD_ZERO = 1,
# PAD_NAN = 2; the attribute is absent for the default).
TRITON_PADDING = {None: PAD_ZERO, 1: PAD_ZERO, 2: PAD_NAN}

# The quiet NaN a padded element of each floating type holds.
NAN_BITS = {'f16': 0x7E00, 'bf16': 0x7FC0, 'f32': 0x7FC00000,
            'f8E4M3FN': 0x7F, 'f8E5M2': 0x7F}

# Triton's DESCRIPTOR_REDUCE_KIND, onto the ISA's atomic operations.
REDUCE_KIND = {1: 'add', 2: 'min', 3: 'max', 6: 'and', 7: 'or', 8: 'xor'}

# What the transfer engine is asked to move, by element type. A window can
# be read back (or fed to the MMA) only for these.
DMA_ELEMS = ('f16', 'bf16', 'f32')

# Largest box the engine is asked to place in one window, in bytes.
DMA_MAX_BYTES = 1024


def _unsupported(imp, what, **kw):
    from .frontend import Unsupported
    return Unsupported(what, **kw)


def esize(imp, elem):
    n = imp._ELEM_BYTES.get(elem)
    if n is None:
        raise _unsupported(imp, f'a tensor descriptor of {elem}',
                           because='the value model holds 1, 2 and 4-byte elements; '
                                   '64-bit tensors go through `wide.py`, which has no '
                                   'descriptor form yet')
    return n


def as_value(imp, x):
    return imp.b.const(x) if isinstance(x, int) else x


# ---------------------------------------------------------------------------
# Building a descriptor
# ---------------------------------------------------------------------------

def make(imp, op):
    """`tl.make_tensor_descriptor(base, shape, strides, block_shape,
    padding_option)`, any rank 1-5, shapes and strides constant or not."""
    from .frontend import _pointee, _parse_type
    ptr = op.get_operand(0)
    n = op.get_num_operands()
    rank = (n - 1) // 2
    if not 1 <= rank <= 5:
        raise _unsupported(imp, f'a rank-{rank} tensor descriptor',
                           because='Triton and the transfer engine define ranks 1-5')
    elem = _pointee(str(ptr.get_type()))
    es = esize(imp, elem)
    extents, strides = [], []
    for i in range(rank):
        k = imp._scalar_const(op.get_operand(1 + i).id())
        extents.append(k if k is not None else imp._value(op.get_operand(1 + i)))
    for i in range(rank):
        k = imp._scalar_const(op.get_operand(1 + rank + i).id())
        if k is not None:
            strides.append(k * es)
        else:
            s = imp._value(op.get_operand(1 + rank + i))
            strides.append(s if es == 1 else
                           imp.b.sbini('shl', s, es.bit_length() - 1,
                                       comment='stride in bytes'))
    shape, _ = _parse_type(str(op.get_result(0).get_type()).replace(
        '!tt.tensordesc<', 'tensor<').rstrip('>') + '>')
    block = list(shape) if isinstance(shape, tuple) else [shape or 1]
    if len(block) != rank:
        raise _unsupported(imp, f'a descriptor whose block {block} is not rank {rank}')
    pad = TRITON_PADDING.get(op.get_int_attr('padding'))
    if pad is None:
        raise _unsupported(imp, f"padding option {op.get_int_attr('padding')}")
    imp._descs[op.get_result(0).id()] = imp._build_descriptor(
        base=imp._value(ptr), elem=elem, extents=extents, byte_strides=strides,
        block_shape=block, what=f'tl.make_tensor_descriptor (rank {rank})',
        pad=pad, check=False)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def extents_of(imp, d):
    if d.extents is not None:
        return d.extents
    return [imp.b.sload(d.address, offset=8 + 8 * i, size='d')
            for i in range(len(d.block))]


def origin(imp, d, coords):
    """base + sum(coord * byte_stride): the address of box element 0."""
    return imp._descriptor_origin(d, coords)


def row_index(block, r):
    """Box row r (rows are the outer dimensions in row-major order) as the
    index of each outer dimension."""
    outer = block[:-1]
    at = []
    for b in reversed(outer):
        at.append(r % b)
        r //= b
    return list(reversed(at))


def row_address(imp, d, base, at):
    addr = base
    for i, stride in zip(at, d.byte_strides[:-1]):
        if i:
            addr = imp._scaled_address(addr, i, stride)
    return addr


def lane_positions(imp, coord):
    """coord + lane, the contiguous-dimension index of each lane."""
    return imp.b.vbin('add', imp.b.splat(coord), imp.b.arange(), dtype='i32')


def in_bounds(imp, pos, extent):
    """0 <= pos < extent, as a predicate (pos a vector)."""
    b = imp.b
    return b.pbin('and', b.vcmp('ge', pos, b.vmovi(0)),
                  b.vcmp('lt', pos, b.splat(as_value(imp, extent))))


def row_mask(imp, d, coords, extents, at, inner):
    """Lanes of box row `at` that are inside the tensor and inside the box."""
    b = imp.b
    mask = inner
    for c, i, e in zip(coords[:-1], at, extents[:-1]):
        pos = b.splat(c if not i else b.sbini('add', c, i))
        mask = b.pbin('and', mask, in_bounds(imp, pos, e))
    return imp._shape_mask(d.block[-1], mask)


def inner_mask(imp, coords, extents):
    return in_bounds(imp, lane_positions(imp, coords[-1]), extents[-1])


def fill(imp, d):
    """(immediate, runtime vector or None) for out-of-bounds load lanes.

    A descriptor this kernel built knows its padding; a host descriptor's is
    in its TensorMap, so it is read and selected at run time.
    """
    if d.host:
        word = imp.b.sload(d.address, offset=104, size='d')
        pad = imp.b.sbini('and', imp.b.sbini('lshr', word, 48), 0xFF)
        nan = NAN_BITS.get(d.elem)
        if nan is None:
            return 0, None
        is_nan = imp.b.vcmp('eq', imp.b.splat(pad), imp.b.vmovi(PAD_NAN))
        return 0, imp.b.vselect(is_nan, imp.b.vmovi(nan, dtype=d.elem),
                                imp.b.vmovi(0, dtype=d.elem))
    if d.pad == PAD_NAN:
        return NAN_BITS[d.elem], None
    return 0, None


def _filled(imp, v, mask, fillv):
    if fillv is None or mask is None:
        return v
    return imp.b.vselect(mask, v, fillv)


def suffix(imp, elem, load):
    from .frontend import _SIZE_SUFFIX, _load_suffix
    s = _SIZE_SUFFIX[esize(imp, elem)]
    return _load_suffix(s, elem) if load else s


# ---------------------------------------------------------------------------
# Lane maps for a box that fits one register
# ---------------------------------------------------------------------------

def lane_index(imp, block, dim):
    """For each lane l < prod(block), the index along `dim` of box element l
    in row-major order, as a vector (lanes past the box hold junk; they are
    masked off)."""
    b = imp.b
    inner = math.prod(block[dim + 1:])
    size = block[dim]
    if inner & (inner - 1) == 0 and size & (size - 1) == 0:
        v = b.arange()
        if inner > 1:
            v = b.vbin('lshr', v, b.vmovi(inner.bit_length() - 1), dtype='i32')
        if dim > 0:
            v = b.vbin('and', v, b.vmovi(size - 1), dtype='i32')
        return v
    v = b.vmovi(0, dtype='i32')
    for lane in range(min(math.prod(block), imp.target.vlen)):
        b.vinsert(v, lane, b.const((lane // inner) % size))
    return v


def gather_plan(imp, d, coords, extents):
    """Per-lane element offsets from the descriptor base, and the lanes that
    are inside the tensor and inside the box."""
    b = imp.b
    es = esize(imp, d.elem)
    count = math.prod(d.block)
    offset, mask = None, None
    for dim, (c, e, s) in enumerate(zip(coords, extents, d.byte_strides)):
        pos = b.vbin('add', b.splat(c), lane_index(imp, d.block, dim), dtype='i32')
        ok = in_bounds(imp, pos, e)
        mask = ok if mask is None else b.pbin('and', mask, ok)
        selem = (b.const(s // es) if isinstance(s, int) else
                 (s if es == 1 else b.sbini('ashr', s, es.bit_length() - 1)))
        term = b.vbin('mul', pos, b.splat(selem), dtype='i32')
        offset = term if offset is None else b.vbin('add', offset, term, dtype='i32')
    return offset, imp._shape_mask(count, mask)


# ---------------------------------------------------------------------------
# Choosing the transfer engine
# ---------------------------------------------------------------------------

def users_of(imp, value_id):
    # Register chunks retain the original TTIR consumer contract. Treating
    # a synthetic ID as unused could select a packed DMA window even when
    # an arithmetic consumer needs individually unpacked elements.
    if isinstance(value_id, tuple) and value_id[0] in ('register-chunk', 'register-row-chunk'):
        value_id = value_id[1]
    return imp._value_users.get(value_id, [])


def window_consumers_ok(imp, rid, elem, cols):
    """Whether every use of a window holding this box can read it.

    An MMA operand reads a window directly; a store through a descriptor
    writes one out with t.st.async; a reshape or a trans that only relabels
    is followed through. A 32-bit tile one row per register is readable by
    anything (its registers ARE the rows). A 16-bit window is not readable
    one element per lane (v.unpack is reserved), so anything else rules the
    engine out for it.
    """
    if imp._ELEM_BYTES[elem] == 4 and cols == imp.target.vlen:
        return True
    seen, work = set(), [rid]
    while work:
        v = work.pop()
        if v in seen:
            continue
        seen.add(v)
        for u, k in users_of(imp, v):
            name = u.get_name()
            if name == 'tt.dot' and k in (0, 1):
                continue
            if name == 'tt.descriptor_store':
                continue
            if name == 'tt.reshape':
                work.append(u.get_result(0).id())
                continue
            return False
    return True


def dma_ok(imp, d, rid, rows, cols, *, store=False):
    if d.host or d.value is None:
        return False
    if d.elem not in DMA_ELEMS:
        return False
    if not imp.target.has('tensor_dma'):
        return False
    if store and not imp.target.has('tensor_store'):
        return False
    if len(d.block) > getattr(imp.target, 'dma_max_rank', 2):
        return False
    es = imp._ELEM_BYTES[d.elem]
    if (d.block[-1] * es) % 8 or any(b > 256 for b in d.block):
        return False
    if rows * cols * es > DMA_MAX_BYTES:
        return False
    if store:
        return True
    return window_consumers_ok(imp, rid, d.elem, cols)


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------

def load(imp, op):
    import math as _m
    d = imp._descs[op.get_operand(0).id()]
    coords = [imp._value(op.get_operand(1 + i)) for i in range(op.get_num_operands() - 1)]
    rid = op.get_result(0).id()
    block = d.block
    count = _m.prod(block)
    vlen = imp.target.vlen
    if block[-1] > vlen:
        raise _unsupported(imp, f'a descriptor block {block} whose last dimension is '
                                f'wider than one vector ({vlen})',
                           because='one box row is one vector register')
    extents = extents_of(imp, d)
    imm, fillv = fill(imp, d)
    sfx = suffix(imp, d.elem, load=True)
    base = origin(imp, d, coords)
    inner = inner_mask(imp, coords, extents)
    rows = count // block[-1]

    if rows == 1:
        # One run of the contiguous dimension: one masked load.
        mask = row_mask(imp, d, coords, extents, [0] * (len(block) - 1), inner)
        v = _filled(imp, imp.b.load(base, mask=mask, other=imm, dtype=d.elem, size=sfx),
                    mask, fillv)
        imp._v[rid] = v
        if len(block) > 1:
            # Held both ways: as a one-row tile, the form pointer loads of
            # the same shape take and the tile lowering expects, and as one
            # register, the form a change of rank (expand_dims, trans,
            # reshape) needs. packed.lower picks.
            imp._packed.add(rid)
            imp._tiles[rid] = Tile('rows', (v,), 1, block[-1], d.elem)
        return

    if count <= vlen:
        # A small N-D box: every element in its row-major lane, one gather.
        imp.target.require('gather_scatter', 'emit v.ld.gather for a descriptor box')
        offset, mask = gather_plan(imp, d, coords, extents)
        v = imp.b.gather(d.base, offset, mask=mask, other=imm, dtype=d.elem, size=sfx)
        imp._v[rid] = _filled(imp, v, mask, fillv)
        imp._packed.add(rid)
        return

    if dma_ok(imp, d, rid, rows, block[-1]):
        tile = imp._dma_in(d, coords, [rows, block[-1]], d.elem,
                           f'tt.descriptor_load (rank {len(block)})')
        if len([x for x in block[:-1] if x > 1]) > 1:
            tile.desc = None      # rows are not one stride apart
        imp._tiles[rid] = tile
        return

    def row(r):
        at = row_index(block, r)
        mask = row_mask(imp, d, coords, extents, at, inner)
        v = imp.b.load(row_address(imp, d, base, at), mask=mask, other=imm,
                       dtype=d.elem, size=sfx)
        return _filled(imp, v, mask, fillv)
    imp._tiles[rid] = imp._materialise(row, rows, block[-1], d.elem)


# ---------------------------------------------------------------------------
# Stores, reductions
# ---------------------------------------------------------------------------

def _store_operands(op):
    """`desc[coords], value`: the operands are (desc, value?, coords...) in
    an order MLIR prints differently from how it stores them, so the value
    is found by its type."""
    from .frontend import _parse_type
    value = None
    for i in range(1, op.get_num_operands()):
        cand = op.get_operand(i)
        if isinstance(_parse_type(str(cand.get_type()))[0], (tuple, int)) and \
                str(cand.get_type()).startswith('tensor<'):
            value = cand
            break
    coords = [op.get_operand(i) for i in range(1, op.get_num_operands())
              if value is None or op.get_operand(i).id() != value.id()]
    return value, coords


def _rows_of_value(imp, value, d):
    """('tile', Tile), ('vector', register) or ('rows', row getter) for the
    value being written: a tile, one register (1-D or packed N-D), or any
    other N-D value the importer can produce a row at a time (a dense
    constant, an affine index tile)."""
    from .frontend import _parse_type
    tile = imp._tiles.get(value.id())
    if tile is not None:
        return 'tile', tile
    shape = _parse_type(str(value.get_type()))[0]
    if not isinstance(shape, tuple) or value.id() in imp._packed or \
            math.prod(shape) <= imp.target.vlen and value.id() in imp._v:
        return 'vector', imp._vec(value, d.elem)
    rows = math.prod(shape[:-1])
    return 'rows', imp._row_getter(value, rows, d.elem)


def store(imp, op):
    d = imp._descs[op.get_operand(0).id()]
    value, coord_ops = _store_operands(op)
    if value is None:
        raise _unsupported(imp, 'a tt.descriptor_store with no value to store')
    coords = [imp._value(c) for c in coord_ops]
    block = d.block
    count = math.prod(block)
    rows = count // block[-1]
    kind, src = _rows_of_value(imp, value, d)
    extents = extents_of(imp, d)
    base = origin(imp, d, coords)
    sfx = suffix(imp, d.elem, load=False)

    if kind == 'tile' and src.kind == 'window' and src.rows == rows and \
            src.cols == block[-1] and dma_ok(imp, d, None, rows, block[-1], store=True):
        imp._dma_out(d.value, coords, src, d.elem,
                     f'tt.descriptor_store (rank {len(block)})')
        return

    inner = inner_mask(imp, coords, extents)
    if kind == 'vector':
        v = src
        if rows == 1:
            mask = row_mask(imp, d, coords, extents, [0] * (len(block) - 1), inner)
            imp.b.store(base, v, mask=mask, size=sfx)
            return
        imp.target.require('gather_scatter', 'emit v.st.scatter for a descriptor box')
        offset, mask = gather_plan(imp, d, coords, extents)
        imp.b.scatter(d.base, offset, v, mask=mask, size=sfx)
        return

    if kind == 'tile':
        tile = src
        if tile.rows != rows or tile.cols != block[-1]:
            raise _unsupported(imp, f'a {tile.rows}x{tile.cols} tile stored through a '
                                    f'descriptor block {block}')
        get, held = imp._tile_row_getter(tile), tile.store
    else:
        get, held = src, d.elem
    for r in range(rows):
        at = row_index(block, r)
        v = get(r)
        if d.elem != held:
            v = imp.b.vcvt(held, d.elem, v,
                           comment=f'row {r}: {held} -> {d.elem} (ISA sec 8.4)')
        imp.b.store(row_address(imp, d, base, at), v,
                    mask=row_mask(imp, d, coords, extents, at, inner), size=sfx)


def reduce(imp, op):
    """`desc.atomic_add/min/max/and/or/xor(offsets, value)`: one atomic per
    in-bounds element, as tt.atomic_rmw does it (ISA sec 9.2)."""
    d = imp._descs[op.get_operand(0).id()]
    kind = REDUCE_KIND.get(op.get_int_attr('kind'))
    if kind is None:
        raise _unsupported(imp, f"descriptor reduction kind {op.get_int_attr('kind')}")
    elem = d.elem
    es = esize(imp, elem)
    if es not in (4, 8):
        raise _unsupported(imp, f'a descriptor atomic on {elem}',
                           because='ISA sec 9.2 atomics are 4 or 8 bytes')
    if elem.startswith('f') and (kind != 'add' or elem != 'f32'):
        raise _unsupported(imp, f'a descriptor atomic {kind} on {elem}',
                           because='the only floating-point atomic is FP32 add '
                                   '(a compare-and-swap loop, ISA sec 9.2)')
    value, coord_ops = _store_operands(op)
    coords = [imp._value(c) for c in coord_ops]
    block = d.block
    rows = math.prod(block) // block[-1]
    extents = extents_of(imp, d)
    base = origin(imp, d, coords)
    inner = inner_mask(imp, coords, extents)
    kind_v, src = _rows_of_value(imp, value, d)
    get = (imp._tile_row_getter(src) if kind_v == 'tile' else
           src if kind_v == 'rows' else (lambda r: src))
    from .frontend import _SIZE_SUFFIX
    size = _SIZE_SUFFIX[es]
    b = imp.b
    for r in range(rows):
        at = row_index(block, r)
        mask = row_mask(imp, d, coords, extents, at, inner)
        if kind_v == 'vector' and rows > 1:
            # A packed box: the row's elements are lanes r*cols.. of one register.
            raise _unsupported(imp, 'a descriptor atomic on a box of several rows that '
                                    'fits one register')
        active = b.vselect(mask, b.vmovi(1), b.vmovi(0)) if mask is not None else None
        values = get(r)
        addr0 = row_address(imp, d, base, at)
        for c in range(block[-1]):
            end = imp._control_label('desc_atomic_skip')
            if active is not None:
                b.branch_nz(b.scmp('eq', b.vextract(active, c), b.const(0)), end)
            addr = addr0 if c == 0 else b.sbini('add', addr0, c * es)
            v = b.vextract(values, c, dtype=elem)
            if elem.startswith('f'):
                imp._atomic_fadd(addr, v, elem, 'relaxed', 'device', size, f'd{r},{c}')
            else:
                b.atomic(kind, addr, v, size=size, order='relaxed', scope='device',
                         dtype=elem)
            b.label(end)


# ---------------------------------------------------------------------------
# Gather / scatter (a 2-D descriptor with a one-row block)
# ---------------------------------------------------------------------------

def _gather_operands(imp, op, store):
    d = imp._descs[op.get_operand(0).id()]
    if len(d.block) != 2 or d.block[0] != 1:
        raise _unsupported(imp, 'a descriptor gather/scatter that is not 2-D with a '
                                'one-row block')
    from .frontend import _parse_type
    # By type, not position: the row offsets are a 1-D tensor, the value
    # (scatter only) a 2-D one, the column offset a scalar.
    value = xoff = yoff = None
    for i in range(1, op.get_num_operands()):
        v = op.get_operand(i)
        shape = _parse_type(str(v.get_type()))[0]
        if isinstance(shape, tuple):
            value = v
        elif shape is not None:
            xoff = v
        else:
            yoff = v
    if xoff is None or yoff is None or (store and value is None):
        raise _unsupported(imp, f'the operands of {op.get_name()}')
    return d, value, imp._vec(xoff, 'i32'), imp._value(yoff)


def gather(imp, op):
    from .frontend import _parse_type
    d, _, xv, y = _gather_operands(imp, op, store=False)
    shape, _ = _parse_type(str(op.get_result(0).get_type()))
    rows, cols = shape
    if rows > imp.target.vlen or cols > imp.target.vlen:
        raise _unsupported(imp, f'a descriptor gather of {rows}x{cols}',
                           because='the row offsets and each row are one register')
    extents = extents_of(imp, d)
    imm, fillv = fill(imp, d)
    sfx = suffix(imp, d.elem, load=True)
    b = imp.b
    inner = in_bounds(imp, lane_positions(imp, y), extents[1])
    col0 = imp._scaled_address(d.base, y, d.byte_strides[1])

    def row(r):
        x = b.vextract(xv, r)
        ok = in_bounds(imp, b.splat(x), extents[0])
        mask = imp._shape_mask(cols, b.pbin('and', inner, ok))
        v = b.load(imp._scaled_address(col0, x, d.byte_strides[0]), mask=mask,
                   other=imm, dtype=d.elem, size=sfx)
        return _filled(imp, v, mask, fillv)
    imp._tiles[op.get_result(0).id()] = imp._materialise(row, rows, cols, d.elem)


def scatter(imp, op):
    d, value, xv, y = _gather_operands(imp, op, store=True)
    tile = imp._tile_of(value)
    extents = extents_of(imp, d)
    sfx = suffix(imp, d.elem, load=False)
    b = imp.b
    inner = in_bounds(imp, lane_positions(imp, y), extents[1])
    col0 = imp._scaled_address(d.base, y, d.byte_strides[1])
    get = imp._tile_row_getter(tile)
    for r in range(tile.rows):
        x = b.vextract(xv, r)
        ok = in_bounds(imp, b.splat(x), extents[0])
        v = get(r)
        if d.elem != tile.store:
            v = b.vcvt(tile.store, d.elem, v)
        b.store(imp._scaled_address(col0, x, d.byte_strides[0]), v,
                mask=imp._shape_mask(tile.cols, b.pbin('and', inner, ok)), size=sfx)
