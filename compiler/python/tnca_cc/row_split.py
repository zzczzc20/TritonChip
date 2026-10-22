"""Run a program whose rows are independent as a loop over 16-row chunks.

Triton block kernels size their M tile to the problem (`BLOCK_SIZE_M` up to
128). Unrolled, every operation on a 128-row tile is eight row chunks of code,
and a GEMM body that fits the 128 KiB code store at 16 rows does not at 32. If
no operation couples one row of the tile to another, the program is the same
as running its body once per 16-row chunk, each chunk seeing rows
[r0, r0 + 16) of that dimension; the body is then emitted once, inside a loop.

Which dimension is "the rows" is found, not assumed: a union-find over every
tensor dimension of every value, joined through each operation's semantics
(elementwise operands and results share dimensions, `tt.dot` joins A's rows
with C's, `tt.broadcast` and `tt.expand_dims` map them). `tt.make_range`,
`tt.splat` and dense constants make fresh dimensions at each use, since one
`arange(0, 128)` may index both M and K. A class is split only when:

  * no reduction, scan, dot K, reshape, transpose, join, gather, atomic or
    descriptor operation touches it;
  * it is the first dimension of every 2-D value it appears in, every member
    has the same size, a multiple of VLEN larger than VLEN, and an arange
    supplies its index;
  * no load follows a store in program order (a later chunk must not read
    what an earlier chunk wrote).

Everything else -- values without the dimension, like a GEMM's B tile -- is
recomputed in each chunk. Only the extended lowering (frontend.large_tiles)
asks for this.
"""
import re

_ELEMENTWISE_PREFIX = ('arith.', 'math.')
_ELEMENTWISE = {'tt.addptr', 'tt.load', 'tt.store', 'tt.fp_to_fp', 'tt.bitcast',
                'tt.clampf', 'tt.precise_sqrt', 'tt.precise_divf', 'tt.mulhiui',
                'tt.extern_elementwise', 'tt.int_to_ptr', 'tt.ptr_to_int'}
_POLY = ('tt.make_range', 'tt.splat', 'arith.constant')
_IGNORE = {'tt.return', 'tt.get_program_id', 'tt.get_num_programs', 'scf.yield',
           'tt.reduce.return', 'llvm.intr.assume', 'tt.func', 'builtin.module'}


def _shape(value):
    m = re.match(r'^tensor<((?:\d+x)+)', str(value.get_type()))
    return tuple(int(n) for n in m.group(1).split('x') if n) if m else ()


class Plan:
    def __init__(self, imp, size, dims, poly_dims, use_dims):
        self.imp, self.size = imp, size
        self.dims = dims            # value id -> set of split dimensions
        self.poly_dims = poly_dims  # poly value id -> split dimensions of its chunk version
        self.use_dims = use_dims    # (consumer id, operand) -> dims, for poly operands
        self.row0 = None


def plan(imp, ops, bodies):
    """A Plan, or None when the program has no splittable row dimension."""
    vlen = imp.target.vlen
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    defs = {}
    for o in ops:
        for i in range(o.get_num_results()):
            defs[o.get_result(i).id()] = o

    def poly(v):
        d = defs.get(v.id())
        return d is not None and d.get_name() in _POLY and _shape(v)

    sizes, blocked, ranges, rank2 = {}, set(), set(), {}

    def node(v, d, use=None):
        if use is not None and poly(v):
            key = ('use', use[0], use[1], d)
            if defs[v.id()].get_name() == 'tt.make_range':
                ranges.add(key)
        else:
            key = ('v', v.id(), d)
        sizes[key] = _shape(v)[d]
        rank2.setdefault(key, len(_shape(v)))
        return key

    def operand(o, i, d):
        return node(o.get_operand(i), d, (id(o), i))

    loads_after_store = False
    stored = False
    for o in ops:
        name = o.get_name()
        if name in ('tt.store', 'tt.atomic_rmw', 'tt.atomic_cas', 'tt.descriptor_store'):
            stored = True
        if name in ('tt.load', 'tt.descriptor_load') and stored:
            loads_after_store = True
        if name.startswith('cf.'):
            return None
        tensors_in = [i for i in range(o.get_num_operands()) if _shape(o.get_operand(i))]
        tensors_out = [i for i in range(o.get_num_results()) if _shape(o.get_result(i))]
        if name in _POLY or name in _IGNORE or (not tensors_in and not tensors_out):
            continue
        if name in _ELEMENTWISE or name.startswith(_ELEMENTWISE_PREFIX):
            rank = len(_shape(o.get_operand(tensors_in[0]) if tensors_in else o.get_result(tensors_out[0])))
            for d in range(rank):
                keys = [operand(o, i, d) for i in tensors_in] + \
                       [node(o.get_result(i), d) for i in tensors_out]
                for k in keys[1:]:
                    union(keys[0], k)
        elif name == 'tt.broadcast':
            src, res = o.get_operand(0), o.get_result(0)
            for d, (a, b) in enumerate(zip(_shape(src), _shape(res))):
                if a == b:
                    union(operand(o, 0, d), node(res, d))
                else:
                    node(res, d)
        elif name == 'tt.expand_dims':
            axis = o.get_int_attr('axis')
            res = o.get_result(0)
            for d in range(len(_shape(res))):
                if d != axis:
                    union(node(res, d), operand(o, 0, d if d < axis else d - 1))
        elif name == 'tt.reduce':
            axis = o.get_int_attr('axis')
            for i in tensors_in:
                rank = len(_shape(o.get_operand(i)))
                blocked.add(operand(o, i, axis))
                for d in range(rank):
                    if d != axis and o.get_num_results() > i and _shape(o.get_result(i)):
                        union(operand(o, i, d), node(o.get_result(i), d if d < axis else d - 1))
        elif name == 'tt.dot':
            r = o.get_result(0)
            union(operand(o, 0, 0), node(r, 0))
            union(operand(o, 2, 0), node(r, 0))
            union(operand(o, 1, 1), node(r, 1))
            union(operand(o, 2, 1), node(r, 1))
            blocked.add(operand(o, 0, 1))
            blocked.add(operand(o, 1, 0))
        elif name == 'scf.for':
            body = bodies.get(o.get_region(0).id(), [])
            if not body:
                return None
            block = body[0].get_block()
            yields = [x for x in body if x.get_name() == 'scf.yield' and x.get_block().id() == block.id()]
            for i in range(o.get_num_results()):
                res = o.get_result(i)
                for d in range(len(_shape(res))):
                    k = node(res, d)
                    union(k, operand(o, 3 + i, d))
                    union(k, node(block.get_argument(i + 1), d))
                    for y in yields:
                        union(k, operand(y, i, d))
        else:
            for i in tensors_in:
                for d in range(len(_shape(o.get_operand(i)))):
                    blocked.add(operand(o, i, d))
            for i in tensors_out:
                for d in range(len(_shape(o.get_result(i)))):
                    blocked.add(node(o.get_result(i), d))
    if loads_after_store:
        return None
    classes = {}
    for k in list(sizes):
        classes.setdefault(find(k), []).append(k)
    bad = {find(k) for k in blocked}
    best = None
    for root, members in classes.items():
        size = sizes[members[0]]
        if root in bad or size <= vlen or size % vlen or any(sizes[m] != size for m in members):
            continue
        if not any(m in ranges for m in members):
            continue
        if any(rank2[m] == 2 and m[-1] != 0 for m in members) or any(rank2[m] > 2 for m in members):
            continue
        score = sum(1 for m in members if rank2[m] == 2)
        if score and (best is None or score > best[0]):
            best = (score, root, size)
    if best is None:
        return None
    _score, root, size = best
    dims, use_dims, poly_dims = {}, {}, {}
    for k in classes[root]:
        if k[0] == 'v':
            dims.setdefault(k[1], set()).add(k[2])
        else:
            use_dims.setdefault((k[1], k[2]), set()).add(k[3])
    # A poly operand's split dimensions are its consumer's: one chunk version
    # per value, which every split use must agree on.
    consumers = {}
    for o in ops:
        for i in range(o.get_num_operands()):
            consumers[(id(o), i)] = o.get_operand(i)
    for use, ds in use_dims.items():
        v = consumers[use]
        prev = poly_dims.setdefault(v.id(), frozenset(ds))
        if prev != frozenset(ds):
            return None
    return Plan(imp, size, dims, poly_dims, use_dims)


def _narrow(type_text, dims, vlen):
    m = re.match(r'^tensor<((?:\d+x)+)(.*)$', type_text)
    if not m or not dims:
        return type_text
    shape = [n for n in m.group(1).split('x') if n]
    for d in dims:
        shape[d] = str(vlen)
    return 'tensor<' + 'x'.join(shape) + 'x' + m.group(2)


class _MValue:
    def __init__(self, value, p, use=None, chunk_version=False):
        self.value, self._p = value, p
        vid = value.id()
        if chunk_version:
            self._dims = p.poly_dims.get(vid, frozenset())
            self._chunk = True
        elif use is not None and use in p.use_dims:
            self._dims = p.use_dims[use]
            self._chunk = True
        else:
            self._dims = p.dims.get(vid, set())
            self._chunk = False

    def id(self):
        return ('row-split', self.value.id()) if self._chunk else self.value.id()

    def get_type(self):
        return _narrow(str(self.value.get_type()), self._dims, self._p.imp.target.vlen)

    def __getattr__(self, name):
        return getattr(self.value, name)


class _MBlock:
    def __init__(self, block, p):
        self.block, self._p = block, p

    def get_argument(self, i):
        return _MValue(self.block.get_argument(i), self._p)

    def __getattr__(self, name):
        return getattr(self.block, name)


class _MOp:
    def __init__(self, op, p, chunk_version=False):
        self.op, self._p, self._chunk = op, p, chunk_version

    def get_operand(self, i):
        return _MValue(self.op.get_operand(i), self._p, use=(id(self.op), i))

    def get_result(self, i):
        return _MValue(self.op.get_result(i), self._p, chunk_version=self._chunk)

    def get_block(self):
        return _MBlock(self.op.get_block(), self._p)

    def __getattr__(self, name):
        return getattr(self.op, name)

    def __str__(self):
        return str(self.op)


def run(imp, p, top_ops, dispatch):
    """Emit `top_ops` (without tt.return) once per row chunk, in a loop."""
    b = imp.b
    vlen = imp.target.vlen
    wrapped = {}

    def wrap(op, chunk=False):
        key = (id(op), chunk)
        if key not in wrapped:
            wrapped[key] = _MOp(op, p, chunk)
        return wrapped[key]

    def chunk_poly(op):
        return (op.get_num_results() and op.get_result(0).id() in p.poly_dims
                and op.get_name() != 'tt.make_range')

    def sequence(ops):
        out = []
        for o in ops:
            if chunk_poly(o):
                out.append(wrap(o, True))
            out.append(wrap(o))
        return out

    saved_bodies = dict(imp._bodies)
    imp._bodies = {rid: sequence(body) for rid, body in saved_bodies.items()}
    ranges = [o for o in imp._all_ops if o.get_name() == 'tt.make_range'
              and o.get_result(0).id() in p.poly_dims]
    p.row0 = imp._copy_value(b.const(0))
    caches = ('_scalar_cache', '_fconst_cache', '_byte_offsets', '_aff_cache')
    saved = {n: dict(getattr(imp, n)) for n in caches}
    start, end = imp._control_label('rows'), imp._control_label('endrows')
    b.label(start)
    b.comment(f'rows [r0, r0 + {vlen}) of a {p.size}-row program (row_split)')
    try:
        # The chunk's own row index, wherever a split arange is used: at the
        # top of the iteration, where it dominates every use.
        for op in ranges:
            vid = op.get_result(0).id()
            first = op.get_int_attr('start') or 0
            rid = ('row-split', vid)
            row = p.row0 if not first else b.sbini('add', p.row0, first)
            imp._v[rid] = b.vbin('add', b.arange(), b.splat(row), dtype='i32')
            imp._uscalar[rid] = row
            imp._offsets[rid] = ('contiguous', rid)
        printed = getattr(imp, '_printed_consts', None)
        if printed:
            for vid in list(p.poly_dims):
                if vid in printed:
                    printed[('row-split', vid)] = printed[vid]
        for op in sequence(top_ops):
            dispatch(op)
        nxt = b.sbini('add', p.row0, vlen)
        b._emit('assign', [], [p.row0, nxt])
        b.branch_nz(b.scmp('lt', p.row0, b.const(p.size)), start)
        b.label(end)
    finally:
        imp._bodies = saved_bodies
        for n, cache in saved.items():
            setattr(imp, n, cache)
