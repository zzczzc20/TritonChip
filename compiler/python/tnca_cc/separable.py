"""Separable 2-D indices: a per-row part and a per-column part.

An `Affine2D` (tiles.py) is `row_stride*row + col_stride*col + c` with
compile-time strides. Triton's block kernels are often not that:

    offs_am = (pid_m * BM + tl.arange(0, BM)) % M      # rows wrap at M
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] + offs_bn[None, :] * stride_bn

The row index is data (a remainder), and the strides are runtime arguments.
What such an index still is, is separable: every term depends on the row or
on the column, never on both,

    offset(r, c) = S(r) + L(c)

so row r of the tile is ONE scalar address S(r) and one lane vector L that
is the same for every row. A contiguous L (the plain lane index) makes row r
a single predicated vector load; any other L is one gather a row. Without
this the importer materialised the whole index as an i32 tile in scratch
and read it back for every access, and re-wrote it on every iteration a
pointer advanced.

`Sep2D` is that sum, as terms

    ('one',    None, k, s)   k * s1 * s2 ...            (a scalar)
    ('row',    None, k, s)   r * k * s...               (the row index)
    ('col',    None, k, s)   lane * k * s...
    ('rowvec', v,    k, s)   v[r] * k * s...            (lane r of a 1-D value)
    ('colvec', v,    k, s)   v * k * s...               (a 1-D value across lanes)

with `k` an integer and `s` a tuple of scalar Values (runtime factors).
Only the extended lowering builds one (frontend `large_tiles`); the default
lowering's code is unchanged.
"""


class Sep2D:
    __slots__ = ('terms',)

    def __init__(self, terms=()):
        self.terms = [t for t in terms if t[2] != 0]

    @classmethod
    def from_affine(cls, aff):
        terms = [('row', None, aff.row_stride, ()), ('col', None, aff.col_stride, ()),
                 ('one', None, aff.const, ())]
        terms += [('one', None, c, (v,)) for v, c in aff.dynamic]
        return cls(terms)

    def plus(self, other):
        return Sep2D(self.terms + other.terms)

    def scaled(self, k, scalars=()):
        return Sep2D([(kind, v, c * k, s + tuple(scalars)) for kind, v, c, s in self.terms])

    def scalar_only(self):
        return all(kind == 'one' for kind, _v, _c, _s in self.terms)

    def lane_terms(self):
        return [t for t in self.terms if t[0] in ('col', 'colvec')]

    def row_terms(self):
        return [t for t in self.terms if t[0] not in ('col', 'colvec')]

    def contiguous(self):
        """True when the lane part is exactly the lane index."""
        lanes = self.lane_terms()
        return len(lanes) == 1 and lanes[0][0] == 'col' and lanes[0][2] == 1 and not lanes[0][3]

    def __repr__(self):
        return 'Sep2D(' + ' + '.join(f'{k}*{kind}' + ('*dyn' if s else '')
                                     for kind, _v, k, s in self.terms) + ')'


class Emitter:
    """Code for a Sep2D's parts, with the runtime factors computed once."""

    def __init__(self, imp):
        self.imp = imp
        self.b = imp.b
        # Scalars that are compile-time constants fold into the coefficient.
        self._known = {id(v): k for k, v in imp._scalar_cache.items()}

    @property
    def _cache(self):
        # The importer's index cache, which a loop or a branch saves and
        # restores (frontend._op_scf_for): a value computed in a loop body is
        # never reused after it. Read at each use, not held.
        return self.imp._aff_cache

    def _fold(self, k, scalars):
        rest = []
        for s in scalars:
            known = self._known.get(id(s))
            if known is None:
                rest.append(s)
            else:
                k *= known
        return k, tuple(rest)

    def _factor(self, k, scalars):
        """k * s1 * s2 ... as a scalar Value (or an int when there are none)."""
        k, scalars = self._fold(k, scalars)
        if not scalars or k == 0:
            return k if scalars == () else 0
        key = ('sep-factor', k, tuple(id(s) for s in scalars))
        v = self._cache.get(key)
        if v is None:
            v = scalars[0]
            for s in scalars[1:]:
                v = self.b.sbin('mul', v, s)
            if k != 1:
                v = self.b.sbini('mul', v, k)
            self._cache[key] = v
        return v

    def scalar(self, sep, r, *, scale=1):
        """S(r) * scale, in elements times `scale` (bytes when scale is the
        element size): an int when every term is a compile-time constant,
        else a scalar Value."""
        b = self.b
        const = 0
        total = None

        def add(v):
            nonlocal total
            total = v if total is None else b.sbin('add', total, v)

        rowvecs, ones = [], []
        for kind, value, k, scalars in sep.row_terms():
            f = self._factor(k * scale, scalars)
            if kind == 'rowvec':
                rowvecs.append((value, f))
            elif isinstance(f, int):
                const += f * (r if kind == 'row' else 1)
            elif kind == 'one':
                ones.append(f)
            elif r:
                ones.append(b.sbini('mul', f, r))
        # The lane-indexed part as one vector, built once; row r is one
        # extract of it. (Holding a scalar per row ran the 64 scalar
        # registers out on a 16-row tile split eight ways.)
        if rowvecs:
            key = ('sep-rowvec', tuple((id(v), f if isinstance(f, int) else ('s', id(f)))
                                       for v, f in rowvecs))
            vec = self._cache.get(key)
            if vec is None:
                for v, f in rowvecs:
                    if isinstance(f, int):
                        term = v if f == 1 else b.vbin('mul', v, b.splat(b.const(f), dtype='i32'),
                                                       dtype='i32')
                    else:
                        term = b.vbin('mul', v, b.splat(f, dtype='i32'), dtype='i32')
                    vec = term if vec is None else b.vbin('add', vec, term, dtype='i32')
                self._cache[key] = vec
            add(b.vextract(vec, r))
        if ones:
            key = ('sep-ones', tuple(sorted(id(f) for f in ones)))
            total_ones = self._cache.get(key)
            if total_ones is None:
                total_ones = ones[0]
                for f in ones[1:]:
                    total_ones = b.sbin('add', total_ones, f)
                self._cache[key] = total_ones
            add(total_ones)
        if total is None:
            return const
        return total if not const else b.sbini('add', total, const)

    def lanes(self, sep):
        """L, the lane part in elements, as a vector Value; None when it is
        the plain lane index (a contiguous row)."""
        if sep.contiguous():
            return None
        key = ('sep-lanes', tuple(sorted((kind, id(v) if v is not None else None, k,
                                          tuple(id(x) for x in s))
                                         for kind, v, k, s in sep.lane_terms())))
        held = self._cache.get(key)
        if held is not None:
            return held
        b = self.b
        out = None
        for kind, value, k, scalars in sep.lane_terms():
            base = b.arange() if kind == 'col' else value
            f = self._factor(k, scalars)
            if isinstance(f, int):
                term = base if f == 1 else b.vbin('mul', base, b.splat(b.const(f), dtype='i32'),
                                                  dtype='i32')
            else:
                term = b.vbin('mul', base, b.splat(f, dtype='i32'), dtype='i32')
            out = term if out is None else b.vbin('add', out, term, dtype='i32')
        if out is None:
            out = b.vmovi(0)
        self._cache[key] = out
        return out

    def row_value(self, sep, r):
        """Row r of the index itself, as data (a mask compares it)."""
        s = self.scalar(sep, r)
        lanes = self.lanes(sep)
        b = self.b
        if lanes is None:
            lanes = b.arange()
        if isinstance(s, int):
            if s == 0:
                return lanes
            s = b.const(s)
        return b.vbin('add', lanes, b.splat(s, dtype='i32'), dtype='i32')
