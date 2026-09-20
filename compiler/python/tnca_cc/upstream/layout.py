"""The real upstream LinearLayout, called directly.

doc/phase3_5_tasks_and_acceptance.md section 3 and R35-UP-01/R35-LL-01.

Nothing here computes a layout. `libtriton.linear_layout.LinearLayout`
does that; this validates what goes in, because upstream answers a
domain error by aborting the process rather than raising, and it names
the TNCA meaning of each dimension, because upstream has no opinion
about what a "lane" is.

Section 3.1 draws the line this file has to hold:

    logical tensor indices + valid mask
          ^ LL mapping                        <- upstream owns this
    virtual register / lane / packed element / MMA fragment position
          v TNCA allocation + window constraints   <- we own this
    physical VRF register -> bank / row

So a layout here maps a *virtual* position to tensor indices. It says
nothing about banks: a bank is a property of the physical register a
later allocation pass chooses, and section 3.1 is explicit that reading
bank behaviour off a virtual layout is not allowed.
"""
from .version import require, UpstreamError


class LayoutError(Exception):
    """A layout that upstream would have aborted on, refused first."""


def available():
    """True when the pinned upstream LinearLayout can be called."""
    try:
        require('layout')
        return True
    except UpstreamError:
        return False


def _ll():
    require('layout')
    from triton._C import libtriton
    return libtriton.linear_layout.LinearLayout


def _check_bases(bases, out_dims, out_sizes):
    if not bases:
        raise LayoutError('a layout with no input dimension')
    for name, rows in bases:
        if not isinstance(name, str) or not name:
            raise LayoutError(f"input dimension {name!r} is not a name")
        for row in rows:
            if len(row) != len(out_dims):
                raise LayoutError(
                    f"input dimension {name!r} has a basis of width "
                    f"{len(row)} and there are {len(out_dims)} output "
                    f"dimension(s) {list(out_dims)} -- upstream aborts the "
                    f"process on this, so it is refused here")
    if len(out_dims) != len(set(out_dims)):
        raise LayoutError(f"repeated output dimension in {list(out_dims)}")
    if out_sizes is not None and len(out_sizes) != len(out_dims):
        raise LayoutError(
            f"{len(out_sizes)} size(s) for {len(out_dims)} dimension(s)")
    for s in (out_sizes or ()):
        if s <= 0 or (s & (s - 1)):
            raise LayoutError(
                f"output size {s} is not a positive power of two; "
                f"LinearLayout is a GF(2) map and cannot represent it. "
                f"Section 3.1 keeps general strides and non-power-of-two "
                f"bounds in ordinary index arithmetic.")


class LinearLayoutAdapter:
    """One TNCA layout, computed by upstream.

    `bases` is [(in_dim, [[out...], ...]), ...] exactly as upstream takes
    it. `out_dims` and `out_sizes` name and bound the tensor side.
    """

    def __init__(self, bases, out_dims, out_sizes=None,
                 require_surjective=True, *, what=''):
        _check_bases(bases, out_dims, out_sizes)
        self.what = what
        self.out_dims = list(out_dims)
        try:
            self._ll = _ll().from_bases(
                [(n, [list(map(int, r)) for r in rows]) for n, rows in bases],
                list(out_dims),
                None if out_sizes is None else [int(s) for s in out_sizes],
                bool(require_surjective))
        except Exception as e:                            # noqa: BLE001
            raise LayoutError(
                f"upstream refused this layout{' for ' + what if what else ''}"
                f": {type(e).__name__}: {e}") from e

    # -- what upstream says about it ---------------------------------------
    @property
    def in_dims(self):
        return list(self._ll.get_in_dim_names())

    def is_injective(self):
        return bool(self._ll.is_injective())

    def is_surjective(self):
        return bool(self._ll.is_surjective())

    def is_invertible(self):
        return bool(self._ll.is_invertible())

    def apply(self, inputs):
        """One position to its tensor indices."""
        if not isinstance(inputs, dict):
            raise LayoutError('apply takes a dict of input dimensions')
        missing = set(self.in_dims) - set(inputs)
        if missing:
            raise LayoutError(f"apply is missing {sorted(missing)}")
        extra = set(inputs) - set(self.in_dims)
        if extra:
            raise LayoutError(f"apply was given {sorted(extra)}, which this "
                              f"layout does not take")
        return dict(self._ll.apply({k: int(v) for k, v in inputs.items()}))

    def compose(self, other):
        """This layout followed by `other`.

        Refused unless the dimensions line up: upstream prints
        `LLVM ERROR: Dimensions must match` and exits, which would take
        the compiler with it.
        """
        if not isinstance(other, LinearLayoutAdapter):
            raise LayoutError('compose takes another adapter')
        mine, theirs = set(self.out_dims), set(other.in_dims)
        if mine != theirs:
            raise LayoutError(
                f"cannot compose: this layout produces {sorted(mine)} and "
                f"the next consumes {sorted(theirs)}. Upstream aborts the "
                f"process on this rather than raising.")
        out = LinearLayoutAdapter.__new__(LinearLayoutAdapter)
        out.what = f"{self.what} o {other.what}".strip(' o')
        out._ll = self._ll.compose(other._ll)
        out.out_dims = list(out._ll.get_out_dim_names())
        return out

    def invert(self):
        """The inverse, when there is exactly one.

        Section 3.1: a broadcast is not a bijection and must not be
        inverted by pretending it is. This refuses rather than returning
        one of several sources.
        """
        if not self.is_invertible():
            raise LayoutError(
                f"this layout{' (' + self.what + ')' if self.what else ''} "
                f"is not invertible: injective={self.is_injective()}, "
                f"surjective={self.is_surjective()}. A broadcast has "
                f"several sources for one output and section 3.1 requires "
                f"one to be chosen explicitly, not produced by inversion.")
        out = LinearLayoutAdapter.__new__(LinearLayoutAdapter)
        out.what = f"{self.what}^-1"
        out._ll = self._ll.invert()
        out.out_dims = list(out._ll.get_out_dim_names())
        return out

    # -- what this stage needs on top --------------------------------------
    def enumerate(self, sizes):
        """Every input position and where it lands. For the oracle.

        R35-LL-01 compares this against an independent enumeration, so
        this deliberately does the dullest possible thing: walk the
        product of the input ranges and ask upstream about each one.
        """
        names = self.in_dims
        missing = [n for n in names if n not in sizes]
        if missing:
            raise LayoutError(f"no size given for {missing}")
        out = {}
        total = 1
        for n in names:
            total *= sizes[n]
        if total > 4096:
            raise LayoutError(
                f"{total} positions is more than this enumerates; "
                f"section 6.2 caps a host case at 64 and this is meant "
                f"for small tiles")
        idx = [0] * len(names)
        for _ in range(total):
            key = tuple(idx)
            out[key] = self.apply(dict(zip(names, idx)))
            for d in range(len(names) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < sizes[names[d]]:
                    break
                idx[d] = 0
        return out

    def covers(self, sizes, out_sizes):
        """Which outputs are hit, and how often.

        R35-LL-02: a conversion must cover every valid element with no
        duplicate write and no missing output. This reports the counts
        and lets the caller decide, because a broadcast legitimately
        hits an output more than once.
        """
        hits = {}
        for _pos, o in self.enumerate(sizes).items():
            hits[tuple(o[d] for d in self.out_dims)] = \
                hits.get(tuple(o[d] for d in self.out_dims), 0) + 1
        total = 1
        for d in self.out_dims:
            total *= out_sizes[d]
        return {'distinct_outputs': len(hits), 'expected_outputs': total,
                'covered': len(hits) == total,
                'max_multiplicity': max(hits.values()) if hits else 0,
                'unhit': total - len(hits)}


def identity(size, in_dim, out_dim):
    """The identity of `size`, from upstream."""
    require('layout')
    out = LinearLayoutAdapter.__new__(LinearLayoutAdapter)
    out.what = f"identity({size})"
    out._ll = _ll().identity_1d(int(size), in_dim, out_dim)
    out.out_dims = list(out._ll.get_out_dim_names())
    return out
