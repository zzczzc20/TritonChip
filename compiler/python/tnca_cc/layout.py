"""What a layout view costs, decided with the upstream LinearLayout.

doc/phase3_5_tasks_and_acceptance.md sections 3.1, 3.2 and R35-LL-01/03.

Section 3.2 says every conversion must fall into one of three classes
and be recorded as such:

  1. a compile-time rename: no value moves, zero extra data instructions
  2. realisable with instructions that exist: name them, and charge for
     the temporaries, the bytes and the waits
  3. not expressible on this ISA: a diagnostic with a position, or keep
     the original layout

This decides which, and the deciding is done by upstream. A chain of
views is composed with `LinearLayout.compose` and the result compared
against the identity that `LinearLayout.identity_1d` produces -- so the
question "do these two views cancel?" is answered by the same algebra
Triton uses, not by a look at the bases with our own eyes.

Section 3.1's boundary is kept: a layout here maps a VIRTUAL position to
tensor indices. It says nothing about banks. A bank is a property of the
physical register that allocation chooses later, and reading bank
behaviour off a virtual layout is exactly what section 3.1 forbids.
"""
from .ir import Value
from .upstream.layout import LinearLayoutAdapter, LayoutError, identity

# The three answers, spelled as section 3.2 spells them.
RENAME = 'rename'
INSTRUCTIONS = 'instructions'
REFUSED = 'refused'


class ViewPlan:
    """One view chain, and what the compiler decided about it."""

    def __init__(self, ops, category, why, layout=None):
        self.ops = list(ops)
        self.category = category
        self.why = why
        self.layout = layout

    def __repr__(self):
        return f"<{self.category}: {self.why}>"


def _adapter(op):
    return LinearLayoutAdapter(
        [(n, rows) for n, rows in op.attrs['bases']],
        op.attrs['out_dims'], op.attrs['out_sizes'],
        what=op.attrs.get('what') or 'view')


def classify(ops, use_upstream=True):
    """Every view chain in `ops`, classified.

    A chain is a view whose input is another view. They are composed and
    the composition is what gets classified -- which is the whole point:
    two views that cancel cost nothing together and something each.

    `use_upstream=False` is the ablation. It does not compose and does
    not compare; every view is materialised. That is a legal plan and it
    is what a compiler without layout algebra would have to do, which is
    what section 4.3's O2-no-LL asks for: "关闭 LL 驱动布局/分配决策,
    使用 legacy 合法计划; 不能删计算".
    """
    views = [o for o in ops if o.kind == 'view' and o.results]
    by_result = {id(o.results[0]): o for o in views}
    # A chain is walked back from its TAIL. Starting from whichever view
    # came first in the program gives every chain one element, because
    # the one it feeds has not been reached yet -- and then nothing ever
    # composes with anything.
    feeds_a_view = {id(by_result[id(o.operands[0])])
                    for o in views if id(o.operands[0]) in by_result}
    chains, seen = [], set()
    for o in views:
        if id(o) in feeds_a_view or id(o) in seen:
            continue
        chain = [o]
        seen.add(id(o))
        while True:
            src = chain[0].operands[0]
            prev = by_result.get(id(src))
            if prev is None or id(prev) in seen:
                break
            chain.insert(0, prev)
            seen.add(id(prev))
        chains.append(chain)
    # Any view not reached from a tail (a cycle cannot happen in SSA, but
    # a view whose consumer was removed can) is its own chain.
    for o in views:
        if id(o) not in seen:
            chains.append([o])
            seen.add(id(o))

    plans = []
    for chain in chains:
        if not use_upstream:
            plans.append(ViewPlan(
                chain, INSTRUCTIONS,
                f"{len(chain)} view(s) materialised one at a time: the "
                f"layout algebra is switched off, so there is nothing "
                f"that could notice they compose"))
            continue
        try:
            net = _adapter(chain[0])
            for nxt in chain[1:]:
                net = net.compose(_adapter(nxt))
        except LayoutError as e:
            plans.append(ViewPlan(chain, REFUSED, str(e)))
            continue

        size = 1
        for s in chain[0].attrs['out_sizes']:
            size *= int(s)
        in_dim = net.in_dims[0] if net.in_dims else 'reg'
        try:
            ident = identity(size, in_dim, net.out_dims[0])
        except LayoutError as e:                            # pragma: no cover
            plans.append(ViewPlan(chain, REFUSED, str(e)))
            continue

        # Is the composition the identity? Ask by evaluating both on
        # every position: upstream computed the composition, and this
        # compares two of its answers rather than inspecting bases.
        same = (len(net.out_dims) == 1 and
                all(net.apply({in_dim: i}) == ident.apply({in_dim: i})
                    for i in range(size)))
        if same:
            plans.append(ViewPlan(
                chain, RENAME,
                f"{len(chain)} view(s) compose to the identity over "
                f"{size} position(s), so no value moves", net))
        else:
            cover = net.covers({in_dim: size},
                               {d: s for d, s in zip(
                                   net.out_dims,
                                   chain[-1].attrs['out_sizes'])})
            if not cover['covered']:
                plans.append(ViewPlan(
                    chain, REFUSED,
                    f"the composition leaves {cover['unhit']} output "
                    f"position(s) with no source; section 3.2 refuses "
                    f"rather than inventing one"))
            else:
                plans.append(ViewPlan(
                    chain, INSTRUCTIONS,
                    f"a real permutation over {size} position(s), "
                    f"realised with per-lane addressing", net))
    return plans


def index_vector_recipe(net, size, in_dim, out_sizes):
    """How to build this layout's index vector from the lane index.

    `v.permute Vd, Va, Vb` reads `Vd[i] = Va[Vb[i]]`, so materialising a
    view needs Vb: lane i holding the position the layout sends i to.

    A LinearLayout is linear over GF(2), which is what makes this
    constructible at all: the image of i is the XOR of the bases its set
    bits select. So for each bit k, take that bit of the lane index,
    spread it to a full mask, AND it with basis k, and XOR it in.

    Returned as a recipe rather than emitted here, so the caller can
    count what it will cost before deciding to pay it -- which is what
    section 3.2 means by charging for a conversion.
    """
    bits = max(1, (size - 1).bit_length())
    basis = []
    for k in range(bits):
        # Where does 2^k land? That is basis k, flattened to a position.
        out = net.apply({in_dim: 1 << k})
        # Row-major over the output dimensions, which is how a tile sits
        # in the lanes: the last dimension is contiguous.
        flat, stride = 0, 1
        for d, s in reversed(list(zip(net.out_dims, out_sizes))):
            flat += out[d] * stride
            stride *= int(s)
        basis.append(flat)
    # iota, then per bit: lshri, andi, sub-from-zero, andi, xor
    return {'bits': bits, 'basis': basis,
            'instructions': 1 + 5 * bits,
            'temporaries': 3}


def apply_plans(kernel, use_upstream=True):
    """Rewrite each role's views according to their plan.

    A rename is deleted and its users repointed at the source. Anything
    else is left for codegen to materialise, which is the honest cost.
    """
    out = {'roles': [], 'renamed': 0, 'materialised': 0, 'refused': 0}
    for role in kernel.roles:
        ops = role.region.ops
        plans = classify(ops, use_upstream=use_upstream)
        note = {'role': role.name,
                'plans': [(p.category, p.why) for p in plans]}
        drop = set()
        repoint = {}
        for p in plans:
            if p.category != RENAME:
                out['materialised' if p.category == INSTRUCTIONS
                    else 'refused'] += len(p.ops)
                # Each view in a materialised chain is emitted on its
                # own: the ablation is not allowed to quietly get the
                # composition it just turned off.
                for o in p.ops:
                    o.attrs['perm_basis'] = declared_basis(o)
                continue
            # The chain's result is its input, unchanged.
            src = p.ops[0].operands[0]
            for o in p.ops:
                repoint[id(o.results[0])] = src
                drop.add(id(o))
            out['renamed'] += len(p.ops)
        if repoint:
            # Resolve chains: a view of a view maps to the original.
            for o in ops:
                o.operands = [_final(repoint, v) for v in o.operands]
                for k, a in list(o.attrs.items()):
                    if isinstance(a, Value):
                        o.attrs[k] = _final(repoint, a)
            role.region.ops = [o for o in ops if id(o) not in drop]
        note['dropped'] = len(drop)
        out['roles'].append(note)
    return out


def declared_basis(op):
    """The permutation this ONE view declares, read straight off it.

    No composition and no upstream call. A view states its bases, and
    flattening them row-major over its output dimensions is arithmetic,
    not layout algebra.

    That distinction is what keeps the ablation honest. O2-no-LL is
    supposed to have the layout ALGEBRA switched off, not to be unable
    to emit a permutation at all; if it reached for LinearLayout to
    build its index vector, "without LL" would not mean anything.

    Both sides use the same flattening, and it has to be said out loud
    because getting it wrong is silent: a tile sits in the lanes
    ROW-MAJOR, so the LAST dimension is the one that varies fastest and
    therefore occupies the low bits of the lane index. Concatenating the
    input dimensions in declaration order instead put `row` in the low
    bits, and then a transpose composed with its own inverse came out as
    a transpose -- the right answer for the layout that was written down
    and the wrong one for the layout that was meant. Section 3.1 asks
    for exactly this correspondence to be explicit, and this is it.
    """
    sizes = [int(s) for s in op.attrs['out_sizes']]
    basis = []
    for _name, rows in reversed(list(op.attrs['bases'])):
        for row in rows:
            flat, stride = 0, 1
            for value, s in reversed(list(zip(row, sizes))):
                flat += int(value) * stride
                stride *= s
            basis.append(flat)
    return basis


def _in_sizes(op):
    """How many positions the view's input has.

    One basis row per bit of the input index, so the input size is two
    to the number of rows.
    """
    return [1 << len(rows) for _n, rows in op.attrs['bases']]


def _final(repoint, v):
    seen = set()
    while isinstance(v, Value) and id(v) in repoint and id(v) not in seen:
        seen.add(id(v))
        v = repoint[id(v)]
    return v
