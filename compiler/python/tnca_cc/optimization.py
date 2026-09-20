"""Optimisations that run on TNCA IR, above O0.

doc/phase3_5_tasks_and_acceptance.md section 4.1 and 4.3.

There is one pass here so far: the pure scalar values of each role are
sent across the MLIR bridge, upstream canonicalisation and CSE decide
which of them are the same value, and this applies that decision.

What it is allowed to change is deliberately narrow. Operands are
repointed at the representative upstream chose, and a pure definition
that nothing names afterwards is dropped. No effectful operation is
moved, removed, duplicated or reordered -- the order of the op list is
untouched apart from those deletions, and a load, store, transfer,
matrix operation, barrier or anything that can fault never crosses the
bridge at all.
"""
from .ir import Op, Value, Region
from .upstream.bridge import PureScalarBridge, BridgeError, _is_pure


def _values_in(op):
    """Every Value this op names, including the ones in attrs.

    A predicate arrives as `attrs['mask']` rather than as an operand, so
    an optimisation that rewrote only operands would leave a mask
    pointing at a definition it had just deleted.
    """
    for v in op.operands:
        if isinstance(v, Value):
            yield v
    for a in op.attrs.values():
        if isinstance(a, Value):
            yield a
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, Value):
                    yield x


def _rewrite(op, eq):
    op.operands = [eq.get(id(v), v) if isinstance(v, Value) else v
                   for v in op.operands]
    for k, a in list(op.attrs.items()):
        if isinstance(a, Value):
            op.attrs[k] = eq.get(id(a), a)
        elif isinstance(a, list):
            op.attrs[k] = [eq.get(id(x), x) if isinstance(x, Value) else x
                           for x in a]


def _opaque(op):
    """True when this op carries structure the pass will not reason about.

    A nested region is a control-flow body, and a value defined outside
    it may be live across it in ways the flat list does not show. Rather
    than model that, a role containing one is left alone.
    """
    for a in op.attrs.values():
        if isinstance(a, Region):
            return True
        if isinstance(a, (list, tuple)) and any(isinstance(x, (Op, Region))
                                                for x in a):
            return True
    return False


def pure_scalar_cse(kernel):
    """Merge each role's duplicate pure scalar values, using upstream.

    Returns what happened, per role, for the report. Never raises for a
    region it cannot express: that region is left exactly as it was and
    says so, which is section 4.3's "not-applicable" rather than a
    failure.
    """
    out = {'roles': [], 'merged': 0, 'removed_ops': 0}
    for role in kernel.roles:
        ops = role.region.ops
        note = {'role': role.name, 'applied': False, 'why': None}

        if any(_opaque(o) for o in ops):
            note['why'] = 'the region has nested control flow'
            out['roles'].append(note)
            continue

        pure = [o for o in ops if _is_pure(o) and len(o.results) == 1]
        if len(pure) < 2:
            note['why'] = f"{len(pure)} pure scalar value(s); nothing to pair"
            out['roles'].append(note)
            continue

        # Only a forest whose operands are themselves in the forest can
        # cross: a pure op reading a loaded value is pure, but upstream
        # would need that value, and it is not ours to send.
        known = set()
        forest = []
        for o in pure:
            if all(id(v) in known for v in o.operands):
                forest.append(o)
                known.add(id(o.results[0]))
        if len(forest) < 2:
            note['why'] = 'no closed pure scalar forest'
            out['roles'].append(note)
            continue

        try:
            eq, info = PureScalarBridge().equivalence(forest)
        except BridgeError as e:
            note['why'] = f"not expressible: {e}"
            out['roles'].append(note)
            continue

        for o in ops:
            _rewrite(o, eq)

        # A pure definition nothing names any more is dead. Effectful ops
        # are never dropped, whatever names them.
        live = set()
        for o in ops:
            for v in _values_in(o):
                live.add(id(v))
        before = len(ops)
        role.region.ops = [o for o in ops
                           if not _is_pure(o)
                           or any(id(r) in live for r in o.results)]
        note.update(applied=True, merged=info['merged'],
                    values=info['values'], distinct=info['distinct'],
                    removed_ops=before - len(role.region.ops),
                    upstream=info['removed'], pipeline=info['pipeline'],
                    protocol=info['protocol'])
        out['merged'] += info['merged']
        out['removed_ops'] += note['removed_ops']
        out['roles'].append(note)
    return out
