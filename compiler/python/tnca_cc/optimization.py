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


def drop_dead_pure(kernel):
    """Remove pure scalar definitions that nothing names.

    Its own pass rather than a tail of the CSE, because the two have
    different reasons to run. Strength reduction leaves a constant with
    no users whether or not any value merged, and a kernel with a single
    pure value has nothing to merge but can still have something dead.

    Only pure definitions go. An effectful operation is never removed
    for being unused: a store nobody reads is still a store.
    """
    out = {'removed': 0, 'roles': []}
    for role in kernel.roles:
        if any(_opaque(o) for o in role.region.ops):
            out['roles'].append({'role': role.name,
                                 'why': 'nested control flow'})
            continue
        # To a fixed point: dropping one definition can make the one it
        # named dead in turn.
        n = 0
        while True:
            live = set()
            for o in role.region.ops:
                for v in _values_in(o):
                    live.add(id(v))
            keep = [o for o in role.region.ops
                    if not _is_pure(o)
                    or any(id(r) in live for r in o.results)]
            if len(keep) == len(role.region.ops):
                break
            n += len(role.region.ops) - len(keep)
            role.region.ops = keep
        out['removed'] += n
        out['roles'].append({'role': role.name, 'removed': n})
    return out


def strength_reduce(kernel):
    """Multiply by a constant power of two becomes a shift.

    This one is OURS, and saying so matters. Upstream canonicalisation
    was measured on the exact module where this shows up -- the `pid *
    16` of every Triton elementwise kernel -- and it leaves arith.muli
    alone and produces no arith.shli. So this is a TNCA lowering
    decision, and reporting it as an upstream win would be false.

    What it saves is not the multiply. It is the constant: `s.mul` needs
    its operand in a register, and materialising it costs a whole
    instruction (four, before the constant lowering). `s.shli` takes the
    amount as an immediate, so the constant loses its only user and the
    dead-value sweep takes it.
    """
    out = {'sites': 0, 'roles': []}
    for role in kernel.roles:
        if any(_opaque(o) for o in role.region.ops):
            out['roles'].append({'role': role.name,
                                 'why': 'nested control flow'})
            continue
        const_of = {}
        n = 0
        for op in role.region.ops:
            if op.kind == 'const' and len(op.results) == 1:
                const_of[id(op.results[0])] = int(op.attrs['value'])
                continue
            if op.kind != 'sbin' or op.attrs.get('op') != 'mul':
                continue
            if len(op.operands) != 2:
                continue
            a, b = op.operands
            v = const_of.get(id(b))
            # Only the right operand: multiplication commutes, but the
            # IR's operand order is the source's, and swapping it here
            # would make the dump disagree with the program.
            if v is None or v <= 0 or (v & (v - 1)):
                continue
            shift = v.bit_length() - 1
            op.kind = 'sbini'
            op.operands = [a]
            op.attrs = dict(op.attrs, op='shl', imm=shift,
                            comment=f"x{v} as a shift")
            n += 1
        out['sites'] += n
        out['roles'].append({'role': role.name, 'sites': n})
    return out


# The widest signed immediate `v.cmpi` takes (ISA sec 0x38).
_CMPI_MIN, _CMPI_MAX = -(1 << 15), (1 << 15) - 1


def fold_broadcast_compare(kernel):
    """A compare against a broadcast constant becomes an immediate compare.

    `v.cmp` takes two vectors, so comparing against a constant costs
    three instructions: materialise the scalar, splat it, compare. The
    ISA has `v.cmpi` with a signed 16-bit immediate, which is one.

    This is ours, not upstream's: at this point the splat and the compare
    are TNCA vector operations that no MLIR pass has an opinion about.

    The splat is left in place. It is a pure value, and if the compare
    was its only user the dead sweep takes it; if something else reads
    it, removing it here would be wrong.
    """
    out = {'sites': 0, 'roles': []}
    for role in kernel.roles:
        if any(_opaque(o) for o in role.region.ops):
            out['roles'].append({'role': role.name,
                                 'why': 'nested control flow'})
            continue
        const_of, splat_of = {}, {}
        n = 0
        for op in role.region.ops:
            if op.kind == 'const' and len(op.results) == 1:
                const_of[id(op.results[0])] = int(op.attrs['value'])
            elif op.kind == 'splat' and len(op.results) == 1 \
                    and len(op.operands) == 1:
                v = const_of.get(id(op.operands[0]))
                if v is not None:
                    splat_of[id(op.results[0])] = v
            elif op.kind == 'vcmp' and len(op.operands) == 2:
                a, b = op.operands
                # Floating point stays alone: the immediate form is an
                # integer one, and reading a float's bits as an integer
                # immediate would change the comparison.
                if _is_float(a):
                    continue
                v = splat_of.get(id(b))
                if v is None or not (_CMPI_MIN <= v <= _CMPI_MAX):
                    continue
                op.operands = [a]
                op.attrs = dict(op.attrs, imm=v)
                n += 1
        out['sites'] += n
        out['roles'].append({'role': role.name, 'sites': n})
    return out


def _is_float(v):
    return str(getattr(v, 'dtype', '')).startswith(('f', 'bf'))


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
