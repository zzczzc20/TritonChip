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

# Operations that define a value and do nothing else: no memory, no
# ownership, no barrier, nothing that can fault. Deleting one that
# nothing names changes no observable behaviour.
#
# `_is_pure` is a narrower question -- whether the bridge can express it
# as an arith op -- and a splat is the case that separates them. It has
# no effect and there is no arith op for it, so it can be deleted but
# not sent across.
#
# Floating-point arithmetic is deliberately absent. A dead vf.add is
# very probably removable too, but whether it is depends on what the
# machine does with exception state, and section 7's R35-OPT-03 owns
# that question. Nothing here needs it, so nothing here assumes it.
PURE_NO_EFFECT = frozenset(_is_pure.__globals__['PURE_KINDS']) | {
    'splat', 'vmovi', 'arange', 'scmp', 'vcmp',
}


def _no_effect(op):
    return op.kind in PURE_NO_EFFECT


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
    if op.kind in ('label', 'branch_nz', 'assign'):
        return True
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
                                 'why': 'control flow or mutable assignments'})
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
                    if not _no_effect(o)
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
                                 'why': 'control flow or mutable assignments'})
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
                                 'why': 'control flow or mutable assignments'})
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


# Consumers whose inactive lanes are documented not to matter, and why.
#
#   reduce  ISA sec 8.7: inactive lanes contribute the operation's
#           identity, "so a masked reduction needs no pre-select". The
#           hardware supplies the identity; what the register held is
#           not read.
#   store   a predicated store does not write an inactive lane, so what
#           that lane held was never observable.
#
# Nothing else is on this list. A vector arithmetic op reading a lane of
# rubbish and producing a NaN that a LATER instruction observes would be
# a wrong answer, and the fact that this kernel happens not to do that
# is not a licence for the next one.
_DONT_CARE_CONSUMERS = {'reduce': 'inactive lanes contribute the identity '
                                  '(ISA sec 8.7)',
                        'store': 'a predicated store does not write an '
                                 'inactive lane'}

# Operations whose output lane i depends on input lane i and nothing
# else. Rubbish in an inactive lane stays in that lane, so if the lane is
# never observed at the end of the chain it is never observed at all.
#
# This is safe here because the ISA has no observable floating-point
# state: the special register table (sec 7.2) has no status word, no
# sticky flags and no exception register, so an operation whose result
# nobody reads has no other way to be noticed. A machine with an inexact
# flag would need this list to be empty for floating point.
#
# Cross-lane operations are absent and must stay absent: a gather, a
# scatter, a shuffle or a reduction moves lane i somewhere else, and
# then an inactive lane's rubbish arrives in an active one.
_LANE_LOCAL = {'vbin', 'vun', 'vselect', 'vcmp', 'vmovi', 'splat',
               'vfma', 'vfms', 'vcvt', 'pbin', 'pun'}


def drop_dead_load_other(kernel):
    """Stop materialising `other=` when nothing can observe it.

    A masked load leaves its inactive lanes holding whatever the register
    had, so the compiler writes `other` into the register first. That
    instruction is only needed if something reads those lanes.

    It is dropped only when EVERY consumer of the loaded value is
    predicated by the same predicate value AND is one of the consumers
    above. Same predicate value, not merely an equal-looking one: two
    compares that happen to produce the same mask are still two values,
    and this does not try to prove them equal.

    `other` itself must be the identity the consumer assumes -- zero for
    an add-reduction. A non-zero `other` is left alone, because then the
    program is asking for something and this would silently stop
    providing it.
    """
    out = {'sites': 0, 'roles': [], 'kept': []}
    for role in kernel.roles:
        if any(_opaque(o) for o in role.region.ops):
            out['roles'].append({'role': role.name,
                                 'why': 'control flow or mutable assignments'})
            continue
        ops = role.region.ops
        n = 0
        for op in ops:
            if op.kind != 'load' or op.attrs.get('mask') is None:
                continue
            if op.attrs.get('other') not in (0, 0.0):
                out['kept'].append(f"{role.name}: other= is not zero")
                continue
            if op.attrs.get('no_other'):
                continue
            mask = op.attrs['mask']
            loaded = op.results[0]
            consumers = [o for o in ops
                         if any(v is loaded for v in _values_in(o))]
            if not consumers:
                continue
            why = _observable(loaded, mask, ops, {})
            if why:
                out['kept'].append(f"{role.name}: {why}")
                continue
            op.attrs = dict(op.attrs, no_other=True)
            n += 1
        out['sites'] += n
        out['roles'].append({'role': role.name, 'sites': n})
    return out


def narrow_lane0_store(kernel):
    """A vector store of a broadcast, masked to lane 0, is a scalar store.

    A reduction produces a scalar. Putting it back in memory as a vector
    store costs a broadcast into a register, a mask that selects one lane
    and the vector store itself. The ISA has `s.st` (major LSU, func 8),
    which writes that word directly.

    Conditions, all of them checked rather than assumed:

      * the stored value is a splat of a scalar, so every lane holds the
        same thing and lane 0 is that thing;
      * the mask is `idx < 1` over the lane index, so exactly lane 0 is
        written -- proved from the operations that built it, not from
        the mask's name;
      * nothing else reads the splat or the mask, so removing them
        removes nothing else's operand. (The dead sweep does the
        removing; this only stops them being needed.)

    The address and the bytes written are identical either way: the
    vector store writes lane 0 to [base+0], and so does the scalar one.
    """
    out = {'sites': 0, 'roles': []}
    for role in kernel.roles:
        if any(_opaque(o) for o in role.region.ops):
            out['roles'].append({'role': role.name,
                                 'why': 'control flow or mutable assignments'})
            continue
        ops = role.region.ops
        iota, const_of, splat_of, lane0 = set(), {}, {}, set()
        n = 0
        for op in ops:
            if op.kind == 'arange' and op.results:
                iota.add(id(op.results[0]))
            elif op.kind == 'const' and len(op.results) == 1:
                const_of[id(op.results[0])] = int(op.attrs['value'])
            elif op.kind == 'splat' and len(op.results) == 1 \
                    and len(op.operands) == 1:
                splat_of[id(op.results[0])] = op.operands[0]
            elif op.kind == 'vcmp' and op.attrs.get('cc') == 'lt' \
                    and op.operands and id(op.operands[0]) in iota:
                # `idx < 1`, however the 1 arrived: as an immediate the
                # fold left behind, or as a splat constant.
                v = op.attrs.get('imm')
                if v is None and len(op.operands) > 1:
                    s = splat_of.get(id(op.operands[1]))
                    v = const_of.get(id(s)) if s is not None else None
                if v == 1 and op.results:
                    lane0.add(id(op.results[0]))
            elif op.kind == 'store' and len(op.operands) == 2:
                base, value = op.operands
                mask = op.attrs.get('mask')
                if mask is None or id(mask) not in lane0:
                    continue
                src = splat_of.get(id(value))
                if src is None:
                    continue
                op.kind = 'sstore'
                op.operands = [base, src]
                op.attrs = dict(op.attrs, mask=None, offset=0,
                                comment='lane 0 of a broadcast is one word')
                n += 1
        out['sites'] += n
        out['roles'].append({'role': role.name, 'sites': n})
    return out


def _observable(value, mask, ops, memo):
    """Why this value's inactive lanes can be observed, or '' if they cannot.

    Follows the value forward. A terminal consumer that ignores inactive
    lanes ends that path; a lane-local operation continues it through its
    own result; anything else ends the search with a reason.
    """
    key = id(value)
    if key in memo:
        return memo[key]
    memo[key] = ''          # assume unobservable while we look; SSA is acyclic
    consumers = [o for o in ops if any(v is value for v in _values_in(o))]
    if not consumers:
        # Nothing reads it at all, so nothing observes it either.
        return ''
    for c in consumers:
        if c.kind in _DONT_CARE_CONSUMERS:
            if c.attrs.get('mask') is not mask:
                memo[key] = (f"a {c.kind} under a different predicate reads "
                             f"the inactive lanes")
                return memo[key]
            continue
        if c.kind in _LANE_LOCAL and len(c.results) == 1:
            why = _observable(c.results[0], mask, ops, memo)
            if why:
                memo[key] = why
                return why
            continue
        memo[key] = f"{c.kind} reads the inactive lanes"
        return memo[key]
    return ''


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
            note['why'] = 'the region has control flow or mutable assignments'
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

        # Constants are cheap to rematerialise; merging distant copies can
        # keep dozens of scalars live across every transfer in an unrolled
        # convolution. Bound the lifetime extension instead of exhausting
        # the scalar file to save a single immediate.
        positions = {id(v): i for i, o in enumerate(ops) for v in o.results}
        eq = {key: value for key, value in eq.items()
              if abs(positions.get(key, 0)-positions.get(id(value), 0)) <= 32}
        info['merged'] = sum(key != id(value) for key, value in eq.items())
        info['distinct'] = info['values'] - info['merged']
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
                           if not _no_effect(o)
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
