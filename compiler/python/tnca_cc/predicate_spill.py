"""Keep excess immutable predicates as exact scalar lane-bit masks.

P.ballot/P.frombits are ordinary ISA operations, not memory spilling. This
only changes programs whose predicate intervals exceed the target capacity.
Mutable loop carries and predicates with an assigned physical register are
left alone. The ordinary scalar allocator still rejects an oversized plan.
"""
from .asm import FIRST_PREG
from .codegen import CodeGen
from .ir import Op, Value, PRED, SCALAR


def peak(kernel, target, ops):
    cg = CodeGen(kernel, target, check_protocol=False)
    cg._compute_last_uses(ops)
    events, starts = {}, {}
    for i, op in enumerate(ops):
        for value in op.results:
            if value.kind == PRED and value not in starts:
                starts[value] = i
                events.setdefault(i, [[], []])[0].append(value)
                events.setdefault(cg._last_use[id(value)] + 1, [[], []])[1].append(value)
    active, maximum = set(), set()
    for i in sorted(events):
        born, dead = events[i]
        active.difference_update(dead)
        active.update(born)
        if len(active) > len(maximum):
            maximum = set(active)
    return maximum, {value: cg._last_use[id(value)] - start for value, start in starts.items()}


def spill(kernel, target):
    capacity = target.pregs_per_rctx - FIRST_PREG
    if target.vlen > 64 or capacity < 3:
        return dict(predicates=0, roles=[])
    rows, total = [], 0
    for role in kernel.roles:
        original = role.region.ops
        live, spans = peak(kernel, target, original)
        if len(live) <= capacity:
            continue
        mutable = {op.operands[0] for op in original if op.kind == 'assign' and op.operands}
        candidates = {value for op in original for value in op.results
                      if value.kind == PRED and value.reg is None and value not in mutable}
        chosen = set()
        current = original
        def rewrite():
            packed = {v: Value(f'predicate_bits_{role.index}_{i}', SCALAR, 'u64')
                      for i, v in enumerate(sorted(chosen, key=lambda v: v.name))}
            result, counter = [], 0
            for op in original:
                reloads = {}
                for value in op.operands + [op.attrs.get('mask')]:
                    if value in packed and value not in reloads:
                        counter += 1
                        temporary = Value(f'predicate_reload_{role.index}_{counter}', PRED, 'i1')
                        result.append(Op('pred_from_scalar', [temporary], [packed[value]]))
                        reloads[value] = temporary
                attrs = dict(op.attrs)
                if attrs.get('mask') in reloads:
                    attrs['mask'] = reloads[attrs['mask']]
                result.append(Op(op.kind, list(op.results),
                                 [reloads.get(v, v) for v in op.operands], attrs))
                for value in op.results:
                    if value in packed:
                        result.append(Op('pred_to_scalar', [packed[value]], [value]))
            return result
        while len(live) > capacity:
            possible = (live & candidates) - chosen
            if not possible:
                break
            chosen.add(max(possible, key=lambda v: (spans.get(v, 0), v.name)))
            current = rewrite()
            live, _ = peak(kernel, target, current)
        if chosen:
            role.region.ops = current
            rows.append(dict(role=role.index, predicates=len(chosen), remaining_predicate_peak=len(live)))
            total += len(chosen)
    return dict(predicates=total, roles=rows)
