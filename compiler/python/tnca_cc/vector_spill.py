"""Bound VRF pressure using private per-program scratch for straight-line IR."""
import copy
from .ir import VECTOR, Value


def _values(obj):
    if isinstance(obj, Value):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _values(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _values(value)


def spill(imp):
    region = imp.kernel.role(0).region
    ops = list(region.ops)
    # Crossing a control-flow edge needs dominance/loop analysis. Keep the
    # old explicit resource refusal until that analysis is available.
    if any(op.kind in ('label', 'branch', 'branch_nz', 'jump', 'loop', 'loop_end') for op in ops):
        return {'refused': 'control flow requires spill dominance analysis'}
    definitions, last, order = {}, {}, {}
    for at, op in enumerate(ops):
        for value in op.results:
            if value.kind == VECTOR:
                key = id(value)
                definitions[key] = at
                order[key] = len(order)
                last.setdefault(key, at)
        for value in _values([op.operands, op.attrs]):
            if value.kind == VECTOR:
                last[id(value)] = at
    limit = max(1, imp.target.vregs_per_rctx - 12)
    chosen = set()
    def pressure(at):
        return [key for key, start in definitions.items()
                if key not in chosen and start <= at <= last[key]]
    before = max((len(pressure(at)) for at in range(len(ops))), default=0)
    while True:
        peak, at = max(((len(pressure(at)), at) for at in range(len(ops))), default=(0, 0))
        if peak <= limit:
            break
        candidates = [key for key in pressure(at) if last[key] > definitions[key] + 2]
        if not candidates:
            return {'refused': 'no long-lived vector available to spill', 'estimated_peak': before}
        chosen.add(max(candidates, key=lambda key: last[key] - definitions[key]))
    if not chosen:
        return {'slots': 0, 'scratch_bytes': 0, 'estimated_peak': before}
    bytes_per_register = imp.target.vlen * 4  # physical 32-bit lanes, including packed dtypes
    slots = {key: imp._take_scratch(bytes_per_register) for key in sorted(chosen, key=lambda k: (definitions[k], order[k]))}
    previous = len(region.ops)
    base = imp._scratch_base()
    prefix = region.ops[previous:]
    region.ops[:] = prefix
    loads = 0
    def address(offset):
        return imp.b.sbini('add', base, offset) if offset else base
    def replace(obj, loaded):
        nonlocal loads
        if isinstance(obj, Value) and id(obj) in chosen:
            key = id(obj)
            if key not in loaded:
                loaded[key] = imp.b.load(address(slots[key]), dtype=obj.dtype, size='w')
                loads += 1
            return loaded[key]
        if isinstance(obj, list):
            return [replace(value, loaded) for value in obj]
        if isinstance(obj, tuple):
            return tuple(replace(value, loaded) for value in obj)
        if isinstance(obj, dict):
            return {key: replace(value, loaded) for key, value in obj.items()}
        return obj
    for op in ops:
        loaded = {}
        rewritten = copy.copy(op)
        rewritten.operands = replace(op.operands, loaded)
        rewritten.attrs = replace(op.attrs, loaded)
        region.ops.append(rewritten)
        for value in op.results:
            if id(value) in chosen:
                imp.b.store(address(slots[id(value)]), value, size='w', comment='private VRF spill')
    return {'slots': len(chosen), 'scratch_bytes': len(chosen) * bytes_per_register,
            'stores': len(chosen), 'loads': loads, 'estimated_peak_before': before,
            'estimated_unspilled_peak': peak, 'tier': 'per-program scratch'}
