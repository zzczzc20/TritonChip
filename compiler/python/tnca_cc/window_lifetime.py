"""Color single-role windows only after their asynchronous readers finish.

The backing registers stay reserved against ordinary temporaries. Reuse is
between windows with disjoint complete lifetimes, including alias reads and
loop back edges. Shared producer/consumer roles retain the fixed layout.
"""
from .diagnostics import ResourceError
from .ir import TOKEN


def assign(kernel, limit, last_uses):
    if len(kernel.roles) != 1:
        raise ResourceError('window lifetime reuse requires a single IR role')
    if not kernel.windows:
        return 0
    ops = kernel.role(0).region.ops
    labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
    loops = [(labels[op.attrs['target']], i) for i, op in enumerate(ops)
             if op.kind == 'branch_nz' and labels.get(op.attrs['target'], len(ops)) <= i]
    for i, op in enumerate(ops):
        if op.kind == 'branch_nz':
            destination = labels.get(op.attrs['target'], len(ops))
            guarded_loop = any(destination == end + 1 and i <= end and
                all(not any(getattr(v, 'kind', None) == 'window'
                            for v in inner.operands + inner.results)
                    for inner in ops[min(i + 1, start):max(i, start)])
                for start, end in loops)
            if destination > i and not guarded_loop:
                raise ResourceError('window reuse requires straight-line code or structured loops',
                                    because=f'branch {i} -> {op.attrs["target"]} ({destination}); '
                                            f'loop intervals {loops}')
    windows = {id(w): w for w in kernel.windows}
    first, last = {}, {}
    aliases = {}
    waits = {}
    for i, op in enumerate(ops):
        if op.kind == 'barrier_wait':
            waits.setdefault(id(op.operands[0]), []).append(i)
        if op.kind == 'window_reg':
            aliases[id(op.results[0])] = id(op.operands[0])
    for i, op in enumerate(ops):
        for value in op.operands:
            key = id(value)
            if key in windows:
                first.setdefault(key, i)
                last[key] = max(last.get(key, i), last_uses.get(key, i))
            if key in aliases:
                window = aliases[key]
                first.setdefault(window, i)
                last[window] = max(last.get(window, i), last_uses.get(key, i))
        if op.kind in ('dma_load', 'dma_store', 'mma_async'):
            token = next(v for v in op.operands if getattr(v, 'kind', None) == TOKEN)
            completion = next((j for j in waits.get(id(token), []) if j > i), len(ops))
            issued_in = {(start, end) for start, end in loops if start <= i <= end}
            waited_in = {(start, end) for start, end in loops if start <= completion <= end}
            if issued_in != waited_in:
                # A loop may execute zero times or more than once. An await
                # in another loop scope is not an unconditional completion.
                completion = len(ops)
            for value in op.operands:
                if id(value) in windows:
                    last[id(value)] = max(last[id(value)], completion)
    layouts = []
    peak = 0
    for key in sorted(first, key=lambda k: (first[k], -windows[k].length)):
        window = windows[key]
        candidates = range(limit - window.length + 1)
        start = next((a for a in candidates if all(
            last[other] < first[key] or last[key] < first[other] or
            a + window.length <= b or b + windows[other].length <= a
            for other, b in layouts)), None)
        if start is None:
            raise ResourceError('overlapping live windows exceed the vector register capacity',
                                remedy='retile or shorten the awaited window lifetimes')
        layouts.append((key, start))
        peak = max(peak, start + window.length)
    for key, start in layouts:
        windows[key].reg = limit - peak + start
    for key, window in windows.items():
        if key not in first:
            if window.length > limit:
                raise ResourceError('unused window still exceeds the hardware window size')
            window.reg = limit - window.length
    return peak
