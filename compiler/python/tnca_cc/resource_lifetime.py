"""Number barriers, accumulators, descriptors and window ids by lifetime, when
the fixed numbering does not fit.

sec 10.1 gives each of these a fixed slot for the whole program, and
`TargetInfo.check_budget` refuses a plan that needs more. That is right for a
pipelined kernel whose roles share them, and it is what every kernel that fits
keeps. A kernel that issues a fragment per K step and column -- a dot over a
128-deep K, or a 128-wide N -- names far more of them than the target has,
while only a few are ever live at once: each matrix command is armed, issued
and awaited before the next. Those are numbered here by live range instead:
two values share a slot only when one's last use precedes the other's first,
in a single role, with loops taken into account.

The live range of a value is the interval of operation indices from its first
to its last use in program order. A loop is the interval between a label and
the backward branch to it. A value used inside a loop is live across the back
edge -- so its interval covers the whole loop -- when it is also used outside
the loop, or when its first use inside is not a definition: a barrier that is
not armed first, an accumulator that is not zeroed or written by an MMA into
zero first, a window not defined (`w.def`) first. A window's id is what is
renumbered (`wid`): its registers are tnca_cc.window_lifetime's. Asynchronous
completion is a use: a barrier lives until the wait on
its token, and an accumulator or descriptor an asynchronous command reads lives
until that command's wait.

Barriers whose tokens another role constructs (`barrier_token`,
`barrier_arrive`, `barrier_init`) are not renumbered: their epochs are shared
state, and the fixed layout is kept.
"""
from .diagnostics import ResourceError
from .ir import TOKEN

_SHARED_BARRIER_OPS = ('barrier_token', 'barrier_token_dyn', 'barrier_arrive', 'barrier_init')


def _pool(kernel, kind):
    return {'barrier': kernel.barriers, 'accumulator': kernel.accumulators,
            'descriptor': kernel.descriptors, 'window': kernel.windows}[kind]


def _loops(ops):
    labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
    return [(labels[op.attrs['target']], i) for i, op in enumerate(ops)
            if op.kind in ('branch_nz', 'branch') and labels.get(op.attrs.get('target'), len(ops)) <= i]


def _uses(kernel, kind):
    """{id(resource): [(op index, is_definition)]} for one resource class."""
    ops = kernel.role(0).region.ops
    pool = _pool(kernel, kind)
    members = {id(v): v for v in pool}
    uses = {k: [] for k in members}
    token_owner = {}           # id(token) -> barrier
    waits = {}                 # id(token) -> [wait indices]
    for i, op in enumerate(ops):
        if op.kind == 'barrier_wait':
            waits.setdefault(id(op.operands[0]), []).append(i)
    for i, op in enumerate(ops):
        if kind == 'barrier':
            if op.kind == 'barrier_arm':
                bar = op.operands[0]
                token_owner[id(op.results[0])] = id(bar)
                uses[id(bar)].append((i, True))
                continue
            if op.kind in _SHARED_BARRIER_OPS:
                for v in op.operands:
                    if id(v) in members:
                        return None
            for v in op.operands:
                if getattr(v, 'kind', None) == TOKEN and id(v) in token_owner:
                    uses[token_owner[id(v)]].append((i, False))
            continue
        token = next((v for v in op.operands if getattr(v, 'kind', None) == TOKEN), None)
        for v in list(op.operands) + list(op.results):
            if id(v) not in members:
                continue
            definition = (kind == 'accumulator' and op.kind == 'acc_zero') or \
                         (kind == 'descriptor' and op.kind == 'desc_snapshot') or \
                         (kind == 'window' and op.kind == 'window_def')
            uses[id(v)].append((i, definition))
            if token is not None:
                # Read or written asynchronously: live until the wait.
                for j in waits.get(id(token), []):
                    if j > i:
                        uses[id(v)].append((j, False))
                        break
                else:
                    uses[id(v)].append((len(ops), False))
    return uses


def _intervals(kernel, uses):
    ops = kernel.role(0).region.ops
    loops = _loops(ops)
    spans = {}
    for key, points in uses.items():
        if not points:
            continue
        points.sort()
        lo, hi = points[0][0], points[-1][0]
        changed = True
        while changed:
            changed = False
            for start, end in loops:
                inside = [p for p in points if start <= p[0] <= end]
                if not inside:
                    continue
                if lo <= start and hi >= end:
                    continue
                outside = len(inside) != len(points) or lo < start or hi > end
                if outside or not inside[0][1]:
                    nlo, nhi = min(lo, start), max(hi, end)
                    if (nlo, nhi) != (lo, hi):
                        lo, hi, changed = nlo, nhi, True
        spans[key] = (lo, hi)
    return spans


def color(kernel, kind, limit):
    """Give every `kind` value a slot so that values with overlapping live
    ranges never share one. Returns the number of slots used, or None when
    this class cannot be renumbered (shared barrier epochs)."""
    if len(kernel.roles) != 1:
        return None
    uses = _uses(kernel, kind)
    if uses is None:
        return None
    pool = _pool(kernel, kind)
    spans = _intervals(kernel, uses)
    attr = 'wid' if kind == 'window' else 'reg'
    slots = []                       # slot -> end of the interval holding it
    for v in sorted(pool, key=lambda v: spans.get(id(v), (-1, -1))):
        if id(v) not in spans:
            setattr(v, attr, 0)          # never used: any slot names it
            continue
        lo, hi = spans[id(v)]
        for slot, end in enumerate(slots):
            if end < lo:
                slots[slot] = hi
                setattr(v, attr, slot)
                break
        else:
            setattr(v, attr, len(slots))
            slots.append(hi)
    if len(slots) > limit:
        raise ResourceError(f"{len(slots)} {kind}s live at once, target has {limit}",
                            remedy='a smaller tile')
    return len(slots)
