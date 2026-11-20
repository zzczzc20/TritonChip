"""Software pipelining of a transfer/MMA loop in one role (R42 gemm).

The importer lowers Inductor's GEMM template one operation at a time, so a
K step is

    arm; t.ld A; wait;  arm; t.ld B; wait;  arm; m.mma; wait

with a barrier per command: the transfer engine idles while the matrix
engine runs and the matrix engine idles while the transfers run, and a tile
with n fragments holds 2n+... of the core's barriers (eight on the R41
profiles, four a context when two programs share a core).

This pass rewrites the innermost such loop into a rotated two-stage
pipeline on two sets of transfer windows:

        test -> E
        LOAD[0]                     ; transfers of step 0, one barrier
    P:  MMA[0]                      ; matrix commands of step i, not awaited
        increment; test -> D0
        LOAD[1]                     ; transfers of step i+1 into the OTHER windows,
                                    ;   while the matrix engine runs step i
        wait MMA[0]
        MMA[1]
        increment; test -> D1
        LOAD[0]
        wait MMA[1]
        branch P
    D0: wait MMA[0]; branch E
    D1: wait MMA[1]
    E:

What makes it correct:

  * accumulation order. The matrix engine executes one command at a time in
    issue order (an m.mma issued while it is busy waits at issue), and the
    commands are issued in the original order into the original
    accumulators: every output element sees the same sequence of sec 9C.1
    accumulations. Numerics are bit-identical.
  * source windows. LOAD[s] overwrites the windows MMA[s] of two steps ago
    read. Those commands are complete: the stage in between waited for them
    ("wait MMA[s]") before it issued its own. The windows the running
    commands read belong to the other stage. No source-free event is needed.
  * barriers. A barrier is re-armed only after the wait on its previous
    generation (tnca_barrier: an arm replaces the counts). All transfers of
    a step count on one barrier and all matrix commands of a step on
    another, so the loop holds two barriers whatever the tile.
  * the accumulator is not read in the loop, and every exit path waits for
    the last matrix commands before the code after the loop runs.

Refused (the loop is left exactly as it was) when the body is not
`<load part> <matrix part>` in that order, when the matrix part reads a
register value the load part computes, when the load part touches an
accumulator or issues a tensor store, or when a window a matrix command
reads is written in the loop by anything but a transfer.
"""
from . import partition
from .ir import Op, Value

_RESOURCE = partition._RESOURCE


class NotPipelined(Exception):
    """Why the loop keeps its serial form. Not an error."""


# Operations that may sit between two transfers whose barriers are merged:
# address arithmetic and window definitions only -- nothing that reads what a
# transfer wrote, writes what one reads, or changes a descriptor.
_BETWEEN_TRANSFERS = frozenset({'const', 'sbin', 'sbini', 'scmp', 'comment', 'window_def'})
_MATRIX_PART = frozenset({'barrier_arm', 'mma_async', 'barrier_wait', 'comment'})
_LOAD_FORBIDDEN = frozenset({'dma_store', 'acc_read', 'acc_write', 'acc_zero', 'mma_async',
                             'barrier_arrive', 'barrier_init', 'barrier_token',
                             'barrier_token_dyn', 'trap', 'assert_eq', 'assert_lane',
                             'label', 'branch', 'branch_nz', 'exit'})


def _uses(ops, value):
    return [op for op in ops if any(v is value for v in partition._operands(op))]


def _merge_transfer_barriers(part):
    """`arm; t.ld; wait` runs of the load part onto one barrier each.

    A run is a sequence of such triples with only address arithmetic and
    window definitions between them. The first arm takes the run's operation
    count (ISA sec 9A.3: one count per complete command), every transfer
    counts on its token, and the one wait stands where the last was -- so
    nothing after the run reads a window before its transfer completed, and
    the transfers are issued back to back for an engine that overlaps them.
    Returns the new list."""
    triples = {}                                   # index of arm -> (load index, wait index)
    for i, op in enumerate(part):
        if op.kind != 'barrier_arm' or op.attrs.get('arrivals') or op.attrs.get('ops') != 1:
            continue
        users = _uses(part, op.results[0])
        if (len(users) == 2 and users[0].kind == 'dma_load' and users[1].kind == 'barrier_wait'
                and users[0].attrs.get('mask') is None):
            li, wi = part.index(users[0]), part.index(users[1])
            if i < li < wi and all(p.kind in _BETWEEN_TRANSFERS for p in part[i + 1:li]) \
                    and all(p.kind in _BETWEEN_TRANSFERS for p in part[li + 1:wi]):
                triples[i] = (li, wi)
    out, drop, retoken = list(part), set(), {}
    arms = sorted(triples)
    j = 0
    while j < len(arms):
        run = [arms[j]]
        while j + 1 < len(arms):
            nxt = arms[j + 1]
            last_wait = triples[run[-1]][1]
            if nxt > last_wait and all(p.kind in _BETWEEN_TRANSFERS for p in part[last_wait + 1:nxt]):
                run.append(nxt)
                j += 1
            else:
                break
        j += 1
        if len(run) < 2:
            continue
        head = part[run[0]]
        head.attrs['ops'] = len(run)
        for a in run[1:]:
            drop.add(a)
            retoken[id(part[a].results[0])] = head.results[0]
        for a in run[:-1]:
            drop.add(triples[a][1])
    if not drop:
        return out
    merged = []
    for i, op in enumerate(part):
        if i in drop:
            continue
        op.operands = [retoken.get(id(v), v) if isinstance(v, Value) else v for v in op.operands]
        merged.append(op)
    return merged


def _matrix_part(part):
    """The matrix part with every command on one barrier and no wait.

    Returns (ops, token): the caller places the wait."""
    arms = [op for op in part if op.kind == 'barrier_arm']
    mmas = [op for op in part if op.kind == 'mma_async']
    waits = [op for op in part if op.kind == 'barrier_wait']
    if not mmas or len(arms) != len(mmas) or len(waits) != len(mmas):
        raise NotPipelined('the matrix commands are not one armed, awaited command each')
    if len(mmas) > 255:
        raise NotPipelined('more matrix commands a step than a barrier counts')
    tokens = {}
    for arm in arms:
        if arm.attrs.get('arrivals') or arm.attrs.get('ops') != 1:
            raise NotPipelined('a matrix barrier is armed for more than one command')
        users = _uses(part, arm.results[0])
        if len(users) != 2 or users[0].kind != 'mma_async' or users[1].kind != 'barrier_wait':
            raise NotPipelined('a matrix completion has more than one command or wait')
        if users[0].attrs.get('source_free'):
            raise NotPipelined('a matrix command signals source-free')
        tokens[id(arm.results[0])] = True
    if any(id(op.operands[3]) not in tokens for op in mmas):
        raise NotPipelined('a matrix command counts on a barrier armed outside the loop')
    head = arms[0]
    head.attrs['ops'] = len(mmas)
    token = head.results[0]
    out = []
    for op in part:
        if op.kind == 'barrier_wait' or (op.kind == 'barrier_arm' and op is not head):
            continue
        if op.kind == 'mma_async':
            op.operands = list(op.operands)
            op.operands[3] = token
        out.append(op)
    return out, token


def pipeline_loop(kernel):
    """Rewrite the kernel's transfer/MMA loop in place, or raise NotPipelined
    with the kernel untouched."""
    if len(kernel.roles) != 1:
        raise NotPipelined('one logical role is pipelined')
    region = kernel.roles[0].region
    ops = region.ops
    try:
        loop = partition.find_ring_loop(ops)
    except partition.NotSpecialised as error:
        raise NotPipelined(str(error)) from error
    test_ops = ops[loop.head + 1:loop.test + 1]
    body = ops[loop.test + 1:loop.latch]
    incr = ops[loop.latch:loop.back]
    back = ops[loop.back]
    exit_label = ops[loop.back + 1].attrs['name']
    if ops[loop.test].attrs.get('target') != exit_label:
        raise NotPipelined('the loop does not exit to the label after it')
    if any(op.kind in ('label', 'branch', 'branch_nz', 'exit') for op in incr):
        raise NotPipelined('the increment is not straight-line')

    # ---- split the body ----------------------------------------------------
    mma_at = [i for i, op in enumerate(body) if op.kind == 'mma_async']
    first = None
    for i, op in enumerate(body):
        if op.kind == 'barrier_arm' and any(u.kind == 'mma_async' for u in _uses(body, op.results[0])):
            first = i
            break
    if first is None or not mma_at:
        raise NotPipelined('no armed matrix command in the loop')
    while first > 0 and body[first - 1].kind == 'comment':
        first -= 1
    load_part, matrix = body[:first], body[first:]
    if any(op.kind not in _MATRIX_PART for op in matrix):
        raise NotPipelined('the loop does more than matrix commands after its first one')
    if any(op.kind in _LOAD_FORBIDDEN for op in load_part):
        raise NotPipelined('the load part reads an accumulator, stores a tile or branches')
    if not any(op.kind == 'dma_load' for op in load_part):
        raise NotPipelined('the loop issues no transfer before its matrix commands')
    computed = {id(r) for op in load_part for r in op.results if r.kind not in _RESOURCE}
    if any(id(v) in computed for op in matrix for v in partition._operands(op)):
        raise NotPipelined('a matrix command reads a register the load part computes')
    # The increment may read what the body computed (the importer writes
    # `k + 1` where it is first needed). Here the increment runs after the
    # matrix part and before the next load part, so those values are
    # computed again beside it: pure scalar arithmetic on values the loop
    # does not change before the increment's own assignments.
    defined = {id(r): op for op in load_part for r in op.results}
    need, order, stack = set(), [], [v for op in incr for v in partition._operands(op)]
    while stack:
        v = stack.pop()
        op = defined.get(id(v))
        if op is None or id(op) in need:
            continue
        if op.kind not in ('const', 'sbin', 'sbini', 'scmp'):
            raise NotPipelined('the increment reads a value the load part computes with an effect')
        need.add(id(op))
        order.append(op)
        stack.extend(partition._operands(op))
    incr_pre = [op for op in load_part if id(op) in need]
    carried = {id(r) for op in matrix for r in op.results}
    if any(id(v) in carried for op in incr + test_ops for v in partition._operands(op)):
        raise NotPipelined('the loop control reads a matrix completion')
    # Windows the matrix commands read: each is refilled by transfers of the
    # load part (and gets a second copy), or is not written in the loop.
    ring = {id(w) for w in loop.windows}
    read = {id(v): v for op in matrix if op.kind == 'mma_async' for v in op.operands[1:3]}
    written = {}
    for op in body:
        if op.kind in ('dma_load', 'window_put'):
            written.setdefault(id(op.operands[0]), set()).add(op.kind)
    for key in read:
        kinds = written.get(key, set())
        if kinds - {'dma_load'}:
            raise NotPipelined('a matrix source window is written by the vector unit in the loop')
        if kinds and key not in ring:
            raise NotPipelined('a matrix source window is not one of the loop\'s transfer windows')
    # A window the load part fills and reads back synchronously (an unpack)
    # is not live across a stage; one read by a matrix command is doubled.
    doubled = [w for w in loop.windows if id(w) in read]
    if not doubled:
        raise NotPipelined('no transfer window feeds a matrix command')
    # The accumulators must not be read by anything in the loop.
    if any(op.kind in ('acc_read', 'acc_write', 'acc_zero') for op in body + incr + test_ops):
        raise NotPipelined('the loop reads or writes an accumulator directly')

    saved = (list(ops), list(kernel.windows), list(kernel.barriers),
             [(op, list(op.operands), dict(op.attrs)) for op in body])
    try:
        load_part = _merge_transfer_barriers(load_part)
        matrix, _ = _matrix_part(matrix)

        names = {op.attrs['name'] for op in ops if op.kind == 'label'}

        def fresh(stem):
            name, n = stem, 0
            while name in names:
                n += 1
                name = f'{stem}_{n}'
            names.add(name)
            return name
        head_label, drain = fresh('gemm_pipe'), [fresh('gemm_pipe_drain0'), fresh('gemm_pipe_drain1')]

        stage_windows = [{}, {}]
        for w in doubled:
            c = Value(name=f'{w.name}_p1', kind=w.kind, dtype=w.dtype, length=w.length)
            kernel.windows.append(c)
            stage_windows[1][id(w)] = c

        def copy(part, stage, shared=None):
            vmap = dict(stage_windows[stage])
            if shared:
                vmap.update(shared)
            out = [partition._clone(op, vmap) for op in part]
            return out, vmap

        def increment():
            vmap = {}
            return [partition._clone(op, vmap) for op in incr_pre + incr]

        def test_to(label):
            vmap = {}
            out = [partition._clone(op, vmap) for op in test_ops]
            out[-1].attrs['target'] = label
            return out

        def token_of(part):
            return next(op.results[0] for op in part if op.kind == 'barrier_arm')

        new = list(ops[:loop.test + 1])
        new += copy(load_part, 0)[0]
        new.append(Op(kind='label', results=[], operands=[], attrs={'name': head_label}))
        m0, _ = copy(matrix, 0)
        new += m0
        new += increment()
        new += test_to(drain[0])
        new += copy(load_part, 1)[0]
        new.append(Op(kind='barrier_wait', results=[], operands=[token_of(m0)], attrs={}))
        m1, _ = copy(matrix, 1)
        new += m1
        new += increment()
        new += test_to(drain[1])
        new += copy(load_part, 0)[0]
        new.append(Op(kind='barrier_wait', results=[], operands=[token_of(m1)], attrs={}))
        one = Value(name='gemm_pipe_one', kind='scalar', dtype='i64')
        new += [Op(kind='const', results=[one], operands=[], attrs={'value': 1}),
                Op(kind='branch_nz', results=[], operands=[one], attrs={'target': head_label}),
                Op(kind='label', results=[], operands=[], attrs={'name': drain[0]}),
                Op(kind='barrier_wait', results=[], operands=[token_of(m0)], attrs={})]
        one2 = Value(name='gemm_pipe_one2', kind='scalar', dtype='i64')
        new += [Op(kind='const', results=[one2], operands=[], attrs={'value': 1}),
                Op(kind='branch_nz', results=[], operands=[one2], attrs={'target': exit_label}),
                Op(kind='label', results=[], operands=[], attrs={'name': drain[1]}),
                Op(kind='barrier_wait', results=[], operands=[token_of(m1)], attrs={})]
        new += ops[loop.back + 1:]
        # Barriers no command or arm names any more leave the kernel: they
        # count against the core's (and a context slice's) barriers.
        used = {id(v) for op in new for v in op.operands if isinstance(v, Value)}
        kernel.barriers[:] = [b for b in kernel.barriers if id(b) in used]
        region.ops[:] = new
    except BaseException:
        region.ops[:] = saved[0]
        kernel.windows[:] = saved[1]
        kernel.barriers[:] = saved[2]
        for op, operands, attrs in saved[3]:
            op.operands, op.attrs = operands, attrs
        raise
    del back
    return dict(stages=2, roles=1, windows=len(doubled) * 2, barriers=2,
                transfers=sum(op.kind == 'dma_load' for op in load_part),
                matrix_commands=sum(op.kind == 'mma_async' for op in matrix))
