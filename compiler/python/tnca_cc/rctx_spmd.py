"""R39-05 Mode B: every Role Context of the core runs programs of one grid row.

A launch starts each context in its role mask at the same PC with the same
pid (rtl/core/tnca_core.v), and each reads ROLE_ID. The kernel body becomes
a loop over its programs on the x axis,

    x = ROLE_ID
    do { body with program_id(0) = x;  x += R } while (x < num_programs(0))

with R the contexts the target has, so context r runs programs r, r + R,
r + 2R, ... and one launch runs a whole row of the grid. The runtime
launches each (y, z) once, starting min(R, gx) contexts (manifest
execution.program_to_role). The contexts interleave instruction by
instruction, each with its own registers, so one issues while another waits.

What the loop does not depend on is computed once before it: the scalar
argument loads, constants and the other axes' ids, from the kernel's entry
block. Vector values stay where they are -- hoisting one keeps a register
live across the whole loop, and the allocator has no spill.

Accumulators, barriers, descriptors and windows are per core and named by
immediates every context would use alike, and the scratch the kernel builds
its TensorMaps in is one buffer per launch. On a target with `rctx_slice`
(RSLICE) a launch gives each context its own slice of each file -- the
hardware ORs the context number into the top bits of every index -- so a
kernel whose counts fit a slice is eligible, and its scratch is offset by
ROLE_ID * scratch_bytes into a buffer the runtime makes that many times
larger. Without the capability only a kernel that uses none of them is.
"""
from .ir import Op, Value, SCALAR

# Operations on per-core resources two concurrent programs would share.
SHARED = frozenset({
    'acc_zero', 'acc_write', 'acc_read', 'mma_async',
    'window_def', 'window_reg', 'window_put', 'dma_load', 'dma_store',
    'barrier_init', 'barrier_arm', 'barrier_wait', 'barrier_token',
    'barrier_token_dyn', 'barrier_arrive', 'desc_snapshot',
})

_BODY, _LATCH = '__rctx_program', '__rctx_next'
_CONTROL = frozenset({'label', 'branch_nz', 'exit'})
_PURE_SCALAR = frozenset({'sbin', 'sbini', 'scmp'})
# Operations that write an operand in place (tnca_cc.schedule.MUTATES).
_MUTATES = {'vinsert': (0,), 'assign': (0,)}


def slices(target, contexts):
    """{kind: how many a context may use} when the files are cut into
    `contexts` slices (RSLICE), or None without it. Descriptors and windows
    share the slicing of the smaller file."""
    caps = target.capabilities if isinstance(target.capabilities, dict) \
        else dict(target.capabilities)
    if not caps.get('rctx_slice'):
        return None
    res = min(target.num_descriptors, target.num_windows)
    return {'accumulators': target.num_accumulators // contexts,
            'barriers': target.num_barriers // contexts,
            'descriptors': res // contexts, 'windows': res // contexts}


def ineligible(kernel, scratch_bytes=0, slice_sizes=None):
    """Why this kernel cannot run its programs on every context, or None.
    `slice_sizes` is slices(target, contexts)."""
    if len(kernel.roles) != 1:
        return 'the kernel has more than one role'
    used = {'accumulators': len(kernel.accumulators), 'barriers': len(kernel.barriers),
            'descriptors': len(kernel.descriptors), 'windows': len(kernel.windows)}
    if slice_sizes is None:
        if scratch_bytes:
            return 'the kernel uses scratch memory (one buffer per launch)'
        if any(used.values()):
            return 'the kernel holds per-core resources and the target cannot slice them'
    else:
        for kind, n in used.items():
            if n > slice_sizes[kind]:
                return f'the kernel uses {n} {kind}, more than a slice of {slice_sizes[kind]}'
    ops = kernel.roles[0].region.ops
    for op in ops:
        if slice_sizes is None and op.kind in SHARED:
            return f'the kernel uses {op.kind}'
        if op.kind in ('barrier_token', 'barrier_token_dyn'):
            # A token that names an epoch counted from the kernel's start
            # (the pipelined lowering's) is wrong for the second program on
            # a context: the barrier's epochs carry on from the first.
            return 'the kernel names barrier epochs counted from its start'
        if op.kind == 'label' and op.attrs.get('name') in (_BODY, _LATCH):
            return f'the kernel already has a label {op.attrs["name"]}'
    if not any(op.kind == 'program_id' and op.attrs.get('dim', 0) == 0 for op in ops):
        return 'the kernel never reads program_id(0)'
    return None


def _hoistable(op, invariant, arg_base, mutated):
    """Whether `op`, in the entry block, gives the same result for every
    program of a launch and may be computed once before the loop.

    Not a value something writes in place (`assign`, `vinsert`): a loop
    counter's `const 0` computed once would start the second program's
    loop where the first one's ended."""
    if any(id(v) not in invariant for v in op.operands) or op.attrs.get('mask') is not None:
        return False
    if any(r.kind != SCALAR or id(r) in mutated for r in op.results):
        return False
    k = op.kind
    if k == 'const' or k == 'num_programs':
        return True
    if k == 'special':
        return op.attrs.get('name') != 'role_id'
    if k == 'program_id':
        return op.attrs.get('dim', 0) != 0
    if k in _PURE_SCALAR:
        return True
    # An argument: read from the launch's argument block, which nothing
    # writes while the kernel runs.
    return k == 'sload' and id(op.operands[0]) in arg_base


def apply(kernel, contexts, scratch_slot=None, scratch_bytes=0, hoist=True):
    """Wrap the role's body in the loop over its programs. Returns the number
    of operations computed once before the loop.

    `scratch_slot` is the byte offset in the argument block of the scratch
    base (the slot after the declared arguments). Every load of it -- the
    importer loads it again after a loop, where an earlier load does not
    dominate -- is offset here by ROLE_ID * scratch_bytes, so each context
    builds its TensorMaps and repacks its tiles in a buffer of its own.

    `hoist=False` computes nothing once: every value the loop would have
    carried across programs is recomputed in each, which is what a kernel
    whose scalars do not fit otherwise needs (the allocator has no spill)."""
    region = kernel.roles[0].region
    ops = region.ops
    entry = next((i for i, op in enumerate(ops) if op.kind in _CONTROL), len(ops)) \
        if hoist else 0
    invariant, arg_base = set(), set()
    mutated = {id(op.operands[i]) for op in ops
               for i in _MUTATES.get(op.kind, ()) if i < len(op.operands)}
    # A TensorMap the entry block builds from invariant values -- Inductor's
    # are built from the arguments, and the tile's place is given at each
    # load -- is built once per context too: its stores into the context's
    # scratch, and the snapshot once every store to that map has moved. Only
    # snapshots read a map, so moving its stores reorders nothing observable.
    maps = {id(op.operands[0]) for op in ops if op.kind == 'desc_snapshot'}
    left = {}                      # map address -> its stores still in the body
    before, body = [], []
    for i, op in enumerate(ops):
        if i < entry and _hoistable(op, invariant, arg_base, mutated):
            before.append(op)
            invariant.update(id(r) for r in op.results)
            if op.kind == 'special' and op.attrs.get('name') == 'arg_base':
                arg_base.add(id(op.results[0]))
        elif (i < entry and op.kind == 'sstore' and id(op.operands[0]) in maps
              and all(id(v) in invariant for v in op.operands)):
            before.append(op)
        elif (i < entry and op.kind == 'desc_snapshot' and id(op.operands[0]) in invariant
              and not left.get(id(op.operands[0]))):
            before.append(op)
        else:
            if op.kind == 'sstore' and id(op.operands[0]) in maps:
                left[id(op.operands[0])] = left.get(id(op.operands[0]), 0) + 1
            body.append(op)

    # The loop variable is the first program_id(0)'s own value, so the body
    # reads it where it read the program id; any other read is a copy.
    pids = [op for op in body if op.kind == 'program_id' and op.attrs.get('dim', 0) == 0]
    x = pids[0].results[0]
    npid = Value(name='rctx_programs', kind=SCALAR, dtype='i32')
    one = Value(name='rctx_one', kind=SCALAR, dtype='i64')
    out = []
    for op in body:
        if op is pids[0]:
            continue
        if op.kind == 'program_id' and op.attrs.get('dim', 0) == 0:
            out.append(Op('sbini', op.results, [x], {'op': 'add', 'imm': 0,
                                                    'comment': 'this program'}))
        elif op.kind == 'exit':
            out.append(Op('branch_nz', [], [one], {'target': _LATCH}))
        else:
            out.append(op)
    # The exits at the end fall through to the latch (the importer ends a
    # function with two: tt.return's and its own).
    while True:
        last = next((i for i in range(len(out) - 1, -1, -1) if out[i].kind != 'comment'), None)
        if last is None or out[last].kind != 'branch_nz' or out[last].attrs.get('target') != _LATCH:
            break
        del out[last]
    head = [Op('special', [x], [], {'name': 'role_id'}),
            Op('num_programs', [npid], [], {'dim': 0})]
    if any(op.kind == 'branch_nz' and op.attrs.get('target') == _LATCH for op in out):
        head.append(Op('const', [one], [], {'value': 1}))
    head.append(Op('label', [], [], {'name': _BODY}))
    nxt = Value(name='rctx_next', kind=SCALAR, dtype=x.dtype)
    more = Value(name='rctx_more', kind=SCALAR, dtype='i32')
    # The compare reads the incremented value, not the copy, so it does not
    # wait for the `assign`.
    tail = [Op('label', [], [], {'name': _LATCH}),
            Op('sbini', [nxt], [x], {'op': 'add', 'imm': contexts,
                                     'comment': f'the next program of this context: + {contexts}'}),
            Op('scmp', [more], [nxt, npid], {'cc': 'lt'}),
            Op('assign', [], [x, nxt]),
            Op('branch_nz', [], [more], {'target': _BODY}),
            Op('exit', [], [])]
    if scratch_slot is not None and scratch_bytes:
        bases = {id(op.results[0]) for op in before + out
                 if op.kind == 'special' and op.attrs.get('name') == 'arg_base'}
        before = _own_scratch(before, bases, scratch_slot, scratch_bytes, 'pre')
        out = _own_scratch(out, bases, scratch_slot, scratch_bytes, 'body')
    region.ops = before + head + out + tail
    return len(before)


def _own_scratch(ops, bases, slot, nbytes, tag):
    """After each load of the scratch base in `ops` (an sload at `slot` from
    one of the `bases`, the arg_base values), the base plus ROLE_ID * nbytes
    under the loaded value's name, so every use is the context's."""
    out = []
    for i, op in enumerate(ops):
        out.append(op)
        if not (op.kind == 'sload' and op.attrs.get('offset') == slot
                and op.operands and id(op.operands[0]) in bases):
            continue
        base = op.results[0]
        raw = Value(name=f'{base.name}_launch', kind=SCALAR, dtype=base.dtype)
        op.results = [raw]
        rid = Value(name=f'rctx_{tag}{i}_role', kind=SCALAR, dtype='i64')
        size = Value(name=f'rctx_{tag}{i}_bytes', kind=SCALAR, dtype='i64')
        off = Value(name=f'rctx_{tag}{i}_off', kind=SCALAR, dtype='i64')
        out += [Op('special', [rid], [], {'name': 'role_id'}),
                Op('const', [size], [], {'value': nbytes}),
                Op('sbin', [off], [rid, size], {'op': 'mul'}),
                Op('sbin', [base], [raw, off],
                   {'op': 'add', 'comment': "this context's own scratch"})]
    return out
