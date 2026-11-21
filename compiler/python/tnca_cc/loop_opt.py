"""Loop optimisation on TNCA IR (R42): the reduction loops of the standard path.

The passes of optimization.py leave a role alone as soon as it has a label, a
branch or an `assign` -- so a kernel with a loop (every Inductor reduction:
`for roffset in range(0, rnumel, RBLOCK)`) reached the emitter exactly as the
importer wrote it. On a core that issues one instruction per cycle, in order,
that is what a looped kernel's time is made of: the importer rebuilds every
constant, every broadcast and every loop-invariant expression (a reciprocal
and its Newton step, the row's mean, a reciprocal square root) in every
iteration, masks every access with `index < rnumel` even when the trip count
makes the mask all-true, and carries each accumulator through two register
moves.

Seven steps, on the flat operation list of a role (labels, branches and
register assignments; a role with a nested region is left alone), each of
them a rewrite of the same program -- no operation's result changes, no
memory access is added, removed or reordered:

  0. `drop_unused_maps`: a TensorMap whose descriptor nothing names is not
     built (its scratch words, its snapshot, and so the fence before it).
  1. `fold_true_masks`: with a counted loop `for (iv = a; iv < n; iv += s)`
     whose bounds are constants, the predicate `splat(iv) + arange < N` (or
     `>= N`) is all-true in every iteration when the last iteration's last
     lane satisfies it. Loads, stores and reductions under such a predicate
     lose the mask, `select(p, a, b)` is `a`, and `p & q` is `q`.
  2. `forward_copies`: `assign(T, v); assign(K, T)` with T a temporary nothing
     else names and v a value nothing writes in place is `assign(K, v)`.
  3. dead pure definitions are dropped (optimization.drop_dead_pure with
     flat control: the lane indices the folded masks no longer need).
  4. `hoist_invariants`: a pure operation in a loop whose inputs are all
     defined before the loop and never written in it moves in front of the
     loop (innermost loops first, so it can move again). The ISA has no
     floating-point exception state and no pure operation can fault, so
     executing one the loop would have skipped is unobservable.
  5. `induction_pointers`: an address that is linear in the loop counter is
     a pointer advanced by a constant, and accesses a constant apart share it.
  6. `rotate_loops`: `L: if (iv >= n) goto E; body; iv = next; goto L; E:`
     becomes a guarded bottom-tested loop -- one branch per iteration and no
     constant for the unconditional jump.

A loop is touched only if every operation in it is one this file knows:
pure register operations, vector and scalar loads and stores, gathers and
scatters, labels, branches and assignments. A loop with a transfer, a matrix
command, a barrier, a window, a descriptor, an atomic or a fence keeps its
code (those loops are shaped by partition.py and the Role Context passes).
Nothing here looks at a kernel's name, its shapes or its sizes.

Register pressure: a hoisted value holds a register across its loop and the
allocator has no spill. The caller (tnca.aot) compiles a kernel again without
this pass when the optimised one does not fit.
"""
from .ir import Op, Value, SCALAR, VECTOR, PRED
from .optimization import _values_in, _rewrite, _opaque

# Operations with a result and nothing else: no memory, no ownership, no
# fault, no exception state (the ISA has none).
PURE = frozenset({
    'const', 'sbin', 'sbini', 'scmp', 'arange', 'splat', 'vmovi', 'vbin',
    'vun', 'vfma', 'vfms', 'vcvt', 'vextract', 'vshuffle', 'vcmp', 'pbin',
    'pun', 'vselect', 'reduce',
})
MEMORY = frozenset({'load', 'store', 'gather', 'scatter', 'sload', 'sstore'})
# `vinsert` writes its vector operand in place (tracked like `assign`): never moved.
CONTROL = frozenset({'label', 'branch_nz', 'assign', 'comment', 'vinsert'})
SIMPLE = PURE | MEMORY | CONTROL
# Operations that write an operand in place (tnca_cc.schedule.MUTATES).
MUTATES = {'vinsert': (0,), 'assign': (0,), 'window_put': (1, 2)}


class _Info:
    """Where every value is defined and written in a flat op list."""

    def __init__(self, ops):
        self.ops = ops
        self.defs = {}       # id(value) -> [positions of ops with it as a result]
        self.writes = {}     # id(value) -> [positions of in-place writes]
        self.def_op = {}
        for i, op in enumerate(ops):
            for r in op.results:
                self.defs.setdefault(id(r), []).append(i)
                self.def_op[id(r)] = op
            for k in MUTATES.get(op.kind, ()):
                if k < len(op.operands) and isinstance(op.operands[k], Value):
                    self.writes.setdefault(id(op.operands[k]), []).append(i)

    def ssa(self, v):
        """One definition and nothing writes it in place."""
        return len(self.defs.get(id(v), ())) == 1 and id(v) not in self.writes

    def const(self, v, depth=0):
        """The integer a never-written scalar holds, or None."""
        if not isinstance(v, Value) or not self.ssa(v) or depth > 8:
            return None
        op = self.def_op[id(v)]
        if op.kind == 'const':
            return int(op.attrs['value'])
        if op.kind == 'sbini' and op.attrs.get('op') in ('add', 'sub', 'shl', 'mul'):
            a = self.const(op.operands[0], depth + 1)
            if a is None:
                return None
            imm = int(op.attrs['imm'])
            return {'add': a + imm, 'sub': a - imm, 'shl': a << imm, 'mul': a * imm}[op.attrs['op']]
        return None


def _simple_role(role):
    return not any(_opaque(o) and o.kind not in ('label', 'branch_nz', 'assign')
                   for o in role.region.ops)


def find_loops(ops):
    """[(header, latch)] positions of the loops this file may touch, innermost
    first: a label and a later branch back to it, every operation between them
    SIMPLE, entered only through the header and properly nested."""
    labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
    if len(labels) != sum(1 for op in ops if op.kind == 'label'):
        return []                                   # a label defined twice: not ours
    back = {}
    for i, op in enumerate(ops):
        if op.kind == 'branch_nz':
            at = labels.get(op.attrs.get('target'))
            if at is not None and at < i:
                back[at] = max(back.get(at, -1), i)
    loops = sorted(back.items(), key=lambda p: p[1] - p[0])
    ok = []
    for h, e in loops:
        if any(ops[i].kind not in SIMPLE for i in range(h, e + 1)):
            continue
        # entered only by falling into the header: no branch from outside to
        # the header (it would skip what is hoisted in front) or into the body
        entered = False
        for i, op in enumerate(ops):
            if op.kind == 'branch_nz' and not (h <= i <= e):
                at = labels.get(op.attrs.get('target'))
                if at is not None and h <= at <= e:
                    entered = True
        if entered:
            continue
        # properly nested with every other loop
        if any((h2 < h <= e2 < e) or (h < h2 <= e < e2) for h2, e2 in loops):
            continue
        ok.append((h, e))
    return ok


def _counted(ops, info, h, e):
    """The counted form of loop (h, e), or None:

        L:  c = scmp ge (iv, lim);  branch_nz c -> EXIT
            ...                         (iv read, never written)
            assign(iv, nxt)             nxt = iv + step
            branch_nz <const 1> -> L
        EXIT:

    Returns dict(iv, lim, nxt, cmp, exit, init, limit, step) -- the last three
    integers or None when not constant."""
    if e - h < 4 or e + 1 >= len(ops):
        return None
    latch = ops[e]
    # the importer writes the latch's `const 1` between the assign and the branch
    at = e - 2 if (ops[e - 1].kind == 'const' and ops[e - 1].results[0] is latch.operands[0]) else e - 1
    cmp_, br, asg, exit_ = ops[h + 1], ops[h + 2], ops[at], ops[e + 1]
    if cmp_.kind != 'scmp' or cmp_.attrs.get('cc') != 'ge' or br.kind != 'branch_nz':
        return None
    if exit_.kind != 'label' or br.attrs.get('target') != exit_.attrs.get('name'):
        return None
    if br.operands[0] is not cmp_.results[0] or asg.kind != 'assign':
        return None
    iv, lim = cmp_.operands
    if asg.operands[0] is not iv or info.writes.get(id(iv)) != [at]:
        return None
    if len(info.defs.get(id(iv), ())) != 1 or info.defs[id(iv)][0] >= h:
        return None
    if info.const(latch.operands[0]) != 1:
        return None
    nxt = asg.operands[1]
    if not isinstance(nxt, Value) or not info.ssa(nxt) or not (h < info.defs[id(nxt)][0] < at):
        return None
    nop = info.def_op[id(nxt)]
    step = None
    if nop.kind == 'sbin' and nop.attrs.get('op') == 'add' and nop.operands[0] is iv:
        step = info.const(nop.operands[1])
        step_value = nop.operands[1]
    elif nop.kind == 'sbini' and nop.attrs.get('op') == 'add' and nop.operands[0] is iv:
        step = int(nop.attrs['imm'])
        step_value = None
    else:
        return None
    # the header's label is the target of the latch only; the compare has one user
    name = ops[h].attrs['name']
    if sum(1 for o in ops if o.kind == 'branch_nz' and o.attrs.get('target') == name) != 1:
        return None
    if sum(1 for o in ops for v in _values_in(o) if v is cmp_.results[0]) != 1:
        return None
    iop = info.def_op[id(iv)]
    init = None
    if iop.kind == 'const':
        init = int(iop.attrs['value'])
    elif iop.kind == 'sbini' and iop.attrs.get('op') == 'add':
        a = info.const(iop.operands[0])
        init = None if a is None else a + int(iop.attrs['imm'])
    return dict(iv=iv, lim=lim, nxt=nxt, cmp=cmp_, exit=exit_.attrs['name'], init=init, assign=at,
                limit=info.const(lim) if info.ssa(lim) else None, step=step,
                step_value=step_value)


def _lane_offset(info, v):
    """c when `v` is the lane vector arange() + c (c >= 0; a wide block's
    chunk is `arange + vmovi(c)`), else None."""
    if not isinstance(v, Value) or not info.ssa(v):
        return None
    d = info.def_op[id(v)]
    if d.kind == 'arange' and not d.operands:
        return 0
    if d.kind == 'vbin' and d.attrs.get('op') == 'add' and len(d.operands) == 2:
        for x, y in (d.operands, d.operands[::-1]):
            dx = info.def_op.get(id(x)) if isinstance(x, Value) and info.ssa(x) else None
            dy = info.def_op.get(id(y)) if isinstance(y, Value) and info.ssa(y) else None
            if dx is not None and dy is not None and dx.kind == 'arange' and not dx.operands \
                    and dy.kind == 'vmovi' and 0 <= int(dy.attrs['imm']) < (1 << 30):
                return int(dy.attrs['imm'])
    return None


def _iv_offset(info, x, iv, h, i):
    """c when the scalar `x`, read at position i of the loop at h, is iv + c
    (c >= 0): iv itself, or iv plus a constant computed in this iteration."""
    if x is iv:
        return 0
    if not isinstance(x, Value) or not info.ssa(x) or not (h < info.defs[id(x)][0] < i):
        return None
    d = info.def_op[id(x)]
    c = None
    if d.kind == 'sbini' and d.attrs.get('op') == 'add' and d.operands[0] is iv:
        c = int(d.attrs['imm'])
    elif d.kind == 'sbin' and d.attrs.get('op') == 'add' and len(d.operands) == 2:
        a, b = d.operands
        c = info.const(b) if a is iv else info.const(a) if b is iv else None
    return c if c is not None and 0 <= c < (1 << 30) else None


def fold_true_masks(role, vlen):
    """Step 1. Returns the number of masks, selects and predicate ANDs folded."""
    ops = role.region.ops
    info = _Info(ops)
    true = set()          # id(pred) proven all-true wherever it is read
    keep = {}             # id -> Value, to keep the objects alive
    for h, e in find_loops(ops):
        c = _counted(ops, info, h, e)
        if c is None or c['init'] is None or c['limit'] is None or not c['step'] or c['step'] <= 0:
            continue
        init, limit, step = c['init'], c['limit'], c['step']
        if init < 0 or init >= limit:
            continue
        last = init + ((limit - 1 - init) // step) * step      # iv of the last iteration
        iv = c['iv']
        for i in range(h, c['assign']):
            op = ops[i]
            if op.kind != 'vcmp' or op.attrs.get('cc') not in ('lt', 'ge') or len(op.operands) != 2:
                continue
            lanes, bound = op.operands
            if not (isinstance(lanes, Value) and info.ssa(lanes) and h < info.defs[id(lanes)][0] < i):
                continue
            lop = info.def_op[id(lanes)]
            if lop.kind != 'vbin' or lop.attrs.get('op') != 'add' or len(lop.operands) != 2:
                continue
            off = None
            for x, y in (lop.operands, lop.operands[::-1]):
                d = info.def_op.get(id(x)) if isinstance(x, Value) and info.ssa(x) else None
                if d is not None and d.kind == 'splat' and h < info.defs[id(x)][0] < i:
                    first = _iv_offset(info, d.operands[0], iv, h, i)
                    lane = _lane_offset(info, y)
                    if first is not None and lane is not None:
                        off = first + lane
                        break
            if off is None:
                continue
            bop = info.def_op.get(id(bound)) if isinstance(bound, Value) and info.ssa(bound) else None
            if bop is None:
                continue
            if bop.kind == 'vmovi':
                n = int(bop.attrs['imm'])
            elif bop.kind == 'splat':
                n = info.const(bop.operands[0])
            else:
                continue
            if n is None or not (0 <= n < (1 << 31)) or last + off + vlen - 1 >= (1 << 31):
                continue
            if (op.attrs['cc'] == 'lt' and last + off + vlen - 1 < n) \
                    or (op.attrs['cc'] == 'ge' and init + off >= n):
                true.add(id(op.results[0]))
                keep[id(op.results[0])] = op.results[0]
    if not true:
        return 0
    folded = 0
    eq = {}               # id(value) -> the Value that replaces it

    def resolve(v):
        while id(v) in eq:
            v = eq[id(v)]
        return v
    changed = True
    while changed:
        changed = False
        for op in ops:
            if op.kind == 'pbin' and op.attrs.get('op') == 'and' and id(op.results[0]) not in eq \
                    and info.ssa(op.results[0]):
                a, b = (resolve(v) for v in op.operands)
                if id(a) in true or id(b) in true:
                    eq[id(op.results[0])] = b if id(a) in true else a
                    folded += 1
                    changed = True
    eq = {k: resolve(v) for k, v in list(eq.items())}
    out = []
    for op in ops:
        _rewrite(op, eq)
        mask = op.attrs.get('mask')
        if isinstance(mask, Value) and id(mask) in true and op.kind in (
                'load', 'store', 'gather', 'scatter', 'reduce'):
            op.attrs['mask'] = None
            folded += 1
        if op.kind == 'vselect' and id(op.operands[0]) in true and info.ssa(op.results[0]):
            eq[id(op.results[0])] = eq.get(id(op.operands[1]), op.operands[1])
            folded += 1
            continue
        if op.kind == 'pbin' and id(op.results[0]) in eq:
            continue
        out.append(op)
    for op in out:
        _rewrite(op, eq)
    role.region.ops = out
    return folded


def forward_copies(role):
    """Step 2. Returns the number of temporaries removed."""
    ops = role.region.ops
    info = _Info(ops)
    uses = {}
    for i, op in enumerate(ops):
        for v in _values_in(op):
            uses.setdefault(id(v), []).append(i)
    drop = set()
    n = 0
    for i, op in enumerate(ops):
        if op.kind != 'assign' or i in drop:
            continue
        tmp, src = op.operands
        if not (isinstance(src, Value) and isinstance(tmp, Value)):
            continue
        # T: one plain initial definition, this one write, and one reader -- a later assign
        if info.writes.get(id(tmp)) != [i] or len(info.defs.get(id(tmp), ())) != 1:
            continue
        if info.def_op[id(tmp)].kind not in ('vmovi', 'const', 'sbini', 'splat'):
            continue
        readers = [j for j in uses.get(id(tmp), ()) if j != i]
        if len(readers) != 1:
            continue
        j = readers[0]
        user = ops[j]
        if j <= i or user.kind != 'assign' or user.operands[1] is not tmp or user.operands[0] is tmp:
            continue
        # one basic block: the first assign always runs before the second
        if any(ops[k].kind in ('label', 'branch_nz', 'exit') for k in range(i + 1, j)):
            continue
        # v is never written in place: it holds the same value at the second assign
        if not info.ssa(src):
            continue
        user.operands = [user.operands[0], src]
        drop.add(i)
        n += 1
    # A scalar copy is `T = v + 0` (the importer's _copy_value): `assign(K, T)`
    # with T read nowhere else and v never written in place is `assign(K, v)`.
    for i, op in enumerate(ops):
        if op.kind != 'assign' or i in drop:
            continue
        tmp = op.operands[1]
        if not (isinstance(tmp, Value) and tmp.kind == SCALAR and info.ssa(tmp)):
            continue
        d = info.def_op[id(tmp)]
        if d.kind != 'sbini' or d.attrs.get('op') != 'add' or int(d.attrs.get('imm', 1)) != 0:
            continue
        src = d.operands[0]
        if uses.get(id(tmp)) != [i] or not (isinstance(src, Value) and info.ssa(src)) \
                or src.kind != op.operands[0].kind:
            continue
        op.operands = [op.operands[0], src]
        n += 1
    if drop:
        role.region.ops = [op for i, op in enumerate(ops) if i not in drop]
    return n


def hoist_invariants(role):
    """Step 4. Returns the number of operations moved in front of their loop."""
    moved = 0
    seen = set()
    while True:
        ops = role.region.ops
        info = _Info(ops)
        todo = [(h, e) for h, e in find_loops(ops) if ops[h].attrs['name'] not in seen]
        if not todo:
            return moved
        h, e = todo[0]                                # innermost first
        seen.add(ops[h].attrs['name'])
        hoisted, hoisted_ids = [], set()
        for i in range(h + 1, e):
            op = ops[i]
            if op.kind not in PURE or not op.results:
                continue
            if any(not info.ssa(r) for r in op.results):
                continue
            ok = True
            for v in _values_in(op):
                if id(v) in hoisted_ids:
                    continue
                d = info.defs.get(id(v), ())
                w = info.writes.get(id(v), ())
                # defined (and last written) before the loop, never written in it
                if not d or max(d) >= h or any(x >= h for x in w):
                    ok = False
                    break
            if not ok:
                continue
            hoisted.append(i)
            hoisted_ids.update(id(r) for r in op.results)
        if hoisted:
            pick = set(hoisted)
            role.region.ops = (ops[:h] + [ops[i] for i in hoisted]
                               + [op for i, op in enumerate(ops[h:], h) if i not in pick])
            moved += len(hoisted)


def merge_equal(role):
    """Equal pure definitions in one basic block become one: the same
    operation on the same operands with the same attributes, results and
    operands never written in place. (Hoisting collects what each chunk of a
    wide block computed for itself -- the same broadcast, the same row
    offset -- in front of the loop.) Returns the number merged."""
    ops = role.region.ops
    info = _Info(ops)
    eq, seen, n = {}, {}, 0
    for op in ops:
        if op.kind in ('label', 'branch_nz', 'exit'):
            seen = {}
            continue
        _rewrite(op, eq)
        if op.kind not in PURE or len(op.results) != 1 or not info.ssa(op.results[0]):
            continue
        named = list(_values_in(op))
        if any(not info.ssa(v) for v in named):
            continue
        r = op.results[0]
        try:
            key = (op.kind, r.kind, r.dtype, tuple(id(v) if isinstance(v, Value) else ('c', v) for v in op.operands),
                   tuple(sorted((k, id(a) if isinstance(a, Value) else repr(a))
                                for k, a in op.attrs.items() if k != 'comment')))
        except TypeError:
            continue
        first = seen.get(key)
        if first is None:
            seen[key] = r
        else:
            eq[id(r)] = first
            n += 1
    if eq:
        role.region.ops = [op for op in ops
                           if not (len(op.results) == 1 and id(op.results[0]) in eq)]
        for op in role.region.ops:
            _rewrite(op, eq)
    return n


_IMM16 = 1 << 15


def _affine(info, ops, h, at, iv):
    """{id(scalar): form} for the scalars of loop body (h, at): a form is
    {leaf: coefficient} with leaves 'iv', 1 (the constant term) and the ids
    of scalars the loop does not change. Only ring operations (add, sub,
    shl, mul by a constant), so a form holds modulo the register width."""
    forms, leaves = {}, {}

    def outside(x):
        d = info.defs.get(id(x), ())
        return bool(d) and max(d) < h and not any(w >= h for w in info.writes.get(id(x), ()))

    def form(x):
        if x is iv:
            return {'iv': 1}
        if not isinstance(x, Value) or x.kind != SCALAR:
            return None
        if id(x) in forms:
            return forms[id(x)]
        if outside(x):
            c = info.const(x)
            if c is not None:
                return {1: c}
            leaves[id(x)] = x
            return {id(x): 1}
        return None

    def scale(f, k):
        return {l: c * k for l, c in f.items()}

    def add(f, g, sign=1):
        out = dict(f)
        for l, c in g.items():
            out[l] = out.get(l, 0) + sign * c
        return out
    for i in range(h + 1, at):
        op = ops[i]
        if len(op.results) != 1 or op.results[0].kind != SCALAR or not info.ssa(op.results[0]):
            continue
        f = None
        if op.kind == 'const':
            f = {1: int(op.attrs['value'])}
        elif op.kind == 'sbini' and op.attrs.get('op') in ('add', 'sub', 'shl', 'mul'):
            g = form(op.operands[0])
            imm = int(op.attrs['imm'])
            if g is not None:
                f = {'add': lambda: add(g, {1: imm}), 'sub': lambda: add(g, {1: -imm}),
                     'shl': lambda: scale(g, 1 << imm) if 0 <= imm < 32 else None,
                     'mul': lambda: scale(g, imm)}[op.attrs['op']]()
        elif op.kind == 'sbin' and op.attrs.get('op') in ('add', 'sub', 'mul') and len(op.operands) == 2:
            g, k = form(op.operands[0]), form(op.operands[1])
            if g is not None and k is not None:
                if op.attrs['op'] == 'mul':
                    f = scale(g, k[1]) if set(k) == {1} else scale(k, g[1]) if set(g) == {1} else None
                else:
                    f = add(g, k, 1 if op.attrs['op'] == 'add' else -1)
        if f is not None:
            forms[id(op.results[0])] = {l: c for l, c in f.items() if c or l == 1}
    return forms


def induction_pointers(role):
    """Step 5. An address `base + ((iv + x) << k)` recomputed every iteration
    becomes a pointer advanced by a constant: the address of the first
    iteration is computed in front of the loop (the same operations, reading
    iv before the loop changes it), and accesses whose addresses differ by a
    constant share one pointer through the immediate offset of `v.ld`/`v.st`.
    Returns the number of accesses rewritten."""
    n = 0
    seen = set()
    while True:
        ops = role.region.ops
        info = _Info(ops)
        todo = [(h, e) for h, e in find_loops(ops) if ops[h].attrs['name'] not in seen]
        if not todo:
            return n
        h, e = todo[0]
        seen.add(ops[h].attrs['name'])
        c = _counted(ops, info, h, e)
        if c is None or not c['step'] or c['step'] <= 0:
            continue
        at, iv = c['assign'], c['iv']
        forms = _affine(info, ops, h, at, iv)
        groups = {}
        for i in range(h + 1, at):
            op = ops[i]
            if op.kind not in ('load', 'store') or not op.operands:
                continue
            f = forms.get(id(op.operands[0]))
            if not f or not f.get('iv'):
                continue
            key = tuple(sorted((str(l), co) for l, co in f.items() if l != 1))
            groups.setdefault(key, []).append((i, f.get(1, 0) + int(op.attrs.get('offset', 0) or 0), f['iv']))
        if not groups:
            continue
        before, bump, replace = [], [], {}
        for key, members in groups.items():
            stride = members[0][2] * c['step']
            low = min(m[1] for m in members)
            ref = next(m for m in members if m[1] == low)
            if not (0 < abs(stride) < _IMM16) or int(ops[ref[0]].attrs.get('offset', 0) or 0):
                continue
            fit = [m for m in members if 0 <= m[1] - low < _IMM16]
            # the first iteration's address: the reference access's own
            # address computation, copied in front of the loop
            copies = {}

            def clone(x):
                if id(x) in copies:
                    return copies[id(x)]
                d = info.defs.get(id(x), ())
                if x is iv or not d or max(d) < h:
                    return x
                src = info.def_op[id(x)]
                new = Value(x.name + '_p', x.kind, x.dtype)
                copy = Op(src.kind, [new], [clone(v) if isinstance(v, Value) else v for v in src.operands],
                          dict(src.attrs))
                before.append(copy)
                copies[id(x)] = new
                return new
            pointer = clone(ops[ref[0]].operands[0])
            nxt = Value(pointer.name + '_n', SCALAR, pointer.dtype)
            bump += [Op('sbini', [nxt], [pointer], {'op': 'add', 'imm': stride,
                                                    'comment': 'the next iteration\'s address'}),
                     Op('assign', [], [pointer, nxt])]
            for i, const, _a in fit:
                replace[i] = (pointer, const - low)
        if not replace:
            continue
        for i, (pointer, offset) in replace.items():
            ops[i].operands = [pointer] + list(ops[i].operands[1:])
            ops[i].attrs['offset'] = offset
            n += 1
        role.region.ops = ops[:h] + before + ops[h:at] + bump + ops[at:]


def rotate_loops(role):
    """Step 6. Returns the number of loops rotated."""
    n = 0
    seen = set()
    while True:
        ops = role.region.ops
        info = _Info(ops)
        todo = [(h, e) for h, e in find_loops(ops) if ops[h].attrs['name'] not in seen]
        if not todo:
            return n
        h, e = todo[0]
        seen.add(ops[h].attrs['name'])
        c = _counted(ops, info, h, e)
        if c is None:
            continue
        iv, lim = c['iv'], c['lim']
        # the bound must hold the same value at the bottom test
        if not isinstance(lim, Value) or any(h <= x <= e for x in info.writes.get(id(lim), ())) \
                or any(x >= h for x in info.defs.get(id(lim), ())):
            continue
        cmp_, br, latch = ops[h + 1], ops[h + 2], ops[e]
        more = Value(cmp_.results[0].name + '_more', SCALAR, cmp_.results[0].dtype)
        bottom = Op('scmp', [more], [iv, lim], {'cc': 'lt'})
        back = Op('branch_nz', [], [more], dict(latch.attrs))
        known = (c['init'] is not None and c['limit'] is not None and c['init'] < c['limit'])
        guard = [] if known else [cmp_, br]           # at least one trip: no guard
        role.region.ops = (ops[:h] + guard + [ops[h]] + ops[h + 3:e] + [bottom, back] + ops[e + 1:])
        n += 1


def drop_unused_maps(kernel):
    """Step 0. A TensorMap the kernel builds and never transfers through.

    Inductor's kernels make a tensor descriptor for every buffer they read
    through one; the importer lowers a descriptor load of a register row to a
    vector load from the map's base, and then nothing names the descriptor.
    Its snapshot (`td.load`), the words stored for it in the kernel's private
    scratch, and the fence the snapshot needs (added later, only where a
    snapshot remains) are then work with no reader. A map is dropped only if
    nothing names its descriptor, and its words only if the address they are
    stored through has no other use. Returns the number of maps dropped."""
    dropped = 0
    for role in kernel.roles:
        if not _simple_role(role):
            continue
        ops = role.region.ops
        named = set()
        for op in ops:
            for v in _values_in(op):
                named.add(id(v))
        dead = [op for op in ops if op.kind == 'desc_snapshot' and len(op.results) == 1
                and id(op.results[0]) not in named and len(op.operands) == 1]
        if not dead:
            continue
        gone = {id(op) for op in dead}
        for snap in dead:
            base = snap.operands[0]
            users = [op for op in ops if any(v is base for v in _values_in(op))]
            if all(op is snap or (op.kind == 'sstore' and op.operands[0] is base
                                  and all(v is not base for v in op.operands[1:]))
                   for op in users):
                gone.update(id(op) for op in users)
        role.region.ops = [op for op in ops if id(op) not in gone]
        for snap in dead:
            kernel.descriptors[:] = [d for d in kernel.descriptors if d is not snap.results[0]]
        dropped += len(dead)
    return dropped


def optimize(kernel, vlen):
    """All five steps on every role with flat control. Returns the counts
    (all zero when nothing changed)."""
    from .optimization import drop_dead_pure
    out = dict(maps=0, masks=0, copies=0, dead=0, hoisted=0, merged=0, pointers=0, rotated=0)
    out['maps'] = drop_unused_maps(kernel)
    touched = bool(out['maps'])
    for role in kernel.roles:
        if not _simple_role(role) or not find_loops(role.region.ops):
            continue
        out['masks'] += fold_true_masks(role, vlen)
        out['copies'] += forward_copies(role)
        touched = True
    if not touched:
        return out
    out['dead'] += drop_dead_pure(kernel, flat_control=True)['removed']
    for role in kernel.roles:
        if not _simple_role(role) or not find_loops(role.region.ops):
            continue
        out['hoisted'] += hoist_invariants(role)
        out['merged'] += merge_equal(role)
        out['pointers'] += induction_pointers(role)
    if out['pointers']:
        out['dead'] += drop_dead_pure(kernel, flat_control=True)['removed']
    for role in kernel.roles:
        if _simple_role(role) and find_loops(role.region.ops):
            out['rotated'] += rotate_loops(role)
    return out
