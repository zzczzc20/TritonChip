"""Instruction scheduling over TNCA IR (R39-01).

The core issues one instruction per cycle, in order, and an instruction that
reads a register still being written waits at issue (the scoreboard). The
emitter used to keep program order, so every dependence the importer wrote
next to its producer stalled for the producer's whole latency -- 32.6 % of
all cycles on the standard path were that stall (doc/r39_tasks_and_acceptance
.md sec 0.2). This pass reorders each basic block of a role so independent
work fills those cycles.

What it may move, and what it may not:

  * Pure operations (`PURE`: arithmetic, compares, splats, constants,
    reductions) move freely subject to their data dependences.
  * Memory reads (`load`, `gather`, `sload`) may pass each other and pure
    operations, never a write, a fence, a barrier, a transfer or anything
    else with an effect.
  * Everything else keeps its order relative to every other effectful
    operation and to the reads.
  * `vinsert` and `assign` write an operand in place: they are ordered after
    every earlier reader of it, and later readers after them.
  * A value that aliases a window register (`window_reg`, `view`) can change
    under a transfer, so a reader of one stays ahead of the next effectful
    operation it preceded.
  * Labels, branches and exits end a block; nothing crosses them.

The machine model is measured, not assumed: spec/r39_sched_model.json, from
tools/r39/latency_probe.py on the profile's register file. A block is a list
schedule against it -- each step issues, among the operations whose inputs
are scheduled, the one that can issue earliest, ties to the longest latency
path to the end of the block, then to program order. A pressure guard keeps
the number of live vector values near what program order had, because the
allocator has no spill.
"""
import json

from ._datapath import find
from .ir import PRED, SCALAR, VECTOR

PURE = frozenset({
    'const', 'program_id', 'num_programs', 'special', 'sbin', 'sbini', 'scmp',
    'arange', 'splat', 'vmovi', 'vbin', 'vun', 'vfma', 'vfms', 'vcvt',
    'vextract', 'vshuffle', 'vcmp', 'pbin', 'pun', 'vselect', 'reduce',
})
MEMREAD = frozenset({'load', 'gather', 'sload'})
# Operations that write the register of an operand rather than a result.
MUTATES = {'vinsert': (0,), 'assign': (0,)}
# Operations whose result names a register something else can write.
ALIASES = frozenset({'window_reg', 'view'})
BLOCK_END = frozenset({'branch_nz', 'exit'})

_SFU = frozenset({'exp2', 'rcp', 'rsqrt', 'log2'})
_FP = frozenset({'f32', 'f16', 'bf16'})


def _values_in(op):
    from .optimization import _values_in as v
    return v(op)


class Model:
    """Latency and issue cost of an IR operation on one register file."""

    def __init__(self, name, data):
        self.name = name
        self.lat = data['latency']
        self.issue = data['issue']
        self.lsu = data['lsu']

    @classmethod
    def for_target(cls, target):
        with open(find('spec', 'r39_sched_model.json')) as fh:
            spec = json.load(fh)
        name = spec['by_target'].get(getattr(target, 'name', None), spec['default'])
        return cls(name, spec['models'][name])

    def klass(self, op):
        k = op.kind
        if k in ('sbin', 'sbini'):
            return 'smul' if op.attrs.get('op') == 'mul' else 'salu'
        if k in ('const', 'program_id', 'num_programs', 'special', 'scmp', 'assign'):
            return 'salu'
        if k == 'sload':
            return 'sload'
        if k in ('load', 'gather'):
            return 'vload'
        if k == 'vun' and op.attrs.get('op') in _SFU:
            return 'sfu'
        if k in ('vbin', 'vun', 'vfma', 'vfms', 'vcvt'):
            res = op.results[0] if op.results else None
            fp = (res is not None and res.dtype in _FP) or k in ('vfma', 'vfms', 'vcvt')
            return 'vfp' if fp else 'valu'
        if k == 'vcmp':
            return 'vcmp'
        if k in ('pbin', 'pun'):
            return 'pred'
        if k == 'reduce':
            return 'reduce_b' if op.attrs.get('broadcast') else 'reduce'
        if k == 'vextract':
            return 'vextract'
        if k in ('arange', 'splat', 'vmovi', 'vselect', 'vshuffle'):
            return 'valu'
        if k in ('vinsert', 'acc_read'):
            return k
        if k in ALIASES:
            return 'alias'
        return 'other'

    def latency(self, op):
        """Cycles from this operation's issue to the issue of one that reads
        its result. A load's result is there when the unit lets it go."""
        if op.kind in MEMREAD:
            return self.lsu_busy(op)
        return int(self.lat.get(self.klass(op), self.lat['other']))

    _BYTES = {'b': 1, 'h': 2, 'w': 4, 'd': 8}

    def lsu_busy(self, op):
        """Cycles a memory access holds the load/store unit: a fixed cost
        and one per beat of the data port."""
        size = self._BYTES.get(op.attrs.get('size'), 4)
        lanes = 1 if op.kind in ('sload', 'sstore', 'atomic', 'atomic_cas') else 16
        beats = -(-size * lanes // int(self.lsu['bytes_per_beat']))
        return int(self.lsu['base']) + max(1, beats)

    def occupancy(self, op):
        """Issue cycles: a vector operation that reads the register file
        holds issue for the read on the banked file."""
        if op.kind in ('comment', 'label') or self.klass(op) == 'alias':
            return 0
        if any(getattr(v, 'kind', None) == VECTOR for v in _values_in(op)):
            return int(self.issue['vector_reads_vrf'])
        if any(getattr(r, 'kind', None) == VECTOR for r in op.results):
            return int(self.issue['vector_no_vrf_read'])
        return int(self.issue['scalar'])

    def uses_lsu(self, op):
        return op.kind in ('load', 'gather', 'sload', 'store', 'scatter', 'sstore',
                           'atomic', 'atomic_cas')


def _blocks(ops):
    """Split a flat op list into basic blocks: a label starts one, a branch
    or an exit ends one."""
    out, cur = [], []
    for op in ops:
        if op.kind == 'label' and cur:
            out.append(cur)
            cur = []
        cur.append(op)
        if op.kind in BLOCK_END:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


class _Node:
    __slots__ = ('ops', 'op', 'idx', 'preds', 'succs', 'height', 'kind')

    def __init__(self, ops, idx):
        self.ops = ops            # the op and the comments that precede it
        self.op = ops[-1]
        self.idx = idx
        self.preds = {}           # node -> latency
        self.succs = []
        self.height = 0


def _units(block):
    """Comments travel with the operation after them; a label stays first and
    a branch or exit last."""
    units, pending = [], []
    for op in block:
        if op.kind == 'comment':
            pending.append(op)
            continue
        units.append(pending + [op])
        pending = []
    tail = pending
    return units, tail


def _edge(a, b, lat):
    if a is b:
        return
    if b.preds.get(a, -1) < lat:
        if a not in b.preds:
            a.succs.append(b)
        b.preds[a] = lat


def _graph(units, model, aliased):
    nodes = [_Node(u, i) for i, u in enumerate(units)]
    writer = {}          # value id -> node that last defined or wrote it
    readers = {}         # value id -> nodes that read it since that write
    last_effect = None
    reads_since = []     # memory reads since last_effect
    alias_readers = []   # nodes reading an aliased value, waiting for the next effect
    for n in nodes:
        op = n.op
        if op.kind == 'label':
            n.kind = 'fixed'
        elif op.kind in PURE:
            n.kind = 'pure'
        elif op.kind in MEMREAD:
            n.kind = 'read'
        else:
            n.kind = 'effect'
        reads = [v for v in _values_in(op) if hasattr(v, 'kind')]
        mut = [op.operands[i] for i in MUTATES.get(op.kind, ())
               if i < len(op.operands)]
        for v in reads:
            w = writer.get(id(v))
            if w is not None:
                _edge(w, n, model.latency(w.op))
            readers.setdefault(id(v), []).append(n)
        for v in mut:
            for r in readers.get(id(v), []):
                _edge(r, n, 0)
            w = writer.get(id(v))
            if w is not None:
                _edge(w, n, 0)
        for v in list(op.results) + mut:
            writer[id(v)] = n
            readers[id(v)] = []
        if n.kind in ('effect', 'fixed'):
            if last_effect is not None:
                _edge(last_effect, n, 0)
            for r in reads_since:
                _edge(r, n, 0)
            for r in alias_readers:
                _edge(r, n, 0)
            reads_since, alias_readers = [], []
            last_effect = n
        elif n.kind == 'read':
            if last_effect is not None:
                _edge(last_effect, n, 0)
            reads_since.append(n)
        if n.kind != 'effect' and any(id(v) in aliased for v in reads):
            alias_readers.append(n)
    # The first node of a block that starts with a label stays first; the
    # last one, a branch or an exit, stays last.
    if nodes and nodes[0].op.kind == 'label':
        for m in nodes[1:]:
            _edge(nodes[0], m, 0)
    if nodes and nodes[-1].op.kind in BLOCK_END:
        for m in nodes[:-1]:
            _edge(m, nodes[-1], 0)
    # Heights: the longest latency path from a node to the end of the block.
    for n in reversed(nodes):
        h = model.occupancy(n.op)
        for s in n.succs:
            h = max(h, s.preds[n] + s.height)
        n.height = h
    return nodes


_FILES = (VECTOR, SCALAR, PRED)
# How far past program order's peak a block may go, per register file, on
# the first attempt. aot._emit retries with none before giving up on
# scheduling a kernel.
SLACK = {VECTOR: 4, SCALAR: 4, PRED: 1}
# How far ahead of the oldest unscheduled operation, in program order, an
# operation may be taken. Enough to cover the longest latency the model has
# with independent work; small enough that a block of thousands of
# independent operations is not hoisted wholesale (which no register file
# survives).
WINDOW = 48


class _Pressure:
    """Live values per register file, as the allocator will see them.

    The allocator (codegen.CodeGen._compute_last_uses) frees a value after
    its last use in the final order, and keeps a value defined before a loop
    and used inside it until the loop's closing branch. What a block's order
    can change is when the values that DIE in it die: those defined here and
    those live into it, whose last use (so extended) is here. They are
    counted -- the live-ins from the block's start -- and nothing else is,
    because everything else is live across the whole block whatever the
    order.
    """

    def __init__(self, nodes, dies_here):
        self.remaining = {}
        kinds = {}
        for n in nodes:
            for v in _values_in(n.op):
                if getattr(v, 'kind', None) in _FILES:
                    self.remaining[id(v)] = self.remaining.get(id(v), 0) + 1
                    kinds[id(v)] = v.kind
        defined = {id(r) for n in nodes for r in n.op.results}
        self.dies = dies_here
        # Live into the block and dying in it: live from the start.
        self.live_in = {v: k for v, k in kinds.items() if v not in defined and v in dies_here}
        self.reset()

    def _keeps(self, vid):
        return vid not in self.dies or self.rem.get(vid, 0) > 0

    def reset(self):
        self.rem = dict(self.remaining)
        self.defined = dict(self.live_in)
        self.live = {k: 0 for k in _FILES}
        for k in self.live_in.values():
            self.live[k] += 1

    def _uses(self, op):
        uses = {}
        for x in _values_in(op):
            if getattr(x, 'kind', None) in _FILES:
                uses[id(x)] = uses.get(id(x), 0) + 1
        return uses

    def growth(self, op):
        """Live values per file after `op` minus before it."""
        g = {k: 0 for k in _FILES}
        for r in op.results:
            k = getattr(r, 'kind', None)
            if k in _FILES and self._keeps(id(r)):
                g[k] += 1
        for v, k in self._uses(op).items():
            if v in self.defined and self.rem[v] == k and v in self.dies:
                g[self.defined[v]] -= 1
        return g

    def step(self, op):
        for v, k in self._uses(op).items():
            if v in self.rem:
                self.rem[v] -= k
            if v in self.defined and self.rem.get(v, 0) == 0 and v in self.dies:
                self.live[self.defined.pop(v)] -= 1
        for r in op.results:
            k = getattr(r, 'kind', None)
            if k in _FILES and self._keeps(id(r)):
                self.live[k] += 1
                self.defined[id(r)] = k

    def peaks(self, nodes):
        self.reset()
        best = {k: 0 for k in _FILES}
        for n in nodes:
            self.step(n.op)
            for k in _FILES:
                best[k] = max(best[k], self.live[k])
        self.reset()
        return best


def schedule_block(block, model, dies_here, aliased, slack=None, window=None):
    """One basic block, list-scheduled. Returns the new op list.

    Each step issues, among the operations whose predecessors are all
    issued, the one that can issue soonest, ties to the longest path to the
    end of the block, then to program order. While a register file is at its
    cap -- program order's peak plus `slack` -- an operation that would add a
    live value to that file goes last.
    """
    slack = SLACK if slack is None else slack
    window = WINDOW if window is None else window
    units, tail = _units(block)
    if len(units) < 3:
        return block
    nodes = _graph(units, model, aliased)
    pressure = _Pressure(nodes, dies_here)
    peaks = pressure.peaks(nodes)
    cap = {k: peaks[k] + slack[k] for k in _FILES}

    unscheduled_preds = {n: len(n.preds) for n in nodes}
    ready = [n for n in nodes if not n.preds]
    issue_at = {}
    t, lsu_free = 0, 0
    order = []
    while ready:
        def start(n):
            s = t
            for p, lat in n.preds.items():
                s = max(s, issue_at[p] + lat)
            if model.uses_lsu(n.op):
                s = max(s, lsu_free)
            return s

        full = [k for k in _FILES if pressure.live[k] >= cap[k]]

        def over(n):
            if not full:
                return False
            g = pressure.growth(n.op)
            return any(g[k] > 0 for k in full)

        # Every predecessor of the oldest unscheduled node is older, so it is
        # ready and the candidates are never empty.
        oldest = min(n.idx for n in ready)
        cands = [n for n in ready if n.idx < oldest + window]
        best = min(cands, key=lambda n: (over(n), start(n), -n.height, n.idx))
        s = start(best)
        issue_at[best] = s
        t = s + model.occupancy(best.op)
        if model.uses_lsu(best.op):
            lsu_free = s + model.lsu_busy(best.op)
        order.append(best)
        ready.remove(best)
        pressure.step(best.op)
        for s_ in best.succs:
            unscheduled_preds[s_] -= 1
            if unscheduled_preds[s_] == 0:
                ready.append(s_)
    if len(order) != len(nodes):
        raise AssertionError('schedule: a dependence cycle in a basic block')
    # The guard steers; it cannot always hold the cap (when every ready
    # operation adds a live value, one of them goes). A block whose peak
    # went over keeps program order.
    after = pressure.peaks(order)
    if any(after[k] > cap[k] for k in _FILES):
        return block
    out = []
    for n in order:
        out.extend(n.ops)
    return out + tail


def estimate(block, model):
    """Issue cycles of a block in the given order under the model -- what the
    scheduler minimises, for reports and tests."""
    units, _ = _units(block)
    nodes = _graph(units, model, set())
    issue_at, t, lsu_free = {}, 0, 0
    for n in nodes:
        s = t
        for p, lat in n.preds.items():
            s = max(s, issue_at[p] + lat)
        if model.uses_lsu(n.op):
            s = max(s, lsu_free)
        issue_at[n] = s
        t = s + model.occupancy(n.op)
        if model.uses_lsu(n.op):
            lsu_free = s + model.lsu_busy(n.op)
    return t


def last_uses(ops):
    """Value id -> position of its last use, as the allocator computes it
    (codegen.CodeGen._compute_last_uses): operands and the mask, with a
    value defined before a loop and used in it kept to the loop's branch.
    A value that is never used dies where it is defined."""
    defined_at, last = {}, {}
    for i, op in enumerate(ops):
        for v in op.results:
            defined_at.setdefault(id(v), i)
            last.setdefault(id(v), i)
    for i, op in enumerate(ops):
        for v in list(op.operands) + [op.attrs.get('mask')]:
            if v is not None and getattr(v, 'kind', None):
                last[id(v)] = max(last.get(id(v), -1), i)
    labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
    loops = [(labels[op.attrs['target']], i) for i, op in enumerate(ops)
             if op.kind == 'branch_nz' and op.attrs.get('target') in labels]
    for start, end in sorted(loops, key=lambda p: p[1] - p[0]):
        for key, at in list(last.items()):
            if defined_at.get(key, -1) < start and start <= at <= end:
                last[key] = end
    return last


def drop_unread_argument_loads(ops):
    """Remove loads from the argument block whose value nothing reads.

    The importer loads every declared argument at entry, used or not (an
    elementwise kernel never reads `xnumel` once its mask is folded). Each
    is a memory instruction, and the load/store unit takes one at a time --
    about nine cycles on the board -- in every program of the grid. A load
    from the argument block cannot fault (the runtime wrote that block for
    this launch), so one nobody reads has no effect to keep.
    """
    arg_base = {id(r) for op in ops if op.kind == 'special' and op.attrs.get('name') == 'arg_base'
                for r in op.results}
    read = {id(v) for op in ops for v in _values_in(op)}
    keep = [op for op in ops
            if not (op.kind == 'sload' and op.operands and id(op.operands[0]) in arg_base
                    and not any(id(r) in read for r in op.results))]
    return keep, len(ops) - len(keep)


def schedule_kernel(kernel, target, *, model=None, slack=None, window=None):
    """Reorder every basic block of every role. Returns a report:
    {'model', 'blocks', 'moved', 'estimate_before', 'estimate_after',
    'dropped_argument_loads'}."""
    model = model or Model.for_target(target)
    report = dict(model=model.name, blocks=0, moved=0, estimate_before=0, estimate_after=0,
                  dropped_argument_loads=0)
    for role in kernel.roles:
        role.region.ops, dropped = drop_unread_argument_loads(role.region.ops)
        report['dropped_argument_loads'] += dropped
        ops = role.region.ops
        aliased = set()
        for op in ops:
            if op.kind in ALIASES:
                aliased.update(id(r) for r in op.results)
        blocks = _blocks(ops)
        last = last_uses(ops)
        new = []
        pos = 0
        for bi, blk in enumerate(blocks):
            lo, hi = pos, pos + len(blk) - 1
            pos += len(blk)
            dies_here = {vid for vid, at in last.items() if lo <= at <= hi}
            before = estimate(blk, model)
            out = schedule_block(blk, model, dies_here, aliased, slack, window)
            after = estimate(out, model)
            if after > before:
                out = blk                      # never worse than program order
                after = before
            report['blocks'] += 1
            report['moved'] += sum(1 for a, b in zip(blk, out) if a is not b)
            report['estimate_before'] += before
            report['estimate_after'] += after
            new.extend(out)
        role.region.ops = new
    return report
