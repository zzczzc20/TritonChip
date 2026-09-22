"""
Automatic Context Specialization: milestone C4.

compiler_overview.md sec 7 sets out six steps, and sec 15 C4 says to reuse the
upstream partitioning infrastructure at this point -- graph construction,
partition propagation, loop cloning, producer/consumer analysis -- while
replacing what is target-specific. This module is the TNCA half: the
classification, the policy, the communication choice and the mapping. The
shape follows sec 7.2's suggested adapter, so the target-specific decisions
sit in one place instead of being scattered through the algorithm.

Two constraints shape everything here.

sec 7.6, last step:

    不能让物理 RCTX 数决定最开始的数据流图表示；否则两角色和四角色版本会形成
    两套编译器。

So partitioning is logical and the RCTX count is applied only at the end, by
merging. The four-role plan and the two-role plan are the same graph mapped
differently, not two compilers.

sec 7.3, on communication:

    Scalar RF 是每 RCTX 私有的。

So a barrier token produced by one role does not appear in another's scalar
file. For a static pipeline each role computes the epoch from its own
iteration number; only a genuinely dynamic scalar needs a mailbox.
"""
from dataclasses import dataclass, field
from enum import Enum

from .diagnostics import Diagnostic
from .ir import Op, Value


class OpClass(Enum):
    """sec 7.2's classification. Deliberately not NVIDIA's.

    The plan is explicit that policies like "TMEM and MMA must be partitioned
    thus", "CLC must own a worker" and warpgroup register budgets are not to
    be inherited -- they describe a different machine.
    """
    ADDRESS = 'address/control'
    TRANSFER = 'load/transfer'
    MATRIX = 'matrix'
    VECTOR = 'vector/reduction/sfu'
    EPILOGUE = 'store/epilogue'
    SYNC = 'synchronization'


class EdgeKind(Enum):
    """sec 7.1: SSA def-use alone is not the graph.

    A partitioner that sees only def-use will happily put an accumulator's
    updates in two roles, or move a write across an asynchronous command that
    is still reading the register.
    """
    DEF_USE = 'def-use'
    LOOP_CARRIED = 'loop-carried'
    MEMORY = 'memory'
    CONTROL = 'control'
    ACCUMULATOR = 'accumulator-order'
    ASYNC_RESOURCE = 'async-resource'


@dataclass
class Node:
    index: int
    op: object
    cls: OpClass

    def __repr__(self):
        return f"n{self.index}:{self.op.kind}:{self.cls.value}"


@dataclass
class Edge:
    src: int
    dst: int
    kind: EdgeKind
    value: object = None


@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def successors(self, i):
        return [e for e in self.edges if e.src == i]

    def predecessors(self, i):
        return [e for e in self.edges if e.dst == i]

    def of_class(self, cls):
        return [n for n in self.nodes if n.cls is cls]


# ---------------------------------------------------------------------------
# sec 7.2: classification, behind one adapter
# ---------------------------------------------------------------------------
_CLASS_BY_OP = {
    'const': OpClass.ADDRESS, 'program_id': OpClass.ADDRESS,
    'special': OpClass.ADDRESS, 'sbin': OpClass.ADDRESS,
    'sbini': OpClass.ADDRESS, 'scmp': OpClass.ADDRESS,
    'assign': OpClass.ADDRESS, 'label': OpClass.ADDRESS,
    'branch_nz': OpClass.ADDRESS, 'comment': OpClass.ADDRESS,
    'exit': OpClass.ADDRESS, 'trap': OpClass.ADDRESS,
    'window_def': OpClass.ADDRESS, 'desc_snapshot': OpClass.ADDRESS,
    'window_reg': OpClass.ADDRESS,

    'dma_load': OpClass.TRANSFER, 'dma_store': OpClass.TRANSFER,

    'mma_async': OpClass.MATRIX, 'acc_zero': OpClass.MATRIX,
    'acc_read': OpClass.MATRIX,

    'arange': OpClass.VECTOR, 'splat': OpClass.VECTOR,
    'vmovi': OpClass.VECTOR, 'vbin': OpClass.VECTOR,
    'vcmp': OpClass.VECTOR, 'vselect': OpClass.VECTOR,
    'vun': OpClass.VECTOR, 'vfma': OpClass.VECTOR,
    'vfms': OpClass.VECTOR, 'vcvt': OpClass.VECTOR,
    'load': OpClass.VECTOR, 'gather': OpClass.VECTOR,
    'reduce': OpClass.VECTOR,

    'store': OpClass.EPILOGUE, 'scatter': OpClass.EPILOGUE,
    'sstore': OpClass.EPILOGUE, 'sload': OpClass.ADDRESS,

    'barrier_init': OpClass.SYNC, 'barrier_arm': OpClass.SYNC,
    'barrier_wait': OpClass.SYNC, 'barrier_arrive': OpClass.SYNC,
    'barrier_token': OpClass.SYNC, 'barrier_token_dyn': OpClass.SYNC,

    'assert_eq': OpClass.EPILOGUE, 'assert_lane': OpClass.EPILOGUE,
}


class SpecializationTargetInfo:
    """sec 7.2's adapter: the target-specific decisions, in one place.

    The plan proposes this shape precisely so the upstream algorithm is not
    edited in a dozen places to accommodate a different machine.
    """

    def __init__(self, target):
        self.target = target

    def classify(self, op):
        cls = _CLASS_BY_OP.get(op.kind)
        if cls is None:
            raise Diagnostic(
                f"no partition class for IR operation {op.kind!r}",
                because='sec 7.2 classifies every operation; an unclassified '
                        'one would be placed arbitrarily',
                source='compiler_overview.md sec 7.2')
        return cls

    def can_rematerialize(self, op):
        """sec 7.3 step 2: cheap pure address arithmetic is recomputed in the
        consuming role rather than communicated."""
        # Pure, cheap, and with no memory or resource effect. A predicate is
        # here deliberately: it is one v.cmp, and the predicate file is
        # per-RCTX private (sec 7.3), so recomputing it costs an instruction
        # where communicating it would cost a mailbox and a synchronisation.
        return op.kind in ('const', 'program_id', 'special', 'sbin', 'sbini',
                           'arange', 'barrier_token', 'barrier_token_dyn',
                           'vcmp', 'splat',
                           'vmovi')

    def estimate_cross_partition_use(self, value):
        """A rough cost, in registers held across the boundary."""
        if value.kind == 'vector':
            return self.target.vlen * 4
        if value.kind == 'window':
            return value.length * self.target.vlen * 4
        return 8

    def can_merge(self, a, b):
        """Which logical partitions may share one Role Context.

        Transfer and matrix are kept apart while there is room: they are the
        two that overlap in time, and merging them serialises the pipeline
        that specialisation exists to create.
        """
        return not ({a, b} == {OpClass.TRANSFER, OpClass.MATRIX})

    def budget(self):
        t = self.target
        return {'rctx': t.num_rctx, 'barriers': t.num_barriers,
                'accumulators': t.num_accumulators, 'windows': t.num_windows,
                'vregs': t.vregs_per_rctx}


# ---------------------------------------------------------------------------
# sec 7.1: the graph
# ---------------------------------------------------------------------------
def build_graph(region, info):
    """Build the dataflow graph, with the edges sec 7.1 lists beyond def-use."""
    g = Graph()
    producer = {}
    for i, op in enumerate(region.ops):
        node = Node(index=i, op=op, cls=info.classify(op))
        g.nodes.append(node)
        for v in op.results:
            producer[id(v)] = i

    label_at = {op.attrs['name']: i
                for i, op in enumerate(region.ops) if op.kind == 'label'}

    last_memory_write = None
    acc_last_update = {}
    live_async = []          # (node index, window value)

    for i, op in enumerate(region.ops):
        # --- def-use, including values consumed as attributes ---
        operands = list(op.operands)
        mask = op.attrs.get('mask')
        if mask is not None:
            operands.append(mask)
        for v in operands:
            src = producer.get(id(v))
            if src is not None:
                g.edges.append(Edge(src, i, EdgeKind.DEF_USE, v))

        # --- control: a branch depends on everything before its target ---
        if op.kind == 'branch_nz':
            target = op.attrs.get('target')
            if target in label_at and label_at[target] < i:
                # A loop back edge. Every node in the body carries a
                # loop-carried dependence on the branch, because the next
                # iteration cannot start before it resolves.
                for j in range(label_at[target], i):
                    g.edges.append(Edge(i, j, EdgeKind.LOOP_CARRIED))

        # --- memory: stores order against each other and against loads ---
        if op.kind in ('store', 'scatter', 'sstore'):
            if last_memory_write is not None:
                g.edges.append(Edge(last_memory_write, i, EdgeKind.MEMORY))
            last_memory_write = i
        elif op.kind in ('load', 'gather', 'sload') \
                and last_memory_write is not None:
            g.edges.append(Edge(last_memory_write, i, EdgeKind.MEMORY))

        # --- accumulator update order (ISA sec 9C.1 fixes it) ---
        if op.kind in ('mma_async', 'acc_zero', 'acc_read'):
            acc = op.operands[0]
            prev = acc_last_update.get(id(acc))
            if prev is not None:
                g.edges.append(Edge(prev, i, EdgeKind.ACCUMULATOR, acc))
            if op.kind != 'acc_read':
                acc_last_update[id(acc)] = i

        # --- asynchronous source/destination resource conflicts (sec 13.4) ---
        if op.kind == 'dma_load':
            window = op.operands[0]
            for (src_i, w) in live_async:
                if w is window:
                    g.edges.append(Edge(src_i, i, EdgeKind.ASYNC_RESOURCE, w))
            live_async.append((i, window))
        elif op.kind == 'mma_async':
            for w in op.operands[1:3]:
                for (src_i, lw) in live_async:
                    if lw is w:
                        g.edges.append(
                            Edge(src_i, i, EdgeKind.ASYNC_RESOURCE, w))
        elif op.kind == 'barrier_wait':
            live_async = []

    return g


# ---------------------------------------------------------------------------
# sec 7.3: how a value crosses a partition boundary
# ---------------------------------------------------------------------------
class Communication(Enum):
    COPY = 'copy the constant'
    REMATERIALIZE = 'recompute in the consuming role'
    ABI = 'each role reads it from the argument buffer'
    SHARED_WINDOW = 'shared RF window'
    MAILBOX = 'synchronised RF mailbox'
    GLOBAL = 'global temporary'
    # Not one of sec 7.3's six, because it is not communication at all.
    # Barriers, descriptors, windows and accumulators are per-core objects
    # allocated for the whole program (sec 10.1), so every role names the same
    # one by the same static index. Treating a descriptor handle as a value to
    # be moved would send it through a global temporary, which is both a copy
    # and wrong.
    SHARED_RESOURCE = 'a per-core resource, named by the same index in every role'


def choose_communication(value, producing_op, info, *, is_kernel_arg=False):
    """sec 7.3's order, applied in order.

    The order matters: a tensor must reach step 4 and become a shared window,
    because sending it through a global temporary would be a copy through
    memory in a machine whose whole point is not having one.
    """
    if value.kind in ('desc', 'barrier', 'acc'):
        return Communication.SHARED_RESOURCE
    if producing_op is not None and producing_op.kind == 'const':
        return Communication.COPY
    if is_kernel_arg:
        return Communication.ABI
    if producing_op is not None and info.can_rematerialize(producing_op):
        return Communication.REMATERIALIZE
    if value.kind in ('window', 'vector'):
        return Communication.SHARED_WINDOW
    if value.kind in ('scalar', 'token', 'event'):
        return Communication.MAILBOX
    return Communication.GLOBAL


# ---------------------------------------------------------------------------
# sec 7.2 / 7.6: partitioning, then mapping
# ---------------------------------------------------------------------------
@dataclass
class Partition:
    label: str
    cls: OpClass
    nodes: list = field(default_factory=list)

    def __repr__(self):
        return f"{self.label}({len(self.nodes)} ops)"


@dataclass
class PartitionPlan:
    partitions: list = field(default_factory=list)
    assignment: dict = field(default_factory=dict)   # node index -> label
    crossings: list = field(default_factory=list)    # (value, from, to, how)
    roles: dict = field(default_factory=dict)        # rctx -> [labels]
    merges: list = field(default_factory=list)

    def describe(self):
        lines = ["logical partitions:"]
        for p in self.partitions:
            lines.append(f"  {p.label:<10s} {p.cls.value:<22s} "
                         f"{len(p.nodes)} op(s)")
        lines.append("cross-partition values:")
        if not self.crossings:
            lines.append("  none")
        for v, a, b, how in self.crossings:
            lines.append(f"  {v.name:<12s} {a} -> {b:<10s} {how.value}")
        lines.append("physical mapping:")
        for rctx in sorted(self.roles):
            lines.append(f"  RCTX {rctx}: {', '.join(self.roles[rctx])}")
        for m in self.merges:
            lines.append(f"  merged: {m}")
        return '\n'.join(lines)


def partition_graph(graph, info):
    """Assign every node to a logical partition, by class.

    Logical only: sec 7.6 keeps the physical RCTX count out of this step, so
    the same graph serves a two-role and a four-role machine.
    """
    order = [OpClass.ADDRESS, OpClass.TRANSFER, OpClass.MATRIX,
             OpClass.VECTOR, OpClass.EPILOGUE, OpClass.SYNC]
    labels = {OpClass.ADDRESS: 'A', OpClass.TRANSFER: 'P',
              OpClass.MATRIX: 'M', OpClass.VECTOR: 'V',
              OpClass.EPILOGUE: 'E', OpClass.SYNC: 'S'}
    plan = PartitionPlan()
    by_cls = {}
    for cls in order:
        nodes = graph.of_class(cls)
        if not nodes:
            continue
        p = Partition(label=labels[cls], cls=cls,
                      nodes=[n.index for n in nodes])
        by_cls[cls] = p
        plan.partitions.append(p)
        for n in nodes:
            plan.assignment[n.index] = p.label

    # Which values cross, and how they will travel.
    seen = set()
    for e in graph.edges:
        if e.kind is not EdgeKind.DEF_USE or e.value is None:
            continue
        a = plan.assignment.get(e.src)
        b = plan.assignment.get(e.dst)
        if a is None or b is None or a == b:
            continue
        key = (id(e.value), a, b)
        if key in seen:
            continue
        seen.add(key)
        how = choose_communication(e.value, graph.nodes[e.src].op, info)
        plan.crossings.append((e.value, a, b, how))
    return plan


def map_to_rctx(plan, info):
    """sec 7.6: the physical mapping, last.

    When there are more logical partitions than Role Contexts they are merged,
    rather than the graph having been built differently. A four-context
    machine runs the same plan with fewer merges.
    """
    budget = info.budget()
    available = budget['rctx']
    if available < 1:
        raise Diagnostic('a target with no Role Contexts')

    # Address and synchronisation are not roles of their own: every role needs
    # its own addresses (sec 7.3 rematerialises them) and its own barrier
    # operations.
    workers = [p for p in plan.partitions
               if p.cls not in (OpClass.ADDRESS, OpClass.SYNC)]
    if not workers:
        plan.roles = {0: [p.label for p in plan.partitions]}
        return plan

    groups = [[p] for p in workers]
    while len(groups) > available:
        best = None
        for i in range(len(groups) - 1):
            a, b = groups[i][-1], groups[i + 1][0]
            if not info.can_merge(a.cls, b.cls):
                continue
            # Merge the pair whose crossing traffic is largest: the values
            # that stop crossing are the saving.
            weight = sum(1 for v, x, y, _ in plan.crossings
                         if {x, y} == {a.label, b.label})
            if best is None or weight > best[0]:
                best = (weight, i)
        if best is None:
            # Nothing may merge, yet there are too many groups. sec 11.1:
            # say so rather than silently serialising.
            raise Diagnostic(
                f"{len(groups)} partitions cannot be merged onto "
                f"{available} Role Context(s)",
                because='the remaining pairs are ones that must stay apart; '
                        'merging transfer with matrix would serialise the '
                        'pipeline specialisation exists to create',
                remedy='a target with more Role Contexts, or a coarser '
                       'classification',
                source='compiler_overview.md sec 7.6, sec 10.3')
        _, i = best
        merged = groups[i] + groups[i + 1]
        plan.merges.append(
            f"{'+'.join(p.label for p in groups[i])} with "
            f"{'+'.join(p.label for p in groups[i + 1])} "
            f"({available} Role Contexts available)")
        groups[i:i + 2] = [merged]

    plan.roles = {r: [p.label for p in group] for r, group in enumerate(groups)}
    # Address and synchronisation live in every role.
    for p in plan.partitions:
        if p.cls in (OpClass.ADDRESS, OpClass.SYNC):
            for r in plan.roles:
                plan.roles[r].append(p.label)
    return plan


def specialize(region, target):
    """The whole pipeline of sec 7, in order."""
    info = SpecializationTargetInfo(target)
    graph = build_graph(region, info)
    plan = partition_graph(graph, info)
    map_to_rctx(plan, info)
    return graph, plan


# ---------------------------------------------------------------------------
# sec 7.4 / 7.5: realise the plan as roles over an RF channel
# ---------------------------------------------------------------------------
def specialize_gemm(target, gemm_plan, *, name='cs_gemm', a_desc_addr,
                    b_desc_addr, out_addr, stages=2):
    """Build a specialised GEMM from an automatic partition.

    sec 15 C4 says to support GEMM-shaped loops first -- 先支持 GEMM 类循环,
    再扩大控制流范围 -- and sec 7.5's first version restricts itself to
    structured loops with an iteration count every role can derive and a fixed
    number of channels. This is that case.

    The partitioner decides the roles; this realises them. The producer role
    is the TRANSFER partition, the consumer is MATRIX merged with EPILOGUE
    where only two Role Contexts exist. On a four-context machine the same
    plan would merge less, which is sec 7.6's point: one graph, two mappings,
    not two compilers.
    """
    from .gemm import build_gemm
    from .ir import Builder, make_kernel

    info = SpecializationTargetInfo(target)

    # 1. Build the un-specialised kernel and partition it. The plan is read
    #    for its decisions; the loop below realises them.
    single, _ = build_gemm(target, gemm_plan, name=name,
                           a_desc_addr=a_desc_addr, b_desc_addr=b_desc_addr,
                           out_addr=out_addr)
    graph = build_graph(single.role(0).region, info)
    plan = partition_graph(graph, info)
    map_to_rctx(plan, info)

    producer_role = next((r for r, labels in plan.roles.items()
                          if 'P' in labels), None)
    if producer_role is None:
        raise Diagnostic(
            'the partition put no transfer work in any role',
            source='compiler_overview.md sec 7.6')
    consumer_role = next((r for r in plan.roles if r != producer_role), None)
    if consumer_role is None:
        raise Diagnostic(
            'only one Role Context: there is nothing to specialise',
            because='sec 6.3 requires a 2-RCTX profile to be supported, and '
                    'this target reports fewer',
            source='compiler_overview.md sec 6.3')

    steps = gemm_plan.k_steps
    stages = min(stages, max(1, steps))

    k = make_kernel(name, num_roles=2)
    b = Builder(k, k.role(0).region)

    # 2. Whole-program resources first (sec 10.1). Two windows per slot,
    #    because A and B are read simultaneously by one MMA command.
    wa = [b.window(gemm_plan.window_registers, name=f"wa{s}")
          for s in range(stages)]
    wb = [b.window(gemm_plan.window_registers, name=f"wb{s}")
          for s in range(stages)]
    ready = [b.barrier(name=f"ready{s}") for s in range(stages)]
    free = [b.barrier(name=f"free{s}") for s in range(stages)]
    mma_bar = b.barrier(name='mma')
    acc = b.accumulator(name='acc')

    s_a = b.const(a_desc_addr)
    da = b.descriptor(s_a, name='da')
    s_b = b.const(b_desc_addr)
    db = b.descriptor(s_b, name='db')

    # 3. Producer: the TRANSFER partition.
    b.comment(f"role {producer_role}: partition P (load/transfer). "
              f"{steps} k step(s) over a {stages}-slot RF channel.")
    # No startup flag: barriers are quiescent at kernel entry (ISA sec 9A.7),
    # so the consumer may sleep on `ready` generation 1 immediately.

    fm, fn, fk = target.mma_shape
    for step in range(steps):
        slot = step % stages
        gen = step // stages + 1
        b.comment(f"k step {step}: slot {slot}, generation {gen}")
        if step >= stages:
            b.barrier_wait(b.barrier_token(free[slot], gen - 1))
        tok = b.barrier_arm(ready[slot], arrivals=0, ops=2)
        b.dma_load(wa[slot], da, [b.const(0), b.const(step * fk)], tok,
                   dtype=gemm_plan.dtype)
        b.dma_load(wb[slot], db, [b.const(step * fk), b.const(0)], tok,
                   dtype=gemm_plan.dtype)
    b.exit()

    # 4. Consumer: MATRIX, merged with EPILOGUE on a two-context target.
    cb = Builder(k, k.role(1).region)
    cb.comment(f"role {consumer_role}: partitions "
               f"{'+'.join(l for l in plan.roles[consumer_role] if l in 'ME')}"
               " (matrix, and the epilogue merged in because this target has "
               f"{target.num_rctx} Role Contexts -- sec 7.6)")
    cb.acc_zero(acc)

    for step in range(steps):
        slot = step % stages
        gen = step // stages + 1
        cb.comment(f"k step {step}: slot {slot}, generation {gen}")
        cb.barrier_wait(cb.barrier_token(ready[slot], gen))
        mtok = cb.barrier_arm(mma_bar, arrivals=0, ops=1)
        cb.mma_async(acc, wa[slot], wb[slot], mtok, dtype=gemm_plan.dtype)
        # ISA v0.1 has one completion covering source and destination, so the
        # fragments are not reusable until the whole command retires. Releasing
        # the slot before this wait would let the producer overwrite a fragment
        # the matrix unit is still reading (ISA sec 9C.3).
        cb.barrier_wait(mtok)
        cb.barrier_arm(free[slot], arrivals=1, ops=0)
        cb.barrier_arrive(free[slot])

    # 5. Epilogue, in the same role because of the merge.
    cb.comment("epilogue: the accumulator leaves a row at a time")
    s_out = cb.const(out_addr)
    row_bytes = gemm_plan.n * 4
    mask = None
    if gemm_plan.n_tail:
        mask = cb.vcmp('lt', cb.arange(), cb.splat(cb.const(gemm_plan.n)))
    for row in range(gemm_plan.m):
        v = cb.acc_read(acc, cb.const(row))
        addr = (s_out if row == 0
                else cb.sbini('add', s_out, row * row_bytes))
        cb.store(addr, v, mask=mask)
    cb.exit()

    return k, plan


def specialize_gemm_rolled(target, gemm_plan, *, name='cs_gemm_rolled',
                           a_desc_addr, b_desc_addr, out_addr, stages=2):
    """The same specialisation, with the k loop rolled rather than unrolled.

    sec 15 C4 lists loop cloning among what this milestone builds, and sec 7.5
    restricts the first version to structured loops whose trip count every
    role can derive. This is that: each role gets its own copy of the loop,
    the loop body covers `stages` consecutive k steps so every W, V and A
    number stays static (sec 9 -- they are encoding fields, and `slot = i % 2`
    cannot become a computed register base), and the trip count is the same in
    both roles.

    The generation is computed rather than baked in. sec 7.3: 对于静态
    pipeline，epoch 通常可以由各 role 根据迭代编号独立计算. The consumer cannot
    read the producer's token -- the scalar file is per-RCTX private -- but it
    can derive the same epoch from its own trip counter.
    """
    from .ir import Builder, make_kernel

    info = SpecializationTargetInfo(target)
    steps = gemm_plan.k_steps
    if steps % stages:
        raise Diagnostic(
            f"{steps} k steps do not divide into {stages}-step loop trips",
            because='the body covers one trip through the slot ring, so every '
                    'W and V number stays static (sec 9)',
            remedy=f"a K that is a multiple of {stages * gemm_plan.frag_k}, "
                   'or the unrolled form',
            source='compiler_overview.md sec 7.5, sec 9')
    trips = steps // stages
    if trips < 1:
        raise Diagnostic('a rolled loop needs at least one trip')

    k = make_kernel(name, num_roles=2)
    b = Builder(k, k.role(0).region)

    wa = [b.window(gemm_plan.window_registers, name=f"wa{s}")
          for s in range(stages)]
    wb = [b.window(gemm_plan.window_registers, name=f"wb{s}")
          for s in range(stages)]
    ready = [b.barrier(name=f"ready{s}") for s in range(stages)]
    free = [b.barrier(name=f"free{s}") for s in range(stages)]
    mma_bar = b.barrier(name='mma')
    acc = b.accumulator(name='acc')

    da = b.descriptor(b.const(a_desc_addr), name='da')
    db = b.descriptor(b.const(b_desc_addr), name='db')
    fm, fn, fk = target.mma_shape

    # ---- producer -------------------------------------------------------
    b.comment(f"role 0: partition P, k loop rolled -- {trips} trip(s) of "
              f"{stages} step(s). Every W and V number in the body is static.")
    trip = b.const(0)
    k_off = b.const(0)
    with b.loop(trip, trips, 'p_loop'):
        for slot in range(stages):
            if trips > 1:
                # Generation `trip`, not `trip + 1`: this waits for the
                # consumer's release of the PREVIOUS trip through the ring.
                # On trip 0 that is epoch 0, which every barrier starts at
                # (ISA sec 9A.7), so the first trip does not block.
                b.barrier_wait(b.barrier_token_dyn(free[slot], trip))
            tok = b.barrier_arm(ready[slot], arrivals=0, ops=2)
            c1 = (k_off if slot == 0
                  else b.sbini('add', k_off, slot * fk))
            b.dma_load(wa[slot], da, [b.const(0), c1], tok,
                       dtype=gemm_plan.dtype)
            b.dma_load(wb[slot], db, [c1, b.const(0)], tok,
                       dtype=gemm_plan.dtype)
        nxt = b.sbini('add', k_off, stages * fk)
        b._emit('assign', [], [k_off, nxt])
    b.exit()

    # ---- consumer -------------------------------------------------------
    cb = Builder(k, k.role(1).region)
    cb.comment("role 1: partitions M+E, the same loop cloned. The generation "
               "is derived from this role's own trip counter (sec 7.3).")
    cb.acc_zero(acc)
    ctrip = cb.const(0)
    with cb.loop(ctrip, trips, 'c_loop'):
        cgen = cb.sbini('add', ctrip, 1)
        for slot in range(stages):
            cb.barrier_wait(cb.barrier_token_dyn(ready[slot], cgen))
            mtok = cb.barrier_arm(mma_bar, arrivals=0, ops=1)
            cb.mma_async(acc, wa[slot], wb[slot], mtok,
                         dtype=gemm_plan.dtype)
            cb.barrier_wait(mtok)
            cb.barrier_arm(free[slot], arrivals=1, ops=0)
            cb.barrier_arrive(free[slot])

    cb.comment("epilogue")
    s_out = cb.const(out_addr)
    row_bytes = gemm_plan.n * 4
    mask = None
    if gemm_plan.n_tail:
        mask = cb.vcmp('lt', cb.arange(), cb.splat(cb.const(gemm_plan.n)))
    for row in range(gemm_plan.m):
        v = cb.acc_read(acc, cb.const(row))
        addr = s_out if row == 0 else cb.sbini('add', s_out, row * row_bytes)
        cb.store(addr, v, mask=mask)
    cb.exit()

    graph = build_graph(k.role(0).region, info)
    plan = partition_graph(graph, info)
    map_to_rctx(plan, info)
    return k, plan


# ---------------------------------------------------------------------------
# sec 7.5: split one region into per-role regions
# ---------------------------------------------------------------------------

# Operations with no memory access, no resource effect and no engine command.
# A pure sub-DAG can be recomputed in whichever role needs it instead of being
# communicated, which is sec 7.3 steps 1-3 taken to their conclusion: a
# constant is copied, cheap address arithmetic is rematerialised, and the
# distinction between the two is not worth drawing when both are free.
_PURE = {'const', 'program_id', 'special', 'sbin', 'sbini', 'scmp',
         'arange', 'splat', 'vmovi', 'vbin', 'vun', 'vcmp', 'vselect',
         'vfma', 'vfms', 'vcvt',
         'barrier_token', 'barrier_token_dyn', 'comment'}


def _rematerializable(index, ops, producer_of, memo):
    """True when op `index` and everything it depends on is pure.

    Recursive on purpose. A predicate is one `v.cmp`, but it is computed from
    `pid*BLOCK + arange`, and recomputing the comparison while communicating
    its inputs would be worse than recomputing the lot. Anything rooted in
    memory or in an engine stops the recursion and has to travel as a value.
    """
    if index in memo:
        return memo[index]
    op = ops[index]
    if op.kind not in _PURE:
        memo[index] = False
        return False
    memo[index] = True          # provisional, breaks cycles
    for v in list(op.operands) + [op.attrs.get('mask')]:
        if v is None or not getattr(v, 'kind', None):
            continue
        src = producer_of.get(id(v))
        if src is None:
            continue            # a kernel argument: every role has it
        if not _rematerializable(src, ops, producer_of, memo):
            memo[index] = False
            return False
    return True


@dataclass
class SplitResult:
    kernel: object
    plan: PartitionPlan
    crossings: list = field(default_factory=list)   # (Value, producer, consumers)
    rematerialised: int = 0
    windows: int = 0

    def describe(self):
        lines = [f"roles: {len(self.kernel.roles)}"]
        for r, labels in sorted(self.plan.roles.items()):
            workers = [l for l in labels if l not in ('A', 'S')]
            lines.append(f"  role {r}: {'+'.join(workers) or 'address only'} "
                         f"({len(self.kernel.role(r).region.ops)} ops)")
        lines.append(f"values crossing on a shared window: {len(self.crossings)}")
        for v, prod, cons in self.crossings:
            lines.append(f"  {v.name} role {prod} -> {sorted(cons)}")
        lines.append(f"operations rematerialised: {self.rematerialised}")
        return '\n'.join(lines)


def split_roles(kernel, target, *, name=None):
    """Turn a single-role kernel into one region per Role Context.

    compiler_overview.md sec 7.5 restricts the first version of role-loop
    construction to structured loops whose trip count every role can derive.
    This handles the straight-line case, which is what the Triton frontend
    produces for one program id, and it is the step that makes the path from
    a @triton.jit kernel to context-specialised ISA continuous rather than
    two halves that each work.

    What it does, in sec 7.3's order:

      * a value whose whole dependency chain is pure is recomputed in every
        role that needs it, rather than communicated;
      * a tensor that cannot be recomputed -- anything rooted in memory or in
        an engine -- crosses on a shared RF window, which sec 7.3 ranks above
        every other option and which is the only one TNCA has, there being no
        scratchpad;
      * the handoff is ordered by a barrier, not by hope: the producer arrives
        after its last write and each consumer waits before its first read.
        ISA sec 9A.5 makes that arrival cover the outstanding register write,
        which is exactly the guarantee needed.
    """
    from .ir import Builder, make_kernel

    region = kernel.role(0).region
    info = SpecializationTargetInfo(target)
    graph = build_graph(region, info)
    plan = partition_graph(graph, info)
    map_to_rctx(plan, info)

    ops = region.ops
    producer_of = {}
    for i, op in enumerate(ops):
        for v in op.results:
            producer_of[id(v)] = i

    # Address and synchronisation are not roles; every role carries its own.
    replicated = {'A', 'S'}
    owner = {}
    for i, label in plan.assignment.items():
        if label in replicated:
            continue
        for r, labels in sorted(plan.roles.items()):
            if label in labels:
                owner[i] = r
                break

    n_roles = len(plan.roles)
    if n_roles < 2:
        return SplitResult(kernel=kernel, plan=plan)

    memo = {}

    def operands_of(op):
        out = [v for v in op.operands if getattr(v, 'kind', None)]
        m = op.attrs.get('mask')
        if m is not None:
            out.append(m)
        return out

    # ---- what each role must contain ------------------------------------
    # Ops every role must carry whatever its partition: control flow, and
    # anything with an effect. Pure address arithmetic is NOT seeded -- it is
    # pulled in only by a role that actually uses it, or the epilogue role
    # ends up recomputing the load addresses it never reads.
    structural = {'label', 'branch_nz', 'assign', 'trap', 'exit',
                  'assert_eq', 'assert_lane', 'sstore', 'barrier_init',
                  'barrier_arm', 'barrier_wait', 'barrier_arrive',
                  'window_def', 'desc_snapshot'}

    need = {r: set() for r in plan.roles}          # op indices
    crossing = {}                                  # op index -> consumer roles
    for r in plan.roles:
        work = [i for i, o in owner.items() if o == r]
        work += [i for i, o in enumerate(ops)
                 if o.kind in structural and plan.assignment.get(i) in replicated]
        seen = set()
        while work:
            i = work.pop()
            if i in seen:
                continue
            seen.add(i)
            for v in operands_of(ops[i]):
                src = producer_of.get(id(v))
                if src is None or src in seen:
                    continue
                if owner.get(src) == r or plan.assignment.get(src) in replicated:
                    work.append(src)
                elif _rematerializable(src, ops, producer_of, memo):
                    work.append(src)
                else:
                    crossing.setdefault(src, set()).add(r)
        need[r] = seen

    # A crossing consumed by its own producing role is not a crossing.
    for src in list(crossing):
        crossing[src].discard(owner.get(src))
        if not crossing[src]:
            del crossing[src]

    # ---- build the new kernel -------------------------------------------
    out = make_kernel(name or kernel.name, num_roles=n_roles)
    shared = Builder(out, out.role(0).region)

    ordered = sorted(crossing)
    window = None
    if ordered:
        window = shared.window(len(ordered), name='xfer')
    slot_of = {src: k for k, src in enumerate(ordered)}

    # One barrier per producing role. The producer arrives once, after its
    # last handoff; every consumer waits once, before its first read.
    producers = sorted({owner[src] for src in ordered if src in owner})
    ready = {r: shared.barrier(name=f"ready{r}") for r in producers}

    builders = {}
    for r in range(n_roles):
        builders[r] = shared if r == 0 else Builder(out, out.role(r).region)

    remat_count = 0
    for r in range(n_roles):
        b = builders[r]
        vmap = {}
        workers = [l for l in plan.roles[r] if l not in replicated]
        b.comment(f"role {r}: partition {'+'.join(workers) or 'address'}"
                  + (", regenerating the pure values it needs (sec 7.3)"
                     if r else ""))

        tok = None
        if r in ready:
            tok = b.barrier_arm(ready[r], arrivals=1, ops=0)

        waited = set()
        for i, op in enumerate(ops):
            if i not in need[r]:
                continue
            if op.kind == 'exit':
                continue

            # Before the first use of a value another role produces, wait.
            for v in operands_of(op):
                src = producer_of.get(id(v))
                if src in crossing and r in crossing[src] and src not in waited:
                    prod = owner[src]
                    if prod not in waited:
                        b.comment(f"wait for role {prod}'s handoff")
                        b.barrier_wait(b.barrier_token(ready[prod], 1))
                        waited.add(prod)
                    waited.add(src)
                    vmap[id(v)] = b.window_reg(window, slot_of[src],
                                               dtype=v.dtype)

            if plan.assignment.get(i) not in replicated and owner.get(i) != r:
                remat_count += 1

            clone = Op(kind=op.kind,
                       results=[Value(name=v.name, kind=v.kind, dtype=v.dtype,
                                      length=v.length) for v in op.results],
                       operands=[vmap.get(id(v), v) for v in op.operands],
                       attrs=dict(op.attrs))
            m = clone.attrs.get('mask')
            if m is not None:
                clone.attrs['mask'] = vmap.get(id(m), m)
            for old, new in zip(op.results, clone.results):
                vmap[id(old)] = new
            b.region.ops.append(clone)

            # Hand off anything this role produces for another.
            if i in crossing and owner.get(i) == r:
                b.comment(f"hand {op.results[0].name} to "
                          f"role(s) {sorted(crossing[i])}")
                b.window_put(window, slot_of[i], clone.results[0])

        if r in ready:
            b.comment("release the consumers: the arrival covers this role's "
                      "outstanding register write (ISA sec 9A.5)")
            b.barrier_arrive(ready[r])
        b.exit()

    return SplitResult(kernel=out, plan=plan,
                       crossings=[(ops[src].results[0], owner[src],
                                   crossing[src]) for src in ordered],
                       rematerialised=remat_count,
                       windows=1 if ordered else 0)
