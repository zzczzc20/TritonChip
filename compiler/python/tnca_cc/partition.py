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
    'acc_read': OpClass.MATRIX, 'acc_write': OpClass.MATRIX,

    'arange': OpClass.VECTOR, 'splat': OpClass.VECTOR,
    'vmovi': OpClass.VECTOR, 'vbin': OpClass.VECTOR,
    'vcmp': OpClass.VECTOR, 'vselect': OpClass.VECTOR,
    'vun': OpClass.VECTOR, 'vfma': OpClass.VECTOR,
    'pbin': OpClass.VECTOR, 'pun': OpClass.VECTOR,
    'vextract': OpClass.VECTOR, 'vinsert': OpClass.VECTOR,
    'vshuffle': OpClass.VECTOR,
    'vfms': OpClass.VECTOR, 'vcvt': OpClass.VECTOR,
    'load': OpClass.VECTOR, 'gather': OpClass.VECTOR,
    'reduce': OpClass.VECTOR,

    'atomic': OpClass.EPILOGUE, 'atomic_cas': OpClass.EPILOGUE,
    'fence': OpClass.EPILOGUE,
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
         'vfma', 'vfms', 'vcvt', 'pbin', 'pun',
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


# ---------------------------------------------------------------------------
# R39-12: Mode A on the standard path -- the upstream pipeline, on the
# importer's IR
# ---------------------------------------------------------------------------
#
# Upstream's automatic warp specialisation (triton
# lib/Dialect/TritonGPU/Transforms/WarpSpecialization) runs, in order:
#
#   PartitionScheduling  "data" ops are seeded -- descriptor loads, MMAs --
#                        and merged under heuristics and constraints; then
#                        propagatePartitions pulls every non-data op into
#                        each partition that uses it (duplicating it), and
#                        duplicateCheapOps recomputes cheap chains instead of
#                        communicating them;
#   InsertAref/LowerAref every value that crosses partitions inside the loop
#                        becomes a ring of numStages buffers with a full and
#                        an empty barrier per buffer;
#   PartitionLoops       each partition gets its own copy of the loop.
#
# The same three steps run here, with this machine's answers where upstream's
# are NVIDIA's (sec 7.2):
#
#   * the producer is the TRANSFER partition -- each `t.ld.async` feeding an
#     MMA, and whatever builds what it reads -- and the consumer is MATRIX
#     merged with the EPILOGUE: two roles per program, one Role Context pair
#     (RPAIR) each (sec 7.6);
#   * a ring buffer is a window per stage: the register file is the only
#     buffer there is (sec 7.3);
#   * each role counts the epochs it waits on itself -- the scalar file is
#     per context (sec 7.3), so nothing but the barriers is exchanged;
#   * a W number is an encoding field (sec 9), so the loop body is unrolled
#     once per stage instead of indexing the ring.
#
# Twill (Soi et al., "Optimal Software Pipelining and Warp Specialization for
# Tensor Core GPUs", arXiv 2512.18134) frames the choice as one of initiation
# interval: variable-latency transfers on a warp of their own, and the
# schedule taken only where its steady state beats the serial one. The paper
# releases no code; `ring_estimate` asks its question of the R39-01 machine
# model and the result is recorded with the kernel.

_EFFECT = frozenset({
    'dma_load', 'dma_store', 'mma_async', 'acc_zero', 'acc_write', 'acc_read',
    'barrier_init', 'barrier_arm', 'barrier_wait', 'barrier_arrive',
    'window_def', 'window_put', 'desc_snapshot', 'store', 'scatter', 'sstore',
    'fence', 'atomic', 'atomic_cas', 'assert_eq', 'assert_lane', 'trap'})
_MEM_WRITES = frozenset({'store', 'scatter', 'sstore', 'fence', 'atomic',
                         'atomic_cas'})
_ALWAYS = frozenset({'label', 'branch_nz', 'exit', 'comment'})
_IN_PLACE = {'assign': 0, 'vinsert': 0}
_RESOURCE = frozenset({'window', 'desc', 'barrier', 'acc'})
PRODUCER, CONSUMER = 0, 1
BOTH = 2          # an op each role runs its own copy of


class NotSpecialised(Exception):
    """Why a kernel keeps one role per program. Not an error: the kernel is
    compiled as it was."""


@dataclass
class RingLoop:
    head: int            # the loop's label
    test: int            # its exit branch
    latch: int           # first op of the increment that ends the body
    back: int            # the back branch
    windows: list        # windows a transfer fills and an MMA reads
    direct: bool         # transfers complete on the ring's barrier


def _operands(op):
    out = list(op.operands)
    m = op.attrs.get('mask')
    if m is not None:
        out.append(m)
    return [v for v in out if isinstance(v, Value)]


def find_ring_loop(ops):
    """The loop to specialise: the innermost one holding both a transfer
    into a window and an MMA reading that window, in the importer's shape --

        L:  <exit test> branch_nz E
            <body, straight-line>
            <increment>  branch_nz L
        E:
    """
    labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
    loops = [(labels[op.attrs['target']], i) for i, op in enumerate(ops)
             if op.kind == 'branch_nz' and labels.get(op.attrs.get('target'), i) < i]
    found = []
    for head, back in loops:
        if any(head < h < back for h, b in loops if (h, b) != (head, back)):
            continue                                   # not innermost
        body = ops[head + 1:back]
        fills = {id(op.operands[0]): op for op in body if op.kind == 'dma_load'}
        reads = {id(w) for op in body if op.kind == 'mma_async' for w in op.operands[1:3]}
        windows = [op.operands[0] for k, op in fills.items() if k in reads]
        if windows:
            found.append((head, back, windows))
    if not found:
        raise NotSpecialised('no loop fills a window that an MMA in it reads')
    if len(found) > 1:
        raise NotSpecialised(f'{len(found)} loops feed an MMA; one is specialised')
    head, back, windows = found[0]
    end = ops[back + 1] if back + 1 < len(ops) else None
    test = next((i for i in range(head + 1, back) if ops[i].kind == 'branch_nz'), None)
    if (test is None or end is None or end.kind != 'label'
            or ops[test].attrs.get('target') != end.attrs['name']):
        raise NotSpecialised('the loop does not exit by one test at its head')
    # The increment: the assigns and the pure scalar ops at the end of the
    # body that only they and the back branch read.
    latch = back
    while latch - 1 > test:
        op = ops[latch - 1]
        if op.kind != 'assign' and not (op.kind in ('const', 'sbin', 'sbini', 'scmp')
                                         and all(id(r) in _read_from(ops, latch, back + 1)
                                                 and id(r) not in _read_from(ops, test + 1, latch - 1)
                                                 for r in op.results)):
            break
        latch -= 1
    for i in range(test + 1, latch):
        if ops[i].kind in ('label', 'branch_nz', 'exit'):
            raise NotSpecialised('the loop body is not straight-line')
    body = ops[test + 1:latch]
    # The transfers complete on the ring's barrier unless the producer
    # writes memory in the loop -- the gathered-and-repacked operand of a
    # target without `tensor_transpose` -- or builds a map in it: then it
    # waits for its own transfers before overwriting what they read, and
    # arrives on the ring itself.
    direct = not any(op.kind in _MEM_WRITES or op.kind == 'desc_snapshot' for op in body)
    return RingLoop(head, test, latch, back, windows, direct)


def _read_from(ops, a, b):
    return {id(v) for op in ops[a:b] for v in _operands(op)}


def _clone(op, vmap):
    """A copy of `op` with fresh results, reading through `vmap`. A window,
    descriptor, barrier or accumulator is a per-core resource and keeps its
    identity -- the kernel's lists name it -- unless `vmap` maps it."""
    results = []
    for r in op.results:
        if id(r) not in vmap and r.kind not in _RESOURCE:
            vmap[id(r)] = Value(name=r.name, kind=r.kind, dtype=r.dtype, length=r.length)
        results.append(vmap.get(id(r), r))
    new = Op(kind=op.kind, results=results,
             operands=[vmap.get(id(v), v) if isinstance(v, Value) else v for v in op.operands],
             attrs=dict(op.attrs))
    m = new.attrs.get('mask')
    if isinstance(m, Value):
        new.attrs['mask'] = vmap.get(id(m), m)
    return new


def _unroll(kernel, ops, loop, stages):
    """The loop body once per ring stage, each stage on windows of its own:

        L:  test; body[stage 0]; increment;
                  test; body[stage 1]; increment; ... branch L
        E:

    Every copy keeps the exit test, so any trip count is right; a role's
    epochs are counted per stage, so one that leaves after stage 0 still
    agrees with the other role about every barrier."""
    test_ops = ops[loop.head + 1:loop.test + 1]
    body = ops[loop.test + 1:loop.latch]
    incr = ops[loop.latch:loop.back]
    out = list(ops[:loop.test + 1])
    ring = []                              # [stage] -> {window id: window}
    for s in range(stages):
        vmap = {}
        if s:
            for w in loop.windows:
                c = Value(name=f"{w.name}_s{s}", kind=w.kind, dtype=w.dtype, length=w.length)
                kernel.windows.append(c)
                vmap[id(w)] = c
            tmap = {}
            out += [_clone(op, tmap) for op in test_ops]
        ring.append({id(w): vmap.get(id(w), w) for w in loop.windows})
        for op in body:
            c = _clone(op, vmap)
            c.attrs['_stage'] = s
            out.append(c)
        if s < stages - 1:
            imap = {}
            out += [_clone(op, imap) for op in incr]
    out += ops[loop.latch:]
    return out, ring


def _assign(ops, loop, ring_ids):
    """PartitionScheduling: every op with an effect gets a partition.

    Seeds as upstream's: the transfers filling a ring window (descriptor
    loads) are the producer's and the MMAs the consumer's. Then the merges
    upstream's heuristics would make on this IR -- the whole load chain with
    its load (`load_local_alloc`, `load_epilog`), the epilogue with the MMA
    when there are two roles (sec 7.6) -- written as the rule each op meets.
    Pure ops get none here: `_pull` takes them into each role that needs
    them (propagatePartitions, duplicateCheapOps)."""
    first = loop.test + 1
    last = max(i for i, op in enumerate(ops) if op.attrs.get('_stage') is not None)
    token_of = {}
    for i, op in enumerate(ops):
        if op.kind in ('dma_load', 'dma_store', 'mma_async'):
            token_of[id(op.operands[2] if op.kind != 'mma_async' else op.operands[3])] = i
    part = {}
    for i, op in enumerate(ops):
        k = op.kind
        if k == 'dma_load' and id(op.operands[0]) in ring_ids:
            part[i] = PRODUCER
        elif k in ('mma_async', 'acc_zero', 'acc_write', 'acc_read'):
            part[i] = CONSUMER
        elif k == 'window_def' and id(op.results[0]) in ring_ids:
            part[i] = PRODUCER
        elif k in _EFFECT and first <= i <= last:
            part[i] = PRODUCER          # the transfer chain, and its waits
        elif k in ('store', 'scatter', 'sstore', 'fence') and i < first:
            # Memory written before the loop may be read by either role (a
            # carried tile's spill, its initial value): each writes it
            # itself. The producer's copy is ordered before the consumer's
            # first read of anything by the ring; the values are the same.
            part[i] = BOTH
        elif k in _EFFECT:
            part[i] = CONSUMER          # the epilogue, and anything else outside
    # A barrier op goes with the command its token counts.
    arm_of = {}
    for i, op in enumerate(ops):
        if op.kind == 'barrier_arm':
            user = token_of.get(id(op.results[0]))
            if user is not None:
                part[i] = part.get(user, part.get(i))
            arm_of[id(op.results[0])] = i
    for i, op in enumerate(ops):
        if op.kind == 'barrier_wait' and id(op.operands[0]) in arm_of:
            part[i] = part[arm_of[id(op.operands[0])]]
    # A TensorMap's snapshot goes with its transfers, and its stores with it.
    users = {}
    for i, op in enumerate(ops):
        for v in _operands(op):
            users.setdefault(id(v), []).append(i)
    maps = {}
    for i, op in enumerate(ops):
        if op.kind == 'desc_snapshot':
            roles = {part[u] for u in users.get(id(op.results[0]), []) if u in part}
            if len(roles) > 1:
                raise NotSpecialised(f'descriptor {op.results[0].name} is read by both roles')
            if roles:
                part[i] = roles.pop()
            maps[id(op.operands[0])] = part[i]
    for i, op in enumerate(ops):
        if op.kind == 'sstore' and id(op.operands[0]) in maps:
            part[i] = maps[id(op.operands[0])]
    return part


_WRITES = frozenset({'store', 'scatter', 'sstore', 'atomic', 'atomic_cas'})
_READS_MEM = frozenset({'load', 'gather', 'sload', 'atomic', 'atomic_cas'})


def _where(ops, v, producer):
    """An address as (the value it is built on, a byte offset or None):
    `sbini add` chains folded, anything else the root."""
    off = 0
    while True:
        j = producer.get(id(v))
        if j is None or ops[j].kind != 'sbini' or ops[j].attrs.get('op') != 'add':
            return id(v), off
        off += ops[j].attrs.get('imm', 0)
        v = ops[j].operands[0]


def _memory_crossings(ops, roles, first, last):
    """Refuse a split in which one role may read memory the other writes
    unordered. The ring's barriers order the windows and nothing else, so a
    tile one role spills to scratch in the loop and the other reads back --
    or a store the consumer makes that the producer's next program reads --
    would be a race. An address is compared as its root value and constant
    offset; a scatter's, or one that is not a constant from its root,
    covers the whole root. A TensorMap is its role's own: only that role's
    transfers read it."""
    producer = {id(r): i for i, op in enumerate(ops) for r in op.results}
    maps = {id(op.operands[0]) for op in ops if op.kind == 'desc_snapshot'}
    def spans(role, kinds, where):
        out = []
        for i in roles[role]:
            op = ops[i]
            if op.kind not in kinds or not where(i):
                continue
            if op.kind == 'sstore' and id(op.operands[0]) in maps:
                continue
            root, off = _where(ops, op.operands[0], producer)
            out.append((root, None if op.kind in ('scatter', 'gather') else off, op.kind))
        return out
    def clash(writes, reads):
        for wr, wo, wk in writes:
            for rr, ro, rk in reads:
                if wr == rr and (wo is None or ro is None or abs(wo - ro) < 64):
                    return f'{rk} may read what a {wk} of the other role wrote'
        return None
    in_loop = lambda i: first <= i <= last
    anywhere = lambda i: True
    why = (clash(spans(PRODUCER, _WRITES, in_loop), [r for r in spans(CONSUMER, _READS_MEM, anywhere)])
           or clash(spans(CONSUMER, _WRITES, anywhere), spans(PRODUCER, _READS_MEM, in_loop)))
    if why:
        raise NotSpecialised(f'memory crosses roles: {why}')


def _pull(ops, part, role):
    """propagatePartitions for one role: its own ops, the control flow, and
    everything they read -- duplicated rather than communicated, as
    duplicateCheapOps would. A value only another role's op can produce is
    a crossing; a ring window, a descriptor, a barrier or an accumulator is
    a per-core resource both name alike (sec 10.1) and is not one."""
    producer, mutators = {}, {}
    for i, op in enumerate(ops):
        for r in op.results:
            producer[id(r)] = i
        k = _IN_PLACE.get(op.kind)
        if k is not None and k < len(op.operands):
            mutators.setdefault(id(op.operands[k]), []).append(i)
    keep = set()
    work = [i for i, op in enumerate(ops)
            if part.get(i) in (role, BOTH) or op.kind in _ALWAYS]
    while work:
        i = work.pop()
        if i in keep:
            continue
        keep.add(i)
        op = ops[i]
        reads = _operands(op)
        if op.kind in _IN_PLACE:
            reads = [v for j, v in enumerate(op.operands) if j != _IN_PLACE[op.kind]
                     and isinstance(v, Value)]
        for v in reads:
            for j in [producer.get(id(v))] + mutators.get(id(v), []):
                if j is None or j in keep:
                    continue
                other = part.get(j)
                if other is not None and other not in (role, BOTH):
                    if v.kind in _RESOURCE:
                        continue
                    raise NotSpecialised(
                        f'{v.name} ({v.kind}) crosses from role {other} to role {role} '
                        f'at {ops[j].kind}')
                work.append(j)
    return sorted(keep)


def _ring_sync(role_ops, role, ring, full, empty, direct, stages):
    """InsertAref + LowerAref for one role: the waits, arms and arrivals of
    each stage's full/empty pair, with the epochs this role counts."""
    use = [Value(name=f'ring_use{s}', kind='scalar', dtype='i64') for s in range(stages)]
    out = [Op('const', [u], [], {'value': 0}) for u in use]
    span = {}
    for i, op in enumerate(role_ops):
        s = op.attrs.get('_stage')
        if s is not None:
            a, b = span.get(s, (i, i))
            span[s] = (min(a, i), max(b, i))
    start = {a: s for s, (a, b) in span.items()}
    end = {b: s for s, (a, b) in span.items()}
    ring_ids = {id(w): s for s in range(stages) for w in ring[s].values()}
    if role == PRODUCER and direct:
        # The importer arms a barrier per transfer and waits for it; on the
        # ring the transfers complete on the stage's `full` instead, and the
        # consumer is the one who waits.
        dropped, token_stage, count = set(), {}, {}
        for op in role_ops:
            if op.kind == 'dma_load' and id(op.operands[0]) in ring_ids:
                s = ring_ids[id(op.operands[0])]
                token_stage[id(op.operands[2])] = s
                count[s] = count.get(s, 0) + 1
        for i, op in enumerate(role_ops):
            if op.kind == 'barrier_arm' and id(op.results[0]) in token_stage:
                dropped.add(i)
            elif op.kind == 'barrier_wait' and id(op.operands[0]) in token_stage:
                dropped.add(i)
        full_tok = {}
    mma_wait = {}
    if role == CONSUMER:
        mma_tok = {}
        for op in role_ops:
            if op.kind == 'mma_async':
                for w in op.operands[1:3]:
                    if id(w) in ring_ids:
                        mma_tok[id(op.operands[3])] = ring_ids[id(w)]
        for i, op in enumerate(role_ops):
            if op.kind == 'barrier_wait' and id(op.operands[0]) in mma_tok:
                mma_wait[i] = mma_tok[id(op.operands[0])]
    for i, op in enumerate(role_ops):
        s = start.get(i)
        if s is not None:
            if role == PRODUCER:
                # Wait until the consumer has released this stage as many
                # times as it has been filled; on the first trip that is
                # epoch 0, which every barrier starts at (ISA sec 9A.7).
                tk = Value(name=f'ring_empty{s}', kind='token', dtype='i64')
                out += [Op('comment', [], [], {'text': f'ring stage {s}: wait until the consumer released it'}),
                        Op('barrier_token_dyn', [tk], [empty[s], use[s]]),
                        Op('barrier_wait', [], [tk], {'retires': [full[s]]})]
            else:
                g = Value(name=f'ring_gen{s}', kind='scalar', dtype='i64')
                tk = Value(name=f'ring_full{s}', kind='token', dtype='i64')
                out += [Op('comment', [], [], {'text': f'ring stage {s}: wait until the producer filled it'}),
                        Op('sbini', [g], [use[s]], {'op': 'add', 'imm': 1}),
                        Op('assign', [], [use[s], g]),
                        Op('barrier_token_dyn', [tk], [full[s], use[s]]),
                        Op('barrier_wait', [], [tk])]
        if role == PRODUCER and direct:
            if i in dropped:
                continue
            if op.kind == 'dma_load' and id(op.operands[0]) in ring_ids:
                s2 = ring_ids[id(op.operands[0])]
                if s2 not in full_tok:
                    full_tok[s2] = Value(name=f'ring_fill{s2}', kind='token', dtype='i64')
                    out.append(Op('barrier_arm', [full_tok[s2]], [full[s2]],
                                  {'arrivals': 0, 'ops': count[s2]}))
                op = Op(op.kind, op.results, [op.operands[0], op.operands[1], full_tok[s2]]
                        + list(op.operands[3:]), dict(op.attrs))
        out.append(op)
        if role == CONSUMER and i in mma_wait:
            s2 = mma_wait[i]
            t = Value(name=f'ring_release{s2}', kind='token', dtype='i64')
            out += [Op('barrier_arm', [t], [empty[s2]], {'arrivals': 1, 'ops': 0}),
                    Op('barrier_arrive', [], [empty[s2]])]
        s = end.get(i)
        if s is not None and role == PRODUCER:
            if not direct:
                t = Value(name=f'ring_filled{s}', kind='token', dtype='i64')
                out += [Op('barrier_arm', [t], [full[s]], {'arrivals': 1, 'ops': 0}),
                        Op('barrier_arrive', [], [full[s]])]
            n = Value(name=f'ring_next{s}', kind='scalar', dtype='i64')
            out += [Op('sbini', [n], [use[s]], {'op': 'add', 'imm': 1}),
                    Op('assign', [], [use[s], n])]
    if role == PRODUCER and direct:
        full_tok.clear()
    return out


def _rename_labels(ops, role):
    for op in ops:
        if op.kind == 'label':
            op.attrs = dict(op.attrs, name=f"{op.attrs['name']}__r{role}")
        elif op.kind == 'branch_nz':
            op.attrs = dict(op.attrs, target=f"{op.attrs['target']}__r{role}")


def specialize_loop(kernel, target, *, stages=2):
    """R39-12: one program as a producer and a consumer role over a ring.

    Takes a one-role kernel (after rctx_spmd, if it ran) and returns a
    description of what was done; the kernel is rebuilt in place with two
    roles. Raises NotSpecialised, leaving the kernel untouched, when it does
    not have the shape this handles.
    """
    if len(kernel.roles) != 1:
        raise NotSpecialised('the kernel already has more than one role')
    ops = kernel.roles[0].region.ops
    loop = find_ring_loop(ops)
    import copy
    saved = copy.copy(kernel.windows), copy.copy(kernel.barriers)
    try:
        unrolled, ring = _unroll(kernel, ops, loop, stages)
        ring_ids = {id(w) for st in ring for w in st.values()}
        # Where the unrolled loop is now.
        heads = [i for i, op in enumerate(unrolled) if op is ops[loop.head]]
        test = next(i for i in range(heads[0] + 1, len(unrolled))
                    if unrolled[i].kind == 'branch_nz')
        part = _assign(unrolled, RingLoop(heads[0], test, 0, 0, loop.windows, loop.direct),
                       ring_ids)
        roles_idx = {PRODUCER: _pull(unrolled, part, PRODUCER),
                     CONSUMER: _pull(unrolled, part, CONSUMER)}
        staged = [i for i, op in enumerate(unrolled) if op.attrs.get('_stage') is not None]
        _memory_crossings(unrolled, roles_idx, min(staged), max(staged))
        prod = [unrolled[i] for i in roles_idx[PRODUCER]]
        cons = [unrolled[i] for i in roles_idx[CONSUMER]]
    except NotSpecialised:
        kernel.windows[:], kernel.barriers[:] = saved
        raise
    full = [Value(name=f'ring_full{s}', kind='barrier') for s in range(stages)]
    empty = [Value(name=f'ring_empty{s}', kind='barrier') for s in range(stages)]
    kernel.barriers += full + empty
    roles = []
    for role, role_ops in ((PRODUCER, prod), (CONSUMER, cons)):
        # A role's own copies: the same op object in two roles would share
        # its attrs and, worse, its result Values' registers.
        vmap = {}
        mine = [_clone(op, vmap) for op in role_ops]
        mine = _ring_sync(mine, role, [{k: vmap.get(id(w), w) for k, w in st.items()}
                                       for st in ring], full, empty, loop.direct, stages)
        for op in mine:
            op.attrs.pop('_stage', None)
        _rename_labels(mine, role)
        roles.append(mine)
    from .ir import Role, Region
    kernel.roles = [Role(index=r, name=('producer', 'consumer')[r], region=Region(ops=roles[r]))
                    for r in (PRODUCER, CONSUMER)]
    used = {id(v) for r in roles for op in r for v in _operands(op)}
    kernel.barriers = [b for b in kernel.barriers if id(b) in used]
    kernel.windows = [w for w in kernel.windows
                      if any(op.kind == 'window_def' and op.results[0] is w for r in roles for op in r)
                      or id(w) in used]
    return {'stages': stages, 'direct': loop.direct, 'ring_windows': len(ring_ids),
            'producer_ops': len(roles[0]), 'consumer_ops': len(roles[1])}
