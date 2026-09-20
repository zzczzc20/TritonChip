#!/usr/bin/env python3
"""
A finite-resource performance model -- doc/phase3_tasks_and_acceptance.md R3-02.

    "显式建模依赖 DAG、issue 限制、RF bank/port、ARF 更新、DMA 请求/返回、
     MMA/SFU/reduction 占用、barrier/channel、有限队列 ... 输入为编译后的
     schedule/操作与 profile，不是反向读取该 workload 的 RTL stall trace 来
     '预测'"

The last clause is the one that decides whether a model is worth anything. A
model tuned against the RTL trace of the workload it is predicting is a
lookup table with extra steps: it will be exactly right on everything it has
seen and silently wrong on everything it has not. So this takes operations
and a machine description and nothing else, and it has no input through which
a measured cycle count could reach it.

What it models, and what each costs:

    issue        a width. An operation issues in a cycle where a slot is free
    RF ports     banked reads and a fixed number of writes; two reads of one
                 bank in one cycle is a conflict, and conflicts serialise
    units        MMA, SFU, DMA and the vector ALU each occupy for a duration
    queues       finite. A request to a full queue does not issue
    events       a barrier completes when its operations do; a lease is held
                 until it does

What it refuses to do, because each would make it flattering:

    - a free layout conversion. Every operation costs at least one issue slot
    - an infinite queue. A queue with no depth is a queue that never stalls
    - computing before the data arrives. A consumer cannot issue in the same
      cycle its producer completes unless the contract says forwarding exists
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def _unit(spec):
    """(occupancy, latency, instances) from a two- or three-tuple."""
    if len(spec) == 2:
        return spec[0], spec[1], 1
    return spec[0], spec[1], spec[2]


class ModelError(Exception):
    """An illegal machine or an unschedulable graph.

    Raised rather than returning a cycle count, because the failure mode
    R3-02 MODEL-03 names is a model that answers 0 for a configuration that
    cannot work. Zero cycles is the most attractive wrong answer available.
    """


@dataclass
class Machine:
    """What the model is allowed to assume about the hardware."""
    issue_width: int = 1
    rf_banks: int = 1
    rf_read_ports_per_bank: int = 1
    rf_write_ports: int = 2
    arf_write_ports: int = 1
    # unit -> (occupancy, latency, instances).
    #
    # Three numbers, not two, and the third is the one the first draft was
    # missing. Occupancy is how long ONE unit is busy; latency is when the
    # result is usable; instances is how many there are. Without the third,
    # four independent ALU operations took four cycles at any issue width --
    # which is correct for one ALU and was being read as a scheduling result.
    # An issue width wider than the units behind it is a model of a machine
    # that cannot exist.
    units: dict = field(default_factory=lambda: {
        'alu': (1, 1, 1),
        'mma': (1, 224, 1),
        'dma': (1, 72, 1),
        'sfu': (1, 7, 1),
    })
    queues: dict = field(default_factory=lambda: {'dma': 4, 'mma': 2})
    forwarding: bool = False

    def validate(self):
        problems = []
        if self.issue_width < 1:
            problems.append(f"issue_width={self.issue_width} issues nothing")
        if self.rf_banks < 1:
            problems.append(f"rf_banks={self.rf_banks}")
        if self.rf_read_ports_per_bank < 1:
            problems.append(
                f"rf_read_ports_per_bank={self.rf_read_ports_per_bank}: a "
                "bank with no read port cannot be read")
        for name, depth in self.queues.items():
            if depth is None:
                problems.append(
                    f"queue {name!r} has no depth. An unbounded queue never "
                    "stalls, which is the single most flattering thing a "
                    "model can assume")
            elif depth < 1:
                problems.append(f"queue {name!r} has depth {depth}")
        for name, spec in self.units.items():
            occ, lat, count = _unit(spec)
            if occ < 1:
                problems.append(
                    f"unit {name!r} occupies {occ} cycles: an operation that "
                    "takes no time is a free operation")
            if lat < occ:
                problems.append(
                    f"unit {name!r}: latency {lat} < occupancy {occ}")
            if count < 1:
                problems.append(
                    f"unit {name!r} has {count} instances")
        if problems:
            raise ModelError('; '.join(problems))
        return self


@dataclass
class Op:
    """One operation in the compiled schedule."""
    name: str
    unit: str = 'alu'
    reads: tuple = ()          # register numbers, for bank conflicts
    writes: tuple = ()
    deps: tuple = ()           # names of ops whose RESULT this needs
    queue: str = None          # a queue it occupies until it completes


@dataclass
class Result:
    cycles: int
    issued: dict
    stalls: dict
    bottleneck: str
    trace: list
    machine: dict
    kind: str = 'model projection'   # never 'RTL measured'

    def as_dict(self):
        return dict(cycles=self.cycles, issued=self.issued,
                    stalls=self.stalls, bottleneck=self.bottleneck,
                    machine=self.machine, kind=self.kind,
                    trace=self.trace[:64])


def simulate(ops, machine=None, *, max_cycles=1_000_000, trace=False):
    """Cycle-by-cycle, with every resource finite.

    Returns a Result. Raises ModelError if the graph cannot finish, because
    "it deadlocked" and "it took zero cycles" must not look the same.
    """
    m = (machine or Machine()).validate()
    by_name = {}
    for op in ops:
        if op.name in by_name:
            raise ModelError(f"two operations named {op.name!r}")
        by_name[op.name] = op
    for op in ops:
        for d in op.deps:
            if d not in by_name:
                raise ModelError(f"{op.name!r} depends on unknown {d!r}")
        if op.unit not in m.units:
            raise ModelError(
                f"{op.name!r} uses unit {op.unit!r}, which this machine does "
                f"not have ({', '.join(sorted(m.units))})")
        if op.queue and op.queue not in m.queues:
            raise ModelError(
                f"{op.name!r} uses queue {op.queue!r}, which this machine "
                f"does not have")

    # A cycle in this graph would make it unschedulable, and the symptom
    # without this check is a deadlock reported as a cycle count.
    _check_acyclic(by_name)

    # An operation whose OWN reads exceed a bank's ports can never issue: it
    # needs two ports on one bank in one cycle and there is one. Without this
    # the scheduler span a million cycles and then reported "cannot be
    # scheduled", which is true and useless -- it does not say which
    # operation, or that the machine is at fault rather than the graph.
    #
    # A real design answers this with an operand collector that gathers reads
    # over several cycles. That is R3-04's work; until it exists, the
    # configuration is refused rather than silently assumed.
    for op in ops:
        need = {}
        for r in op.reads:
            b = r % m.rf_banks
            need[b] = need.get(b, 0) + 1
        for bank, n in sorted(need.items()):
            if n > m.rf_read_ports_per_bank:
                raise ModelError(
                    f"{op.name!r} reads {n} register(s) from bank {bank} and "
                    f"a bank has {m.rf_read_ports_per_bank} port(s): it can "
                    f"never issue. Either the machine needs more ports, or "
                    f"it needs an operand collector to gather reads over "
                    f"several cycles -- which this model does not assume it "
                    f"has")
        if len(op.writes) > m.rf_write_ports:
            raise ModelError(
                f"{op.name!r} writes {len(op.writes)} register(s) and the "
                f"machine has {m.rf_write_ports} write port(s)")

    pending = list(ops)
    done_at = {}                 # name -> cycle its result is usable
    # Per INSTANCE, so a machine with two ALUs can issue two ALU operations
    # in one cycle and a machine with one cannot.
    unit_free = {u: [0] * _unit(spec)[2] for u, spec in m.units.items()}
    queue_used = {q: [] for q in m.queues}     # completion cycles
    issued = {u: 0 for u in m.units}
    stalls = dict(issue=0, rf_bank=0, unit=0, queue=0, dependency=0)
    events = []

    cycle = 0
    while pending:
        if cycle > max_cycles:
            raise ModelError(
                f"nothing issued for {max_cycles} cycles with "
                f"{len(pending)} operation(s) left: the graph cannot be "
                f"scheduled on this machine",
                )
        # Retire queue entries whose operations have completed.
        for q in queue_used:
            queue_used[q] = [c for c in queue_used[q] if c > cycle]

        slots = m.issue_width
        bank_reads = {b: 0 for b in range(m.rf_banks)}
        writes_this_cycle = 0
        progressed = False

        for op in list(pending):
            if slots == 0:
                stalls['issue'] += 1
                break
            # A dependency that has not issued yet is not in `done_at` at
            # all, and reading it raised KeyError -- a crash where a stall
            # belongs. An unissued producer means the consumer waits.
            if any(d not in done_at for d in op.deps):
                stalls['dependency'] += 1
                continue
            ready_at = max((done_at[d] for d in op.deps), default=0)
            if cycle < ready_at:
                stalls['dependency'] += 1
                continue
            instance = next((i for i, free in enumerate(unit_free[op.unit])
                             if free <= cycle), None)
            if instance is None:
                stalls['unit'] += 1
                continue
            if op.queue and len(queue_used[op.queue]) >= m.queues[op.queue]:
                stalls['queue'] += 1
                continue
            # RF bank conflicts: each read takes a port on its bank.
            banks = [r % m.rf_banks for r in op.reads]
            conflict = False
            for b in banks:
                if bank_reads[b] + banks.count(b) > m.rf_read_ports_per_bank \
                        and bank_reads[b] >= m.rf_read_ports_per_bank:
                    conflict = True
                    break
            need = {}
            for b in banks:
                need[b] = need.get(b, 0) + 1
            for b, n in need.items():
                if bank_reads[b] + n > m.rf_read_ports_per_bank:
                    conflict = True
            if conflict:
                stalls['rf_bank'] += 1
                continue
            if op.writes and writes_this_cycle + len(op.writes) > \
                    m.rf_write_ports:
                stalls['rf_bank'] += 1
                continue

            occ, lat, _count = _unit(m.units[op.unit])
            for b, n in need.items():
                bank_reads[b] += n
            writes_this_cycle += len(op.writes)
            unit_free[op.unit][instance] = cycle + occ
            done_at[op.name] = cycle + lat
            if op.queue:
                queue_used[op.queue].append(cycle + lat)
            issued[op.unit] += 1
            pending.remove(op)
            slots -= 1
            progressed = True
            if trace:
                events.append(dict(cycle=cycle, op=op.name, unit=op.unit,
                                   done=cycle + lat))
        cycle += 1
        if not progressed and not pending:
            break

    total = max(done_at.values()) if done_at else 0
    worst = max(stalls, key=lambda k: stalls[k]) if any(stalls.values()) \
        else 'none'
    return Result(cycles=total, issued=issued, stalls=stalls,
                  bottleneck=worst, trace=events,
                  machine=dict(issue_width=m.issue_width,
                               rf_banks=m.rf_banks,
                               rf_read_ports_per_bank=m.rf_read_ports_per_bank,
                               rf_write_ports=m.rf_write_ports,
                               queues=dict(m.queues),
                               units={k: list(_unit(v))
                                      for k, v in m.units.items()}))


def _check_acyclic(by_name):
    colour = {}

    def visit(name, stack):
        state = colour.get(name)
        if state == 'done':
            return
        if state == 'open':
            raise ModelError(
                'a dependency cycle: ' + ' -> '.join(stack + [name]))
        colour[name] = 'open'
        for d in by_name[name].deps:
            visit(d, stack + [name])
        colour[name] = 'done'

    for name in by_name:
        visit(name, [])


def conservation(result, ops):
    """MODEL-03: every operation issued exactly once, nothing left over."""
    problems = []
    total_issued = sum(result.issued.values())
    if total_issued != len(ops):
        problems.append(
            f"{total_issued} operation(s) issued, {len(ops)} in the graph")
    for unit, n in result.issued.items():
        want = sum(1 for o in ops if o.unit == unit)
        if n != want:
            problems.append(f"unit {unit}: issued {n}, graph has {want}")
    return problems


def sweep(graph, candidates, *, limit=128):
    """A bounded exploration. Never generates RTL, never runs a simulator."""
    if len(candidates) > limit:
        raise ModelError(
            f"{len(candidates)} candidates exceeds the {limit} the contract "
            "allows. A sweep that large is a search, and R3 asks for a "
            "choice")
    out = []
    for name, machine in candidates:
        try:
            r = simulate(graph, machine)
            out.append(dict(candidate=name, cycles=r.cycles,
                            bottleneck=r.bottleneck, ok=True))
        except ModelError as e:
            # A rejected candidate is reported WITH ITS REASON rather than
            # dropped: "it did not appear in the results" is not an answer.
            out.append(dict(candidate=name, ok=False, reason=str(e)))
    return out
