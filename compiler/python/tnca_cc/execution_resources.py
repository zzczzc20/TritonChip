"""Static execution resource facts shared by compilation and model analysis.

This describes the emitted program. It is neither a numerical oracle nor a
cycle predictor. Dynamic masks, loop counts and overlap still require native
execution; a static command count must not be presented as executed traffic.
"""
from collections import Counter


def describe(kernel):
    roles = []
    for role in kernel.roles:
        ops = role.region.ops
        labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
        loops = []
        for end, op in enumerate(ops):
            start = labels.get(op.attrs.get('target'))
            if op.kind in ('branch', 'branch_nz') and start is not None and start < end:
                body = Counter(item.kind for item in ops[start:end + 1])
                loops.append(dict(label=op.attrs['target'], start=start, end=end,
                                  static_mma_commands=body['mma_async'],
                                  static_acc_reads=body['acc_read'],
                                  static_acc_writes=body['acc_write'],
                                  static_acc_zeros=body['acc_zero']))
        counts = Counter(op.kind for op in ops)
        shared = set(kernel.windows + kernel.accumulators)
        uses = {v: [] for v in shared}
        dependencies = []
        for i, op in enumerate(ops):
            for v in op.operands + op.results:
                if v in uses:
                    uses[v].append(i)
            if op.kind in ('mma_async', 'dma_load', 'dma_store',
                           'barrier_arm', 'barrier_wait', 'event_wait'):
                dependencies.append(dict(position=i, kind=op.kind,
                                         tokens=[v.name for v in op.operands + op.results
                                                 if v.kind in ('token', 'event', 'barrier')]))
        roles.append(dict(role=role.index, static_op_counts=dict(sorted(counts.items())),
                          loops=loops, command_dependencies=dependencies,
                          shared_use_intervals=[dict(name=v.name, kind=v.kind,
                                                     first=min(sites), last=max(sites))
                                                for v, sites in uses.items() if sites]))
    return dict(schema='tnca-execution-resources/1', scope='static emitted IR',
                dynamic_counts='require native execution',
                valid_and_padding_macs='require logical tile and runtime valid masks',
                overlap='not inferred from command admission', roles=roles)
