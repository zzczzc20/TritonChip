"""Bounded one-role double-buffer pipeline for eligible transfer/MMA loops.

The transformation is opt-in. It issues the next iteration's transfers while
an earlier MMA is outstanding, with independent RF windows and a wait before
reusing the accumulator. Odd trip counts drain the outstanding MMA on their
own exit path. Unsupported or ambiguous bodies are refused without mutation.
"""
from . import partition
from .ir import Op,Value


class NotPipelined(Exception):
    """A proven-safe schedule was not found; retain the original loop."""


def pipeline_loop(kernel, *, stages=2):
    if stages!=2:raise NotPipelined('single-role software pipeline requires exactly two stages')
    if len(kernel.roles)!=1:raise NotPipelined('single-role pipeline requires one logical role')
    ops=kernel.roles[0].region.ops
    try:loop=partition.find_ring_loop(ops)
    except partition.NotSpecialised as error:raise NotPipelined(str(error)) from error
    body=ops[loop.test+1:loop.latch]
    mma=[op for op in body if op.kind=='mma_async']
    if not loop.direct or len(mma)!=1 or sum(op.kind=='mma_async' for op in ops)!=1:
        raise NotPipelined('requires one MMA and no loop-carried memory writes')
    forbidden={'dma_store','store','scatter','sstore','fence','atomic','atomic_cas',
               'acc_read','acc_write','acc_zero','desc_snapshot','trap','assert_eq','assert_lane'}
    if any(op.kind in forbidden for op in body):raise NotPipelined('unsafe side effect in loop body')
    windows={id(w) for w in loop.windows}
    if not windows or any(id(op.operands[0]) not in windows for op in body if op.kind=='dma_load'):
        raise NotPipelined('all loop transfers must fill the MMA source windows')
    token=mma[0].operands[3]
    uses=[op for op in body if any(v is token for v in partition._operands(op))]
    if len(uses)!=2 or uses[0] is not mma[0] or uses[1].kind!='barrier_wait':
        raise NotPipelined('MMA completion must have one unambiguous wait')
    if not ops or ops[-1].kind!='exit':raise NotPipelined('kernel needs a terminal exit for the drain block')
    if any(op.kind in ('branch','branch_nz','label','exit') for op in body):
        raise NotPipelined('pipeline body must have straight-line control')
    old_windows=list(kernel.windows);old_barriers=list(kernel.barriers)
    try:
        unrolled,ring=partition._unroll(kernel,ops,loop,2)
        stage0=next(op for op in unrolled if op.kind=='mma_async' and op.attrs.get('_stage')==0)
        stage1=next(op for op in unrolled if op.kind=='mma_async' and op.attrs.get('_stage')==1)
        waits=[op for op in unrolled if op.kind=='barrier_wait' and op.attrs.get('_stage')==0 and op.operands[0] is stage0.operands[3]]
        if len(waits)!=1:raise NotPipelined('unrolled MMA wait is ambiguous')
        wait0=waits[0];first1=next(i for i,op in enumerate(unrolled) if op.attrs.get('_stage')==1)
        test1=next((op for op in reversed(unrolled[:first1]) if op.kind=='branch_nz'),None)
        if test1 is None or test1.attrs.get('target')!=ops[loop.test].attrs['target']:
            raise NotPipelined('second iteration has no matching exit guard')
        # Keep the pending MMA's windows live while filling the other slot.
        arm1=next(op for op in unrolled if op.kind=='barrier_arm' and op.results[0] is stage1.operands[3])
        unrolled.remove(wait0);unrolled.insert(unrolled.index(arm1),wait0)
        existing={op.attrs['name'] for op in unrolled if op.kind=='label'}
        label='single_role_pipeline_drain';suffix=0
        while label in existing:suffix+=1;label=f'single_role_pipeline_drain_{suffix}'
        exit_target=test1.attrs['target'];test1.attrs['target']=label
        one=Value(name='single_role_pipeline_one',kind='scalar',dtype='i64')
        end=next(i for i,op in enumerate(unrolled) if op.kind=='label' and op.attrs.get('name')==exit_target)
        # A non-taken back-edge skips the odd-trip drain; the second-iteration
        # exit guard enters it. Both targets remain forward branches.
        unrolled[end:end]=[Op(kind='const',results=[one],operands=[],attrs={'value':1}),
                          Op(kind='branch_nz',results=[],operands=[one],attrs={'target':exit_target}),
                          Op(kind='label',results=[],operands=[],attrs={'name':label}),
                          Op(kind='barrier_wait',results=[],operands=[stage0.operands[3]],attrs={})]
        for op in unrolled:op.attrs.pop('_stage',None)
        kernel.roles[0].region.ops=unrolled
    except BaseException:
        kernel.windows[:]=old_windows;kernel.barriers[:]=old_barriers
        raise
    return dict(stages=2,roles=1,ring_windows=sum(len(x) for x in ring),
                drain='conditional odd-trip completion before original epilogue')
