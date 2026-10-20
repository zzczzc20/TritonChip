#!/usr/bin/env python3
"""IR semantic oracle for bounded role chunks, not hardware acceptance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compiler/python'))
from tnca_cc.ir import Kernel,Role,Region,Value,Op,SCALAR
from tnca_cc.rctx_spmd import apply


def kernel():
    x,y,z,n= [Value(name,SCALAR,'i32') for name in ('x','y','z','n')]
    base=Value('base',SCALAR,'i64')
    # The invariant store models descriptor construction: a tail role must
    # exit before executing it, even with hoisting enabled.
    ops=[Op('special',[base],[],{'name':'arg_base'}),Op('sstore',[],[base],{}),
         Op('program_id',[x],[],{'dim':0}),Op('program_id',[y],[],{'dim':1}),
         Op('program_id',[z],[],{'dim':2}),Op('num_programs',[n],[],{'dim':0}),
         Op('trace',[],[x,y,z,n]),Op('exit')]
    return Kernel('chunk',roles=[Role(0,'role0',Region(ops))])


def execute(k,pid,npid,role):
    ops=k.roles[0].region.ops
    labels={op.attrs['name']:i for i,op in enumerate(ops) if op.kind=='label'}
    values={}; traces=[]; stores=0; pc=0
    for _ in range(10000):
        op=ops[pc];pc+=1
        args=[values[id(v)] for v in op.operands]
        kind=op.kind;result=None
        if kind=='special':result=role if op.attrs['name']=='role_id' else 0x10000
        elif kind=='program_id':result=pid[op.attrs['dim']]
        elif kind=='num_programs':result=npid[op.attrs['dim']]
        elif kind=='const':result=op.attrs['value']
        elif kind=='sbin':result=args[0]+args[1]
        elif kind=='sbini':
            result=args[0]>>op.attrs['imm'] if op.attrs['op']=='lshr' else args[0]+op.attrs['imm']
        elif kind=='scmp':result=int(args[0]>=args[1]) if op.attrs['cc']=='ge' else int(args[0]<args[1])
        elif kind=='assign':values[id(op.operands[0])]=args[1]
        elif kind=='branch_nz':
            if args[0]:pc=labels[op.attrs['target']]
        elif kind=='sstore':stores+=1
        elif kind=='trace':traces.append(tuple(args))
        elif kind=='exit':return traces,stores
        elif kind!='label':raise AssertionError(kind)
        if result is not None:values[id(op.results[0])]=result
    raise AssertionError('unterminated role')


def main():
    cases=0
    for width,pair in ((1,False),(2,False),(4,False),(1,True),(2,True)):
        for hoist in (False,True):
            for gx,gy,gz in ((1,1,1),(5,2,3),(17,3,2),(6145,1,1)):
                k=kernel();apply(k,width,pair=pair,hoist=hoist,physical_chunk=True)
                seen=[]
                for z in range(gz):
                    for y in range(gy):
                        for start in range(0,gx,width):
                            for role in range(width*(2 if pair else 1)):
                                traces,stores=execute(k,(start,y,z),(gx,gy,gz),role)
                                wanted=start+(role//2 if pair else role)
                                assert traces==([(wanted,y,z,gx)] if wanted<gx else []),(width,pair,start,role,traces)
                                assert stores==int(wanted<gx),(start,role,stores)
                                if not pair or role%2==0:seen.extend(traces)
                assert sorted(seen)==sorted((x,y,z,gx) for z in range(gz) for y in range(gy) for x in range(gx))
                cases+=1
    legacy=kernel();apply(legacy,4)
    assert execute(legacy,(0,0,0),(17,1,1),1)[0]==[(x,0,0,17) for x in (1,5,9,13)]
    invalid=kernel();before=str(invalid)
    try:apply(invalid,3,physical_chunk=True)
    except ValueError:pass
    else:raise AssertionError('invalid chunk width accepted')
    assert str(invalid)==before
    print(f'PHYSICAL CHUNK IR PASS {cases} cases; legacy row preserved; invalid width refused')
if __name__=='__main__':main()
