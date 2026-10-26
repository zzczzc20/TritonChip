"""SPMD map hoisting must publish dynamic fields before engine snapshots."""
import unittest
from tnca_cc.frontend import TTIRImporter
from tnca_cc.target import TargetInfo
from tnca_cc.rctx_spmd import apply


def execute(ops, *, initial=None, programs=3):
    """Two independent requesters: the last LSU store may still be pending.

    A following LSU store drains its predecessor; an engine snapshot alone
    does not. This exposes stale fields without relying on zeroed memory.
    """
    values={};memory=dict(initial or {});pending=[];snapshots=[]
    labels={op.attrs['name']:i for i,op in enumerate(ops) if op.kind=='label'}
    pc=0
    for _ in range(1000):
        op=ops[pc];pc+=1;k=op.kind
        args=[values[id(v)] for v in op.operands]
        result=None
        if k=='const':result=op.attrs['value']
        elif k=='special':result=0
        elif k=='num_programs':result=programs
        elif k=='label':continue
        elif k in ('sbin','sbini'):
            a=args[0];b=args[1] if k=='sbin' else op.attrs['imm']
            result={'add':lambda:a+b,'mul':lambda:a*b,'shl':lambda:a<<b,'lshr':lambda:a>>b}[op.attrs['op']]()
        elif k=='scmp':result=int(args[0]<args[1])
        elif k=='assign':values[id(op.operands[0])]=args[1]
        elif k=='branch_nz':
            if args[0]:pc=labels[op.attrs['target']]
        elif k=='sstore':
            memory.update(pending);pending=[(args[0]+op.attrs['offset'],args[1])]
        elif k=='fence':memory.update(pending);pending=[]
        elif k=='desc_snapshot':snapshots.append(memory.get(args[0],0));result=0
        elif k=='exit':return snapshots
        else:raise AssertionError(k)
        if op.results:values[id(op.results[0])]=result
    raise AssertionError('SPMD loop did not terminate')


class DescriptorPublicationTests(unittest.TestCase):
    def program(self,dynamic):
        imp=TTIRImporter(TargetInfo.default());b=imp.b
        addr=b.const(0x20200);pid=b.program_id(0)
        base=b.sbini('add',b.sbini('shl',pid,6),0x20600) if dynamic else b.const(0x20600)
        b.sstore(addr,base)
        b.sstore(addr,b.const(17),offset=8)
        b.descriptor(addr);b.exit()
        return imp.kernel

    def test_dynamic_map_base_is_published_in_each_program(self):
        kernel=self.program(True);apply(kernel,1)
        ops=kernel.role(0).region.ops
        self.assertEqual(execute(ops,initial={0x20200:0x1234}),[0x20600,0x20640,0x20680])
        # Removing the publication must expose the stale base, so a test cannot
        # accidentally pass merely because unused simulator RAM began at zero.
        self.assertNotEqual(execute([op for op in ops if op.kind!='fence'],initial={0x20200:0x1234}),
                            [0x20600,0x20640,0x20680])

    def test_invariant_map_is_published_before_hoisted_snapshot(self):
        kernel=self.program(False);apply(kernel,1)
        self.assertEqual(execute(kernel.role(0).region.ops),[0x20600])

    def test_readonly_host_descriptor_needs_no_new_publication(self):
        imp=TTIRImporter(TargetInfo.default());b=imp.b
        addr=b.const(0x20200);b.program_id(0);b.descriptor(addr);b.exit()
        apply(imp.kernel,1)
        ops=imp.kernel.role(0).region.ops
        self.assertFalse(any(op.kind=='fence' for op in ops))
        self.assertEqual(execute(ops,initial={0x20200:0x20600}),[0x20600])


if __name__=='__main__':unittest.main()
