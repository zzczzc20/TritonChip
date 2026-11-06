"""Real importer IR: prove ordering/guard and reject unsafe bodies."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler/python'),str(ROOT/'compiler/test')]
import triton_bridge
from tnca_cc.frontend import TTIRImporter
from tnca_cc.target import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.single_role_pipeline import pipeline_loop,NotPipelined
if triton_bridge.available():
    from test_fragment_residency import fragment_loop

@unittest.skipUnless(triton_bridge.available(),'Triton is required')
class SingleRolePipeline(unittest.TestCase):
    def kernel(self,k=64,old_use=False,columns=16):
        sig={x:'*fp16' for x in ('A','B')};sig.update(C='*fp32',K='constexpr',N='constexpr',OLD_USE='constexpr')
        module,context=triton_bridge.make_ttir(fragment_loop,sig,dict(K=k,N=columns,OLD_USE=old_use))
        target=TargetInfo.load('tnca-sim-vrf-v5');imp=TTIRImporter(target,'fragment_loop',index_opt=True,scratch_addr=0x80000)
        kernel,_=imp.run(module,'fragment_loop',dict(A=0x40000,B=0x44000,C=0x48000),param_names=['A','B','C'])
        return kernel,target
    def test_two_slots_overlap_and_drain_odd_trip_before_epilogue(self):
        kernel,target=self.kernel(k=48);info=pipeline_loop(kernel);ops=kernel.roles[0].region.ops
        self.assertEqual(info['roles'],1);self.assertEqual(info['stages'],2);self.assertEqual(len(kernel.roles),1)
        mmas=[(i,o) for i,o in enumerate(ops) if o.kind=='mma_async'];self.assertEqual(len(mmas),2)
        first,second=mmas;token=first[1].operands[3]
        waits=[i for i,o in enumerate(ops) if o.kind=='barrier_wait' and o.operands[0] is token]
        self.assertEqual(len(waits),2)
        self.assertTrue(any(o.kind=='dma_load' for o in ops[first[0]+1:waits[0]]))
        self.assertLess(waits[0],second[0])
        label=next(i for i,o in enumerate(ops) if o.kind=='label' and o.attrs['name'].startswith('single_role_pipeline_drain'))
        self.assertGreater(waits[1],label)
        self.assertTrue(any(o.kind=='branch_nz' and o.attrs.get('target')==ops[label].attrs['name'] for o in ops[first[0]+1:second[0]]))
        for checking in (False,True):
            from copy import deepcopy
            trial=deepcopy(kernel)
            cg=CodeGen(trial,target,check_protocol=checking,window_reuse=False);cg.assign_shared();cg.run()
    def test_old_accumulator_reader_refuses_without_mutation(self):
        kernel,_=self.kernel(old_use=True);before=str(kernel);windows=list(kernel.windows)
        with self.assertRaises(NotPipelined):pipeline_loop(kernel)
        self.assertEqual(str(kernel),before);self.assertEqual(kernel.windows,windows)
    def test_multiple_mma_refuses_without_mutation(self):
        kernel,_=self.kernel(columns=32);before=str(kernel)
        with self.assertRaises(NotPipelined):pipeline_loop(kernel)
        self.assertEqual(str(kernel),before)
    def test_non_two_stage_refuses_without_mutation(self):
        kernel,_=self.kernel();before=str(kernel)
        for stages in (1,3):
            with self.assertRaises(NotPipelined):pipeline_loop(kernel,stages=stages)
            self.assertEqual(str(kernel),before)
if __name__=='__main__':unittest.main()
