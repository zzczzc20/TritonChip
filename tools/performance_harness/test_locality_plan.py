"""Metadata safety tests using actual Session ownership and admission resolver.

No numerical execution or performance evidence is claimed by these unit tests.
"""
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm')]
from tnca import aot
from tnca.runtime import Session,Buffer
from tnca.tensor import TensorDescriptor
from tnca.locality import plan_commands,replay_with_locality
from tnca.errors import TncaError

class LocalityOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.paths={}
        for alias in (False,True):
            cls.paths[alias]=aot.compile_aot(lambda b,t:b.exit(),kernel_name='metadata_only',
                signature={'a':'*i8','b':'*i8'},source_kind='tnca-ir',options={'alias':[['a','b']]} if alias else {},
                output=str(Path(cls.tmp.name)/f'{alias}.tnca')).path
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        self.s=Session(profile='core-sim',backend='sim-file');self.addCleanup(self.s.close)
        self.module={k:self.s.load(v) for k,v in self.paths.items()}
    def command(self,a,b,alias=True,epoch=None):
        return (self.s,self.module[alias],{'a':a,'b':b},(1,1,1),(),(),self.s.epoch if epoch is None else epoch,{})
    def test_views_coalesce_and_incomplete_accesses_never_get_aggressive_hints(self):
        a,b=self.s.allocate(128),self.s.allocate(64)
        plan=plan_commands([self.command(a,b),self.command(a,(a,64))]);report=plan.report()
        self.assertEqual(len(plan.ranges),2)
        first=next(r for r in plan.ranges if r.buffer_id==a.id)
        self.assertEqual((first.first_command,first.last_command),(0,1));self.assertTrue(first.aliases)
        self.assertFalse(report['aggressive_hints']);self.assertEqual(plan.policy,1)
        self.assertTrue(all(r.nacc==0xffffffff and not r.bypass and not r.complete_readers and not r.complete_writers for r in plan.ranges))
    def test_undeclared_overlap_refuses_before_admission(self):
        a=self.s.allocate(64);before=self.s.latest_event
        with self.assertRaises(TncaError):plan_commands([self.command(a,a,alias=False)])
        self.assertIs(self.s.latest_event,before)
    def test_descriptor_tracks_underlying_storage(self):
        a=self.s.allocate(128);desc=TensorDescriptor((a,64),(8,),(1,),(8,),'i8')
        plan=plan_commands([self.command(desc,a)])
        self.assertEqual(len(plan.ranges),1);self.assertTrue(plan.ranges[0].aliases)
        self.assertEqual(plan.ranges[0].address,a.addr)
    def test_forged_same_session_handle_is_not_storage_ownership(self):
        a=self.s.allocate(64);forged=Buffer(a.id,a.addr,a.size,self.s.id,self.s.epoch)
        with self.assertRaises(TncaError):plan_commands([self.command(forged,a)])
    def test_freed_or_stale_capture_is_refused(self):
        a,b=self.s.allocate(64),self.s.allocate(64);command=self.command(a,b)
        self.s.free(a)
        with self.assertRaises(TncaError):plan_commands([command])
        with self.assertRaises(TncaError):plan_commands([self.command(b,b,epoch=self.s.epoch+1)])
    def test_unsupported_backend_refuses_before_any_control_admission(self):
        from types import SimpleNamespace
        a=self.s.allocate(64);graph=SimpleNamespace(commands=[self.command(a,a)])
        before=self.s.latest_event
        with self.assertRaises(TncaError):replay_with_locality(graph,lambda:self.fail('numerical execution admitted'))
        self.assertIs(self.s.latest_event,before)
    def test_capacity_refuses_entire_plan(self):
        buffers=[self.s.allocate(64) for _ in range(9)]
        with self.assertRaises(TncaError):plan_commands([self.command(b,b) for b in buffers])
        self.assertIsNone(self.s.latest_event)

if __name__=='__main__':unittest.main()
