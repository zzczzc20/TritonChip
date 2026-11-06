"""Later fragments retain only proven alignment and same-block deferral."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tnca_cc.frontend import TTIRImporter
from tnca_cc.target import TargetInfo

class Value:
    def __init__(self,key):self.key=key
    def id(self):return self.key
    def get_type(self):return 'tensor<16xi32>'
class Op:
    def __init__(self,name='tt.make_range',start=16,key='chunk'):
        self.name,self.start,self.key=name,start,key
    def get_name(self):return self.name
    def get_int_attr(self,name):return self.start if name=='start' else self.start+16
    def get_result(self,i):return Value(self.key)
    def get_operand(self,i):return Value(self.key)
    def get_block(self):return SimpleNamespace(id=lambda:'block')

class IndexProofs(unittest.TestCase):
    def make(self,start=16,nonnegative=True):
        imp=TTIRImporter(TargetInfo.default(),scratch_addr=0x3000,index_opt=True)
        op=Op(start=start);op._register_chunk=True
        imp._op_tt_make_range(op)
        imp._offsets['uniform']=('uniform','pid32')
        imp._v['pid32']=imp.b.program_id(0)
        imp._multiple['pid32']=32
        if nonnegative:imp._nonneg.add('pid32')
        imp._note_affine('sum',Value('uniform'),Value('chunk'),'add')
        return imp
    def test_later_fragment_remainder_remains_affine(self):
        imp=self.make()
        self.assertEqual(imp._multiple['sum'],16)
        self.assertIn('sum',imp._nonneg)
        self.assertTrue(imp._contiguous_div(Op(key='sum'),True,128))
    def test_unproven_sign_retains_general_remainder(self):
        self.assertFalse(self.make(nonnegative=False)._contiguous_div(Op(key='sum'),True,128))
    def test_unaligned_fragment_retains_general_remainder(self):
        self.assertFalse(self.make(start=7)._contiguous_div(Op(key='sum'),True,128))
    def test_nonmultiple_extent_retains_wrapping_general_path(self):
        self.assertFalse(self.make()._contiguous_div(Op(key='sum'),True,127))
    def test_aligned_identity_for_many_origins(self):
        for extent in (16,32,48,128,256):
            for origin in range(0,1024,16):
                self.assertEqual([(origin+lane)%extent for lane in range(16)],
                                 [origin%extent+lane for lane in range(16)])
    def deferral(self,middle=(),users=1):
        imp=TTIRImporter(TargetInfo.default(),scratch_addr=0x3000,index_opt=True)
        source,user=Op('tt.load',key='source'),Op('tt.dot',key='user')
        imp._value_users={'source':[(user,0)]*users}
        imp._all_ops=[source,*middle,user]
        imp._op_pos={id(op):i for i,op in enumerate(imp._all_ops)}
        return imp._deferrable(SimpleNamespace(op=source))
    def test_ordinary_register_chunk_defers_without_large_tile_mode(self):
        self.assertTrue(self.deferral())
    def test_intervening_aliasing_store_prevents_deferral(self):
        self.assertFalse(self.deferral((Op('tt.store'),)))
    def test_multiple_consumers_prevent_deferral(self):
        self.assertFalse(self.deferral(users=2))
if __name__=='__main__':unittest.main()
