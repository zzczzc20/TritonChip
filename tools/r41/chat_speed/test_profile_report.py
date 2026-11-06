#!/usr/bin/env python3
"""Profile report selects actual decode scopes and excludes skipped calls."""
import json
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/models'))
import decode_profile_report as report

class ProfileReportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  out=self.root/'out';(out/'profile').mkdir(parents=True)
  self.report=dict(model='test',decoder_layers=1,decode_weight_bytes=1,steps=[
   dict(step=0,kind='prefill',transfers_inside_step={'device_cycles':1}),
   dict(step=1,kind='decode',transfers_inside_step={'device_cycles':2})])
  (out/'report.json').write_text(json.dumps(self.report))
 def row(self,index,step,kind=None,skipped=False):
  r=dict(index=index,step=step,kernel='test',grid=[1,1,1],args=[],counters={'core_0_instructions':2},
   device_cycles=7,code_load=True,code_bytes=64,evicted=0)
  if kind is not None:r.update(scope_kind=kind,skipped=skipped)
  return r
 def save(self,rows):
  (self.root/'out/profile/launches.json').write_text(json.dumps({'launches':rows}))
 def test_v2_excludes_warmup_prefill_and_skipped(self):
  self.save([self.row(0,0,'warmup'),self.row(1,1,'prefill'),self.row(2,2,'decode',True),self.row(3,2,'decode')])
  got=report.analyse(self.root,1);self.assertEqual(got['step'],2);self.assertEqual(got['report_step']['kind'],'decode');self.assertEqual([r['index'] for r in got['rows']],[3])
  pre=report.analyse(self.root,1,step=1);self.assertEqual(pre['report_step']['kind'],'prefill')
 def test_legacy_profile_keeps_step_mapping(self):
  self.save([self.row(0,0),self.row(1,1)]);got=report.analyse(self.root,1)
  self.assertEqual(got['step'],1);self.assertEqual(got['report_step']['kind'],'decode')
 def test_missing_counters_are_not_zero(self):
  r=self.row(0,2,'decode');r['counters']=None;self.save([r])
  with self.assertRaises(SystemExit):report.analyse(self.root,1)
if __name__=='__main__':unittest.main()
