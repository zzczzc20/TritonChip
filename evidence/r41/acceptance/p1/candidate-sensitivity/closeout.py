"""Derive one-factor uncertainty results from actual archived native runs."""
import hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
archive=HERE/'runs-v2.tgz';reports={};hashes={}
with tarfile.open(archive) as t:
 for member in t.getmembers():
  if not member.isfile():continue
  data=t.extractfile(member).read();hashes[member.name]=hashlib.sha256(data).hexdigest()
  if member.name.endswith('/case/report.json'):reports[member.name.split('/')[1]]=json.loads(data)
  if member.name.endswith('/registration.json'):registration=json.loads(data)
assert set(reports)==set(registration['variants'])
base=reports['base'];rows={}
for tag,d in reports.items():
 assert d['status']=='functional-pass' and d['bitwise'] and len(d['rows'])==4,tag
 assert d['software_tree']==base['software_tree']
 assert len(d['model_identities'])==1 and len(base['model_identities'])==1
 identity=d['model_identities'][0];base_identity=base['model_identities'][0]
 assert identity['binary_sha256']==base_identity['binary_sha256']
 cfg={k:int(v) for k,v in identity['config'].items()};basecfg={k:int(v) for k,v in base_identity['config'].items()}
 assert cfg['core_pipeline']==1
 delta={k:cfg[k]-basecfg[k] for k in cfg if cfg[k]!=basecfg[k]}
 assert delta==registration['variants'][tag],(tag,delta)
 entries=[]
 for actual,control in zip(d['rows'],base['rows']):
  assert actual['status']=='pass' and actual['shape']==control['shape'] and actual['graph']==control['graph']
  assert actual['output_f32_sha256']==control['output_f32_sha256']
  assert [v['code_sha256'] for v in actual['artifacts']]==[v['code_sha256'] for v in control['artifacts']]
  entries.append(dict(graph=actual['graph'],shape=actual['shape'],base_cycles=control['device_cycles'],cycles=actual['device_cycles'],ratio=actual['device_cycles']/control['device_cycles']))
 rows[tag]=dict(parameter_deltas=registration['variants'][tag],observations=entries,interpretation='uncalibrated one-factor uncertainty; not an RTL candidate counterfactual or gain')
# Verify archived X8 comparisons separately: they actually exercise byte transpose.
root=HERE.parents[4]
x8=root/'evidence/r41/stage2/candidates/X8/x8-evidence.tgz'
with tarfile.open(x8) as t:
 def result(config):
  d=json.load(t.extractfile('./whatif/'+config+'-results.json'))
  assert d['backend']=='systemc-tlm-grid'
  assert all(r['status']=='pass' and not r.get('problems') and not r.get('fallbacks') for r in d['points'])
  return {r['point']:r for r in d['points']}
 old=result('core-vrf-sync-v5-g1-pf0');new=result('core-vrf-sync-v5-x8-g1-pf0')
 assert set(old)==set(new)
 x8rows=[dict(point=k,baseline_cycles=old[k]['device_cycles'],x8_cycles=new[k]['device_cycles'],speedup=old[k]['device_cycles']/new[k]['device_cycles']) for k in sorted(old)]
summary=dict(schema='r41-p1-candidate-sensitivity/1',numeric_rows=sum(len(d['rows']) for d in reports.values()),rows=rows,x8=x8rows,coverage='F01-F10 and T2 one-factor timing uncertainty; X8 archived capability comparison',limitations=['No proxy is a quantitative physical candidate attribution. Cycle-preserving retiming requires physical frequency/area evidence.','Small standard Torch GEMMs do not establish workload-general gain. Zero or mixed response is retained.','All model results remain uncalibrated; board remeasurement is pending.'])
(HERE/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
(HERE/'index.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=hashes,x8_archive_sha256=hashlib.sha256(x8.read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(dict(numeric_rows=summary['numeric_rows'],variants=len(reports),x8_points=len(x8rows))))
