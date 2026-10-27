"""Verify archived P1 observations; never promote simulation into board evidence."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
members={}
def read(archive):
    result={}
    with tarfile.open(ROOT/archive) as t:
        for m in t.getmembers():
            if not m.isfile():continue
            data=t.extractfile(m).read()
            members[archive+':'+m.name]=hashlib.sha256(data).hexdigest()
            if m.name.endswith('.json'):
                try:result[m.name]=json.loads(data)
                except (ValueError,UnicodeDecodeError):pass
    return result
fixed=read('fixed-model-and-audit.tgz');final=read('identities-and-final-audit.tgz');rtl=read('rtl-sensitivity.tgz');read('model-evidence.tgz')
def cases(data,prefix,count):
    reports={k:v for k,v in data.items() if k.startswith(prefix) and k.endswith('/report.json')}
    assert len(reports)==count,(prefix,len(reports))
    for name,d in reports.items():
        assert d['status']=='functional-pass' and d['bitwise'],name
        assert all(r['status']=='pass' for r in d['rows']),name
    return reports
model=cases(fixed,'fixed-v2/',8);native=cases(rtl,'rtl-fixed-v3/',4);sensitivity=cases(rtl,'sensitivity/',8)
for key in ['profile-compare.json']:
    assert fixed[key]['summary']==dict(points=113,unchanged=113,changed=0,not_compared=0)
assert final['frontend-default-compare.json']['summary']==dict(points=113,unchanged=113,changed=0,not_compared=0)
service_transfer=read('service-transfer.tgz')
service={k:v for k,v in service_transfer.items() if k.startswith('service-current/') and k.endswith('/report.json')}
assert len(service)==3 and all(v['status']=='functional-pass' for v in service.values())
queue=cases(service_transfer,'transfer-queue/',2)
for a,b in zip(queue['transfer-queue/queue-1/report.json']['rows'],queue['transfer-queue/queue-2/report.json']['rows']):
    assert a['output_f32_sha256']==b['output_f32_sha256']
pairs=[]
for name,d in model.items():
    if d['columns']!=32:continue
    other=model[name.replace('-32-','-16-')]
    for wide,narrow in zip(d['rows'],other['rows']):
        assert wide['shape']==narrow['shape'] and wide['graph']==narrow['graph']
        pairs.append(dict(dtype=d['dtype'],strided=d['strided'],graph=wide['graph'],shape=wide['shape'],narrow_cycles=narrow['device_cycles'],wide_cycles=wide['device_cycles'],wide_over_narrow=wide['device_cycles']/narrow['device_cycles']))
result=dict(schema='r41-p1-observations/1',model_numeric_rows=sum(len(d['rows']) for d in model.values()),rtl_numeric_rows=sum(len(d['rows']) for d in native.values()),sensitivity_numeric_rows=sum(len(d['rows']) for d in sensitivity.values()),default_audit=dict(points=113,changed=0),pairs=pairs,sensitivity={k:[dict(graph=r['graph'],shape=r['shape'],cycles=r['device_cycles']) for r in d['rows']] for k,d in sensitivity.items()},limitations=['Uncalibrated SystemC sensitivity parameters are hypotheses, not hardware ablations or board results. Some perturbations are nonmonotonic; no causal throughput attribution is inferred.','RTL regression covers four small dtype/layout configurations, not the full model or original suite.','Prior board failures remain failures until matched remeasurement; tail wide fragments remain slower in the model.'])
result['memory_service']={k:dict(cycles=v['device_cycles'],outlets=v['outlets'],model_identity=v.get('model_identity'),counter_delta=v.get('counter_delta')) for k,v in service.items()}
result['transfer_queue']={k:[dict(graph=r['graph'],shape=r['shape'],cycles=r['device_cycles']) for r in v['rows']] for k,v in queue.items()}
(ROOT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'index.json').write_text(json.dumps(dict(archives={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.tgz')},members=members),indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['model_numeric_rows','rtl_numeric_rows','sensitivity_numeric_rows','default_audit']}))
