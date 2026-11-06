"""Validate retained three-sample, BF16 RTL and default-code audit evidence."""
import hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/r40'))
from engine_overlap import selected
archive=HERE/'pipeline-final-preflight.tgz'
index={};contents={}
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if m.isfile():
            data=t.extractfile(m).read();index[m.name]=hashlib.sha256(data).hexdigest()
            if m.name.endswith(('report.json','compare-v5.json','rtl-engine.log','compiler-check-v5-retry.log')):
                contents[m.name]=data
j=lambda p:json.loads(contents[p])
audit=j('pipeline-default-audit-v5/report.json');comparison=j('pipeline-default-compare-v5.json')
assert audit['status']=='complete' and audit['summary']==dict(points=113,passed=113)
assert comparison['summary']==dict(points=113,unchanged=113,changed=0,not_compared=0)
models=[j('pipeline-fixed-model-v3/'+m+'/report.json') for m in ('baseline','single','paired')]
assert all(r['status']=='functional-pass' and r['software_tree']==audit['software_tree'] and r['timed_samples']==3 for r in models)
for report in models:
    assert len(report['rows'])==6
    for row in report['rows']:
        assert row['bitwise_reference']['equal'] and row['cold_bitwise_same']
        assert len(row['timed_samples'])==3 and all(s['output_bits_sha256']==row['reference_bits_sha256'] for s in row['timed_samples'])
rtl=j('pipeline-single-rtl-v2/report.json');assert rtl['status']=='functional-pass' and rtl['dtype']=='bfloat16'
assert len(rtl['rows'])==2 and all(r['shape']==[16,16,64] and r['bitwise_reference']['equal'] and r['single_role_plans'] for r in rtl['rows'])
sequences=[x for row in rtl['rows'] for x in row['rtl_sequences']]
traces=selected(contents['pipeline-single-rtl-v2/rtl-engine.log'].decode(),sequences)
assert len(traces)==2 and all(v['overlap_cycles']>0 for v in traces.values())
assert json.loads(json.dumps(traces))==rtl['rtl_engine_audit']
log=contents['pipeline-compiler-check-v5-retry.log'].decode()
assert 'compiler-unit          PASS' in log and 'compiler-ir            SKIP' in log
result=dict(schema='r41-p2-pipeline-preflight/1',status='passed-with-mlir-skipped',software_tree=audit['software_tree'],
    default_audit=comparison['summary'],model_rows=sum(len(x['rows']) for x in models),model_timed_samples=sum(len(r['timed_samples']) for x in models for r in x['rows']),
    rtl_bf16_rows=len(rtl['rows']),rtl_bf16_trace=traces,compiler='make compiler passed; 53 Python tests and the tnca-opt MLIR suite skipped',
    limitations=['The model uses an uncalibrated timing domain.','The retained RTL trace was built from an earlier source identity.','Physical U280 results and final identity reproduction remain pending.'])
(HERE/'pipeline-preflight-summary.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'pipeline-preflight-index.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=index),indent=2)+'\n')
print(json.dumps(dict(status=result['status'],audit=result['default_audit'],model_samples=result['model_timed_samples'],rtl_bf16_rows=result['rtl_bf16_rows'])))
