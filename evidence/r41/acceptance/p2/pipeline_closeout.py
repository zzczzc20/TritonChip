"""Validate native model/RTL evidence; no fabricated timing or board claim."""
import hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/r40'))
from engine_overlap import selected
archive=HERE/'pipeline-model-rtl.tgz'
files={}
with tarfile.open(archive) as t:
    for member in t.getmembers():
        if member.isfile():files[member.name]=t.extractfile(member).read()
def j(path):return json.loads(files[path])
def sha(data):return hashlib.sha256(data).hexdigest()
models={mode:j(f'pipeline-fixed-model-v2/{mode}/report.json') for mode in ('baseline','single','paired')}
assert len({r['software_tree'] for r in models.values()})==1
identities=[r['model_identities'][0] for r in models.values()]
assert len({x['binary_sha256'] for x in identities})==1
assert len({x['config_sha256'] for x in identities})==1
expected={(graph,(16,16,k)) for graph in ('mm','addmm') for k in (32,48,64)}
rows={}
for mode,report in models.items():
    assert report['status']=='functional-pass' and report['backend']=='systemc-tlm-grid' and report['bitwise']
    knobs=report['knobs'];assert knobs['TNCA_ACTIVE_CORES']=='3' and knobs['TNCA_RCTX_SPMD_SHARED']=='2'
    assert int(identities[0]['config']['nx'])==3
    indexed={(r['graph'],tuple(r['shape'])):r for r in report['rows']}
    assert set(indexed)==expected and len(report['rows'])==len(expected)
    for (graph,shape),row in indexed.items():
        assert row['status']=='pass' and row['bitwise_reference']['equal'] and row['cold_bitwise_same']
        assert row['output_bits_sha256']==row['reference_bits_sha256']
        for entry in row['inputs']:
            payload=files[f"pipeline-fixed-model-v2/{mode}/"+entry['logical_contiguous_bytes']]
            assert sha(payload)==entry['sha256']
        outname=f"pipeline-fixed-model-v2/{mode}/artifacts/{graph}-{'-'.join(map(str,shape))}-out.bin"
        assert sha(files[outname])==row['output_bits_sha256']
        assert len(row['artifacts'])==1 and row['device_cycles']>0
        art=row['artifacts'][0];execution=art['execution'];schedule=art['scheduling']
        assert execution['role_mask']==3 and all(x['grid']==[1,1,1] and x['upload_bytes']==x['download_bytes']==0 for x in row['launches'])
        assert art['plan']['gemm']['static_mma_commands']['match'] is True
        assert art['plan']['resources']['vrf_limit']==64
        if mode=='paired':
            assert schedule['rctx_roles']==2 and schedule['rctx_roles_pairs']==1 and schedule['rctx_roles_ring']['stages']==2
            assert len(execution['program_to_role']['0'])==2
        else:
            assert schedule['rctx_spmd']==2 and len(execution['program_to_role'])==2
            assert all(len(v)==1 for v in execution['program_to_role'].values())
            assert bool(schedule.get('single_role_pipeline'))==(mode=='single')
        rows.setdefault((graph,shape),{})[mode]=row
comparison=[]
for (graph,shape),arms in sorted(rows.items()):
    assert len({tuple(x['sha256'] for x in r['inputs']) for r in arms.values()})==1
    assert len({r['output_bits_sha256'] for r in arms.values()})==1
    cycles={m:r['device_cycles'] for m,r in arms.items()}
    comparison.append(dict(graph=graph,shape=shape,cycles=cycles,
        single_speedup=cycles['baseline']/cycles['single'],paired_speedup=cycles['baseline']/cycles['paired'],
        resources={m:r['artifacts'][0]['plan']['resources'] for m,r in arms.items()},
        role_operations={m:[x['static_op_counts'] for x in r['artifacts'][0]['plan']['roles']] for m,r in arms.items()}))
rtl=j('pipeline-single-rtl-v1/report.json')
assert rtl['status']=='functional-pass' and rtl['backend']=='sim-file' and len(rtl['rows'])==4
sequences=[]
for row in rtl['rows']:
    assert row['status']=='pass' and row['bitwise_reference']['equal'] and row['cold_bitwise_same'] and row['single_role_plans']
    assert row['output_bits_sha256']==row['reference_bits_sha256'];sequences+=row['rtl_sequences']
trace=files['pipeline-single-rtl-v1/rtl-engine.log'].decode()
audit=selected(trace,sequences)
assert len(audit)==4
assert all(v['overlap_cycles']>0 for v in audit.values())
assert json.loads(json.dumps(audit))==rtl['rtl_engine_audit']
result=dict(schema='r41-p2-pipeline-evidence/1',status='model-and-rtl-pass',
    software_tree=models['baseline']['software_tree'],model_identity=identities[0],
    total_model_rows=sum(len(x['rows']) for x in models.values()),comparison=comparison,
    rtl_software_tree=rtl['software_tree'],rtl_rows=len(rtl['rows']),rtl_engine_audit=audit,
    limitations=['SystemC timing is uncalibrated and does not establish board speedup.',
                 'RTL trace uses an earlier software tree and one core; the final-image board comparison remains pending.',
                 'The same three-core image and two-context mask are configured for all arms; these grid-one cases exercise one program, not three-core scaling. Mode B uses one 64-register context per program; the paired roles share one 64-register base through RPAIR.'])
(HERE/'pipeline-summary.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'pipeline-index.json').write_text(json.dumps(dict(archive_sha256=sha(archive.read_bytes()),members={k:sha(v) for k,v in files.items()}),indent=2)+'\n')
print(json.dumps(dict(status=result['status'],model_rows=result['total_model_rows'],rtl_rows=result['rtl_rows'])))
