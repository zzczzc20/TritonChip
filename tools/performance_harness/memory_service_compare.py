"""Retain service/depth observations without assuming more outlets win."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',action='append',required=True,help='label=report.json')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--domains',default='model,rtl',
                    help='comma list of cohorts: model, rtl, board (physical fpga-u280-multicore)')
    args=ap.parse_args()
    domains=args.domains.split(',')
    if not domains or set(domains)-{'model','rtl','board'} or len(set(domains))!=len(domains):
        raise ValueError('domains are a unique subset of model, rtl, board')
    cases={}
    for value in args.case:
        label,path=value.split('=',1)
        if label in cases:raise ValueError('duplicate service case label')
        cases[label]=json.loads(Path(path).read_text())
    reference=next(iter(cases.values()))
    expected={(d,p,o) for d in domains for p,o in ((1,1),(1,8),(3,8))}
    actual={(label.split('-',1)[0],r['ports'],r['outstanding']) for label,r in cases.items()}
    if actual!=expected or len(cases)!=len(expected):raise ValueError('complete unique declared outlet/depth cohort required')
    model_policies={r['model_identity'].get('host_handoff_policy','coherent-direct')
                    for label,r in cases.items() if label.startswith('model-')}
    if 'model' in domains and len(model_policies)!=1:raise ValueError('candidate cohort mixes model host-ownership policies')
    result=[]
    for label,r in cases.items():
        if r['status']!='functional-pass':raise ValueError('service case did not pass numerically')
        if r['software_tree']!=reference['software_tree']:raise ValueError('service cohort mixes product source identities')
        if type(r['device_cycles']) is not int or r['device_cycles']<=0:raise ValueError('positive completed device cycles required')
        if r['elements']!=reference['elements'] or [i['sha256'] for i in r['inputs']]!=[i['sha256'] for i in reference['inputs']]:
            raise ValueError('service candidate changed workload or actual inputs')
        domain=label.split('-',1)[0]
        if domain=='model' and not r['backend'].startswith('systemc-tlm'):
            raise ValueError('model label does not identify a model backend')
        if domain=='rtl' and r['backend']!='fpga-multicore-sim':
            raise ValueError('RTL service cohort must use the actual MMIO stand-in')
        if domain=='board' and r['backend']!='fpga-u280-multicore':
            raise ValueError('board service cohort must use the physical multicore image')
        baseline=next(v for name,v in cases.items() if name.startswith(domain+'-') and (v['ports'],v['outstanding'])==(1,1))
        code=[a['code_sha256'] for a in r['artifacts']]
        if code!=[a['code_sha256'] for a in baseline['artifacts']]:raise ValueError('controls unexpectedly changed emitted shader')
        delta=r.get('counter_delta',{})
        ledger={k:v for k,v in delta.items() if any(s in k for s in ('home_','outlet_','lsu_','dma_','link_'))}
        peer=cases.get(f"model-{label.split('-',1)[1]}") if domain=='board' else None
        result.append(dict(label=label,ports=r['ports'],outstanding=r['outstanding'],
            model_minus_board_over_board=((peer['device_cycles']-r['device_cycles'])/r['device_cycles']
                                          if peer else None),
            device_cycles=r['device_cycles'],cycle_ratio_to_domain_baseline=r['device_cycles']/baseline['device_cycles'],
            actual_input_values_equal=True,same_shader_as_domain_baseline=True,shader_code_sha256=code,
            bindings_equal_to_domain_baseline=(r['placements']==baseline['placements'] and r['output_placement']==baseline['output_placement']),
            actual_outlet_line_deltas=r['outlets'],model_accumulated_service_ledger=ledger,
            model_timing_status=r.get('model_timing_status'),
            host_dma_handoffs=delta.get('host_dma_handoffs'),
            high_water_scope='raw before/after snapshots retain peak counters; peak deltas are not occupancy'))
    report=dict(schema='tnca-memory-service-study/1',domains=domains,
        status=('board-cohort-recorded' if 'board' in domains else 'screening-and-RTL-diagnostic-complete'),rows=result,
        scope='resident standard Torch graph after warm call; no explicit inter-call CPU download; native ABI maintenance remains included',
        cycle_metric='completed launch cycles only; prelaunch maintenance outside launch windows; raw service deltas surround the full native call',
        model_host_handoff_policy=next(iter(model_policies)) if model_policies else None,
        cache_boundary=('explicit native DMA maintenance is included; physical cache/controller timing equivalence is unproven'
                        if model_policies=={'dma-maintenance'} else
                        'physical DMA ownership handoffs and coherent model host access differ; cache warmth is not matched'),
        model_scope='uncalibrated; accumulated resource service/queue cycles may exceed elapsed makespan',
        board_closure=('board cohort recorded; a model residual needs model cases on the same product source'
                       if 'board' in domains and 'model' not in domains else
                       'model and board cases compared on one product source; the model remains uncalibrated'
                       if 'board' in domains else 'not-measured'),
        decision='retain existing product defaults; diagnostic outlet/depth baselines do not select a physical policy')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],rows=len(result),board_closure=report['board_closure']),indent=2))


if __name__=='__main__':main()
