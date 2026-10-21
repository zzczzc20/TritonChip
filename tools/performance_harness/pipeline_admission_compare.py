"""Compare native ring cases without promoting model rankings to board wins."""
import argparse
import json
from pathlib import Path


def key(row):
    return row['graph'],tuple(row['shape'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',action='append',required=True,help='label=report.json')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    reports={}
    for value in args.case:
        label,path=value.split('=',1)
        if label in reports:raise ValueError('duplicate pipeline case label')
        reports[label]=json.loads(Path(path).read_text())
    expected={'model-baseline',*(f'model-stage{s}' for s in (1,2,3)),*(f'rtl-stage{s}' for s in (1,2,3))}
    if set(reports)!=expected or any(r['status']!='functional-pass' for r in reports.values()):
        raise ValueError('all registered cases and the model baseline must complete numerically')
    rows={label:{key(row):row for row in report['rows']} for label,report in reports.items()}
    if any(not rows[label] or len(rows[label])!=len(report['rows']) for label,report in reports.items()):
        raise ValueError('empty or duplicate pipeline workload population')
    baseline=rows['model-baseline'];result=[]
    for label,group in rows.items():
        if set(group)!=set(baseline):
            raise ValueError('candidate workload population changed')
        for point,row in group.items():
            base=baseline[point]
            if [i['sha256'] for i in row['inputs']] != [i['sha256'] for i in base['inputs']]:
                raise ValueError('candidate changed actual input values')
            stage=row['ring_stages']
            expected_stage=None if label=='model-baseline' else int(label[-1])
            if stage!=expected_stage:raise ValueError('pipeline label disagrees with executed ring stage')
            peer=rows.get(f'model-stage{stage}',{}).get(point) if label.startswith('rtl-') else None
            code=[a['code_sha256'] for a in row['artifacts']]
            peer_code=[a['code_sha256'] for a in peer['artifacts']] if peer else None
            # Different memory/front-end domains preclude a model/RTL
            # makespan accuracy claim even when the same shader is executed.
            ledger={k:sum(c.get(k,0) for c in row['model_counter_deltas'].values()) for k in
                    sorted({k for c in row['model_counter_deltas'].values() for k in c})}
            selected={k:v for k,v in ledger.items() if any(s in k for s in
                ('dma_admission','dma_commands','matrix_engine','engine_service','engine_queue','outlet_'))}
            trace=reports[label].get('rtl_engine_audit',{})
            audits=[trace[str(seq)] for seq in row['rtl_sequences']] if trace else []
            result.append(dict(label=label,graph=point[0],shape=list(point[1]),stage=stage,
                device_cycles=row['device_cycles'],reference_status=row['status'],
                model_timing_status=row.get('model_timing_status'),
                model_cycle_ratio_to_baseline=(row['device_cycles']/base['device_cycles'] if label.startswith('model-') else None),
                actual_input_values_equal=True,bindings_equal_to_baseline=(row['input_placements']==base['input_placements'] and
                     row['output_placement']==base['output_placement']),
                artifact_code_sha256=code,model_peer_code_sha256=peer_code,
                same_code_as_model_peer=(code==peer_code if peer else None),
                model_accumulated_counter_ledger=selected,
                peak_counter_scope='use raw before/after high-water marks in case reports; peak deltas are not current occupancy',
                rtl_engine_audit=audits,rtl_overlap_cycles=sum(a['overlap_cycles'] for a in audits)))
    report=dict(schema='tnca-pipeline-admission-study/1',status='screening-and-RTL-diagnostic-complete',rows=result,
        model_rank_qualification='uncalibrated; accumulated service/queue counters are not exclusive fractions of makespan',
        cross_domain_scope='RTL checks shared code and actual engine service; memory profiles differ from the board model',
        board_closure='not-measured',decision='retain legal stages; Mode A remains default-off pending physical evaluation')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],rows=len(result),board_closure=report['board_closure']),indent=2))


if __name__=='__main__':
    main()
