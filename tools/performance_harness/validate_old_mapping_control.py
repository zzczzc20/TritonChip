"""Validate the isolated legacy-mapping control before any board measurement."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_old_mapping_control import FROZEN_TREE, REGISTER_BLOCKS, REGISTER_BLOCKS_SHA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(row,suffix):
    return sum(v for counters in row['model_counter_deltas'].values()
               for name,v in counters.items() if name.endswith(suffix))


def validate(proof):
    prep=json.loads((proof/'preparation.json').read_text())
    if (prep['status']!='isolated-control-prepared' or prep['frozen_software_tree']!=FROZEN_TREE or
        prep['changed_product_files']!=[REGISTER_BLOCKS] or prep['source_file_sha256']!=REGISTER_BLOCKS_SHA or
        prep['control_file_sha256']!='0e81a1cb06742046c2aad20ff99e32e693cfbbb1207fa0d9bfce54c5363521a5'):
        raise ValueError('unexpected source ablation; the original single-change control is required')
    reports={};inputs={};identities=[]
    for variant in ('old','fixed'):
        for label in ('16','32','32-k33'):
            path=proof/'cases'/f'{variant}-{label}'/'report.json'
            report=json.loads(path.read_text());inputs[str(path.relative_to(proof))]=sha(path)
            expected_tree=prep['control_software_tree'] if variant=='old' else FROZEN_TREE
            if report['status']!='functional-pass' or report['software_tree']!=expected_tree:
                raise ValueError('numerical completion and correct source identity required')
            if report['columns']!=(16 if label=='16' else 32) or report['dtype']!='float16':
                raise ValueError('registered tile width and input dtype required')
            if report['backend']!='systemc-tlm-grid':
                raise ValueError('registered native-model binding required')
            models=report['model_identities']
            models=list(models.values()) if isinstance(models,dict) else models
            if not models:raise ValueError('missing actual native model identity')
            identities.extend(tuple(m[k] for k in ('binary_sha256','config_sha256','dram_path')) for m in models)
            rows={row['graph']:row for row in report['rows']}
            if set(rows)!={'mm','addmm'} or len(report['rows'])!=2:
                raise ValueError('exact mm/addmm cohort required')
            expected_k=33 if label.endswith('k33') else 65
            if any(row['status']!='pass' or row['shape']!=[17,31,expected_k] for row in rows.values()):
                raise ValueError('registered shape and per-graph numerical pass required')
            reports[variant,label]=rows
    if len(set(identities))!=1:
        raise ValueError('control and repaired cases changed the actual model binary/config/path')
    result=[]
    for label in ('16','32','32-k33'):
        for graph in ('mm','addmm'):
            old,fixed=[reports[v,label][graph] for v in ('old','fixed')]
            if (old['shape'],old['inputs'],old['rtol'],old['atol'])!=(fixed['shape'],fixed['inputs'],fixed['rtol'],fixed['atol']):
                raise ValueError('actual input values, shape or numerical acceptance changed')
            codes=lambda row:[a['code_sha256'] for a in row['artifacts']]
            if count(old,'_mma_commands')!=count(fixed,'_mma_commands'):
                raise ValueError('control changed the useful MMA population')
            trips=lambda row:{k:sum(a['plan']['accumulator_round_trips'][k] for a in row['artifacts'])
                              for k in ('k_loop_reads','k_loop_writes','k_loop_zeros')}
            old_trips,fixed_trips=trips(old),trips(fixed)
            if label=='16':
                if codes(old)!=codes(fixed):raise ValueError('narrow control executable changed')
            elif (codes(old)==codes(fixed) or not all(old_trips.values()) or any(fixed_trips.values()) or
                  count(old,'_class_a_row')<=count(fixed,'_class_a_row')):
                raise ValueError('legacy K-loop traffic was not reproduced and removed')
            result.append(dict(label=label,graph=graph,shape=old['shape'],inputs_equal=True,
                executable_equal=codes(old)==codes(fixed),old_code_sha256=codes(old),fixed_code_sha256=codes(fixed),
                old_k_loop_acc_operations=old_trips,fixed_k_loop_acc_operations=fixed_trips,
                old_row_commands=count(old,'_class_a_row'),fixed_row_commands=count(fixed,'_class_a_row'),
                old_instructions=count(old,'_instructions'),fixed_instructions=count(fixed,'_instructions'),
                old_cycles=old['device_cycles'],fixed_cycles=fixed['device_cycles'],
                bindings_equal=(old['input_placements']==fixed['input_placements'] and
                                old['output_placement']==fixed['output_placement'])))
    for graph in ('mm','addmm'):
        short,long=[next(r for r in result if r['graph']==graph and r['label']==label) for label in ('32-k33','32')]
        if short['fixed_row_commands']!=long['fixed_row_commands'] or short['old_row_commands']>=long['old_row_commands']:
            raise ValueError('K-scaling check did not isolate invariant epilogue traffic')
    return dict(schema='tnca-old-mapping-control-validation/1',status='old-mapping-control-validated',
        preparation_sha256=sha(proof/'preparation.json'),input_report_sha256=inputs,
        validator_sha256=sha(Path(__file__)),rows=result,
        model_identity=dict(zip(('binary_sha256','config_sha256','dram_path'),identities[0])),
        scope='native model numerical and structural validation; no board performance or Stage 1 acceptance claim')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proof',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();report=validate(args.proof)
    with args.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],rows=len(report['rows'])),indent=2))


if __name__=='__main__':main()
