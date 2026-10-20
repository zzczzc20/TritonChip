#!/usr/bin/env python3
"""Audit a native SystemC run and expose falsifiable performance hypotheses.

This is architecture screening, not physical timing sign-off. Queue residence
sums are pressure indicators across concurrent requests, not critical-path
fractions. No fit, gate promotion, synthesis or FPGA operation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    report=json.loads(path.read_text())
    if (report['status']!='completed-prediction' or not report['inputs_unchanged']
            or report['identity']['backend']!='systemc-tlm-prediction'):
        raise ValueError('a completed, immutable SystemC prediction is required')
    for name,expected in report['output_sha256'].items():
        if Path(name).name!=name or digest(path.parent/name)!=expected:
            raise ValueError(f'output changed: {name}')
    summary=json.loads((path.parent/'summary.json').read_text())
    if summary!=report['summary']:
        raise ValueError('summary does not match sealed report')
    sockets=json.loads((path.parent/'sockets.json').read_text())
    if not sockets['complete'] or sockets['mode']!=report['identity']['monitor']:
        raise ValueError('incomplete or mismatched socket monitor')
    return report,sockets


def inspect(path):
    report,sockets=load(path)
    cycles=report['summary']['visible_cycles']
    boundaries=[]
    for name,s in sockets['sockets'].items():
        if s['active'] or s['errors'] or sum(s['latency_log2_histogram'])!=s['calls']:
            raise ValueError('inconsistent socket accounting')
        boundaries.append(dict(socket=name,transactions=s['calls'],bytes=s['bytes'],
                               mean_residence_cycles=s['latency_sum_cycles']/s['calls'] if s['calls'] else None,
                               max_residence_cycles=s['latency_max_cycles'],peak_concurrency=s['peak'],
                               busy_fraction=s['busy_cycles']/cycles if cycles else None,
                               scope='home includes cache/downstream; HBM includes outlet/PC/ack; excludes upstream NoC'))
    counters=report['summary']['counters']
    pressure=[]
    for name,value in counters.items():
        suffix=next((s for s in ('_queue_cycles','_wait_cycles') if name.endswith(s)),None)
        if suffix and value:
            resource=name[:-len(suffix)]
            pressure.append(dict(resource=resource,kind=suffix[1:],aggregate_request_cycles=value,
                                 service_cycles=counters.get(resource+'_service_cycles'),
                                 hypothesis='contention contributes to request residence; vary this resource alone and preserve useful work'))
    pressure.sort(key=lambda row:row['aggregate_request_cycles'],reverse=True)
    return dict(schema='tnca-systemc-diagnosis/1',status='audited-prediction',identity=report['identity'],
                visible_cycles=cycles,host_elapsed_seconds=report['host_elapsed_seconds'],
                socket_boundaries=boundaries,resource_pressure=pressure,
                interpretation='nested boundaries overlap; queue sums may exceed elapsed time; neither identifies critical path alone',
                next_experiment='change one capacity or mapping; rerun identical native code/data/reference and examine predicted cycles, queue and traffic deltas',
                model_qualification='partial-ISA, timing hypotheses; no physical-routing prediction',
                prediction_error_bound_cycles=None,rtl_speedup=None,performance_gate='not-measured',
                input_report_sha256=digest(path),analysis_sha256=digest(Path(__file__)))


def compare(reference,candidate):
    a,b=reference['identity'],candidate['identity']
    # Paths and node names may differ, contents and execution semantics may not.
    for field in ('binary_sha256','mode','trace_sha256','sources','systemc_library','contract_sha256'):
        if a[field]!=b[field]:raise ValueError(f'incompatible comparison: {field}')
    if sorted(a['dependencies_sha256'].values())!=sorted(b['dependencies_sha256'].values()):
        raise ValueError('different native code/data/reference dependencies')
    return dict(reference_cycles=reference['visible_cycles'],candidate_cycles=candidate['visible_cycles'],
                predicted_relative_change=(candidate['visible_cycles']/reference['visible_cycles']-1)
                    if reference['visible_cycles'] else None,
                decision='unqualified sensitivity result; require independent calibration and functional equivalence before spending implementation budget',
                confidence_bound=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=inspect(a.report)
    if a.candidate:
        candidate=inspect(a.candidate)
        result['comparison']=compare(result,candidate)
        result['candidate']=candidate
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(f"audited-prediction: {len(result['resource_pressure'])} resource pressure indicators; performance gate not-measured")


if __name__=='__main__':main()
