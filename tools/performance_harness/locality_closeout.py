#!/usr/bin/env python3
"""Validate a real locality_case report and derive its comparison ledger."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report['schema'] == 'tnca-r41-locality-case/1'
    assert report['status'] == 'functional-pass' and len(report['graphs']) == 2
    rows = []
    for graph in report['graphs']:
        assert graph['status'] == 'functional-pass'
        assert [run['locality'] for run in graph['runs']] == [False, True, False]
        hashes = {run['output_sha256'] for run in graph['runs']}
        assert len(hashes) == 1
        artifacts = [entry['artifact'] for entry in graph['capture']]
        for run in graph['runs']:
            assert run['status'] == 'pass'
            assert [entry['artifact'] for entry in run['launches']] == artifacts
            assert all(entry['replay'] and entry['upload_bytes'] == 0 and
                       entry['download_bytes'] == 0 for entry in run['launches'])
            before = {entry['port']:entry for entry in run['counters_before']}
            after = {entry['port']:entry for entry in run['counters_after']}
            assert before.keys() == after.keys()
            traffic = {key:sum(after[port][key]-before[port][key] for port in before)
                       for key in ('read_lines', 'write_lines')}
            assert all(value >= 0 for value in traffic.values())
            row = dict(graph=graph['name'], locality=run['locality'],
                       output_sha256=run['output_sha256'],
                       device_cycles=sum(entry['device_cycles'] for entry in run['launches']),
                       host_scope_seconds=run['host_scope_seconds'], traffic=traffic)
            if run['locality']:
                controls = run['controls']
                assert controls['status'] == 'completed'
                assert controls['registered']['policy'] == 1 and controls['restored']['policy'] == 0
                assert not controls['plan']['aggressive_hints']
                assert all(not r['bypass'] and not r['complete_readers'] and
                           not r['complete_writers'] for r in controls['plan']['ranges'])
                row['control_events'] = len(controls['control_events'])
            rows.append(row)
        fault = graph['fault']
        assert fault['status'] == 'pass' and fault['off_replay_refused']
        assert fault['cleanup']['status'] == 'failed-preserved'
        assert fault['cleanup']['restored']['policy'] == 0
    result = dict(schema='tnca-r41-locality-closeout/1', status='functional-pass',
                  source_report=str(args.report), source_software_tree=report['software_tree'],
                  backend=report['backend'], rows=rows,
                  fault_checks=[dict(graph=g['name'], status=g['fault']['status'],
                                     restored_policy=g['fault']['cleanup']['restored']['policy'],
                                     off_replay_refused=g['fault']['off_replay_refused'])
                                for g in report['graphs']],
                  interpretation='RTL functional observations; not physical-board or performance acceptance')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], rows=len(rows))))


if __name__ == '__main__':
    main()
