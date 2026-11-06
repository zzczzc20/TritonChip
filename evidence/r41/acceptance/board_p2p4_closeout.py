#!/usr/bin/env python3
"""Validate the lead-run P2/P4 raw archive and derive its evidence index."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'evidence/r41/acceptance'
ARCHIVE = BASE / 'board-p2p4-7b10e7b-try2.tgz'
PREFIX = 'r41_accept_p2_p4_7b10e7b_try2'
REMOTE_ROOT = '/home/zzhouch/r41_accept_7b10e7b/'
OUT_ROOT = '/home/zzhouch/r41_accept_p2_p4_7b10e7b_try2/'
COMMIT = '7b10e7b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{COMMIT}:{path}'])


def main():
    with tarfile.open(ARCHIVE) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    assert files[f'{PREFIX}/DONE'] == b'commands completed; per-point acceptance still required\n'
    results = {}
    common_tree = None
    for experiment, expected_steps in (
        ('s2-p2-locality', ['locality']),
        ('s2-p4-lifecycle', ['aliases', 'eviction', 'barrier', 'grid']),
    ):
        report = json.loads(files[f'{PREFIX}/{experiment}/report.json'])
        spec = json.loads(blob(f'spec/r41/stage2/acceptance/{experiment}.json'))
        assert report['status'] == 'completed-experiment'
        assert report['identity']['manifest_sha256'] == sha(blob(f'spec/r41/stage2/acceptance/{experiment}.json'))
        assert [s['id'] for s in report['steps']] == expected_steps == [s['id'] for s in spec['steps']]
        for name, expected in report['identity']['sources'].items():
            assert name.startswith(REMOTE_ROOT)
            assert sha(blob(name[len(REMOTE_ROOT):])) == expected, name
        common_tree = common_tree or report['identity']['software_tree']
        assert report['identity']['software_tree'] == common_tree
        for step in report['steps']:
            assert step['returncode'] == 0 and step['status'] == 'completed'
            board = step['board']
            assert board['image_uuid'] == spec['board']['image_uuid']
            assert board['image_sha256'] == spec['board']['image_sha256']
            assert board['clock_hz'] == spec['board']['clock_hz'] == 100000000
            for probe in (board['probe_before'], board['probe_after']):
                assert probe['loaded_uuid'] == board['image_uuid']
                assert probe['clock_counter_mhz'][0] == board['clock_hz'] // 1000000
                assert probe['contexts'] == probe['outstanding_execs'] == 0
                assert not probe['client_pid']
            for absolute, expected in list(step['outputs'].items()) + list(step['logs'].items()):
                assert absolute.startswith(OUT_ROOT)
                assert sha(files[PREFIX + absolute[len(OUT_ROOT) - 1:]]) == expected, absolute
        results[experiment] = report

    locality = json.loads(files[f'{PREFIX}/s2-p2-locality/locality/report.json'])
    assert locality['status'] == 'functional-pass' and locality['backend'] == 'fpga-u280-multicore'
    assert locality['software_tree'] == common_tree
    locality_rows = []
    for graph in locality['graphs']:
        assert graph['status'] == 'functional-pass'
        assert [r['locality'] for r in graph['runs']] == [False, True, False]
        assert len({r['output_sha256'] for r in graph['runs']}) == 1
        capture_artifacts = [e['artifact'] for e in graph['capture']]
        for run in graph['runs']:
            assert run['status'] == 'pass'
            assert [e['artifact'] for e in run['launches']] == capture_artifacts
            assert all(e['backend'] == 'fpga-u280-multicore' and e['replay'] for e in run['launches'])
            assert all(e['upload_bytes'] == e['download_bytes'] == 0 for e in run['launches'])
        controls = graph['runs'][1]['controls']
        assert controls['status'] == 'completed' and not controls['plan']['aggressive_hints']
        assert controls['registered']['policy'] == 1 and controls['restored']['policy'] == 0
        assert graph['fault']['status'] == 'pass' and graph['fault']['off_replay_refused']
        assert graph['fault']['cleanup']['restored']['policy'] == 0
        rows = []
        for run in graph['runs']:
            traffic = [dict(port=before['port'],
                            read_lines=after['read_lines'] - before['read_lines'],
                            write_lines=after['write_lines'] - before['write_lines'])
                       for before, after in zip(run['counters_before'], run['counters_after'])]
            rows.append(dict(locality=run['locality'], host_scope_seconds=run['host_scope_seconds'],
                             device_cycles=[e['device_cycles'] for e in run['launches']],
                             memory_port_line_changes=traffic))
        locality_rows.append(dict(graph=graph['name'], output_sha256=graph['runs'][0]['output_sha256'], runs=rows))

    aliases = json.loads(files[f'{PREFIX}/s2-p4-lifecycle/aliases/report.json'])
    assert aliases['status'] == 'pass' and aliases['software_tree'] == common_tree
    assert {r['name'] for r in aliases['checks']} == {
        'undeclared-overlapping-storage-arguments', 'out-of-bounds-view',
        'forged-high-bit-storage-owner', 'distinct-storage-handles-overlap'}
    assert all(r['status'] == 'refused-before-admission' for r in aliases['checks'])

    eviction_step = next(s for s in results['s2-p4-lifecycle']['steps'] if s['id'] == 'eviction')
    eviction_log_name = next(n for n in files if '/steps/eviction/' in n and n.endswith('/stdout.log'))
    eviction_text = files[eviction_log_name].decode()
    match = re.search(r'PASS\s+eviction\s+(\d+) graphs; (\d+) module\(s\) unloaded', eviction_text)
    assert match and int(match.group(2)) > 0 and 'resident data intact' in eviction_text
    assert 'r40-device-tests  PASS  (1/1)' in eviction_text
    eviction = dict(graphs=int(match.group(1)), modules_unloaded=int(match.group(2)),
                    host_wall_seconds=eviction_step['host_wall_seconds'])

    barrier = json.loads(files[f'{PREFIX}/s2-p4-lifecycle/barrier/report.json'])
    assert barrier['status'] == 'pass' and barrier['event']['ok']
    assert barrier['observed']['completed_arm_wait_iterations'] == barrier['requested_arms'] > 65536

    grid = json.loads(files[f'{PREFIX}/s2-p4-lifecycle/grid/report.json'])
    assert grid['status'] == 'pass' and all(c['status'] == 'pass' for c in grid['checks'])
    checks = {c['name']: c for c in grid['checks']}
    assert checks['exact-grid-and-drain']['unique_program_ids'] == grid['requested_programs']
    assert checks['exact-grid-and-drain']['completion_atomics'] == grid['requested_programs'] + len(grid['events']) - 1
    assert checks['bounded-queue-overflow']['admitted_dependent'] == len(grid['events']) - 1
    assert all(e['state'] == 'SUCCEEDED' for e in grid['events'])

    summary = {
        'schema': 'r41-p2p4-board-closeout/1',
        'archive': str(ARCHIVE.relative_to(ROOT)),
        'archive_sha256': sha(ARCHIVE.read_bytes()),
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', COMMIT], text=True).strip(),
        'software_tree': common_tree,
        'interpreter': results['s2-p2-locality']['identity']['python'],
        'board': {k: results['s2-p2-locality']['steps'][0]['board'][k]
                  for k in ('image_uuid', 'image_sha256', 'clock_hz')},
        'step_wall_seconds': {e['experiment'] + '/' + s['id']: s['host_wall_seconds']
                              for e in results.values() for s in e['steps']},
        'locality': locality_rows,
        'aliases': aliases['checks'],
        'eviction': eviction,
        'barrier': barrier,
        'grid': grid,
        'judgments': {
            'S2-03.A.5': 'passed: physical off/on/off traffic and latency are supplied; no performance gain is claimed',
            'S2-04.D.3': 'passed: compound high-bit, bounds and illegal-alias refusals occur before admission',
            'S2-04.D.4': 'passed: opt-in, bounded queue, exact completion/drain and advertised-capacity refusal',
            'S2-04.D.6': 'passed: more than 65,536 physical barrier arm/wait iterations completed',
            'S2-04.D.7': 'not measured: targeted eviction passes, but a persistent full-model session is still required',
            'S2-03.B': 'not measured: this queue did not execute the separate pipeline manifest',
        },
    }
    (BASE / 'board-p2p4-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    index = {'schema': 'r41-p2p4-board-index/1', 'archive_sha256': summary['archive_sha256'],
             'members': {name: {'size': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}}
    (BASE / 'board-p2p4-index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'status': 'validated', 'judgments': summary['judgments'],
                      'step_wall_seconds': summary['step_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
