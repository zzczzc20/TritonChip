"""Recheck archived service evidence without compiling kernels or accessing devices."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'evidence/r41/stage1/kickoff/memory-service-v1-evidence.tgz'
COMPARATOR = ROOT / 'tools/performance_harness/memory_service_compare.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    checks = []
    with tarfile.open(ARCHIVE) as archive:
        reports = {}
        for member in archive:
            if not member.name.endswith('/report.json'):
                continue
            report = json.load(archive.extractfile(member))
            reports[member.name.removesuffix('/report.json')] = report
            if report['status'] != 'functional-pass':
                continue
            for item in report['inputs']:
                path = str(Path(member.name).parent / item['path'])
                assert sha(archive.extractfile(path).read()) == item['sha256'], path
        failed = {name: r['status'] for name, r in reports.items()
                  if r['status'] != 'functional-pass'}
    rtl = {
        'rtl-p1-d1': reports['memory-service-rtl-p1-d1-v3'],
        'rtl-p1-d8': reports['memory-service-rtl-p1-d8-v1'],
        'rtl-p3-d8': reports['memory-service-rtl-diagnostic-v1'],
    }
    cohorts = {
        'coherent': ['memory-service-p1-d1-diagnostic-v1',
                     'memory-service-p1-d8-diagnostic-v1', 'memory-service-diagnostic-v1'],
        'maintenance': ['host-handoff-diagnostic-v1/maintenance-p1-d1',
                        'host-handoff-diagnostic-v1/maintenance-p1-d8',
                        'host-handoff-diagnostic-v1/maintenance'],
        'small-cache': ['service-cache-geometry-v1/p1-d1',
                        'service-cache-geometry-v1/p1-d8', 'service-cache-geometry-v1/p3-d8'],
    }
    with tempfile.TemporaryDirectory(prefix='r41-service-review-') as directory:
        temp = Path(directory)

        def compare(name, cases, should_pass, extra=()):
            command = [sys.executable, str(COMPARATOR), '--out', str(temp / 'out.json')]
            for i, (label, report) in enumerate(cases.items()):
                path = temp / f'case-{i}.json'
                path.write_text(json.dumps(report))
                command += ['--case', f'{label}={path}']
            result = subprocess.run(command + list(extra), capture_output=True, text=True)
            checks.append(dict(name=name, expected='accept' if should_pass else 'reject',
                               returncode=result.returncode,
                               result=result.stdout.strip() if result.returncode == 0 else
                               result.stderr.strip().splitlines()[-1]))
            assert (result.returncode == 0) == should_pass, checks[-1]
            if should_pass:
                (HERE / f'{name}.json').write_text((temp / 'out.json').read_text())

        for name, paths in cohorts.items():
            cases = dict(rtl)
            cases.update({label: reports[path] for label, path in
                          zip(('model-p1-d1', 'model-p1-d8', 'model-p3-d8'), paths)})
            compare(name, cases, True)
        baseline = copy.deepcopy(cases)
        missing = copy.deepcopy(baseline)
        del missing['rtl-p3-d8']
        compare('missing-case', missing, False)
        for name, field, value in (
            ('failed-numerics', 'status', 'failed-preserved'),
            ('mixed-source', 'software_tree', 'files:wrong-source'),
            ('zero-cycles', 'device_cycles', 0),
            ('wrong-backend', 'backend', 'systemc-tlm-grid'),
        ):
            altered = copy.deepcopy(baseline)
            altered['rtl-p1-d8'][field] = value
            compare(name, altered, False)
        altered = copy.deepcopy(baseline)
        altered['rtl-p1-d8']['inputs'][0]['sha256'] = 'wrong-input'
        compare('changed-input', altered, False)
        altered = copy.deepcopy(baseline)
        altered['rtl-p1-d8']['artifacts'][0]['code_sha256'] = 'wrong-shader'
        compare('changed-shader', altered, False)
        altered = copy.deepcopy(baseline)
        altered['model-p1-d8']['model_identity']['host_handoff_policy'] = 'coherent-direct'
        compare('mixed-host-policy', altered, False)
        compare('invalid-domain', baseline, False, ('--domains', 'model,rtl,unknown'))
    summary = dict(schema='r41-memory-service-archive-review/1',
                   scope='offline revalidation of historical reports; no new execution measurement',
                   archive_sha256=sha(ARCHIVE.read_bytes()),
                   comparator_sha256=sha(COMPARATOR.read_bytes()),
                   case_sha256=sha((COMPARATOR.parent / 'memory_service_case.py').read_bytes()),
                   successful_reports=sum(r['status'] == 'functional-pass' for r in reports.values()),
                   archived_input_hashes='verified for every successful report',
                   preserved_failures=failed, checks=checks,
                   board_status='not-measured; lead owns the board queue',
                   final_identity_status='historical cohorts do not bind the frozen Stage 1 product')
    (HERE / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(checks)} comparator checks passed; {len(failed)} historical failures preserved')


if __name__ == '__main__':
    main()
