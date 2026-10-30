#!/usr/bin/env python3
"""Run a registered sequence of board experiments and publish its result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(root, out, python, registration):
    root, out, python = Path(root).resolve(), Path(out).resolve(), Path(python).resolve()
    rows = registration.get('experiments')
    if not isinstance(rows, list) or not rows:
        raise ValueError('a stage must register at least one experiment')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or any(not x.replace('-', '').replace('_', '').isalnum() for x in ids):
        raise ValueError('experiment IDs must be unique simple names')
    summary = out / 'stage.json'
    if summary.exists():
        raise ValueError('stage result already exists')
    for row in rows:
        manifest = root / row['manifest']
        if not manifest.is_file() or sha(manifest) != row['sha256']:
            raise ValueError('registered manifest missing or changed: ' + str(manifest))
        if (out / row['id'] / 'report.json').exists():
            raise ValueError('experiment result already exists: ' + row['id'])
        if not row.get('checks'):
            raise ValueError('each experiment needs explicit result checks')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        target = out / row['id']
        if sha(root / row['manifest']) != row['sha256']:
            raise ValueError('registered manifest changed before execution: ' + row['id'])
        command = [str(python), str(root / 'tools/performance_harness/experiment.py'),
                   str(root / row['manifest']), '--root', str(root), '--out', str(target),
                   '--python', str(python)]
        with (out / (row['id'] + '.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha(root / row['manifest']) != row['sha256']:
            raise ValueError('registered manifest changed during execution: ' + row['id'])
        report = target / 'report.json'
        if json.loads(report.read_text()).get('status') != 'completed-experiment':
            raise RuntimeError('experiment did not complete: ' + row['id'])
        checked = []
        for check in row['checks']:
            path = target / check['path']
            value = json.loads(path.read_text())
            for key in check['field']:
                value = value[key]
            if value != check['equals']:
                raise RuntimeError('experiment result check failed: ' + row['id'] + ' ' + str(path))
            checked.append(dict(path=str(path), sha256=sha(path)))
        results.append(dict(id=row['id'], manifest_sha256=row['sha256'], report_sha256=sha(report), checks=checked))
    tmp = out / 'stage.json.tmp'
    tmp.write_text(json.dumps(dict(schema='tnca-run2-stage/1', status='pass', experiments=results), indent=2) + '\n')
    tmp.replace(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.out, a.python, json.loads(a.registration.read_text()))


if __name__ == '__main__':
    main()
