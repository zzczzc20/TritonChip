#!/usr/bin/env python3
"""Run a registered sequence of board experiments and publish its result."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import auto

PROFILE = 'core-vrf-sync-v5-x8-g2c4-dp'


def check_coverage(root, rows, config):
    suite_path = root / config['suite']
    if sha(suite_path) != config['suite_sha256']:
        raise ValueError('registered suite changed')
    points = json.loads(suite_path.read_text())['points']
    expected = {p['id']: p['sha256'] for p in points}
    if len(expected) != len(points) or len(expected) != config['count']:
        raise ValueError('suite has duplicate points or wrong size')
    found = {}
    for row in rows:
        spec = json.loads((root / row['manifest']).read_text())
        for batch in spec.get('board_batches', {}).values():
            for case in batch.get('configurations', []):
                name = case['point']
                if name in found:
                    raise ValueError('point appears more than once: ' + name)
                knobs = case.get('knobs', {})
                if knobs.get('TNCA_TILED_POINTWISE', '0') != config['tiled_pointwise']:
                    raise ValueError('wrong tiled pointwise setting: ' + name)
                found[name] = case['point_sha256']
    if found != expected:
        raise ValueError('registered experiment points do not exactly match the suite')


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
    identity = registration.get('image_identity', {})
    image_sha = identity.get('sha256', auto.IMAGE_SHA)
    image_uuid = identity.get('uuid', auto.IMAGE_UUID)
    if (not isinstance(image_sha, str) or len(image_sha) != 64 or
            any(c not in '0123456789abcdef' for c in image_sha) or
            not isinstance(image_uuid, str) or len(image_uuid) != 36):
        raise ValueError('invalid registered image identity')
    if len(ids) != len(set(ids)) or any(not x.replace('-', '').replace('_', '').isalnum() for x in ids):
        raise ValueError('experiment IDs must be unique simple names')
    summary = out / 'stage.json'
    if summary.exists():
        raise ValueError('stage result already exists')
    for row in rows:
        manifest = root / row['manifest']
        if not manifest.is_file() or sha(manifest) != row['sha256']:
            raise ValueError('registered manifest missing or changed: ' + str(manifest))
        spec = json.loads(manifest.read_text())
        board = spec.get('board', {})
        args = board.get('backend_args', {})
        if (spec.get('id') != row['id'] or board.get('image_sha256') != image_sha
                or board.get('image_uuid') != image_uuid or board.get('clock_hz') != 100_000_000
                or board.get('profile') != PROFILE or args.get('image_profile') != PROFILE
                or args.get('core_count') != 3 or args.get('memory_ports') != 3
                or args.get('hbm_bytes') != 8 * (1 << 30) or args.get('hbm_dma') is not True):
            raise ValueError('manifest is not registered for the run-2 board configuration: ' + row['id'])
        if (out / row['id']).exists():
            raise ValueError('experiment output already exists: ' + row['id'])
        if not row.get('checks'):
            raise ValueError('each experiment needs explicit result checks')
        if 'budget_seconds' in row and (type(row['budget_seconds']) not in (int, float)
                or not math.isfinite(row['budget_seconds']) or not 0 < row['budget_seconds'] <= 10800):
            raise ValueError('registered experiment budget must be at most three hours')
    if registration.get('coverage'):
        check_coverage(root, rows, registration['coverage'])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        target = out / row['id']
        if sha(root / row['manifest']) != row['sha256']:
            raise ValueError('registered manifest changed before execution: ' + row['id'])
        command = [str(python), str(root / 'tools/performance_harness/experiment.py'),
                   str(root / row['manifest']), '--root', str(root), '--out', str(target),
                   '--python', str(python)]
        if 'budget_seconds' in row:
            command += ['--budget-seconds', str(row['budget_seconds'])]
        with (out / (row['id'] + '.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha(root / row['manifest']) != row['sha256']:
            raise ValueError('registered manifest changed during execution: ' + row['id'])
        report = target / 'report.json'
        if json.loads(report.read_text()).get('status') != 'completed-experiment':
            raise RuntimeError('experiment did not complete: ' + row['id'])
        checked = []
        for check in row['checks']:
            relative = Path(check['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('result check must stay under its experiment output')
            path = target / relative
            value = json.loads(path.read_text())
            for key in check['field']:
                value = value[key]
            if 'equals' in check:
                passed = value == check['equals'] and type(value) is type(check['equals'])
            elif 'length_equals' in check:
                passed = isinstance(value, (list, dict)) and len(value) == check['length_equals']
            elif 'minimum' in check:
                passed = type(value) in (int, float) and math.isfinite(value) and value >= check['minimum']
            else:
                raise ValueError('result check needs equals, length_equals or minimum')
            if not passed:
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
