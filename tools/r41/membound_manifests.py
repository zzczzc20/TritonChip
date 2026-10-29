#!/usr/bin/env python3
"""Board manifests for the baseline of the registered R41 memory-bound suite.

    python3 tools/r41/membound_manifests.py [--tiled2] [--check]

Writes spec/r41/hbm_bw/membound/<id>.json: experiment.py manifests whose one
`fpga` step runs tools/performance_harness/board_batch.py over a slice of
archbench/suites/r41_membound.json on the already-approved run-1 100 MHz image
(UUID 49964b17-...), through the standard path with default knobs, with the
same harness, sampling rule and timing boundary as `s2-original-suite`
(spec/r41/stage2/board/s2-original-suite.json, which this copies its board,
plan and environment declarations from).

Every registered point appears in exactly one manifest, in the suite's order
(size, then family). The slices exist only because acceptance_queue.py gives
one queue at most three hours; the split uses an ESTIMATE of board time, fitted
here from the run-1 records in evidence/r41/hbm_bw/link_budget/board_run1/
(per-process wall time against moved bytes and device time, device time from
the slowest measured streaming rate of the operator's class). The estimate
decides only where a slice ends; it is not a result.

--tiled2 writes the parallel spec/r41/hbm_bw/membound_tiled2/ set with only
TNCA_TILED_POINTWISE=2 added to the run knobs, plus the s2-original-suite copy.
Those manifests are non-default default-change-gate evidence, not target results.

--check regenerates in memory and exits 1 if a written manifest differs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SUITE = 'archbench/suites/r41_membound.json'
TEMPLATE = 'spec/r41/stage2/board/s2-original-suite.json'
RUN1 = 'evidence/r41/hbm_bw/link_budget/board_run1/run1_streaming.tgz'
OUT = 'spec/r41/hbm_bw/membound'
TILED_OUT = 'spec/r41/hbm_bw/membound_tiled2'
SLICE_HOURS = 1.5
KNOBS = {
    # The same explicit set as s2-original-suite. TNCA_ACTIVE_CORES=3 equals the
    # code default (all physical cores, python/tnca/runtime.py
    # _negotiate_physical_grid) and is stated because board_batch.py checks the
    # launch map against it; TNCA_TRANSFER_POLICY=resident is the driver default
    # (python/tnca/triton_backend/driver.py); TNCA_TORCH_DEVICE=native selects
    # tensors on device "tnca", the standard path of doc/engineering_discipline.md.
    'TNCA_ACTIVE_CORES': '3',
    'TNCA_TORCH_DEVICE': 'native',
    'TNCA_TRANSFER_POLICY': 'resident',
}
CLASSES = {
    'stream': ('copy', 'clone', 'cast', 'relu', 'add', 'sub', 'mul'),
    'sfu': ('silu', 'gelu', 'exp', 'sigmoid', 'div'),
    'row': ('sum', 'mean', 'max', 'layernorm', 'rmsnorm', 'softmax'),
}
REFERENCE_FAMILIES = {'stream': ('add', 'copy'), 'sfu': ('silu', 'gelu_tanh'),
                      'row': ('softmax', 'rmsnorm', 'layernorm')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run1_records():
    """(point id, family, tier, moved bytes, device seconds, wall seconds) of run-1 runs."""
    rows = []
    with tarfile.open(ROOT/RUN1) as tar:
        state = json.load(tar.extractfile('state.json'))
        for stem, entry in sorted(state['runs'].items()):
            if not entry.get('ok'):
                continue
            member = 'runs/' + Path(entry['raw']).name
            try:
                record = json.loads(tar.extractfile(member).read().decode().splitlines()[0])
            except KeyError:
                continue
            d = record['demand']
            rows.append(dict(id=record['point_id'], family=record['family'], tier=record['tier'],
                             bytes=d['read_bytes'] + d['write_bytes'], device=entry['seconds'],
                             timed=entry['timed'], wall=entry['wall_seconds']))
    return rows


def solve3(rows):
    """Least squares wall = a + b * MiB + c * device seconds x (1 + timed calls)."""
    xs = [(1.0, r['bytes'] / (1 << 20), r['device'] * (1 + r['timed'])) for r in rows]
    ys = [r['wall'] for r in rows]
    m = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    v = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    for i in range(3):                        # Gauss-Jordan on the 3x3 normal equations
        p = max(range(i, 3), key=lambda k: abs(m[k][i]))
        m[i], m[p], v[i], v[p] = m[p], m[i], v[p], v[i]
        for k in range(3):
            if k != i:
                f = m[k][i] / m[i][i]
                m[k] = [a - f * b for a, b in zip(m[k], m[i])]
                v[k] -= f * v[i]
    return [v[i] / m[i][i] for i in range(3)]


def estimator():
    rows = run1_records()
    fit = solve3(rows)
    rate = {}
    for cls, fams in REFERENCE_FAMILIES.items():
        big = [r for r in rows if r['family'] in fams and r['tier'] in ('medium', 'large', 'hbm')]
        rate[cls] = min(r['bytes'] / r['device'] for r in big)
    cls_of = {f: c for c, fams in CLASSES.items() for f in fams}

    def estimate(point):
        d = point['demand']
        moved = d['read_bytes'] + d['write_bytes']
        device = moved / rate[cls_of[point['family']]]
        short = device <= 1.0
        one = fit[0] + fit[1] * moved / (1 << 20) + fit[2] * device * 2
        # device rule: a short point runs again with three timed calls
        return one + ((fit[0] + fit[1] * moved / (1 << 20) + fit[2] * device * 4) if short else 0.0)
    return estimate, dict(fit_wall_seconds=dict(constant=fit[0], per_mib=fit[1], per_device_second=fit[2]),
                          device_bytes_per_second=rate, reference_runs=len(rows), source=RUN1,
                          source_sha256=sha(ROOT/RUN1))


def build(tiled2=False):
    suite_path = ROOT/SUITE
    suite = json.loads(suite_path.read_text())
    template = json.loads((ROOT/TEMPLATE).read_text())
    out_dir = TILED_OUT if tiled2 else OUT
    knobs = dict(KNOBS, **({'TNCA_TILED_POINTWISE': '2'} if tiled2 else {}))
    sys.path.insert(0, str(ROOT))
    from archbench import registry
    estimate, method = estimator()
    points = []
    for entry in suite['points']:
        p = registry.get(entry['id'])
        if p.definition_sha256() != entry['sha256']:
            raise SystemExit(f'{entry["id"]}: registry differs from the frozen suite')
        points.append(dict(id=p.id, sha256=entry['sha256'], family=p.family, size=p.params['size'],
                           demand=p.demand.as_dict()))
    order = {'small': 0, 'medium': 1, 'large': 2}
    points.sort(key=lambda p: (order[p['size']], p['id']))
    slices, current, seconds = [], [], 0.0
    for p in points:
        s = estimate(p)
        if current and (seconds + s > SLICE_HOURS * 3600 or p['size'] != current[-1]['size']):
            slices.append((current, seconds))
            current, seconds = [], 0.0
        current.append(p)
        seconds += s
    slices.append((current, seconds))
    manifests = {}
    counts = {}
    for points_, seconds in slices:
        size = points_[0]['size']
        counts[size] = counts.get(size, 0) + 1
        mid = f'membound-{size}-{counts[size]}'
        manifests[mid] = (points_, seconds)
    out = {}
    for mid, (pts, seconds) in manifests.items():
        name = mid.replace('-', '_')
        spec = dict(
            schema='tnca-experiment-spec/1', id=mid, allowed_hosts=template['allowed_hosts'],
            hypothesis=('Non-default TNCA_TILED_POINTWISE=2 gate measurement on the run-1 100 MHz image through '
                        'the standard path: every point of this slice is recorded with its device time and '
                        'numerical result, pass or fail.' if tiled2 else
                        'Baseline of the registered R41 memory-bound suite (archbench/suites/r41_membound.json) on the '
                        'run-1 100 MHz image through the standard path with default knobs: every point of this slice '
                        'is recorded with its device time and numerical result, pass or fail.'),
            competing_explanations=[
                'a point fails, falls back on the host or times out and would be silently dropped',
                'a large point is skipped or its workload reduced',
                'card state from earlier points leaks into later ones',
                'the loaded image, clock or card ownership changes during the slice',
            ],
            expected_counter_changes=([
                'eligible pointwise kernels use tiled lane-group lowering; launch counts, grids and per-launch '
                'device cycles may change against the default baseline',
                'no host fallback and no upload/download inside a resident call',
                'only TNCA_TILED_POINTWISE changes; the image, other knobs, sampling and timing boundary are unchanged',
            ] if tiled2 else [
                'none against a previous run: this is the first measurement of these points; it records '
                'per-launch device cycles (serial grid and graph makespan) for every point',
                'no host fallback and no upload/download inside a resident call',
                'launch counts and grids are those the default compiler produces; no knob changes them',
            ]),
            expected_observations=[
                f'{len(pts)} configurations recorded, each pass or a preserved failure with its reason',
                'device cycles per passing point, from which full-link bandwidth (demand bytes / device time) '
                'is computed for evidence/r41/hbm_bw/LINK_BUDGET.md',
            ],
            board=template['board'], sources=sorted(set(template['sources'] + [
                SUITE, 'archbench/membound.py', f'{out_dir}/{mid}.json'])),
            env=template['env'],
            registration=dict(
                status='registered-not-run', registered_utc='2026-10-28',
                contract=('evidence/r41/hbm_bw/g4/gate_default_change (board gate evidence, not a target result)'
                          if tiled2 else
                          'evidence/r41/hbm_bw/BANDWIDTH_PLAN.md sections 1 and 3 (baseline of the registered suite)'),
                launch=('evidence/r41/hbm_bw/launches/membound_tiled2.sh' if tiled2 else
                        'evidence/r41/hbm_bw/launches/membound_baseline.sh'),
                generator='tools/r41/membound_manifests.py',
                suite=dict(path=SUITE, sha256=sha(suite_path), suite_sha256=suite['sha256']),
                expected_board_hours=round(seconds / 3600, 3),
                estimate_method=method,
                knobs=knobs,
                sampling_rule=template['registration']['sampling_rule'],
                reference=('evidence/r41/hbm_bw/membound_baseline: the same points with default knobs'
                           if tiled2 else
                           'none: these points have no earlier board record; the non-regression set keeps the '
                           's2-original-suite run-1 records as its baseline')),
            steps=[dict(
                id=name, kind='fpga',
                argv=['{python}', 'tools/performance_harness/board_batch.py', '--manifest', f'{out_dir}/{mid}.json',
                      '--batch', name, '--root', '{root}', '--out', '{out}/' + name, '--python', '{python}'],
                timeout_seconds=10800,
                timing_boundary=template['steps'][0]['timing_boundary'],
                cache_state=template['steps'][0]['cache_state'],
                outputs=['{out}/' + name + '/batch.json', '{out}/' + name + '/formal.jsonl'])],
            plan=template['plan'], plan_sha256=template['plan_sha256'], measurement=template['measurement'],
            suites=[dict(path=SUITE, sha256=sha(suite_path))],
            board_batches={name: dict(
                sampling='device-rule',
                configurations=[dict(id=p['id'], point=p['id'], point_sha256=p['sha256'], knobs=dict(knobs))
                                for p in pts])})
        if tiled2:
            spec['registration']['non_default'] = True
            spec['registration']['evidence_scope'] = 'default-change gate only; not a bandwidth-target result'
        out[f'{out_dir}/{mid}.json'] = json.dumps(spec, indent=1) + '\n'
    covered = [c['point'] for text in out.values()
               for b in json.loads(text)['board_batches'].values() for c in b['configurations']]
    if sorted(covered) != sorted(p['id'] for p in points) or len(set(covered)) != len(covered):
        raise SystemExit('the slices do not partition the suite exactly')
    if tiled2:
        rel = f'{TILED_OUT}/s2-original-suite.json'
        nonreg = json.loads(json.dumps(template))
        nonreg['hypothesis'] = ('Non-default TNCA_TILED_POINTWISE=2 gate measurement: the original R40 full suite '
                                'passes its numeric references on the run-1 image with no fallback.')
        nonreg['expected_counter_changes'] = [
            'eligible pointwise kernels use tiled lane-group lowering; launch counts, grids and per-launch '
            'device cycles may change against the default run-1 records',
            'no new host fallback or upload/download inside resident calls',
            'only TNCA_TILED_POINTWISE changes; the image, other knobs, sampling and timing boundary are unchanged',
        ]
        nonreg['sources'] = sorted((set(template['sources']) - {TEMPLATE}) |
                                   {rel, 'tools/r41/membound_manifests.py'})
        nonreg['registration'].update(
            contract='evidence/r41/hbm_bw/g4/gate_default_change (board gate evidence, not a target result)',
            launch='evidence/r41/hbm_bw/launches/membound_tiled2.sh',
            generator='tools/r41/membound_manifests.py', knobs=knobs,
            non_default=True, evidence_scope='default-change gate only; not a bandwidth-target result')
        nonreg['steps'][0]['argv'] = [rel if arg == TEMPLATE else arg
                                      for arg in nonreg['steps'][0]['argv']]
        for expansion in nonreg['board_batches']['original']['expand']:
            expansion['knobs'] = dict(knobs)
        out[rel] = json.dumps(nonreg, indent=1) + '\n'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--tiled2', action='store_true',
                    help='generate the parallel non-default TNCA_TILED_POINTWISE=2 board gate manifests')
    args = ap.parse_args()
    out = build(tiled2=args.tiled2)
    out_dir = TILED_OUT if args.tiled2 else OUT
    bad = 0
    for rel, text in out.items():
        path = ROOT/rel
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f'differs: {rel}')
                bad += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        spec = json.loads(text)
        batch = next(iter(spec['board_batches'].values()))
        count = (len(batch['configurations']) if 'configurations' in batch else
                 sum(len(json.loads((ROOT/e['suite']).read_text())['points']) for e in batch['expand']))
        print(f"{rel}: {count} points, estimated {spec['registration']['expected_board_hours']} h")
    if args.check:
        stale = sorted(set(str(p.relative_to(ROOT)) for p in (ROOT/out_dir).glob('*.json')) - set(out))
        for rel in stale:
            print(f'not generated: {rel}')
        bad += len(stale)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
