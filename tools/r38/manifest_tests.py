#!/usr/bin/env python3
"""S1-03 acceptance: records and candidate manifests.

doc/r38_stage1_tasks_and_acceptance.md S1-03, tests 1-5. Test 1 reads the
records the smoke run wrote (run `tools/r38/run.py --suite smoke` first).
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import manifest as M                                        # noqa: E402

RECORDS = ROOT / 'sim' / 'build' / 'r38' / 'smoke' / 'records.json'
FAILS = []


def check(name, problems):
    FAILS.extend(f'{name}: {p}' for p in problems)
    print(f'{"FAIL" if problems else "PASS"} {name}' + ''.join(f'\n     {p}' for p in problems[:6]))


def must_reject(rec, edit, why):
    r = copy.deepcopy(rec)
    edit(r)
    return [] if M.validate(r) else [f'accepted: {why}']


def main():
    if not RECORDS.exists():
        check('1 smoke records valid', ['no smoke records; run tools/r38/run.py --suite smoke'])
        return 1
    records = json.loads(RECORDS.read_text())
    check('1 smoke records valid', [f'{r["workload_id"]}: {p}' for r in records for p in M.validate(r)]
          + ([] if records else ['no records']))

    good = next(r for r in records if r['bounds']['eta'] is not None)
    bad = []
    bad += must_reject(good, lambda r: r.pop('workload_id'), 'missing field')
    bad += must_reject(good, lambda r: r['time'].pop('device_cycles'), 'missing nested field')
    bad += must_reject(good, lambda r: r.update(evidence_level='board-measured'), 'illegal evidence level')
    bad += must_reject(good, lambda r: r['time'].update(device_cycles=0), 'zero standing in for unknown cycles')
    bad += must_reject(good, lambda r: r['time'].update(host_seconds=0.0), 'zero host time')
    bad += must_reject(good, lambda r: r['resources'].update(lut=0), 'a LUT count in Stage 1')
    bad += must_reject(good, lambda r: r['time'].update(achieved_clock_hz=1e8), 'a clock in Stage 1')
    bad += must_reject(good, lambda r: r['bounds'].update(eta=1.2), 'eta above 1')
    bad += must_reject(good, lambda r: r['bounds'].update(eta=good['bounds']['eta'] * 1.01), 'eta inconsistent')
    bad += must_reject(good, lambda r: r.update(candidate_hash='abc'), 'not a hash')
    bad += must_reject(good, lambda r: r['time'].update(device_cycles='13549'), 'cycles as a string')
    check('2 malformed records rejected', bad)

    # 3 and 4: content hashes, in copies of the tree.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dirs = ['rtl', 'tb', 'compiler', 'isa', 'python/tnca', 'spec', 'tools/r37', 'tools/r38']
        for tag in ('a', 'b'):
            for d in dirs:
                shutil.copytree(ROOT / d, tmp / tag / d,
                                ignore=shutil.ignore_patterns('__pycache__', 'build', 'data'))
        env = {'fixed': True}
        a, b = M.candidate(tmp / 'a', env=env), M.candidate(tmp / 'b', env=env)
        check('4 same content, two directories, same candidate',
              [] if a == b else [f'{a["components"]} vs {b["components"]}'])
        moved = []
        for key, rel in (('H', 'rtl/core/tnca_lsu.v'), ('C', 'compiler/python/tnca_cc/frontend.py'),
                         ('I', 'isa/tnca-isa-v0.1.json'), ('R', 'python/tnca/runtime.py'),
                         ('P', 'spec/r38/counters.json'), ('C', 'python/tnca/aot.py')):
            p = tmp / 'b' / rel
            if not p.exists():
                moved.append(f'{rel} does not exist')
                continue
            original = p.read_bytes()
            p.write_bytes(original + b'\n')
            c = M.candidate(tmp / 'b', env=env)
            changed = {k for k in a['components'] if a['components'][k] != c['components'][k]}
            if changed != {key}:
                moved.append(f'editing {rel} changed {sorted(changed)}, expected [{key}]')
            if c['candidate_hash'] == a['candidate_hash']:
                moved.append(f'editing {rel} left the candidate hash unchanged')
            p.write_bytes(original)
        e2 = M.candidate(tmp / 'b', env={'fixed': False})
        if {k for k in a['components'] if a['components'][k] != e2['components'][k]} != {'E'}:
            moved.append('environment change did not move E alone')
        check('3 each layer moves only its own hash', moved)

    wrong = []
    for field, value in (('bounds', 'r38-bounds-v999'), ('time', 'device-makespan')):
        r = copy.deepcopy(good)
        if field == 'bounds':
            r['bounds']['version'] = value
        else:
            r['time']['timing'] = value
        try:
            M.compatible([r], bound_version='r38-bounds-v0', timing='serial-grid')
            wrong.append(f'accepted a record with {field} {value}')
        except ValueError:
            pass
    check('5 scorer refuses other bound versions and timing protocols', wrong)
    print(f'manifest: {"FAIL" if FAILS else "PASS"}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
