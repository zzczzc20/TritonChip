#!/usr/bin/env python3
"""R40-07.policy board A/B: the automatic pipeline policy against pipeline-off, at 90 MHz.

    v3_policy_board.py plan  --inventory evidence/r40/v3/R40-07.policy/inventory_existing/report.json \
                             --out DIR [--base B40-N]
    v3_policy_board.py run   --out DIR --xclbin X --operating-clock-evidence C
    v3_policy_board.py judge --out DIR

Only the tiny/small/medium points whose executable payload the frozen automatic
policy changes (tools/r40/v3_policy_inventory.py: TNCA_RCTX_ROLES=1 at 2 or 3
stages differs from pipeline-off) are measured; every other point runs the
identical executable on the same image and clock under both settings.

Control: the base knobs (doc/r40_v3/plan.json baseline_knobs) with
TNCA_RCTX_ROLES=0. Policy: the same with TNCA_RCTX_ROLES=1, TNCA_PIPELINE=2.
Collection reuses tools/r40/v3_campaign.py's runner exactly (one ArchBench
process per configuration, cold call apart, one qualification sample, then
three timed calls when both sides are <= 1 s; seeded order; identity, clock,
knob, launch-map and sampling checks on every record). The judgement is per
point in seconds at the measured clock: a policy/control ratio above 1.05 is an
unresolved > 5 % regression (doc R40-07 acceptance 3); above 1.02 is listed.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm'),
                str(ROOT / 'tools/r40')]
import ablate  # noqa: E402
import contract_v3  # noqa: E402
import v3_campaign as C  # noqa: E402

NAME = 'R40-07.policy'


def plan(a):
    inv = json.loads(Path(a.inventory).read_text())
    if inv.get('status') != 'complete' or inv['suite'] != 'spec/r40/suites/existing.json':
        raise SystemExit('the inventory must be complete over spec/r40/suites/existing.json')
    changed = sorted(p for p, row in inv['summary'].items()
                     if any((row.get(c) or {}).get('code_identical_to_off') is not True
                            for c in ('gated2', 'gated3')))
    suite = {p['id']: p for p in json.loads((ROOT / inv['suite']).read_text())['points']}
    pl = C.load_plan()
    base = dict(pl['baseline_knobs'][a.base])
    sides = dict(control=dict(base, TNCA_RCTX_ROLES='0'),
                 candidate=dict(base, TNCA_RCTX_ROLES='1', TNCA_PIPELINE='2'))
    rng = random.Random(pl['sample_plan']['bootstrap']['seed'])
    configs, rows = {}, []
    for pid in changed:
        point = suite[pid]
        keys = {}
        for side, knobs in sides.items():
            key = C.config_key(pid, knobs, 'device')
            keys[side] = key
            configs[key] = dict(key=key, point=dict(id=pid, sha256=point['sha256']), knobs=knobs,
                                klass='device', experiments=[NAME])
        order = ['control', 'candidate']
        rng.shuffle(order)
        rows.append(dict(point=pid, sha256=point['sha256'], keys=keys, order=order))
    campaign = dict(schema='tnca-r40-v3-policy-board/1', planned=C.utc(),
                    contract_sha256=contract_v3.LOCK_SHA256, harness_sha256=C.sha_file(__file__),
                    campaign_harness_sha256=C.sha_file(ROOT / 'tools/r40/v3_campaign.py'),
                    inventory=str(a.inventory), inventory_sha256=C.sha_file(a.inventory), base=a.base,
                    sides=sides, order=[NAME],
                    experiments={NAME: dict(name=NAME, klass='device', timing='device-cycles-serial-grid',
                                            points=rows)},
                    configs=configs)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'campaign.json').write_text(json.dumps(campaign, indent=1) + '\n')
    print(f'{len(rows)} point(s), {len(configs)} configuration(s): {changed}')


def run(a):
    campaign = json.loads((a.out / 'campaign.json').read_text())
    r = C.Runner(a.out, campaign, a)
    d = contract_v3.delivery(r.plan)
    from archbench.provenance import software_tree
    if software_tree(refresh=True) != d['software_tree']:
        raise SystemExit('product tree is not the v3 delivery identity')
    if C.sha_file(a.xclbin) != d['image_sha256']:
        raise SystemExit('xclbin is not the v3 delivery image')
    exp = campaign['experiments'][NAME]
    failed = {}
    for row in exp['points']:
        try:
            r.device_experiment(dict(exp, points=[row]))
        except Exception:                                    # noqa: BLE001  preserve and continue
            failed[row['point']] = traceback.format_exc()[-2000:]
    r.state['experiments'][NAME] = dict(status='collected' if not failed else 'failed-preserved',
                                        failed=failed, ended=C.utc())
    if software_tree(refresh=True) != d['software_tree']:
        raise SystemExit('product tree changed during collection')
    r.state.update(status='complete', current=None)
    r.save()


def judge(a):
    campaign = json.loads((a.out / 'campaign.json').read_text())
    state = json.loads((a.out / 'state.json').read_text())
    pl = C.load_plan()
    d = contract_v3.delivery(pl)
    points, ok = [], state.get('status') == 'complete'
    for row in campaign['experiments'][NAME]['points']:
        sec = {}
        for side in ('control', 'candidate'):
            key = row['keys'][side]
            entry = state['runs'].get(f'{key}-f3') or state['runs'].get(f'{key}-q')
            if not entry or not entry.get('ok'):
                sec[side] = None
                continue
            rec = ablate.load(entry['raw'])[0]
            samples = ablate.samples_of(rec, 'device-cycles-serial-grid')[1]
            sec[side] = statistics.median(samples) / d['clock_hz']
        ratio = (sec['candidate'] / sec['control']) if all(sec.values()) else None
        points.append(dict(point=row['point'], control_s=sec['control'], policy_s=sec['candidate'],
                           ratio=ratio, over_2pct=bool(ratio and ratio > 1.02),
                           over_5pct=bool(ratio is None or ratio > 1.05)))
        ok &= ratio is not None and ratio <= 1.05
    result = dict(schema='tnca-r40-v3-policy-board-result/1', ok=bool(ok), software_tree=d['software_tree'],
                  image_sha256=d['image_sha256'], clock_hz=d['clock_hz'], base=campaign['base'],
                  sides=campaign['sides'], points=[p['point'] for p in points if p['ratio'] is not None],
                  per_point=points, campaign_sha256=C.sha_file(a.out / 'campaign.json'),
                  state_sha256=C.sha_file(a.out / 'state.json'))
    (a.out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    for p in points:
        print(f"{p['point']:<48s} control {p['control_s']} policy {p['policy_s']} ratio {p['ratio']}")
    print(f"R40-07.policy board A/B: {'no unresolved >5 % regression' if ok else 'NOT PASSED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base', default='B40-N', choices=('B40-N', 'B40-S'))
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--xclbin', type=Path, required=True)
    r.add_argument('--operating-clock-evidence', type=Path, required=True)
    j = sub.add_parser('judge')
    j.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    return {'plan': plan, 'run': run, 'judge': judge}[a.cmd](a) or 0


if __name__ == '__main__':
    sys.exit(main())
