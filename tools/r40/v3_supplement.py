#!/usr/bin/env python3
"""Supplementary diagnostic A/B runs on the v3 delivery identity, outside the frozen plan.

    v3_supplement.py plan  --out DIR --name NAME --subset SUITE.json --baseline B40-N \
                           --knob TNCA_RTL_MEM_PORTS --control 1 --candidate 3
    v3_supplement.py run   --out DIR --xclbin X --operating-clock-evidence C
    v3_supplement.py judge --out DIR

Same collection, sampling and judgement as tools/r40/v3_campaign.py (it
reuses that module), but the experiment is not one of the frozen v3
experiments: its result is a diagnostic for a decision and never records a
gate. The judgement says so.
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_campaign as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--subset', required=True)
    p.add_argument('--baseline', choices=('B40-S', 'B40-N'), required=True)
    p.add_argument('--knob', required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--timing', default='graph-makespan')
    p.add_argument('--reason', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--xclbin', type=Path, required=True)
    r.add_argument('--operating-clock-evidence', type=Path, required=True)
    j = sub.add_parser('judge')
    j.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == 'plan':
        plan = C.load_plan()
        base = dict(plan['baseline_knobs'][a.baseline])
        sides = {s: dict(base, **{a.knob: v}) for s, v in (('control', a.control), ('candidate', a.candidate))}
        rng = random.Random(plan['sample_plan']['bootstrap']['seed'])
        configs, rows = {}, []
        for point in sorted(json.loads((C.ROOT/a.subset).read_text())['points'], key=lambda q: q['id']):
            keys = {}
            for side, knobs in sides.items():
                key = C.config_key(point['id'], knobs, 'device')
                keys[side] = key
                configs[key] = dict(key=key, point=dict(id=point['id'], sha256=point['sha256']), knobs=knobs,
                                    klass='device', experiments=[a.name])
            order = ['control', 'candidate']
            rng.shuffle(order)
            rows.append(dict(point=point['id'], sha256=point['sha256'], keys=keys, order=order))
        exp = dict(name=a.name, subset=a.subset, vary=[a.knob], control=a.control, candidate=a.candidate,
                   candidate_value=a.candidate, baseline=a.baseline, timing=a.timing, target=None,
                   rule='significant-gain', klass='device', points=rows,
                   note='SUPPLEMENTARY DIAGNOSTIC, not a frozen v3 experiment; never records a gate',
                   reason=a.reason)
        campaign = dict(schema='tnca-r40-v3-supplement/1', planned=C.utc(), contract_sha256=C.contract_v3.LOCK_SHA256,
                        plan_sha256=C.sha_file(C.PLAN), harness_sha256=C.sha_file(C.__file__),
                        supplement_sha256=C.sha_file(__file__), order=[a.name], experiments={a.name: exp},
                        configs=configs, supplementary=True)
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out/'campaign.json').write_text(json.dumps(campaign, indent=1)+'\n')
        print(f'{a.name}: {len(configs)} configurations')
        return 0
    campaign = json.loads((a.out/'campaign.json').read_text())
    assert campaign.get('supplementary') is True
    if a.cmd == 'run':
        C.Runner(a.out, campaign, a).run()
        return 0
    state = json.loads((a.out/'state.json').read_text())
    report = C.judge(a.out, campaign, state, write_evidence=False)
    report['supplementary'] = 'diagnostic outside the frozen v3 plan; not gate evidence'
    (a.out/'judgement.json').write_text(json.dumps(report, indent=1)+'\n')
    for name, res in report['experiments'].items():
        print(name, res['status'], res.get('s_geo'), res.get('ci95'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
