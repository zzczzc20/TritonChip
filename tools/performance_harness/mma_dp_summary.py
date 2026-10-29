#!/usr/bin/env python3
"""Derive the m.mma.dp.async summaries from raw run reports (no number is typed here).

    mma_dp_summary.py cases --base BASE/report.json --dp DP/report.json [--neg NEG/report.json] --out summary.json
    mma_dp_summary.py chat  --base BASE/report.json --dp DP/report.json --out summary.json

`cases`: fp8_cases.py reports of one profile with and without MATRIX_DP (the SystemC/TLM
model, uncalibrated): pass counts, the matrix-command census by opcode, and per case the model's
device cycles of the steady launches. `chat`: chat.py reports: model device cycles per step and
per kernel class from `transfers_inside_step`. Every value is read from the files named.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census(row):
    out = {}
    for key, n in (row.get('graph') or {}).get('mma', {}).items():
        op = key.split(': ', 1)[1]
        out[op.split('.k')[0]] = out.get(op.split('.k')[0], 0) + n
    return out


def cycles(row):
    return sum(l.get('device_cycles') or 0 for l in (row.get('graph') or {}).get('launches', []))


def cases(args):
    base, dp = (json.loads(Path(p).read_text()) for p in (args.base, args.dp))
    neg = json.loads(Path(args.neg).read_text()) if args.neg else None
    rows = {}
    for r in dp['rows']:
        rows[(r['group'], r['case'])] = dict(dp=r)
    for r in base['rows']:
        rows.setdefault((r['group'], r['case']), {})['base'] = r
    out = dict(schema='tnca-mma-dp-cases-summary/1', domain='SystemC/TLM model (uncalibrated)',
               inputs={k: dict(path=str(v), sha256=sha(v)) for k, v in
                       (('base', args.base), ('dp', args.dp)) + ((('neg', args.neg),) if args.neg else ())},
               reference_dot_mode=dict(base=base.get('reference_dot_mode'), dp=dp.get('reference_dot_mode'),
                                       neg=neg.get('reference_dot_mode') if neg else None),
               status=dict(base=base['summary'], dp=dp['summary']), opcodes=dict(base={}, dp={}), cases=[])
    for (group, name), pair in sorted(rows.items()):
        b, d = pair.get('base'), pair.get('dp')
        if b is None or d is None:
            out['cases'].append(dict(group=group, case=name, note='present in only one run'))
            continue
        for side, row in (('base', b), ('dp', d)):
            for op, n in census(row).items():
                out['opcodes'][side][op] = out['opcodes'][side].get(op, 0) + n
        cb, cd = cycles(b), cycles(d)
        entry = dict(group=group, case=name, base_status=b['status'], dp_status=d['status'],
                     base_device_cycles=cb, dp_device_cycles=cd,
                     ratio=round(cb / cd, 4) if cb and cd else None)
        if group in ('scaled_mm', 'linear'):
            out['cases'].append(entry)
    gm = [c['ratio'] for c in out['cases'] if c.get('ratio')]
    if gm:
        prod = 1.0
        for v in gm:
            prod *= v
        out['geomean_cycle_ratio_base_over_dp_scaled_mm_and_linear'] = round(prod ** (1 / len(gm)), 4)
        out['cases_with_ratio'] = len(gm)
    if neg:
        out['negative_control'] = dict(
            note='the dp-compiled kernels judged against the m.mma (sec 9C.1) reference: the cases must be '
                 'able to tell the two reductions apart', status=neg['summary'],
            failing=[dict(group=r['group'], case=r['case']) for r in neg['rows'] if r['status'] != 'pass'])
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'opcodes')}, indent=1))


def chat(args):
    base, dp = (json.loads(Path(p).read_text()) for p in (args.base, args.dp))
    out = dict(schema='tnca-mma-dp-chat-summary/1', domain='SystemC/TLM model (uncalibrated), reduced depth',
               inputs={k: dict(path=str(v), sha256=sha(v)) for k, v in (('base', args.base), ('dp', args.dp))},
               model=base.get('model'), decoder_layers=base.get('decoder_layers'), runs={})
    for side, rep in (('base', base), ('dp', dp)):
        steps = []
        for s in rep['steps']:
            t = s['transfers_inside_step']
            steps.append(dict(step=s['step'], kind=s['kind'], input_tokens=s['input_tokens'], token=s['token'],
                              launches=t['launches'], device_cycles=t['device_cycles'],
                              by_kernel=t.get('by_kernel')))
        out['runs'][side] = dict(status=rep.get('status'), generated_tokens=rep.get('generated_tokens'),
                                 teacher=rep.get('teacher_forced_cpu_w8a8'), steps=steps)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cases')
    c.add_argument('--base', required=True)
    c.add_argument('--dp', required=True)
    c.add_argument('--neg')
    c.add_argument('--out', required=True)
    c.set_defaults(fn=cases)
    h = sub.add_parser('chat')
    h.add_argument('--base', required=True)
    h.add_argument('--dp', required=True)
    h.add_argument('--out', required=True)
    h.set_defaults(fn=chat)
    a = ap.parse_args()
    a.fn(a)
