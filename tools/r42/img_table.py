#!/usr/bin/env python3
"""R42-img: per-point table of the arms of the combined-image study (rtl-3c records of tools/r42/lineelig_mc.py).

    img_table.py BASE_ARM_NAME ROOT [--arms A,L,M,...]

ROOT holds directories <ARM>_<set> (one or more sets per arm) with the link_budget_rtl.py records (*.jsonl) and digest.jsonl.
For every point: device cycles (serial grid, the timed call) of each arm, the ratio against the base arm, the status, the largest relative
error against the CPU reference and the tolerance, and whether the sha256 of the compared output tensors equals the base arm's (bit-identical).
A point measured twice on one arm (two sets) keeps the first record and lists the repeat as `rep`.
Domain: rtl-3c, never board.
"""
import glob, json, math, os, re, sys


def load(root):
    arms = {}
    for d in sorted(glob.glob(os.path.join(root, '*_*'))):
        m = re.match(r'(.+?)_([A-Za-z0-9]+)$', os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        arm = m.group(1)
        dig = {}
        dp = os.path.join(d, 'digest.jsonl')
        if os.path.exists(dp):
            for l in open(dp):
                r = json.loads(l)
                dig.setdefault(r['point'], []).append(r['output_sha256'])
        for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
            if f.endswith('digest.jsonl'):
                continue
            for l in open(f):
                r = json.loads(l)
                c = r.get('correctness') or {}
                s = r.get('device_cycles_samples') or []
                rec = dict(cycles=s[0] if s else None, status=r.get('status'), err=c.get('max_rel_err') if isinstance(c, dict) else None,
                           tol=c.get('tolerance') if isinstance(c, dict) else None, wall=r.get('wall_seconds'),
                           sha=dig.get(r['point'], [None]), dir=os.path.basename(d),
                           cold=((r.get('time') or {}).get('first_call') or {}).get('device_cycles'))
                arms.setdefault(arm, {}).setdefault(r['point'], []).append(rec)
    return arms


def main():
    base = sys.argv[1]
    root = sys.argv[2]
    order = None
    if '--arms' in sys.argv:
        order = sys.argv[sys.argv.index('--arms') + 1].split(',')
    arms = load(root)
    names = order or sorted(arms, key=lambda a: (a != base, a))
    points = sorted({p for a in arms.values() for p in a})
    print(f'rtl-3c, device cycles of the timed call (serial grid); ratio = arm / {base}; = : output sha256 equal to {base}\'s, ! : differs\n')
    print(f'{"point":58s}' + ''.join(f'{n:>16s}' for n in names))
    logs = {n: [] for n in names}
    for p in points:
        row = f'{p:58s}'
        b = arms.get(base, {}).get(p)
        bc = b[0]['cycles'] if b and b[0]['status'] == 'pass' else None
        for n in names:
            r = arms.get(n, {}).get(p)
            if not r:
                row += f'{"-":>16s}'
                continue
            r0 = r[0]
            if r0['status'] != 'pass':
                # the failure stays the first record; later records of the same arm and point (reruns) are listed beside it
                again = ','.join(str(x['cycles']) for x in r[1:] if x['status'] == 'pass')
                row += f'{("FAIL>" + again) if again else "FAIL":>16s}'
                continue
            txt = f'{r0["cycles"]}'
            if n != base and bc:
                ratio = r0['cycles'] / bc
                logs[n].append(math.log(ratio))
                eq = '' if not b[0]['sha'][0] or not r0['sha'][0] else ('=' if b[0]['sha'][0] == r0['sha'][0] else '!')
                txt += f' {ratio:.3f}{eq}'
            if len(r) > 1:
                txt += f' rep{r[1]["cycles"]}'
            row += f'{txt:>16s}'
        print(row)
    print()
    for n in names:
        if n != base and logs[n]:
            print(f'geometric mean {n}/{base}: {math.exp(sum(logs[n]) / len(logs[n])):.3f} over {len(logs[n])} points')
    # status and error summary
    bad = [(n, p) for n, d in arms.items() for p, rs in d.items() for r in rs if r['status'] != 'pass']
    print(f'\nrecords: {sum(len(rs) for d in arms.values() for rs in d.values())}, not pass: {bad or "none"}')


if __name__ == '__main__':
    main()
