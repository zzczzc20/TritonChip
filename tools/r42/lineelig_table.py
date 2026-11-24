#!/usr/bin/env python3
"""R42 lineelig: per-point cycle table from link_budget_rtl.py records (domain: rtl-3c, the three-core board-top RTL stand-in).

    lineelig_table.py BASE_DIR ARM_DIR [ARM_DIR ...]

Each directory holds one <point>.jsonl per point (tools/r41/link_budget_rtl.py --out). Prints, per point, the
serial-grid device cycles of the timed call in each directory, the ratio of each arm to the first directory, the
status and max_rel_err of every record, and whether max_rel_err is identical (bit-identical results give identical
errors against the same reference). A point missing from an arm is printed as '-'. Raw records are never edited.
"""
import glob, json, math, os, sys


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            s = (r.get('device_cycles_samples') or [None])
            out[r['point']] = dict(cycles=s[0], status=r.get('status'),
                                   err=(r.get('correctness') or {}).get('max_rel_err'),
                                   launches=[x['cycles'] for x in r.get('timed_launches') or []])
    return out


def main():
    dirs = sys.argv[1:]
    recs = [load(d) for d in dirs]
    names = [os.path.basename(d.rstrip('/')) for d in dirs]
    pts = sorted(set().union(*[set(r) for r in recs]))
    print(f"{'point':58s} " + ' '.join(f'{n:>14s}' for n in names) + ' ' + ' '.join(f'{"ratio "+n:>14s}' for n in names[1:]) + '  status  err-identical')
    logs = [[] for _ in dirs[1:]]
    for p in pts:
        row = [r.get(p) for r in recs]
        cyc = [x['cycles'] if x else None for x in row]
        ratios = []
        for i, c in enumerate(cyc[1:], 1):
            if c and cyc[0]:
                ratios.append(c / cyc[0]); logs[i - 1].append(math.log(c / cyc[0]))
            else:
                ratios.append(None)
        st = ','.join(str(x['status'] if x else '-') for x in row)
        errs = [x['err'] for x in row if x]
        same = 'yes' if len(set(errs)) == 1 and len(errs) == len(row) else ('-' if len(errs) < len(row) else 'NO')
        print(f"{p:58s} " + ' '.join(f"{c if c else '-':>14}" for c in cyc) + ' ' +
              ' '.join(f"{r:14.3f}" if r else f"{'-':>14}" for r in ratios) + f'  {st}  {same}')
    for n, l in zip(names[1:], logs):
        if l:
            print(f'geometric mean ratio {n} / {names[0]}: {math.exp(sum(l) / len(l)):.3f} over {len(l)} points')


if __name__ == '__main__':
    main()
