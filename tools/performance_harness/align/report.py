#!/usr/bin/env python3
"""Render alignment.json files (tracediff.py gate) as a Markdown table.

    report.py --cycles A.json [--order B.json] [--title ...]

A: the cycle-exact gate; B (optional): the same points judged --order-only
--data, whose columns say whether the sequences agree even where cycles do not.
"""
import argparse
import json
from pathlib import Path


def first(v):
    if v == 'equal':
        return 'equal'
    if isinstance(v, dict) and 'index' in v:
        return f"L{v['launch']} #{v['index']}"
    if isinstance(v, dict) and 'launch' in v:
        return f"L{v['launch']}"
    return str(v)[:24]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cycles', required=True)
    ap.add_argument('--order')
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    g = json.loads(Path(a.cycles).read_text())
    o = json.loads(Path(a.order).read_text()) if a.order else None
    agg = g['aggregates']
    out = []
    if a.title:
        out.append(f'### {a.title}\n')
    out.append(f"Verdict: **{g['verdict']}** ({g['reason']}). Points {agg['points']}, aligned {agg['aligned']}, "
               f"diverged {agg['diverged']}, missing {agg['missing']}"
               + (f"; |cycle error| median {agg['median_abs_cycle_error']:.1%}, p95 {agg['p95_abs_cycle_error']:.1%}."
                  if agg['median_abs_cycle_error'] is not None else '.') + '\n')
    head = '| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |'
    sep = '|---|---:|---:|---:|---:|---|---|'
    if o:
        head += ' order-only: issue | request | data |'
        sep += '---|---|---|'
    out += [head, sep]
    for k in sorted(g['points']):
        p = g['points'][k]
        if p.get('status') in ('missing', 'malformed'):
            out.append(f"| {k} | {p['status']} | | | | {p.get('error', '')[:60]} | |" + (' | | |' if o else ''))
            continue
        s = p['streams']
        row = (f"| {k} | {p['aligned_launches']}/{p['launches_a']} | {p['cycles_a']} | {p['cycles_b']} | "
               f"{p.get('cycle_error', 0):+.2%} | {first(s['issue'])} | {first(s['request'])} |")
        if o:
            q = o['points'].get(k, {}).get('streams', {})
            row += f" {first(q.get('issue'))} | {first(q.get('request'))} | {first(q.get('response'))} |"
        out.append(row)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
