#!/usr/bin/env python3
"""Markdown tables of the registered xl tier, from archbench/suites/xl.json (nothing typed in).

    xl_table.py [--suite archbench/suites/xl.json] [--compile DIR_OF_COMPILE_SUMMARY.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def human(sec):
    return f'{sec:.1f} s' if sec < 120 else f'{sec / 60:.1f} min'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', type=Path, default=ROOT / 'archbench' / 'suites' / 'xl.json')
    ap.add_argument('--compile', type=Path, help='the summary.json written by xl_compile_summary.py')
    a = ap.parse_args()
    doc = json.loads(a.suite.read_text())
    comp = {r['point']: r for r in json.loads(a.compile.read_text())['points']} if a.compile else {}
    print('| # | point | live MiB | bound Mcycles (binding) | est. device time (eta table) | pessimistic | tolerance |' +
          (' compile-only |' if comp else ''))
    print('|--:|---|--:|--:|--:|--:|--:|' + ('---|' if comp else ''))
    for i, e in enumerate(doc['points'], 1):
        es = e['estimate']
        row = (f'| {i} | `{e["id"]}` | {e["live_bytes"] / 2**20:.0f} | {es["bound_cycles"] / 1e6:,.2f} ({es["binding"]}) | '
               f'{human(es["seconds"])} | {human(es["seconds_pessimistic"])} | {e["tolerance"]:g} |')
        if comp:
            c = comp.get(e['id'])
            row += (f' {c["verdict"]}, {c["compile_s"]:.0f} s, {c["launches"]} launches, {c["kernels"]} kernels |'
                    if c else ' not run |')
        print(row)
    print()
    print(f'Total estimated device time {human(doc["estimate_total_seconds"])} '
          f'(pessimistic {human(doc["estimate_total_seconds_pessimistic"])}); largest point {human(doc["estimate_max_seconds"])} '
          f'(pessimistic {human(doc["estimate_max_seconds_pessimistic"])}); FP8 matrix points '
          f'{doc["fp8_matmul_points"]}/{len(doc["points"])} = {doc["fp8_matmul_share"]:.0%}; largest live set '
          f'{doc["max_live_bytes"] / 2**20:.0f} MiB of {doc["memory_cap_bytes"] / 2**20:.0f} MiB.')


if __name__ == '__main__':
    main()
