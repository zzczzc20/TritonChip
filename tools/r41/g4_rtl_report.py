#!/usr/bin/env python3
"""RTL link probes of the tiled element-wise lowering against the default (G4).

    python3 tools/r41/g4_rtl_report.py [--write]

Reads the RTL-simulation records of tools/r41/link_budget_rtl.py:
  default   evidence/r41/hbm_bw/link_budget/rtl/batch*/*.jsonl (G3) and
            evidence/r41/hbm_bw/g4/rtl/mode0*.jsonl (G4, points G3 lacks);
  mode 1/2  evidence/r41/hbm_bw/g4/rtl/mode{1,2}*.jsonl (NON-DEFAULT
            TNCA_TILED_POINTWISE, recorded in every record).
Stage metrics come from link_budget_report.rtl_rows. Domain: RTL simulation of
the run-1 image parameters with the testbench memory model, never board.
With --write the table goes to evidence/r41/hbm_bw/g4/rtl/rtl_table.{md,json}.
"""
import argparse
import glob
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/r41'))
from link_budget_report import rtl_rows  # noqa: E402

G4 = ROOT/'evidence/r41/hbm_bw/g4/rtl'
METRICS = ('cycles', 'per_core', 'ipc', 'stall_lsu', 'port_words_per_cycle', 'port_mean_latency',
           'home_engine_busy', 'line_read_bytes_per_demand_read', 'outlet_busy')


def load(paths):
    rows = {}
    for r in rtl_rows(paths):
        rows[r['point']] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--arm', action='append', default=[], metavar='NAME=GLOB',
                    help='an extra column: the RTL records matching GLOB (repository-relative)')
    a = ap.parse_args()
    base = load(sorted(glob.glob(str(ROOT/'evidence/r41/hbm_bw/link_budget/rtl/batch*/*.jsonl'))) +
                sorted(glob.glob(str(G4/'mode0*.jsonl'))))
    arms = {m: load(sorted(glob.glob(str(G4/f'mode{m}*.jsonl')))) for m in (1, 2)}
    extra = {kv.split('=', 1)[0]: load(sorted(glob.glob(str(ROOT/kv.split('=', 1)[1])))) for kv in a.arm}
    points = sorted(set(arms[1]) | set(arms[2]) | {p for e in extra.values() for p in e})
    lines = ['| Point (rtl, 64 KiB) | default cycles | mode 1 cycles | mode 2 cycles | default B/cycle/core | '
             'mode 1 | mode 2 | mode 2 speedup | mode 2 IPC | mode 2 LSU stall | mode 2 port words/cycle | '
             'mode 2 port latency | mode 2 home busy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    out = []
    sp = {1: [], 2: []}
    for pid in points:
        b = base.get(pid, {})
        r = {m: arms[m].get(pid, {}) for m in (1, 2)}
        def cell(x, k, f='{:.3f}'):
            v = x.get(k)
            return f.format(v) if isinstance(v, (int, float)) else (x.get('status') or '–')
        for m in (1, 2):
            if b.get('cycles') and r[m].get('cycles'):
                sp[m].append(b['cycles'] / r[m]['cycles'])
        s2 = (b['cycles'] / r[2]['cycles']) if b.get('cycles') and r[2].get('cycles') else None
        lines.append(f"| {pid.split('/')[0][3:]} {pid.split('/')[2].split('-r')[0]} | {cell(b, 'cycles', '{:,}')} | "
                     f"{cell(r[1], 'cycles', '{:,}')} | {cell(r[2], 'cycles', '{:,}')} | {cell(b, 'per_core')} | "
                     f"{cell(r[1], 'per_core')} | {cell(r[2], 'per_core')} | "
                     f"{'–' if s2 is None else f'{s2:.2f}x'} | {cell(r[2], 'ipc')} | {cell(r[2], 'stall_lsu')} | "
                     f"{cell(r[2], 'port_words_per_cycle')} | {cell(r[2], 'port_mean_latency', '{:.0f}')} | "
                     f"{cell(r[2], 'home_engine_busy')} |")
        out.append(dict(point=pid, default={k: b.get(k) for k in METRICS + ('status', 'source')},
                        **{f'mode{m}': {k: r[m].get(k) for k in METRICS + ('status', 'source')} for m in (1, 2)}))
    summary = []
    for m in (1, 2):
        if sp[m]:
            summary.append(f'mode {m}: {len(sp[m])} points with a default, geometric-mean speedup '
                           f'{statistics.geometric_mean(sp[m]):.3f}, min {min(sp[m]):.3f}, max {max(sp[m]):.3f}, '
                           f'slower by > 2 %: {sum(1 for x in sp[m] if x < 0.98)}')
    if extra:
        lines += ['', '| Point (rtl, 64 KiB) | default cycles (run-1 image) | ' +
                  ' | '.join(f'{n} cycles | {n} B/cycle/core' for n in extra) + ' |',
                  '|---|---:|' + '---:|---:|' * len(extra)]
        for pid in points:
            b = base.get(pid, {})
            cells = []
            for n, e in extra.items():
                r = e.get(pid, {})
                cells += [f"{r['cycles']:,}" if r.get('cycles') else (r.get('status') or '–'),
                          f"{r['per_core']:.2f}" if r.get('per_core') else '–']
            lines.append(f"| {pid.split('/')[0][3:]} {pid.split('/')[2].split('-r')[0]} | "
                         f"{b['cycles']:,} | " if b.get('cycles') else f"| {pid} | – | ")
            lines[-1] += ' | '.join(cells) + ' |'
            for n, e in extra.items():
                r = e.get(pid, {})
                if r.get('cycles'):
                    out.append(dict(point=pid, arm=n, **{k: r.get(k) for k in METRICS + ('status', 'source')}))
    text = '\n'.join(lines + [''] + summary) + '\n'
    print(text)
    if a.write:
        (G4/'rtl_table.md').write_text(text)
        (G4/'rtl_table.json').write_text(json.dumps(dict(domain='rtl-simulation', board=False, rows=out,
                                                         summary=summary), indent=1) + '\n')


if __name__ == '__main__':
    main()
