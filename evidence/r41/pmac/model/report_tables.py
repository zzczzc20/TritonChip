#!/usr/bin/env python3
"""Markdown tables for README.md from summary.json (summarize.py).

    report_tables.py summary.json > tables.md

WHAT-IF only: uncalibrated model timing of a hypothetical opcode. Reduced
scope (2026-10-22): FP8 W8A8 and BF16 linears, M=1/128, K x N = 2048x6144 and
2560x9216. Primary what-if: lane V3 (E4M3xE4M3 at G=4, lane latency +1, so
mtx_pipe_cycles 4 -> 5); 16-bit G=2 (lane V2, >= +2 registers) is secondary.
"""
import json
import sys

POINTS = ['op.w8a8_linear/hbm/bf16-m1-k2048-n6144', 'op.w8a8_linear/hbm/bf16-m1-k2560-n9216',
          'op.w8a8_linear/hbm/bf16-m128-k2048-n6144', 'op.w8a8_linear/hbm/bf16-m128-k2560-n9216',
          'op.bf16_linear/hbm/bf16-m1-k2048-n6144', 'op.bf16_linear/hbm/bf16-m1-k2560-n9216',
          'op.bf16_linear/hbm/bf16-m128-k2048-n6144', 'op.bf16_linear/hbm/bf16-m128-k2560-n9216']
BASE = 'c-g1-pf0'
# (run set, label for FP8 points, label for BF16 points); None = not shown for that class.
CONFIGS = [
    ('c-g1-pf0', 'G=1 (today)', 'G=1 (today)'),
    ('c-g1-pf1', 'G=1, B prefetch', 'G=1, B prefetch'),
    ('c-g4-pf0-pipe5', '**V3: G=4, pipe+1**', 'V3 image, BF16 stays G=1 (pipe+1 only)'),
    ('c-g4-pf1-pipe5', '**V3: G=4, B prefetch, pipe+1**', 'V3 image + B prefetch, BF16 stays G=1'),
    ('c-g2g4-pf0', 'G=4, pipe+0 (lower bound on pipe cost)', 'G=2, pipe+0 (optimistic)'),
    ('c-g2g4-pf1', 'G=4, B prefetch, pipe+0', 'G=2, B prefetch, pipe+0 (optimistic)'),
    ('c-g2g4-pf0-pipe6', None, 'secondary V2: G=2, pipe+2'),
    ('c-g2g4-pf1-pipe6', None, 'secondary V2: G=2, B prefetch, pipe+2'),
]
COUNTERS = [('matrix_engine_busy_cycles', 'MMA busy'), ('matrix_engine_wait_cycles', 'MMA wait (exposed)'),
            ('dependency_wait_cycles', 'dependency wait'), ('lsu_wait_cycles', 'LSU wait'),
            ('pipeline_freeze_cycles', 'pipeline freeze'), ('barrier_sleep_cycles', 'barrier sleep'),
            ('dma_admission_wait_cycles', 'DMA admission wait')]


def c(v):
    return f'{v:,}' if isinstance(v, int) else '-'


def main():
    rows = json.load(open(sys.argv[1]))['rows']
    by = {(r['point'], r['config']): r for r in rows}
    out = []
    # Overview.
    out += ['| point | G=1 cycles | MMA busy share | MMA wait share | V3 G=4 | V3 G=4 + pf | G=1 + pf | 16-bit G=2 pipe+2 | 16-bit G=2 + pf pipe+2 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in POINTS:
        b = by.get((p, BASE))
        if not b:
            continue
        fp8 = 'w8a8' in p

        def sp(cfg):
            r = by.get((p, cfg))
            return f"{r['speedup_vs_g1']:.3f}x" if r and r.get('speedup_vs_g1') else '-'
        out.append(f"| `{p.split('/', 1)[0][3:]} {p.rsplit('/', 1)[1][5:]}` | {b['device_cycles']:,} | "
                   f"{100 * b['matrix_busy_share']:.1f} % | {100 * b['per_core_fraction']['matrix_engine_wait_cycles']:.1f} % | "
                   f"{sp('c-g4-pf0-pipe5') if fp8 else 'n/a'} | {sp('c-g4-pf1-pipe5') if fp8 else 'n/a'} | {sp('c-g1-pf1')} | "
                   f"{'n/a' if fp8 else sp('c-g2g4-pf0-pipe6')} | {'n/a' if fp8 else sp('c-g2g4-pf1-pipe6')} |")
    out.append('')
    # Per point.
    for p in POINTS:
        if (p, BASE) not in by:
            continue
        fp8 = 'w8a8' in p
        out += [f'#### `{p}`', '',
                '| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cfg, l8, l16 in CONFIGS:
            label = l8 if fp8 else l16
            r = by.get((p, cfg))
            if not label or not r:
                continue
            if not r.get('device_cycles'):
                out.append(f"| {label} | {r.get('status')} | | | | | | | |")
                continue
            out.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                label, c(r['device_cycles']),
                f"{r['speedup_vs_g1']:.3f}x" if r.get('speedup_vs_g1') else '-',
                c(r.get('matrix_engine_busy_cycles')),
                f"{r['busy_per_mma']:.0f}" if r.get('busy_per_mma') else '-',
                f"{100 * r['matrix_busy_share']:.1f} %",
                f"{r['busy_ratio_vs_g1']:.2f}x" if r.get('busy_ratio_vs_g1') else '-',
                f"{100 * r['per_core_fraction']['matrix_engine_wait_cycles']:.1f} %",
                f"{r['amdahl_ceiling']:.2f}x" if r.get('amdahl_ceiling') else '-'))
        out.append('')
    # Where the time goes: G=1 against the primary what-if (fraction of core-cycles; overlapping).
    out += ['| point | config | ' + ' | '.join(k[1] for k in COUNTERS) +
            ' | home queue | LSU instr / MMA | core-cycles / MMA |',
            '|---|---|' + '---:|' * (len(COUNTERS) + 3)]
    for p in POINTS:
        fp8 = 'w8a8' in p
        for cfg in (BASE, 'c-g4-pf1-pipe5' if fp8 else 'c-g2g4-pf1-pipe6'):
            r = by.get((p, cfg))
            if not r or not r.get('per_core_fraction'):
                continue
            f = r['per_core_fraction']
            n = r['counts']
            label = 'G=1' if cfg == BASE else ('V3 G=4 pf' if fp8 else 'G=2 pf pipe+2')
            out.append(f"| `{p.split('/', 1)[0][3:]} {p.rsplit('/', 1)[1][5:]}` | {label} | " +
                       ' | '.join(f'{f.get(k, 0):.3f}' for k, _ in COUNTERS) +
                       f" | {r['home_fraction']['engine_queue_cycles']:.2f} | {n['class_lsu'] / n['mma_commands']:.1f} | "
                       f"{r['core_cycles_per_mma']:.0f} |")
    out.append('')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
