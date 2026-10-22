#!/usr/bin/env python3
"""Markdown tables for README.md from summary.json (summarize.py).

    report_tables.py summary.json > tables.md

WHAT-IF only: uncalibrated model timing of a hypothetical opcode.
"""
import json
import sys

MAIN = ['c-g1-pf0', 'c-g1-pf1', 'c-g2g4-pf0', 'c-g2g4-pf1', 'c-g2g4-pf0-pipe6', 'c-g2g4-pf1-pipe6']
SHORT = {'c-g1-pf0': 'G=1', 'c-g1-pf1': 'G=1 pf', 'c-g2g4-pf0': 'G=2/4', 'c-g2g4-pf1': 'G=2/4 pf',
         'c-g2g4-pf0-pipe6': 'G=2/4 pipe+2', 'c-g2g4-pf1-pipe6': 'G=2/4 pf pipe+2'}
LEVERS = [('dma4', 'transfer engine 4x deeper'), ('home2', 'home slices ~2x faster'),
          ('t64', 'tile 16,64,16 (A reuse x4)'), ('ring', 'producer/consumer ring (RCTX_ROLES=2, 3 stages)'),
          ('best', 'all four together')]
POINT_ORDER = ['op.w8a8_linear/hbm/bf16-m1-k2048-n6144', 'op.w8a8_linear/hbm/bf16-m1-k2560-n9216',
               'op.w8a8_linear/hbm/bf16-m1-k9216-n2560', 'op.w8a8_linear/hbm/bf16-m128-k2048-n6144',
               'op.w8a8_linear/hbm/bf16-m128-k2560-n9216', 'op.w8a8_linear/hbm/bf16-m128-k9216-n2560',
               'op.bf16_linear/hbm/bf16-m1-k2048-n6144', 'op.bf16_linear/hbm/bf16-m1-k2560-n9216',
               'op.bf16_linear/hbm/bf16-m1-k9216-n2560', 'op.bf16_linear/hbm/bf16-m128-k2048-n6144',
               'op.bf16_linear/hbm/bf16-m128-k2560-n9216', 'op.bf16_linear/hbm/bf16-m128-k9216-n2560',
               'op.matmul/large/fp16-m2048-k2048-n2048', 'op.matmul/medium/fp16-m128-k192-n128',
               'op.attention_prefill/large/fp16-b1-h32-s512-d64', 'op.attention_prefill/medium/fp16-b1-h4-s96-d32']


def fmt_c(v):
    return f'{v:,}' if isinstance(v, int) else '-'


def main():
    rows = json.load(open(sys.argv[1]))['rows']
    by = {(r['point'], r['config']): r for r in rows}
    points = [p for p in POINT_ORDER if (p, 'c-g1-pf0') in by]
    out = []
    # 1. Overview: speedup of every configuration, one row per point.
    out += ['| point | MMA cmds | busy share at G=1 | ' + ' | '.join(SHORT[c] for c in MAIN[1:]) + ' |',
            '|---|---:|---:|' + '---:|' * (len(MAIN) - 1)]
    for p in points:
        b = by[(p, 'c-g1-pf0')]
        cells = []
        for c in MAIN[1:]:
            r = by.get((p, c))
            cells.append(f"{r['speedup_vs_g1']:.3f}x" if r and r.get('speedup_vs_g1') else '-')
        out.append(f"| `{p}` | {b['counts']['mma_commands']:,} | {100 * b['matrix_busy_share']:.1f} % | " +
                   ' | '.join(cells) + ' |')
    out.append('')
    # 2. Per point: cycles, busy, share, speedup.
    for p in points:
        out += [f'#### `{p}`', '',
                '| configuration | device cycles | speedup | MMA busy cycles (3 cores) | busy / MMA | busy share | busy ratio | Amdahl ceiling |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for c in MAIN:
            r = by.get((p, c))
            if not r:
                continue
            out.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                SHORT[c], fmt_c(r.get('device_cycles')),
                f"{r['speedup_vs_g1']:.3f}x" if r.get('speedup_vs_g1') else '-',
                fmt_c(r.get('matrix_engine_busy_cycles')),
                f"{r['busy_per_mma']:.0f}" if r.get('busy_per_mma') else '-',
                f"{100 * r['matrix_busy_share']:.1f} %" if r.get('matrix_busy_share') is not None else '-',
                f"{r['busy_ratio_vs_g1']:.2f}x" if r.get('busy_ratio_vs_g1') else '-',
                f"{r['amdahl_ceiling']:.2f}x" if r.get('amdahl_ceiling') else '-'))
        out.append('')
    # 3. Where the time goes (per core-cycle; overlapping, not additive).
    keys = [('matrix_engine_busy_cycles', 'MMA busy'), ('dma_admission_wait_cycles', 'DMA admission wait'),
            ('barrier_sleep_cycles', 'barrier sleep'), ('dependency_wait_cycles', 'dependency wait'),
            ('lsu_wait_cycles', 'LSU wait'), ('pipeline_freeze_cycles', 'pipeline freeze'),
            ('memory_port_service_cycles', 'core memory port busy')]
    out += ['| point | config | core-cycles / MMA | ' + ' | '.join(k[1] for k in keys) +
            ' | home engine busy | home queue | outlet busy | HBM bytes / call |',
            '|---|---|---:|' + '---:|' * (len(keys) + 4)]
    for p in points:
        for c in ('c-g1-pf0', 'c-g2g4-pf1'):
            r = by.get((p, c))
            if not r or not r.get('per_core_fraction'):
                continue
            f = r['per_core_fraction']
            out.append(f"| `{p}` | {SHORT[c]} | {r['core_cycles_per_mma']:.0f} | " if r.get('core_cycles_per_mma') else
                       f"| `{p}` | {SHORT[c]} | - | ")
            out[-1] += ' | '.join(f"{f.get(k, 0):.2f}" for k, _ in keys) + \
                f" | {r['home_fraction']['engine_service_cycles']:.2f} | {r['home_fraction']['engine_queue_cycles']:.2f}" \
                f" | {r['outlet_fraction']['service_cycles']:.2f} | {r.get('pc_bytes') or 0:,} |"
    out.append('')
    # 4. Sensitivity levers.
    spoints = [p for p in points if (p, 's-dma4-g1') in by or (p, 's-t64-g1') in by]
    if spoints:
        out += ['| point | lever | G=1 cycles | vs today | G=2/4 pf cycles | vs today | packed gain within lever | busy share G=1 -> G=2/4 pf |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
        for p in spoints:
            b = by[(p, 'c-g1-pf0')]
            g = by.get((p, 'c-g2g4-pf1'))
            out.append(f"| `{p}` | today | {b['device_cycles']:,} | 1.000x | {g['device_cycles']:,} | "
                       f"{b['device_cycles'] / g['device_cycles']:.3f}x | {b['device_cycles'] / g['device_cycles']:.3f}x | "
                       f"{100 * b['matrix_busy_share']:.0f} % -> {100 * g['matrix_busy_share']:.0f} % |")
            for lever, label in LEVERS:
                r1, r2 = by.get((p, f's-{lever}-g1')), by.get((p, f's-{lever}-g2g4-pf1'))
                if not r1 or not r1.get('device_cycles'):
                    out.append(f'| `{p}` | {label} | {r1["status"] if r1 else "-"} | | | | | |')
                    continue
                c1 = r1['device_cycles']
                c2 = r2.get('device_cycles') if r2 else None
                out.append(f"| `{p}` | {label} | {c1:,} | {b['device_cycles'] / c1:.3f}x | "
                           f"{fmt_c(c2)} | {(b['device_cycles'] / c2) if c2 else 0:.3f}x | "
                           f"{(c1 / c2) if c2 else 0:.3f}x | "
                           f"{100 * r1['matrix_busy_share']:.0f} % -> {100 * r2['matrix_busy_share']:.0f} % |"
                           if c2 else f"| `{p}` | {label} | {c1:,} | {b['device_cycles'] / c1:.3f}x | - | - | - | - |")
        out.append('')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
