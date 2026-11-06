#!/usr/bin/env python3
"""Summarise the packed-MMA (m.mma.dp) WHAT-IF runs.

    summarize.py --runs RUNS_DIR --out summary.json [--markdown tables.md]

RUNS_DIR/<set>-<config>/results.json are run_graphs.py outputs: set c (the
main what-if matrix, configs/<config>.cfg), s (sensitivity levers, see
README) and i (default-identity controls). Per (point, configuration) this reads
the measured (steady) call's device cycles and the model counters of that call
alone: the final snapshot minus counters_untimed_cold.json (taken when timing
switched on after the untimed cold call).

Every number is an UNCALIBRATED model prediction of a hypothetical opcode
(R40 holdout median error 21.5 %); compare ratios, not absolute cycles.
"""
import argparse
import json
from pathlib import Path
import re

BASE = 'c-g1-pf0'
CONFIG_LABEL = {
    'c-g1-pf0': 'G=1 (today)',
    'c-g1-pf1': 'G=1 + B prefetch',
    'c-g2g4-pf0': 'G=2/4',
    'c-g2g4-pf1': 'G=2/4 + B prefetch',
    'c-g2g4-pf0-pipe6': 'G=2/4, pipe+2',
    'c-g2g4-pf1-pipe6': 'G=2/4 + B prefetch, pipe+2',
}
# Per-core time counters (cycles; several overlap, see README).
CORE_TIME = ('matrix_engine_busy_cycles', 'matrix_engine_wait_cycles', 'barrier_sleep_cycles', 'dependency_wait_cycles',
             'lsu_wait_cycles', 'pipeline_freeze_cycles', 'context_park_cycles', 'fence_drain_cycles',
             'arrive_drain_cycles', 'dma_admission_wait_cycles', 'memory_port_service_cycles',
             'memory_port_queue_cycles', 'issue_service_cycles', 'issue_queue_cycles', 'branch_refetch_cycles',
             'sfu_queue_cycles', 'vrf_bank_conflict_cycles')
CORE_COUNT = ('instructions', 'mma_commands', 'matrix_engine_commands', 'matrix_dp_whatif_commands', 'class_lsu',
              'class_mma', 'class_barrier', 'class_salu', 'class_vint', 'class_vfp', 'class_fence', 'class_desc',
              'dma_read_words', 'dma_write_words', 'dma_commands_issued')
HOME = ('engine_service_cycles', 'engine_queue_cycles', 'hits', 'misses', 'merges', 'mshr_full_events',
        'resolution_cycles', 'writebacks', 'evictions', 'set_busy_events')
OUTLET = ('service_cycles', 'queue_cycles', 'reads', 'writes', 'wait_cycles')


def flat(path):
    if not path.exists():
        return None
    c = json.loads(path.read_text())
    return c.get('counters', c) if isinstance(c, dict) else {}


def aggregate(c):
    """Sum per-unit counters: core_N_x -> core x, home_N_x -> home x, outlet_N_x, pc_N_bytes."""
    out = {}
    for k, v in c.items():
        if not isinstance(v, (int, float)):
            continue
        m = re.match(r'^(core|home|outlet|pc)_(\d+)_(.*)$', k)
        name = f'{m.group(1)}.{m.group(3)}' if m else k
        if m and m.group(1) == 'core' and 'peak' in m.group(3):
            out[name] = max(out.get(name, 0), v)
        else:
            out[name] = out.get(name, 0) + v
    return out


def per_core(c, cores):
    out = {}
    for k, v in c.items():
        m = re.match(r'^core_(\d+)_matrix_engine_busy_cycles$', k)
        if m and int(m.group(1)) < cores:
            out[int(m.group(1))] = v
    return out


def one(row, config):
    rec = json.loads(Path(row['record']).read_text()) if row.get('record') else {}
    model = (rec.get('systemc_model') or [{}])[-1]
    mdir = Path(model['model_dir']) if model.get('model_dir') else None
    final = flat(mdir/'counters.json') if mdir else None
    cold = flat(mdir/'counters_untimed_cold.json') if mdir else None
    delta = None
    if final is not None:
        cold = cold or {}
        delta = {k: v - cold.get(k, 0) for k, v in final.items() if isinstance(v, (int, float))
                 and not k.endswith(('peak_outstanding', 'peak_mshrs', 'peak_units', 'peak_tags'))}
    t = rec.get('time') or {}
    cores = int((rec.get('platform') or {}).get('knobs', {}).get('TNCA_ACTIVE_CORES', 1))
    dev = t.get('device')
    out = dict(point=row['point'], config=config, status=row.get('status'), max_rel_err=row.get('max_rel_err'),
               tol=row.get('tol'), device_cycles=dev, graph_cycles=(rec.get('model_graph') or {}).get('graph_cycles'),
               launches=[dict(kernel=p.get('kernel'), grid=p.get('grid'), cycles=p.get('cycles'))
                         for p in t.get('per_launch') or []],
               cores=cores, model_dir=str(mdir) if mdir else None, process_wall_s=row.get('process_wall_s'),
               model_launch_host_s=row.get('model_launch_host_s'), timing_status=row.get('model_timing_status'),
               knobs=row.get('knobs'))
    if delta is not None and dev:
        agg = aggregate(delta)
        busy = agg.get('core.matrix_engine_busy_cycles', 0)
        pc = per_core(delta, cores)
        out.update(counters=agg, matrix_engine_busy_cycles=busy,
                   matrix_busy_share=busy / (cores * dev),
                   matrix_busy_share_max_core=max(pc.values()) / dev if pc else None,
                   per_core_fraction={k.split('.', 1)[1]: agg.get(k, 0) / (cores * dev)
                                      for k in (f'core.{n}' for n in CORE_TIME)},
                   home_fraction={n: agg.get(f'home.{n}', 0) / (cores * dev) for n in HOME},
                   outlet_fraction={n: agg.get(f'outlet.{n}', 0) / (cores * dev) for n in OUTLET},
                   counts={n: agg.get(f'core.{n}', 0) for n in CORE_COUNT},
                   core_cycles_per_mma=(cores * dev / agg['core.mma_commands']) if agg.get('core.mma_commands') else None,
                   busy_per_mma=(busy / agg['core.mma_commands']) if agg.get('core.mma_commands') else None,
                   pc_bytes=agg.get('pc.bytes'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--markdown', type=Path)
    ap.add_argument('--prefixes', default='c-,s-,i-', help='run-set directory prefixes to read')
    a = ap.parse_args()
    rows = []
    for d in sorted(a.runs.iterdir()):
        if not (d.is_dir() and d.name.startswith(tuple(a.prefixes.split(','))) and (d/'results.json').exists()):
            continue
        config = d.name
        for r in json.loads((d/'results.json').read_text())['points']:
            rows.append(one(r, config))
    base = {r['point']: r for r in rows if r['config'] == BASE}
    for r in rows:
        b = base.get(r['point'])
        if b and b.get('device_cycles') and r.get('device_cycles'):
            r['speedup_vs_g1'] = b['device_cycles'] / r['device_cycles']
            if b.get('matrix_engine_busy_cycles') and r.get('matrix_engine_busy_cycles'):
                r['busy_ratio_vs_g1'] = b['matrix_engine_busy_cycles'] / r['matrix_engine_busy_cycles']
                # Amdahl ceiling: only the busy cycles shrink, all else stays.
                s = b['matrix_busy_share']
                r['amdahl_ceiling'] = 1 / ((1 - s) + s / r['busy_ratio_vs_g1'])
    a.out.write_text(json.dumps(dict(
        schema='tnca-pmac-whatif-summary/1',
        label='WHAT-IF: uncalibrated SystemC/TLM model timing of a hypothetical packed MMA (m.mma.dp, not built); '
              'numerics are the existing per-K path; R40 holdout median error 21.5 %: compare ratios only',
        rows=rows), indent=1, default=str) + '\n')
    if a.markdown:
        lines = []
        for point in dict.fromkeys(r['point'] for r in rows):
            rs = sorted((r for r in rows if r['point'] == point and r['config'] in CONFIG_LABEL), key=lambda r: list(CONFIG_LABEL).index(r['config'])
                        if r['config'] in CONFIG_LABEL else 99)
            lines += [f'#### {point}', '',
                      '| configuration | status | device cycles | speedup | MMA busy cycles | busy share | busy ratio | Amdahl ceiling |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for r in rs:
                lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    CONFIG_LABEL.get(r['config'], r['config']), r['status'],
                    f"{r['device_cycles']:,}" if r.get('device_cycles') else '-',
                    f"{r['speedup_vs_g1']:.3f}x" if r.get('speedup_vs_g1') else '-',
                    f"{r['matrix_engine_busy_cycles']:,}" if r.get('matrix_engine_busy_cycles') is not None else '-',
                    f"{100 * r['matrix_busy_share']:.1f} %" if r.get('matrix_busy_share') is not None else '-',
                    f"{r['busy_ratio_vs_g1']:.2f}x" if r.get('busy_ratio_vs_g1') else '-',
                    f"{r['amdahl_ceiling']:.3f}x" if r.get('amdahl_ceiling') else '-'))
            lines.append('')
        a.markdown.write_text('\n'.join(lines) + '\n')
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
