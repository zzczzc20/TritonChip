#!/usr/bin/env python3
"""WHAT-IF check of the packed-MMA timing keys (m.mma.dp, hypothetical).

    whatif_matrix_check.py --harness HARNESS_DIR --binary tnca-chip --out FRESH_DIR [--base-config CFG]

Runs validate_matrix.py's single-MMA case (every dtype x K16/32/64, exact
Fraction reference, results checked byte for byte by the model's `expect`)
under the default configuration and under each what-if setting, and checks:

  * numerics: every case still completes with the reference bytes (the packed
    path changes timing only; the per-K arithmetic is untouched);
  * timing: the change in matrix_engine_busy_cycles equals the change of the
    sweep formula below, and matrix_dp_whatif_commands counts the one MMA.

    sweep = A + steps*(t_rf + 16*mac)          (no prefetch)
    sweep = A + t_rf + steps*16*mac            (prefetch: one exposed B read)
    A = a_regs*t_rf if buffered else 0; steps = K/G; mac = t_mac if buffered
    else t_mac_unbuf; buffered = ABUF and (K != 64 for FP8, K == 16 for 16-bit)

The output is a what-if timing check, not a prediction of packed numerics.
"""
import argparse
import json
from pathlib import Path
import random
import sys

T = {}   # mtx_* constants of the base configuration


def config(base, path, **changes):
    values = dict(line.split('#')[0].strip().split('=', 1) for line in base.read_text().splitlines()
                  if '=' in line.split('#')[0])
    values.update({k: str(v) for k, v in changes.items()})
    path.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
    return path


def sweep(dtype, K, g16=1, g8=1, pf=0):
    T_RF, T_MAC, T_UNBUF, abuf = T['mtx_rf_read_cycles'], T['mtx_mac_cycles'], T['mtx_mac_unbuffered_cycles'], T['mtx_abuf']
    nbytes = 1 if dtype >= 2 else 2
    a_regs = (16 * K * nbytes + 63) // 64
    buffered = bool(abuf) and (K != 64 if nbytes == 1 else K == 16)
    mac = 16 * (T_MAC if buffered else T_UNBUF)
    a_term = a_regs * T_RF if buffered else 0
    g = g8 if nbytes == 1 else g16
    steps = K // g
    return a_term + (T_RF + steps * mac if pf else steps * (T_RF + mac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harness', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--base-config', type=Path, help='default: HARNESS/configs/r40-board-m2.cfg')
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness))
    import validate_matrix as vm
    base_cfg = a.base_config or a.harness/'configs/r40-board-m2.cfg'
    vals = dict(line.split('#')[0].strip().split('=', 1) for line in base_cfg.read_text().splitlines()
                if '=' in line.split('#')[0])
    T.update({k: int(v) for k, v in vals.items() if k.startswith('mtx_')})
    a.out.mkdir(parents=True, exist_ok=False)
    settings = {'default': {}, 'keys-at-default': dict(mtx_dp_group_16=1, mtx_dp_group_8=1, mtx_dp_b_prefetch=0),
                'g1-pf1': dict(mtx_dp_b_prefetch=1),
                'g2g4-pf0': dict(mtx_dp_group_16=2, mtx_dp_group_8=4),
                'g2g4-pf1': dict(mtx_dp_group_16=2, mtx_dp_group_8=4, mtx_dp_b_prefetch=1),
                'g2g2-pf0': dict(mtx_dp_group_16=2, mtx_dp_group_8=2)}
    rows, ok = [], True
    base = {}
    for name, keys in settings.items():
        d = a.out/name
        d.mkdir()
        cfg = config(base_cfg, d/'chip.cfg', trace_events=10000, **keys)
        for dtype in range(4):
            for K in (16, 32, 64):
                case = f'mma-{vm.DT[dtype]}-k{K}'
                rng = random.Random(9071 + 97 * dtype + K)      # same operands for every setting
                r = vm.mma_case(d, a.binary, cfg, case, dtype, K, rng)   # asserts the reference bytes
                summary = json.loads((d/case/'summary.json').read_text())['counters']
                dp = summary.get('core_0_matrix_dp_whatif_commands')
                if name == 'default':
                    base[case] = r['matrix_engine_busy_cycles']
                g16, g8, pf = keys.get('mtx_dp_group_16', 1), keys.get('mtx_dp_group_8', 1), keys.get('mtx_dp_b_prefetch', 0)
                want_delta = sweep(dtype, K, g16, g8, pf) - sweep(dtype, K)
                got_delta = r['matrix_engine_busy_cycles'] - base[case]
                active = g16 > 1 or g8 > 1 or pf
                good = got_delta == want_delta and (dp == 1 if active else dp is None)
                ok &= good
                rows.append(dict(setting=name, case=case, dtype=vm.DT[dtype], K=K, numerics='reference bytes matched',
                                 mma_sweep_formula=sweep(dtype, K, g16, g8, pf), mma_busy_formula=sweep(dtype, K, g16, g8, pf) + T['mtx_pipe_cycles'] + 1 + T['mtx_complete_cycles'],
                                 busy_delta_model=got_delta, busy_delta_formula=want_delta,
                                 matrix_dp_whatif_commands=dp, ok=good, visible_cycles=r['cycles']))
    report = dict(schema='tnca-pmac-whatif-matrix-check/1', status='passed' if ok else 'FAILED',
                  base_config=str(base_cfg), constants=T,
                  label='WHAT-IF timing only (m.mma.dp is not built); numerics are the unchanged per-K path',
                  rows=rows)
    (a.out/'whatif_matrix_check.json').write_text(json.dumps(report, indent=1) + '\n')
    for r in rows:
        print(f"{r['setting']:16s} {r['case']:14s} sweep {r['mma_sweep_formula']:5d} busy-delta model {r['busy_delta_model']:6d} "
              f"formula {r['busy_delta_formula']:6d} dp {r['matrix_dp_whatif_commands']} {'ok' if r['ok'] else 'FAIL'}")
    print(report['status'])
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
