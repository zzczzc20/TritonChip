#!/usr/bin/env python3
"""Per-point cycle tables from the recorded runs in records/ (computes every number from the raw files).

    python3 evidence/r42/copy/tables.py            # single-core RTL screens and TLM predictions
    python3 evidence/r42/copy/tables.py census     # static instruction counts per point

Domains: `rtl` = single-core RTL simulation (profile core-vrf-sync-v5-x8-g2c4-dp, a screen, never a board number);
`tlm` = the SystemC/TLM model of the three-core image (tools/r42/kfuse_speed_tlm_run.py, calibrated config cal.cfg), a
model prediction; `census` = instructions in the assembly of every kernel of a point (code size, both sides of a branch).
"""
import json
import math
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / 'records'


def rtl(name):
    out = {}
    for line in (R / f'{name}.jsonl').open():
        r = json.loads(line)
        out[r['point_id']] = (r['status'], (r.get('time') or {}).get('device'),
                              (r.get('correctness') or {}).get('max_rel_err'), r.get('counters') or {})
    return out


def tlm(name):
    out = {}
    for line in (R / f'{name}.jsonl').open():
        r = json.loads(line)
        if 'key' in r:
            out[r['key']] = (r['status'], r.get('device_cycles'), r.get('max_rel_err'), {})
    return out


def table(title, base, cand, extra=''):
    print(f'\n## {title}')
    print(f'{"point":62s} {"base":>10s} {"new":>10s} {"base/new":>9s}  {"issued base":>11s} {"issued new":>10s}  status  err')
    ratios = []
    for p in sorted(cand):
        if p not in base:
            continue
        b, c = base[p], cand[p]
        ib, ic = b[3].get('perf_issued', ''), c[3].get('perf_issued', '')
        if b[1] and c[1] and c[0] == 'pass':
            ratios.append(b[1] / c[1])
            print(f'{p:62s} {b[1]:>10d} {c[1]:>10d} {b[1] / c[1]:>9.3f}  {ib!s:>11s} {ic!s:>10s}  {c[0]:6s}  {c[2]}')
        else:
            print(f'{p:62s} {str(b[1]):>10s} {str(c[1]):>10s} {"-":>9s}  {ib!s:>11s} {ic!s:>10s}  {c[0]:6s}  {c[2]}')
    if ratios:
        print(f'{len(ratios)} points, geometric mean base/new {math.exp(sum(map(math.log, ratios)) / len(ratios)):.3f}')


def census():
    def load(f):
        d = {}
        for line in (R / f).open():
            p = line.split()
            d[p[0]] = int(p[2].split('=')[1])
        return d
    b, n = load('census_base.txt'), load('census_new.txt')
    print(f'{"point":62s} {"base":>7s} {"new":>7s} {"new/base":>9s}')
    for p in sorted(b, key=lambda k: n[k] / b[k]):
        if n[p] != b[p]:
            print(f'{p:62s} {b[p]:>7d} {n[p]:>7d} {n[p] / b[p]:>9.2f}')
    print(f'{sum(1 for p in b if n[p] != b[p])} of {len(b)} points have different code; '
          f'{sum(1 for p in b if n[p] == b[p])} identical')


def prediction():
    """The TLM base/new ratio of each of the 13 points applied to its `ab-int2` board cycles (a model prediction, not a board result)."""
    board = json.load((R / 'board_ab_int2_13points.json').open())['rows']
    base, new = tlm('tlm_base'), tlm('tlm_new')
    proxy = {'op.transpose/hbm/fp32-r8192-c8192': 'op.transpose/large/fp32-r4096-c4096'}
    print(f'{"point":62s} {"board ab-int2":>13s} {"tlm ratio":>9s} {"predicted":>13s}  note')
    logs, n_all = [], 113
    for p, row in sorted(board.items()):
        q = proxy.get(p, p)
        ratio = base[q][1] / new[q][1]
        logs.append(math.log(ratio))
        print(f'{p:62s} {row["ab_int2_cycles"]:>13d} {ratio:>9.2f} {row["ab_int2_cycles"] / ratio:>13.0f}  {"ratio of " + q if q != p else ""}')
    g = math.exp(sum(logs) / n_all)
    half = math.exp(sum(logs) / 2 / n_all)
    print(f'13 points; effect on the 113-point geometric mean if the board follows the model ratios: x{g:.3f}; '
          f'if only half of each log-gain materialises: x{half:.3f}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prediction':
        prediction()
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'census':
        census()
        sys.exit()
    for title, base, cand, kind in (
            ('rtl: embedding, transpose, cumsum, tiny and small (base = R42 head 02b7578e)', 'base_all', 'final_families', rtl),
            ('rtl: tiny tier, all 27 points, final tree', 'base_tiny', 'final_tiny', rtl),
            ('rtl: tiny tier, all 27 points, first candidate tree', 'base_tiny', 'cand_tiny', rtl),
            ('rtl: GEMM epilogue and conv screens', 'gm_copy_base', 'gm_copy_dev', rtl),
            ('rtl: conv2d small', 'cv_copy_base', 'cv_copy_dev', rtl),
            ('tlm: three-core image model, base tree', 'tlm_base', 'tlm_new', tlm)):
        try:
            table(title, kind(base), kind(cand))
        except FileNotFoundError as e:
            print(f'\n## {title}\n(missing {e.filename})')
