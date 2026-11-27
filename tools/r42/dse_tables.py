#!/usr/bin/env python3
"""Regenerate evidence/r42/dse/tables.md from the raw collected files (domain: SystemC/TLM model against board and single-core RTL).

    python3 tools/r42/dse_tables.py > evidence/r42/dse/tables.md

Every number is computed here from files under evidence/r42/dse/col (model runs), the board records of evidence/r42/board
(through dse_common.board) and the single-core RTL cycles printed in evidence/r42/xfer/tables.txt section 3b.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C   # noqa: E402

COL = C.ROOT / 'evidence/r42/dse/col'
GEMM = {'matmul', 'linear', 'mlp', 'gemv', 'bmm', 'gpt2_block', 'llama_prefill', 'llama_decode'}
EXC = {'embedding', 'transpose', 'cumsum'}   # lowered differently when the RTL numbers were taken (before R42-copy)


def rtl_1c():
    for p in (Path('/home/ubuntu/TritonChip/evidence/r42/xfer/tables.txt'), C.ROOT / 'evidence/r42/xfer/tables.txt'):
        if p.exists():
            out = {}
            for l in p.read_text().splitlines():
                m = re.match(r'\s+(\S+/tiny/\S+)\s+pass\s+(\d+)', l)
                if m:
                    out[m.group(1)] = int(m.group(2))
            return out
    raise FileNotFoundError('xfer tables')


def table_1c():
    rtl = rtl_1c()
    rows = [('`ideal` (R41 G4 model, zero-latency memory, 1 core)', 'v1_tiny_1c_ideal'), ('+ hybrid LSU (`lsu_hybrid`)', 'v2_tiny_1c_ideal_hy'),
            ('+ every vector-datapath instruction takes 2 issue cycles (`issue_vector_cycles=2`)', 't27_iv2'),
            ('vector integer only 2 (`issue_vint_cycles`)', 'cls_c1'), ('vint + vfp 2', 'cls_c2'), ('vint + shuffle 2', 'cls_c5'),
            ('**all vector 2, shuffle 1 = DSE base**', 'cls_c8'), ('all vector 2, vfp and shuffle 1', 'cls_c7'),
            ('`issue_cycles=2` for every instruction', 't27_is2'), ('all vector 2 and taken branch 6', 't27_iv2_br6')]
    print('| single-core TLM variant (zero-latency memory) | n | geo TLM/RTL | median abs err | gemm-shaped geo | other geo |')
    print('|---|--:|--:|--:|--:|--:|')
    for n, f in rows:
        c = json.loads((COL / f'{f}.json').read_text())['points']
        a, g, o = [], [], []
        for p, v in rtl.items():
            if p in c and c[p].get('device_cycles') and C.family_of(p) not in EXC:
                r = c[p]['device_cycles'] / v
                a.append(r)
                (g if C.family_of(p) in GEMM else o).append(r)
        med = sorted(abs(i - 1) for i in a)[len(a) // 2]
        print(f'| {n} | {len(a)} | {C.geo(a):.3f} | {med:.3f} | {C.geo(g):.3f} | {C.geo(o):.3f} |')


def table_family_tsm():
    B = C.board()

    def fam(f):
        out = {}
        for p, m in C.load_collected(COL / f).items():
            if m.get('device_cycles'):
                out.setdefault(B[p]['family'], []).append(m['device_cycles'] / B[p]['r2'])
        return out
    v1, v3 = fam('v1_tsm_cal.json'), fam('val_dse_base_tsm.json')
    print('| family | n (tiny/small/medium) | first validation geo model/board (`cal`) | DSE base geo model/board |')
    print('|---|--:|--:|--:|')
    for f in sorted(v1):
        print(f'| {f} | {len(v1[f])} | {C.geo(v1[f]):.3f} | {C.geo(v3[f]):.3f} |')
    a1 = [x for v in v1.values() for x in v]
    a3 = [x for v in v3.values() for x in v]
    for n, a in (('first', a1), ('base', a3)):
        e = [abs(x - 1) for x in a]
        print(f'\n{n}: n={len(a)} geo {C.geo(a):.3f} median abs err {sorted(e)[len(e) // 2]:.3f} within 20 %: {sum(x <= .2 for x in e)}')


def table_all():
    d = json.loads((C.ROOT / 'evidence/r42/dse/validation_all_val_dse_base.json').read_text())
    print('| family | n | direct/proxy/wave | geo model/board | median abs err | max abs err | within 20 % | verdict |')
    print('|---|--:|--:|--:|--:|--:|--:|---|')
    for f, v in sorted(d['by_family'].items()):
        print(f'| {f} | {v["n"]} | {"/".join(map(str, v["kinds"]))} | {v["geo"]:.3f} | {v["median_abs_err"]:.3f} | {v["max_abs_err"]:.3f} | {v["within20"]} | {"usable" if v["usable"] else "NOT USABLE"} |')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1c', 'tsm', 'all', 'screen', 'full']
    for w in which:
        print(f'\n### {w}\n')
        if w == '1c':
            table_1c()
        elif w == 'tsm':
            table_family_tsm()
        elif w == 'all':
            table_all()
        elif w == 'screen':
            print(subprocess.run([sys.executable, str(Path(__file__).with_name('dse_report.py')), 'screen'], capture_output=True, text=True).stdout)
