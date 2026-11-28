#!/usr/bin/env python3
"""core_hybrid_clk.py -- cycle-identity and function check of the R42-clk CLKFIX bits on the hybrid-LSU core bench (tb_core_hybrid, RTL simulation).

  core_hybrid_clk.py [--fixes 0,1] [--progs a,b,...] [--jobs 3] [--out DIR] [--extra -Ptb_core_hybrid.X=Y ...]

For every CLKFIX value one vvp is built (tnca_core on the line-wide port: LSU_TAG_W 8, HAS_TENSOR_XPOSE 2, TENSOR_LINE 1, memory latency 6 with random stalls and
reordering, as the line-eligibility core check); every program of the list runs on each and the table `prog  PASS/FAIL cycles per fix` is printed.
The first fix of the list is the reference: a later fix may take more cycles, the programs' own self-checks decide function (PASS). Run it from the tree root.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGS = ('t00_scalar', 't01_vector', 't02_mem', 't03_fp', 't06_alu', 't12_tensor', 't16_vecadd', 't20_sfu', 't21_cvt', 't22_atomic', 't25_mmasf',
         't30_xpose', 't33_tdesc_load', 't34_tdesc_store', 't36_reduce', 't37_lsu_straddle', 't38_wbfreeze', 't39_freeze_drop', 't40_f2i')   # t41_mmadp/t42_mmaf32/t42_mmaovl need MAC_DP/MAC_F32 and fail on the unchanged bench: pass them with --extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixes', default='0,1')
    ap.add_argument('--progs', default=','.join(PROGS))
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--base', action='store_true', help='tree without CLKFIX (R42 head): the parameter is not passed; give --fixes 0')
    ap.add_argument('--out', default=None)
    ap.add_argument('--extra', nargs='*', default=[])
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / 'tools/asm'))
    from tnca_asm import Assembler, build_image
    fixes = [int(x) for x in a.fixes.split(',')]
    progs = a.progs.split(',')
    tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix='clkcore-'))
    tmp.mkdir(parents=True, exist_ok=True)
    vvps = {}
    for fx in fixes:
        v = tmp / f'core_fx{fx}.vvp'
        cmd = ['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/common', '-y', 'rtl/core', '-y', 'rtl/gen', '-y', 'tb/common', '-s', 'tb_core_hybrid',
               '-Ptb_core_hybrid.MEM_LATENCY=6', '-Ptb_core_hybrid.RANDOM_STALL=1', '-Ptb_core_hybrid.MEM_REORDER=1', '-Ptb_core_hybrid.LSU_TAG_W=8',
               '-Ptb_core_hybrid.HAS_TENSOR_XPOSE=2', '-Ptb_core_hybrid.TENSOR_LINE=1', ] + ([] if a.base else [f'-Ptb_core_hybrid.CLKFIX={fx}']) + a.extra + ['-o', str(v), 'tb/core/tb_core_hybrid.v']
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=600)
        vvps[fx] = v
    for p in progs:
        h = tmp / f'{p}.hex'
        h.write_text(''.join(f'{w:016x}\n' for w in build_image(Assembler().assemble((ROOT / f'tb/core/progs/{p}.s').read_text()), size_words=65536)))

    def run(job):
        p, fx = job
        r = subprocess.run(['vvp', str(vvps[fx]), f'+hex={tmp / (p + ".hex")}', '+maxcycles=2000000'], cwd=ROOT, capture_output=True, text=True, timeout=3000).stdout
        ok = re.search(r'^PASS', r, re.M) is not None and 'FAIL' not in r and 'FATAL' not in r
        m = re.search(r'finish called at (\d+)', r)
        fast = re.search(r'HYBRID CORE fast=(\d+) slow=(\d+)', r)
        (tmp / f'{p}.fx{fx}.log').write_text(r)
        return p, fx, ok, (int(m.group(1)) // 10000 if m else -1), (int(fast.group(1)) if fast else -1)
    jobs = [(p, fx) for p in progs for fx in fixes]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = {(p, fx): (ok, c, f) for p, fx, ok, c, f in ex.map(run, jobs)}
    bad = 0
    print('%-20s' % 'program' + ''.join('  fix=%-3d cycles  fast' % fx for fx in fixes) + '   delta vs first')
    for p in progs:
        row = [res[(p, fx)] for fx in fixes]
        st = 'PASS' if all(r[0] for r in row) else 'FAIL'
        bad += st == 'FAIL'
        d = ' '.join('%+d' % (r[1] - row[0][1]) for r in row[1:])
        print('%-20s' % p + ''.join('  %s %8d %5d' % ('ok  ' if r[0] else 'FAIL', r[1], r[2]) for r in row) + '   ' + d)
    print('CORE-HYBRID-CLK %s: %d programs x %d fixes, %d failing programs' % ('PASS' if not bad else 'FAIL', len(progs), len(fixes), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
