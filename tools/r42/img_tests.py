#!/usr/bin/env python3
"""R42-img: unit tests of the integrated tnca_matrix (RTL simulation, Icarus Verilog).

    python3 tools/r42/img_tests.py matlock --ref tnca_matrix_ref.v [--seeds N] [--cmds N]
    python3 tools/r42/img_tests.py matovl  [--seeds N] [--segs N]

matlock  the engine at default parameters (OVL 0; MAC_F32 0 and 1 with no binary32 command given; ABUF 0 and 1) beside the delivered
         engine (--ref: rtl/core/tnca_matrix.v of origin/R42 with the module renamed tnca_matrix_ref), one random command list,
         one vector-file model: every output of both compared in every cycle (tb/core/tb_matrix_lock.v).
matovl   tb/core/tb_matrix_ovl.v (the overlapped sequencer against the sequencer it replaces) with the binary32 element format in the
         command list: MAC_F32 1 on both sides.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
INC = ['-g2005', '-I', 'rtl/common', '-y', 'rtl/core', '-y', 'rtl/common', '-y', 'rtl/gen']


def build(tmp, top, tb, params, extra=()):
    out = str(Path(tmp) / f'{top}.vvp')
    cmd = ['iverilog'] + INC + ['-s', top] + [f'-P{top}.{k}={v}' for k, v in params.items()] + ['-o', out, tb] + [str(x) for x in extra]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=600)
    return out


def run(vvp, args=()):
    r = subprocess.run(['vvp', vvp] + list(args), cwd=ROOT, capture_output=True, text=True, timeout=7200)
    return r.stdout


def matlock(a):
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='imgtest-') as t:
        for dp in (1, 0):
            for vm in (0, 1):
                for abuf in (0, 1):
                    for f32 in (0, 1):
                        vvp = build(t, 'tb_matrix_lock', 'tb/core/tb_matrix_lock.v',
                                    dict(MAC_DP=dp, VRF_MODE=vm, ABUF=abuf, MAC_F32=f32), [a.ref])
                        for seed in range(1, a.seeds + 1):
                            out = run(vvp, [f'+seed={seed}', f'+cmds={a.cmds}'])
                            ok = 'MATRIX LOCK PASS' in out and 'FAIL' not in out and 'FATAL' not in out
                            n += 1; bad += not ok
                            line = next((l for l in out.splitlines() if 'MATRIX LOCK' in l), out.strip().splitlines()[-1] if out.strip() else 'no output')
                            print(f"{'PASS' if ok else 'FAIL'} {line}", flush=True)
                            if not ok:
                                print(out[-1500:], flush=True)
    print(f'MATLOCK {n - bad} of {n} configurations cycle-identical', flush=True)
    return 1 if bad else 0


def matovl(a):
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='imgtest-') as t:
        for dp in (1, 0):
            for vm in (0, 1, 2):
                for f32 in (0, 1):
                    vvp = build(t, 'tb_matrix_ovl', 'tb/core/tb_matrix_ovl.v', dict(MAC_DP=dp, VRF_MODE=vm, MAC_F32=f32))
                    for seed in range(1, a.seeds + 1):
                        out = run(vvp, [f'+seed={seed}', f'+segs={a.segs}'])
                        ok = re.search(r'^PASS', out, re.M) is not None and 'FATAL' not in out
                        n += 1; bad += not ok
                        line = next((l for l in out.splitlines() if l.startswith('random:')), 'no summary')
                        print(f"{'PASS' if ok else 'FAIL'} MAC_DP={dp} VRF_MODE={vm} MAC_F32={f32} seed={seed} {line}", flush=True)
                        if not ok:
                            print(out[-1500:], flush=True)
    print(f'MATOVL {n - bad} of {n} configurations pass', flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('matlock'); p.add_argument('--ref', required=True); p.add_argument('--seeds', type=int, default=3); p.add_argument('--cmds', type=int, default=400)
    p = sub.add_parser('matovl'); p.add_argument('--seeds', type=int, default=3); p.add_argument('--segs', type=int, default=60)
    a = ap.parse_args()
    sys.exit({'matlock': matlock, 'matovl': matovl}[a.mode](a))


if __name__ == '__main__':
    main()
