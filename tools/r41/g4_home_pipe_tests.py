#!/usr/bin/env python3
"""tnca_home_pipe (R41 G4): randomised and directed regressions (RTL simulation).

    python3 tools/r41/g4_home_pipe_tests.py [--quick]

* tb_home_pipe: tb_nuca_wide's byte-shadow traffic (hot, streaming, random
  byte enables, dirty evictions, merged misses) against a reordering, stalling
  DRAM model, over the full-line no-fill option, response back-pressure,
  DRAM latency 5/40/200 with and without reordering, MSHR files of 2/4/8 and
  target pools of 4/8/16, five seeds each; with maintenance, a half-range and
  a full-range writeback/invalidate whose memory image must equal the shadow.
* the single-line PRESSURE stimulus (target exhaustion).
* tb_home_pipe_special: every scalar atomic at every 8-byte position, cold and
  hot; a merged read/store/read miss; the TMU register page (with and without
  the no-fill option).
* service: hot reads, streaming reads and streaming full-line stores, with
  their span in cycles (the throughput of one home in this fixture).
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(binary, top, params, source, args=()):
    cmd = ['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/common', '-y', 'rtl/nuca', '-y', 'tb/common',
           '-s', top, '-o', binary] + [f'-P{top}.{k}={v}' for k, v in params.items()] + [source]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=120)
    out = subprocess.run(['vvp', binary, *args], cwd=ROOT, text=True, capture_output=True, timeout=600).stdout
    ok = 'PASS' in out and 'FAIL' not in out and 'FATAL' not in out
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    seeds = (1, 7) if a.quick else (1, 7, 39, 101, 2026)
    n = bad = 0
    with tempfile.TemporaryDirectory(prefix='tnca-home-pipe-') as t:
        b = str(Path(t)/'h.vvp')
        cases = []
        for nofill in (0, 1):
            for stall in (0, 1):
                for lat, ooo in ((5, 0), (40, 1), (200, 1)):
                    for mshr, tgt in ((2, 4), (4, 8), (8, 16)):
                        for seed in seeds:
                            cases.append(({'LINE_WRITE_NOFILL': nofill, 'RSP_STALL': stall, 'DRAM_LAT': lat,
                                           'DRAM_OOO': ooo, 'MSHR_N': mshr, 'TARGET_N': tgt, 'MAINTENANCE': 1},
                                          ['+nreq=3000', f'+seed={seed}']))
        cases.append(({'PRESSURE': 1, 'DRAM_LAT': 200}, ['+nreq=100', '+min_wb=0']))
        for params, args in cases:
            ok, out = run(b, 'tb_home_pipe', params, 'tb/nuca/tb_home_pipe.v', args)
            m = re.search(r'MEMORY reads=(\d+) writes=(\d+) nofill_allocs=(\d+)', out)
            mt = re.search(r'MAINTENANCE lower_half_lines=(\d+) mismatches=(\d+)', out)
            print(f"CASE {'PASS' if ok else 'FAIL'} {params} {' '.join(args)} "
                  f"{m.group(0) if m else ''} {mt.group(0) if mt else ''}", flush=True)
            if not ok:
                bad += 1
                print('\n'.join(l for l in out.splitlines() if 'FAIL' in l or 'FATAL' in l)[:2000])
            n += 1
        for nofill in (0, 1):
            ok, out = run(b, 'tb_home_pipe_special', {'LINE_WRITE_NOFILL': nofill}, 'tb/nuca/tb_home_pipe_special.v')
            print(f"SPECIAL {'PASS' if ok else 'FAIL'} nofill={nofill} "
                  f"{(re.search(r'HOME PIPE SPECIAL PASS.*', out) or [''])[0]}", flush=True)
            bad += not ok
            n += 1
        for mode in (1, 2, 3):
            for mshr in (8, 32):
                ok, out = run(b, 'tb_home_pipe', {'SERVICE_MODE': mode, 'LINE_WRITE_NOFILL': 1, 'MSHR_N': mshr,
                                                  'TARGET_N': max(16, mshr)},
                              'tb/nuca/tb_home_pipe.v', ['+nreq=2048', '+min_wb=0'])
                m = re.search(r'HOME SERVICE .*', out)
                print(f"SERVICE {'PASS' if ok else 'FAIL'} mshr={mshr} {m.group(0) if m else ''}", flush=True)
                bad += not ok
                n += 1
    print(f'HOME PIPE {n - bad} of {n} cases pass', flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
