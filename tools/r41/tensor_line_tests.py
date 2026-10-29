#!/usr/bin/env python3
"""The line-wide, multi-command transfer engine against the word engine
(tb/core/tb_tensor_line.v, tnca_tensor LINE_ENGINE 1 vs 0), as a miter.

    python3 tools/r41/tensor_line_tests.py [--xperf]

Seeds x (register-file refusal %, memory refusal %, latency); every
configuration must end with identical memories, register files and
completion counts, and with the line engine having moved lines.
Domain: RTL simulation (Icarus Verilog).
"""
import re, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def xperf():
    """The Hub FP8 matmul's transposed weight transfer (tb XPERF): cycles of the
    word engine and the line engine at fixed memory latencies (rtl)."""
    with tempfile.TemporaryDirectory(prefix='tnca-tensor-xperf-') as t:
        b = str(Path(t) / 'xp.vvp')
        for lat in (20, 60, 100):
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-s', 'tb_tensor_line',
                            '-Ptb_tensor_line.XPERF=1', f'-Ptb_tensor_line.FIXLAT={lat}',
                            '-Ptb_tensor_line.VW_STALL=0', '-Ptb_tensor_line.REQ_STALL=0',
                            '-o', b, 'tb/core/tb_tensor_line.v', 'rtl/core/tnca_tensor.v'],
                           cwd=ROOT, check=True, timeout=120)
            out = subprocess.run(['vvp', b], cwd=ROOT, capture_output=True, text=True, timeout=1800).stdout
            m = re.search(r'XPERF .*', out)
            ok = m is not None and 'PASS' in out and 'FATAL' not in out
            print(f"{'PASS' if ok else 'FAIL'} {m.group(0) if m else out[-400:]}", flush=True)
            if not ok:
                return 1
        # another requester's store to the weight lines: ordered before the second
        # load (both engines must return the new bytes), and racing with it (reported)
        for xr in (1, 2):
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-s', 'tb_tensor_line',
                            '-Ptb_tensor_line.XPERF=1', f'-Ptb_tensor_line.XRACE={xr}', '-Ptb_tensor_line.FIXLAT=60',
                            '-Ptb_tensor_line.VW_STALL=0', '-Ptb_tensor_line.REQ_STALL=0',
                            '-o', b, 'tb/core/tb_tensor_line.v', 'rtl/core/tnca_tensor.v'],
                           cwd=ROOT, check=True, timeout=120)
            out = subprocess.run(['vvp', b], cwd=ROOT, capture_output=True, text=True, timeout=1800).stdout
            ok = 'TENSOR LINE MITER PASS' in out and 'FATAL' not in out
            m = re.search(r'XRACE 2.*', out)
            print(f"{'PASS' if ok else 'FAIL'} XRACE={xr} {m.group(0) if m else ''}", flush=True)
            if not ok:
                return 1
    return 0


def main():
    if '--xperf' in sys.argv:
        return xperf()
    configs = [(seed, vw, rq, lat) for seed in (1, 2, 3, 4, 5, 6)
               for vw, rq, lat in ((20, 20, 40), (0, 0, 8), (40, 40, 120))]
    bad = 0
    with tempfile.TemporaryDirectory(prefix='tnca-tensor-line-') as t:
        for seed, vw, rq, lat in configs:
            b = str(Path(t) / 'tl.vvp')
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-s', 'tb_tensor_line',
                            f'-Ptb_tensor_line.SEED={seed}', f'-Ptb_tensor_line.VW_STALL={vw}',
                            f'-Ptb_tensor_line.REQ_STALL={rq}', f'-Ptb_tensor_line.MAXLAT={lat}',
                            '-o', b, 'tb/core/tb_tensor_line.v', 'rtl/core/tnca_tensor.v'],
                           cwd=ROOT, check=True, timeout=120)
            out = subprocess.run(['vvp', b], cwd=ROOT, capture_output=True, text=True, timeout=1800).stdout
            m = re.search(r'TENSOR LINE MITER PASS.*', out)
            ok = m is not None and 'FATAL' not in out
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'} seed={seed} vw_stall={vw} req_stall={rq} maxlat={lat} "
                  f"{m.group(0) if m else out[-400:]}", flush=True)
    n = len(configs)
    print(f'TENSOR LINE {n - bad} of {n} configurations pass', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
