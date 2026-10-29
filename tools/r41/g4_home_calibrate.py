#!/usr/bin/env python3
"""Service rate of the pipelined home (rtl/nuca/tnca_home_pipe.v), to calibrate
the TLM home occupancies (tools/performance_harness/systemc tl_slice_*).

    python3 tools/r41/g4_home_calibrate.py [--out FILE.json]

tb/nuca/tb_home_pipe.v in SERVICE_MODE 1 (hot reads), 2 (streaming read
misses), 3 (streaming full-line writes, no fill), 4 (a copy: alternating
streaming reads and full-line writes), 64-byte words, 32 MSHRs, at several
memory latencies, with no memory stalls. Reports cycles per request. If the
rate does not change with latency, the home (not the requester's 32 tags)
sets it. Domain: RTL simulation of one home slice.
"""
import argparse, json, re, subprocess, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def run(binary, params, nreq):
    cmd = ['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/common', '-y', 'rtl/nuca', '-y', 'tb/common',
           '-s', 'tb_home_pipe', '-o', binary] + [f'-Ptb_home_pipe.{k}={v}' for k, v in params.items()] + \
          ['tb/nuca/tb_home_pipe.v']
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=120)
    out = subprocess.run(['vvp', binary, f'+nreq={nreq}', '+min_wb=0'], cwd=ROOT, text=True,
                         capture_output=True, timeout=900).stdout
    m = re.search(r'HOME SERVICE mode=(\d+) .*accepted=(\d+) responses=(\d+) span_cycles=(\d+) hit=(\d+) miss=(\d+)', out)
    ok = 'PASS' in out and 'FAIL' not in out and 'FATAL' not in out and m is not None
    return ok, m, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out')
    ap.add_argument('--mshr', type=int, default=32, help='MSHR_N (targets: the same number)')
    ap.add_argument('--modes', default='1,2,3,4')
    ap.add_argument('--req-tag-w', type=int, default=5, help="the requester's tag bits (2^N requests in flight)")
    ap.add_argument('--image-sets', action='store_true', help='the image slice (32 sets x 8 ways), not the bench default (16 x 4)')
    ap.add_argument('--fill-chain', type=int, default=0, help='FILL_CHAIN (HOME_LINE_PIPE 2)')
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix='tnca-home-cal-') as t:
        for mode in [int(x) for x in a.modes.split(',')]:
            for lat in (10, 40, 80):
                params = dict(SERVICE_MODE=mode, LINE_WRITE_NOFILL=1, MSHR_N=a.mshr, TARGET_N=a.mshr,
                              DRAM_LAT=lat, DRAM_STALL=0, DRAM_OOO=1, FILL_CHAIN=a.fill_chain)
                params['TAG_W'] = a.req_tag_w
                if a.image_sets:
                    params.update(SETS=32, SET_W=5, WAYS=8, WAY_W=3)
                ok, m, out = run(str(Path(t) / 'cal.vvp'), params, 2048)
                row = dict(mode=mode, dram_latency=lat, ok=ok)
                if m:
                    n, span = int(m.group(2)), int(m.group(4))
                    row.update(requests=n, span_cycles=span, cycles_per_request=round(span / n, 3),
                               hits=int(m.group(5)), misses=int(m.group(6)))
                rows.append(row)
                print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(domain='rtl-simulation', module='rtl/nuca/tnca_home_pipe.v', fill_chain=a.fill_chain, mshr_n=a.mshr, image_sets=a.image_sets, requester_tag_w=a.req_tag_w,
                                               rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
