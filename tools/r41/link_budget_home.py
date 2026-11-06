#!/usr/bin/env python3
"""Home-slice service ceilings at the run-1 image geometry (RTL simulation).

    PYTHONPATH=. python3 tools/r41/link_budget_home.py --out DIR

One NUCA home slice (rtl/nuca/tnca_nuca_slice.v) in tb/nuca/tb_nuca_wide.v at
the run-1 tile geometry: 64-bit words (the core port), 32 sets x 8 ways, 8 MSHRs,
16 targets, registered victim select/rank/owner, cache data pipeline,
registered DCO ranges. The generator offers a new request every cycle it can
(32 tags outstanding); the slice is the only limit apart from the synthetic
DRAM, whose latency and random stalls are swept.

Modes: 1 hot reads (16 resident lines), 2 streaming reads (every line a miss),
3 streaming full-word writes (every line a write miss, then a write-back).
Every response is checked against the shadow memory. The span is first
accepted request to last response. RTL simulation of one slice, not the chip
and not the board.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.r41.wide_home_tests import run_case  # noqa: E402

GEOMETRY = dict(WORD_BITS=64, SETS=32, SET_W=5, WAYS=8, WAY_W=3, MSHR_N=8, MSHR_W=3, TARGET_N=16, TARGET_W=4,
                CACHE_DATA_PIPELINE=1, REGISTER_VICTIM_SELECT=1, REGISTER_VICTIM_RANK=1,
                REGISTER_VICTIM_OWNER=1, DCO_REGISTERED_RANGES=1, HIT_CHAIN=0, FAST_FILL_READ=0,
                BACKGROUND_READ_REQUESTS=0)
SERVICE = re.compile(r'HOME SERVICE mode=(\d+) word_bytes=(\d+) accepted=(\d+) responses=(\d+) '
                     r'span_cycles=(\d+) hit=(\d+) miss=(\d+)')
SUMMARY = re.compile(r'hit=(\d+) miss=(\d+) merge=(\d+) wb=(\d+) mshr_full=(\d+) target_full=(\d+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--words', type=int, default=32768)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix='r41-link-home-') as temp:
        for mode in (1, 2, 3):
            for latency, stall in ((40, 1), (40, 0), (1, 0)):
                params = dict(GEOMETRY, SERVICE_MODE=mode, DRAM_LAT=latency, DRAM_STALL=stall)
                label = f'mode{mode}-lat{latency}-stall{stall}'
                log = run_case(str(Path(temp)/'home.vvp'), 'tb_nuca_wide', params, 'tb/nuca/tb_nuca_wide.v',
                               [f'+nreq={a.words}', '+min_wb=0'])
                (a.out/f'{label}.log').write_text(log)
                m = SERVICE.search(log)
                s = SUMMARY.search(log)
                row = dict(zip(('mode', 'word_bytes', 'accepted', 'responses', 'span_cycles', 'hit', 'miss'),
                               map(int, m.groups())))
                row.update(dict(zip(('hit_total', 'miss_total', 'merge', 'wb', 'mshr_full', 'target_full'),
                                    map(int, s.groups()))))
                if row['accepted'] != a.words or row['responses'] != a.words:
                    raise SystemExit(f'{label}: lost handshakes {row}')
                row.update(label=label, dram_latency=latency, dram_random_stall=bool(stall),
                           useful_bytes=a.words * row['word_bytes'],
                           bytes_per_cycle=a.words * row['word_bytes'] / row['span_cycles'],
                           cycles_per_word=row['span_cycles'] / a.words)
                rows.append(row)
                print(label, json.dumps(row, sort_keys=True), flush=True)
    sources = [ROOT/'rtl/nuca/tnca_nuca_slice.v', ROOT/'tb/nuca/tb_nuca_wide.v', Path(__file__).resolve()]
    (a.out/'results.json').write_text(json.dumps(dict(
        schema='tnca-link-budget-home/1', evidence='rtl-simulation', board=False,
        scope='one home slice with a synthetic DRAM; no NoC, core, AXI outlet or HBM',
        geometry=GEOMETRY, words=a.words,
        sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
