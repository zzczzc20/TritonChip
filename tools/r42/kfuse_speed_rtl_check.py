#!/usr/bin/env python3
"""Single-core RTL simulation versus the model on one core, G = 1 and G = 2 (domain: RTL simulation, model prediction).

    python3 tools/r42/kfuse_speed_rtl_check.py COLLECTED_DIR

RTL: evidence/r42/kfuse_speed/raw/rtl_single_core_records.tgz (tb/device single-core simulator, default knobs, the checkout's
compiler; profile core-vrf-sync-v5-x8 = legacy `m.mma.async` (G = 1) and core-vrf-sync-v5-x8-g2c4-dp = `m.mma.dp.async` (G = 2)).
Model: collected runs `c1_<profile>_<config>` (TNCA_ACTIVE_CORES=1) of the configurations `ideal` (all memory latencies at their
minimum, as the device's one-cycle memory) and `cal_lat0`/`run1` (memory of the board).
"""
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rtl = {}
with tarfile.open(ROOT / 'evidence/r42/kfuse_speed/raw/rtl_single_core_records.tgz') as t:
    for m in t.getmembers():
        for line in t.extractfile(m).read().decode().splitlines():
            r = json.loads(line)
            rtl[(r['platform']['profile'], r['point_id'])] = r['time']['device']
P1, P2 = 'core-vrf-sync-v5-x8', 'core-vrf-sync-v5-x8-g2c4-dp'
pts = ['op.matmul/tiny/fp16-m16-k32-n16', 'op.linear/tiny/fp16-m16-k64-n64', 'op.matmul/small/fp16-m64-k128-n128']
col = Path(sys.argv[1])


def model(cfg, prof, pid):
    f = col / f'c1_{prof}_{cfg}.json'
    return json.loads(f.read_text())['points'][pid]['device_cycles'] if f.exists() else None


print(f'{"point":<40}{"RTL G1":>9}{"RTL G2":>9}{"G2/G1":>7} | ' + ' | '.join(f'{c}: G1 G2 G2/G1 G2/RTL' for c in ('ideal', 'cal_lat0', 'run1')))
for pid in pts:
    a, b = rtl[(P1, pid)], rtl[(P2, pid)]
    out = f'{pid:<40}{a:>9,}{b:>9,}{b / a:>7.3f} | '
    parts = []
    for cfg in ('ideal', 'cal_lat0', 'run1'):
        m1, m2 = model(cfg, P1, pid), model(cfg, P2, pid)
        parts.append(f'{m1:>7,} {m2:>7,} {m2 / m1:.3f} {m2 / b:.2f}' if m1 and m2 else 'n/a')
    print(out + ' | '.join(parts))
