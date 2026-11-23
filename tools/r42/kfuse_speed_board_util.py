#!/usr/bin/env python3
"""Matrix-engine utilisation of the single-GEMM board points, from board cycles alone (domain: board + RTL command time).

    python3 tools/r42/kfuse_speed_board_util.py

For a point that is one GEMM template launch (matmul fp16, linear), the number of K16 FP16 commands is programs x K/16
(programs = the launch's grid, K from the point id) and one command occupies the engine 168 cycles (single-core RTL probe,
evidence/r42/gemm/ANALYSIS.md sec 2; the SystemC model gives 169). Utilisation = commands x 168 / (3 cores x board cycles).
No model is involved except the command time: it bounds what any lane widening can remove if the engine time were fully serial.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kfuse_speed_validate as V   # noqa: E402

CMD = 168
b = V.board_records()
print(f'{"point":<54}{"programs":>10}{"K":>6}{"commands":>11}{"board cycles":>15}{"engine util":>12}')
tot = []
for pid, r in sorted(b.items()):
    fam = V.family_of(pid)
    if fam not in ('matmul', 'linear') or 'fp32' in pid:
        continue
    pl = r['time']['per_launch']
    if len(pl) != 1:
        continue
    k = int(re.search(r'-k(\d+)', pid).group(1))
    prog = pl[0]['grid'][0]
    cmds = prog * (k // 16)
    util = cmds * CMD / (3 * r['time']['device'])
    tot.append(util)
    print(f'{pid:<54}{prog:>10}{k:>6}{cmds:>11,}{r["time"]["device"]:>15,}{util:>12.3f}')
print(f'range {min(tot):.3f}..{max(tot):.3f}')
