#!/usr/bin/env python3
"""board_vs_sim.py BOARD.jsonl REF.jsonl: ArchBench device cycles on the board against
the same points run on the RTL simulator (fpga-u50-sim), point by point. Exit 0
only if every point has the same status pair and identical cycles: the board
runs the RTL, so a difference is a fault in the board path, not noise."""
import json, sys

def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r['point_id']] = r
    return out

b, r = load(sys.argv[1]), load(sys.argv[2])
same = diff = 0
print(f"{'point':<52s} {'board':>10s} {'sim':>10s}  status")
for pid in sorted(set(b) | set(r)):
    bb, rr = b.get(pid), r.get(pid)
    bc = bb and bb.get('time', {}).get('device')
    rc = rr and rr.get('time', {}).get('device')
    st = f"{bb['status'] if bb else '-'}/{rr['status'] if rr else '-'}"
    mark = 'same' if (bc is not None and bc == rc) else 'DIFF'
    same += mark == 'same'; diff += mark == 'DIFF'
    print(f"{pid:<52s} {bc if bc is not None else '-':>10} {rc if rc is not None else '-':>10}  {st} {mark}")
print(f"B4: {same} identical, {diff} different, of {len(set(b) | set(r))} points")
# No points is not agreement.
sys.exit(0 if diff == 0 and same > 0 else 1)
