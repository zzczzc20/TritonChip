"""Shared helpers of the R42 design-space exploration (domain: SystemC/TLM model against board records).

Board records are read from the archived comparison files and record tarballs of evidence/r42/board (nothing is typed in):
  b41   = R41 three-core run-3 image, software of main (b41-run3-base: 113 points; xl-base: 32 points), the yardstick baseline
  r2    = the same image, round-2 software (ab-int2: 113 points; xl-cand2: 32 points), the newest board record that exists
Model records are the JSON files written by tools/r42/kfuse_speed_collect.py (per point: status, device cycles, per-launch cycles).
"""
import json
import math
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# the evidence/ directory is not part of the sparse worktree: read it from the lead checkout when it is not here
BOARD = [p for p in (ROOT / 'evidence/r42/board', Path('/home/ubuntu/TritonChip/evidence/r42/board')) if (p / 'ab-int2').exists()][0]


def family_of(pid):
    return pid.split('/')[0].split('.', 1)[1]


def tier_of(pid):
    return pid.split('/')[1]


def _formal(tgz, suffix):
    out = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if m.name.endswith(suffix) and 'identity' not in m.name:
                for line in t.extractfile(m).read().decode().splitlines():
                    r = json.loads(line)
                    out[r['point_id']] = r
    return out


def board():
    """pid -> dict(b41, r2, family, tier, demand, per_launch_r2, xl)"""
    out = {}
    c113 = json.loads((BOARD / 'ab-int2/comparison.json').read_text())
    cxl = json.loads((BOARD / 'xl-cand2/comparison.json').read_text())
    f113 = _formal(BOARD / 'ab-int2/records.tgz', 'original/formal.jsonl')
    fxl = _formal(BOARD / 'xl-cand2/records.tgz', 'xl/formal.jsonl')
    for comp, forms, xl in ((c113, f113, False), (cxl, fxl, True)):
        for r in comp['rows']:
            pid = r['point']
            f = forms[pid]
            out[pid] = dict(b41=r['old_cycles'], r2=r['new_cycles'], family=family_of(pid), tier=tier_of(pid), xl=xl,
                            demand=f['demand'], per_launch_r2=f['time']['per_launch'], eta_r2=f.get('eta'))
    return out


def load_collected(path):
    return json.loads(Path(path).read_text())['points']


def geo(xs):
    xs = [x for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')
