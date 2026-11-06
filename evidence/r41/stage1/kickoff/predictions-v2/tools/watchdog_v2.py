#!/usr/bin/env python3
"""Keep the v2 sealing runs inside the GB10 limits; kill only by PID.

Every 20 s: sum the RSS of this user's processes whose command line or
working tree belongs to ~/r41s1-c2c96d3. If total RSS > 35 GB or
MemAvailable < 40 GB, kill (SIGKILL by PID, with its process group when it
leads one) the largest slow-path HBM run_graphs child, and log it.
"""
import json
import os
from pathlib import Path
import signal
import time

H = Path.home()/'r41s1-c2c96d3'
LOG = H/'watchdog.log'
LIMIT_RSS = 35 << 30
MIN_AVAILABLE = 40 << 30


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) << 10


def mine():
    rows = []
    uid = os.getuid()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != uid:
                continue
            cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(p/'cwd')
            if 'r41s1-c2c96d3' not in cmd and 'r41s1-c2c96d3' not in cwd:
                continue
            rss = int((p/'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            pgid = os.getpgid(int(p.name))
            rows.append(dict(pid=int(p.name), pgid=pgid, rss=rss, cmd=cmd[:300], cwd=cwd))
        except (OSError, ValueError):
            continue
    return rows


def log(**row):
    with LOG.open('a') as fh:
        fh.write(json.dumps(dict(row, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')


peak = 0
while True:
    rows = mine()
    total, avail = sum(r['rss'] for r in rows), available()
    peak = max(peak, total)
    log(total_rss_gib=round(total/2**30, 2), available_gib=round(avail/2**30, 1), peak_rss_gib=round(peak/2**30, 2),
        processes=len(rows))
    if total > LIMIT_RSS or avail < MIN_AVAILABLE:
        # Groups led by a slow-path HBM graph child (run_graphs starts each in its own session).
        groups = {}
        for r in rows:
            groups.setdefault(r['pgid'], []).append(r)
        candidates = [(sum(x['rss'] for x in g), pgid) for pgid, g in groups.items()
                      if any('--child' in x['cmd'] for x in g) and any('gemm-hbm' in x['cwd'] and '-slow' in x['cwd'] for x in g)]
        if candidates:
            size, pgid = max(candidates)
            log(action='kill', pgid=pgid, rss_gib=round(size/2**30, 2),
                reason='total RSS over 35 GB or available memory under 40 GB')
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if not rows and Path(H/'watchdog.stop').exists():
        break
    time.sleep(20)
