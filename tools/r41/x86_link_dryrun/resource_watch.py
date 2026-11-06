#!/usr/bin/env python3
"""Run a command in a fresh session, cap summed process-tree RSS, log samples."""
import os
import signal
import subprocess
import sys
import time

limit_kib = 12 * 1024 * 1024
with open(sys.argv[1], 'w', buffering=1) as log:
    child = subprocess.Popen(sys.argv[2:], start_new_session=True)
    peak = 0
    while child.poll() is None:
        rows = subprocess.check_output(['ps', '-eo', 'pid,ppid,rss'], text=True).splitlines()[1:]
        procs = {}
        for row in rows:
            fields = row.split()
            if len(fields) == 3:
                procs[int(fields[0])] = (int(fields[1]), int(fields[2]))
        descendants = {child.pid}
        while True:
            more = {pid for pid, (ppid, _) in procs.items() if ppid in descendants}
            if more <= descendants:
                break
            descendants |= more
        rss = sum(procs.get(pid, (0, 0))[1] for pid in descendants)
        peak = max(peak, rss)
        log.write(f'{time.time():.3f} rss_kib={rss} peak_kib={peak} processes={len(descendants)}\n')
        if rss > limit_kib:
            log.write('STOP: summed RSS exceeded 12 GiB\n')
            for pid in sorted(descendants, reverse=True):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            child.wait()
            sys.exit(125)
        time.sleep(2)
    log.write(f'EXIT={child.returncode} PEAK_RSS_KIB={peak}\n')
    sys.exit(child.returncode)
