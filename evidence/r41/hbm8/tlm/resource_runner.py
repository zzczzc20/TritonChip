#!/usr/bin/env python3
"""Run an owned process tree on <=6 CPUs with a 16 GiB summed RSS guard.

Usage: resource_runner.py OUTPUT_PREFIX COMMAND [ARG ...]
Samples include shared pages in each process's RSS (a conservative bound).
No service credentials or host helpers are embedded here.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def field(pid, filename, key):
    try:
        for line in Path(f'/proc/{pid}/{filename}').read_text().splitlines():
            if line.startswith(key + ':'):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def descendants(pid):
    found = [pid]
    for parent in found:
        try:
            found += [int(p) for p in Path(f'/proc/{parent}/task/{parent}/children').read_text().split()]
        except OSError:
            pass
    return found


def main():
    prefix = Path(sys.argv[1])
    command = sys.argv[2:]
    available = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])
    if available < 40 * (1 << 20):
        raise RuntimeError(f'less than 40 GiB available: {available} KiB')
    cpus = sorted(os.sched_getaffinity(0))[:6]
    os.sched_setaffinity(0, cpus)
    started = time.monotonic()
    peak_rss = peak_pss = 0
    refusal = None
    with prefix.with_suffix('.log').open('w') as log, prefix.with_suffix('.resources.jsonl').open('w') as samples:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            pids = descendants(proc.pid)
            rss = sum(field(p, 'status', 'VmRSS') for p in pids)
            pss = sum(field(p, 'smaps_rollup', 'Pss') for p in pids)
            peak_rss, peak_pss = max(peak_rss, rss), max(peak_pss, pss)
            samples.write(json.dumps(dict(seconds=round(time.monotonic() - started, 3), pids=pids,
                                          rss_kib=rss, pss_kib=pss)) + '\n')
            samples.flush()
            if rss >= 16 * (1 << 20) or time.monotonic() - started > 4 * 3600:
                refusal = '16 GiB summed RSS or four hour wall guard exceeded'
                for pid in reversed(pids):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(2)
        code = proc.wait()
    prefix.with_suffix('.resources.json').write_text(json.dumps(dict(
        command=command, cpus=cpus, initial_available_kib=available, peak_summed_rss_kib=peak_rss,
        peak_summed_pss_kib=peak_pss, wall_seconds=round(time.monotonic() - started, 3),
        returncode=code, guard_refusal=refusal), indent=2) + '\n')
    return code if not refusal else 1


if __name__ == '__main__':
    sys.exit(main())
