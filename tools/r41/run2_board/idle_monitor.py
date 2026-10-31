#!/usr/bin/env python3
"""Continuously record a monotonic idle interval without touching the card."""
import json
import os
from pathlib import Path
import time
import auto


def main():
    work = Path.home() / 'r41run3'
    sysfs = Path('/sys/bus/pci/devices')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    tracker = auto.IdleTracker(max_gap=15)
    while True:
        try:
            row = auto.probe(sysfs)
            tracker.observe(time.monotonic(), row)
        except (OSError, ValueError):
            tracker = auto.IdleTracker(max_gap=15)
        state = dict(schema='tnca-idle-monitor/1', boot_id=boot, pid=os.getpid(),
                     start=tracker.start, last=tracker.last, identity=tracker.identity)
        temporary = work / 'idle_monitor.json.tmp'
        temporary.write_text(json.dumps(state) + '\n')
        temporary.replace(work / 'idle_monitor.json')
        time.sleep(5)


if __name__ == '__main__':
    main()
