#!/usr/bin/env python3
"""Does the run-3 board setup hold the 3 GiB live-byte cap of the xl tier? Read from the run-3 board records only.

    xl_memory_check.py RECORDS.tgz [--suite archbench/suites/xl.json] [--out memory_check.json]

RECORDS.tgz is a records archive of the 113 original points on the run-3 image (evidence/r42/board/b41-run3-base/
records.tgz). Domain: board. Reports the heap capacity the runtime saw (`native_memory.session.heap_capacity_bytes`),
the memory window of the backend (`native_memory.window`), and the largest heap peak and tensor peak of any run that
passed, with the point that has it; then, when --suite is given, the live bytes each xl point declares against that
capacity. The largest xl point needing more than the capacity would be reported, not changed.
"""
import argparse
import json
from pathlib import Path
import tarfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('records', type=Path)
    ap.add_argument('--suite', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    caps, windows, peaks = set(), set(), []
    with tarfile.open(a.records) as tar:
        for m in tar.getmembers():
            if m.isfile() and '/original/runs/' in m.name and m.name.endswith('.jsonl'):
                for line in tar.extractfile(m).read().decode().splitlines():
                    r = json.loads(line)
                    nm = r.get('native_memory') or {}
                    s = nm.get('session') or {}
                    if r.get('status') != 'pass' or not s:
                        continue
                    caps.add(s['heap_capacity_bytes'])
                    windows.add((nm['window']['base'], nm['window']['size'], nm['window']['local_limit']))
                    peaks.append((s['heap_peak_bytes'], nm.get('tensor_peak_bytes'), r['point_id']))
    peaks.sort(reverse=True)
    out = dict(domain='board (U280), image 791b8af2, run-3 records', records=str(a.records),
               passing_runs_with_memory=len(peaks), heap_capacity_bytes=sorted(caps), windows=sorted(windows),
               largest_heap_peaks=[dict(heap_peak_bytes=h, tensor_peak_bytes=t, point=p) for h, t, p in peaks[:3]])
    cap = min(caps) if caps else None
    if a.suite:
        suite = json.loads(a.suite.read_text())
        live = sorted(((p['live_bytes'], p['id']) for p in suite['points']), reverse=True)
        out['suite'] = dict(path=str(a.suite), memory_cap_bytes=suite.get('memory_cap_bytes'),
                            largest_live=[dict(live_bytes=b, point=i) for b, i in live[:3]],
                            heap_capacity_over_largest_live=(cap/live[0][0] if cap else None),
                            over_capacity=[i for b, i in live if cap and b > cap])
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')


if __name__ == '__main__':
    main()
