#!/usr/bin/env python3
"""Compare gate lane logs of the head and base trees.

For each lane run on both trees: exit codes, and the multiset of result lines
(lines containing PASS, FAIL, cycles= or mismatch), with tree paths and
timestamps normalised. Prints a JSON summary; computes, never types, results.
"""
import collections, json, re, sys
from pathlib import Path

def codes(d):
    out = {}
    for line in (d/'exit_codes.txt').read_text().splitlines():
        name, rc = line.rsplit(' ', 1)
        out[name] = int(rc)  # the last record of a lane wins
    return out

PAT = re.compile(r'PASS|FAIL|cycles=|mismatch', re.I)
def lines(path):
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('# ') or not PAT.search(line):
            continue
        line = re.sub(r'/home/\w+/r41fix_gate/(head|base|final)', '<TREE>', line)
        line = re.sub(r'\d+(\.\d+)?\s*s\b', '<SECONDS>', line)
        line = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>', line)
        rows.append(line.rstrip())
    return collections.Counter(rows)

def main():
    head, base = Path(sys.argv[1]), Path(sys.argv[2])
    ch, cb = codes(head), codes(base)
    report = {}
    for lane in sorted(set(ch) | set(cb)):
        row = dict(head_exit=ch.get(lane), base_exit=cb.get(lane))
        if lane in ch and lane in cb:
            lh, lb = lines(head/f'{lane}.log'), lines(base/f'{lane}.log')
            row.update(result_lines_head=sum(lh.values()), result_lines_base=sum(lb.values()),
                       pass_head=sum(v for k, v in lh.items() if 'PASS' in k),
                       pass_base=sum(v for k, v in lb.items() if 'PASS' in k),
                       identical=lh == lb,
                       base_lines_missing_from_head=sum((lb - lh).values()),
                       only_head=sorted((lh - lb).elements())[:20],
                       only_base=sorted((lb - lh).elements())[:20],
                       only_head_count=sum((lh - lb).values()))
        report[lane] = row
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main()
