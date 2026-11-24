#!/usr/bin/env python3
"""usage: census.py FILE.probe   engine events of a single-core RTL run (tools/xf_probe.v with +gprobe_ev):
transfer command durations (issue 'T' to completion 't'), matrix command durations ('M' to 'm'), per launch."""
import sys, re, statistics, collections
T, t, M, m = [], [], [], []
for line in open(sys.argv[1]):
    p = line.split()
    if not p: continue
    if p[0] == 'T': T.append((int(p[1]), p[2], p[3]))
    elif p[0] == 't': t.append(int(p[1]))
    elif p[0] == 'M': M.append((int(p[1]), p[3]))
    elif p[0] == 'm': m.append(int(p[1]))
print('transfer commands', len(T), 'completions', len(t), '| matrix commands', len(M), 'completions', len(m))
import bisect
t.sort(); m.sort()
# one transfer command at a time in the word engine: a command's completion is the first completion after its issue
td = collections.defaultdict(list)
for c, ctx, w in T:
    k = bisect.bisect_right(t, c)
    if k < len(t): td[w].append(t[k] - c)
for w in sorted(td, key=lambda x: int(x[1:])):
    d = td[w]
    print('  transfer window %-4s n=%-3d cycles median %d min %d max %d' % (w, len(d), statistics.median(d), min(d), max(d)))
md = []
for c, op in M:
    k = bisect.bisect_right(m, c)
    if op == 'op1' and k < len(m): md.append(m[k] - c)
print('  m.mma commands n=%d: cycles median %d min %d max %d' % (len(md), statistics.median(md), min(md), max(md)) if md else '  no m.mma')
