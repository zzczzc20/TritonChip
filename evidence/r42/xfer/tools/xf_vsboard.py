#!/usr/bin/env python3
"""usage: xf_vsboard.py BOARD.json COLLECTED.json TAG...   TLM cycles / board cycles per point (ab-int2 board records)"""
import json, sys, math
board = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2])); tags = sys.argv[3:]
pts = sorted(set.intersection(*[set(c[t]) for t in tags]) & set(board))
print('%-58s %12s' % ('point', 'board') + ''.join(' %12s' % t[:12] for t in tags))
g = {t: [] for t in tags}
for p in pts:
    row = '%-58s %12d' % (p, board[p])
    for t in tags:
        v = c[t][p]
        if v[0] is None or v[1] != 'pass':
            row += ' %12s' % 'FAIL'; continue
        r = v[0] / board[p]; g[t].append(r); row += ' %12.3f' % r
    print(row)
print('%-58s %12s' % ('geomean model/board', '') + ''.join(' %12.3f' % math.exp(sum(map(math.log, g[t])) / len(g[t])) for t in tags))
