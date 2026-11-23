import re, sys, collections
d = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.search(r'KS=(\d+) XB=(\d+) XM=(\d+) XN=(\d+) fixlat=(\d+) word_engine_cycles=(\d+) line_engine_cycles=(\d+)', l)
    if not m: continue
    ks, xb, xm, xn, lat, w, ln = map(int, m.groups())
    d[(ks, xb, xm, lat)][xn] = (w, ln)
print('per K step of KS elements: engine cycles (transfers only, ideal memory pipe of fixed latency); word engine (single-core device) / line engine (board image)')
print('%-34s' % 'pattern (A tiles/step, B kind)' + ''.join('  lat=%-3d (word/line)' % l for l in (12, 24, 40, 60)))
for (ks, xb, xm) in sorted({k[:3] for k in d}, key=lambda t: (t[2], t[1], t[0])):
    row = '%-34s' % ('KS=%d XM=%d B=%s' % (ks, xm, 'W^T(xpose)' if xb == 0 else 'plain row-major'))
    for lat in (12, 24, 40, 60):
        e = d[(ks, xb, xm, lat)]
        if 4 in e and 8 in e:
            row += '   %6.0f / %-6.0f   ' % ((e[8][0] - e[4][0]) / 4, (e[8][1] - e[4][1]) / 4)
        else:
            row += '   ---   '
    print(row)
