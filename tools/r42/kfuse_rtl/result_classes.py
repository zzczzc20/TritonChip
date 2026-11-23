#!/usr/bin/env python3
"""result_classes.py G VECFILE... -- how the expected results of a vector file (made by kfuse_rtl_vectors.py from the model)
fall into classes that matter for the shape of the back end: subnormal results that differ from c, overflow to infinity with a
finite c, NaN, infinity, zero (+0 / -0), results equal to c."""
import sys

G = int(sys.argv[1])
for path in sys.argv[2:]:
    cnt = dict(n=0, sub_not_c=0, sub=0, ovf=0, nan=0, inf=0, pz=0, nz=0, eq_c=0)
    with open(path) as f:
        for l in f:
            w = int(l, 16)
            c = (w >> (16 * G)) & 0xffffffff
            y = (w >> (16 * G + 32)) & 0xffffffff
            e, fr = (y >> 23) & 0xff, y & 0x7fffff
            ce = (c >> 23) & 0xff
            cnt['n'] += 1
            if e == 0 and fr != 0:
                cnt['sub'] += 1
                cnt['sub_not_c'] += (y != c)
            if e == 255 and fr == 0 and ce != 255:
                cnt['ovf'] += 1
            cnt['nan'] += (e == 255 and fr != 0)
            cnt['inf'] += (e == 255 and fr == 0)
            cnt['pz'] += (y == 0)
            cnt['nz'] += (y == 0x80000000)
            cnt['eq_c'] += (y == c)
    print('%s G=%d: ' % (path.split('/')[-1], G) + ' '.join('%s=%d' % kv for kv in cnt.items()))
