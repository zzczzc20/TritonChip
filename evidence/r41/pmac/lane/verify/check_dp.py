#!/usr/bin/env python3
"""check_dp.py -- compare lane_check dumps with the exact oracle (tools/model/fdp.py).

    check_dp.py [-j N] DUMP...

Each line: mode alt_a alt_b a b c sub_c y1 y2l y2d y3 (hex where hex). The
oracle is evaluated once per line and compared with every variant that defines
the case: V2L and V2D always, V1 in DP4 mode, V3 in DP4 mode with E4M3 x E4M3.
Also counts the lines where RO_48 changed the result against exact rounding of
C + sum, i.e. where the opcode's defined rounding is visible.
"""
import argparse
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))
import fdp  # noqa: E402


def work(lines):
    st = {'lines': 0, 'V1': 0, 'V2L': 0, 'V2D': 0, 'V3': 0, 'fail': 0, 'ro_visible': 0,
          'nan': 0, 'inf': 0, 'zero': 0, 'sub': 0}
    bad = []
    for ln in lines:
        f = ln.split()
        if len(f) != 11:
            continue
        mode, aa, ab = int(f[0]), int(f[1]), int(f[2])
        a, b, c = int(f[3], 16), int(f[4], 16), int(f[5], 16)
        sub = int(f[6])
        y = {'V1': int(f[7], 16), 'V2L': int(f[8], 16), 'V2D': int(f[9], 16), 'V3': int(f[10], 16)}
        want = fdp.lane_dp_ref(mode, a, b, c, aa, ab, sub)
        exact = fdp.lane_dp_ref(mode, a, b, c, aa, ab, sub, exact=True)
        st['lines'] += 1
        if want != exact:
            st['ro_visible'] += 1
        e = (want >> 23) & 0xff
        if e == 0xff:
            st['nan' if want & 0x7fffff else 'inf'] += 1
        elif e == 0:
            st['zero' if (want & 0x7fffff) == 0 else 'sub'] += 1
        for v in ('V1', 'V2L', 'V2D', 'V3'):
            if v == 'V1' and mode != fdp.MODE_DP4:
                continue
            if v == 'V3' and (mode != fdp.MODE_DP4 or aa or ab):
                continue
            st[v] += 1
            if y[v] != want:
                st['fail'] += 1
                if len(bad) < 20:
                    bad.append(f"{v}: {ln.strip()} want {want:08x}")
    return st, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=6)
    ap.add_argument('dumps', nargs='+')
    args = ap.parse_args()
    total = None
    allbad = []
    with Pool(args.j) as pool:
        for path in args.dumps:
            with open(path) as fh:
                lines = fh.readlines()
            chunks = [lines[i:i + 20000] for i in range(0, len(lines), 20000)]
            st_f = None
            for st, bad in pool.imap_unordered(work, chunks):
                st_f = st if st_f is None else {k: st_f[k] + st[k] for k in st}
                allbad += bad
            print(f"{os.path.basename(path)}: " + ' '.join(f"{k}={v}" for k, v in st_f.items()))
            sys.stdout.flush()
            total = st_f if total is None else {k: total[k] + st_f[k] for k in total}
    for b in allbad[:20]:
        print("MISMATCH", b)
    print("TOTAL: " + ' '.join(f"{k}={v}" for k, v in total.items()))
    ok = total['fail'] == 0 and total['lines'] > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
