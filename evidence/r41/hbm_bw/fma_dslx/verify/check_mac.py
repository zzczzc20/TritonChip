#!/usr/bin/env python3
"""check_mac.py -- compare mac_check dumps with the exact oracle
(tools/model/fdp.py lane_mac_ref).

    check_mac.py [-j N] DUMP...

Each line: mode fa fb a b c sub_c y (hex where hex). Also counts the lines
where RO_48 changed the result against exact rounding of C + sum (where the
opcode's defined rounding is visible) and the result classes.
"""
import argparse
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))
import fdp  # noqa: E402


def work(lines):
    st = {'lines': 0, 'fail': 0, 'ro_visible': 0, 'nan': 0, 'inf': 0, 'zero': 0, 'sub': 0}
    bad = []
    for ln in lines:
        f = ln.split()
        if len(f) != 8:
            continue
        mode, fa, fb = int(f[0]), int(f[1]), int(f[2])
        a, b, c = int(f[3], 16), int(f[4], 16), int(f[5], 16)
        sub, y = int(f[6]), int(f[7], 16)
        want = fdp.lane_mac_ref(mode, a, b, c, fa, fb, sub)
        exact = fdp.lane_mac_ref(mode, a, b, c, fa, fb, sub, exact=True)
        st['lines'] += 1
        if want != exact:
            st['ro_visible'] += 1
        e = (want >> 23) & 0xff
        if e == 0xff:
            st['nan' if want & 0x7fffff else 'inf'] += 1
        elif e == 0:
            st['zero' if (want & 0x7fffff) == 0 else 'sub'] += 1
        if y != want:
            st['fail'] += 1
            if len(bad) < 10:
                bad.append(f"{ln.strip()}  want {want:08x}")
    return st, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=4)
    ap.add_argument('dumps', nargs='+')
    args = ap.parse_args()
    total = None
    allbad = []
    with Pool(args.j) as pool:
        for path in args.dumps:
            lines = open(path).read().splitlines()
            chunks = [lines[i:i + 20000] for i in range(0, len(lines), 20000)]
            fst = None
            for st, bad in pool.map(work, chunks):
                fst = st if fst is None else {k: fst[k] + st[k] for k in fst}
                allbad += bad
            fst = fst or {}
            print(f"{os.path.basename(path)}: " + ' '.join(f"{k} {v}" for k, v in fst.items()))
            total = fst if total is None else {k: total[k] + fst.get(k, 0) for k in total}
    print("TOTAL: " + ' '.join(f"{k} {v}" for k, v in (total or {}).items()))
    for b in allbad[:30]:
        print("MISMATCH", b)
    ok = total and total['fail'] == 0 and total['lines'] > 0
    print("check_mac:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
