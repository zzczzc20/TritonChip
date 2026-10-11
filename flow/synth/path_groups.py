#!/usr/bin/env python3
"""Group a vivado_ooc_timing.tcl paths CSV by (start module, end module).

    python3 flow/synth/path_groups.py sim/build/ooc/<unit>/<unit>_paths.csv [depth]

A cell name like u_shell/u_core/u_mtx/g_mac[3].u_mac/u_fma/g_l2.u_xls/p0_x_reg[7]
is cut to its module path (`depth` hierarchy levels, default 4, indices
dropped), so sixteen lanes and 32 bits of one bus become one row: how many
endpoints, the worst delay and its logic levels, and an example pair.
"""
import csv
import re
import sys


def module(cell, depth):
    parts = [re.sub(r'\[\d+\]', '', p) for p in cell.split('/')]
    parts = [p for p in parts if not p.endswith(('_reg', '_reg_rep')) and '_reg_' not in p][:depth]
    return '/'.join(parts) or cell


def main():
    path = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    groups = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (module(row['start'], depth), module(row['end'], depth))
            d = float(row['delay'])
            g = groups.setdefault(key, [0, 0.0, 0, row['start'], row['end']])
            g[0] += 1
            if d > g[1]:
                g[1], g[2], g[3], g[4] = d, int(row['levels']), row['start'], row['end']
    print(f"{'delay':>7} {'lv':>3} {'paths':>6}  start -> end")
    for (s, e), (n, d, lv, ex_s, ex_e) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"{d:7.3f} {lv:3d} {n:6d}  {s} -> {e}")


if __name__ == '__main__':
    main()
