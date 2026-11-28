#!/usr/bin/env python3
"""R42 issue probe: cost of a taken backward branch.  A loop of K independent vector instructions plus a scalar
counter decrement and `br.nz`, run for N and 2N iterations on the core bench variants; cycles per iteration =
(cycles(2N) - cycles(N)) / N, to be compared with K + 2 instructions.   (run on GB10 from the tree root)

  python3 probe_loops.py BUILDDIR OUT.tsv
"""
import os
import re
import subprocess
import sys

builddir, outtsv = sys.argv[1:3]
ASM = [sys.executable, "tools/asm/tnca_asm.py"]
N = 60
rows = []
for body in ("v2", "f3", "none"):
    for k in (0, 1, 2, 4, 8):
        if body == "none" and k != 0:
            continue
        cyc = {}
        for n in (N, 2 * N):
            lines = [".org 0", "start:", "    .li s5, #%d" % n, "    .li s1, #5", "    v.movi v1, #3", "    v.movi v2, #4", "    v.movi v3, #5",
                     "loop:"]
            for i in range(k):
                d = 16 + i
                lines.append(f"    v.add v{d}, v1, v2" if body == "v2" else f"    vf.fma v{d}, v1, v2, v3")
            lines += ["    s.subi s5, s5, #1", "    br.nz s5, loop", "    exit"]
            sfile = f"{builddir}/loop_{body}_{k}_{n}.s"
            open(sfile, "w").write("\n".join(lines) + "\n")
            hexf = sfile[:-2] + ".hex"
            subprocess.run(ASM + [sfile, "-o", hexf, "--size", "16384"], check=True, stdout=subprocess.DEVNULL)
            for v in ("run2", "fwd", "nopf", "behav"):
                r = subprocess.run(["nice", "-n", "19", "vvp", f"{builddir}/tb_{v}.vvp", f"+hex={hexf}", "+maxcycles=100000"],
                                   capture_output=True, text=True)
                m = re.search(r"^cycles=(\d+)", r.stdout, re.M)
                cyc[(v, n)] = int(m.group(1)) if m else None
                if "PASS" not in r.stdout:
                    print("FAIL", sfile, v)
        for v in ("run2", "fwd", "nopf", "behav"):
            per = (cyc[(v, 2 * N)] - cyc[(v, N)]) / N
            rows.append((body, k, v, per, k + 2))
with open(outtsv, "w") as fh:
    fh.write("body\tK\tvariant\tcycles_per_iteration\tinstructions_per_iteration\n")
    for r in rows:
        fh.write("%s\t%d\t%s\t%.2f\t%d\n" % r)
print(open(outtsv).read())
