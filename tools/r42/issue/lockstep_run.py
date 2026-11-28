#!/usr/bin/env python3
"""R42 issue lockstep (GB10): run each program of a directory on two builds (reference, candidate) with 1, 2 and 4
contexts and compare the memory image written by the program and the cycle counts.

  python3 lockstep_run.py PROGDIR VVP_REF VVP_CAND OUT.tsv [roles list, e.g. 1,2,4] [extra plusargs...]

Results (the memory image) must be identical; the candidate is not required to be faster, the cycle counts are
reported (ratio) so that a slower program is visible.
"""
import hashlib
import os
import re
import subprocess
import sys

progdir, ref, cand, outtsv = sys.argv[1:5]
roles_list = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,4").split(",")]
extra = sys.argv[6:]
ASM = [sys.executable, "tools/asm/tnca_asm.py"]
tmp = os.path.join(os.path.dirname(os.path.abspath(outtsv)), "ls_tmp")
os.makedirs(tmp, exist_ok=True)


def run(vvp, hexf, roles, dump):
    args = ["nice", "-n", "19", "vvp", vvp, f"+hex={hexf}", "+maxcycles=400000", f"+dump={dump}", "+dumpfirst=512", "+dumpcount=768"]
    if roles > 1:
        args.append(f"+roles={roles}")
    r = subprocess.run(args + extra, capture_output=True, text=True)
    m = re.search(r"^cycles=(\d+)", r.stdout, re.M)
    ok = "PASS" in r.stdout and "FAIL" not in r.stdout
    h = hashlib.sha256(open(dump, "rb").read()).hexdigest()[:16] if os.path.exists(dump) else None
    return (int(m.group(1)) if m else None), ok, h, r.stdout


rows, bad = [], 0
for fn in sorted(os.listdir(progdir)):
    if not fn.endswith(".s"):
        continue
    hexf = os.path.join(tmp, fn[:-2] + ".hex")
    subprocess.run(ASM + [os.path.join(progdir, fn), "-o", hexf, "--size", "16384"], check=True, stdout=subprocess.DEVNULL)
    for roles in roles_list:
        c0, ok0, h0, o0 = run(ref, hexf, roles, os.path.join(tmp, "ref.dump"))
        c1, ok1, h1, o1 = run(cand, hexf, roles, os.path.join(tmp, "cand.dump"))
        same = (h0 == h1) and h0 is not None and ok0 and ok1
        if not same:
            bad += 1
            open(os.path.join(os.path.dirname(os.path.abspath(outtsv)), f"FAIL_{fn[:-2]}_r{roles}.txt"), "w").write(o0[-2000:] + "\n=====\n" + o1[-2000:])
        rows.append((fn[:-2], roles, c0, c1, h0, h1, ok0, ok1, same))
with open(outtsv, "w") as fh:
    fh.write("prog\troles\tcycles_ref\tcycles_cand\thash_ref\thash_cand\tpass_ref\tpass_cand\tidentical\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
n = len(rows)
slower = sum(1 for r in rows if r[2] and r[3] and r[3] > r[2])
import math
g = math.exp(sum(math.log(r[3] / r[2]) for r in rows if r[2] and r[3]) / max(1, sum(1 for r in rows if r[2] and r[3])))
print(f"runs {n}  mismatches {bad}  candidate slower in {slower}  geomean cand/ref cycles {g:.4f}")
sys.exit(1 if bad else 0)
