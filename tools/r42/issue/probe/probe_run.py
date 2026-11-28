#!/usr/bin/env python3
"""R42 issue probe runner (GB10). Assembles each probe program, runs it on the core bench variants with +trace and
reports issue-to-issue cycles per instruction of the stream (the last N instructions before `exit`).

  python3 probe_run.py PROGDIR BUILDDIR OUT.tsv [variant ...]
"""
import collections
import os
import re
import statistics
import subprocess
import sys

progdir, builddir, outtsv = sys.argv[1:4]
variants = sys.argv[4:] or ["run2", "nopf", "nobyp", "behav", "fwd"]
ROOT = os.getcwd()
ROLES = int(os.environ.get("ROLES", "1"))   # contexts started (each runs the same stream)
ASM = [sys.executable, os.path.join(ROOT, "tools/asm/tnca_asm.py")]
pat = re.compile(r"^\[\s*(\d+)\] rctx=(\d+) pc=([0-9a-f]+) maj=([0-9a-f]+) fn=([0-9a-f]+)")
rows = []
for fn in sorted(os.listdir(progdir)):
    if not fn.endswith(".s"):
        continue
    name = fn[:-2]
    if os.environ.get("PROGS") and not re.search(os.environ["PROGS"], name):
        continue
    hexf = os.path.join(builddir, name + ".hex")
    subprocess.run(ASM + [os.path.join(progdir, fn), "-o", hexf, "--size", "16384"], check=True,
                   stdout=subprocess.DEVNULL)
    n = int(re.search(r"N=(\d+)", open(os.path.join(progdir, fn)).readline()).group(1))
    for v in variants:
        r = subprocess.run(["nice", "-n", "19", "vvp", os.path.join(builddir, f"tb_{v}.vvp"), f"+hex={hexf}",
                            "+maxcycles=200000", "+trace"] + ([f"+roles={ROLES}"] if ROLES > 1 else []), capture_output=True, text=True)
        times = []
        for line in r.stdout.splitlines():
            m = pat.match(line)
            if m:
                times.append(int(m.group(1)))
        st = {}
        for line in r.stdout.splitlines():
            if line.startswith("cycles="):
                st.update(dict(kv.split("=") for kv in line.split()))
            if line.startswith("stall_total="):
                st.update(dict(kv.split("=") for kv in line.split()))
        passed = "PASS" in r.stdout and "FAIL" not in r.stdout
        # with ROLES > 1 every context runs the stream: the delta is then per instruction of one context and
        # the aggregate rate is ROLES / delta; use only rctx 0 trace lines
        t0 = []
        for line in r.stdout.splitlines():
            m = pat.match(line)
            if m and m.group(2) == "0":
                t0.append(int(m.group(1)))
        stream = t0[-(n + 1):-1]
        # aggregate issue rate of all contexts over the same window (instructions per cycle)
        deltas = [(b - a) // 10000 for a, b in zip(stream, stream[1:])]
        mid = deltas[20:-5]
        hist = collections.Counter(mid)
        rows.append((name, v, n, len(stream), statistics.mean(mid) if mid else float("nan"),
                     statistics.median(mid) if mid else float("nan"), dict(sorted(hist.items())),
                     st.get("cycles", "?"), st.get("x_hazard", "?"), st.get("x_vrf", "?"), st.get("x_fpu", "?"),
                     st.get("x_rsv", "?"), st.get("x_wb", "?"), st.get("x_vrfwb", "?"), st.get("x_frontend", "?"),
                     passed))
with open(outtsv, "w") as fh:
    fh.write("prog\tvariant\tN\tseen\tmean_cyc_per_instr\tmedian\thist\ttotal_cycles\tx_hazard\tx_vrf\tx_fpu\tx_rsv\tx_wb\tx_vrfwb\tx_frontend\tpass\n")
    for r in rows:
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in r) + "\n")
print(open(outtsv).read())
