#!/usr/bin/env python3
"""Static instruction counts of a compiled Hub FP8 block-matmul kernel (.s as written by TNCA_KERNEL_DUMP=1).

usage: static_count.py KERNEL.s
Counts are of assembly lines in the K loop (label for_25 .. its back edge), computed here from the file.
'Fallback arm' is the byte-gather loop the compiler emits before each transposed transfer for windows whose
columns wrap; it is skipped at run time when the run-time check `ok` holds (the 'br.nz ok, x8dot_done_N')."""
import re
import sys
from collections import Counter

L = open(sys.argv[1]).read().split("\n")
ins = lambda l: (l.split("//")[0].strip() or None) if not l.split("//")[0].strip().endswith(":") else None
lab = {m.group(1): i for i, l in enumerate(L) if (m := re.match(r"^([A-Za-z_0-9]+):", l))}
start = lab["for_25"]
end = next(i for i, l in enumerate(L) if re.search(r"br\.nz\s+\w+, for_25", l))
body = [i for i in range(start + 1, end + 1) if ins(L[i])]
arms = []
for i in body:
    m = re.search(r"br\.nz\s+\w+, (x8dot_done_\d+)", L[i])
    if m:
        arms.append(sum(1 for k in range(i + 1, lab[m.group(1)]) if ins(L[k])))
op = Counter(re.sub(r"^@p\d\s+", "", ins(L[i])).split()[0] for i in body)
print(f"file lines {len(L)}; K loop lines {start + 1}..{end + 1}")
print(f"K-loop instructions (static) {len(body)}; fallback-arm instructions {sum(arms)} in {len(arms)} arms "
      f"({sorted(set(arms))} each); fast path {len(body) - sum(arms)}")
steps = op["t.ld.async.b.t"]
print(f"transposed weight transfers t.ld.async.b.t {steps}; gathered-fallback tile transfers t.ld.async.b "
      f"{op['t.ld.async.b']}; mma {op['m.mma.async.k32.e4m3']}; weight-side vector loads: none "
      f"(v.ld.gather.b {op['v.ld.gather.b']} are only in the fallback arms)")
fixed = len(body) - sum(arms)
print(f"weight bytes per K-loop iteration: {steps} transfers x 16 rows x 32 B = {steps * 512} B = {steps * 512 // 64} x 64 B")
print(f"fast-path instructions per iteration {fixed} = {fixed / (steps * 512 / 64):.1f} per 64 B of weights "
      f"({fixed / steps:.1f} per 512 B transfer)")
for k in ["v.ld.gather.w", "v.ld.w", "v.st.w", "v.st.b", "a.vread", "a.vwrite", "vf.cvt.f32.e4m3", "vf.cvt.f32.bf16",
          "vf.cvt.bf16.f32", "vf.fms", "vf.mul", "v.rcp.approx", "mb.arm", "mb.wait", "fence.acq_rel.device",
          "m.zero", "td.load"]:
    print(f"{k:24s} {op[k]}")
whole = Counter(re.sub(r"^@p\d\s+", "", ins(l)).split()[0] for l in L if ins(l))
print("whole file: td.load", whole["td.load"], "t.ld.async.b.t", whole["t.ld.async.b.t"])
