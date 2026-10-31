#!/usr/bin/env python3
"""Lockstep RTL regression for the optional matrix VRF response register."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

PARAMS = [
    "USE_BANKED_VRF=1", "VRF_BYPASS=1", "FE_NOBUBBLE=1",
    "VRF_PREFETCH=1", "RCTX_SWITCH=2", "LSU_INFLIGHT=2",
    "FPU_OVERLAP=1", "RSLICE=1", "NUM_RCTX=4", "RCTX_W=2",
    "PHYS_W=8", "HAS_TENSOR_XPOSE=2", "RPAIR=1", "BAR_N=8",
    "BAR_W=3", "BAR_EPOCH_W=38",
]


def run(cmd, log, cwd=ROOT):
    text = subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)
    with log.open("w") as fh:
        fh.write(f"$ {text}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode:
        tail = log.read_text().splitlines()[-30:]
        raise RuntimeError(f"command failed ({proc.returncode}): {text}\n" + "\n".join(tail))
    return log.read_text()


def events(text):
    out = []
    for line in text.splitlines():
        if not line.startswith("["):
            continue
        line = re.sub(r"^\[\d+\] ", "", line)
        line = re.sub(r"cyc=\d+ ", "", line)
        out.append(line)
    return out


def metric(text, name):
    match = re.search(rf"^{name}=(\d+)", text, re.M)
    if not match:
        raise RuntimeError(f"missing {name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"sim/build/mtx-vrf-reg")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"failure.txt").exists():
        (out/"failure.txt").replace(out/"previous-failure.txt")
    subprocess.run(["make", "sim/build/t11_matrix.hex", "sim/build/t25_mmasf.hex"],
                   cwd=ROOT, check=True)
    records = []
    try:
        for abuf in (0, 1):
            for memory in ("normal", "ooo"):
                outputs = {}
                for fixed in (0, 1):
                    tag = f"a{abuf}-{memory}-r{fixed}"
                    vvp = out/f"{tag}.vvp"
                    params = PARAMS + [f"MTX_ABUF={abuf}", f"MTX_VRF_REG={fixed}"]
                    if memory == "ooo":
                        params += ["MEM_LATENCY=6", "RANDOM_STALL=1", "MEM_REORDER=1"]
                    cmd = ["iverilog", "-g2005", "-I", "rtl/common", "-y", "rtl/core",
                           "-y", "rtl/common", "-y", "rtl/gen", "-y", "tb/common"]
                    for param in params:
                        cmd += ["-P", f"tb_tnca_core.{param}"]
                    cmd += ["-o", str(vvp), "tb/core/tb_tnca_core.v"]
                    run(cmd, out/f"{tag}-compile.log")
                    outputs[fixed] = vvp
                for program in ("t11_matrix", "t25_mmasf"):
                    texts = {}
                    for fixed in (0, 1):
                        log = out/f"a{abuf}-{memory}-r{fixed}-{program}.log"
                        cmd = ["vvp", str(outputs[fixed]), f"+hex=sim/build/{program}.hex",
                               "+maxcycles=400000", "+trace", "+mtxtrace"]
                        texts[fixed] = run(cmd, log)
                        if "FAIL" in texts[fixed] or not re.search(r"^PASS", texts[fixed], re.M):
                            raise RuntimeError(f"self-check failed: {log}")
                    base_events, fixed_events = events(texts[0]), events(texts[1])
                    if base_events != fixed_events:
                        for i, (a, b) in enumerate(zip(base_events, fixed_events)):
                            if a != b:
                                raise RuntimeError(f"event {i} differs: {a!r} != {b!r}")
                        raise RuntimeError(f"event counts differ: {len(base_events)} != {len(fixed_events)}")
                    reads = [metric(texts[f], "mtx_vrf_reads") for f in (0, 1)]
                    cycles = [metric(texts[f], "cycles") for f in (0, 1)]
                    counters = [{k: int(v) for k, v in re.findall(r"(cycles|issued|hazard|lsu|wb|squash)=(\d+)",
                                re.search(r"^cycles=.*$", texts[f], re.M).group(0))} for f in (0, 1)]
                    if reads[0] != reads[1]:
                        raise RuntimeError(f"accepted matrix reads differ: {reads}")
                    for name in ("issued", "hazard", "wb", "squash"):
                        if counters[0][name] != counters[1][name]:
                            raise RuntimeError(f"unexplained {name} delta: {counters}")
                    lsu_delta = counters[1]["lsu"] - counters[0]["lsu"]
                    expected = reads[0] + lsu_delta
                    if cycles[1] - cycles[0] != expected:
                        raise RuntimeError(f"cycle delta {cycles[1]-cycles[0]} != reads {reads[0]} + LSU delta {lsu_delta}")
                    record = dict(abuf=abuf, memory=memory, program=program, base_cycles=cycles[0],
                                  fixed_cycles=cycles[1], delta=cycles[1]-cycles[0],
                                  accepted_reads=reads[0], lsu_stall_delta=lsu_delta,
                                  compared_events=len(base_events), status="PASS")
                    records.append(record)
                    print(f"PASS lockstep abuf={abuf} mem={memory} {program}: "
                          f"cycles={cycles[0]}->{cycles[1]} delta={record['delta']} "
                          f"reads={reads[0]} lsu_delta={lsu_delta} events={len(base_events)}", flush=True)
    except Exception as exc:
        (out/"failure.txt").write_text(str(exc)+"\n")
        raise
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    identity = {
        "domain": "rtl-simulation",
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "exported-tree",
        "working_tree_sha256": hashlib.sha256(b"".join(
            p.read_bytes() for p in (ROOT/"rtl/core/tnca_core.v", ROOT/"tb/core/tb_tnca_core.v"))).hexdigest(),
        "records": records,
    }
    (out/"summary.json").write_text(json.dumps(identity, indent=1)+"\n")
    print(f"PASS: {len(records)} lockstep comparisons; raw logs and summary in {out}")


if __name__ == "__main__":
    sys.exit(main())
