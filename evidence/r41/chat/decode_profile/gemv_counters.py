#!/usr/bin/env python3
"""Per-core TLM counters of FP8 GEMV launches in one steady decode step of a decode_profile.py run.
usage: gemv_counters.py launches.json.gz STEP [INDEX ...]  (default: every w8a8 launch of the step)"""
import gzip, json, re, sys
rows = json.load(gzip.open(sys.argv[1]))["launches"]
step = int(sys.argv[2]); want = {int(x) for x in sys.argv[3:]}
for r in rows:
    if r["step"] != step or not r["counters"]:
        continue
    if want and r["index"] not in want or not want and not r["kernel"].startswith("w8a8"):
        continue
    c = r["counters"]; cyc = r["device_cycles"]
    g = lambda p: sum(v for k, v in c.items() if re.fullmatch(p, k))
    pc = lambda k: [c.get(f"core_{i}_{k}", 0) for i in range(3)]
    w = sum(a["view_bytes"] for a in r["args"] if a.get("role") == "weight")
    steps = max(pc("class_xfer")) // 2
    print(f"#{r['index']} {r['kernel'][:44]} grid {r['grid']} cycles {cyc} weight_bytes {w}")
    print(f"  weight B/cycle (chip) {w / cyc:.3f}; TTE bytes {g(r'core_\d+_dma_read_words') * 8} "
          f"(= {g(r'core_\d+_dma_read_words') * 8 / w:.3f} x weights); NUCA bytes {g(r'home_\d+_(hits|misses|merges)') * 8} "
          f"({g(r'home_\d+_(hits|misses|merges)') * 8 / cyc:.3f} B/cycle)")
    print(f"  DRAM reads {g(r'outlet_\d+_reads') * 64} B ({g(r'outlet_\d+_reads') * 64 / w:.2f} x weights), "
          f"writes {g(r'outlet_\d+_writes') * 64} B; home hits {g(r'home_\d+_hits')} misses {g(r'home_\d+_misses')} merges {g(r'home_\d+_merges')}")
    for k in ["programs_completed", "instructions", "class_lsu", "class_xfer", "class_mma", "class_a_vread",
              "dma_read_words", "barrier_sleep_cycles", "dependency_wait_cycles", "matrix_engine_busy_cycles"]:
        print(f"  core_{{0,1,2}}_{k}: {pc(k)}")
    print(f"  critical core steps (xfer/2) {steps}: cycles per 512 B transfer step {cyc / steps:.0f}; "
          f"instructions/cycle (core 1) {pc('instructions')[1] / cyc:.3f}")
