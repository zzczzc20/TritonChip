#!/usr/bin/env python3
"""Per-kernel and per-class tables for one steady decode step, from a decode_profile.py run.

Reads RUN/out/profile/launches.json and RUN/out/report.json (raw, as the run wrote them) and computes:
  - per launch: class, device cycles, demand bytes by role (the argument footprint: weights, activations, KV cache,
    linear-attention state), TLM-measured NUCA word requests and DRAM line traffic, B/cycle, code-store loads;
  - per class: share of decode cycles and of demand bytes, per decoder layer and scaled to full depth.

Scaling to full depth is analytic: full = fixed part + (full layers / profiled layers) x per-layer part, where a
launch belongs to the per-layer part when any of its arguments is a decoder-layer parameter or a layer's
cache/state, or, with no parameter or cache argument, when the launch before it does. The profiled layer
pattern must repeat in the full model (Qwen3-VL: every layer alike; Qwen3.5: 3 linear-attention + 1
full-attention layer, repeated), which --period-check verifies against the layer types the caller passes.

Classification is by the kernel name and its arguments only (rules in CLASS_RULES, first match wins). Fused
Inductor kernels carry several operations; the full name of each kernel is in the per-kernel table so the
assignment can be checked.
"""
import argparse
import json
from pathlib import Path
import re

LINE = 64          # bytes per NUCA line / DRAM transaction (TLM timeline memory)
WORD = 8           # bytes per core word request into a NUCA home (TLM timeline memory, word())


def tokens(name):
    return set(t for t in re.split(r"_+", name) if t)


def classify(row):
    """(class, reason)."""
    name = row["kernel"]
    t = tokens(name)
    roles = {a.get("role") for a in row["args"]}
    wdtypes = {a["dtype"] for a in row["args"] if a.get("role") == "weight"}
    if "fp8" in name and "matmul" in name:
        return "fp8_gemv", "Hub FP8 block matmul (activation quant and block scales fused inside)"
    if "quant" in t or "dequant" in t:
        return "dequant_scale", "separate quant/dequant kernel"
    if "state" in roles or t & {"convolution", "cumsum", "softplus", "triangular", "solve", "sizes"}:
        return "linear_attn", ("touches a linear-attention state, or conv/cumsum/softplus/triangular-solve, or the "
                               "qkv split (split_with_sizes) that only the linear-attention block has")
    if t & {"mm", "addmm"} and "weight" in roles and wdtypes & {"bfloat16", "float16", "float32"}:
        return "bf16_gemv", "Inductor matmul template over an unquantized weight"
    if "kv" in roles or ("scaled" in t and "attention" in t) or "bmm" in t:
        return "attention", "touches the KV cache, or SDPA/bmm"
    if "rsqrt" in t and "cat" in t:
        return "norm_rotary", "rmsnorm fused with rotary (cat)"
    if "cat" in t and "neg" in t:
        return "rotary", "rotate_half (cat, neg)"
    if "rsqrt" in t or {"mean", "pow"} <= t:
        return "norm", "rmsnorm (mean, pow, rsqrt), with any fused residual add"
    if t & {"silu", "sigmoid", "gelu", "exp"}:
        return "act_gate", "silu/sigmoid/exp gating"
    if name == "_copy_kernel" or t <= {"triton", "poi", "red", "fused", "to", "copy", "clone", "view", "transpose",
                                       "slice", "cat", "zeros", "expand", "unsqueeze", "select", "unsafe", "add",
                                       "arange", "index", "squeeze", "split", "with", "sizes", "constant", "pad",
                                       "nd", "like"} | {str(i) for i in range(1000)}:
        return "copy_cast", "data movement only (copy/clone/cast/view family)"
    return "other", "no rule matched"


CLASS_ORDER = ["fp8_gemv", "bf16_gemv", "dequant_scale", "attention", "norm", "norm_rotary", "rotary", "act_gate",
               "linear_attn", "copy_cast", "other"]
CLASS_TEXT = {
    "fp8_gemv": "FP8 matmul/GEMV (weights; act-quant + block dequant fused)",
    "bf16_gemv": "BF16 matmul/GEMV (unquantized weights, incl. lm_head)",
    "dequant_scale": "dequant/scale (separate kernels)",
    "attention": "attention (QK, softmax, AV, KV write/read)",
    "norm": "rmsnorm and other norms (+ residual add / embedding lookup)",
    "norm_rotary": "rmsnorm fused with rotary",
    "rotary": "rotary",
    "act_gate": "activation/gating (silu*mul etc.)",
    "linear_attn": "linear-attention / conv ops (Qwen3.5)",
    "copy_cast": "copies, casts, clones",
    "other": "other",
}


def counter_sums(c):
    if not c:
        return None
    def tot(pat):
        return sum(v for k, v in c.items() if re.fullmatch(pat, k))
    return dict(
        nuca_requests=tot(r"home_\d+_(hits|misses|merges)"),
        nuca_bytes=tot(r"home_\d+_(hits|misses|merges)") * WORD,
        dram_bytes=(tot(r"outlet_\d+_reads") + tot(r"outlet_\d+_writes")) * LINE,
        dram_read_bytes=tot(r"outlet_\d+_reads") * LINE,
        tte_commands=tot(r"core_\d+_dma_commands_issued"),
        tte_read_words=tot(r"core_\d+_dma_read_words"),
        tte_read_bytes=tot(r"core_\d+_dma_read_words") * WORD,
        mma=tot(r"core_\d+_class_mma"),
        lsu=tot(r"core_\d+_class_lsu"),
        xfer=tot(r"core_\d+_class_xfer"),
        instructions=tot(r"core_\d+_instructions"),
        dma_admission_wait=tot(r"core_\d+_dma_admission_wait_cycles"),
        barrier_sleep=tot(r"core_\d+_barrier_sleep_cycles"),
        matrix_busy=tot(r"core_\d+_matrix_engine_busy_cycles"),
    )


def demand(row):
    seen, out = set(), dict(weight=0, act=0, kv=0, state=0)
    for a in row["args"]:
        role = a.get("role")
        if role not in out:
            continue
        key = (a["storage"], tuple(a["shape"]), tuple(a["stride"]))
        if key in seen:
            continue
        seen.add(key)
        out[role] += a["view_bytes"]
    out["total"] = sum(out.values())
    return out


def layer_of(row):
    for a in row["args"]:
        n = a.get("name") or ""
        m = re.search(r"layers\.(\d+)\.", n) or re.match(r"layer(\d+)\.", n)
        if m:
            return int(m.group(1))
    return None


def analyse(run, full_layers, step=None):
    prof = json.loads((run / "out/profile/launches.json").read_text())
    report = json.loads((run / "out/report.json").read_text())
    steps = [s for s in report["steps"] if s["kind"] == "decode"]
    decode_steps = sorted({r["step"] for r in prof["launches"] if r["step"] >= 1})
    # Steady decode: the last decode step (no compilation; chat.py refuses recompilation from the second decode).
    step = step if step is not None else decode_steps[-1]
    rows = [r for r in prof["launches"] if r["step"] == step]
    profiled_layers = report["decoder_layers"]
    current = None
    out = []
    for r in rows:
        cls, why = classify(r)
        lay = layer_of(r)
        has_param = any(a.get("role") in ("weight", "kv", "state") for a in r["args"])
        if lay is not None:
            current = lay
            part = "layer"
        elif has_param:
            current = None
            part = "fixed"
        else:
            part = "layer" if current is not None else "fixed"
        d = demand(r)
        cs = counter_sums(r["counters"])
        if cs is None:
            raise SystemExit(f"launch {r['index']} has no counters; this table needs the TLM backend")
        cyc = r["device_cycles"]
        out.append(dict(index=r["index"], kernel=r["kernel"], cls=cls, why=why, part=part, layer=lay,
                        grid=r["grid"], cycles=cyc, demand=d, counters=cs,
                        demand_B_per_cycle=d["total"] / cyc if cyc else None,
                        dram_B_per_cycle=(cs["dram_bytes"] / cyc) if cyc and cs["dram_bytes"] is not None else None,
                        nuca_B_per_cycle=(cs["nuca_requests"] * WORD / cyc) if cyc and cs["nuca_requests"] is not None else None,
                        code_load=r["code_load"], code_bytes=r["code_bytes"], evicted=r["evicted"]))
    scale = full_layers / profiled_layers
    step_report = next(s for s in report["steps"] if s["step"] == step)
    return dict(run=str(run), model=report["model"], profiled_layers=profiled_layers, full_layers=full_layers,
                scale=scale, step=step, decode_steps=decode_steps, rows=out, report_step=step_report,
                all_decode_cycles=[s["transfers_inside_step"]["device_cycles"] for s in steps],
                weight_bytes_profiled=report["decode_weight_bytes"])


def aggregate(a):
    by = {}
    for r in a["rows"]:
        k = r["cls"]
        b = by.setdefault(k, dict(launches_fixed=0, launches_layer=0, cycles_fixed=0, cycles_layer=0,
                                   bytes_fixed=0, bytes_layer=0, weight_layer=0, weight_fixed=0,
                                   dram_fixed=0, dram_layer=0, loads_fixed=0, loads_layer=0,
                                   nuca_fixed=0, nuca_layer=0, tte_fixed=0, tte_layer=0, evict_fixed=0,
                                   evict_layer=0))
        p = r["part"]
        b["launches_" + p] += 1
        b["cycles_" + p] += r["cycles"] or 0
        b["bytes_" + p] += r["demand"]["total"]
        b["weight_" + p] += r["demand"]["weight"]
        b["dram_" + p] += r["counters"]["dram_bytes"] or 0
        b["loads_" + p] += int(r["code_load"])
        b["evict_" + p] += r["evicted"]
        b["nuca_" + p] += r["counters"]["nuca_bytes"]
        b["tte_" + p] += r["counters"]["tte_read_bytes"]
    s = a["scale"]
    for b in by.values():
        b["launches_full"] = b["launches_fixed"] + s * b["launches_layer"]
        b["cycles_full"] = b["cycles_fixed"] + s * b["cycles_layer"]
        b["bytes_full"] = b["bytes_fixed"] + s * b["bytes_layer"]
        b["weight_full"] = b["weight_fixed"] + s * b["weight_layer"]
        b["dram_full"] = b["dram_fixed"] + s * b["dram_layer"]
        b["loads_full"] = b["loads_fixed"] + s * b["loads_layer"]
        b["evict_full"] = b["evict_fixed"] + s * b["evict_layer"]
        b["nuca_full"] = b["nuca_fixed"] + s * b["nuca_layer"]
        b["tte_full"] = b["tte_fixed"] + s * b["tte_layer"]
    tot = {k: sum(b[k] for b in by.values()) for k in next(iter(by.values()))}
    return by, tot


def fmt(x, nd=0):
    if x is None:
        return "—"
    if isinstance(x, float) and nd:
        return f"{x:,.{nd}f}"
    return f"{x:,.0f}" if isinstance(x, (int, float)) else str(x)


def tables(a, clock_hz, rates_gbps, board=None):
    by, tot = aggregate(a)
    classes = [k for k in CLASS_ORDER if k in by]
    L = []
    L.append(f"#### Per kernel: steady decode step {a['step']}, {a['profiled_layers']} profiled layer(s) "
             f"(domain: TLM)\n")
    L.append("Footprint = bytes of the tensor arguments (weights / activations / KV cache or linear-attention "
             "state), each distinct view once. NUCA bytes = core word requests into the NUCA homes x 8 B "
             "(TLM-measured, includes re-reads); TTE bytes = the tensor-transfer engine's share of them; DRAM = "
             "64-byte lines at the outlets (reads + writes).\n")
    L.append("| # | Part | Class | Kernel | Grid | Cycles | Footprint W / A / KV B | NUCA B | NUCA B/cyc | "
             "TTE B | DRAM B | DRAM B/cyc | Code load / evicted |")
    L.append("|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|")
    for r in a["rows"]:
        d, c = r["demand"], r["counters"]
        L.append(f"| {r['index']} | {r['part']}{'' if r['layer'] is None else ' L' + str(r['layer'])} | {r['cls']} | "
                 f"`{r['kernel']}` | {'x'.join(map(str, r['grid']))} | {fmt(r['cycles'])} | "
                 f"{fmt(d['weight'])} / {fmt(d['act'])} / {fmt(d['kv'] + d['state'])} | {fmt(c['nuca_bytes'])} | "
                 f"{fmt(r['nuca_B_per_cycle'], 3)} | {fmt(c['tte_read_bytes'])} | {fmt(c['dram_bytes'])} | "
                 f"{fmt(r['dram_B_per_cycle'], 3)} | {'load' if r['code_load'] else '-'} / {r['evicted']} |")
    L.append("")
    L.append(f"#### Per class, scaled to {a['full_layers']} layers (TLM per-layer part x{a['scale']:g} + fixed part; "
             f"the scaling is analytic)\n")
    L.append("| Class | Launches/token | Cycles/token | Share of cycles | NUCA bytes/token | Share of NUCA bytes | "
             "NUCA B/cycle | TTE bytes/token | DRAM bytes/token | Code loads / evictions per token |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k in classes:
        b = by[k]
        L.append(f"| {CLASS_TEXT[k]} | {fmt(b['launches_full'])} | {fmt(b['cycles_full'])} | "
                 f"{100 * b['cycles_full'] / tot['cycles_full']:.1f}% | {fmt(b['nuca_full'])} | "
                 f"{100 * b['nuca_full'] / tot['nuca_full']:.1f}% | "
                 f"{fmt(b['nuca_full'] / b['cycles_full'] if b['cycles_full'] else None, 3)} | {fmt(b['tte_full'])} | "
                 f"{fmt(b['dram_full'])} | {fmt(b['loads_full'])} / {fmt(b['evict_full'])} |")
    L.append(f"| **total** | {fmt(tot['launches_full'])} | {fmt(tot['cycles_full'])} | 100% | {fmt(tot['nuca_full'])} | "
             f"100% | {fmt(tot['nuca_full'] / tot['cycles_full'], 3)} | {fmt(tot['tte_full'])} | {fmt(tot['dram_full'])} | "
             f"{fmt(tot['loads_full'])} / {fmt(tot['evict_full'])} |")
    L.append("")
    L.append(f"Predicted full-depth device time per token (analytic scaling of TLM): "
             f"{tot['cycles_full'] / clock_hz:.3f} s at {clock_hz / 1e6:g} MHz "
             f"({fmt(tot['cycles_full'])} cycles, {fmt(tot['launches_full'])} launches).")
    host = None
    if board:
        for b in board:
            h = (b['wall'] - b['cycles'] / clock_hz) / b['launches']
            host = h if host is None else max(host, h)
            L.append(f"\nBoard steady decode step {b['step']} (full depth; domain: board): {fmt(b['cycles'])} device "
                     f"cycles ({b['cycles'] / clock_hz:.3f} s), {b['launches']} launches, {b['wall']:.3f} s wall: "
                     f"{b['wall'] - b['cycles'] / clock_hz:.3f} s outside device time = "
                     f"{1e3 * h:.2f} ms per launch. TLM full-depth prediction / board device cycles = "
                     f"{tot['cycles_full'] / b['cycles']:.3f}.")
    small = sorted(r["cycles"] for r in a["rows"] if r["cycles"])
    L.append(f"\nSmallest launch in the step: {fmt(small[0])} cycles (TLM), an upper bound on the fixed device "
             f"cost of a launch; x {fmt(tot['launches_full'])} launches = {fmt(small[0] * tot['launches_full'])} "
             f"cycles/token.")
    L.append("")
    L.append("#### Bound analysis (analytic, from the TLM-measured NUCA bytes per class)\n")
    L.append("Seconds per token if a class's NUCA bytes moved at chip-level rate R and nothing else cost time. "
             "The last rows add the fixed per-launch costs: device (smallest launch above) and host (board "
             "ms per launch, worst steady step).\n")
    L.append("| Class | " + " | ".join(f"{r:g} GB/s" for r in rates_gbps) + " |")
    L.append("|---|" + "---:|" * len(rates_gbps))
    for k in classes:
        L.append(f"| {CLASS_TEXT[k]} | " + " | ".join(f"{by[k]['nuca_full'] / (r * 1e9):.4f}" for r in rates_gbps) + " |")
    t_bytes = [tot['nuca_full'] / (r * 1e9) for r in rates_gbps]
    t_dev = small[0] * tot['launches_full'] / clock_hz
    L.append("| **all classes, s/token** | " + " | ".join(f"{t:.4f}" for t in t_bytes) + " |")
    L.append("| **tokens/s, bytes only** | " + " | ".join(f"{1 / t:.3f}" for t in t_bytes) + " |")
    L.append(f"| **tokens/s, + device launch cost ({t_dev:.4f} s)** | " +
             " | ".join(f"{1 / (t + t_dev):.3f}" for t in t_bytes) + " |")
    wmin = sum(by[k]["weight_full"] for k in ("fp8_gemv", "bf16_gemv") if k in by)
    L.append(f"| *lower bound: each matmul weight read once ({fmt(wmin)} B; embedding table excluded), tokens/s* | " +
             " | ".join(f"{(r * 1e9) / wmin:.3f}" for r in rates_gbps) + " |")
    if host is not None:
        t_host = host * tot['launches_full']
        L.append(f"| **tokens/s, + device and board host launch cost ({t_host:.2f} s)** | " +
                 " | ".join(f"{1 / (t + t_dev + t_host):.4f}" for t in t_bytes) + " |")
    L.append("")
    return "\n".join(L), by, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory (with out/report.json and out/profile/launches.json)")
    ap.add_argument("--full-layers", type=int, required=True)
    ap.add_argument("--clock-hz", type=float, default=100e6)
    ap.add_argument("--rates", default="1,2,4,8,19")
    ap.add_argument("--board-report", type=Path,
                    help="a board run's chat report.json (raw); its steady decode steps are compared")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    a = analyse(args.run, args.full_layers)
    board = None
    if args.board_report:
        br = json.loads(args.board_report.read_text())
        board = [dict(step=s["step"], cycles=s["transfers_inside_step"]["device_cycles"],
                      launches=s["transfers_inside_step"]["launches"], wall=s["seconds"])
                 for s in br["steps"] if s["kind"] == "decode" and s["new_graphs"] == 0]
    text, by, tot = tables(a, args.clock_hz, [float(x) for x in args.rates.split(",")], board)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(dict(analysis=a, by_class=by, total=tot), indent=1) + "\n")


if __name__ == "__main__":
    main()
