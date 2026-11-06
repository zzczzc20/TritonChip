#!/usr/bin/env python3
"""R41 run-2 throughput-matching model (analytic; no simulation is run here).

Every number is either (a) a constant read from RTL / spec with its source named in CONST,
(b) computed from a recorded raw file in this repository (read below), or (c) a stated assumption
in SCEN / ASSUME. Domains: rtl, spec, analytic, tlm, board. Nothing is typed in from a result.

    python3 tools/r41/balance_model.py --out evidence/r41/hbm_bw/balance/balance_model.json \
        [--update-md evidence/r41/hbm_bw/BALANCE.md]

--update-md rewrites the blocks between `<!-- BEGIN name -->` and `<!-- END name -->` in the document.
"""
import argparse
import json
import math
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HB = ROOT / "evidence/r41/hbm_bw"
KERNEL_TGZ = ROOT / "evidence/r41/fp8/kernels/fp8-kernels-evidence.tgz"
ASM = "asm/w8a8_block_dynamic_fp8_matmul_kernel/w8a8_block_dynamic_fp8_matmul_kernel.s"
KREPORT = "runs/final/report.json"
BOARD = ROOT / "evidence/r41/chat/board-51aef19-summary.json"
PLAN = {"vl": ROOT / "evidence/r41/fp8/qwen/vl/plan.json", "q35": ROOT / "evidence/r41/fp8/qwen/q35/plan.json"}
CHIPVAL = HB / "mma_dp/chip_validation.json"
CASES = HB / "mma_dp/cases_summary.json"

# ---- (a) constants: RTL / spec, with source ----------------------------------------------------
CONST = {
    "clock_hz": (100e6, "spec: 100 MHz run image (evidence/r41/hbm_bw/g4/run2/README.md)"),
    "cores": (3, "rtl: MEM_PORTS 3 / nx=3 (g4/run2/image.tcl, configs/r40-three-core.cfg)"),
    "vlen": (16, "rtl: tnca_matrix VLEN=16, one tnca_mac_lane per lane"),
    "step_overhead": (1, "rtl: tnca_matrix S_LOADB, one cycle per k-step (S_MAC -> S_LOADB)"),
    "rows": (16, "rtl: tnca_matrix ROWS=16, S_MAC sweeps mm=0..15"),
    "vreg_bytes": (64, "rtl: VLEN*ELEN = 512 bit"),
    "phys_vregs": (256, "rtl: image PHYS_VREGS 256 (4 contexts x 64)"),
    "acc_slots": (16, "rtl: tnca_matrix ACC_N=16"),
    "vrf_banks": (4, "rtl: tnca_core NUM_BANKS(4), one read and one write port per bank, 5 read requesters"),
    "sfu_elem_per_cycle": (1, "rtl: tnca_sfu one slice, one lane per cycle; image sets no SFU_LANES"),
    "issue_per_cycle": (1, "rtl/doc: one instruction per cycle per core (G4_DESIGN sec 1)"),
    "line_bytes": (64, "rtl: line = 64 B"),
    "nuca_bytes_per_home": (32 * 8 * 64, "rtl: image SETS 32 x WAYS 8 x 64 B"),
    "pipe_dp": (1 + 4, "rtl: OPND_REG + popcount(MAC_CUTS 0b11101) = result latency"),
    "pipe_legacy": (2 + 1, "rtl: FMA_LAT 2 + OPND_REG 1"),
}
C = {k: v[0] for k, v in CONST.items()}

# memory-path per-core ceilings (bytes/cycle/core). Source tags are part of the table output.
PATH = [
    ("task-brief read ceiling", 29.0, "brief: 'the 29 B/cycle/core read ceiling' (no recorded source found in the repo)"),
    ("task-brief copy-like ceiling", 38.0, "brief: '38 for copy-like' (no recorded source found in the repo)"),
    ("core port / home / outlet, 1 line per cycle", 64.0, "rtl/spec: one 64 B line per cycle (G4_DESIGN sec 6, 9.6; 512-bit outlet)"),
    ("vector-LSU path best measured (clone fp32)", 11.23, "rtl: G4_DESIGN sec 8.5, s4 image, 4-13 lines in flight"),
    ("transposed X8 tile (word walk)", 8.0, "rtl: G4_DESIGN sec 9.6, 8 B units, 8 B/cycle at most"),
    ("run-1 LSU memory-level parallelism", 0.88, "rtl+analytic: LINK_BUDGET sec 4 (2 commands x 32 B / 73 cycles)"),
]

# Linear layers per decoder layer: (name, K, N, count). From recorded shapes: the K/N pairs below are the
# ones listed in the archived fp8 evidence (models-tnca-evidence.tgz: K 2048/6144 N 2048/1024/6144 for
# VL; K 2560/4096/9216 N 8192/1024/4096/9216/2560 for Qwen3.5). Layer counts are checked against the
# recorded plan.json fp8_gemm_flops (see check_plan).
MODELS = {
    "vl": dict(
        title="Qwen3-VL-2B-Instruct-FP8", hidden=2048, vocab=151936, layers=28,
        groups=[dict(n=28, lin=[("q", 2048, 2048), ("k", 2048, 1024), ("v", 2048, 1024), ("o", 2048, 2048),
                                ("gate", 2048, 6144), ("up", 2048, 6144), ("down", 6144, 2048)])],
        attn=dict(layers=28, heads=16, kv_heads=8, hd=128), inter=6144, linattn_layers=0),
    "q35": dict(
        title="Qwen3.5-4B-FP8", hidden=2560, vocab=248320, layers=32,
        groups=[dict(n=8, lin=[("q+gate", 2560, 8192), ("k", 2560, 1024), ("v", 2560, 1024), ("o", 4096, 2560),
                               ("gate", 2560, 9216), ("up", 2560, 9216), ("down", 9216, 2560)]),
                dict(n=24, lin=[("qkv", 2560, 8192), ("z", 2560, 4096), ("out", 4096, 2560),
                                ("gate", 2560, 9216), ("up", 2560, 9216), ("down", 9216, 2560)])],
        attn=dict(layers=8, heads=16, kv_heads=4, hd=256), inter=9216, linattn_layers=24),
}
BLK = 128   # weight block and K block of the registered kernels (block_size [128,128])
TOK = 16    # token chunk of the compiled kernel (row_split chunk = VLEN)


def read_tar(name):
    with tarfile.open(KERNEL_TGZ) as t:
        names = {m.name.lstrip("./"): m for m in t.getmembers()}
        return t.extractfile(names[name]).read().decode()


# ---- (b) recorded inputs ------------------------------------------------------------------------
def asm_profile():
    lines = read_tar(ASM).splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "for_26:")
    end = next(i for i, l in enumerate(lines) if l.strip() == "endfor_27:")
    body = [l for l in lines[start + 1:end] if l.strip() and not l.strip().startswith("//")]
    mn = lambda l: re.sub(r"^\s*@p\d+\s+", "", l).split()[0]
    mmas = [i for i, l in enumerate(body) if mn(l).startswith("m.mma.async")]
    first, last = mmas[0], mmas[-1]
    # the instructions after the last mma that belong to the last window: its mb.wait and following scale code
    regions = {"pre_mma (act-quant, B scatter)": body[:first], "mma_window (weight row loads, mma)": body[first:last + 2],
               "post_mma (scale, accumulate, store)": body[last + 2:]}

    def cls(m):
        if m.startswith(("s.", "p.", "w.", "fence", "mb.", "br", ".")): return "scalar/predicate/ctl"
        if m.startswith(("v.ld", "v.st")): return "vector load/store"
        if m.startswith(("m.", "a.")): return "matrix/accumulator"
        if m.startswith("vf."): return "vector fp"
        return "vector int/move"
    out = {"loop_instr": len(body), "mma": len(mmas), "regions": {}, "mnemonics": {}, "classes": {}}
    for r, ls in regions.items():
        out["regions"][r] = len(ls)
    for l in body:
        m = mn(l)
        out["mnemonics"][m] = out["mnemonics"].get(m, 0) + 1
        out["classes"][cls(m)] = out["classes"].get(cls(m), 0) + 1
    return out


def engine_cycles():
    d = json.load(open(CHIPVAL))["results"]
    dp = {k: d["pairs"][f"dp-e4m3-e4m3-k{k}"]["matrix_engine_busy_cycles"] for k in (16, 32, 64)}
    leg = {k: d["fma_compare"][f"e4m3-k{k}"]["fma_matrix_engine_busy_cycles"] for k in (16, 32, 64) if f"e4m3-k{k}" in d["fma_compare"]}
    f16 = {k: d["fma_compare"][f"f16-k{k}"]["dp_matrix_engine_busy_cycles"] for k in (16, 32) if f"f16-k{k}" in d["fma_compare"]}
    f16leg = {k: d["fma_compare"][f"f16-k{k}"]["fma_matrix_engine_busy_cycles"] for k in (16, 32) if f"f16-k{k}" in d["fma_compare"]}
    return dict(dp_e4m3=dp, legacy_e4m3=leg, dp_f16=f16, legacy_f16=f16leg)


def tlm_points():
    rep = json.loads(read_tar(KREPORT))
    pts = {}
    for c in rep["cases"]:
        m = re.search(r"block-dynamic (\d+)x(\d+)x(\d+)", c["case"])
        if m and c["launches"]:
            pts[tuple(int(x) for x in m.groups())] = (c["launches"][0]["device_cycles"], c["launches"][0]["grid"])
    return pts, rep["knobs"]["TNCA_PROFILE"]


def board_decode():
    d = json.load(open(BOARD))
    c = next(s for s in d["steps"] if s["step"] == "vl-full-chat")["chat"]
    st = {s["kind"] + str(s["step"]): s for s in c["steps"]}
    return dict(prefill11=st["prefill0"]["device_cycles"], decode=st["decode2"]["device_cycles"],
                decode3=st["decode3"]["device_cycles"], wall=c["steady_decode_seconds"],
                prompt_tokens=c["prompt_tokens"], decode_weight_bytes=c.get("decode_weight_bytes"))


def check_plan(model):
    m = MODELS[model]
    p = json.load(open(PLAN[model]))
    st = {s["stage"]: s for s in p["stages"]}
    params = sum(g["n"] * sum(k * n for _, k, n in g["lin"]) for g in m["groups"])
    flops = 2 * params
    rec = st["decode_step"]["fp8_gemm_flops"]
    lm = m["hidden"] * m["vocab"] * 2
    return dict(params_fp8=params, flops_from_shapes=flops, flops_recorded=rec, rel_diff=(flops - rec) / rec,
                lm_head_bytes_bf16=lm, recorded_decode_weight_bytes_read=st["decode_step"]["graph_weight_bytes_read"],
                shapes_fp8_bytes_plus_lm=params + lm, prefill_prompt_tokens=p.get("prompt_tokens"))


# ---- (c) the model ------------------------------------------------------------------------------
def peak_macs():
    vl, st = C["vlen"], C["step_overhead"]
    rows = C["rows"]
    out = {}
    for name, g in (("legacy m.mma.async (any format; FP32 vector FMA is 16/cycle too)", 1), ("DP2 (FP16/BF16/mixed FP8)", 2), ("DP4 (E4M3 x E4M3)", 4)):
        raw = vl * g
        step = vl * g * rows / (rows + st)       # LOADB bubble per k-step
        out[name] = dict(group=g, macs_per_cycle_core=raw, macs_per_cycle_after_step_bubble=step,
                         chip_gmac_s=raw * C["cores"] * C["clock_hz"] / 1e9,
                         chip_tflops=2 * raw * C["cores"] * C["clock_hz"] / 1e12)
    return out


def iters(m, k, n, tok=TOK):
    return math.ceil(m / tok) * math.ceil(n / BLK) * math.ceil(k / BLK)


def model_iters_per_chunk(model):
    """FP8 K-block iterations per 16-token chunk, whole model (all layers)."""
    m = MODELS[model]
    return sum(g["n"] * sum(math.ceil(k / BLK) * math.ceil(n / BLK) for _, k, n in g["lin"]) for g in m["groups"])


def scenarios(prof, eng, tlm_t_iter):
    I = prof["loop_instr"]
    n_mma = prof["mma"]
    Imma = prof["regions"]["mma_window (weight row loads, mma)"]
    Ipre = prof["regions"]["pre_mma (act-quant, B scatter)"]
    Ipost = prof["regions"]["post_mma (scale, accumulate, store)"]
    t_leg = n_mma * eng["legacy_e4m3"][32]
    t_dp = n_mma * eng["dp_e4m3"][32]
    macs = TOK * BLK * BLK
    w_bytes = BLK * BLK                      # e4m3 weights of one iteration
    a_bytes = TOK * BLK * 2                  # bf16 activations read by the fused-quant kernel
    # "fixed" issue: pre-quantised activations (act-quant out of the N loop: assumed amortised to 0),
    # weight fragments by tensor-engine descriptor (one descriptor load + arm/wait per mma ~ 6 instr),
    # the scale epilogue as compiled today. Assumption, not a measurement.
    I_fixed = n_mma * 6 + Ipost
    S = {}
    S["run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine)"] = dict(domain="tlm", t_opt=tlm_t_iter, t_pess=tlm_t_iter)
    S["run-2, kernel as compiled, legacy engine (no DP)"] = dict(domain="analytic", issue=I, mma=t_leg, mem=(w_bytes + a_bytes) / PATH[0][1])
    S["run-2, kernel as compiled, DP4 engine"] = dict(domain="analytic", issue=I, mma=t_dp, mem=(w_bytes + a_bytes) / PATH[0][1])
    S["run-2, DP4, issue cut by software (tile loads, quant hoisted)"] = dict(domain="analytic projection", issue=I_fixed, mma=t_dp, mem=(w_bytes) / PATH[0][1])
    for s in S.values():
        if "issue" in s:
            s["t_opt"] = max(s["issue"], s["mma"], s["mem"])      # perfect overlap of the three
            s["t_pess"] = s["issue"] + s["mma"] + s["mem"]         # no overlap
        s["macs"] = macs
        s["util_dp4_peak_opt"] = macs / s["t_opt"] / (C["vlen"] * 4)
        s["util_dp4_peak_pess"] = macs / s["t_pess"] / (C["vlen"] * 4)
        s["binding"] = max(("issue", s.get("issue", 0)), ("mma", s.get("mma", 0)), ("mem", s.get("mem", 0)), key=lambda x: x[1])[0] if "issue" in s else "memory latency / LSU MLP (run-1)"
    return S, dict(I=I, Ipre=Ipre, Imma=Imma, Ipost=Ipost, I_fixed=I_fixed, t_leg=t_leg, t_dp=t_dp, macs=macs, w_bytes=w_bytes, a_bytes=a_bytes)


def e2e(model, scen, T_list, rest_per_lm_byte):
    m = MODELS[model]
    per_chunk = model_iters_per_chunk(model)
    lm_bytes = m["hidden"] * m["vocab"] * 2
    res = {}
    for sname, s in scen.items():
        for bound in ("t_opt", "t_pess"):
            ti = s[bound]
            row = {}
            for T in T_list:
                chunks = math.ceil(T / TOK)
                gemm = chunks * per_chunk * ti / C["cores"]
                # non-GEMM (analytic estimates, see doc): attention matmul on DP2 at 50 % of DP2 peak (bf16; the
                # same estimate for every scenario), softmax exp at 1 element/cycle/core, SwiGLU silu 2 SFU ops.
                a = m["attn"]
                att_mac = a["layers"] * 2 * a["heads"] * a["hd"] * T * (T + 1) / 2
                att = att_mac / (C["cores"] * 0.5 * C["vlen"] * 2)
                exp = a["layers"] * a["heads"] * T * (T + 1) / 2 / C["cores"] / C["sfu_elem_per_cycle"]
                silu = m["layers"] * T * m["inter"] * 2 / C["cores"] / C["sfu_elem_per_cycle"]
                other = att + exp + silu
                cyc = gemm + other
                row[T] = dict(gemm_cycles=gemm, other_cycles=other, att=att, exp=exp, silu=silu, cycles=cyc,
                              tokens_per_s=T / (cyc / C["clock_hz"]))
            # decode: one chunk (M=1 padded to 16), lm_head at the same weight bytes per cycle as the body
            body_iter = per_chunk
            body = body_iter * ti / C["cores"]
            body_bytes = body_iter * BLK * BLK
            lm_same = lm_bytes / body_bytes * body
            lm_path = lm_bytes / (PATH[0][1] * C["cores"])
            if sname.startswith("run-1"):
                # run-1: the non-body work (lm_head, attention, norms, launch gaps) is the board's decode cycles minus
                # the TLM-calibrated body, per lm_head byte; exact for Qwen3-VL by construction, extrapolated for Qwen3.5
                lm_path = lm_same = rest_per_lm_byte * lm_bytes
            row["decode"] = dict(body_cycles=body, lm_head_cycles_same_rate=lm_same, lm_head_cycles_path_bound=lm_path,
                                 tokens_per_s_lm_body_rate=C["clock_hz"] / (body + lm_same),
                                 tokens_per_s=C["clock_hz"] / (body + lm_path))
            res[(sname, bound)] = row
    return res


def ai_table(T_list):
    """Arithmetic intensity (FLOP per byte from NUCA/DRAM), per layer shape, current tiling vs a 4x4-chunk tile."""
    rows = []
    shapes = [("VL q/o", 2048, 2048), ("VL gate/up", 2048, 6144), ("VL down", 6144, 2048),
              ("Q35 gate/up", 2560, 9216), ("Q35 down", 9216, 2560), ("Q35 lin-attn qkv", 2560, 8192)]
    for name, k, n in shapes:
        for T in T_list + [1]:
            flop = 2 * T * n * k
            # current: weights re-read per 16-token chunk, bf16 activations re-read per 128-column program, bf16 out
            w = math.ceil(T / TOK) * n * k
            a = math.ceil(n / BLK) * T * k * 2
            o = T * n * 2
            cur = flop / (w + a + o)
            # 64 x 64 output tile (16 accumulator slots), activations pre-quantised to e4m3 (1 B), weights 1 B
            tm, tn = 64, 64
            w2 = math.ceil(T / tm) * n * k
            a2 = math.ceil(n / tn) * T * k
            fixed = flop / (w2 + a2 + o)
            rows.append(dict(shape=name, K=k, N=n, T=T, ai_current=cur, ai_tile64=fixed, w_bytes_current=w))
    return rows


def md_table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def f(x, d=0):
    return f"{x:,.{d}f}"


def build():
    prof = asm_profile()
    eng = engine_cycles()
    tlm, tlm_prof = tlm_points()
    cyc, grid = tlm[(16, 9216, 2560)]
    progs = grid[0] * grid[1]
    iters_per_core = progs / C["cores"] * (2560 // BLK)
    t_tlm = cyc / iters_per_core
    board = board_decode()
    plans = {m: check_plan(m) for m in MODELS}
    scen, aux = scenarios(prof, eng, t_tlm)
    T_list = [128, 512, 2048]
    it_vl0 = model_iters_per_chunk("vl")
    rest_per_byte = (board["decode"] - it_vl0 * t_tlm / C["cores"]) / plans["vl"]["lm_head_bytes_bf16"]
    res = {m: e2e(m, scen, T_list, rest_per_byte) for m in MODELS}
    out = dict(const={k: v for k, v in CONST.items()}, peak=peak_macs(), prof=prof, eng=eng, tlm_point=dict(cycles=cyc, grid=grid, profile=tlm_prof, t_iter=t_tlm),
               board=board, plans=plans, aux=aux, scen=scen, ai=ai_table(T_list), tlm_all={f"{a}x{b}x{c}": v[0] for (a, b, c), v in tlm.items()},
               tlm_grid={f"{a}x{b}x{c}": v for (a, b, c), v in tlm.items()})
    out["e2e"] = {m: {f"{s}|{b}": r for (s, b), r in res[m].items()} for m in res}
    # board cross-check of the run-1 per-iteration cost
    it_vl = model_iters_per_chunk("vl")
    out["board_check"] = dict(iters_vl=it_vl, t_iter_from_board_decode_upper=board["decode"] * C["cores"] / it_vl,
                              t_iter_from_board_prefill11_upper=board["prefill11"] * C["cores"] / it_vl,
                              decode_from_tlm_iter=it_vl * t_tlm / C["cores"],
                              rest_cycles=board["decode"] - it_vl * t_tlm / C["cores"],
                              lm_bytes=plans["vl"]["lm_head_bytes_bf16"])
    return out, res, scen, aux


def tables(out, res, scen, aux):
    T = {}
    pk = out["peak"]
    T["peak"] = md_table(["Mode (per core)", "G", "MAC/cycle (raw)", "after 1-cycle step bubble", "chip GMAC/s (3 cores)", "chip TFLOP/s-eq"],
                         [(n, v["group"], v["macs_per_cycle_core"], f(v["macs_per_cycle_after_step_bubble"], 1), f(v["chip_gmac_s"], 1), f(v["chip_tflops"], 4)) for n, v in pk.items()])
    p = out["prof"]
    T["asm"] = md_table(["Region of one K-block loop iteration", "instructions"], [(k, f(v)) for k, v in p["regions"].items()] + [("total loop body", f(p["loop_instr"]))]) \
        + "\n" + md_table(["Class", "instructions", "share"], [(k, f(v), f"{100 * v / p['loop_instr']:.1f} %") for k, v in sorted(p["classes"].items(), key=lambda x: -x[1])])
    roles = [("activation quantisation, distinctive", ["v.ld.h", "vf.cvt.bf16.f32", "vf.abs", "vf.max.ieee", "vf.reduce.max.b", "v.rcp.approx", "vf.cvt.f32.e4m3", "v.st.scatter.b"]),
             ("IEEE division glue (amax/448, x/scale)", ["vf.fms", "vf.neg", "vf.cmp.o", "v.sel"]),
             ("block scale and accumulate, distinctive", ["a.vread", "vf.add", "a.vwrite", "v.st.scatter.w"]),
             ("weight and scale loads", ["v.ld.w"]), ("lane extract / splat / bit-merge bookkeeping", ["v.extract", "v.splat", "v.or", "v.and", "v.mov", "v.movi", "v.lshr"]),
             ("scalar address arithmetic and predicates", ["s.addi", "s.add", "s.movz", "s.mov", "p.and"]),
             ("matrix command and wait", ["m.mma.async.k32.e4m3", "mb.arm", "mb.wait"]), ("vf.mul", ["vf.mul"])]
    mm = p["mnemonics"]
    T["mnem"] = md_table(["Role (my reading of the mnemonics)", "mnemonics", "instructions in the loop body"],
                         [(r, ", ".join(f"`{m}` {mm.get(m, 0)}" for m in ms), f(sum(mm.get(m, 0) for m in ms))) for r, ms in roles])
    e = out["eng"]
    T["eng"] = md_table(["K", "legacy `m.mma.async` e4m3 (busy cycles)", "`m.mma.dp.async` e4m3 x e4m3", "ratio", "MAC/cycle legacy", "MAC/cycle DP4", "DP4 % of 64"],
                        [(k, f(e["legacy_e4m3"][k]), f(e["dp_e4m3"][k]), f(e["legacy_e4m3"][k] / e["dp_e4m3"][k], 2), f(16 * 16 * k / e["legacy_e4m3"][k], 1), f(16 * 16 * k / e["dp_e4m3"][k], 1), f"{100 * 16 * 16 * k / e['dp_e4m3'][k] / 64:.0f} %") for k in (16, 32, 64)])
    rows = []
    for name, s in scen.items():
        rows.append((name, s["domain"], f(s.get("issue", 0)) if "issue" in s else "-", f(s.get("mma", 0)) if "mma" in s else "-", f(s.get("mem", 0)) if "mem" in s else "-",
                     f(s["t_opt"]), f(s["t_pess"]), s["binding"], f"{100 * s['util_dp4_peak_opt']:.1f}-{100 * s['util_dp4_peak_pess']:.1f} %"))
    T["scen"] = md_table(["Scenario (cycles per core per 16-token x 128 x 128 iteration)", "domain", "issue", "engine (32 cmds)", "memory", "T opt (overlap)", "T pess (serial)", "binding stage (opt)", "% of DP4 peak"], rows)
    out_rows = []
    for mname in MODELS:
        for (sn, b), r in res[mname].items():
            if b != "t_opt":
                continue
            out_rows.append((MODELS[mname]["title"], sn, *[f(r[Tn]["tokens_per_s"], 2) for Tn in (128, 512, 2048)], f(r["decode"]["tokens_per_s"], 3) + " (" + f(r["decode"]["tokens_per_s_lm_body_rate"], 3) + ")"))
    T["e2e_opt"] = md_table(["Model", "Scenario (T opt)", "prefill tok/s T=128", "T=512", "T=2048", "decode tok/s, lm_head at path ceiling (at body rate)"], out_rows)
    out_rows = []
    for mname in MODELS:
        for (sn, b), r in res[mname].items():
            if b != "t_pess":
                continue
            out_rows.append((MODELS[mname]["title"], sn, *[f(r[Tn]["tokens_per_s"], 2) for Tn in (128, 512, 2048)], f(r["decode"]["tokens_per_s"], 3) + " (" + f(r["decode"]["tokens_per_s_lm_body_rate"], 3) + ")"))
    T["e2e_pess"] = md_table(["Model", "Scenario (T pess)", "prefill tok/s T=128", "T=512", "T=2048", "decode tok/s, lm_head at path ceiling (at body rate)"], out_rows)
    ai = [r for r in out["ai"]]
    T["ai"] = md_table(["Shape (K x N)", "T tokens", "weights re-read (MB)", "AI current (FLOP/B)", "AI 64x64 pre-quant (FLOP/B)", "B/cycle/core needed at DP4 peak, current", "same, 64x64"],
                       [(f"{r['shape']} {r['K']}x{r['N']}", r["T"], f(r["w_bytes_current"] / 1e6, 1), f(r["ai_current"], 1), f(r["ai_tile64"], 1),
                         f(128 / r["ai_current"], 1), f(128 / r["ai_tile64"], 1)) for r in ai])
    # non-GEMM split for T=2048 under the DP4 as-compiled scenario and software-fixed
    rows = []
    for mname in MODELS:
        for sn in ("run-2, kernel as compiled, DP4 engine", "run-2, DP4, issue cut by software (tile loads, quant hoisted)"):
            for T_ in (512, 2048):
                r = res[mname][(sn, "t_opt")][T_]
                rows.append((MODELS[mname]["title"], sn, T_, f(r["gemm_cycles"] / 1e6, 0), f(r["att"] / 1e6, 1), f(r["exp"] / 1e6, 1), f(r["silu"] / 1e6, 1), f(100 * r["other_cycles"] / r["cycles"], 1) + " %"))
    T["other"] = md_table(["Model", "Scenario", "T", "GEMM Mcycles", "attention matmul Mcycles", "softmax exp Mcycles", "SwiGLU SFU Mcycles", "non-GEMM share"], rows)
    # ---- stage ladders --------------------------------------------------------------------------
    a = out["aux"]
    P = {n: v for n, _, v in [(x[0], None, x[1]) for x in PATH]}
    mac = a["macs"]
    pre_rows = [
        ("MAC lanes, DP4 peak (16 lanes x 4)", 64.0, "rtl"),
        ("MAC lanes, DP4 after the 1-cycle step bubble", out["peak"]["DP4 (E4M3 x E4M3)"]["macs_per_cycle_after_step_bubble"], "rtl (analytic)"),
        ("Engine, whole K32 command as measured", 16 * 16 * 32 / out["eng"]["dp_e4m3"][32], "tlm"),
        ("VRF operand supply: 8 B/cycle needed vs 64 B/cycle port, ABUF on", None, "rtl"),
        ("Vector issue, kernel as compiled (11,192 instr, IPC 1)", mac / a["I"], "analytic on asm"),
        ("  of which pre-mma region alone", mac / a["Ipre"], "analytic on asm"),
        ("  of which mma window (weight row loads) alone", mac / a["Imma"], "analytic on asm"),
        ("  of which post-mma scale/accumulate alone", mac / a["Ipost"], "analytic on asm"),
        ("Vector issue, software projection (2,652 instr)", mac / a["I_fixed"], "analytic projection"),
        ("Memory path 29 B/cycle at AI 25.3 FLOP/B (12.7 MAC/B)", 29.0 * 12.65, "analytic"),
        ("Memory path 29 B/cycle at 64x64 tile (31 MAC/B)", 29.0 * 31.1, "analytic"),
        ("Memory path 8 B/cycle (X8 word walk) at 12.7 MAC/B", 8.0 * 12.65, "analytic"),
    ]
    T["ladder_prefill"] = md_table(["Stage (prefill, per core)", "MAC/cycle sustained", "chip GMAC/s (3 cores, 100 MHz)", "domain"],
                                   [(n, "-" if v is None else f(v, 1), "-" if v is None else f(v * C["cores"] * C["clock_hz"] / 1e9, 2), d) for n, v, d in pre_rows])
    wb = a["w_bytes"]
    dec_rows = [
        ("Memory path, task-brief read ceiling", 29.0, "brief"),
        ("Memory path, vector-LSU best measured", 11.23, "rtl"),
        ("Memory path, X8 transposed word walk", 8.0, "rtl"),
        ("Engine geometry at M=1 (DP4 peak; 4 weight bytes per cycle)", 4.0, "rtl (analytic)"),
        ("Engine geometry at M=1 (32 K32 commands, 182 cycles each)", wb / a["t_dp"], "tlm + analytic"),
        ("Vector issue as compiled (16,384 B per 11,192 instr)", wb / a["I"], "analytic on asm"),
        ("Vector issue, software projection (2,652 instr)", wb / a["I_fixed"], "analytic projection"),
        ("Run-1 TLM point (16 KiB per 126,080 cycles)", wb / out["tlm_point"]["t_iter"], "tlm"),
    ]
    T["ladder_decode"] = md_table(["Stage (decode, weights consumed)", "B/cycle/core", "chip GB/s (3 cores, 100 MHz)", "domain"],
                                  [(n, f(v, 2), f(v * C["cores"] * C["clock_hz"] / 1e9, 2), d) for n, v, d in dec_rows])
    # decode memory bound
    rows = []
    for mname in MODELS:
        by = out["plans"][mname]["recorded_decode_weight_bytes_read"]
        for n, v, d in dec_rows[:5]:
            rows.append((MODELS[mname]["title"], n, f(by / 1e9, 3), f(C["clock_hz"] * C["cores"] * v / by, 3)))
    T["decode_bound"] = md_table(["Model", "Ceiling", "bytes per token (recorded plan.json)", "tokens/s at that ceiling"], rows)
    # sensitivity: VL prefill T=512 and decode
    sens = []
    per_chunk = model_iters_per_chunk("vl")
    for label, issue_n, ipc, cmd in [
            ("as compiled, IPC 1.0, cmd 182", a["I"], 1.0, 182), ("as compiled, IPC 0.7", a["I"], 0.7, 182), ("as compiled, IPC 0.5", a["I"], 0.5, 182),
            ("software projection, IPC 1.0, cmd 182", a["I_fixed"], 1.0, 182), ("software projection, IPC 0.5", a["I_fixed"], 0.5, 182),
            ("software projection, cmd 150 (RTL structural)", a["I_fixed"], 1.0, 150), ("software projection, cmd 220 (+20 %)", a["I_fixed"], 1.0, 220),
            ("software projection, issue 5,000 instr", 5000, 1.0, 182)]:
        ti = max(issue_n / ipc, 32 * cmd, 565)
        pp = max(issue_n / ipc, 32 * cmd, 565) + 0   # opt
        ps = issue_n / ipc + 32 * cmd + 565            # pess
        for nm, tt in (("opt", pp), ("pess", ps)):
            cyc = math.ceil(512 / TOK) * per_chunk * tt / C["cores"]
            sens.append((label, nm, f(tt), f(512 / (cyc / C["clock_hz"]), 2), f(C["clock_hz"] / (per_chunk * tt / C["cores"] + out["plans"]["vl"]["lm_head_bytes_bf16"] / (29.0 * C["cores"])), 3)))
    T["sens"] = md_table(["Case (Qwen3-VL-2B)", "overlap", "T iter (cycles)", "prefill tok/s at T=512 (GEMM only)", "decode tok/s"], sens)
    # board / TLM cross-check
    bc = out["board_check"]
    bd = out["board"]
    T["xcheck"] = md_table(["Quantity (Qwen3-VL-2B, run-1 image)", "value", "domain"], [
        ("FP8 K-block iterations per 16-token chunk, whole model (from shapes)", f(bc["iters_vl"]), "analytic"),
        ("TLM point 16x9216x2560: cycles / grid", f"{out['tlm_point']['cycles']:,} / {out['tlm_point']['grid']}", "tlm"),
        ("cycles per iteration per core from the TLM point", f(out["tlm_point"]["t_iter"]), "tlm"),
        ("board steady decode device cycles (calls 2, 3)", f(bd["decode"]) + " / " + f(bd["decode3"]), "board"),
        ("board prefill, 11 tokens, device cycles", f(bd["prefill11"]), "board"),
        ("cycles per iteration implied by the board decode (upper bound, all work attributed)", f(bc["t_iter_from_board_decode_upper"]), "board + analytic"),
        ("decode cycles predicted from the TLM iteration cost (body only)", f(bc["decode_from_tlm_iter"]), "tlm + analytic"),
        ("board minus predicted body (lm_head, attention, norms, launch gaps)", f(bc["rest_cycles"]), "board - model"),
        ("prediction / board", f(bc["decode_from_tlm_iter"] / bd["decode"], 3), "-"),
    ]) + "\n" + md_table(["TLM point (run-1 profile, block-dynamic kernel)", "grid", "cycles", "iterations per core", "cycles per iteration"],
                         [(k, "x".join(str(g) for g in v[1]), f(v[0]), f(iters(*[int(x) for x in k.split("x")]) / C["cores"]), f(v[0] / (iters(*[int(x) for x in k.split("x")]) / C["cores"])))
                          for k, v in out["tlm_grid"].items() if k in ("7x6144x2048", "7x2048x6144", "16x9216x2560", "129x2048x2048")])
    return T


def update_md(path, T):
    s = Path(path).read_text()
    for k, v in T.items():
        s = re.sub(rf"(<!-- BEGIN {k} -->\n).*?(<!-- END {k} -->)", lambda m: m.group(1) + v + m.group(2), s, flags=re.S)
    Path(path).write_text(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--update-md")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    out, res, scen, aux = build()
    T = tables(out, res, scen, aux)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    if a.update_md:
        update_md(a.update_md, T)
    if a.print or not (a.out or a.update_md):
        for k, v in T.items():
            print(f"## {k}\n{v}")
        print(json.dumps({k: out[k] for k in ("plans", "board_check", "tlm_point", "aux")}, indent=1, default=str))
