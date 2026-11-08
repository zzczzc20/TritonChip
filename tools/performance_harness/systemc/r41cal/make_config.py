#!/usr/bin/env python3
"""The SystemC model's R41 configuration, from a device description and its per-core profile.

    python3 make_config.py --device spec/devices/u280-3core-r41.json --profiles spec/r1_profiles.json \
        --base configs/r41-hbm8-device-service.cfg --out configs/r41-sim.cfg [--sources configs/r41-sim.sources.json]

The target is the evaluator's RTL simulation of the device (the board-register
bench, 100 MHz, three HBM stand-ins of tb/common/tnca_axi_line_model.v), not
the card. Following the author's method (configs/r40-board-m2.sources.json),
every key is either read from the image (the device description and the
profile, so an agent's change to them reaches the model), derived from the
RTL's structure, read from the testbench, or a model control; nothing here is
fitted. The evidence for each value is the R41 structure inventory
(2026-11-08), which cites file:line for every row; the provenance written
next to the configuration names the source of each key.

Stage A (this file): only keys the model already has. What no key can express
is listed under "not expressible" in the provenance and is the next stage's
code: the hybrid LSU (line path + LSU_INFLIGHT word units), same-line spacing
at the home, per-outlet testbench latency and slot jitter, line-engine
eligibility, grid admission and the outlet drain at a launch's end.
"""
import argparse
import json
import re
from pathlib import Path

PROFILE = "core-vrf-sync-v5-x8-g2c4-dp-board"


def config(desc, prof):
    """{key: (value, class, source)}: every key this file decides."""
    u, p = desc["uncore"], desc["platform"]
    rp = prof["implementation"]["rtl_parameters"]
    el = prof["elaboration"]
    cores = int(p["core_count"])
    out = {}

    def put(k, v, cls, src):
        out[k] = (int(v), cls, src)

    # --- the image: device description (python/tnca/devices.py systemc_overrides, extended) ---
    put("nx", cores, "image", "device platform.core_count (3x1 mesh, board top NX)")
    put("ny", 1, "image", "board top: one row")
    put("sets", u["SETS"], "image", "device uncore.SETS -> home pipe SETS (tile)")
    put("ways", u["WAYS"], "image", "device uncore.WAYS")
    put("outlets", cores, "image", "one outlet per core (home s -> outlet s % PORTS, tnca_mem_ports.v)")
    if u["MEM_PORTS"] not in (1, cores):
        raise SystemExit(f"MEM_PORTS={u['MEM_PORTS']}: the model routes to one outlet or to all {cores}")
    put("active_outlets", u["MEM_PORTS"], "image", "device uncore.MEM_PORTS")
    put("outstanding", u["AXI_OUT_N"], "image", "device uncore.AXI_OUT_N: outlet slots and the admission limit (MMIO reset)")
    put("clock_hz", p["kernel_clock_mhz"] * 1_000_000, "image", "device platform.kernel_clock_mhz (bench clock #5 = 100 MHz)")
    mshr = u.get("HOME_MSHR_N", 8)
    put("mshrs", mshr, "image", "device uncore.HOME_MSHR_N -> MSHR_N (multicore shell)")
    put("targets", 2 * mshr if mshr > 8 else 16, "derived", "TARGET_N = HOME_MSHR_N>8 ? 2*HOME_MSHR_N : 16 (multicore shell)")
    put("noc_stages", u["NOC_LINK_STAGES"] + 1, "derived", "NOC_LINK_STAGES + router stage per XY hop (router/mesh unchanged since R40)")
    link = u["MEM_HOME_STAGES"] + u["LINK_STAGES"]
    put("tl_slice_to_outlet_cycles", 2 + link, "derived",
        "home issues its memory request at accept+3 (model: accept+1) + MEM_HOME_STAGES + LINK_STAGES link stages")
    put("tl_outlet_to_slice_cycles", link + 2, "derived",
        "MEM_HOME_STAGES + LINK_STAGES response registers + MS_FILL latch + fill-unit start (R40 FIT 13 retired)")
    issue = 1 if u.get("AXI_PIPELINED_ISSUE") else 2
    put("tl_outlet_read_issue_cycles", issue, "derived", "AXI_PIPELINED_ISSUE: one issue register, next line on the AR handshake")
    put("tl_outlet_write_issue_cycles", issue, "derived", "AXI_PIPELINED_ISSUE: a write leaves after AW and W")
    put("tl_outlet_beat_cycles", 512 // p["AXI_DATA_W"] - 1, "image", "AXI_DATA_W 512: one beat per 64-byte line")
    hybrid = bool(u.get("HYBRID_LSU"))
    pipe = int(u.get("HOME_LINE_PIPE", 0))
    if hybrid and pipe:
        put("tl_slice_hit_occupancy", 1, "derived", "home pipe takes one request per cycle (tnca_home_pipe.v)")
        put("tl_slice_hit_latency", 4, "derived", "accept, S1 tag read, S2 decide, S3 data -> response valid at +4")
        put("tl_slice_miss_occupancy", 1, "derived", "S2 allocates an MSHR in one cycle")
        put("tl_slice_mreq_occupancy", 0, "derived", "the memory request leaves from MSHR state on its own port")
        put("tl_slice_fill_occupancy", 0, "derived", "separate fill unit; pairs with wake 1")
        put("tl_slice_wake_occupancy", 1 if pipe >= 2 else 2, "derived",
            "one target per cycle; HOME_LINE_PIPE 2 chains MSHR fills with no gap (1 leaves one)")
        put("tl_slice_writeback_occupancy", 0, "derived", "victims queue in S3 (4-entry WBQ), issued before reads")
        put("tl_full_line_nofill", u.get("HOME_LINE_WRITE_NOFILL", 0), "image", "device HOME_LINE_WRITE_NOFILL")
    if hybrid:
        put("tl_core_beat_bytes", 64, "image",
            "HYBRID_LSU: 512-bit core port, one line request per line-eligible access (the model applies it to every access: stage B)")
        put("tl_noc_payload_bytes", 64, "derived", "one 64-byte line per NoC flit (NOC_PAYLOAD_W)")
    if u.get("TENSOR_LINE"):
        put("dma_line_engine", 1, "image", "device TENSOR_LINE -> transfer engine LINE_ENGINE (eligibility: stage B)")
        put("dma_line_buffers", u.get("LINE_SLOTS", 64), "derived",
            "LSLOT_N = 2^(MTAG_W-1), the port tag width from LINE_SLOTS (inferred)")
        put("dma_queue", 8, "derived", "line-eligible transfer commands overlap up to CMD_N 8 (word walk: 1)")

    # --- the image: per-core profile (rtl_parameters / elaboration) ---
    if el.get("MAC_DP"):
        cuts = int(el.get("MAC_CUTS", 0)) & 31
        put("mtx_dp", 1, "image", "profile elaboration.MAC_DP")
        put("mtx_pipe_cycles", 1 + bin(cuts).count("1") + 1, "derived",
            "PIPE = OPND_REG + popcount(MAC_CUTS[4:0]) on the MAC lane, drain PIPE+1 (tnca_matrix.v)")
        put("mtx_dp_extra_latency", 0, "derived", "dp and plain m.mma share the MAC lane and its PIPE")
        put("mtx_dp_prefetch", 0, "derived", "each group step reads B before its MACs")
    reg = int(rp.get("MTX_VRF_REG", 0))
    put("mtx_rf_read_cycles", 2 + reg, "image", "profile MTX_VRF_REG registers every matrix VRF read (2 + MTX_VRF_REG)")
    put("mtx_mac_unbuffered_cycles", 2 + reg, "image", "one matrix VRF read per unbuffered MAC (2 + MTX_VRF_REG)")
    put("mtx_abuf", rp.get("MTX_ABUF", 0), "image", "profile MTX_ABUF")
    put("vrf_prefetch", rp.get("VRF_PREFETCH", 0), "image", "profile VRF_PREFETCH")

    # --- derived from the core's RTL (inventory 1.3-1.7) ---
    for k, v, src in [
        ("scalar_latency", 1, "X_PROD_S forwarding"),
        ("predicate_latency", 1, "single-cycle VALU op"),
        ("reduction_latency", 3, "integer reductions are in the late class, no X_PROD"),
        ("fp_reduction_latency", 16, "4 levels x (FP_LAT+1), then done -> red_pend -> W"),
        ("sfu_occupancy", 27, "SFU pass VLEN+LAT plus the core's done -> red_pend -> writeback (author's reading 22; open)"),
        ("sfu_latency", 28, "as sfu_occupancy (author's reading 23; open)"),
        ("barrier_latency", 1, "mb.arm / mb.test write in X"),
        ("reduction_vector_extra", 1, "integer reduction broadcast in the late class"),
        ("dispatch_cycles", 6, "per-program dispatch (R40 derivation, path unchanged), DISPATCH_STAGES 2"),
        ("launch_cycles", 5, "per program (grid admission +4 per grid launch: stage B)"),
    ]:
        put(k, v, "derived", src)

    # --- the testbench (tnca_multicore_board_regs.v / tnca_axi_line_model.v), as expectations ---
    lat = [25 + 19 * q for q in range(cores)]
    jit_r = sum((s * 11) % 31 for s in range(31)) / 31
    jit_w = sum((s * 7) % 23 for s in range(23)) / 23
    put("hbm_latency_cycles", round(sum(lat) / cores + 2 + jit_r + 0.45), "testbench",
        f"mean over outlets of LATENCY 25+19p ({lat}) + 2 + mean slot jitter (id*11)%31 ({jit_r:.1f}) + mean arready wait 0.45;"
        " per-outlet latency and slot identity: stage B")
    put("tl_hbm_write_latency_cycles", round(sum(lat) / cores + 16 + jit_w + 1.08), "testbench",
        f"mean LATENCY + 15 + 1 + mean (id*7)%23 ({jit_w:.1f}) + mean AW/W wait 1.08")
    put("host_launch_cycles", 0, "testbench", "the count runs from MMIO decode to done: host time is outside it")
    put("host_transfer_cycles", 0, "testbench", "as host_launch_cycles")
    put("host_transfer_cycles_per_kib", 0, "testbench", "as host_launch_cycles")
    put("dram_path", 0, "testbench", "the testbench HBM has fixed latency and no banks")

    # --- model controls ---
    put("memory_model", 1, "model", "timeline memory: the only path with line-granular ports")
    put("core_pipeline", 1, "model", "core pipeline: units, freezes, park, writeback steal")
    return out


NOT_EXPRESSIBLE = {
    "LSU_INFLIGHT": "word-path units of the hybrid LSU (tl_lsu_units refuses 1 and applies to every access)",
    "HYBRID_LSU line eligibility": "only aligned vector accesses inside one line take the line path; the rest take the word unit",
    "memory issue limit": "one memory instruction per 3 cycles per core",
    "path-switch drain": "switching between line and word paths drains the other",
    "home same-line spacing": "two requests to one line at least 2 cycles apart; RQ_N 16 request buffer",
    "per-outlet testbench latency": "25 / 44 / 63 + slot jitter by lowest free slot; one hbm_latency_cycles is a mean",
    "line-engine eligibility": "only aligned whole-line boxes take the line path",
    "grid admission and outlet drain": "+4 per grid launch; done waits for every outlet write response",
    "FP-reduction fold occupancy, VALU late class, writeback-slot reservation": "core datapath details",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", type=Path)
    a = ap.parse_args()
    desc = json.loads(a.device.read_text())
    prof = json.loads(a.profiles.read_text())["profiles"][a.profile]
    keys = config(desc, prof)
    lines, seen = [], set()
    head = [f"# R41 RTL-simulation configuration of {desc['name']} / {a.profile}: made by r41cal/make_config.py",
            "# from the device description and the profile; provenance of every key in the .sources.json beside it.",
            f"# Base for the keys this file does not decide: {a.base.name}.", ""]
    for line in a.base.read_text().splitlines():
        m = re.match(r"^([a-z_0-9]+)=(-?\d+)\s*$", line)
        if m and m.group(1) in keys:
            lines.append(f"{m.group(1)}={keys[m.group(1)][0]}")
            seen.add(m.group(1))
        elif not line.startswith("#"):
            lines.append(line)
    extra = [k for k in keys if k not in seen]
    if extra:
        lines += ["", "# R41 keys the base file does not have"] + [f"{k}={keys[k][0]}" for k in extra]
    a.out.write_text("\n".join(head + lines).rstrip() + "\n")
    if a.sources:
        src = {"_note": "Provenance of r41cal/make_config.py's keys: image (device description or profile), derived "
                        "(RTL structure), testbench, model (control). Evidence with file:line: the R41 structure "
                        "inventory of 2026-11-08. Keys not listed keep the base file's value.",
               "_base": a.base.name, "_device": desc["name"], "_profile": a.profile,
               "_not_expressible": NOT_EXPRESSIBLE}
        src.update({k: {"value": v, "class": c, "source": s} for k, (v, c, s) in keys.items()})
        a.sources.write_text(json.dumps(src, indent=1) + "\n")
    print(f"{len(keys)} keys decided ({len(extra)} added) -> {a.out}")


if __name__ == "__main__":
    main()
