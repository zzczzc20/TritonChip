# F07 unit OOC screen: slice MSHR service snapshot (C1, 658c811)

This screens fixed candidate F07 of the next implementation cohort (`frequency_preflight.md`):
- **Structure:** slice `cur_ms` / `ms_way` → data RAM.
- **Reference candidate:** C1, `658c811`.

It is a synthesis-only, out-of-context unit screen. It is not a routed result, and it does not certify the
candidate. No card or image was touched.

| | Base (current RTL) | C1 |
|---|---|---|
| `rtl/nuca/tnca_nuca_slice.v` | `a68ada2f…`, identical to 54249f7 and to F07's responsible-source hash | `4994243e…` (`git show 658c811:rtl/nuca/tnca_nuca_slice.v`) |
| Every other RTL file | the r41s1 tree | the same tree |

**How it was run.** Run on the FPGA host on 2026-10-21, 08:16–08:24 UTC, with `run_f07.sh` from this directory:
- `flow/synth/vivado_ooc_timing.tcl` with `OOC_SV=1`, unit `tnca_nuca_slice`, 10 ns, part `xcu280-fsvh2892-2L-e`;
- generics are the B40 image's slice parameters (`image.tcl`, tile instantiation): `SETS=32 SET_W=5 WAYS=8 WAY_W=3
  MSHR_N=8 MSHR_W=3 TARGET_N=16 TARGET_W=4 TAG_W=5 SLICE_ID=0 FULL_LINE_TAG=1 MAINTENANCE=1 REGISTER_VICTIM_OWNER=1
  REGISTER_VICTIM_SELECT=1 CACHE_DATA_PIPELINE=1 REGISTER_VICTIM_RANK=1 DCO_REGISTERED_RANGES=1`.

| Unrouted estimate, 10 ns | Base | C1 | Change |
|---|---:|---:|---:|
| WNS | +5.308 ns | +5.665 ns | +0.357 ns |
| Worst path | `ms_way_reg` → `data_mem` LUTRAM address, 19 levels, 4.560 ns | `cur_ms_reg_rep` → `data_mem` LUTRAM address, 16 levels, 4.203 ns | −3 levels, −0.357 ns |
| CLB LUTs | 30,211 | 30,785 | +574 (+1.9 %) |
| CLB registers | 4,692 | 4,997 | +305 |
| F7 muxes | 3,095 | 2,805 | −290 |
| Block RAM tiles | 0 | 0 | 0 (data RAM is distributed LUTRAM) |

**Regional resources.** There are three slices, one per SLR, so each SLR gains about +574 LUT and +305 FF. Against
the A3 routed CLB packing, the increase is roughly 0.13 % of an SLR's LUTs. That is small, but SLR0 was already at
95.24 % CLB packing (`frequency-preflight-v2`), so any SLR0 addition still carries packing risk.

**What this does and does not show.**
- C1 removes three logic levels from the slice data-RAM address path. It also moves the worst path's start from the
  live MSHR-way array to the registered `cur_ms` replica. This is the expected effect of capturing the selected
  MSHR's descriptor on the existing E_SEL edge.
- It does not show routed closure. A3's routed failure on this structure was net-dominated (R4 module: 7.757 ns of
  9.203 ns was routed net, address fanout 3,031), and an unrouted screen cannot measure that.
- F07 therefore stays a fixed-cohort candidate, with this unit screen recorded. Routed and regional qualification
  remain Stage 2 work.
- C1's functional equivalence evidence (cycle miter against 54249f7 and an independent oracle) is unchanged; see
  `658c811:evidence/r40/v2/convergence/mshr_service_snapshot/`.

`f07-ooc-evidence.tgz` holds the raw files: summary, path and utilization reports, Vivado logs and the run log.
