## 5. Resources and timing (domain: OOC; tables T1-T6b below)

Flow (`tools/r42/kfuse_rtl/ooc.tcl`, `run_ooc.sh`, `run_queue.sh`; x86 host, at most two Vivado jobs at once, six threads each, no card):
`synth_design -mode out_of_context -flatten_hierarchy rebuilt` (T4: `none`), `create_clock` at 10 ns (also 5 ns and 4 ns) on an ideal source, `opt_design`,
`place_design`, `phys_opt_design`, `route_design`, then `report_utilization`, `report_timing_summary` and the worst path into every register. Tops are in
`tools/r42/kfuse_rtl/ooc_tops.v`: every port of the lane or array is a register of the wrapper (the operand latches before it, the accumulator write after it); the
lane is instantiated with `IN_REG=0`, so the wrapper's operand register is the lane's first register, as in the matrix engine. The flip-flop columns include those
wrapper registers (G=16 lane: 322 of 1,689). The R41 comparison lanes (`tnca_mac_lane`, `MAC_CUTS` 29, `IN_REG=0`) are run in the same flow, in this session.

Reading the tables:
- T1/T2 are the routed results; the array rows are 16 lanes with independent `a`, `b` and `c` per lane (no sharing of the broadcast `a`
  decode, which a real engine would share). An array is within +0.2..+0.9 % of 16 x its lane (linearity check under T2), so the R41 rows
  that are 16 x a lane are linear extrapolations, labelled so.
- The R41 lane with `mode` tied to DP4 is the like-for-like comparison: E4M3 x E4M3, four products, no FP32/DP2 logic. The all-modes lane is
  what the core has today; it also carries the FP32, FP16/BF16 and E5M2 modes, which the K-fused lane does not have.
- T3: WNS is the worst setup slack of the routed design at the constraint; the slack-implied period is the constraint minus WNS, an estimate of the
  lane's own limit, not an image clock. All front-end boundaries have at least 1.5 ns of slack at 5 ns.
- T4 comes from a hierarchy-preserving synthesis, so its totals differ from the routed ones (a different stage of the flow and no flattening).
- T5: the alternatives. `SIGN_MODE=0`: +310 LUT and +16 DSP. `TREE=1` (the adder tree in LUT carry chains instead of DSP48E2): +385 LUT and -15 DSP, which is the
  cost of keeping the tree out of the DSPs. `CUTS=0`: the same arithmetic with no optional register: +249 LUT, 9.7 ns slack-implied period, 859 flip-flops (the cause of the LUT
  increase was not examined).
- T6 (derived): `(a)` the new array replaces the shared-lane array; `(b)` it is added beside it; `(c)` only the front end is added and the existing back end
  is shared (the lane's routed LUT minus the back end's synthesis share, times 16; the mode mux in front of B0 is not counted). SLR0 packing is the recorded
  95.56 % plus the LUT change at 6.13 LUT per CLB over the 55,000 CLB that the two recorded packing figures (95.56 %, 96.74 %) and the 649-CLB headroom imply. It
  is arithmetic on recorded figures; a routed image will differ.
- T6b: equal E4M3 MAC rate (64 per cycle) needs 8 lanes of G=8 or 4 lanes of G=16: 15,832 or 10,152 LUT against 27,664 for 16 R41 DP4 lanes, but the engine's 16-column geometry would change
  and no design for that was made here.
