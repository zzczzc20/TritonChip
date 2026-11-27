# R42 issue: results (worker `issue`, branch R42-issue, base R42 129f213d)

Domains in every number: **rtl-probe** (core testbench, assembly microbenchmarks, one core), **rtl-1c** (single-core RTL simulation of ArchBench points,
`archbench.run --counters`, profile `core-vrf-sync-v5-x8-g2c4-dp`), **rtl-3c** (three-core board-top RTL stand-in at the run-3 image geometry, `tools/r42/img_mc.py`),
**OOC** (Vivado 2023.2 out-of-context synthesis / place and route on xcu280, x86 host), **TLM** (the dse worker's SystemC model; its files were only read),
**derived** (arithmetic on recorded numbers). No board number exists for anything here. No link, card load or board access was made.

## 0. Findings in one paragraph each

1. **The 2-cycle-per-vector-instruction mechanism of `vi1` does not exist in the run-3 core.** With `VRF_PREFETCH 1` + `VRF_BYPASS 1` (the image's parameters) independent vector
   instructions of 20 kinds (0-3 VRF sources, int/FP/convert/shuffle) issue at exactly 1.00 cycle each; 2.00 appears only with `VRF_PREFETCH 0`. What costs cycles is
   (a) a dependent consumer: producer in W, consumer reads the bank in R (request cycle + response cycle), (b) two sources in one VRF bank (+1 cycle per extra source),
   (c) refill after a taken branch. The TLM's `issue_vector_cycles 2` is a calibration knob, not a mechanism: its tiny-tier `vi1` effect is 1.165x, the RTL bound for removing every operand-fetch
   stall cycle is 1.067x (geometric mean over the same 27 points).
2. **`VRF_FWD` (built, default off) is the cheapest change that overlaps the operand fetch of a dependent instruction**: the writeback register is forwarded into R's operand capture.
   Measured (rtl-1c tiny, 27 points): 1.021x geometric mean (1.025x cycle-weighted), derived +1.3 % on the 145-point board geometric mean (central). Price: +0.9 k LUT per tile (OOC synthesis), timing at 5.6 ns see section 6.
3. **The larger issue-side loss is bank conflicts, and the compiler already has the fix, off by default (`TNCA_BANK_ALLOC=1`)**: 1.019x on the 27 tiny points (rtl-1c), derived +1.1 % central, no hardware.
4. **`hl` / zero-cost branch: not built.** Upper bound 1.9 % of cycles (6,376 redirects x 4 cycles over 1.34 M cycles of the 27 tiny points); it needs a front-end change in the path that `clk` is re-pipelining; no ISA change is needed for a predictor but a hardware loop is a new encoding.
5. **`te32` does not fit this core as modelled**: the line engine's word-walk tag width is bounded by the port tag (`TENSOR_TAG_W <= port tag - 3`), the port tag is 2+log2(LINE_SLOTS) = 8 bits, so `te32` (6-bit tags, 32 slots) needs 128 LSU line slots, i.e. a wider LSU completion pick (the image's worst timing family); `te16` (`TENSOR_TAG_W 5`) exists and needs nothing.
6. **`sets128`**: image parameters only (`SETS 128`, `SET_W 7`, `SHRINK`). No benchmark point issues a cache-maintenance operation, so the 2,048-cycle `M_SCAN` costs 0 cycles in every measured launch.

(Sections 1-9 carry the numbers; section 10 is the list of what was not done.)
