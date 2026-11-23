## 6. What failed, what was not done, what is not covered

Failures and surprises, all kept:
1. `SIGN_MODE=0` (sign written as an add/subtract so that the DSP's ALUMODE would carry it) does not map to one DSP48E2 in Vivado
   2023.2: it maps the multiplier and the add/subtract to two (T5: 63 DSP and +310 LUT against 47 DSP for the default). It is
   bit-exact (`f_g16sm0`) and left in the RTL as a parameter. An explicit DSP48E2 instantiation with ALUMODE would need a behavioural
   simulation model of the primitive; it was not built.
2. A first version of the `A` operand gating (a flushed term made zero on A) was replaced by gating B (commit `7cde3e27`): -54 LUT
   per G=16 lane, +256 flip-flops (Vivado stops absorbing DSP2's B register). The first RTL's results are in `ooc_v1/` and T5b;
   its simulations were a matrix that was killed when the RTL changed: 10 runs had finished, all with 0 mismatches, 3 were cut
   (`sim_v1_partial/`); because the matrix straddled the change it is kept but not counted, and the final matrix was run again from scratch.
3. xsim on the x86 host cannot run (`/usr/bin/gcc not found`, no root). The post-synthesis netlist was simulated with iverilog on
   GB10 with the Vivado unisim models of the 13 primitives used. The first G=4 netlist run reported 5 mismatches in the first
   five vectors with `valid_out` low: they were issued inside glbl's 100 ns GSR pulse, which holds every flop at INIT
   (`netlist/g4_run1_gsr_artifact.log`); the bench now waits 250 ns under `NETLIST`, and the rerun is T9.
4. The R41 shared lane measured in this flow (3,146 LUT all modes) is 354 LUT above the recorded 2,792 (a sum of per-stage
   figures); the difference was not reconciled. Both are in T1; the comparison that matters here is the DP4-tied lane (1,729 LUT),
   which is E4M3 x E4M3 only like the new lane.

5. Process: on GB10 up to five of my processes ran for about 20 s (three netlist simulations, the interrupted first netlist job's simulation, and two 40-vector debug
   runs) against the limit of four; the 1-minute load stayed below 11 (limit 16). On the x86 host never more than two Vivado jobs ran at once.

Not done:
- W = 8 and 24 (registered menu) and FP16/BF16 (section 3, items 5 and 6); `sub_c`; the single-DSP-per-product form and a LUT
  product form (no measurement); an explicit-ALUMODE DSP2; a back end narrower than R41's (see below).
- No integration: no matrix-engine change, no ISA mode, no TLM, no image, no board. The lane needs, beyond what is measured here:
  the B latch widened to 16 x G elements per lane and the A-buffer select widened to G elements (doc/r42_arch_headroom.md sec 1.5),
  the capability bit and the new opcode (registered in doc/r42_plan.md, 2026-11-23), a mux between this front end and the shared back end if
  both live in one lane, the oracle for the group size in the SystemC model, and the command overlap of R42-mtx (a K16 command at G=16 is one group).
- The OOC runs have an ideal clock, no SLR crossing and no neighbouring logic: they are screens. A routed image changes every
  number in T1-T3 (the R41 kernel routed 1.6x its post-synthesis delay estimate at 100 MHz, `flow/synth/vivado_ooc.sh`).
- Numbers for the accuracy gate are in `evidence/r42/kfuse/REPORT.md` (model, CPU); this report does not repeat them and adds none.

## 7. Where the cost is, and what to do next (ranked by expected LUT gain per core)

1. The back end is 53 % of the G=16 lane and 80 % of the G=4 lane (T4) and is the R41 FMA back end sized for a 48-bit product
   and subnormal/overflow results that this lane can never produce (|S x 2^L| < 2^(L+28+W), never a subnormal or an overflow
   unless c passes through). A back end written for a 29..32-bit sum and the binary32 c (near/far alignment, no subnormal
   path) is the largest saving available; it was not built or estimated here, and it must reproduce the same `RO`-free rounding
   (the bench above is the check).
2. 68 LUT per product (T4): about half is the one-hot A decode (a 6-input function per bit because the sign is in A). An explicit
   DSP48E2 with ALUMODE (item 1 of section 6) halves it.
3. The G=16 lane is DSP-heavy (47 per lane, 752 per core) because DSP1 and DSP2 are separate: a single-DSP product with operand
   shifts would halve the product DSPs and cost LUTs; the U280 has 9,024 DSP48E2 and the R41 image uses 818, so this is not binding.
