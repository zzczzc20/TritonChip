# R42 architecture headroom: matrix peak, issue width, and what the 113-point cycles are made of

> **Binding:** [`engineering_discipline.md`](engineering_discipline.md). Branch `R42-arch` (from `origin/R42` `5327118c`),
> 2026-11-23, analysis only: no RTL, compiler or ISA file was changed. Every number is quoted from a recorded file
> (`path:line`) or computed from recorded numbers by a script in `evidence/r42/arch/` (labelled **derived**), or it is
> labelled **inference** and the basis is stated. Domains: **board** (U280, image `791b8af2`, 100 MHz, device cycles),
> **routed** (Vivado routed checkpoint of an image), **screen** (out-of-context synthesis/place-and-route of one unit),
> **rtl** (single-core RTL simulation, a diagnostic screen), **structural** (read from RTL source). No new board,
> synthesis or place-and-route run was made.

## 0. Summary

The author asked three questions on 2026-11-23. Short answers first; the evidence follows.

### Q-A. "The matrix peak of about 10 GFLOP/s is too low; mainstream FPGA works reach at least 100 GFLOP/s. Am I wrong?"

**You are right about the FPGA and the sustained number, and the 9.6 is not the peak of the run-3 image.**

- **9.6 GFLOP/s is the 1-product-per-lane rate.** 3 cores x 16 lanes x 2 FLOP x 100 MHz = 9.6 GFLOP/s is the legacy
  `m.mma.async` and also the vector-FMA rate (`BALANCE.md:52`). The run-3 image's default matrix opcode is
  `m.mma.dp.async` (`codegen.py:1429`; the FP16 GEMM dumps contain `m.mma.dp.async.k16.f16`), so the structural matrix
  peak is **19.2 GFLOP/s for FP16/BF16/mixed FP8 (2 products per lane per cycle) and 38.4 GFLOP/s for E4M3 x E4M3
  (4 per lane per cycle)**; 18.1 and 36.1 after the one-cycle B-row load per 16 rows (`tnca_matrix.v:203`, `BALANCE.md:52-54`).
- **Sustained is far below any of them.** The best board point of the 113 (R42 software round 1) reaches **5.29 GFLOP/s**
  (`op.linear/medium/fp16-m128-k320-n256`), the large FP16 GEMMs 4.1-4.3, everything that is not GEMM-shaped 0.03-0.6
  (`evidence/r42/arch/achieved_gflops.txt`). That is 28 % of the 19.2 GFLOP/s FP16 peak at best.
- **What limits it is the design, not the FPGA.** The image uses 818 of 9,024 DSP48E2 (9 %, routed run-2 link,
  `utilization_slr.rpt.gz`); the arithmetic is built from LUTs (about 2,400 LUTs and 2 DSPs per FP32 FMA,
  `RESOURCES.md:136`; 2,792 LUTs and 8 DSPs per shared MAC lane, `fma_dslx/README.md:52-58`), SLR0 is at 95.56 % CLB,
  the clock is 100 MHz, and there are 48 lanes in total. 100 GFLOP/s of FP16 MACs is 500 MAC/cycle at 100 MHz, about
  500 DSPs (5.5 % of the device; **inference**: one 11x11 FP16 mantissa product per DSP per cycle).
- **A K-fused wide engine already exists in two sizes** (`m.mma.dp`, groups of 2 and 4 with one rounding per group,
  ISA sec 9C.4). 100 GFLOP/s of *peak* at 48 lanes and 100 MHz needs a group of **G >= 10.4 products per lane per cycle**
  (2.6x the E4M3 group, 5.2x the FP16 group). Derived cost from the recorded lane screens: **E4M3, exact sum:
  about +146 to +314 LUTs per extra product, +16k to +35k LUTs per core for G = 4 -> 11**; **FP16, exact sum: +1,650 LUTs
  per extra product (+52.8k LUTs per core for G = 2 -> 4), not feasible.** Recorded SLR headroom to the highest packing
  that ever routed (96.74 %): SLR0 about 4k LUTs, SLR1 about 22k, SLR2 about 28k (derived). So an FP8-only G = 8 engine
  (peak 76.8 GFLOP/s) fits SLR1 and SLR2 and not SLR0; G >= 11 (105.6 GFLOP/s) fits only at the low per-product cost and
  needs 12k to 31k LUTs removed from SLR0 or an asymmetric image.
- **Clock:** the run-3 kernel clock closed at +0.111 ns at 10 ns; the ten worst paths are the hybrid LSU (24-25 logic
  levels) and a reset/enable net into `u_vfpu` (one LUT, 9.2-9.5 ns of routing). 75-98 % of their delay is routing. The matrix lane's own
  stages are 6.2/6.5/4.9/4.6/5.2 ns in a unit screen, about 1.54x that in the chip, so they already use the period.
  Nothing recorded supports more than about 100-110 MHz without re-pipelining the core and lowering the packing.
- **Verdict.** 100 GFLOP/s *sustained* on the suite is not realistic: the matrix engine is not what limits the suite
  (a 4x engine gave 0.99-1.04x on FP8 linears in the TLM, 1.09x geometric mean on 59 GEMM cases, `pmac/README.md:148`,
  `mma_dp/README.md:79`). 100 GFLOP/s *peak* is realistic only for E4M3 x E4M3, only with a new group-of-8..16 lane,
  and only if SLR0 loses about 12-31k LUTs or the wide cores live in SLR1/SLR2.
  Confidence: peak arithmetic **high**; area **medium-low** (no screen of a G > 4 lane exists: linear extrapolation);
  clock **medium**.

### Q-B. "Can the core fetch several instructions at once with a scoreboard, or use a very long instruction word?"

**Not first. The evidence supports removing bookkeeping (compiler, then a hardware loop) before widening issue, and
widening issue only once the matrix/transfer path is faster.**

- **Facts verified.** Single issue, in order inside a role context, 4 contexts, F/D/R/X with a scoreboard in R, a
  one-entry fetch buffer, one 64-bit instruction fetch per cycle from a BRAM code store (no instruction cache; no
  extension words, every instruction is 8 bytes), the decoder carries FU classes `SALU, VALU, SFU, LSU, MATRIX, TXFER, BAR`
  (`architecture_overview.md:103-110`, `tnca_core.v:431-470`, `tnca_defs.vh:524-531`, `tnca_asm.py:247`).
- **Where the cycles go** (single-core RTL counters; no board counters exist, `RESOURCES.md:156`): on the GEMM families
  (93.6 % of the suite's device cycles are in families whose points are mostly compute-side) the issue slot is busy only
  14-34 % of the cycles for linear, gemv, mlp, bmm (B39 small tier: 14, 21, 22, 21 %; fresh R42-head tiny: 24, 24, -, 34 %; matmul 42-53 %); the rest is
  transfer-engine and matrix-engine occupancy, barrier sleep and hazards. On the memory-side families (64 of the 113 points) the core issues
  40-95 % of the cycles, the remainder being load/store and FPU latency stalls.
- **Amdahl bounds of a two-slot core** (slot A = scalar/control/predicate, slot B = everything else; no dependence or structural check, so
  upper bounds): per-point geometric mean **1.11x** for in-order pairing within a context (tiny tier, fresh run;
  1.21x if every slot-A instruction could be paired anywhere); removing every slot-A instruction 1.22x. Cycle-weighted over the tiny points the bound is
  dominated by `embedding` (95 % issue-bound, 1.53x). The GEMM families are bounded at 1.05-1.10x.
- **VLIW** needs a new container/encoding, a new assembler/ISA schema/decoder/model/compiler back end and keeps the
  scoreboard anyway (async engines have variable latency): no advantage over a binary-compatible 2-wide front end, and
  it breaks every existing binary. **Not supported.**
- **Cost of the 2-wide front end** (derived from recorded module sizes, **inference** for the sum): +6k to +12k LUT-equivalents per
  core (second decoder, scoreboard check of the pair, 2 more scalar-RF read ports, writeback booking), and a longer R stage
  on a +0.111 ns margin. It does not fit SLR0 (headroom about 4k LUTs); it fits SLR1/SLR2.
- **Cheaper first**: constants (`.li` is 4 instructions), pointer bumps and loop control are 15-50 % of the
  reduction/attention loop bodies and 53 % of the GEMM K-loop body (static; that includes 8 `s.mov` per two steps that the ISA's
  "five consecutive coordinate registers" rule forces); a zero-overhead hardware loop costs about
  0.3-0.5k LUT-equivalents (**inference**), is binary-compatible (new CTRL func) and removes the not-taken-prediction bubble;
  descriptor-driven weight fragments remove the 56 % bookkeeping of the FP8 Hub kernel (`BALANCE.md:228`).
- **What would make issue pay:** a matrix engine with G = 8..16. A K16 command would take 48-31 cycles instead of 150-168 while
  a 16 x 16 tile issues 19 instructions per K step plus a 220-instruction epilogue (static, R42 head): the issue slot is busy 22 % of the tile today,
  68 % at G = 8 and 106 % at G = 16 for K = 256 (derived, sec. 2.6).
  Confidence: facts **high**; counters **medium** (single-core simulator, tiny/small points, no board counters);
  hardware cost **low** (no synthesis).

### Q-C. What the 113 points' device cycles are made of after R42 round 1

- Board total 48.18 G cycles (run-3 base) -> 36.92 G (ab-int1); geometric mean 1.613x (`ab-int1/comparison.json`).
- **One point is two thirds of the total cycles** (`llama_prefill/large`, derived 66.7 %); `llama_prefill` 71.4 % of the
  cycles, `matmul` 11.8 %, `linear` 3.6 %; the 9 families whose points are mostly compute-side carry **93.6 %** of the cycles in 43 of the 113 points.
- The geometric mean weights points equally, so its headroom sits in the memory-side/instruction-bound families: **64 of
  102 points** with a recorded roofline binding are memory-side and carry 6 % of the cycles. Largest remaining distances
  to the ArchBench bound (share of sum log(1/eta)): `conv2d` 8.8 % (eta 0.040), `resnet_block` 8.5 % (0.044), `softmax` 8.2 %
  (0.134), `attention_prefill` 7.9 % (0.055), `attention_decode` 6.6 % (0.089), `rope` 6.1 % (0.108), `matmul` 5.8 %.
- Geometric-mean eta 0.1417 -> 0.2365 (derived, 102 points that have a FLOP bound).
- Compute-bound structure: `llama_prefill`, `matmul`, `linear`, `mlp`, `gpt2_block`, `bmm`, `resnet_block`, `conv2d`,
  `attention_prefill`. `conv2d`/`resnet_block` (FP32) run on the 16-lane vector FPU (no `m.mma` in their code: `vf.fma` + `v.permute` + `a.vread/a.vwrite`), at 0.57 GFLOP/s best.
  Confidence: **high** (board records, arithmetic on them).

---

## 1. Q-A: the matrix peak, what limits it, and a wide dot-product engine

### 1.1 The peak, verified

Structural facts (RTL):

- 16 MAC lanes per core, one output column each; k outer, m inner; one k step is 16 row cycles plus one cycle to load the B
  row (`tnca_matrix.v:47`, `:530`). With `MAC_DP` each lane is `tnca_mac_lane`: mode 0 FP32 (1 product), mode 1 DP2 (two
  products per cycle for FP16/BF16/E5M2/E4M3 pairs), mode 2 DP4 (four E4M3 x E4M3 products); `kstep` is 1, 2 or 4 (`tnca_matrix.v:203`;
  `tnca_mac_lane.v:2-8`). The result is `RNE_f32(C + RO_48(group sum))`, one rounding per group (ISA sec 9C.4,
  `TNCA-ISA-v0.1.md:897-946`); `m.mma.async` keeps one rounding per k (sec 9C.1, `:826`).
- The R41 run-3 profile `core-vrf-sync-v5-x8-g2c4-dp` has the capability `matrix_dp`, and the compiler then emits
  `m.mma.dp.async` for every matrix command (`codegen.py:1429-1440`). A small-tier FP16 GEMM dump of the R42 head contains
  `m.mma.dp.async.k16.f16` and no `m.mma.async` (`evidence/r42/arch/loops_small.txt`, kernel dumps on thor). FP32
  matmul and conv dumps contain **no `m.mma` at all**: the dump of `op.matmul/small/fp32` is 256 `vf.fma`, 256 `v.permute`, 32 `a.vread`, 16 `a.vwrite` (the vector FPU does the arithmetic).

| Mode (3 cores, 16 lanes, 100 MHz) | G | raw GFLOP/s | with 16/17 B-load bubble | MAC/cycle/core | source |
|---|---:|---:|---:|---:|---|
| legacy `m.mma.async` and the FP32 vector FMA | 1 | 9.6 | 9.0 | 16 | `BALANCE.md:52` |
| `m.mma.dp` FP16, BF16, mixed FP8 | 2 | **19.2** | 18.1 | 32 | `BALANCE.md:53` |
| `m.mma.dp` E4M3 x E4M3 | 4 | **38.4** | 36.1 | 64 | `BALANCE.md:54` |

(`evidence/r42/arch/qa_budget.txt`, derived: peak = 3 x 16 x G x 2 FLOP x 100 MHz.)

A whole command is lower still: one `m.mma.dp.async.k16.f16` occupies the engine 168 cycles for 4,096 MACs, 24.4 MAC/cycle
against 32 (`evidence/r42/gemm/ANALYSIS.md:37`, rtl); the TLM puts a K32 E4M3 command at 70 % of the DP4 peak (`BALANCE.md:69`).

### 1.2 What the board reaches

`evidence/r42/arch/achieved_gflops.py` divides the ArchBench demand FLOP of each point by its board device time
(run-3 image, R42 software round 1; `ab-int1/comparison.json`):

| point | GFLOP/s (new) | run-3 base | of 19.2 (DP2 peak) |
|---|---:|---:|---:|
| `op.linear/medium/fp16-m128-k320-n256` | 5.29 | 5.28 | 28 % |
| `op.linear/hbm/fp16-m512-k4096-n11008` | 5.06 | 5.02 | 26 % |
| `op.matmul/hbm/fp16-m4096-k4096-n4096` | 4.27 | 4.26 | 22 % |
| `model.llama_prefill/large/fp16-b1-s512` | 4.10 | 3.11 | 21 % |
| `op.bmm/large/fp16-b32-m512-k64-n512` | 3.58 | 0.43 | 19 % |
| best `conv2d` / `resnet_block` (FP32, vector FPU) | 0.58 / 0.57 | 0.52 / 0.48 | 6 % of 9.6 |
| best `softmax`, `layernorm`, `rope`, `attention_decode` | 0.15, 0.30, 0.06, 0.09 | | memory-side points |

The ArchBench `eta` is relative to a single-core spec bound (64 FLOP/cycle = 6.4 GFLOP/s, 8 B/cycle;
`roofline/roofline.json`, `README.md`), so on the three-core image a compute-side eta of 0.39 is 2.5 GFLOP/s.

Why the GEMM points are at 20-28 % of the peak (single-core RTL, same profile, `evidence/r42/gemm/ANALYSIS.md:21-33`):
MMA busy 70 % of the cycles in `op.linear/small`, transfer engine busy 72 %, both idle 19 %; one K step is 20 instructions and a command is awaited before the
next is issued, so the K loop runs one MMA per 195 cycles (168 busy); epilogue 14 % (small) to 34 % (tiny). The matrix engine is
busy about 48 % of the single-core 2048^3 FP16 matmul (`pmac/README.md:155`). R41's own TLM found that a 4x engine does not move the end-to-end number:
FP8 W8A8 linears 0.99-1.04x, BF16 at G=2 1.00-1.04x (`pmac/README.md:148-150`), the compiled DP lane 1.09x geometric mean on 59 GEMM cases,
at most 1.30x (`mma_dp/README.md:79`; `BALANCE.md:18`). **Arithmetic width is not what limits the suite today.**

### 1.3 FPGA versus design

| quantity | the U280 | this design (run-3 RTL = run-2 link RTL + one latency-width line) | who limits |
|---|---|---|---|
| DSP48E2 | 9,024 (SLR0 2,880, SLR1 3,073, SLR2 3,071, derived from used/percent) | 818 used incl. platform = 9.1 % (SLR 269/276/273); kernel 814 (520 in run 1 + 294 from the shared lane, `RESOURCES.md:458`) | design: FP32 datapaths are LUT-built, the DSPs only form 24x24 products (`RESOURCES.md:136`) |
| LUT | 1,303,680 (`STATUS.md`) | CLB packing SLR0 **95.56 %**, SLR1 90.66 %, SLR2 88.06 % (`link_1f0a5aff/utilization_slr.rpt.gz`, routed run-2 link) | both: one tile is 251k LUTs = 57 % of an SLR |
| URAM | 960 | 0 used | design (NUCA arrays and accumulators are LUTRAM) |
| SLRs | 3, 23,040 SLLs per boundary | one core per SLR; the three outlets and the HBM subsystem are in SLR0 | design placement |
| clock | | 100 MHz, kernel WNS +0.111 ns | design (LSU and control paths, routing 75-98 %) |
| lanes | | 48 matrix lanes, 48 vector FMAs | design |
| HBM | 32 pseudo-channels, 460 GB/s theoretical (`roofline.json`) | three 512-bit outlets, one 64 B/cycle path per core (about 19 GB/s analytic) | design |

Where the tile's LUTs are (kernel synthesis of the run-1 source, `RESOURCES.md:98-111`): `u_vfpu` 62.8k (25.0 %), `u_mtx` 51.1k (20.3 %,
the 16 MACs 42k), NUCA slice 30.6k, LSU pair 25.4k, `u_valu` 23.2k (64 DSPs), VRF 21.3k, transfer engine 14.5k, barrier 7.5k, `u_core` 6.8k.
The FMA rebuild saved 26,976 LUTs per core and the shared lane added 24,100 (`RESOURCES.md:457-458`).

### 1.4 What a wide dot-product engine costs (recorded anchors, then derived)

Recorded per-lane costs (unit screens and OOC place-and-route, 10 ns, SLR0):

| lane | LUT | DSP | products per cycle | LUT per MAC/cycle | source |
|---|---:|---:|---:|---:|---|
| V0 FP32 FMA lane (R41 base) | 2,492 | 2 | 1 | 2,492 | `pmac/README.md:127` |
| V3 E4M3 exact, 38-bit field | 2,929 | 2 | 4 | 732 | `pmac/README.md:131`, `lane/README.md:134` |
| V1 all-FP8 pairs, 66-bit field | 3,433 | 2 | 4 | 858 | `lane/README.md:123` |
| V2D FP16 DP2 + FP8, 64-bit field | 4,142 | 2 | 2 | 2,071 | `lane/README.md:127` |
| **R41 shared lane** (FP32, DP2, DP4), stages 689+639+448+526+490 | **2,792** | **8** | 1/2/4 | 698 (E4M3) | `fma_dslx/README.md:52-58`, `RESOURCES.md:458` |

Of the 2,792 LUTs, the back end shared by every mode (align with C, normalise, round) is 1,464 (52 %): it is paid once per group,
which is why K-fusion is cheap per MAC. Marginal LUTs per extra product, from the V-lanes (derived): **E4M3 exact 146**, all-FP8 66-bit 314,
**FP16 64-bit exact 1,650**. The FP16 field is wide because FP16 products span 2^-48..2^31; a 64-bit field is what makes the group sum exact before RO_48
(`lane/README.md` DP2 field-width paragraph). Where the lane sits in the SLRs (recorded projection, pre-G2 baseline,
`lane/README.md:198-201`): V3 -> SLR0 97.4 %, V1 99.8 %, V2D 103.1 %.

**Budget for raising G per lane** (16 lanes per core; derived by `evidence/r42/arch/qa_budget.py`, linear extrapolation of the marginal
costs, **no G > 4 lane has ever been screened**):

| change | LUTs per core | peak at 100 MHz | fits SLR1 (22k) / SLR2 (28k) / SLR0 (4k)? |
|---|---:|---:|---|
| E4M3 exact, G 4 -> 8 | +9.3k | 76.8 GFLOP/s | yes / yes / no (needs 5.3k removed) |
| E4M3 exact, G 4 -> 11 (100 GFLOP/s peak) | +16.4k | 105.6 | yes / yes / no (needs 12.4k removed) |
| E4M3 exact, G 4 -> 16 | +28.0k | 153.6 | no / borderline (28.1k) / no |
| all-FP8 66-bit, G 4 -> 8 | +20.1k | 76.8 | yes / yes / no |
| FP16 exact, G 2 -> 4 | +52.8k | 38.4 | no / no / no |
| FP16 exact, G 2 -> 8 | +158k | 76.8 | no |

SLR headroom (derived): the packing of the routed run-2 link (SLR0 95.56 %, SLR1 90.66 %, SLR2 88.06 %) up to 96.74 %, the highest that routed (run 1,
`RESOURCES.md:66`), at each SLR's own LUTs per CLB (6.13, 6.79, 6.00): 649 / 3,283 / 4,687 CLBs = **3,975 / 22,292 / 28,133 LUTs**.
(Run 3's own SLR report is not in the recorded tree; run 3 differs from run 2 by one line, `tnca_matrix.v` MAC latency width.)

**Numerics.** Any group larger than the existing 2/4, or any shared-exponent aligned tree that truncates, is a new result-bit contract: it must be a **new
opcode/mode** (ISA sec 9C.1: "An implementation that wants a faster reduction order must expose it as a different opcode"; sec 9C.4 is the precedent: group
sizes 4 and 2 with RO_48), with its own oracle (`tools/model/fdp.py`) and capability bit; the TLM can only model G <= 4 today
(`tnca_model.hpp:101-103`). A truncating aligned adder tree (NVIDIA-style) would be cheaper per product than the exact field, but **no screen of one exists**
(its cost is the missing number in this section); the FP16 row of the table is the exact-sum contract.

### 1.5 What it needs from the register file and the other engines

- The engine does **not** read a 64-byte B row every cycle: `b_row` is latched once per k-group and held for 16 row cycles, so B costs 64 B per 17 cycles; A comes from
  the A buffer, 4 B per cycle at DP4. The matrix port is 9-11 % busy (`BALANCE.md:76`, `tnca_matrix.v:243`).
- At group G the B latch holds 16 x G elements: **E4M3 G = 8: 128 B (2 registers), 7.5 B/cycle; G = 16: 256 B (4 registers), 15.1 B/cycle; FP16 G = 8: 256 B, 15.1 B/cycle** (derived,
  `qa_budget.txt`), against a 64 B/cycle port: below 25 %. The problem is not bandwidth but the B reload latency per group (4 serial register reads against 16 row cycles
  unless double-buffered: `mtx_dp_b_prefetch` is a TLM what-if, R42-mtx is working on it) and the A-buffer select width (G x element bytes per cycle).
- The accumulator file needs the same one 64 B row read and write per cycle at any G.
- What changes is the command: a K16 command at G = 16 sweeps in about 16 cycles plus a 6-cycle drain and the A-buffer fill (8 reads), so command overhead is half of it
  (`tnca_matrix.v:140-150, 530-560`); without command overlap (R42-mtx) the wide engine is idle half the time.
- **Operands arrive through the transfer engine.** 100 GFLOP/s sustained is 167 MAC/cycle/core; with 16 x 16 tiles that is 0.125 element per MAC (0.25 B/MAC FP16), i.e.
  **42 B/cycle/core**, more than the 29 B/cycle read ceiling (`BALANCE.md:39`, the brief's figure) and 5x the 8 B/cycle word walk; a 64 x 64 blocking brings it to about 10 B/cycle/core
  (derived, `BALANCE.md` AI table: 2.1 B/cycle at the DP4 peak with 64 x 64). So the wide engine only pays together with register blocking, the line-wide transfer path and a lower
  issue cost per command.

### 1.6 Clock

Run-3 routed, pre-package (`evidence/r41/hbm_bw/g4/run3/route_prepackage/timing_summary.rpt.gz`, summary at `README.md:5-12`): kernel-clock group setup WNS **+0.111 ns** (hold +0.001) at
10.000 ns; the design-level WNS of -0.085 ns is in the HBM platform clock (`hbm_aclk`), not the kernel. Earlier links: +0.042 (run 1), +0.038 and +0.016 after a reroute (run 2,
`link_1f0a5aff/README.md:9`). The ten worst kernel setup paths (`evidence/r42/arch/run3_kernel_clock_paths.txt`):

- 3 x hybrid-LSU `fast/rr_reg` -> `out_vector/out_scalar` (24-25 logic levels, data path 9.8 ns, 75-78 % routing);
- 1 x hybrid-LSU `fast/valid_reg` -> `control/rsp_status` (14 levels, 9.28 ns, 87 % routing; the clock itself crosses SLR2 -> SLR0, skew -0.351 ns, SLR compensation 0.227 ns);
- 6 x `reset_sync_reg` -> `u_vfpu/acc_reg[..]/CE` (one LUT, **9.2-9.46 ns, 97.7 % routing**: a high-fanout enable spread over the SLR).

So in the image the clock is set by placement and routing in a 95-96 % packed SLR, and by the hybrid LSU, not by the matrix. The matrix lane's five stages are 6.17, 6.45, 4.9, 4.6, 5.2 ns in a unit screen
(`fma_dslx/README.md:52-58`); the pmac study scales OOC to in-chip by 1.54x (`lane/README.md:116,140`), giving 9.5 / 9.9 / 7.5 / 7.1 / 8.0 ns (**derived**): the two front stages already fill 10 ns.
The lane can take two or three more register stages without stalling the sweep (a row returns every 16 cycles, `pmac/README.md:77`), so the lane is not what fixes 100 MHz; the LSU
(24 logic levels), the reset/enable fan-out and the packing are. Run 1 lost 0.9-1.0 ns from tile screen to image at 96.7 % packing (`g4/run2/README.md:37`). **Inference:** about 100-110 MHz is what this
netlist at this packing supports; 125-150 MHz needs a pipelined LSU, a registered/replicated enable tree and a lower SLR packing, none of which has a recorded screen; 200+ MHz is a new physical design
(`r42_plan.md:80`).

### 1.7 Resulting range, and the answer

| configuration | G | clock | peak GFLOP/s | status |
|---|---:|---:|---:|---|
| run-3 image, FP16/BF16 | 2 | 100 | 19.2 | recorded structure; sustained 5.3 best |
| run-3 image, E4M3 x E4M3 | 4 | 100 | 38.4 | idem |
| E4M3 exact, G = 8 in SLR1/SLR2 cores | 8 | 100-110 | 76.8-84.5 | fits headroom on paper (derived) |
| E4M3 exact, G = 11 | 11 | 100 | 105.6 | needs about 12-31k LUTs out of SLR0, or asymmetric cores |
| E4M3 exact, G = 16 | 16 | 100 | 153.6 | +28k LUTs per core: only with real removals |
| FP16 exact | 4 | 100 | 38.4 | +52.8k LUTs per core: no |
| any, 150 MHz | | 150 | x1.5 | no evidence |

100 GFLOP/s peak on the U280 is physically ordinary (5.5 % of the DSPs at 100 MHz); in this design it means a new K-fused lane with G >= 10.4 for FP8 (a new opcode with its own oracle,
a lane screen first), either 12-31k LUTs removed from SLR0 or the wide engine only in the SLR1/SLR2 cores, command overlap and B prefetch (R42-mtx), register blocking, the line-wide
transfer path, and a lower issue cost per command (Q-B). Without those, the sustained number stays near the 4-5 GFLOP/s of today (see 1.2).
What can be removed from SLR0 is limited by the author's earlier decisions (`RESOURCES.md:347-352`: MAC sharing with the vector FPU and a smaller FMA were **rejected**; the
contexts cut was withdrawn); capability-neutral options already taken: `AXI_LINE_LUTRAM`, `NUCA_SHARED_DATA_READ 2`, `LSU_INFLIGHT 1`, 8 barriers, 38-bit epochs, FMA rebuild. Remaining pools in
SLR0 (recorded sizes): DCO/TMU replica 7.6k LUT cells (replicas were added to close cross-SLR TMU paths; `DCO_LOCAL_REPLICAS 0` was screened and rejected, WNS +0.198 ns, `RESOURCES.md:258,481`), NUCA slice 27.8k, `u_valu` 23.2k,
`u_vfpu` about 49k after the FMA rebuild; each is a capability or a timing trade that is the author's to decide.

---

## 2. Q-B: issue

### 2.1 The front end, verified

| fact | where |
|---|---|
| single issue: `issue.width = 1` in the target; "the FU class is kept in the decoder for a future dual-issue option" | `compiler/targets/tnca-sim-vrf-v5-x8-g2c4-dp.json:103-106`, `architecture_overview.md:104` |
| stages F (round-robin among role contexts), D, R (scoreboard), X; in order inside a context; predicted not-taken, a redirect squashes only that context's younger entries; a context asleep in `mb.wait` stops fetching | `architecture_overview.md:103-110`, `tnca_core.v:395-431` |
| **one** instruction fetched per cycle (`imem_addr = pc[f_sel]`, 64-bit `imem_data`), synchronous code store, a **one-entry** fetch buffer | `tnca_core.v:431-460` |
| code memory is a BRAM per core, 16 K x 64 bit (96 RAMB36 for three cores), loaded by the host; there is no instruction cache and no miss path | `RESOURCES.md:58` |
| decoder classes `FU_NONE, SALU (scalar ALU + control + system), VALU (vector int/fp/cvt/pred/shuffle/reduce), SFU, LSU, MATRIX, TXFER, BAR` | `tnca_defs.vh:524-531`, `tnca_decoder.v` |
| instruction = 64 bits, fields `major func d a b c pred X PN mod`; **no instruction uses the extension word**: every instruction is 8 bytes, `.li sN,#imm64` is 4 instructions (`movz` + 3 `movk`) | `TNCA-ISA-v0.1.md:51-66`, `tnca_asm.py:22,247`, `isa/tnca-isa-v0.1.json` `word` |
| scalar RF: 4 contexts x 64 x 64 bit, **7 read ports and 1 write port** (3 operands + 4 more for the 5 transfer coordinates), 3,756 LUTs + 2,072 LUTRAM per core (one LUTRAM copy per read port) | `tnca_scalar_rf.v:4`, `RESOURCES.md:150` |
| writeback slots are booked in an 8-deep reservation shift register; `u_core` own logic (pipeline, scoreboard, counters) is 6,818 LUTs and 5,933 FFs in synthesis, 23,268 LUTs after the tool moves logic across modules | `tnca_core.v:1988-1990`, `RESOURCES.md:94,108` |
| the counters partition every running cycle: `perf_issued + perf_stall_total + perf_x_frontend == perf_cycles`; the exclusive stall causes are `hazard, lsu, fpu, mtx, txf, fence, arrive, rsv, wb, vrf, vrfwb` | `architecture_overview.md:229-244`, `tnca_core.v:2946-2962` |

So the answer to "fetch several instructions at once with a scoreboard": the scoreboard, the FU classes and four independent contexts exist; the fetch (one 64-bit word), the decoder, the
scalar RF write port, the writeback booking and the issue slot do not. The encoding is already a fat 64-bit word, which is the opposite of what a VLIW container needs to be.

### 2.2 Where the cycles go (counters)

Two sources, both **single-core RTL simulation** (no board counters exist):

(a) **Recorded B39 baseline** (`spec/r40/b39/{tiny,small,medium}.jsonl`, profile `core-vrf-sync-v5`, pre-R41 software; `evidence/r42/arch/qb_b39.py`). Small tier, shares of the family's cycles:

| family | n | cycles | issued | hazard | lsu | fpu | mtx | txf | fence | barrier wait / front end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| add | 1 | 31356 | 62 % | 0 % | 22 % | 0 % | 0 % | 0 % | 0 % | 2 % |
| argmax | 1 | 63304 | 76 % | 0 % | 9 % | 0 % | 0 % | 0 % | 0 % | 2 % |
| attention_decode | 1 | 505304 | 73 % | 4 % | 5 % | 3 % | 0 % | 0 % | 0 % | 4 % |
| attention_prefill | 1 | 2587873 | 60 % | 7 % | 19 % | 0 % | 1 % | 0 % | 1 % | 3 % |
| bmm | 1 | 1268824 | 21 % | 17 % | 51 % | 0 % | 0 % | 1 % | 2 % | 5 % |
| conv2d | 1 | 668929 | 60 % | 7 % | 14 % | 0 % | 2 % | 0 % | 1 % | 2 % |
| cumsum | 1 | 145468 | 85 % | 1 % | 0 % | 0 % | 0 % | 0 % | 0 % | 1 % |
| embedding | 1 | 373714 | 95 % | 0 % | 2 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| gelu_tanh | 1 | 140408 | 62 % | 1 % | 1 % | 13 % | 0 % | 0 % | 0 % | 1 % |
| gemv | 1 | 256466 | 21 % | 4 % | 22 % | 0 % | 3 % | 29 % | 0 % | 20 % |
| gpt2_block | 1 | 4453584 | 47 % | 8 % | 12 % | 1 % | 9 % | 1 % | 1 % | 15 % |
| layernorm | 1 | 183823 | 69 % | 1 % | 4 % | 9 % | 0 % | 0 % | 0 % | 2 % |
| linear | 1 | 370979 | 14 % | 2 % | 1 % | 0 % | 30 % | 2 % | 0 % | 50 % |
| llama_decode | 1 | 6672952 | 64 % | 2 % | 10 % | 2 % | 1 % | 7 % | 0 % | 7 % |
| llama_prefill | 1 | 19772226 | 51 % | 8 % | 13 % | 0 % | 7 % | 1 % | 1 % | 12 % |
| matmul | 2 | 529715 | 53 % | 7 % | 1 % | 0 % | 10 % | 1 % | 0 % | 12 % |
| mlp | 1 | 1112351 | 22 % | 11 % | 1 % | 1 % | 22 % | 2 % | 0 % | 35 % |
| resnet_block | 1 | 1425579 | 59 % | 9 % | 14 % | 0 % | 2 % | 0 % | 1 % | 2 % |
| rmsnorm | 1 | 138271 | 72 % | 1 % | 1 % | 2 % | 0 % | 0 % | 0 % | 1 % |
| rope | 1 | 60410 | 64 % | 0 % | 21 % | 0 % | 0 % | 0 % | 0 % | 2 % |
| silu | 1 | 94348 | 55 % | 3 % | 0 % | 22 % | 0 % | 0 % | 0 % | 1 % |
| softmax | 1 | 451818 | 58 % | 11 % | 2 % | 8 % | 0 % | 0 % | 0 % | 6 % |
| swiglu | 1 | 91853 | 61 % | 4 % | 2 % | 13 % | 0 % | 0 % | 0 % | 2 % |
| transpose | 1 | 73104 | 57 % | 2 % | 34 % | 0 % | 0 % | 0 % | 0 % | 1 % |


(b) **Fresh R42-head runs** (profile `core-vrf-sync-v5-x8-g2c4-dp`, the profile of the board image's cores; `qb_family.py`, the issue stream from `arch_probe.v`). "Barrier wait / front end" is `perf_x_frontend`: nothing stalled and nothing
in R, which includes every context asleep in `mb.wait`, that is, waiting for an engine. "Slot A" = scalar ALU, predicate, control, system instructions (dynamic share of issued instructions).
"2-wide in-order bound" = cycles / (cycles - pairs), where pairs are the greedy adjacent (slot A, slot B) pairs of one role context's issue stream (no dependence check: an upper bound).
"All slot A removed" = cycles / (cycles - number of slot-A instructions).

| tier | family | n | cycles | issued | hazard | lsu | fpu | mtx | txf | fence | barrier wait / front end | slot A of issued | 2-wide in-order bound | all slot A removed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | add | 1 | 28959 | 29 % | 0 % | 62 % | 0 % | 0 % | 0 % | 0 % | 0 % | 15 % | 1.01x | 1.05x |
| small | argmax | 1 | 37205 | 68 % | 0 % | 15 % | 0 % | 0 % | 0 % | 0 % | 3 % | 49 % | 1.22x | 1.50x |
| tiny | add | 2 | 3201 | 41 % | 0 % | 50 % | 0 % | 0 % | 0 % | 0 % | 1 % | 24 % | 1.06x | 1.11x |
| tiny | argmax | 1 | 2035 | 68 % | 8 % | 2 % | 0 % | 0 % | 0 % | 0 % | 6 % | 46 % | 1.18x | 1.45x |
| tiny | attention_decode | 1 | 9521 | 59 % | 8 % | 7 % | 10 % | 0 % | 0 % | 0 % | 4 % | 24 % | 1.11x | 1.16x |
| tiny | attention_prefill | 1 | 24890 | 49 % | 16 % | 10 % | 3 % | 0 % | 1 % | 0 % | 6 % | 25 % | 1.08x | 1.14x |
| tiny | bmm | 1 | 1850 | 34 % | 8 % | 5 % | 0 % | 0 % | 9 % | 2 % | 37 % | 54 % | 1.11x | 1.23x |
| tiny | cumsum | 1 | 2221 | 77 % | 0 % | 8 % | 0 % | 0 % | 0 % | 0 % | 1 % | 10 % | 1.03x | 1.09x |
| tiny | embedding | 1 | 46384 | 95 % | 0 % | 2 % | 0 % | 0 % | 0 % | 0 % | 0 % | 47 % | 1.53x | 1.80x |
| tiny | gelu_tanh | 1 | 1971 | 53 % | 1 % | 6 % | 23 % | 0 % | 0 % | 0 % | 1 % | 33 % | 1.11x | 1.21x |
| tiny | gemv | 1 | 4843 | 24 % | 1 % | 2 % | 0 % | 17 % | 44 % | 1 % | 7 % | 53 % | 1.07x | 1.15x |
| tiny | layernorm | 1 | 2046 | 47 % | 3 % | 20 % | 14 % | 0 % | 0 % | 0 % | 2 % | 47 % | 1.09x | 1.28x |
| tiny | linear | 1 | 6318 | 24 % | 12 % | 5 % | 0 % | 13 % | 34 % | 1 % | 6 % | 46 % | 1.06x | 1.13x |
| tiny | llama_decode | 1 | 204816 | 42 % | 6 % | 6 % | 3 % | 9 % | 22 % | 0 % | 4 % | 37 % | 1.11x | 1.18x |
| tiny | matmul | 2 | 4153 | 42 % | 6 % | 4 % | 0 % | 0 % | 8 % | 1 % | 22 % | 20 % | 1.05x | 1.09x |
| tiny | rmsnorm | 1 | 1409 | 47 % | 4 % | 23 % | 14 % | 0 % | 0 % | 0 % | 2 % | 52 % | 1.09x | 1.32x |
| tiny | rope | 1 | 4274 | 56 % | 0 % | 31 % | 0 % | 0 % | 0 % | 0 % | 1 % | 12 % | 1.04x | 1.07x |
| tiny | silu | 1 | 1460 | 46 % | 3 % | 7 % | 28 % | 0 % | 0 % | 0 % | 1 % | 32 % | 1.10x | 1.17x |
| tiny | softmax | 2 | 3810 | 41 % | 20 % | 8 % | 11 % | 0 % | 0 % | 0 % | 8 % | 46 % | 1.09x | 1.23x |
| tiny | swiglu | 1 | 1659 | 46 % | 2 % | 10 % | 24 % | 0 % | 0 % | 0 % | 1 % | 30 % | 1.10x | 1.16x |
| tiny | transpose | 1 | 2298 | 57 % | 1 % | 34 % | 0 % | 0 % | 0 % | 0 % | 2 % | 44 % | 1.24x | 1.33x |
| small | **geometric mean of the per-point bound** | 2 | | | | | | | | | | | **1.108x** | **1.252x** |
| tiny | **geometric mean of the per-point bound** | 22 | | | | | | | | | | | **1.111x** | **1.213x** |


Reading (a) and (b) together:

- **GEMM-shaped families are not issue-bound.** `linear`, `gemv`, `mlp`, `bmm` issue in 14-36 % of the cycles (B39 small: linear 14 %, mlp 22 %, gemv 21 %, bmm 21 %; fresh tiny: linear 24 %, gemv 24 %, bmm 34 %), `matmul` in 42-53 %;
  the rest is transfer-engine occupancy (txf 2-44 %), matrix occupancy (mtx 0-30 %), hazards and barrier sleep (up to 50 %). In `op.linear/small` the issue stream is 48,254 instructions in 244,321 cycles (19.8 %, `gemm/ANALYSIS.md:21-30`).
  These families hold 93.6 % of the suite's cycles (Q-C). A wider issue cannot help them: there are no issue-limited cycles to remove.
- **The memory-side families keep the issue slot busy 40-95 % of the cycles**: `embedding` 95 %, `cumsum` 77-85 %, `argmax` 68-76 %, `attention_decode` 59-73 %, `layernorm`, `rmsnorm`, `rope`, `softmax`, `resnet_block`, `conv2d` 45-70 %. The rest of their cycles are
  `lsu` stalls (add 50 % in the tiny run, rope 31 %, transpose 34 %) and FPU/SFU latency (`silu`, `gelu`, `swiglu` 23-28 %), which a wider issue does not remove (they are latencies, not issue slots). These are the 64 memory-side points of the 102 with a recorded binding.
  R42 round 1 moved `add` from issue-bound (B39 small 62 % issued, 22 % lsu) to port-bound (fresh small 29 % issued, 62 % lsu), which is what its board eta of 0.99 says.
- On the board these families wait **longer** for loads (HBM round trip about 73 cycles against the simulator's short memory latency, `BALANCE.md:268`), so their issue share there is lower than shown unless the four contexts hide it.

### 2.3 What is in the hot loops

Static classification of the innermost loops of the compiler's kernels (`qb_loops.py`; 109 kernels of the 25 small-tier points, dumped from the R42 head; a loop is a backward branch; `.li` counted as 4; the software
integer divide loops `sdiv_*` excluded; `qb_loops_table.md`). Slot A = `inc` (`s.addi/s.add` with destination equal to a source: induction variable or pointer bump) + `cmp` + `br` + `const` (`s.movz/movk`, `.li`) + `mov` + `addr` (all other scalar arithmetic) + `pred`.

| point (small tier) | hot innermost loop | instr | slot A | inc | cmp | br | const | mov | addr | pred | vector/ldst/mtx/txf/bar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| model.gpt2_block_small_fp16-d128-h4-b1-s64 | `__rctx_program` (fused__scaled_dot_product_fl) | 670 | 53 % | 0 | 1 | 1 | 74 | 4 | 225 | 49 | vx 157, ld/st 149, mtx 0, txf 4, bar 2 |
| model.llama_decode_small_fp16-b1-p128 | `for_5` (fused__to_copy__unsafe_view_) | 738 | 48 % | 1 | 35 | 6 | 30 | 0 | 279 | 3 | vx 379, ld/st 5, mtx 0, txf 0, bar 0 |
| model.llama_prefill_small_fp16-b1-s64 | `__rctx_program` (fused__scaled_dot_product_fl) | 3400 | 30 % | 0 | 1 | 1 | 94 | 6 | 785 | 128 | vx 1493, ld/st 760, mtx 128, txf 0, bar 0 |
| model.mlp_small_fp16-t64-d128 | `gemm_pipe` (fused_addmm_gelu_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| model.resnet_block_small_fp32-n1-c16-hw16 | `for_1` (fused__native_batch_norm_leg) | 1748 | 20 % | 0 | 4 | 2 | 120 | 2 | 134 | 96 | vx 1116, ld/st 130, mtx 144, txf 0, bar 0 |
| op.add_small_fp16-n16384 | `tiles_1` (fused_add_0) | 126 | 11 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 64, ld/st 48, mtx 0, txf 0, bar 0 |
| op.argmax_small_fp16-r8-v2048 | `for_1` (fused_argmax_0) | 22 | 50 % | 2 | 1 | 1 | 0 | 0 | 0 | 7 | vx 10, ld/st 1, mtx 0, txf 0, bar 0 |
| op.attention_decode_small_fp16-b1-h4-l256-d32 | `for_1` (fused__scaled_dot_product_fl) | 724 | 39 % | 2 | 65 | 1 | 0 | 0 | 192 | 24 | vx 436, ld/st 4, mtx 0, txf 0, bar 0 |
| op.attention_prefill_small_fp16-b1-h4-s64-d32 | `__rctx_program` (fused__scaled_dot_product_fl) | 859 | 22 % | 0 | 1 | 1 | 9 | 3 | 135 | 32 | vx 573, ld/st 101, mtx 0, txf 0, bar 0 |
| op.bmm_small_fp16-b8-m64-k32-n64 | `gemm_pipe` (fused_bmm_0) | 46 | 61 % | 6 | 2 | 3 | 3 | 8 | 6 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.conv2d_small_fp32-n1-cin16-h16-w16-cout16 | `for_1` (fused_convolution_2) | 1748 | 20 % | 0 | 4 | 2 | 120 | 2 | 134 | 96 | vx 1116, ld/st 130, mtx 144, txf 0, bar 0 |
| op.cumsum_small_fp32-r64-c256 | `for_1` (fused_cumsum_0) | 98 | 5 % | 3 | 1 | 1 | 0 | 0 | 0 | 0 | vx 91, ld/st 2, mtx 0, txf 0, bar 0 |
| op.embedding_small_fp16-v1024-d128-t64 | `__rctx_program` (fused_embedding_0) | 681 | 47 % | 0 | 35 | 5 | 3 | 1 | 276 | 1 | vx 356, ld/st 4, mtx 0, txf 0, bar 0 |
| op.gelu_tanh_small_fp16-n16384 | `tiles_1` (fused_gelu_0) | 540 | 2 % | 1 | 1 | 1 | 0 | 0 | 9 | 0 | vx 496, ld/st 32, mtx 0, txf 0, bar 0 |
| op.gemv_small_fp16-k256-n512 | `gemm_pipe` (fused_mm_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.layernorm_small_fp16-r64-d256 | `for_5` (fused_native_layer_norm_0) | 55 | 13 % | 5 | 1 | 1 | 0 | 0 | 0 | 0 | vx 32, ld/st 16, mtx 0, txf 0, bar 0 |
| op.linear_small_fp16-m64-k256-n256 | `gemm_pipe` (fused_addmm_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.matmul_small_fp16-m64-k128-n128 | `gemm_pipe` (fused_mm_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.matmul_small_fp32-m64-k128-n128 | `for_5` (fused_mm_0) | 610 | 4 % | 1 | 1 | 2 | 16 | 4 | 1 | 0 | vx 545, ld/st 0, mtx 32, txf 4, bar 4 |
| op.rmsnorm_small_fp16-r64-d256 | `for_3` (fused__to_copy_add_mean_mul_) | 38 | 16 % | 4 | 1 | 1 | 0 | 0 | 0 | 0 | vx 20, ld/st 12, mtx 0, txf 0, bar 0 |
| op.rope_small_fp16-b1-h4-s64-d32 | `tiles_1` (fused_add_mul_2) | 336 | 4 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 242, ld/st 80, mtx 0, txf 0, bar 0 |
| op.silu_small_fp16-n16384 | `tiles_1` (fused_silu_0) | 332 | 4 % | 1 | 1 | 1 | 0 | 0 | 9 | 0 | vx 288, ld/st 32, mtx 0, txf 0, bar 0 |
| op.softmax_small_fp16-r64-c512 | `for_3` (fused__softmax_0) | 53 | 9 % | 3 | 1 | 1 | 0 | 0 | 0 | 0 | vx 40, ld/st 8, mtx 0, txf 0, bar 0 |
| op.swiglu_small_fp16-n16384 | `tiles_1` (fused_mul_silu_0) | 382 | 4 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 320, ld/st 48, mtx 0, txf 0, bar 0 |
| op.transpose_small_fp32-r128-c256 | `__rctx_program` (fused_clone_t_0) | 325 | 43 % | 0 | 1 | 1 | 33 | 1 | 71 | 32 | vx 118, ld/st 34, mtx 32, txf 0, bar 0 |


(`__rctx_program` is the whole program body when a kernel has no inner loop.)

- **Element-wise kernels** (`add`, `silu`, `gelu`, `swiglu`, `rope`): 126-540 instructions per iteration, 2-11 % slot A (the loop control is 3 instructions of 126-540). They are vector-issue-bound; a scalar slot has nothing to take.
- **Reductions and norms** (`softmax`, `layernorm`, `rmsnorm`, `cumsum`): 38-98 instructions per iteration, 5-16 % slot A, of which `inc` 3-5 (pointer bumps plus the counter).
- **Argmax, attention, embedding, transpose:** 40-50 % slot A. The `addr` column (192-279 per iteration) is scalar address arithmetic for indexed access; `cmp` 35-65 is predicate/mask construction.
- **The GEMM K loop** (`gemm_pipe`, the R42-gemm software pipeline, unrolled by two): 38 instructions per iteration = **19 per K step and matrix command**, 53 % slot A: `br 3 + cmp 2 + inc 2` loop control, `mov 8` (the ISA wants the five transfer coordinates in *consecutive* scalar registers, so every transfer is preceded by `s.mov` copies), `const 3`, `addr 2`;
  slot B is `bar 8 + txf 8 + mtx 2`. 19 instructions per command while one command occupies the engine 168 cycles: not issue-bound today.
- **Program entry and epilogue**: the entry computes the tile index with a software divide (`sdiv_*` loops: the ISA has no integer divide, `architecture_overview.md:126`), 2 TensorMap builds and a fence; for `op.linear/tiny` this is 557 cycles for two contexts (9 %), the epilogue
  224 instructions per tile (34 % of the tiny point, 14 % of the small one) (`gemm/ANALYSIS.md:46,56`). Both are issue-bound and both are compiler-removable.
- **FP8 Hub kernel** (`BALANCE.md:195-229`, recorded assembly of the run-1 profile): 11,192 instructions per 262,144-MAC iteration, vector int/move 31.6 %, scalar/predicate/control 29.8 %, vector FP 19.5 %, vector load/store 15.2 %, matrix 3.8 %;
  **bookkeeping plus scalar address arithmetic is 6.3 k = 56 %**. Against an engine that needs 5,824 cycles per iteration the kernel is issue-bound (`BALANCE.md:18-19`).

Dynamic share of issued instructions in the fresh tiny runs, per family (slot A = salu + ctrl + pred + sys):

| tier | family | issued | salu | ctrl | pred | sys | ld/st | vector | matrix | transfer | barrier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | add | 8484 | 13.7% | 1.6% | 0.0% | 0.1% | 36.4% | 48.3% | 0.0% | 0.0% | 0.0% |
| small | argmax | 25184 | 13.1% | 4.1% | 31.8% | 0.0% | 4.2% | 46.8% | 0.0% | 0.0% | 0.0% |
| tiny | add | 1308 | 19.9% | 2.4% | 0.0% | 1.8% | 20.2% | 55.7% | 0.0% | 0.0% | 0.0% |
| tiny | argmax | 1378 | 11.5% | 2.6% | 31.5% | 0.4% | 2.9% | 51.1% | 0.0% | 0.0% | 0.0% |
| tiny | attention_decode | 5638 | 19.7% | 1.3% | 2.1% | 0.6% | 5.3% | 71.1% | 0.0% | 0.0% | 0.0% |
| tiny | attention_prefill | 12218 | 18.8% | 1.0% | 4.2% | 0.8% | 5.7% | 68.1% | 1.0% | 0.2% | 0.2% |
| tiny | bmm | 626 | 39.6% | 1.3% | 10.2% | 3.2% | 11.8% | 23.3% | 5.8% | 1.9% | 2.9% |
| tiny | cumsum | 1712 | 8.2% | 1.4% | 0.0% | 0.7% | 2.3% | 87.4% | 0.0% | 0.0% | 0.0% |
| tiny | embedding | 44088 | 45.8% | 0.4% | 0.4% | 0.0% | 0.6% | 52.7% | 0.0% | 0.0% | 0.0% |
| tiny | gelu_tanh | 1044 | 30.3% | 1.1% | 0.0% | 1.1% | 3.8% | 63.6% | 0.0% | 0.0% | 0.0% |
| tiny | gemv | 1186 | 43.3% | 3.2% | 5.4% | 1.2% | 8.8% | 18.2% | 7.1% | 5.7% | 7.1% |
| tiny | layernorm | 956 | 40.2% | 5.9% | 0.0% | 1.3% | 11.7% | 41.0% | 0.0% | 0.0% | 0.0% |
| tiny | linear | 1522 | 38.8% | 2.5% | 4.2% | 0.9% | 11.3% | 26.8% | 5.5% | 4.5% | 5.5% |
| tiny | llama_decode | 85350 | 32.7% | 1.9% | 1.9% | 0.5% | 5.4% | 50.8% | 2.9% | 1.7% | 2.2% |
| tiny | matmul | 1750 | 16.6% | 0.9% | 1.8% | 0.8% | 4.2% | 67.4% | 5.7% | 1.1% | 1.3% |
| tiny | rmsnorm | 660 | 44.2% | 6.1% | 0.0% | 1.8% | 11.5% | 36.4% | 0.0% | 0.0% | 0.0% |
| tiny | rope | 2380 | 9.1% | 1.8% | 0.0% | 1.5% | 10.9% | 76.6% | 0.0% | 0.0% | 0.0% |
| tiny | silu | 668 | 28.1% | 1.8% | 0.0% | 1.8% | 6.0% | 62.3% | 0.0% | 0.0% | 0.0% |
| tiny | softmax | 1574 | 38.4% | 5.5% | 0.3% | 1.5% | 9.0% | 45.4% | 0.0% | 0.0% | 0.0% |
| tiny | swiglu | 768 | 26.6% | 1.6% | 0.0% | 1.6% | 7.8% | 62.5% | 0.0% | 0.0% | 0.0% |
| tiny | transpose | 1312 | 32.3% | 0.6% | 9.8% | 1.2% | 10.4% | 36.0% | 9.8% | 0.0% | 0.0% |


### 2.4 Amdahl bounds of a two-slot core, and of the alternatives

- **Two-wide in-order front end, one scalar/control slot and one vector/matrix/transfer slot (binary compatible):** per-point geometric mean of the bound **1.111x** (tiny, fresh runs); GEMM families 1.05-1.10x; `embedding` 1.53x; `argmax` 1.18x;
  pointwise and norms 1.04-1.11x. The loose bound (any slot-A instruction pairable with any slot-B instruction) is 1.207x. These are upper bounds: they assume that every adjacent pair of different slots can issue together and that every saved issue cycle
  shortens the run, which is false in the stall-dominated families.
- **Very long instruction word with the same two slots:** the same upper bound (it can pair no more than the 2-wide front end can); it adds nop padding for the unpaired half of each bundle, and a bundle must wait for its slowest slot unless the scoreboard stays.
  There is no number for it to beat.
- **Removing slot A instead of hiding it:** the bound of removing *all* of them is 1.213x (geometric mean over the tiny points; `embedding` 1.80x). The reachable part is smaller:
  loop control (`inc` of the counter + `cmp` + `br`) is 3 instructions per iteration, 0.6-2.4 % of element-wise bodies, 5-16 % of reduction bodies, 18 % (7 of 38) of the GEMM K loop;
  `.li` constants (4 instructions each) are hoistable by the compiler; `mov` rotation moves in the pipelined GEMM loop (8 of 38) disappear with a 2x unrolled body; the divide at program entry is removable by strength reduction;
  descriptor loads replace the Hub kernel's weight-row assembly.
- **FP8 Hub kernel** (`BALANCE.md` classes): slot A is 3,339 of 11,192 instructions (29.8 %), slot B 7,853. A perfect two-wide pairing leaves 7,853 issue cycles per iteration (1.43x), still above the engine's 5,824: **still issue-bound**.
  The software route (descriptor weight loads, hoisted activation quantisation, no per-row bookkeeping) gives 2,652 (4.2x) and makes the kernel engine-bound at about 70 % of the DP4 peak (`BALANCE.md:19`, a projection).

### 2.5 The options and their cost

Areas are LUT-equivalents per core, derived from recorded module sizes; **no unit was synthesised** (inference where marked). SLR headroom (sec 1.4): SLR0 about 4k, SLR1 about 22k, SLR2 about 28k LUTs.

| option | what it removes or hides | hardware (basis) | ISA / binaries | software | fit |
|---|---|---|---|---|---|
| **A. 2-wide in-order, slot A (scalar/pred/control) + slot B (rest), same encoding** | hides up to min(slot A, slot B): bound 1.05-1.2x on tiny, 1.43x on the Hub kernel | fetch: 128 bit per cycle from the same RAMB36 (even/odd banks, +0 BRAM, +~130 FF) ; a second decoder (the whole "decoders + salu + prf + mul + mux" bucket is 4,117 LUTs, `RESOURCES.md:110`: +1-2k, inference); scoreboard/R-stage check of the pair (u_core 6.8k LUT synth. / 23.3k routed: +3-7k, inference); 2 more scalar-RF read ports (296 LUTRAM + 536 LUT per port, derived from 7 ports: +0.6k LUTRAM +1.1k LUT) and a second write port or a pairing rule that forbids two scalar writers (+2.1k LUTRAM for a duplicated file); second writeback reservation (+0.3k, inference). **Sum +6k..12k** (inference); the R stage gets one logic level longer on a +0.111 ns margin | existing binaries run unchanged | scheduler gets a pairing-aware priority (`schedule.py` is a list scheduler with a measured latency model, 542 lines) | SLR1/SLR2; **not SLR0** |
| B. cross-context pairing (slot A of context X with slot B of context Y) | same hiding, only when two contexts are live (SPMD GEMM uses two) | as A without the intra-pair dependence check (contexts do not share registers), but 2 fetch ports and a second scoreboard read | unchanged | none | SLR1/SLR2 |
| **C. VLIW, 128-bit bundle, same two slots** | same as A, no better | the scoreboard must stay (matrix, transfer, memory completions have variable latency); saves only the pair-legality logic; code store holds fewer instructions per RAMB36 (nop padding) | **new container and encoding: every binary changes**; assembler (`tnca_asm.py` 727 lines), ISA generator and schema (`tnca_isa.py` 384 lines, `isa/tnca-isa-v0.1.json`), disassembler, RTL decoder (909 lines) and fetch, the SystemC model (`tools/performance_harness/systemc`, about 4.1k header lines) and every instruction-level reference model | codegen (1,549 lines) and scheduler emit bundles; all 113 binaries regenerate | no advantage over A; not recommended |
| **D. zero-overhead hardware loop** (`loop Sn, end` = start/end PC and count per context) | `inc` + `cmp` + `br` of every counted innermost loop (3 instr/iteration) and the not-taken-prediction bubble of the backward branch (`perf_squash` 80 in 6,262 cycles of `linear/tiny`) | 4 contexts x about 64 FF + PC comparators + a fetch-redirect input: **+0.3-0.5k** (inference) | new `CTRL` func (9 of 256 used); old binaries unchanged | frontend emits it for counted `for_N`/`tiles_N` loops, one scheduler rule (loop end ends the block) | fits any SLR incl. SLR0 |
| E. post-increment addressing (`v.ld/v.st/s.ld` with address register += stride) | pointer bumps (`inc` minus the counter): 1-6 of 10-55 per iteration in reductions | needs a second scalar-RF write port (SRF has one): **about the price of A's write port, +2k LUTRAM** (inference) | new `mod` bits in LSU | allocator + scheduler | low value per LUT: rank below D |
| F. macro-tile `m.mma` sweeping K from a window ring | the whole K-loop instruction stream (38 per step) | matrix-sequencer loop + ring/barrier handshake with the transfer engine: new design, the matrix port and `tnca_tensor` interact | new opcode, new oracle | new GEMM lowering | only worth it with a wide engine (sec 2.6) |
| G. software: hoist `.li`, strength-reduce address math and the entry divide, descriptor-driven fragments (FP8 weights), epilogue with `a.vread`-based bias add, unroll by 2 to drop rotation `mov`s | slot A directly, in every family; Hub kernel 11,192 -> 2,652 | none (an A buffer of 16 registers for K32/K64 line eligibility is optional, `BALANCE.md:369`) | none | compiler passes in R42-gemm/red/attn | all images |

### 2.6 Conclusion: what the evidence supports first, and what would make issue pay

1. **First: remove bookkeeping in the compiler (G).** It acts on every family, needs no image, and is the only route that changes the Hub kernel's regime (1.43x ideal for A against 4.2x for descriptors). For the GEMM families
   the removable part is the entry/epilogue and the `mov`/`const`/loop-control share of the K loop, not the K loop's issue (which is not binding).
2. **Second: the hardware loop (D).** Binary-compatible, a few hundred LUTs, fits SLR0, and acts on the reductions/norms/attention loops where 15-50 % of the body is scalar. Its gain is bounded by the loop-control share
   (5-16 % in norms, 18 % in the GEMM K loop, 1-2 % in element-wise bodies).
3. **Third, only with a faster engine: the 2-wide front end (A) or cross-context pairing (B).** Reason, derived from recorded numbers: one K step of the pipelined GEMM loop issues 19 instructions (static `gemm_pipe`, 38 per two steps; 20 before the pipeline,
   `gemm/ANALYSIS.md:47`), and a 16 x 16 output tile adds a 220-instruction epilogue (static, `endfor_6` to `__rctx_next` of the R42-head `op.linear/small` dump; `gemm/ANALYSIS.md:56`: 224) and an entry of at least 142 instructions plus the software divides and a K16 command occupies the engine for `8 + (16/G) x 17 + 6` cycles (structural: A-buffer fill, one 17-cycle sweep per group of G products, drain; 150 for the G = 4 K32 E4M3 case `BALANCE.md:69`):

   | engine group G (FP16 K16; E4M3 K32 at 2G) | engine cycles per K16 command | engine cycles per tile, K = 256 (16 commands) | issue demand per tile: 16 x 19 + 220 | issue slot busy |
   |---:|---:|---:|---:|---:|
   | 2 (today; 168 cycles measured per command) | 150 | 2,400 (2,688 measured) | 524 | 22 % (19.8 % measured: 48,254 / 244,321) |
   | 4 | 82 | 1,312 | 524 | 40 % |
   | 8 | 48 | 768 | 524 | 68 % |
   | 16 | 31 | 496 | 524 | **106 %: issue-bound** |

   (derived from the structural command time `8 + (16/G) x 17 + 6`: A-buffer fill, one 17-cycle sweep per group of G products, drain; the entry (>= 142 instructions + the divides, 557 cycles for two contexts in the base) is not even included.) The K loop alone is 61 % busy at G = 16 (19 per 31 cycles); the per-tile
   epilogue and entry are what saturate the slot. With G (compiler) and D removing the `mov`, `const` and loop-control share of the K step (about 10 of 19) and the epilogue shortened (`gemm/ANALYSIS.md:99-101`: 14 instructions a row where 4-5 would do), the tile demand falls to roughly 250, 33 % at G = 16.
   So issue starts to bind between G = 8 and G = 16, exactly when the engine of sec 1 is widened; for G <= 4 it does not bind in the GEMM families.
4. **VLIW (C): no.** It cannot hide more than A, it keeps the scoreboard, and it invalidates every binary and four tool chains for no gain.
5. **SLR0 headroom** (about 4k LUTs) excludes A in tile 0; B and D are the only options that fit there. A would have to live in the SLR1/SLR2 cores (an asymmetric image) or be paid for by removals.

Confidence: facts of 2.1 **high**; the cycle partitions **medium** (single-core, tiny/small points, no board counters; the B39 columns are older software); the Amdahl bounds **medium** (upper bounds by construction); areas **low**.


---

## 3. Q-C: what the suite's cycles are made of

Source: `evidence/r42/board/ab-int1/comparison.json` (board, run-3 image, R42 software round 1 = `new`; `old` = same image with unchanged `main` software, `b41-run3-base`), per-point eta from
`evidence/r42/baseline/b41_board.json` (run-3 eta; new eta = old eta x old cycles / new cycles, **derived**: the ArchBench bound is a property of the point),
demand FLOP and roofline binding from `evidence/r41/stage2/roofline/points.json`. Script `evidence/r42/arch/qc_families.py`, output `qc_families.txt`.
113/113 points pass on both sides; geometric mean new/old 0.6201 (1.613x), total cycles 48,181,899,903 -> 36,915,323,718.

**Reading the columns.** *share of the log-gain* is each family's part of the sum of ln(new/old); *share of remaining log(1/eta)* is each family's part of the sum of
ln(1/eta_new) over the 102 points that have a FLOP bound (embedding, transpose, copy have none): a measure of how far the family is from the ArchBench bound, weighted the way the geometric mean weights it.
Eta is relative to the single-core spec bound (see 1.2); with three cores a compute-side eta above 1 is possible, so "remaining" is a ranking, not a cap.

| family | n | cycles share old | cycles share new | speed-up (geo) | share of the 1.613x log-gain | eta old | eta new | share of remaining log(1/eta) | compute-side pts | best GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_prefill | 5 | 73.1% | 71.4% | 1.51x | 3.8% | 0.222 | 0.335 | 3.7% | 5/5 | 4.10 |
| matmul | 9 | 10.0% | 11.8% | 1.21x | 3.1% | 0.324 | 0.390 | 5.8% | 7/9 | 4.30 |
| linear | 5 | 2.8% | 3.6% | 1.01x | 0.1% | 0.670 | 0.673 | 1.3% | 5/5 | 5.29 |
| llama_decode | 5 | 2.6% | 2.9% | 1.73x | 5.1% | 0.110 | 0.191 | 5.6% | 0/5 | 0.26 |
| mlp | 4 | 1.8% | 2.3% | 1.02x | 0.2% | 0.519 | 0.532 | 1.7% | 4/4 | 3.98 |
| resnet_block | 4 | 1.6% | 1.7% | 1.25x | 1.7% | 0.035 | 0.044 | 8.5% | 4/4 | 0.57 |
| transpose | 5 | 1.0% | 1.3% | 1.01x | 0.1% | - | - | - | 0/5 | 0.00 |
| gpt2_block | 4 | 1.3% | 1.2% | 1.62x | 3.6% | 0.246 | 0.398 | 2.5% | 4/4 | 3.14 |
| embedding | 5 | 0.7% | 0.9% | 1.00x | 0.0% | - | - | - | 0/5 | 0.00 |
| conv2d | 4 | 0.7% | 0.9% | 1.24x | 1.6% | 0.032 | 0.040 | 8.8% | 3/4 | 0.58 |
| attention_prefill | 4 | 1.5% | 0.6% | 2.66x | 7.2% | 0.021 | 0.055 | 7.9% | 3/4 | 0.48 |
| silu | 5 | 0.6% | 0.4% | 1.79x | 5.4% | 0.145 | 0.259 | 4.6% | 0/5 | 0.19 |
| softmax | 6 | 0.6% | 0.4% | 1.73x | 6.1% | 0.078 | 0.134 | 8.2% | 0/6 | 0.15 |
| add | 6 | 0.4% | 0.1% | 2.96x | 12.1% | 0.334 | 0.989 | 0.0% | 0/6 | 0.16 |
| gemv | 5 | 0.1% | 0.1% | 0.99x | -0.1% | 0.318 | 0.314 | 3.9% | 0/5 | 0.29 |
| rmsnorm | 5 | 0.2% | 0.1% | 3.12x | 10.5% | 0.121 | 0.378 | 3.3% | 0/5 | 0.34 |
| swiglu | 4 | 0.1% | 0.1% | 1.70x | 3.9% | 0.188 | 0.319 | 3.1% | 0/4 | 0.22 |
| bmm | 4 | 0.5% | 0.1% | 4.63x | 11.4% | 0.092 | 0.426 | 2.3% | 3/4 | 3.58 |
| gelu_tanh | 4 | 0.1% | 0.1% | 1.88x | 4.7% | 0.116 | 0.219 | 4.1% | 0/4 | 0.30 |
| attention_decode | 4 | 0.1% | 0.1% | 1.56x | 3.3% | 0.057 | 0.089 | 6.6% | 0/4 | 0.09 |
| copy | 1 | 0.2% | 0.0% | 5.73x | 3.2% | - | - | - | 0/1 | 0.00 |
| argmax | 4 | 0.0% | 0.0% | 1.61x | 3.5% | 0.115 | 0.185 | 4.6% | 0/4 | 0.12 |
| rope | 4 | 0.0% | 0.0% | 1.47x | 2.9% | 0.073 | 0.108 | 6.1% | 0/4 | 0.06 |
| layernorm | 4 | 0.0% | 0.0% | 2.23x | 5.9% | 0.093 | 0.208 | 4.3% | 0/4 | 0.30 |
| cumsum | 3 | 0.0% | 0.0% | 1.17x | 0.9% | 0.198 | 0.231 | 3.0% | 0/3 | 0.03 |


Observations (all derived from the table and the records):

1. **Total cycles are one point.** `model.llama_prefill/large/fp16-b1-s512` is 66.7 % of the suite's new device cycles (1.01e12 FLOP at 4.10 GFLOP/s); `llama_prefill` 71.4 %, `matmul` 11.8 %.
   The nine families whose points are mostly compute-side (`llama_prefill`, `matmul`, `linear`, `mlp`, `resnet_block`, `gpt2_block`, `conv2d`, `attention_prefill`, `bmm`) hold 93.6 % of the cycles in 43 points.
2. **The geometric mean is the other 70 points.** 64 of the 102 points with a recorded binding are memory-side (bytes bound) and hold 6 % of the cycles; with equal weights they are 62 % of the metric.
   After round 1 their eta is 0.09-0.38 (except `add`, 0.99) and they are bound by instruction issue and the load/store path, not by memory (G4 issue ceilings, `G4_DESIGN.md:14-36`): `softmax` 0.134, `attention_decode` 0.089,
   `rope` 0.108, `argmax` 0.185, `layernorm` 0.208, `gelu_tanh` 0.219, `silu` 0.259.
3. **Round 1 closed the cheap gaps:** `add` is at eta 0.989 (its bound), `bmm` 4.63x, `rmsnorm` 3.12x, `add` 2.96x, `attention_prefill` 2.66x; five families did not move (`linear` 1.006x, `mlp` 1.025x, `gemv` 0.99x,
   `embedding` 1.00x, `transpose` 1.006x) and `matmul` moved 1.21x.
4. **Largest remaining distance to the bound:** `conv2d` and `resnet_block` (FP32: no matrix command, vector FPU at 0.58/0.57 GFLOP/s, 6 % of the 9.6 GFLOP/s FMA peak; `R42-fp32` is the work package),
   then `softmax`, `attention_prefill`, `attention_decode`, `rope`, then `matmul` (5.8 %: its 9 points include FP32 matmul at eta 0.1).
5. **Compute-bound structure** (ArchBench binding = FLOP): `linear` (eta 0.67), `mlp` (0.53), `gpt2_block` (0.40), `matmul` (0.39), `llama_prefill` (0.34), `bmm` (0.43; 3 of 4 points). Their gap to a single-core
   spec bound is 1.5-3x and, per sec. 1.2, comes from transfer/issue/epilogue occupancy around the matrix engine, not from its width. The FP32 families (`conv2d`, `resnet_block`, FP32 `matmul`) are compute-bound on the *vector* FPU.
   `llama_decode` is memory-side by binding (eta 0.19) but its cycles are mostly `m.mma` M=1 GEMV-like launches; `gemv` (0.31) did not move.
6. **What the dominant total point says about the two questions:** a further 1.3x on `llama_prefill/large` would save 15.4 % of the suite's total cycles and move the geometric mean by 1.3^(1/113) = 1.002x (derived);
   a 1.3x on each of the 70 memory-side points would move the geometric mean by 1.3^(70/113) = 1.18x (derived). So the width of the matrix engine (Q-A) acts on the cycle total and on 43 points;
   issue and bookkeeping (Q-B) act mostly on the metric.

---

## 4. Provenance, scripts and limits

Scripts and tables (all in `evidence/r42/arch/`, small): `qa_budget.py` -> `qa_budget.txt`; `achieved_gflops.py` -> `achieved_gflops.txt`; `run3_kernel_clock_paths.txt` (extract of the run-3 routed report);
`qc_families.py` -> `qc_families.txt`; `qb_b39.py` -> `qb_b39.txt` (recorded B39 counters); `qb_issue.py`, `qb_family.py` -> `qb_tiny.txt`, `qb_small.txt` (fresh counters + dynamic instruction mix);
`qb_loops.py` -> `loops_small.txt` (static innermost-loop classes of 109 kernels); `tools/arch_probe.v`, `arch_mkprobe.sh` (a second simulation top that logs `x_valid`, `x_major`, `x_func`; derived from `evidence/r42/gemm/tools/gemm_probe.v`; not RTL).

Fresh simulations (domain **rtl**, single-core device `tb/device/tnca_device.v`, iverilog, profile `core-vrf-sync-v5-x8-g2c4-dp`, default knobs, R42 head `5327118c` mirrored to `~/r42/arch` on thor,
`nice -n 19`, at most 4 processes): `archbench.run --platform tnca --profile P --point ID --counters` with `TNCA_DEVICE=~/r42/arch_out/probe/device.vvp`, one process per point; compile-only kernel dumps with
`TNCA_KERNEL_DUMP=1` for the 25 small-tier points. Raw outputs stay on thor (`~/r42/arch_out/{tiny1,small1,dumps_small}`); the tables above are committed.

Limits, stated once:
- No board counters exist (`perf_*` are not wired above the tile, `RESOURCES.md:156`); every issue/stall figure is single-core simulation. The single-core device has the word transfer engine and a short memory latency, the board has the line
  engine, three cores and an HBM round trip of about 73 cycles (`BALANCE.md:268`, `LINK_BUDGET.md` sec 4): load-latency-bound families wait longer on the board, so their issue share there is lower than shown; GEMM families differ in the transfer/matrix ratio
  (`gemm/ANALYSIS.md:5-7`).
- `perf_x_frontend` includes cycles in which every running context sleeps in `mb.wait` (the counter increments when a context is running, nothing stalls and R is empty, `tnca_core.v:2960-2961`); it is **not** the R39 "front-end-bound 4.3 %" fetch-bubble figure of `r42_plan.md:81`.
- The area numbers of sections 1.4 and 2.5 are linear extrapolations of recorded unit screens, not synthesis of the proposed units. The Amdahl bounds ignore dependences and structural hazards on purpose (upper bounds).
- Run 3's own routed SLR report is not in the recorded tree; the run-2 link (same RTL but one line) stands in for it.
