# R38: Board Bring-Up Preparation and Execution Checklist (U50 / U280)

Version 0.5, 2026-10-09. State: **U280 loaded and passed acceptance ladder B1–B6** (acf3019; see §4, §5; 75 MHz and 100 MHz verified, see §6.10, §6.13).
Parent documents: [Stage 1](r38_stage1_tasks_and_acceptance.md) §5 "Hand-off Artifacts Entering Stage 2"; board flow and existing physical measurements: [flow/fpga/README.md](../flow/fpga/README.md); measurement protocol: [archbench/PROTOCOL.md](../archbench/PROTOCOL.md).

## 0. Context: Board Image Lagged Behind r38 RTL

The image recorded in `flow/fpga/README.md` (loaded 2026-09-26, uuid `ccbbcf8b-…`, 150 MHz bus / 37.5 MHz device) was built from the **fpga-u50 branch** RTL. After merging into r38, the device-side RTL had the following changes that the old image **did not contain**:

| Change | Files | Impact on Board Bring-Up |
|---|---|---|
| Counter read port `perf_sel/perf_rdata`, two data-port beat counters, `done_retired` fix (reporting only this command's instruction count) | `rtl/tnca_shell.v` | On the old image, starting from the second program in a grid, `retired` is inflated; counters are unreadable |
| Mailbox op 7 `PERF_READ` (same contract as simulation bench: r0/r1 low/high 32 bits, out-of-bounds `REFUSED`) | `rtl/fpga/tnca_mmio.v`, `rtl/fpga/tnca_fpga_top.v` | Old image returns `REFUSED` for op 7, runtime reports `counters: unavailable` (not 0) |
| Transposed writeback engine (`HAS_TENSOR_XPOSE`, default 0) and unimplemented mod bits changed to `ILLEGAL_INSTRUCTION` | `rtl/core/tnca_tensor.v` | The board profile `core-vrf-sync-v1` does not declare this capability; compiler does not emit it; when disabled, engine matches parent commit |
| R3.7 compiler / ISA coverage | Compiler | `fpga-u50` already included R3.7 (squash), no difference |

Therefore board bring-up was divided into two parallel tracks: **A** use the loaded old image for initial functional validation (no counters); **B** rebuild `.xclbin` with current RTL (~2.5 hours) to obtain an image with counters for formal measurements.

## 1. Preparations Completed Before Board Bring-Up (Verifiable within Repository)

| Item | Evidence |
|---|---|
| fpga-u50 merged into r38, board RTL gates all pass | `make r36-fpga-tests`: R36-FPGA-01…08 (08 is the register protocol bench, including `PERF_READ`) |
| `PERF_READ` usable through full board software stack | `tools/r38/profiler_tests.py --quick --backend fpga-u50-sim --profiles core-vrf-sync-v1`: conservation / identity / monotonic / numerical all PASS |
| PyTorch → Inductor → Triton → tnca → board protocol full flow | `python -m archbench.run --platform tnca --backend fpga-u50-sim --profile core-vrf-sync-v1 …` (same `U50Backend` as board, same `tnca_fpga_top`, swapping only XRT for simulation) |
| Backend switchable via environment without code changes | `TNCA_BACKEND=fpga-u50` (or `fpga-u280`), `TNCA_BACKEND_ARGS='{"xclbin": "<path>"}'` (`python/tnca/triton_backend/driver.py`) |
| Cycles, program count, transferred bytes recorded per kernel launch | `tnca.triton_backend.driver.launch_log()`, ArchBench-recorded `time.per_launch`, `host` |
| "Expected board cycles": board design RTL simulation results | ArchBench tiny/small records on `fpga-u50-sim` (see §4 B4) |
| GPU baseline | ArchBench tiny/small/large records on GB10 (`archbench/PROTOCOL.md` §3.2) |

## 2. Steps Upon Receiving FPGA Server Access

All commands are executed from the repository root; Vivado/v++/XRT environments are loaded per server configuration.

### A. Old Image (if Still Loaded): Functional Bring-Up First, ~30 Minutes

```sh
flow/fpga/program.sh --status                       # currently loaded image, thermals
python tools/r36/conformance.py --xclbin <loaded.xclbin> --require-measured
python -m archbench.run --platform tnca --backend fpga-u50 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "<loaded.xclbin>"}' --tier tiny --jobs 1 --out results/u50_old_tiny.jsonl
```

Expected: in conformance, `fpga-u50` changes from N/A to measured; tiny tier numerical checks all pass; counters report unavailable. **The board can only be used serially: `--jobs 1`.**

### B. Rebuild Image with r38 RTL, ~2.5 Hours (in Parallel with A)

```sh
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs $PWD <build>/pkg
flow/fpga/link.sh <build>/pkg                       # 150 MHz, device /4 = 37.5 MHz; link.sh reads back and verifies frequency
flow/fpga/program.sh --check <build>/pkg/tnca_u50.xclbin
flow/fpga/program.sh --go    <build>/pkg/tnca_u50.xclbin   # only step that changes hardware state; thermal thresholds apply as usual
```

Record: WNS/WHS from link report, resources (LUT/FF/BRAM/DSP), `.xclbin` sha256 and uuid, written into `flow/fpga/README.md`. If timing fails to close, follow "If the link fails on timing again" in that file; do **not** unilaterally reduce frequency.

### C. Acceptance Ladder on New Image

| Step | Command | Passing Condition |
|---|---|---|
| B1 | `python tools/r36/conformance.py --xclbin X --require-measured` | 5/5 backend measured |
| B2 | `python tools/r38/profiler_tests.py --quick --backend fpga-u50 --backend-args '{"xclbin":"X"}' --profiles core-vrf-sync-v1` | Conservation, identity, monotonicity all hold (same checks as `fpga-u50-sim`) |
| B3 | `python -m archbench.run --platform tnca --backend fpga-u50 --profile core-vrf-sync-v1 --backend-args '{"xclbin":"X"}' --tier tiny small --counters --jobs 1 --out results/u50.jsonl` | Numerical checks all pass; no host fallback |
| B4 | `python -m archbench.report results/u50.jsonl results/u50sim.jsonl` and per-launch cycle comparison | **Board cycles = `fpga-u50-sim` cycles, identical launch-by-launch** (same RTL, same fixed-latency memory; any discrepancy indicates a mismatch between board and simulation model and must be explained) |
| B5 | `python -m archbench.report results/u50.jsonl results/gpu.jsonl --tier tiny small` | Board records include `time.device_seconds` (37.5 MHz), first enabling speedup ratio S against GPU |
| B6 | LLM demonstration: `model.llama_decode/small` multi-step, `model.llama_prefill/small`; screenshot includes bitstream hash, command, tokens/s | Board evidence for roadmap §12.4 P2–P3 (P4/P5 require larger memory, see §3) |

Save per run: JSONL records, `program.sh --status` output (uuid, thermals, clock), `xbutil examine` clock counter.

## 3. Known Limitations and Next Decision Points

- **Memory capacity**: Device memory is 512 KiB on-chip BRAM (`WORDS=65536`), heap is 384 KiB; argument storages per launch must fit within it. Hence small tier is designed accordingly, while large tier (1B models) records `capacity-exceeded` on board. Two expansion paths: URAM (~22.5 MB on U50, modify `WORDS/ADDR_W` and `tnca_fpga_mem` inference, requires re-timing) or HBM (add AXI master and DMA, high engineering effort). Needs prioritization decision.
- **Transfer overhead**: In Stage 1, each launch uploads/downloads argument storages word-by-word through the register window; on board this is host time (recorded separately, excluded from device time), causing end-to-end latency to greatly exceed device time. Keeping tensors resident on device is roadmap P2 (PrivateUse1) work.
- **Single-core serial grid**: Device time is the sum of program cycles (`serial-grid`), not concurrent multi-core makespan.
- **Thermals**: Previous run logged FPGA 91–95 °C, HBM 86–89 °C, exceeding the 90/85 °C abort threshold and proceeding only under explicit instructions; this threshold remains active: abort and report if exceeded.

## 4. Actual Board: Alveo U280 on acf3019 (2026-10-07)

The provided FPGA host contains an Alveo U280 rather than a U50. Device RTL, kernel (`tnca_u50`), and register protocol remain unchanged; only card-specific values change, centralized in `flow/fpga/boards.sh` and selected by `TNCA_BOARD`:

| Item | Value (Read on Host) |
|---|---|
| Access | Via GB10 host jumpbox: `ssh zhongchun@143.89.78.224`, then `ssh zzhouch@acf3019.ece.ust.hk` (keys on jumpbox) |
| Host | Ubuntu 22.04, 24 cores, 62 GB RAM, XRT 2.16 (2023.2), Vivado/Vitis 2021.1 and 2023.2; multi-user shared |
| Card | `xilinx_u280_gen3x16_xdma_base_1`, BDF `0000:04:00.1`, interface UUID `fb2b2c5a19ed63593fea95f51fbc8eb9` |
| Platform / Device | `xilinx_u280_gen3x16_xdma_1_202211_1`, `xcu280-fsvh2892-2L-e` |
| Clock | Platform kernel clock 0 defaults to 300 MHz; this design originally requested 150 MHz, device /4 = 37.5 MHz |
| Backend name | `fpga-u280` (uses same `U50Backend` as `fpga-u50`) |

Commands (at repository root on acf3019):

```sh
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs $PWD <build> xcu280-fsvh2892-2L-e
TNCA_BOARD=u280 XILINX_VITIS=/tools/Xilinx/Vitis/2023.2 VPP_JOBS=8 flow/fpga/link.sh <build>
TNCA_BOARD=u280 flow/fpga/program.sh --check <build>/tnca_u280.xclbin
TNCA_BOARD=u280 flow/fpga/program.sh --go    <build>/tnca_u280.xclbin      # see sharing rules below
python -m archbench.run --platform tnca --backend fpga-u280 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "<build>/tnca_u280.xclbin"}' --tier tiny small --counters --out results/u280.jsonl
```

**Build records** (`<build>` = `~/tnca_build/u280`, RTL identical to HEAD of branch):

| Attempt | Placement | Outcome |
|---|---|---|
| 1 | Left to placer | kernel spans SLR0/SLR1; ap_clk WNS −1.919 ns, 32 endpoints, all one path in AXI-Lite adapter (`w_have` → `shadow_lo[*]/CE`: 1 LUT level, 8.2 ns routing, one SLR0→SLR1 crossing). Failed |
| 2 | `--connectivity.slr tnca_u50_1:SLR1` (`boards.sh`) | synth 44 min, impl 3 h 13 min. Routed full-design WNS +0.016 ns, 0 failing endpoints; kernel clock domain WNS +4.229 ns. `.xclbin` reads back DATA_CLK = 150 MHz, device clock 37.5 MHz. `tnca_u280.xclbin` sha256 `f8717adc…a77a52bec` |

Attempt 2 exited v++ with 0, but `link.sh` initially reported failure (exit code 3): its check 1 matched any `AUTO-FREQ-SCALING` rule, whereas v++ printed `AUTO-FREQ-SCALING-08`. That advisory indicates timing closed with slack and the original frequency was maintained ("will select the original specified frequency of 150.0 MHz"). Check 1 now passes 08 while failing other rule IDs (including new ones in future versions). `LINK_RECHECK=1` re-runs checks on an existing link without re-running v++; re-check yielded `link.sh: OK`.

**Sharing rules**: The card had another user's image loaded (an HBM bandwidth test on 2026-10-07, CU idle), and other users ran Vivado tasks on the host. `program.sh --go` replaces the loaded image, so confirm no process is using the device and obtain owner consent before loading; packaging/linking does not touch the board, and `VPP_JOBS` restricts parallelism.

Expected values: Board cycles should match the `core-vrf-sync-v1` simulation bench launch-by-launch (`fpga-u50-sim` already proved top-level cycle equivalence to that bench on RTL); tiny tier expected records are generated by `python -m archbench.run --platform tnca --profile core-vrf-sync-v1 --tier tiny`.

## 5. U280 Board Results (2026-10-08)

With lead approval, loaded `tnca_u280.xclbin` at 2026-10-08 09:38 using `program.sh --go` (sha256 `f8717adc…a77a52bec`, xclbin uuid `083faace-4e12-8434-c841-947e12a47ba8`), replacing prior image `483a6e3f…`. Pre-load gate checks all passed: device idle, no other loading process, FPGA 32 °C, HBM 27 °C. After loading, the HBM sensor reported readings within 2 s (26 °C). Management clock frequency counters (`clock_counter_freqs`) reported kernel clock 150 MHz, giving a 37.5 MHz device clock.

Results recorded in `~/tnca_build/u280/results/` on acf3019.

| Step | Command Highlights | Result |
|---|---|---|
| B1 | `tools/r36/conformance.py --backends fpga-u280 --xclbin X --require-measured` | **5/5 PASS**, 1 of 1 backend measured. Successful launch: 23 cycles, oracle values correct. Fault launch reports code 0x40 and its PC; session remains usable afterwards. |
| B2 | `tools/r38/profiler_tests.py --backend fpga-u280 …` (first `--quick`, then full acceptance) | **PASS**: conservation, monotonicity, non-perturbation all hold, read via PERF_READ; both quick and full passed |
| B3 tiny | `archbench.run … --backend fpga-u280 --tier tiny --counters` | **27/27 pass**, duration 6.3 min, all 27 points include counters |
| B3 small | Same, `--tier small` | **25/25 pass**, duration 7.7 min. LLaMA prefill (S = 64) 155.6 M cycles = 4.15 s; decode (P = 128) 43.9 M cycles = 1.17 s |
| B4 | Board records compared launch-by-launch against `core-vrf-sync-v1` simulation | tiny 27/27 points, 132 launches: **cycles identical launch-by-launch**. (small tier lacks RTL reference as RTL simulation is too slow). |
| B5 | `archbench.report u280_tiny u280_small gpu_gb10 --tier tiny small` | 52/52 vs 52/52. TNCA U280: E = 0.0072, A = 0.021; GB10: E = 0.0051, A = 0.041. Second-scale time ratio T_GPU/T_TNCA between 2×10⁻⁵ and 6×10⁻³ |
| B6 | `python -m archbench.generate --tier small --new 8 …` | 64-token prefill plus 7-step decode, each step consuming device-generated KV cache. All 8 steps within tolerance, token sequence identical to pure FP32 generation. Device time: prefill 15.4 tokens/s, decode 1.01 tokens/s. Host wall clock 327 s, including 194 s inside launch (transfers and waiting), remainder in step-by-step compilation. |

Three bug fixes made and committed during board bring-up:
- **`link.sh` false rejection of `AUTO-FREQ-SCALING-08`**: see §4.
- **`program.sh` failed to read U280 thermals**: XRT 2.16 thermal report uses tabular format, whereas regex expected `FPGA : 45 C`. Both formats now supported.
- **Runtime heap allocator fragmentation**: Freed blocks were not coalesced; LLaMA small ran out of device memory midway ("out of device memory: 131072 bytes needs 131072 and 98304 remain"). After coalescing, heap returns to empty after each launch; tiny cycles identical to pre-fix.

Additionally, conformance tests previously built simulators for every backend, failing on board hosts lacking iverilog; it now builds only for backends requiring a simulator.

**Next steps** (detailed in [architecture_overview.md](architecture_overview.md) §13 and [compiler_overview.md](compiler_overview.md) §15):
1. Extract kernel-only resource utilization from U280 link report.
2. Measure board power (`xbutil examine -r electrical`).
3. Register board evidence into evidence registry (Stage 2 level).
4. Memory capacity roadmap (URAM vs HBM) pending decision.

## 6. Device Clock: 37.5 MHz Bottleneck and Timing Optimizations (2026-10-08 – 2026-10-09)

### 6.1 Where Was the Bottleneck

In the U280 routed link report (`hw_bb_locked_timing_summary_routed.rpt`), the clk_dev (26.668 ns) worst path slack was only 2.689 ns:
- **Endpoints**: `u_mtx/g_mac[12].u_mac/u_fma` `p1_mag_reg → p0_exp_common_reg`.
- **Scale**: 60 logic levels (19 CARRY8, 1 DSP), data path 23.931 ns, routing accounting for 73%.
- **Semantics**: FMA stage 2 (leading-one search, normalization, rounding) result fed via accumulator file write-to-read bypass into next multiply-add stage 0. In stage 0, it passed 24×24 multiplication (DSP) and zero-check before deciding shift direction and common exponent. All completed within one cycle.

### 6.2 Methodology: Minutes-Long OOC Synthesis Instead of 4-Hour Links

- **Tool**: `flow/synth/vivado_ooc.sh` (`make fpga-timing`). Out-of-context synthesis plus timing estimation on individual units or whole device (`tnca_fpga_top`). Unit-level takes 3–5 min, whole device ~25 min.
- **Note**: Reports pre-routing estimates. The matrix path was estimated at 14.7 ns post-synthesis and measured 23.9 ns post-routing (ratio ~1.6, applied below for conversions).
- **Yosys**: Mapping to UltraScale+ primitives was evaluated and unusable: does not infer CARRY8, and `ltp` does not treat FDRE as register boundaries. Yosys remains a structural proxy for `make synth`.

### 6.3 Modifications (All Cycle-Neutral)

| Modification | Files | Post-Synth Estimate |
|---|---|---|
| Removed accumulator file write-to-read bypass. Matrix engine reads row m, writes back FMA_LAT cycles later; same row next read is ROWS+1 cycles later. Row operations and `a.vread` start only after prior command writeback, so bypass never carried used values. `tnca_matrix` asserts this in simulation (`acc_collide`), failing immediately on violation. | `rtl/core/tnca_acc_rf.v`, `rtl/core/tnca_matrix.v` | Matrix unit 14.7 → 8.6 ns |
| FMA generated via XLS (see [`xls.md`](xls.md)). `xls/tnca_fma.x` avoids subnormal-flushing `apfloat`, matching hand-written algorithm step-by-step; pipeline scheduled by XLS. `LAT=2` instance matches hand-written stage count and latency. | `xls/tnca_fma.x`, `rtl/gen/tnca_fma_xls.v`, `rtl/core/tnca_fp32_fma.v` | Vector FPU 9.2 → 6.8 ns |
| Tensor engine window check originally multiplied five block dimensions in series and compared; now multiplies one word per cycle when `td.load` reads descriptor tail, precomputing box element count (saturating to 2^20); transfer only shifts and compares. | `rtl/core/tnca_tensor.v` | Whole-device worst path was 10.1 ns |

Verification:
- `unit`: XLS FMA 41,240 vectors per latency.
- `fma-deep`: 516,240 vectors per latency, 0 difference against exact Fraction oracle.
- `core`, `core-vrf`: 31/31 each, including tensor descriptors t12/t17/t33–t35.
- Other: `r38-xpose-rtl`, `top`, `e2e` all pass with identical cycles (e.g. t10_mesh 18,591).

### 6.4 Frequency

- **Whole-device post-synth**: With all three modifications, whole device (`tnca_fpga_top`, 13.333 ns target, U280) post-synthesis worst path was **10.19 ns, 40 levels**, WNS +3.08 ns, all 525,817 endpoints passing. Path runs from VRF bank BRAM read port through matrix unit A select / format widen into FMA stage 0. Former 60-level 14.7 ns matrix loop and 10.1 ns tensor window check removed from worst path. Resources: LUT 242,474 (18.6%), FF 181,133 (7.0%), BRAM 167.5.
- **New builds initially defaulted to /3**: `rtl/board/tnca_u50_xo.v` `DEV_DIV` parameter changed to default 3 (150 MHz / 3 = 50 MHz, 20 ns period). At 1.6× routing ratio, routed estimate was ~16.3 ns (~61 MHz), ~3.7 ns slack. (Changed to /2 on 2026-10-09, and /1 undivided 100 MHz, see §6.10, §6.13.)
- **Clock domain crossing unaffected**: Adapter uses toggle + synchronizers across domains, independent of divide ratio; divided clock edges align with ap_clk, so CDC constraint remains one ap_clk period.
- **Governed by link results**: `link.sh` fails on unmet timing rather than silently downscaling.
- **Loaded image at the time (uuid `083faace…`) remained /4, 37.5 MHz**: `archbench/platforms/tnca.py` `BOARD_CLOCK_HZ` describes loaded image, updated when a new image is loaded.

### 6.5 Moving Higher: 75 MHz (150 / 2, 13.3 ns)

At 1.6× routing ratio, 75 MHz requires post-synthesis <= ~8.3 ns. Remaining two paths:
- **Matrix unit, 10.2 ns (whole-device worst)**: VRF BRAM read, A select, format widen into FMA stage 0; matrix unit alone accumulates to 8.6 ns. Can register A operand before FMA (MAC scan only adds 1 drain cycle, accumulate loop unaffected), or use `LAT=3` XLS FMA (7.8 ns at unit level).
- **Vector FPU**: 5.8 ns with `LAT=3`.

`LAT=3` requires only changing core `FP_LAT`; FMA itself selects another generated instance. Cost is 1 cycle later floating-point result, requiring refreezing cycle baseline (`spec/r38/baseline_cycles.json`) and updating `fp_result_cycles` in target descriptor. This belongs to next steps and should be recorded as a design change in `spec/r38/design_changes/`.

### 6.6 Second Round: Toward 75 MHz (Night of 2026-10-08)

- **Matrix operand register (`timing-002`)**: Added register stage (`IN_REG`) between `tnca_fpu_multi` format widen and FMA; matrix unit `OPND_REG` enabled by default; result tag and drain extended by 1 cycle. Whole-device post-synth estimate (75 MHz target) dropped from 10.19 ns to **9.30 ns**, WNS +4.02 ns. The 2,672-endpoint VRF→MAC group dropped below 8.9 ns.
- **Cycle cost**: 1 extra cycle per `m.mma`. 14 of 149 R37 O2 cases changed (+1 to +4 cycles), all numerically correct.
- **Frozen baseline amendment**: Refreezing revealed r38-tdesc had changed 17 cases and profile config hash (added `tensor_store`), previously only amending 4 streamk cases because R2 log was truncated. All recorded in `spec/r38/baseline_cycles.json`.
- **Re-stage RTL before re-measuring cycles**: Running `tools/r38/baseline.py --run` directly does not re-stage `python/tnca/data/rtl`; `make r38-baseline-check` stages first.
- **Remaining worst group**: VALU single-cycle masked integer reduction (`x_pm_idx → w_sdata`, 9.30 ns, 37 levels). It sits within a single unit with small fanout, so routing ratio may be lower than matrix path's 1.6×.
- **75 MHz link**: First `DEV_DIV = 2` link launched 2026-10-08 23:58 HKT on acf3019 with this state.
- **Integer reduction and FPU operand select (`timing-003`)**: Reduction split into three trees (add, bitwise, min/max) selected only at end; min/max unified using order-preserving key transform (invert sign bit for signed, invert all for min, identity maps to key 0); vector FPU reduction operand select no longer waits on core issue `start`. Both cycle-neutral. Added `t36_reduce`: 8 integer reductions × 8 vector sets × 8 masks (including empty) + broadcast forms = 544 checks, passing on both old and new VALU. Whole-device post-synth dropped from 9.30 ns to **7.76 ns** (WNS +5.56 ns). Second 75 MHz link started 2026-10-09 00:45 HKT in parallel.

### 6.7 Requirements for 100 MHz and 150 MHz

Whole-device post-synthesis estimates at 150 MHz (6.667 ns) target (RTL after timing-002, before timing-003), grouped by start → end module:

| Delay (ns) | Endpoints | Group | Nature |
|---:|---:|---|---|
| 9.30 | 544 | VALU integer reduction → scalar writeback | timing-003 reduced to 7.76 ns; start became W-stage predicate bypass |
| 8.25 | 3,184 | Issue decode → vector FMA stage 0 | timing-003 removed `start` dependency |
| 7.47 | 467 | X stage → VRF bank control (stall chain) | 29 levels, 79% routing |
| 7.18 | 1,917 | Tensor engine → on-chip memory | Memory address and request |
| 7.00 | 1,062 | LSU → on-chip memory | Same |
| 6.64 | 4 | Core → VRF | |
| 6.35 | 1,488 | Tensor engine internal | Address accumulator, window |
| 6.17 | 1,136 | FMA stages within MAC | Reducible with `LAT=3` XLS instance |
| 6.06 | 112 | Vector FMA stages | Same |
| 5.74 | 9,659 | MAC → accumulator write | |

At 1.6× routing ratio:
- **100 MHz (10 ns)**: Requires post-synthesis <= ~6.25 ns. Demands addressing top 7 groups, including splitting reduction across two stages, and registering stall chains and memory requests.
- **150 MHz (6.67 ns)**: Requires post-synthesis <= ~4.2 ns. Every group above exceeds this threshold; FMA requires 4–5 pipeline stages (`FP_LAT` 4–5), with memory, VRF, and issue logic deeply re-pipelined.
(Note: this assessment was overly pessimistic; the third round in §6.8 reduced all non-FMA groups below 6.2 ns without moving a single cycle.)

### 6.8 Third Round: Cycle-Neutral Reduction to 6.84 ns (Early Morning 2026-10-09)

Evaluated at 150 MHz (6.667 ns) target; five modifications, none moving any cycles:

| Modification | Group Addressed | Before (ns) | After (ns) |
|---|---|---:|---:|
| timing-004: Integer min/max reduction changed to pairwise parallel comparison across all lanes, with one-hot winner selection | VALU reduction → scalar writeback | 7.76 | 6.71 |
| timing-005: FMA array control (operand form, negate, mul-only, reduction override) decoded in R stage, registered into X with opcode | Issue decode → vector FMA stage 0 | 7.17–7.42 | 6.83 |
| timing-006: LSU per-lane word address, byte enables, shifted write data computed in S_SETUP; planner only selects word and ORs | LSU → on-chip memory | 5.6–11.0 | < 5.9 |
| timing-007: VALU immediate select decoded in R; signed high-half correction formed in parallel with multiplier, leaving one subtraction | VALU multiply → vector writeback | 6.71 | 6.07 |
| timing-008: Tensor engine boundary checks evaluate 36-bit offset against registered constants; request fields steered by state rather than late valid | Tensor engine → on-chip memory | 7.20 | 5.73 |

Notes:
- **LSU 11.0 ns was not caused by RTL regression**: In the synthesis run incorporating timing-004, the LSU planner's conditional accumulation over 16 lanes synthesized into a 25-level ripple chain; identical LSU code measured 5.6 ns in other runs. timing-006 eliminated this synthesis sensitivity.
- **Equivalence**: timing-008 moves terms across inequalities assuming no operand approaches 2^63 (addresses/offsets are orders of magnitude smaller). Bounds depend only on t_CB, t_E, t_row_bytes set in S_TCHK, separated from transfer by at least one S_TMUL cycle. t33 (672 items), t34 (400 items), t35, and transpose paths all passed.
- **Verification**: core, core-vrf, top, nuca, mesh, r38-xpose-rtl under all three memory models passed; R2 cycle check passed 149/149 on two consecutive runs.

Additional improvements:
- **timing-009**: Vector FPU format converter inputs registered beside stage 0, entering FP_LAT delay line at stage 1 (previously 6.44 ns combinational logic, 71% routing). Cycle-neutral: t21_cvt unchanged at 1,942 cycles, R2 cycle check 149/149.
- **timing-010**: Banked VRF operand fetch gates only on vector source hazards rather than full `stall_other` (previously 6.17 ns, 77% routing). Older in-flight writers are tracked in the scoreboard, so fetch safely overlaps other stalls. Cycle-neutral on core-sim; 5 core programs faster by 1–14 cycles on banked VRF. (Requires regenerated B4 simulation reference for images with this change).

With timing-009 and timing-010, neither VRF nor converter appears above 5.4 ns. Whole-device post-synthesis reached **6.84 ns** (WNS −0.19 ns at 150 MHz, only 80 failing endpoints out of 538,791). Paths above 6 ns concentrated in/around generated FMA: vector FMA stage 0 (6.84 ns), MAC FMA stages (6.17 ns), FMA writeback (6.15 ns), FMA internal (6.06 ns).

### 6.9 Undivided Attempt: 100 MHz (2026-10-09)

- **Configuration**: Using 5baba1e (third round) RTL, `DEV_DIV = 1`, kernel clock 100 MHz: `package_xo.tcl` `AP_CLK_HZ`, `ooc.xdc` period, and `link.sh` `EXPECT_MHZ` all set to 100 in build tree `~/tnca_srcND` on acf3019. Began linking 2026-10-09 02:57 HKT in parallel with two 75 MHz links (`VPP_JOBS=4`).
- **Safety of undivided clock**: `tnca_axi_lite` crosses domains via toggle and synchronizers; each clk_dev edge is an ap_clk edge, preserving CDC requirements.
- **Fallback behavior**: If 100 MHz does not close, v++ auto-scales frequency and writes `.xclbin`; `link.sh` flags this as failure because it differs from requested design. With `DEV_DIV = 1`, device clock equals kernel clock, so resulting image remains usable at downscaled frequency if `BOARD_CLOCK_HZ` matches.
- **Routed WNS in progress**: Both 75 MHz links showed −0.05 ns post-placement and +0.015 ns initial routing estimate (platform path).
- **Negative slack during routing iterations**: Rip-up-and-reroute produced negative WNS (first link −1.087 to −0.757 ns across three iterations; second link −1.124 ns initially).
- **Backup /2 link**: At 04:50 HKT, cancelled first link (timing-002), launched `u280_75c` (commit d72731d, third round + timing-009, 6.84 ns post-synth, ~0.9 ns faster than second link): build dir `~/tnca_build/u280_75c`, source `~/tnca_srcR3`.

### 6.10 75 MHz Verified on U280 (2026-10-09)

- **Image**: Second /2 link (timing-003 RTL, commit `f307eef`) completed at 05:41 HKT, `LINK_EXIT=0`, reporting only AUTO-FREQ-SCALING-08 (timing closed with slack, requested frequencies kept). `.xclbin` reads back DATA_CLK = 150 MHz, sha256 `16813fa9075f9192f9e3b19f97d9a68185fc13dca395885e8efcaa8c98cc3cee`.
- **Routing results**:
  - Full-design WNS +0.016 ns (platform path).
  - **clk_dev WNS +0.471 ns** (13.333 ns period), 498,254 endpoints, 0 failing.
  - Worst path: `x_pm_idx → w_vdata` (12.68 ns, 28 levels, 80% routing; ~7.8 ns post-synth, matching 1.6× ratio).
- **On-board testing**: Pre-load gates passed (device idle, no other loads, FPGA 31 °C, HBM 26 °C).
  - B1 conformance PASS: oracle, second launch in same session, fault and recovery.
  - B2 profiler PASS.
  - B3 ArchBench tiny 27/27 pass.
  - B4: 27/27 points matched RTL simulation (`fpga-u50-sim`) **identical launch-by-launch** (from op.add at 1,776 cycles to llama_prefill at 5,323,042 cycles) via `tools/r38/board_vs_sim.py`.
- **Clock measurement**:
  - Management frequency counter read 150 MHz kernel clock.
  - `flow/fpga/measure_clock.py` timed two spin kernels on host: **74.965 MHz and 75.017 MHz** (loop ran exactly 6 cycles/iteration both times).
- **Default updated**: `DEV_DIV` defaulted to 2, ArchBench `fpga-u280` clock set to 75e6. `link.sh` reads `DEV_DIV` from wrapper rather than hardcoding /4.
- **Records**: Located in `evidence/r38/u280_75mhz/`.
- Subsequent RTL (third round + timing-009) was 6.84 ns post-synth (~0.9 ns faster, ~2.4 ns slack at /2).

### 6.11 Cycle Cost of Deepening FMA: `FP_LAT = 3`

After timing-010, paths above 6 ns post-synthesis were in the generated FMA:

| Group | Delay (ns) |
|---|---:|
| Vector FMA stage 0 | 6.84 |
| MAC FMA stage | 6.17 |
| FMA output to writeback | 6.13 |
| FMA internal | 6.06 |

The three FMA stages are balanced (~6.1–6.8 ns); reducing further requires deeper pipelining. `FP_LAT` determines both vector FPU and matrix unit FMA latency (`FMA_LAT(FP_LAT)`). XLS 4-stage generated instance (`tnca_fma_l3_xls`) already exists. Tested `FP_LAT = 3` in a temporary branch:
- **Correctness**: All core programs pass.
- **R2 149 cases**: 59 slower (+0.91% total, median +1.8%, max +9.0%: moe_topk_gating; softmax/n37 +7.7%; linear_self_attention/out +7.3%), 90 unchanged, 0 faster. Cost concentrated in small kernels with long FP dependency chains.
- **Post-synthesis timing at 150 MHz target**: **WNS +0.274 ns, 543,528 endpoints, 0 failing** (meeting 150 MHz post-synth). Worst path was 6.22 ns in tensor engine S_TMUL descriptor multiply, followed by core → core (6.09 ns) and vector FPU internal (6.00 ns). FMA groups dropped below 5.8 ns (~10% faster post-synth vs. `FP_LAT = 2` 6.84 ns, trading 1–2% cycles).
- **Costs**: 59 baseline cases require refreezing, config hash updates, B4 reference regeneration.

At 1.6× routing ratio, 6.22 ns yields ~10 ns routed (~100 MHz undivided). Launched trial link `u280_fl3` at 06:51 HKT (`FP_LAT = 3`, `DEV_DIV = 1`, 125 MHz request; build `~/tnca_build/u280_fl3`, source `~/tnca_srcFL3`). Repository default kept at `FP_LAT = 2`.

### 6.12 100 MHz Undivided: Routing Congestion Bottleneck Solved by LUT-RAM Accumulator (timing-011)

- **Symptom**: 100 MHz undivided link (`u280_nd100`, FP_LAT = 2) rip-up-and-reroute began with 513,246 overlap nodes, still had 2,828 after ~2 hours (vs. 75 MHz clearing 357k in under 1 hour).
- **Cause**: Kernel pinned to SLR1 (CLB utilization 97.71%, LUT 70.49%; critical paths 70–80% routing).
- **Largest block**: Matrix unit had 84,354 LUTs and 139,751 FFs; accumulator file alone accounted for 41,869 LUTs and 131,072 FFs due to a whole-array reset loop preventing RAM inference (synthesized as flip-flops with 256:1 read muxes).
- **Fix**: Distributed RAM per lane (256×32) without reset; valid bit per (row, lane) handles reset and clearing (cleared by reset and `z_en`, set on write, returns 0 if unwritten). Functionally identical to original.
- **Outcome**: Matrix unit alone dropped to **49,516 LUTs (2,560 as RAM), 12,758 FFs**, saving ~35k LUTs and 127k FFs in SLR1. Unit WNS +0.573 ns at 150 MHz. Cycle-neutral: lint, unit, core, core-vrf, core-slowmem, top, nuca (122 tests) passed; t11_matrix and t25_mmasf cycles unchanged.
- Launched `u280_nd100c` at 08:19 HKT with timing-011 (replacing `u280_nd100b`).
- **Scalar register file similarly transformed (`timing-012`)**: 128×64 bits, 7 async read ports, previously 8,192 FFs and seven 128:1 muxes (24,669 LUTs) due to reset loop; distributed RAM replicated per read port with valid bit; dropped to **2,115 LUTs (1,036 as RAM), 128 FFs**.
- Total savings across both RFs: ~57k LUTs (~24% of core) and 135k FFs in SLR1.
- Regression passed: 155 tests passed (including core-ooo, nuca).
- Launched `u280_nd100d` at 08:57 HKT (timing-011 + timing-012); terminated original congested `u280_nd100`.

### 6.13 100 MHz Undivided Verified on U280 (2026-10-09)

- **Image**: `u280_nd100c`: RTL `8d8e4e9` (timing-001…011, `FP_LAT = 2`), `DEV_DIV = 1`, kernel clock 100 MHz. Link 08:19–12:05 HKT, `LINK_EXIT=0`, AUTO-FREQ-SCALING-08, DATA_CLK = 100 MHz.
  - sha256: `09cbd3d4992a07330b6f0567cb7f091682311adc3bf212479b4695d1a98323fb`.
  - xclbin UUID: `a446256a-e3be-6912-0701-117f82c9a70a`.
- **Routing results**:
  - SLR1 CLB utilization 85.41% (down from 97.71%). Overlap nodes resolved in ~20 min; converged to +0.016 after three iterations (from −0.541 ns).
  - Full-design WNS +0.016 ns (platform path), 566,734 endpoints, 0 failing.
  - Device domain (kernel clock domain) **WNS +0.127 ns** (10 ns period, 149,372 endpoints, 0 failing).
  - Worst path: LSU `pending → u_mem` BRAM data input (9.469 ns, 14 levels, 87% routing).
- **On-board testing**: Passed all pre-load gates (FPGA 31 °C, HBM 26 °C).
  - B1 conformance PASS.
  - B2 profiler PASS.
  - B3 ArchBench tiny 27/27 pass.
  - B4: 27/27 points matched simulation (`fpga-u50-sim` on commit `1eb6b94`, with timing-010 reference) **point-by-point identical cycles** via `tools/r38/board_vs_sim.py`.
- **Clock measurement**:
  - Sysfs frequency counter reads 100 MHz.
  - `flow/fpga/measure_clock.py` measured **100.067 MHz and 100.009 MHz** (`clock.txt`).
- **Default configuration**:
  - Flow's four clock settings set to 100 MHz undivided (`package_xo.tcl`, `ooc.xdc`, `link.sh`, and `DEV_DIV = 1` in `rtl/board/tnca_u50_xo.v`).
  - ArchBench `fpga-u280` clock set to 100e6 in `archbench/platforms/tnca.py`, matching current loaded image.
  - /2 (150 MHz kernel clock, 75 MHz device) retained as fallback.
- **Records**: Located in `evidence/r38/u280_100mhz/`.
- **Still running** (at time of record):
  - `u280_nd100d`: adding timing-012 (scalar RF), SLR1 CLBs to 76.6%, to check additional margin.
  - `u280_fl3`: `FP_LAT = 3`, 125 MHz request; RTL without timing-011/012.
