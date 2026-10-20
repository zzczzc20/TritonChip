# R40 report: native PyTorch device, three physical cores, NUCA and HBM on the U280

Contract: `doc/r40_tasks_and_acceptance_v3.md` (v3, the one-time relaxation of
v2 for delivery on the loaded 90 MHz image; it inherits v2 and v1). Sign-off
name `R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM_V3`. Machine judgement:
`make r40-v3-acceptance`.

> **Status: closed with failures shown (user decision, 2026-10-20).**
> `R40-10.acceptance` is **not passed** (recorded by `tools/r40/v3_r40_10_gate.py`
> against PR #10, non-Draft, base `main`). Under v3, five performance gates show
> no significant gain and `R40-01.device-time` fails its per-point rule.
> Every gate is judged on its evidence and nothing was relaxed after the
> contract freeze. Each failure has a bottleneck attribution (section 4) and
> becomes an R41 target (section 8).

## 1. What was delivered

| | |
|---|---|
| Board | Alveo U280 (`xcu280-fsvh2892-2L-e`), XRT 2.16, Vitis/Vivado 2023.2 |
| Image | `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` (`tnca_u280.xclbin`, UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`) |
| Cores | 3 physical TNCA cores (core-vrf-sync-v5), one per SLR (SLR0/1/2), 3 memory outlets, TMU per SLR |
| Memory | 4 GiB exposed HBM window over PC0..15; PC16..31 not exposed and not certified |
| Clock | DATA_CLK measured at 90 MHz on the loaded card (MMIO counter samples) |
| Software | product tree `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75` (`doc/r40_v3/identity_amendment.json`) |

### 1.1 Frequency exception (authorized)

The first-round target was 100 MHz. The routed 100 MHz implementation fails
setup: WNS −1.002 ns, TNS −6679 ns. The worst cone runs from the
memory-active-port control to the LSU pending logic: 14 logic levels, 11.18 ns
of data delay, 10.31 ns of it routing. Vitis placed DATA_CLK at 90 MHz, and at
90 MHz the same routed design meets every constraint:

| WNS | TNS | WHS | THS | WPWS | Failing endpoints | Route errors | Routed nets |
|---|---|---|---|---|---|---|---|
| +0.016 ns | 0 | 0.000 ns | 0 | 0.000 ns | 0 | 0 | 1,160,009 |

The user authorized the ~10 MHz reduction (2026-10-18). R40 makes no 100 MHz
claim.

### 1.2 The last implementation was not run (deferred to R41)

R40 had one full synthesis/place/route budget left. Before spending it, the
candidate fixes were built and verified on branch `r40-final-image`:
- M1: registered host grant;
- M2: registered `host_allow`;
- M3: registered local memory controls with an MMIO settle interlock;
- M4: vector-FPU float-to-int per ISA §8.4.

A read-only census of attempt 3's routed checkpoint
(`r40-final-image:evidence/r40/v3/final_image_candidate/census/`) then showed
that M1–M3 remove the known cross-SLR cone but leave **12,799 failing
endpoints**: WNS −0.919 ns, TNS −4,111 ns, in 753 path families inside the
tiles. With M1–M4 alone the link would land at about 91 MHz, and even 95 MHz
needs about ten core-pipeline restructurings.

The launch rule of the plan was not met. The user chose to defer the link to
R41 (2026-10-20), where it is built together with the frequency work. No
image other than a88ebdf5 was loaded in R40.

### 1.3 Resources (placed, per SLR)

| Resource | SLR0 | SLR1 | SLR2 | SLR0 % | SLR1 % | SLR2 % |
|---|---|---|---|---|---|---|
| CLB LUTs | 322,934 | 325,056 | 281,718 | 73.45 | 75.24 | 65.21 |
| CLB registers | 159,717 | 144,350 | 105,429 | 18.16 | 16.71 | 12.20 |
| Block RAM tiles | 139 | 153.5 | 121.5 | 20.68 | 22.84 | 18.08 |
| URAM | 0 | 0 | 0 | 0 | 0 | 0 |
| DSPs | 171 | 189 | 175 | 5.94 | 6.15 | 5.70 |

SLR crossings: SLR1↔SLR0 14,224 of 23,040 (61.7 %); SLR2↔SLR1 8,071 (35.0 %).
CDC: the 56 Critical findings in the vendor platform structures were reviewed
by exact endpoint; no TNCA endpoint appears in the CDC crossing report.
Raw reports: `evidence/r40/v2/convergence/attempt03/qualified90_raw.tar.gz`,
`slr_util_placed_20261018.rpt.gz`, `operating_timing_90mhz_pass.json`.

## 2. Contract changes (v3) and amendments

See `doc/r40_tasks_and_acceptance_v3.md` for the text. In short:

1. All gates are judged on the image above at 90 MHz.
2. **Pass rule for performance gates.** A gate passes on a significant gain:
   the 95 % paired bootstrap lower bound of S_geo is above 1 on the complete
   frozen subset, and every point is correct. The v2 threshold is reported as
   a target, met or missed.
3. **Board device-cycle sampling.** A cold first call (never pooled), then one
   qualification sample, then three formal samples where the call takes at
   most 1 s. Device cycles reproduce within 0.2 % across runs.
4. **Moved out of R40.**
   - `R40-07.matrix` needs new MMA RTL, so it goes with the next frequency
     implementation.
   - `R40-06.locality` (the DCO/locality software path), `R40-09.autotune` and
     `R40-09.holdout` (TNCA_TUNE) go to R41 by the user's decision.

Amendments made after the freeze, each authorized by the user:

- **Identity** (`doc/r40_v3/identity_amendment.json`): the product changed
  twice, 80dd → bc5a → fe1b, only in `python/tnca/inductor.py` and
  `python/tnca/triton_backend/compiler.py`. Every board record was
  re-collected on fe1b.
  - bc5a: scan/sort kernels keep 16 lanes under wide block caps.
  - fe1b:
    - mix-order split reductions are off;
    - a kernel larger than the 128 KiB code store is refused at compile time,
      with a 16-lane fallback;
    - argmax/argmin keep 16 lanes.
- **Design corrections** (`doc/r40_v3/design_corrections.json`):
  - `R40-06.ports` is measured on B40-N (three cores). The frozen B40-S has
    one requester with one request in flight, which cannot use a second
    outlet.
  - `R40-07.pipeline` holds `TNCA_RCTX_ROLES=2` on both sides. With roles
    unset, the pipeline knob compiles to byte-identical executables.

## 3. Gate results

From `make r40-v3-acceptance` on the delivery identity. The capability
auditors' command lines and evidence directories are in
`evidence/r40/v3/capability_fe1b.md`. The gate records themselves are copied
to `evidence/r40/v3/gates/`.

| Gate | Kind | Status | Evidence |
|---|---|---|---|
| R40-00.protocol | capability | pass | 42/42 cases |
| R40-00.b39-reproducible | capability | pass | B39 rerun 77/77, cycle-identical |
| R40-00.frozen | capability | pass | 33/33 as frozen |
| R40-01.device | capability | pass | 79/79: native77 resident inventory on fe1b, image and clock bound |
| R40-01.mixed-device | capability | pass | 32/32; no-CUDA host (board) 16/16 and CUDA host (GB10) 16/16 incl. mixed CUDA+tnca graphs; two earlier GB10 attempts hit CUDA out-of-memory at context creation and are preserved |
| R40-01.device-time | capability (non-regression) | **fail** | S_geo 1.0015 [0.9947, 1.0080]; 4 points > 5 % slower |
| R40-02.zero-transfer | capability | pass | 4/4, fe1b board queue |
| R40-02.resident | perf | pass | S_geo 2.641 [2.365, 2.948] |
| R40-02.batched | perf | pass | S_geo 1.101 [1.009, 1.199] |
| R40-02.stress | capability | pass | fe1b RTL simulation chain incl. replay-1000 |
| R40-03.tiles | capability | pass | 66/66 |
| R40-03.wide-blocks | perf | **no significant gain** | S_geo 1.157 [0.970, 1.266] |
| R40-03.gemm-tile | perf | **no significant gain** | S_geo 0.303 [0.202, 0.444] |
| R40-03.contract | capability | pass | 14/14 |
| R40-04.hbm | capability | pass | 25/25; peak 3,098,945,024 B in the 4 GiB window, headroom 1.39 |
| R40-04.bandwidth | perf | pass | memory-service S_geo 1.229 [1.038, 1.665] |
| R40-04.robustness | capability | pass | 30/30; HBM model fidelity declared uncalibrated (sec 3.4) |
| R40-05.multicore | capability | pass | 94/94 (physical multicore audit on the FPGA host; cross-checked 13/13) |
| R40-05.scaling | perf | pass | three cores 2.392 [2.184, 2.602]; two cores 1.805 [1.689, 1.907] |
| R40-05.dispatch | capability | pass | 31/31 |
| R40-06.nuca | capability | pass | 27/27 |
| R40-06.streaming-scaling | perf | **no significant gain** | streaming-cores 2.559 [2.153, 2.854] passes; ports 1.007 [0.998, 1.025] does not |
| R40-06.locality | perf | deferred (R41) | |
| R40-06.semantics | capability | pass | 21/21 |
| R40-07.pipeline | capability | pass | 47/47 |
| R40-07.long-k | perf | **no significant gain** | S_geo 1.023 [0.955, 1.092] |
| R40-07.matrix | perf | deferred (new MMA RTL) | |
| R40-07.policy | capability | pass | 9/9; the 9 points the frozen policy changes, board A/B at 90 MHz, none > 5 % slower |
| R40-08.attention | capability | pass | 7/7 |
| R40-08.prefill | perf | **no significant gain** | S_geo 0.990 [0.964, 1.026] |
| R40-08.decode | capability | pass | 5/5, fixed-bucket model on the board |
| R40-09.autotune | capability | deferred (R41) | |
| R40-09.holdout | perf | deferred (R41) | |
| R40-09.suites | capability | pass | 9/9: native77, HBM14 and large24 all pass on fe1b |
| R40-09.model | perf | pass | S_geo 2.307 [2.299, 2.315] |
| R40-10.acceptance | capability | **not passed** | this report; the failures above |

## 4. Performance against the v2 targets

Campaign: `evidence/r40/v3/campaign_fe1b/`. 635 runs, 0 failures, on fe1b,
a88ebdf5 and 90 MHz, judged by `tools/r40/v3_campaign.py judge --evidence`.

| Experiment | S_geo | 95 % CI | Verdict | v2 target |
|---|---:|---|---|---|
| R40-02.resident | 2.641 | [2.365, 2.948] | pass | 1.5, met |
| R40-02.batched | 1.101 | [1.009, 1.199] | pass | 1.2, missed |
| R40-03.wide-blocks (64) | 1.157 | [0.970, 1.266] | no significant gain | 1.3, missed |
| R40-03.gemm-tile (16,32,16) | 0.303 | [0.202, 0.444] | no significant gain | 1.15, missed |
| R40-04.memory-service | 1.229 | [1.038, 1.665] | pass | 1.5, missed |
| R40-05.three-cores | 2.392 | [2.184, 2.602] | pass | 2.1, met |
| R40-05.two-cores | 1.805 | [1.689, 1.907] | pass | 1.5, met |
| R40-06.streaming-cores | 2.559 | [2.153, 2.854] | pass | 1.8, met |
| R40-06.ports (B40-N) | 1.007 | [0.998, 1.025] | no significant gain | 1.3, missed |
| R40-07.pipeline (roles 2) | 1.023 | [0.955, 1.092] | no significant gain | 1.2, missed |
| R40-08.prefill (fused SDPA) | 0.990 | [0.964, 1.026] | no significant gain | 1.5, missed |
| R40-09.model | 2.307 | [2.299, 2.315] | pass | 1.5, met |
| R40-01.device-time (native vs CPU path) | 1.0015 | [0.9947, 1.0080] | fail (per-point rule) | non-regression |

### 4.1 Bottlenecks behind each failure and each missed target

Sources:
- the board's per-launch records (`evidence/r40/v3/campaign_fe1b/judge/`);
- the same configuration pairs run through the SystemC/TLM model
  (`evidence/r40/v3/model_attribution/`).

The model is uncalibrated, so it is used only where it reproduces the board's
direction, and the evidence says where it does not.

| Item | What limits it | Evidence |
|---|---|---|
| **R40-03.gemm-tile** 0.30x | Code generation. A 16x32 accumulator is lowered with row-wise accumulator moves, not as two 16x16 MMA fragments: 12x the instructions, 48x the matrix-engine commands, 81x `a.vread`. Useful MMA work grows only 36 %. | The model reproduces the board within 7 % (linear 5.9x vs 6.1x, matmul 3.5x vs 3.8x); `R40-03.gemm-tile/model_diagnosis/` |
| **R40-03.wide-blocks** 1.16x, CI includes 1 | Streaming kernels gain 1.5–1.6x. rope's kernels are 26 % faster on the device, but its makespan is 11x its device cycles: under B40-S per-kernel staging, host gaps between launches dominate, and that is where its 0.94x comes from. layernorm (one launch) is really 4 % slower on the device, in the LSU/memory path at one outstanding line; the model does not reproduce this. | Per-launch board records; model add 1.54x vs board 1.58x |
| **R40-06.ports** 1.01x (B40-N) | The outlets are not the bottleneck. One outlet is busy about 12 % of `op.copy/hbm`. The NUCA home-slice engine and the per-core memory path limit streaming, so more cores scale (2.56x) and more outlets do not. | Model 1.00x vs board 1.00x; outlet and home-engine counters |
| **R40-07.long-k** 1.02x (roles 2) | Pipelining trades barrier-sleep cycles against transfer-engine admission waits. The engine takes one command at a time, so the producer's prefetch serialises. Memory-bound gemv wins (+16–18 %); compute-bound fp16 matmul loses 14–15 %. Only a per-point policy can keep the wins. | Board per-launch; the model shows the trade-off (−35 % barrier sleep, +109 % admission wait) but underestimates the matmul loss |
| **R40-08.prefill** 0.99x | The fused SDPA kernel takes 1.8–1.9x the device cycles of the decomposed attention kernels for the same attention (5.87 G vs 3.12 G on `op.attention_prefill/large`). Its only gain is fewer launches and host gaps, which is why it breaks even. It is register-limited: 64 vector registers per role context, no spill, and it compiles only with packed-window unpack and window reuse. | Board per-launch records; compile preflight |
| **R40-01.device-time**, 4 points > 5 % slower | softmax: on the tnca device, Inductor emits `tl.debug_barrier()` between a store and its reload, lowered to `fence.acq_rel.device`, one more fence per loop than on the CPU path. gemv/silu tiny run byte-identical code; the difference comes from staged-buffer placement (cache set / HBM PC). | `R40-01.device-time/regression_diagnosis.md` |
| R40-02.batched 1.10x (target 1.2 missed) | Graphs of 3–13 launches gain 1.2–1.7x in makespan. The 40–42-launch Llama tiny graphs lose 8–11 %; the cause in the batched path for long graphs is not isolated. | Per-point board records |
| R40-04.memory-service 1.23x (target 1.5 missed) | Outstanding 1→8 helps transpose 2.6x and softmax 1.12x. Streaming ops gain only 1.03–1.06x, limited by the same home-slice engine as R40-06.ports. | Per-point board records |

## 5. Problems found and fixed during delivery

| Problem | Effect | Fix |
|---|---|---|
| Large-model weights overlapped the 4 KiB TMU MMIO page at `0x7fff0000` | original large Llama logits out of tolerance (0.114) | allocator excludes the reserved page (1a7c643); error 0.0027 |
| Probe padding crossed the reserved page | HBM window probe assertion | fillers split at the page (0958db2) |
| `native_model.py` asserted on `warm` after deleting it | final-identity model stage stopped before step 0 | assert moved to the warm path (8f8126e) |
| Wide register blocks have no `tt.scan` lowering | `TNCA_MAX_BLOCK` > 16 refused `op.cumsum` | scan/sort kernels keep 16 lanes (bc5a); all other artifacts byte-identical |
| Mix-order split reductions failed for the tnca device | `x.sum(0)`, `x.t().sum(-1)` did not compile | `mix_order_reduction` off on the TNCA path (fe1b) |
| A kernel larger than the 128 KiB code store compiled, then failed at load | GEMM tile 16,64,16 failed at run time on `model.mlp/medium` | refused at compile time as OutOfResources; wide configurations carry a 16-lane fallback (fe1b) |
| argmax/argmin under wide caps | `op.argmax/medium` 0.61x at `TNCA_MAX_BLOCK=64` | index-carrying reductions keep 16 lanes (fe1b); now 1.00x |
| Vector-FPU float-to-int conversions disagree with ISA §8.4 for \|x\| ≥ 2^23 and some u32 negatives (found by the SystemC oracle; no frozen suite uses them) | none on the frozen suites | M4 on branch `r40-final-image`, deferred with the image to R41 |

## 6. GPU comparison

`evidence/r40/v3/gpu_comparison/` (R40-10 item 4). Both sides run the same
Torch points at the same precision:
- **TNCA:** fe1b board records on a88ebdf5 at 90 MHz.
- **GPU:** NVIDIA GB10 with `inductor-cudagraph` and `inductor-triton`
  (ArchBench), plus `inductor-triton-bigsm` (supplementary).

Geometric means of TNCA time ÷ GPU time over the paired points:

| Suite | Points | Device makespan vs CUDA graph | Resident call vs resident call |
|---|---:|---:|---:|
| native77 (tiny/small/medium) | 77 | 974x | 112x |
| HBM14 | 14 | 6,922x | 4,136x |
| large24 | 24 | 10,365x | 3,454x |

The two systems differ in scope:
- TNCA is a 3-core FPGA prototype at 90 MHz;
- the GB10 is a 48-SM GPU at up to 3 GHz.

No normalisation by area, clock or power is made, and no ASIC is
extrapolated. GB10's `inductor-triton` has no GEMM choice below 68 SMs, so 41
of its GEMM points fail in each harness. Those failures are kept in the
report.

## 7. Reproduction

**PASS** (`evidence/r40/v3/reproduction/`). The run, 2026-10-20 00:46–00:49
UTC on the FPGA host:
- **Source:** `tools/r40/v3_reproduce.sh` taken from a clean `git archive` of
  commit f8a30e054b50 (archive sha256 `e47002be…`), not from any working tree.
- **Caches:** fresh native-extension, Triton and Inductor caches.
- **Identity:** the unpacked tree is the delivery identity
  `files:fe1b5199…`.
- **Checks:** the immutable contract passes, and the protocol tests pass
  42/42.

The three frozen points then ran on the loaded image a88ebdf5 at 90 MHz.
Their device cycles are compared with the fe1b functional re-collection:

| Point | Status | Device cycles | Reference | Difference |
|---|---|---:|---:|---:|
| `op.add/hbm/fp16-n67108864` | pass | 243,988,610 | 243,956,222 | 0.013 % |
| `op.matmul/medium/fp16-m128-k192-n128` | pass | 234,635 | 234,859 | 0.095 % |
| `model.llama_decode/hbm/fp16-b1-p512` | pass | 170,540,745 | 170,630,629 | 0.053 % |

All three are within the 0.2 % determinism bound, with no host fallbacks.

## 8. Carried into R41

Already moved by contract: R40-06.locality (DCO), R40-09.autotune and
R40-09.holdout (TNCA_TUNE), and R40-07.matrix (MMA RTL).

From the failures above:
1. **GEMM tiles.** Lower non-16x16 logical tiles as independent 16x16
   fragments with no row-wise accumulator traffic, then re-run
   R40-03.gemm-tile.
2. **Memory system.** The NUCA home-slice engine, not the outlets, limits
   streaming. It is the target for multi-outlet scaling (R40-06.ports) and for
   memory-service streaming gains.
3. **Transfer engine and pipelining.** Accept more than one command in the
   transfer engine, or choose pipelining per point (TNCA_TUNE /
   R40-07.policy). Without that, long-K pipelining helps memory-bound kernels
   and hurts compute-bound ones.
4. **Fused SDPA.** Under the 64-register budget, the fused kernel must be
   faster per attention than the decomposed kernels, not just save launches.
5. **Native vs CPU path.** Lower Inductor's debug barrier to the ordering
   TNCA actually needs (softmax). Measure buffer-placement sensitivity on tiny
   kernels.
6. **Wide blocks.** Measure the multi-launch points under resident transfer
   (or report device cycles beside makespan), and look into the layernorm LSU
   path.
7. **Hardware.** M1–M4 and the census frontier on branch `r40-final-image`
   (section 1.2) are the starting point of the next image. That image is
   built together with the frequency work and the MMA change.
8. **Performance model.** The SystemC/TLM model is uncalibrated at M2 (holdout
   median 21.5 %). M3, now in progress, adds a switchable cycle-level DRAM
   path (Ramulator2) beside the fast latency path. It targets a holdout median
   of at most 10 %.
