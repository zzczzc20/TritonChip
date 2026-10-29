# R41: `m.mma.dp.async` in the compiler, the toolchain and the SystemC/TLM model

Status: milestone 1 (2026-10-29). Domain of every result below: **SystemC/TLM model** (uncalibrated) or
**CPU/exact-rational oracle**, as labelled. No RTL image decodes the opcode yet (R41 G2 integrates the lane
into `tnca_matrix`, interface in `INTERFACE.md`), and no result here is a board result or closes a board item.
The author's approval of ISA sec 9C.4 (2026-10-29) is recorded in `doc/isa/TNCA-ISA-v0.1.md` sec 9C.4.

## What changed

| Area | Change |
|---|---|
| ISA toolchain | `m.mma.dp.async` (major `0x40`, func `0x02`) in `tools/asm/tnca_isa.py` (suffix class `mmadp`: `.k16/.k32/.k64`, A format then B format, one format names both, `.sf`; `mod[3:2]` A, `mod[6:5]` B, `mod[9:7]` reserved), assembler, disassembler, `rtl/common/tnca_defs.vh` (`TNCA_M_MMA_DP`, `TNCA_CAP_MATRIX_DP = 12`), regenerated schema / opcode map / TableGen input / support matrix, `make compiler` checks. `m.mma.async` is untouched. |
| Target | capability `matrix_dp` (CAPABILITY0 bit 12), emitted into a target file only where true, so every older target file is byte-identical. New profile `core-vrf-sync-v5-x8-g2c4-dp` (= `-g2c4` plus `HAS_MATRIX_DP=1`); it is a compile and model profile only, with no RTL elaboration or bench until G2 binds one. |
| Compiler | `CodeGen._op_mma_async` emits `m.mma.dp.async` when, and only when, the target has `matrix_dp`; the operand format and K step are those of the dot. The group size is `footprints.mma_dp_group(fa, fb)`: 4 for E4M3 x E4M3, 2 otherwise. Nothing reads a name, a shape or a model. |
| SystemC/TLM | opcode in `tools/performance_harness/systemc/src/chip.cpp`; exact group arithmetic in `include/tnca_matrix_dp.hpp` (1024-bit sign-magnitude, RO_48, RNE binary32, specials and signed zeros of sec 9C.4); profile keys `mtx_dp` (default 0; also sets CAP0 bit 12), `mtx_dp_extra_latency` (default 1), `mtx_dp_prefetch` (default 0); config `configs/r41-mma-dp.cfg`. |
| References | `tools/performance_harness/fp8_block_reference.py`: `dot_mode_for_profile`, `fp8_dot_dp` (vectorised exact E4M3 x E4M3), `dp_chain` (every pair, exact rationals through `tools/model/fdp.py`). `fp8_cases.py` and `fp8_kernels_cases.py` choose the reference from the profile's capability (`--reference-dot-mode` exists only for negative controls). |

## The lowering, and what it does not change

The ISA keeps the fragment layout of sec 9C.2 for both opcodes ("the fragment layout and the completion events are those
of `m.mma.async`"). Consequently the windows, the transfers, the footprints, the barriers, the source-free event and the
K stepping of a dot are the ones of `m.mma.async`, and the lowering is the mnemonic. The 32-bit-per-lane packing of the
B operand is done inside the matrix unit from the unchanged flat layout (`INTERFACE.md`); the compiler does not repack.
`compiler/test/test_mma_dp.py` checks this directly: for E4M3, E5M2, FP16 and BF16 at every K step the baseline
compiles, the assembly of the capable target equals the baseline's with `m.mma.dp.async` read as `m.mma.async`, and it
assembles; a shape the baseline cannot compile (a resource limit) fails identically with the capability.

* **K grouping and tails.** Groups are consecutive k aligned at multiples of G from k = 0 of each command. Every K step
  (16, 32, 64) is a multiple of 4, and a longer K is a chain of commands into one accumulator, so a group never straddles
  two commands. A K that is not a multiple of the K step is zero-padded to a whole fragment by the transfer engine
  (padding mode 0, the existing tail rule of `gemm.plan_gemm`); a padded group contributes `RNE_f32(C + 0) = C`, and a
  group with some real and some padded elements equals the group over the real elements. The reference zero-pads the
  same way. The cases with K tails in the table below (K = 40, 72, 144, 200, 328, ...) are the check.
* **Accumulator and rounding.** The accumulator is binary32 and `acc += tl.dot(a, b)` is still accumulated in place
  (`_fuse_dot_add`), so the chain across K blocks is `C <- RNE_f32(C + RO_48(group sum))` in increasing group order.
* **Mixed formats.** Triton converts mixed `tl.dot` operands to a common type before the compiler sees them. The
  existing E4M3 x E5M2 path (exact FP16 conversion of both operands) is unchanged: with `matrix_dp` it is an FP16 x FP16
  `m.mma.dp.async`, group 2, and that equals the native mixed pair E4M3 x E5M2 (group 2, exact products) bit for bit.
* **FP32.** There is no FP32 MMA; FP32 dots stay on the vector FMA path and are untouched.
* **Without the capability** the output is identical to the baseline (test above; and the compile-only audit below).

## Results

### 1. Arithmetic of the model against an independent exact oracle (model arithmetic vs CPU oracle)

`validate_matrix_dp.py` (`numeric_validation.json`, vectors `raw/matrix_dp.txt.gz`): 48,000 group steps, 3,000 per
(A format, B format) pair for all 16 pairs, seed 4129, drawn from boundary pools, moderate-exponent operands, and
exponent-spread groups (where RO_48 matters); oracle `tools/model/fdp.py` (Fractions). **0 mismatches.** The vectors are
generated before the model runs. A negative control in the same script corrupts every 97th expected word (495 words) and requires exactly 495
reported mismatches (`negative_control` in the report): it holds.

### 2. Directed end-to-end programs on the model (`chip_validation.json`)

`validate_matrix_dp_chip.py`: assembled programs on `tnca-chip` with `mtx_dp=1`, accumulator read back with `a.vread`,
44 (A, B, K) combinations over all 16 format pairs (those whose fragments fit 64 vector registers), nonzero C, compared
byte for byte with the oracle: **all pass**. Also: a flipped bit in an expectation fails the run; with `mtx_dp=0` the
opcode faults `ILLEGAL_INSTRUCTION` (CAP0 bit 12 clear); a set reserved modifier bit is illegal.

### 3. Compiled kernels through the standard path on the TLM grid (model domain)

Profile `core-vrf-sync-v5-x8-g2c4-dp` + `configs/r41-mma-dp.cfg` (dp) against the same profile without the capability and
`r40-three-core.cfg` (base). Source: commit `d4881995` (its software-tree digest `files:15b184f256edb1520e1e347f88197e38d859ffbc8b918ed1e73a36981bce34a6`
is recorded in both reports and is the same for both runs); model binary hashes in `binaries.sha256`.
Command: `fp8_cases.py --out DIR` (default settings, `torch.compile` of plain Torch on `tnca` tensors, Inductor, the
TNCA compiler, the TLM grid). Each result is compared bit for bit with the independent reference chosen by the profile.

| Run | Matrix commands in the 59 compiled GEMM graphs | Cases | Reference |
|---|---|---|---|
| base (`raw/f-base-*`) | 59 `m.mma.async` | 100 pass, 0 fail | `mma` (sec 9C.1) |
| dp (`raw/f-dp-*`) | 59 `m.mma.dp.async` | 100 pass, 0 fail | `mma.dp` (sec 9C.4) |
| dp judged by the `mma` reference (negative control, `raw/f-neg-*`, scaled_mm and linear groups) | 59 | 52 pass, **7 fail** | `mma` |

The negative control shows the cases can tell the two reductions apart (e4m3 and e5m2, tensorwise and rowwise, FP32
output, and one large bf16 case differ). In the kernel-level control (`fp8_kernels_cases.py --group v4 --quick`, 12 cases,
`raw/q-dp-*` against `raw/neg-dp-*`) the two references agree on all 12: those inputs do not separate the two
reductions, so that group is evidence of correct execution, not of the difference.

Model device cycles of the same graphs (scaled_mm 49 and linear 10 cases, `cases_summary.json`; steady launch, model
domain, **not a performance claim**): base/dp cycle ratio per case between 1.0025 and 1.2969, geometric mean 1.0922. Small
graphs are dominated by transfers, launch and the vector epilogue; the matrix engine's share is shown by section 4.

### 4. Matrix-engine occupancy of one command (model; `chip_validation.json`, `mtx_dp_extra_latency = 1`)

Busy cycles of one 16 x 16 x K command, `m.mma.async` against `m.mma.dp.async`, same format on both operands:

| K | FP16 / BF16 | E5M2 | E4M3 |
|---:|---|---|---|
| 16 | 317 / 182 = 1.74 | 313 / 178 = 1.76 | 313 / 110 = 2.85 |
| 32 | 1093 / 566 = 1.93 | 589 / 318 = 1.85 | 589 / 182 = 3.24 |
| 64 | not run (fragments exceed the registers) | 2149 / 1094 = 1.96 | 2149 / 566 = 3.80 |

The ratios approach 2 and 4 as K grows (a fixed A-buffer fill and pipeline term). The per-step cost is one B register read
plus 16 MAC cycles for G consecutive k; `mtx_dp_extra_latency` adds to the command's pipeline once, and is a parameter,
not a measurement of any lane (G2 reports +1 or +2; `INTERFACE.md` lists +1 to +3 by lane cuts).

## Non-regression

* `make compiler`: generated files up to date, `compiler-unit` passes.
* `compiler/test/test_mma_dp.py`: 6 tests pass (assembly equality as above).
* `tools/performance_harness/default_change_audit.py` against the tree of `origin/r41-hbm-bw` before this work: see the
  section added by milestone 2.

## Reproduce

Build: `cmake -S tools/performance_harness/systemc -B BUILD ...` (README of that directory); then
`validate_matrix_dp.py --binary BUILD/tnca-numeric-probe --out DIR`, `validate_matrix_dp_chip.py --binary BUILD/tnca-chip --out DIR`,
`TNCA_PROFILE=core-vrf-sync-v5-x8-g2c4-dp TNCA_TLM_CONFIG=.../configs/r41-mma-dp.cfg TNCA_BACKEND=systemc-tlm-grid python tools/performance_harness/fp8_cases.py --out DIR`
(base: profile `core-vrf-sync-v5-x8-g2c4`, config `r40-three-core.cfg`). Binary hashes: `binaries.sha256`.


## Milestone 2 (2026-10-29)

### 5. RTL: the compiled kernels on the core with the shared MAC lane (RTL simulation, one core)

`compiler/test/test_mma_dp_rtl.py` compiles `tl.dot` of E4M3, E5M2, FP16 and BF16 tiles with the compiler (target
`tnca-sim-dp`, profile `core-sim-dp`: `tnca_core` with `MAC_DP=1`, `MAC_CUTS=14`, G2's integrated engine, bench
`tb_core_dp14.vvp` built by `make core-mmadp`) and runs the output on the RTL; every one of the 256 output words is compared
with the exact oracle `tools/model/fdp.py`. Result (`raw/rtl_e2e_test.log`): **16 programs pass**: E4M3 and E5M2 at K16/32/64,
FP16 and BF16 at K16; FP16/BF16 at K32/64 do not compile on this two-context target (the A and B fragments exceed its
64 vector registers; the baseline `m.mma.async` has the same limit and the compile fails identically).
Not an image, not the board.

### 6. Reduced-depth model runs, with and without the packed MAC (SystemC/TLM model, uncalibrated; device cycles of the model)

Profile `core-vrf-sync-v5-x8-g2c4-dp` with `configs/r41-mma-dp.cfg` (`mtx_dp_extra_latency=1`) against `core-vrf-sync-v5-x8-g2c4` with
`r40-three-core.cfg`; default settings; same commands as `evidence/r41/chat/p3/runs.json`
(`chat.py ... --prompt "Say hi." --new-tokens 3 --check-tokens 2 --bucket 64`, `--layers 1` for VL and `--layers 4` for Qwen3.5).
Raw: `raw/chat-*`, summaries `chat-vl.json`, `chat-q35.json` (per-kernel device cycles of every step).

| Model | Step | Base cycles | DP cycles | Base/DP |
|---|---|---:|---:|---:|
| Qwen3-VL-2B-FP8 (1 layer) | prefill (step 0) | 571,516,707 | 539,440,036 | 1.059 |
| Qwen3-VL-2B-FP8 (1 layer) | decode (step 2) | 515,149,716 | 493,464,300 | 1.044 |
| Qwen3-VL-2B-FP8 (1 layer) | decode: `w8a8_block_dynamic_fp8_matmul_kernel` | 111,659,710 | 98,346,976 | 1.135 |
| Qwen3-VL-2B-FP8 (1 layer) | decode: BF16 GEMM templates (`triton_tem_*mm*`) | 395,630,303 | 387,257,322 | 1.022 |
| Qwen3.5-4B-FP8 (4 layers) | prefill (step 0) | 2,410,327,588 | 2,237,260,563 | 1.077 |
| Qwen3.5-4B-FP8 (4 layers) | decode (step 2) | 1,856,633,599 | 1,684,821,273 | 1.102 |
| Qwen3.5-4B-FP8 (4 layers) | decode: `w8a8_block_dynamic_fp8_matmul_kernel` | 949,102,418 | 831,575,689 | 1.141 |
| Qwen3.5-4B-FP8 (4 layers) | decode: BF16 GEMM templates (`triton_tem_*mm*`) | 875,816,140 | 821,472,318 | 1.066 |

Both models generate the same tokens with and without the packed MAC. Teacher-forced relative L2 (gate 0.05):
VL 0.02466 / 0.02574 in both runs (pass). Qwen3.5 0.0352 / 0.0600 in both runs: **the gate fails, identically to
the recorded Qwen3.5 runs of `doc/r41_chat.md`; this is not caused by or fixed by the packed MAC** and both rows are kept as failures.
The decode step gains only 4 to 9 percent because the matrix engine is a small part of a decode step on this model; see the
balance analysis `evidence/r41/hbm_bw/BALANCE.md` for what binds.

FP8 linear layers alone (`qwen_tnca_layers.py linears`, the official Hub-kernel route, real checkpoint weights and captured
activations, steady launch, sum of model device cycles; 14 and 26 module/mode cases, all pass in both runs, the dp run bit-identical to the
exact-sum re-derivation `fp8_block_reference.block_dynamic` in dp mode):

| Layers | Mode | Base cycles | DP cycles | Base/DP |
|---|---|---:|---:|---:|
| Qwen3-VL-2B-FP8, layer 0 (7 linears) | decode (M=1) | 112,239,123 | 98,876,176 | 1.135 |
| Qwen3-VL-2B-FP8, layer 0 (7 linears) | prefill (M=81 / 85) | 974,052,722 | 867,069,938 | 1.123 |
| Qwen3.5-4B-FP8, layers 0 and 3 (13 linears) | decode (M=1) | 469,319,077 | 411,138,961 | 1.141 |
| Qwen3.5-4B-FP8, layers 0 and 3 (13 linears) | prefill (M=81 / 85) | 4,012,166,607 | 3,546,793,642 | 1.131 |

Per module and mode: `lin-vl.json`, `lin-q35.json`. M is the captured reference prompt (81 or 85 tokens) and 1.

### 7. Non-regression of the default (compile-only audit)

`default_change_audit.py`, control = tree of `origin/r41-hbm-bw` at `214f2b3f`, candidate = commit `7c0380c1`:
**113 points unchanged, 0 changed** (`audit/audit.json`, raw reports and logs beside it). `make compiler` passes.

### 8. Issue-stream work (compile-only, in progress; request of 2026-10-29)

`tools/r41/loop_issue_count.py` counts the instructions of the innermost loop of a compiled program by class (the BALANCE.md
classes). On `core-vrf-sync-v5-x8-g2c4-dp` the loop body of the compiled `w8a8_block_dynamic_fp8_matmul_kernel` is **8,928
instructions per 16 x 128 x 128 K-block** (32 matrix commands): scalar/predicate/control 2,732, vector fp 2,314, vector
int/move 1,972, vector load/store 1,358, matrix/accumulator/barrier 552 (`issue/vn_window_sweep.txt`).
A dominator-scoped value numbering pass (`optimization.dominating_value_numbering`, enabled only for targets with `matrix_dp`)
would remove 2,634 of the loop's 8,450 IR operations in an unconstrained count (simulated on the dumped IR text, not executed), but the allocator cannot
spill and the unrolled kernel is already at the scalar register limit: windows of 400 operations fail with "out of scalar
registers"; the largest tried that compile (scalar 64, vector 8) give **8,666 instructions (-2.9 percent)**. Executed on the TLM
(`fp8_kernels_cases.py --group v4 --quick`, 12 cases): 12 of 12 bit-identical to the dp reference, device cycles between -8.7 and +2.8 percent
per case (`raw/vn-dp-kernels-report.json.gz` against `raw/q-dp-report.json.gz`), i.e. no demonstrated gain. The remaining items (descriptor weight
loads and scales, hoisting the quantisation across N blocks, a 64 x 64 accumulator tile, software pipelining of the weight transfers,
an M = 1 lowering) are not done; the register-pressure result above says the scalar address arithmetic must be removed by the
lowering (descriptor loads, row-base reuse), not recovered afterwards.
