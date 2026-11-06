# R41 FP8 ops: generic standard-path blockers B1–B6 and V1–V5 (Qwen3.5, Qwen3-VL)

Branch `r41-fp8-ops`, based on `cb4b0d3` (FP8 T3). The six blockers come from
`evidence/r41/stage1/model_estimates.md`. Each fix is generic: it is keyed on
the TNCA device type or on a TTIR operation, and it names no model. Nothing
falls back to the host on the tnca device.

V1–V5 (the Qwen3-VL blockers that remained after B1–B6) are fixed in
`16a19f6`; see "Qwen3-VL blockers V1–V5" below. Their evidence is in
`ops-evidence-v3.tgz`. The B1–B6 evidence is in `ops-evidence.tgz`.

Raw logs and reports are in `ops-evidence.tgz` (layout below). Every run is on
GB10, with torch 2.14.0 and triton 3.8.0. SystemC/TLM results are model
executions (`systemc-tlm-grid`). RTL results are the single-core Icarus bench
(`sim-file`, profile `core-vrf-sync-v5`). No board was used.

## Status per blocker

| | blocker | fix (generic) | where | status |
|---|---|---|---|---|
| B1 | decompose-K skinny GEMM: `NoValidChoicesError` | `TNCAChoices` (upstream `inductor_choices_class`) removes the decompose-K subgraph for TNCA devices and puts back the mm template that the same condition had admitted. Per-K FP32 FMA order is kept | `python/tnca/inductor.py` | **fixed** |
| B2 | `libdevice.log1p` missing | `tnca_log1pf` extern (`tt.extern_elementwise`) lowered to `mathlib.log1p`, built without the SFU | `triton_backend/libdevice.py`, `tnca_cc/frontend.py`, `tnca_cc/mathlib.py` | **fixed** |
| B3 | split-scan `cumsum` at R0_BLOCK=256 | `Scan.num_splits` returns 1 on TNCA devices, so each row is one looped scan | `python/tnca/inductor.py` | **fixed** for float scans. An int64 looped scan is still refused (see remaining blockers) |
| B4 | `linalg_solve_triangular` left as a host call | decomposition into bmm and pointwise ops: inverted 16×16 diagonal blocks (nilpotent product) plus right-looking block substitution. Fixed a descriptor-hoisting bug that the decomposition exposed (a sliced base pointer was read before it was defined) | `python/tnca/linalg_decomp.py`, `tnca_cc/frontend.py` (`713d02e`) | **fixed** |
| B5 | int64 `position_ids.float()` | `arith.sitofp`/`uitofp` of a 64-bit value: branch-free and exactly rounded (top 32 bits plus a sticky bit, `v.clz`, `v.mulh.u`) | `tnca_cc/wide.py`, `tnca_cc/codegen.py` | **fixed** |
| B6 | `math.cos`/`math.sin` have no lowering | `mathlib.sincos`: Cody–Waite plus Payne–Hanek reduction over the full range. `sin` and `cos` of one value share the reduction | `tnca_cc/mathlib.py`, `tnca_cc/frontend.py` | **fixed** |

The previous agent wrote all six fixes (`66db1db`, `713d02e`). I reviewed them and
validated them again against the current source.

## Tests (standard Torch on tnca, independent CPU reference)

Each case is `torch.compile(backend="inductor", fullgraph=True)` of plain Torch on
`tnca` tensors. Every reference is computed on the CPU. Logs: `v2/std.log`,
`v2/solve.log`, `v2/rope.log`, `v2/rtl-math.log`, `v2/std-rtl.log`,
`v2/solve-rtl.log`.

**SystemC/TLM** (`std_ops_cases.py`, `triangular_solve_cases.py`): **39 pass, 0 fail, 2 refused**.

- **B6, sin/cos.**
  - 1,051,000 binary32 values cover every binade's ends and middle, both signs,
    subnormals, inf and NaN. The device output equals the numpy emulation of
    the emitted sequence bit for bit (0 mismatches). Max error is
    0.716 ulp (sin) and 0.722 ulp (cos) against float64.
  - **Exhaustive, all 2^32 inputs** (emulation against float64): sin max
    0.769 ulp, cos max 0.761 ulp. No result is over 1 ulp, and NaN, inf and
    signed zero all match (`exhaustive/exhaustive.json`, run on the same
    `mathlib.py`, sha256 `2a313a48…`).
  - Rotary tables from int64 positions in [0, 64), [100000, 100048) and
    [262000, 262016): 0 bf16 mismatches against the exact result of the
    binary32 argument, and 0 against torch CPU.
- **B2, log1p.**
  - Same 1.05M inputs: 0 emulation mismatches, max 0.767 ulp. Exhaustive: max
    0.834 ulp, 0 results over 1 ulp.
  - `F.softplus` (Inductor emits `log1p(exp(x))`):
    - float32: ≤ 1 ulp against the exact log1p of the device's exp;
    - bf16: 0 ulp;
    - end to end, bounded by the SFU exp: 1.26e-6 relative (float32),
      3.8e-3 (bf16).
- **B5, int64 → float32.**
  - 65,628 values cover every bit length, both signs, the 2^24, 2^32 and 2^53
    rounding edges, INT64_MIN and INT64_MAX.
  - **0 bit mismatches** against torch CPU, both for the plain cast and for
    `positions.float() * scale`.
- **B6 + B5 fused** (new `rope` group): RMS norm plus rotary on a decode
  token, with angles from int64 positions through a K = 1 bmm.
  - Normwise error against float64 (from the same binary32 angles):
    1.26e-7 (B=1 H=16 D=128) and 9.7e-8 (B=2 H=4 D=64). The bound is 2^-18.
  - Torch CPU fp32 scores 1.2e-7 and 1.7e-7.
- **B3, cumsum** at 64, 256 and 4096 elements, on the last and an inner dim:
  max error/bound 0.995 or less against float64. No split scan appears in the
  generated code.
  - REFUSED (reported, not hidden): int64 cumsum over 64 and 300 elements.
    This is a 64-bit `scf.for` loop, refused at compile time.
- **B1, skinny `F.linear`** (1×2048×16, 3×2048×16, 1×4096×32, 2×1024×8), bf16
  and fp32: **0 bit mismatches** against a per-K binary32 FMA in ascending K.
  No decompose-K in the generated code.
- **B4, `torch.linalg.solve_triangular`**: 12 cases, 0 fallbacks.
  - Covers lower and upper, unit and non-unit, `left=False`, broadcast
    batches, n = 1, 5 and 37 (odd), ill-conditioned 32 and 40, a composed graph,
    and the gram-matrix form of a chunked delta rule (64×64).
  - Forward error ‖X−X64‖∞/‖X64‖∞ ranges from 2.2e-8 to 4.7e-7. Each case is
    within n·u·κ and comparable to torch CPU fp32 (2.2e-8 to 4.6e-7).

**RTL**
- `compiler/test/test_std_math.py`: **4/4 OK**. log1p, sin, cos and int64→f32
  kernels on the RTL bench match the emulation bit for bit, and the emulation
  is checked against float64.
- Torch-level `--quick` runs on `sim-file`: per-group reruns with
  `TNCA_HOST_TIMEOUT=14400` (`v2/rtl/`): **17 pass, 0 fail, 2 refused**.
  - sin over 10,758 values: 0 emulation mismatches, max 0.654 ulp.
  - log1p over 10,758 values: 0 emulation mismatches, max 0.538 ulp.
  - softplus: fp32 ≤ 1 ulp, bf16 0 ulp.
  - int64 → f32 over 2,140 values: 0 bit mismatches.
  - float cumsum (64, 300): passes. int64 cumsum: refused, as on TLM.
  - skinny GEMM bf16 and fp32 (1×1024×16): 0 bit mismatches.
  - rope: 8.7e-8.
  - Triangular solve, 3 cases (37×37 upper, 48×48 right-side, 5×5): all
    within bound.
  - **Not run on RTL at torch level:**
    - cos and the rotary tables. The trig process was stopped by PID after
      sin, which alone took 2 h 20 m. cos is still covered at kernel level
      by `test_std_math.py`.
    - The 64×64 gram solve case: it needs more heap than the single-core
      bench has.
- Failed first attempts are kept in the archive:
  - `v2/std-rtl.log`: one process for all groups. The first sin launch
    exceeded the default 600 s host wait and the session hung, giving 18
    errors.
  - `v2/solve-rtl.log`: the 64×64 gram case does not fit the single-core
    bench heap.
  - `v2/compiler2.log`: run without this tree on `PYTHONPATH`, so it imported
    another checkout's `tnca`. 15 errors.

## No regression

- **`make compiler`** (`v2/compiler3.log`, LLVM 20 tnca-opt): generated files
  are up to date, `compiler-unit` **131 tests pass**, `compiler-ir` **6/6 pass**.
  The 131 include the new descriptor-hoisting test.
- **`default_change_audit.py`**: run on this tree, compared with the r41s1
  control (`~/r41-fp8-audit/run-ctrl`). **113 unchanged, 0 changed**
  (`v2/default_change_audit_ops.json`). The new defaults (no decompose-K, no
  split scan, the solve decomposition) do not change the compiled code of any
  of the 113 original points.

## Full-model recompile (compile-only, meta weights, 8 GiB model window)

**Qwen3.5 language branch, full depth** (`model_compile_estimate.py`,
Stage 1 inputs `model-estimates-v1`, transformers `4321355`, **no shims**):

| graph | status | kernels | failed | host fallbacks | refused ops | code working set | steady peak live |
|---|---|---|---|---|---|---|---|
| prefill (128) | **compiled** | 72 | 0 | 0 | 0 | 555,336 B (4.25× the 128 KiB code store) | 3.82 GB |
| decode (16 slots) | **compiled** | 45 | 0 | 0 | 0 | 182,424 B (1.40×) | 3.79 GB |

- Stage 1 needed four shims (`decompose-k-off`, `log1p-kahan`,
  `scan-no-split`, `int32-positions`) and still failed on the RoPE kernel. Here
  every kernel compiles on the standard path.
- The 48 `linalg_solve_triangular` calls per prefill, the decay cumsum, the
  depthwise conv1d and the softplus are all lowered.
- Prefill compile wall time is 1450 s.
- A code working set above the code store is a performance item (LRU
  reloads), not a compile blocker.

**Qwen3-VL**: the state after B1–B6 had four findings. They are V1–V5, and
the next section closes them.

## Qwen3-VL blockers V1–V5

Commit `16a19f6`. Every fix is keyed on the TNCA device type or on a TTIR
operation, and none names a model or a checkpoint. There is no CPU numerical
fallback. Tests are `std_ops_cases.py` on `systemc-tlm-grid`
(`TNCA_TLM_MEMORY_BYTES=0x200000000`), each against an independent CPU
reference. Final run (`v3/std.log`, tree `files:4556ea11…`): **48 pass,
0 fail, 1 refused** across all nine groups (the B1–B6 groups included).

| | blocker | fix (generic) | where | test | status |
|---|---|---|---|---|---|
| V1 | fused norm + mrope + q·kᵀ decode kernel: "tt.expand_dims of something that is not a range" | The existing TNCA descriptor rule ("one range tree split over two descriptor dimensions → pointer access") is now also applied to the block shape **before** Inductor removes stride-0 dimensions, for the reduction tree only. A periodic read (a 64-value rotary table at `r % 64` over a 128-wide reduction) splits R0 through a stride-0 dimension. The rule now sees that split, and the access takes the pointer lowering. A split of the x tree is degenerate on TNCA (XBLOCK = 1) and keeps its descriptor; widening the rule to it changed 4 attention-decode audit points, so it was narrowed. The unvalidated `descriptor_broadcast.py` is replaced and removed | `python/tnca/inductor.py` (`_record_full_block_shape`, TMA checker) | `rope` group: RMS norm, interleaved 3-row mrope (sections 24/20/20), then q·kᵀ/√D over S slots. f32 and bf16, B=1 H=16 D=128 S=144 and B=2 H=4 D=128 S=40, against float64. **Reproducer:** the same group on a tree with the rule off (`v3/rope-nofix.log`) fails the two S=144 cases with exactly the Qwen3-VL diagnostic | **fixed** |
| V2 | 64-bit loop-carried values refused (`scf.for` over int64) | `scf.for` carries a 64-bit block as its low and high word vectors (`wide.carried_pair`), assigned by the yield like any other carry. 64-bit `arith.divsi/divui/remsi/remui` lanes are exact: power-of-two shift, a reciprocal multiply through `s.mulh.u` (now emitted; `doc/support_matrix.md` regenerated) for other constants, and a 64-step restoring loop for a runtime divisor. Inductor's masked_scatter and bool-index kernels divide int64 indices by constants | `compiler/python/tnca_cc/{frontend,wide,codegen}.py` | `scan`: int64 cumsum over 64, 300 and 4096 (\|x\| < 2⁴⁰, running total past 2⁶²); int64 sum over 4096; a token count `(ids == k).sum()`; `masked_scatter` f32/bf16. **All exact** against torch CPU int64. RTL: `test_std_math.test_int64_divmod` (INT64_MIN/MAX, negative divisors, runtime divisors) bit for bit | **fixed** (exact, no narrowing) |
| V3 | `libdevice.erf` missing (exact GELU) | `mathlib.erf`, no SFU. It is fma(x, P(x²), x) below 1, and 1 − exp(Q(\|x\| − 1)) above, up to 3.92; beyond that ±1. The exponential is built from fma and integer operations (exact reduction, fma(−p, 2ⁿ, 1)). Coefficients come from a Lawson fit, rounded to binary32 one at a time. It is reached through TNCA libdevice as `tnca_erff` | `tnca_cc/mathlib.py`, `tnca_cc/frontend.py`, `triton_backend/libdevice.py` | **Exhaustive, all 2³² inputs** against float64 (`v3/exhaustive/erf.json`, `transcendental_numeric.py`): **max 0.991 ulp, 0 over 1 ulp**, 0 NaN/inf/signed-zero mismatches. Device, 1.05 M values: 0 mismatches against the emulation, max 0.836 ulp. Exact GELU, 65,546 values, against float64 GELU within the bound set by the formula's own roundings: f32 0.575 of the bound, bf16 0.975. RTL `test_erf` bit for bit | **fixed** |
| V4 | host readback of `cu_seqlens` (vision attention `lengths.tolist()`, 72 `_local_scalar_dense` + 216 `_assert_scalar` per image) | See "What V4/V5 compute from what" below. `grid_thw` and `cu_seqlens` are the processor's integer metadata. On the native device (`TNCA_TORCH_DEVICE=native`), `configure()` now compiles kernels over **CPU** tensors for the host (Inductor's C++ backend). Before, they were staged to TNCA implicitly (`cpu_backend="triton"`, which is kept for the B39 CPU-tensor path). `host_fallbacks()` no longer lists a C++ kernel whose every argument is an integer or bool. Those are reported as `host_metadata_kernels`. Any float C++ kernel is still a fallback. `scalar_reads()` counts `.item()` reads as host or device. The vision graph keeps `grid_thw` and `cu_seqlens` on the CPU. The one-image split's own check makes u0 = L, so every TNCA kernel is static | `python/tnca/inductor.py`, `vl_compile_probe.py` | `meta`: packed attention split at CPU `cu_seqlens` [0, 256], H=4 L=256 D=64. 5.7e-7 normwise against float64 (torch CPU 1.1e-6). Host metadata kernel `cpp_fused_slice_sub_0` (int32 `cu[1:] - cu[:-1]`); 3 host `.item()` reads, **0 device reads** | **fixed for one segment per graph** (one image). **REFUSED, documented:** two or more segments (several images, video frames). Their lengths are host values but not compile-time constants (u0, L − u0), so the per-segment kernels are compiled for symbolic sizes, which the TNCA lowering refuses (`scf.if whose branch contains arith.remsi`) |
| V5 | mm-prefill: bool-mask `aten.index` / `index_put_` (DeepStack), `_is_all_true` (`torch._check`) as ATen fallbacks; masked_scatter in int64 | `python/tnca/mask_index.py` (replaces the `check_decomp.py` candidate), registered by `configure()` for TNCA device types only. `x[m] = rows` becomes `masked_scatter`, which is on-device and reads no count back. `x[m] = row` becomes `where`. `x[m]` reads the count back once (`.item()`), the same sync any PyTorch device makes for a boolean index; the rows are gathered on the device (rank = cumsum − 1, scattered, then gathered). `_is_all_true` becomes `all()`, followed by the check's own read of one bool | `python/tnca/mask_index.py`, `inductor.py` | `mask`: the DeepStack form (mask from token ids, `torch._check_tensor_all`, masked_scatter, `h[m] = h[m] + v`) in f32 and bf16; `x[m]`, `x[m] = rows`, `x[m] = row` over a 2-D mask. **0 mismatches** against torch CPU, 0 host fallbacks, device reads exactly as expected (2, 1, 0, 0). A mismatched feature count raises at run time | **fixed**, with the reads below |

### What V4/V5 compute from what (transformers 5.x `modeling_qwen3_vl.py`, `vision_utils.py`)

- **Vision.** `get_vision_attention_seqlens(grid_thw)` gives `cu_seqlens`,
  the cumsum of h·w per frame. Each of the 24 blocks computes
  `lengths = cu_seqlens[1:] - cu_seqlens[:-1]`, then `torch.split(q|k|v,
  lengths.tolist())` (3 × 24 = 72 reads), and attends per segment.
  `vision_utils` exists so that callers precompute these from `grid_thw`
  outside the graph (its docstring says so); they are passed as kwargs.
  These are integer metadata of the input, known on the host before upload.
  They now stay there: one host C++ kernel and 72 host reads, **no device
  readback**. The precomputed rotary position ids and the interpolation
  indices and weights stay on tnca, as before. They are derived from
  `grid_thw` by the official precompute API, are not computed from device
  data, and are not new in this change.
- **mm-prefill.** `get_placeholder_mask` computes the mask from `input_ids
  == image_token_id` **on the device**: `input_ids` must be on tnca for the
  embedding gather, and the API refuses `input_ids` together with
  `inputs_embeds`. So the mask and its count are device data in the
  official module.
  - masked_scatter: on device, no read.
  - `torch_compilable_check(n·hidden == numel)`: device `all()`, then one
    bool read for the check.
  - DeepStack `h[m, :] + v` (×3 layers): one count read each. The add
    makes Dynamo assert count = N, so all shapes after it are static.
  - That is **4 device scalar reads per prefill** (`scalar_reads`), and no
    numerical work on the host.
  - Removing them would need the mask from host `input_ids`, which the
    official forward does not accept. Alternatively,
    `TRANSFORMERS_DISABLE_TORCH_CHECK=1` (transformers' production switch)
    drops the check's read; it was not used here.
  - Text-decode has none.

### Qwen3-VL recompile (`vl_compile_probe.py`, full depth, grid 1×16×16, eager attention, 8 GiB stand-in, no shims, no candidate fixes; `v3/vl/report.json`)

| graph | status | kernels | failed | host fallbacks | scalar reads (host / device) | code bytes |
|---|---|---|---|---|---|---|
| text-prefill (128) | **compiled** | 26 | 0 | 0 | 0 / 0 | 137,200 |
| text-decode (16 slots) | **compiled** (was: 1 failed kernel) | 28 | 0 | 0 | 0 / 0 | 158,712 |
| mm-prefill (128, 64 image tokens) | compiled-with-findings | 33 | 0 | 0 | 0 / 4 | 299,720 |
| vision (256 patches) | **compiled** (was: erf missing, `u0 >= 256`) | 27 | 0 | 0 (1 host metadata kernel) | 72 / 0 | 179,392 |

- **mm-prefill.** All 33 kernels compile, and there is no fallback and no
  refused operator. The remaining finding is the compile-only harness: with
  `TNCA_LAUNCH=skip` the stand-in device holds no data. The first device
  read (the placeholder check) returns garbage, and its assert stops the
  call ("u0 >= 1", 1 launch). That is not a lowering gap, but the steady
  launch count is not measured. On a device that executes, the 4 reads
  return real values.
- **Vision** completes its steady call, 399 launches.

### No regression (final tree)

- **`make compiler`** (`v3/compiler.log`, LLVM 20 `tnca-opt` rebuilt):
  `compiler-unit` **133 tests pass** (new: `test_erf`, `test_int64_divmod`),
  `compiler-ir` **6/6 pass**.
- **RTL:** `test_std_math.py` **6/6 OK** on the Icarus bench
  (`v3/rtl-math.log`).
- **`default_change_audit.py`**, compared with the r41s1 control: **113
  unchanged, 0 changed** (`v3/default_change_audit_ops3.json`, tree
  `files:4556ea11…`).
  - The first audit with the wider V1 rule changed 4
    `op.attention_decode` points: a descriptor split on XBLOCK was refused.
  - The rule was narrowed to the reduction tree, and the audit was rerun.

## Files

- Fixes: `python/tnca/inductor.py`, `python/tnca/linalg_decomp.py`,
  `python/tnca/triton_backend/libdevice.py`, `compiler/python/tnca_cc/{mathlib,wide,frontend,codegen,register_blocks}.py`,
  `doc/support_matrix.md` (`v.mulh.u` and `v.clz` are now emitted).
- Tests: `compiler/test/test_std_math.py`, `compiler/test/test_descriptor_publication.py`,
  `tools/performance_harness/{std_ops_cases,triangular_solve_cases,transcendental_numeric}.py`.
- Probe: `tools/performance_harness/vl_compile_probe.py` (its candidate-fix
  list is now empty: both candidates were replaced by default-on fixes).
- V1–V5: `python/tnca/{inductor,mask_index}.py`,
  `python/tnca/triton_backend/libdevice.py`,
  `compiler/python/tnca_cc/{frontend,wide,codegen,mathlib}.py`,
  `compiler/test/test_std_math.py`,
  `tools/performance_harness/{std_ops_cases,transcendental_numeric,fp8_cases}.py`.

## `ops-evidence-v3.tgz` (V1–V5)

`v3/`:
- `std.log`, `std/report.json`, `std/code/`: all groups, final tree.
- `rope-nofix.*`: V1 reproducer on a tree with the rule off.
- `exhaustive/`: erf over 2³², with source hashes.
- `vl.log`, `vl/report.json`: the Qwen3-VL recompile.
- `compiler.log`, `tnca-opt.log`, `rtl-math.log`.
- `audit.log`, `default_change_audit_ops3.json`, `tree.txt`.

## `ops-evidence.tgz`

`v2/`: this validation (runner log, compiler, RTL math, std, solve, rope,
RTL runs, the Qwen3.5 and Qwen3-VL `report.json` files, audit report and
comparison). `exhaustive/`: the 2^32 runs. `rope-nofix/`: the rope case on a
tree without `descriptor_broadcast`.

## Reproduce (GB10)

```sh
R=~/r41-fp8-ops; export PYTHONPATH=$R/python:$R/compiler/python:$R
export TNCA_TLM_BINARY=~/r41-fp8-ops-build/tnca-chip
P=~/tc_env/torchenv/bin/python; TP=~/r41-stage1/envs/transformers-4321355/bin/python
make compiler PYTHON=$P LLVM_PREFIX=~/tc_env/root/usr/lib/llvm-20
$P -m unittest discover -s compiler/test -p test_std_math.py
$P tools/performance_harness/std_ops_cases.py --out OUT          # TNCA_BACKEND=sim-file ... --quick for RTL
$P tools/performance_harness/triangular_solve_cases.py --out OUT
$P tools/performance_harness/transcendental_numeric.py ...       # exhaustive, see exhaustive/run.log
$P tools/performance_harness/transcendental_numeric.py --functions erf --jobs 4 --out erf.json
$P tools/performance_harness/std_ops_cases.py --out OUT --group erf --group scan --group rope --group meta --group mask
$TP tools/performance_harness/model_compile_estimate.py --config I/config.json --inventory I/inventory.json \
    --sources I/sources.json --out OUT --graphs prefill decode --stand-in-heap-bytes 8589934592
$TP tools/performance_harness/vl_compile_probe.py --config V/config.json --header V/safetensors_header.json \
    --out OUT --graphs text-prefill text-decode mm-prefill vision --stand-in-heap-bytes 8589934592
$P tools/performance_harness/default_change_audit.py run --source-root $R --software-tree files:<sha> --label ops --out A
$P tools/performance_harness/default_change_audit.py compare --control ~/r41-fp8-audit/run-ctrl/report.json \
    --candidate A/report.json --out audit.json
```
