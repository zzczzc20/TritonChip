# R41 FP8 ops: generic standard-path blockers B1–B6 (Qwen3.5, Qwen3-VL)

Branch `r41-fp8-ops`, based on `cb4b0d3` (FP8 T3). The six blockers come from
`evidence/r41/stage1/model_estimates.md`. Each fix is generic: it is keyed on
the TNCA device type or on a TTIR operation, and it names no model. Nothing
falls back to the host on the tnca device.

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

**Qwen3-VL** (`vl_compile_probe.py`, full text and vision depth, grid
1×16×16, eager attention, no shims, no candidate fixes):

| graph | status | kernels | finding |
|---|---|---|---|
| text-prefill | **compiled** | 26 | — |
| text-decode | compiled-with-findings | 27 | 1 kernel (norm + mrope, `select`/`slice` of mrope sections): "tt.expand_dims of something that is not a range". This is a tensor descriptor on a periodic access |
| mm-prefill | compiled-with-findings | 32 | `masked_scatter`/`eq·sum` over int64: 64-bit `scf.for` (2 kernels); host fallbacks `aten.index`, `index_put_` (bool mask, deepstack), `_is_all_true` |
| vision | compiled-with-findings | 27 | `libdevice.erf` missing (exact GELU in the patch merger); `u0 >= 256`, a data-dependent `cu_seqlens` |

- **mrope:** text-prefill compiles.
- **mrope in decode:** fails on one descriptor kernel (blocker V1).
- **Vision patch embedding:** the Conv3d does not appear as a fallback or a
  failed kernel.
- **Vision attention:** blocked by the `cu_seqlens` host readback (V4).

## Remaining blockers

| | blocker | where | state |
|---|---|---|---|
| V1 | Tensor descriptor on a periodic access, after Inductor strips its stride-0 dimension | Qwen3-VL text-decode, 1 kernel | candidate fix `python/tnca/descriptor_broadcast.py`. An earlier run with it (`--candidate-fixes descriptor-broadcast`) compiled text-decode. **Not default-on**: no standalone reproducer yet (the new `rope` case compiles without it), so it is not validated |
| V2 | 64-bit loop-carried values (`scf.for` over int64): an int64 cumsum, or an int64 `eq·sum` reduction | Qwen3-VL mm-prefill (`masked_scatter`, placeholder count), `std_ops_cases` int64 cumsum (REFUSED) | open: `tnca_cc/wide.py` holds 64-bit values only as straight-line pairs |
| V3 | `libdevice.erf` (exact GELU) | Qwen3-VL vision patch merger | open: needs a mathlib erf with an exhaustive ulp check like log1p/sin/cos |
| V4 | **Host readback of `cu_seqlens` in vision attention**: `cu_seqlens.tolist()`, 72 `_local_scalar_dense` + `split_with_sizes` + 216 `_assert_scalar` per image (r41-fp8 `qwen/README.md` B4); here "u0 >= 256" | Qwen3-VL vision | **out of scope, open**: it needs unbacked-symint support or a fixed-shape variant. It is not a lowering gap |
| V5 | Data-dependent bool-mask `index` / `index_put_` (deepstack) and `torch._check` → `_is_all_true` | Qwen3-VL mm-prefill | candidate `python/tnca/check_decomp.py` moves `_is_all_true` onto the device, where it then hits V2. Not default-on. `index`/`index_put_` with a bool mask is data-dependent (same class as V4) |

Neither candidate fix is installed by `tnca.inductor.configure()`. Both are
used only through `vl_compile_probe.py --candidate-fixes`.

## Files

- Fixes: `python/tnca/inductor.py`, `python/tnca/linalg_decomp.py`,
  `python/tnca/triton_backend/libdevice.py`, `compiler/python/tnca_cc/{mathlib,wide,frontend,codegen,register_blocks}.py`,
  `doc/support_matrix.md` (`v.mulh.u` and `v.clz` are now emitted).
- Tests: `compiler/test/test_std_math.py`, `compiler/test/test_descriptor_publication.py`,
  `tools/performance_harness/{std_ops_cases,triangular_solve_cases,transcendental_numeric}.py`.
- Probes and candidates: `tools/performance_harness/vl_compile_probe.py`,
  `python/tnca/{descriptor_broadcast,check_decomp}.py`.

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
$TP tools/performance_harness/model_compile_estimate.py --config I/config.json --inventory I/inventory.json \
    --sources I/sources.json --out OUT --graphs prefill decode --stand-in-heap-bytes 8589934592
$TP tools/performance_harness/vl_compile_probe.py --config V/config.json --header V/safetensors_header.json \
    --out OUT --graphs text-prefill text-decode mm-prefill vision --stand-in-heap-bytes 8589934592
$P tools/performance_harness/default_change_audit.py run --source-root $R --software-tree files:<sha> --label ops --out A
$P tools/performance_harness/default_change_audit.py compare --control ~/r41-fp8-audit/run-ctrl/report.json \
    --candidate A/report.json --out audit.json
```
