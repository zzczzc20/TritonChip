# FP8 as a generic Torch dtype on `tnca`

Branch `r41-fp8`, from the frozen Stage 1 commit `21d09f3`. Development evidence:
the SystemC/TLM results are uncalibrated model predictions and the RTL results
are Icarus simulations. No board was used.

Everything runs on the standard path: a `torch.Tensor` on `tnca`, then
`torch.compile(backend="inductor", fullgraph=True)`, Inductor, Triton, the TNCA
backend, a sealed artifact and the Session. The graphs are the generic patterns
quantisation code is written in. No model, layer or checkpoint name appears in
any change or test.

## What works

| Capability | How |
|---|---|
| fp32/bf16/fp16 ↔ `float8_e4m3fn`/`float8_e5m2` casts in compiled kernels, bit-exact against torch CPU, including NaN sign and payload | The instruction (`vf.cvt`, ISA sec 8.4) computes every number exactly as torch 2.14 does. The compiler rewrites only the NaN lanes, and only for conversions that have an FP8 end (`frontend._fp8_nan_rule` / `_fp8_nan_fix`). |
| `float8_e4m3fn` ↔ `float8_e5m2` | Inductor's Triton codegen refuses a direct FP8→FP8 cast on every device. A tnca lowering policy routes the cast through binary32, which is exact (`policy._fp8_cross_convert`). |
| Host↔tnca copies, views, slices, `as_strided`, `uint8` reinterpretation, `clone`/`copy_` into strided views, compiled slicing and `cat` | Already worked. An eager device copy whose source is FP8 (`d.to(torch.float32)` on a tnca FP8 tensor) used to fail: Triton cannot cast `other=0` to FP8. `copy._copy_kernel` now loads FP8 without `other`. |
| `(x / s).clamp(-448, 448).to(fp8)` with static, rowwise and amax-derived scales; `w.to(bf16) * s` and per-channel dequantisation | Compiled, with no host fallback. |
| `torch._scaled_mm` through Inductor's own `tuned_scaled_mm` (mm template plus `scale_mm_epilogue`) | New TNCA `scaled_mm` template heuristic (`inductor.TNCAScaledMMTemplateConfigHeuristic`). It uses upstream's scaled-mm mixin with the same single legal tile as mm. Without it, the device's plain mm heuristic answered and the template failed (`AssertionError (2, 4, 0, 4)`). |
| `_scaled_mm` cases covered | Tensorwise and rowwise scales; bf16, fp16 and fp32 out; e4m3 and e5m2 out with and without `scale_result`; with and without bias; `use_fast_accum`; e4m3×e4m3 and e5m2×e5m2; e4m3×e5m2 for one 16-deep K block; M, N and K tails; K ≤ 16 (upstream offers no config there, so the heuristic lifts that rule); every legal `TNCA_GEMM_TILE`. |
| `scale_result` on an FP8 output | Inductor's `tuned_scaled_mm` drops `scale_result`, so FP8 results came out unscaled, and wrong in all 1 536 elements of the probe. A tnca lowering now computes it the way torch does: binary32 template output, `* (1/scale_result)`, then the cast (`policy._scaled_mm_result_scale`). |
| FP8 MMA (`m.mma.async.*.e4m3/.e5m2`, CAPABILITY0.MATRIX_FP8) for every same-encoding FP8 `tl.dot` | Already the lowering. Three fixes landed. (1) K32/K64 FP8 dots were **wrong** (the bug predates this work): they were split like elementwise ops and paired A's second K block with B's first. They are now native matrix ops (`register_blocks._native_matrix`). (2) An FP8 tile that only `tl.dot` reads now takes the transfer engine like an FP16 tile, instead of row loads plus a repack. A column-major FP8 operand is gathered straight into the repack (`frontend._fp8_mma_operand`). (3) e4m3×e5m2 becomes an exact FP16 MMA for one K block and is refused with a diagnostic if K is deeper. |
| Static W8A8 linear and weight-only FP8 linear in plain torch, small and medium shapes, including non-multiple-of-16 tails and a 3-D input | Bit-exact against the independent references (see Numerics). |

## Files changed

- `compiler/python/tnca_cc/frontend.py`
  - FP8 conversion NaN rule.
  - FP8 MMA operand transfer and the gathered-repack path.
  - e4m3×e5m2 dot.
  - `_value_def` (a value's defining op).
- `compiler/python/tnca_cc/register_blocks.py`: FP8 counts as a native matrix type, which fixes deep-K FP8 dots. New `view_like`.
- `python/tnca/inductor.py`: the TNCA `scaled_mm` template heuristic, for `tnca` and `cpu`.
- `python/tnca/torch_device/policy.py`: `TNCA_POLICY_FP8_CONVERT` (e4m3↔e5m2 through binary32, and `_scaled_mm` `scale_result` for FP8 outputs).
- `python/tnca/torch_device/copy.py`: eager device copy from an FP8 source.
- New tests:
  - `compiler/test/test_fp8_convert.py` (7 tests, RTL)
  - `compiler/test/test_fp8_mma.py` (6 tests, RTL)
  - `tools/performance_harness/fp8_cases.py` (the Torch-level matrix)
  - `tools/performance_harness/fp8_numeric_exhaustive.py` (every input)

## Conversion semantics: torch, the ISA, and what the compiled kernel returns

torch 2.14's own rules (`c10/util/Float8_e4m3fn.h` and `Float8_e5m2.h`, read from the
installed headers) and the conversion instruction (`xls/tnca_formats.x`
`narrow_from_f32`, and the model's `tnca_numeric.hpp`, which is checked against the RTL):

| Case | torch CPU 2.14 | `vf.cvt` (ISA sec 8.4) | compiled tnca kernel |
|---|---|---|---|
| rounding | RNE (ties to even), subnormals kept | same | same |
| e4m3: \|x\| ≥ 464 (the tie) up to inf, and ±inf | saturates to ±448 (`0x7e`/`0xfe`) | saturates to ±448 | ±448 |
| e5m2: \|x\| ≥ 61440 (the tie), and ±inf | ±inf (`0x7c`/`0xfc`) | ±inf | ±inf |
| NaN → FP8 | `0x7f` with the **source sign** (`0xff` for −NaN) | `0x7f`, always positive | sign kept: `y \| (src >> (w−8) & 0x80)` |
| FP8 NaN → binary32 | sign, quiet bit and payload kept: e4m3 `0x7ff00000\|s`, e5m2 `0x7fc00000\|m<<21\|s` | canonical `0x7fc00000` | torch's encoding (a select on ordered) |
| FP8 NaN → fp16 | e4m3 `0x7f80\|s`, e5m2 `x<<8 \| 0x7e00` | `0x7fff` | torch's encoding |
| FP8 NaN → bf16 | `0x7fc0` (c10 BFloat16: one NaN, no sign) | `0x7fff` | `0x7fc0` |
| ±0, every finite value, max finite 448 / 57344 | exact | exact | exact |

Note: the ISA's "E4M3 overflow saturates to ±448" matches torch 2.14, where
`fp8e4m3fn_from_fp32_value` saturates (older c10 returned NaN above 464).

The instruction therefore differs from torch **only on NaN inputs**. This is
verified for every input, not just a sample
(`fp8_numeric_exhaustive.py`, `runs/exhaustive.json` in the archive):

| Conversion | Inputs | `vf.cvt` ≠ torch | of which non-NaN inputs | compiled ≠ torch |
|---|---:|---:|---:|---:|
| f32→e4m3 | 4 294 967 296 | 8 388 607 (every −NaN) | 0 | **0** |
| f32→e5m2 | 4 294 967 296 | 8 388 607 | 0 | **0** |
| bf16→e4m3 / e5m2 (via f32, as Inductor emits) | 65 536 each | 127 each | 0 | **0** |
| f16→e4m3 / e5m2 (via f32) | 65 536 each | 1 023 each | 0 | **0** |
| e4m3→f32 / f16 / bf16 | 256 each | 2 | 0 | **0** |
| e5m2→f32 / f16 / bf16 | 256 each | 5 / 6 / 6 | 0 | **0** |

**How the rule is implemented, and why it is generic.** A tensor cast is a copy of
bits, so torch's NaN encodings are results in their own right. The compiler
rewrites the NaN lanes of a conversion only when one end is FP8.

- Narrowing to FP8 takes 3 vector ops. Every non-NaN result already carries the
  source sign, so OR-ing the sign in is exact on every lane.
- Widening from FP8 takes 4–5 ops plus a select. It is skipped when every user
  of the widened value is NaN-canonicalising arithmetic
  (`add/sub/mul/div/fma/cmp/min/max`), as in `w.float() * s`.
- Inductor widens a 16-bit load before casting to FP8, and widens FP8 before
  storing bf16/fp16. Widening is exact for numbers, so the pair is torch's single
  conversion. The rule looks through it to the root value, whose bits still hold
  the NaN that the widening dropped (an fp16 NaN widens to the positive canonical
  NaN).
- Conversions between the wide formats are left alone. Changing them would change
  every fp16/bf16 kernel for a NaN payload. The audit below shows that no default
  code changed.

## Test matrix and results

Two trees appear below. **T3** = `files:5bb366a8…` is this worktree, the final
tree. **T2** = `files:891ce053…` is T3 without the `scale_result` lowering in
`policy.py` and without the matching harness cases. T2 has the same compiler
and the same code for everything except `_scaled_mm(scale_result=…, out_dtype=fp8)`.

| Run | Backend | Tree | Result |
|---|---|---|---|
| `fp8_cases.py` (full) | `systemc-tlm-grid` | T3 | **100/100 pass** ([table](#full-model-run)) |
| `fp8_cases.py --quick` | `systemc-tlm` (per-program launches) | T3 | **49/49 pass** |
| `fp8_cases.py --quick` | `sim-file` (single-core RTL, Icarus) | T3 | **49/49 pass**, bitwise. Casts: the structured binary32 set (11 538 values); a sampled bf16/fp16 set that includes every NaN, inf and subnormal pattern; every FP8 pattern; 2-D kernels; eager copies; e4m3↔e5m2. Also plumbing, `_scaled_mm` (incl. e4m3×e5m2) and the linears. |
| `fp8_cases.py --quick --group scaled_mm --group linear` | `fpga-multicore-sim` (3-core RTL stand-in) | T3 | **8/8 pass**: 4 `_scaled_mm` (incl. e4m3×e5m2) and 4 linears, each bitwise equal to the per-K reference. The RTL has no MMA counter, so the count is static only. |
| `scaled_mm` + `linear --quick` under every legal `TNCA_GEMM_TILE`: 16,32,16 / 16,64,16 / 16,16,32 / 16,32,32, and with window reuse 16,64,32 / 16,16,64 | `systemc-tlm-grid` | T2 | **Every same-encoding `_scaled_mm` and every W8A8 case passes at every tile**, with the MMA count matching the tile. e4m3×e5m2 passes at 16,16,16 and 16,32,16 and is refused with a diagnostic at the others. The weight-only linear x(16,64) hits a resource refusal at 16,32,32 and 16,16,64. That is a bf16 GEMM with an in-kernel dequant prologue, and it is **pre-existing**: the r41s1 tree gives the same refusals (`runs/ctrl-tile-*`). |
| `fp8_numeric_exhaustive.py` (every input) | model conversion functions plus the compiler's rule, against torch CPU | T2 | **0 mismatches** |
| `compiler/test/test_fp8_convert.py` (RTL) | `tb_core` | T3 | **7/7 pass**: narrow, look-through, widen to f32/f16/bf16, tile path, no rule where nothing reads the NaN, no rule without an FP8 end. A control shows the bare instruction gives `0x7f` for −NaN on the RTL. |
| `compiler/test/test_fp8_mma.py` (RTL) | `tb_core` | T3 | **6/6 pass**: K32/K64 e4m3/e5m2 as one MMA, operands by transfer, gathered column-major B, e4m3×e5m2 as an FP16 MMA, deeper mixed dot refused. A control shows the old K split is wrong on the RTL. |
| Control: `fp8_cases.py --quick` on the unmodified r41s1 tree | `systemc-tlm-grid` | r41s1 + harness | 15 pass, 21 fail, 8 error. Every cast fails **only on NaN encodings**. Eager FP8-source copies error. e4m3↔e5m2 errors. Every `_scaled_mm` errors. |

The full model run covers:

- **casts**:
  - every 16-bit pattern;
  - every FP8 pattern;
  - 170 258 structured binary32 values: every FP8 value, every midpoint and its
    neighbours, the overflow and subnormal boundaries, NaN payloads, and random
    bit patterns over every exponent;
  - 2-D transposed-read kernels;
  - eager device copies.
- **plumbing**: 9 copy/view checks per format, slice/cat, quantise and dequantise.
- **scaled_mm**: 49 cases.
- **linear**: 10 cases.

Raw reports and logs are in `fp8-evidence.tgz`:

| Path | Contents |
|---|---|
| `runs/final-*` | T3: model full and per-program runs, RTL sim-file and multicore runs |
| `runs/tile-*`, `runs/t2-tlm-linear` | T2 tile sweep and linear group |
| `runs/control` | the r41s1 control run |
| `runs/ctrl-tile-*` | r41s1 under wide tiles |
| `runs/exhaustive.json` | the every-input check |
| `audit/` | both audit runs and the comparisons (`default_change_audit_fp8_final.json` is T3's) |
| `fragments/` | model and RTL K-order reports |
| `logs/` | `make compiler`, `compiler-rtl`, `triton-e2e`, the two FP8 compiler tests |
| `asm/` | every compiled FP8 GEMM and linear kernel |
| `generated/` | Inductor's Triton source per graph |
| `probes/` | cycle comparison before and after the operand change, Inductor-CPU W8A8 and `emulate_precision_casts`, with their scripts |

## Numerics

**`_scaled_mm` against an independent reference: bitwise equal in every case.**

The reference dequantises FP8 exactly to binary32. It accumulates one FMA per K
index in ascending K from +0.0, which is the MMA's order (ISA sec 9C.1). FP8×FP8
products are exact in binary32 (checked), so each add of a product rounds the way
an FMA does. It then applies Inductor's epilogue in binary32,
`acc * (scale_a * scale_b) + bias`, and rounds once to `out_dtype`.

The inputs span several binades, so accumulation does round. Another K order
would change the binary32 accumulator, so a bitwise match shows the ascending K
order (`order_sensitivity` in each row):

| Shape | Order | Accumulators changed |
|---|---|---:|
| 32×48×64 | 16-blocked | 7–26 of 1 536 |
| 32×48×64 | reversed | 24–62 of 1 536 |
| 64×96×128 | reversed | 631 of 6 144 |
| 128×192×384 | 16-blocked | 2 904 of 24 576 |
| 128×192×384 | reversed | 11 197 of 24 576 |

Only the smallest cases (K ≤ 16 and 16×32×8) are order-insensitive.

**Against torch CPU `_scaled_mm`.** bf16/fp16 outputs agree except for 0–13
elements per case, each within 1–3 ulp. fp32 outputs differ in 60–83 % of
elements, by up to 16 384 ulp where the result cancels.

The cause is exact and identified: torch CPU computes something else. It is
`_scaled_mm_out_cpu_emulated`: each operand is multiplied by its scale in binary32
first (one rounding per element), then a binary32 BLAS matmul runs in its own
order. The harness re-implements that: `(a.float()*sa) @ (b.float()*sb) + bias`
in torch equals torch CPU `_scaled_mm` bitwise in **every** case (`torch CPU =
scale-first emulation: 0 mismatches`).

**Static W8A8 linear.** The compiled quantised input and the output are bitwise
equal to the reference: quantisation in binary32, then the `_scaled_mm` reference.
That holds for x(16,64)·w(32,64), the tails x(7,40)·w(24,40) and x(33,72)·w(40,72),
the medium x(64,256)·w(192,256), and the 3-D x(2,13,96)·w(80,96).

Against eager CPU, about 3–4 % of the quantised inputs differ by one FP8 ulp, and
the outputs follow. This is Inductor's precision choice on every Triton device,
not TNCA's:

- Eager rounds `x / input_scale` to bf16 (the tensor's dtype) before the clamp and
  the cast. This is a double rounding, and the harness checks it: eager q is the
  bf16-rounded quotient bit for bit.
- Inductor's Triton kernels keep the binary32 quotient.
- Inductor's own C++ CPU backend also rounds to bf16 and matches eager bitwise.
  TNCA matches Inductor-Triton semantics (`probes/cpuind.log`).
- `emulate_precision_casts=True` reduces the difference but does not remove it.
  It only materialises explicit casts.
- Fed the device's own q, torch CPU `_scaled_mm` agrees with the device output in
  all but 0–3 elements (1 bf16 ulp each).

**Weight-only FP8 linear.** The dequantised weight is rounded once to bf16, then
multiplied by the bf16 activations: per-K binary32 FMA, bias added once in binary32,
one rounding. It is bitwise equal to the reference in every shape. Eager CPU agrees
in all but 2 of 12 288 elements of the medium case (1 ulp).

**NaN through arithmetic.** In the quantise and dequantise rows there are 1 and 26
"NaN-encoding only" differences. The input NaN passed through a divide, min, max or
multiply. TNCA arithmetic returns the canonical NaN (ISA sec 8.2), and its
f32→bf16 narrowing writes `0x7fff`. The host propagates the payload and sign
(`0xff`, `0x7fc0`, `0xfff00000`). Values are never affected.

## Hardware/ISA semantic mismatches found

1. **FP8 conversion NaNs.**
   - `vf.cvt` drops the sign of a NaN narrowed to FP8.
   - It canonicalises an FP8 NaN widened to f32/f16/bf16.
   - torch keeps the sign, and the payload where the format has room.
   - Fixed generically in the compiler, as above.
   - Every other value matches torch, including saturation and inf. Proven for
     all 2^32 binary32 inputs.
2. **NaN canonicalisation in arithmetic and wide narrowing.** This is an ISA
   property and was left as it is. It affects only the encoding of NaN results that
   pass through arithmetic, or an f32→bf16/f16 conversion of a non-FP8 value.
   Rewriting it would change the 113 frozen points' code.
3. **The transfer engine's transposed write handles 16-bit elements only**
   (`rtl/core/tnca_tensor.v`, `S_TSETUP`).
   - `torch._scaled_mm` requires a column-major B, and a linear weight is
     `w.t()`. The FP8 B operand therefore cannot be one transposed transfer the
     way an FP16 one is. It is 16 gathers and 16 byte stores per K step.
   - Model cycles, 64×96×128 (uncalibrated; `probes/perf-cand.log` before, `probes/perf-cand2.log` after):
     - FP8 `_scaled_mm`: 182 979 before the operand changes, 70 870 after.
     - bf16 mm with column-major B: 39 771.
   - At 128×192×384: 2 164 072 → 778 230 against 446 224.
   - An 8-bit transposed write (or a byte pack/shuffle) in the engine would close
     the rest.
4. **One MMA input format per command** (ISA sec 9C `mod[3:2]`). e4m3×e5m2 is an
   exact FP16 MMA of the converted operands for one 16-deep K block. Deeper mixed
   dots are refused with a diagnostic, never miscompiled.
5. **Upstream Inductor limits lifted for tnca.**
   - No direct FP8→FP8 cast: lowered through binary32.
   - No scaled-mm config for K ≤ 16, which upstream refuses because of a CUDA
     Triton crash: K ≤ 16 is one masked MMA fragment here.
   - `scale_result` ignored by `tuned_scaled_mm` (an upstream bug: every
     template path drops it): applied on tnca.

## Does `_scaled_mm` emit the FP8 MMA? Yes.

Statically, each scaled-mm kernel's K loop holds exactly one
`m.mma.async.k16.e4m3` (or `.e5m2`) and no FP16 or BF16 MMA. The mixed case and
the weight-only linear are the expected exceptions: `.f16` and `.bf16`.

Dynamically, the model's `core_*_mma_commands` counters equal
⌈M/BM⌉·⌈N/BN⌉·⌈K/BK⌉·BN/16 in every case. Examples:

- 24 for 32×48×64
- 2 304 for 128×192×384
- 195 for 40×72×200
- 2 for K=8

From `asm/scaled_mm_e4m3_tensorwise_32x48x64_-_bf16_bias/triton_tem_fused__scaled_mm_0.s` in the archive:

```
    m.zero       a0                              // a zero tile carried round the loop
for_5:
    t.ld.async.b w0, d0, s18, s15                // A: one transfer (fp8 TensorMap)
    v.ld.gather.b v4, [s3, v1]                   // B: column-major, gathered ...
    v.st.b       [s12, #0], v4                   //    ... and packed (x16)
    m.mma.async.k16.e4m3 a0, v60, v56, s12       // FP8 MMA, accumulated in place
    br.nz        s17, for_5
endfor_6:
    a.vread      v2, a0, s1                      // epilogue: acc * (sa*sb) + bias
    vf.mul       v3, v0, v1
    vf.mul       v4, v2, v3
    ...
    vf.cvt.f32.bf16 v4, v2
```

## No regression

- **Default compiled code of the 113 original points is unchanged.**
  - `default_change_audit.py run` on the r41s1 tree (`files:6da4bd71…`, a
    `git archive` of `21d09f3`) and on T3.
  - Then `compare`: **113 unchanged, 0 changed**. Code text and launch structure
    are identical for every point (`default_change_audit_fp8.json`).
  - Earlier candidates gave the same 113/113: T2, and the tree with only the cast
    and `scaled_mm` changes.
- **`make compiler` on T3**: generated-file checks pass; `compiler-unit` **126
  tests pass, 0 skipped** (113 before plus 13 new, with the RTL bench present);
  `compiler-ir` **6/6 pass**. Run on GB10 with the Torch/Triton interpreter and
  tnca-opt built against LLVM/MLIR 20.
- **Existing fragment and bitwise tests.** These ran on T2, whose compiled
  fp16/bf16 code is the same as T3's.
  - The Stage 1 bitwise K-order check (`fragment_case.py --bitwise` and
    `compare_fragment_bits.py`) passes on the model: **60 rows bitwise
    identical**. That covers fp16/bf16 × plain/strided/cancelling inputs ×
    16- and 16x32-column fragments. Another K order would differ on 23 (blocked)
    or 27 (reversed) of those rows.
  - On the single-core RTL it is the same check as Stage 1: **16 rows bitwise
    identical**, bf16 strided, fp16 plain and fp16 cancelling, matching Stage 1's 16
    rows. Another K order would differ on 3 (blocked) or 7 (reversed) of those rows.
  - `test_fragment_bits` unit: 4/4 OK.
  - `make compiler-rtl`: **68 tests pass**.
  - `make triton-e2e`: **7 tests pass**.

## Reproduce (GB10)

```sh
T=~/r41-fp8-audit/cand3      # T3, this worktree
export TNCA_TLM_BINARY=...   # tools/performance_harness/systemc build (README there)
$TORCH_PYTHON $T/tools/performance_harness/fp8_cases.py --out OUT [--quick] [--group G]
TNCA_BACKEND=sim-file $TORCH_PYTHON $T/tools/performance_harness/fp8_cases.py --out OUT --quick
TNCA_GEMM_TILE=16,32,32 TNCA_DEADLINE=50000000 $TORCH_PYTHON $T/tools/performance_harness/fp8_cases.py --out OUT --group scaled_mm --quick
$TORCH_PYTHON $T/tools/performance_harness/fp8_numeric_exhaustive.py --out exhaustive.json
PYTHONPATH=$T/python:$T/compiler/python $TORCH_PYTHON $T/compiler/test/test_fp8_convert.py
PYTHONPATH=$T/python:$T/compiler/python $TORCH_PYTHON $T/compiler/test/test_fp8_mma.py
```

The torch environment on GB10 has an editable `tnca` install that points at
another checkout. `PYTHONPATH` must name this tree first, or the tests import the
wrong `tnca`. `fp8_cases.py` puts its own tree first.

## Full model run

`systemc-tlm-grid`, T3 (`files:5bb366a81f7d…`), torch 2.14.0, triton 3.8.0, CPU capability SVE128: **100 pass, 0 fail, 0 error**. Model results are uncalibrated predictions; the cast and GEMM *values* are the model's executed arithmetic, which is checked bit for bit against the RTL.

| group | case | status | result |
|---|---|---|---|
| casts | f32->e4m3 1-D (structured) | pass | 170258 elements, 0 bit mismatches |
| casts | f32->e5m2 1-D (structured) | pass | 170258 elements, 0 bit mismatches |
| casts | bf16->e4m3 1-D (exhaustive) | pass | 65536 elements, 0 bit mismatches |
| casts | bf16->e5m2 1-D (exhaustive) | pass | 65536 elements, 0 bit mismatches |
| casts | f16->e4m3 1-D (exhaustive) | pass | 65536 elements, 0 bit mismatches |
| casts | f16->e5m2 1-D (exhaustive) | pass | 65536 elements, 0 bit mismatches |
| casts | e4m3->f32 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e4m3->bf16 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e4m3->f16 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e5m2->f32 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e5m2->bf16 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e5m2->f16 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | f32->e4m3 2-D transposed read (64, 96) | pass | 6144 elements, 0 bit mismatches |
| casts | f32->e5m2 2-D transposed read (64, 96) | pass | 6144 elements, 0 bit mismatches |
| casts | f16->e4m3 2-D transposed read (64, 96) | pass | 6144 elements, 0 bit mismatches |
| casts | bf16->e5m2 2-D transposed read (64, 96) | pass | 6144 elements, 0 bit mismatches |
| casts | e4m3->f32 2-D transposed read (32, 96) | pass | 3072 elements, 0 bit mismatches |
| casts | e5m2->bf16 2-D transposed read (32, 96) | pass | 3072 elements, 0 bit mismatches |
| casts | e4m3->f16 2-D transposed read (32, 96) | pass | 3072 elements, 0 bit mismatches |
| casts | eager device copy f32->e4m3 | pass | contiguous: 16384 elements, 0 bit mismatches; strided view: 12282 elements, 0 bit mismatches |
| casts | eager device copy f32->e5m2 | pass | contiguous: 16384 elements, 0 bit mismatches; strided view: 12282 elements, 0 bit mismatches |
| casts | eager device copy bf16->e4m3 | pass | contiguous: 13108 elements, 0 bit mismatches; strided view: 9822 elements, 0 bit mismatches |
| casts | eager device copy e4m3->f32 | pass | contiguous: 1027 elements, 0 bit mismatches; strided view: 762 elements, 0 bit mismatches |
| casts | eager device copy e5m2->f16 | pass | contiguous: 1027 elements, 0 bit mismatches; strided view: 762 elements, 0 bit mismatches |
| casts | eager device copy e4m3->bf16 | pass | contiguous: 1027 elements, 0 bit mismatches; strided view: 762 elements, 0 bit mismatches |
| casts | e4m3->e5m2 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| casts | e5m2->e4m3 1-D (all 256 patterns) | pass | 1027 elements, 0 bit mismatches |
| plumbing | host/device copies and views (e4m3) | pass | 9 checks, mismatching: none |
| plumbing | host/device copies and views (e5m2) | pass | 9 checks, mismatching: none |
| plumbing | compiled slice + cat dim 0 (e4m3) | pass | 520 elements, 0 bit mismatches |
| plumbing | compiled cat dim 1 (e5m2) | pass | 4176 elements, 0 bit mismatches |
| plumbing | compiled narrow + transpose + contiguous (e4m3) | pass | 1386 elements, 0 bit mismatches |
| plumbing | clamp-then-cast quantise f32, static scale (e4m3) | pass | 6144 elements, 1 bit mismatches (1 NaN-encoding only) |
| plumbing | clamp-then-cast quantise f32, static scale (e5m2) | pass | 6144 elements, 1 bit mismatches (1 NaN-encoding only) |
| plumbing | clamp-then-cast quantise bf16, static scale (e4m3) | pass | 6144 elements, 1 bit mismatches (1 NaN-encoding only) / vs eager_bf16_intermediates: 6144 elements, 37 bit mismatches (1 NaN-encoding only) |
| plumbing | clamp-then-cast quantise f32, rowwise scale (e4m3) | pass | 6144 elements, 1 bit mismatches (1 NaN-encoding only) |
| plumbing | amax-derived scale, then quantise (e4m3) | pass | 6144 elements, 0 bit mismatches |
| plumbing | dequantise w.to(bf16) * s (e4m3) | pass | 3840 elements, 26 bit mismatches (26 NaN-encoding only) |
| plumbing | dequantise w.float() * s (e4m3) | pass | 3840 elements, 26 bit mismatches (26 NaN-encoding only) |
| plumbing | dequantise w.to(fp16) * s (e5m2) | pass | 3840 elements, 0 bit mismatches |
| plumbing | dequantise per-channel w.to(bf16) * s[:, None] (e4m3) | pass | 3840 elements, 0 bit mismatches |
| scaled_mm | e4m3 tensorwise 32x48x64 -> bf16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> bf16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> f16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> f16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> f32 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1217 bit mismatches (max ulp 236); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> f32 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1017 bit mismatches (max ulp 640); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> bf16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> bf16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> f16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> f16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> f32 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1224 bit mismatches (max ulp 3070); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 32x48x64 -> f32 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1019 bit mismatches (max ulp 1024); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> bf16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> bf16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> f16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> f16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> f32 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1277 bit mismatches (max ulp 13976); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 tensorwise 32x48x64 -> f32 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 989 bit mismatches (max ulp 16384); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> bf16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> bf16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> f16 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> f16 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> f32 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 1212 bit mismatches (max ulp 184); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e5m2 rowwise 32x48x64 -> f32 + bias | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 930 bit mismatches (max ulp 512); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 16x16x16 -> bf16 + bias | pass | vs per-K reference: 256 elements, 0 bit mismatches; vs torch CPU: 256 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 1 (expected 1) |
| scaled_mm | e4m3 rowwise 16x16x16 -> bf16 + bias | pass | vs per-K reference: 256 elements, 0 bit mismatches; vs torch CPU: 256 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 1 (expected 1) |
| scaled_mm | e4m3 tensorwise 5x16x16 -> bf16 + bias | pass | vs per-K reference: 80 elements, 0 bit mismatches; vs torch CPU: 80 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 1 (expected 1) |
| scaled_mm | e4m3 rowwise 5x16x16 -> bf16 + bias | pass | vs per-K reference: 80 elements, 0 bit mismatches; vs torch CPU: 80 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 1 (expected 1) |
| scaled_mm | e4m3 tensorwise 16x32x8 -> bf16 + bias | pass | vs per-K reference: 512 elements, 0 bit mismatches; vs torch CPU: 512 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 2 (expected 2) |
| scaled_mm | e4m3 rowwise 16x32x8 -> bf16 + bias | pass | vs per-K reference: 512 elements, 0 bit mismatches; vs torch CPU: 512 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 2 (expected 2) |
| scaled_mm | e4m3 tensorwise 1x64x64 -> bf16 + bias | pass | vs per-K reference: 64 elements, 0 bit mismatches; vs torch CPU: 64 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 16 (expected 16) |
| scaled_mm | e4m3 rowwise 1x64x64 -> bf16 + bias | pass | vs per-K reference: 64 elements, 0 bit mismatches; vs torch CPU: 64 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 16 (expected 16) |
| scaled_mm | e4m3 tensorwise 17x24x40 -> bf16 + bias | pass | vs per-K reference: 408 elements, 0 bit mismatches; vs torch CPU: 408 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 12 (expected 12) |
| scaled_mm | e4m3 rowwise 17x24x40 -> bf16 + bias | pass | vs per-K reference: 408 elements, 0 bit mismatches; vs torch CPU: 408 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 12 (expected 12) |
| scaled_mm | e4m3 tensorwise 33x48x80 -> bf16 + bias | pass | vs per-K reference: 1584 elements, 0 bit mismatches; vs torch CPU: 1584 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 45 (expected 45) |
| scaled_mm | e4m3 rowwise 33x48x80 -> bf16 + bias | pass | vs per-K reference: 1584 elements, 0 bit mismatches; vs torch CPU: 1584 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 45 (expected 45) |
| scaled_mm | e4m3 tensorwise 64x96x128 -> bf16 + bias | pass | vs per-K reference: 6144 elements, 0 bit mismatches; vs torch CPU: 6144 elements, 1 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 192 (expected 192) |
| scaled_mm | e4m3 rowwise 64x96x128 -> bf16 + bias | pass | vs per-K reference: 6144 elements, 0 bit mismatches; vs torch CPU: 6144 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 192 (expected 192) |
| scaled_mm | e4m3 tensorwise 40x72x200 -> bf16 + bias | pass | vs per-K reference: 2880 elements, 0 bit mismatches; vs torch CPU: 2880 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 195 (expected 195) |
| scaled_mm | e4m3 rowwise 40x72x200 -> bf16 + bias | pass | vs per-K reference: 2880 elements, 0 bit mismatches; vs torch CPU: 2880 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 195 (expected 195) |
| scaled_mm | e4m3 tensorwise 128x192x384 -> bf16 + bias | pass | vs per-K reference: 24576 elements, 0 bit mismatches; vs torch CPU: 24576 elements, 5 bit mismatches (max ulp 1); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 2304 (expected 2304) |
| scaled_mm | e4m3 rowwise 128x192x384 -> bf16 + bias | pass | vs per-K reference: 24576 elements, 0 bit mismatches; vs torch CPU: 24576 elements, 13 bit mismatches (max ulp 3); torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 2304 (expected 2304) |
| scaled_mm | e5m2 rowwise 33x40x48 -> f16 + bias | pass | vs per-K reference: 1320 elements, 0 bit mismatches; vs torch CPU: 1320 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 27 (expected 27) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> bf16 + bias use_fast_accum=True | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> e4m3 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 tensorwise 32x48x64 -> e4m3 scale_result=0.37 | pass | vs per-K reference: 1536 elements, 0 bit mismatches; vs torch CPU: 1536 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 24 (expected 24) |
| scaled_mm | e4m3 rowwise 33x40x48 -> e5m2 scale_result=4.0 | pass | vs per-K reference: 1320 elements, 0 bit mismatches; vs torch CPU: 1320 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 27 (expected 27) |
| scaled_mm | e5m2 tensorwise 17x24x40 -> e5m2 scale_result=0.125 | pass | vs per-K reference: 408 elements, 0 bit mismatches; vs torch CPU: 408 elements, 0 bit mismatches; torch CPU = scale-first emulation: 0 mismatches; FP8 MMA static 1, dynamic 12 (expected 12) |
| scaled_mm | e4m3 x e5m2 tensorwise 16x16x32 -> bf16 | pass | vs per-K reference: 256 elements, 0 bit mismatches; MMA {'triton_tem_fused__scaled_mm_0: m.mma.async.k16.f16': 1} |
| linear | static W8A8 x(16, 64) w(32, 64) | pass | q: 1024 elements, 0 bit mismatches; y: 512 elements, 0 bit mismatches; FP8 MMA static 1, dynamic 8; vs torch CPU _scaled_mm on the same q: 512 elements, 0 bit mismatches; vs eager CPU: q 1024 elements, 39 bit mismatches (max 1 FP8 ulp; eager q = bf16-rounded quotient: 0 mismatches), y 512 elements, 338 bit mismatches (max abs 0.109 of 4.03) |
| linear | weight-only FP8 x(16, 64) w(32, 64) | pass | y: 512 elements, 0 bit mismatches; BF16 MMA static 1, dynamic 8; vs eager CPU: 512 elements, 0 bit mismatches |
| linear | static W8A8 x(7, 40) w(24, 40) | pass | q: 280 elements, 0 bit mismatches; y: 168 elements, 0 bit mismatches; FP8 MMA static 1, dynamic 6; vs torch CPU _scaled_mm on the same q: 168 elements, 0 bit mismatches; vs eager CPU: q 280 elements, 13 bit mismatches (max 1 FP8 ulp; eager q = bf16-rounded quotient: 0 mismatches), y 168 elements, 96 bit mismatches (max abs 0.0469 of 2.59) |
| linear | weight-only FP8 x(7, 40) w(24, 40) | pass | y: 168 elements, 0 bit mismatches; BF16 MMA static 1, dynamic 6; vs eager CPU: 168 elements, 0 bit mismatches |
| linear | static W8A8 x(33, 72) w(40, 72) | pass | q: 2376 elements, 0 bit mismatches; y: 1320 elements, 0 bit mismatches; FP8 MMA static 1, dynamic 45; vs torch CPU _scaled_mm on the same q: 1320 elements, 0 bit mismatches; vs eager CPU: q 2376 elements, 86 bit mismatches (max 1 FP8 ulp; eager q = bf16-rounded quotient: 0 mismatches), y 1320 elements, 771 bit mismatches (max abs 0.148 of 4.25) |
| linear | weight-only FP8 x(33, 72) w(40, 72) | pass | y: 1320 elements, 0 bit mismatches; BF16 MMA static 1, dynamic 45; vs eager CPU: 1320 elements, 0 bit mismatches |
| linear | static W8A8 x(64, 256) w(192, 256) | pass | q: 16384 elements, 0 bit mismatches; y: 12288 elements, 0 bit mismatches; FP8 MMA static 1, dynamic 768; vs torch CPU _scaled_mm on the same q: 12288 elements, 3 bit mismatches (max ulp 1); vs eager CPU: q 16384 elements, 634 bit mismatches (max 1 FP8 ulp; eager q = bf16-rounded quotient: 0 mismatches), y 12288 elements, 10273 bit mismatches (max abs 0.172 of 7) |
| linear | weight-only FP8 x(64, 256) w(192, 256) | pass | y: 12288 elements, 0 bit mismatches; BF16 MMA static 1, dynamic 768; vs eager CPU: 12288 elements, 2 bit mismatches (max ulp 1) |
| linear | static W8A8 x(2, 13, 96) w(80, 96) | pass | q: 2496 elements, 0 bit mismatches; y: 2080 elements, 0 bit mismatches; FP8 MMA static 1, dynamic 60; vs torch CPU _scaled_mm on the same q: 2080 elements, 1 bit mismatches (max ulp 1); vs eager CPU: q 2496 elements, 93 bit mismatches (max 1 FP8 ulp; eager q = bf16-rounded quotient: 0 mismatches), y 2080 elements, 1381 bit mismatches (max abs 0.141 of 4.56) |
| linear | weight-only FP8 x(2, 13, 96) w(80, 96) | pass | y: 2080 elements, 0 bit mismatches; BF16 MMA static 1, dynamic 60; vs eager CPU: 2080 elements, 0 bit mismatches |
