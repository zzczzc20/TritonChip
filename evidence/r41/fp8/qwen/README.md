# Qwen3-VL-2B-Instruct-FP8 and Qwen3.5-4B-FP8: runners, CPU references, TNCA stage requirements

R41 FP8 work, branch `r41-fp8`, 2026-10-22. Code (all new, in `tools/models/`):

* `fp8_blockwise_ref.py`: an independent reference of the fine-grained W8A8 FP8 linear, plus its validation.
* `qwen_fp8_lib.py`: the shared, format-keyed staged runner.
* `qwen3vl_run.py` and `qwen35_run.py`: thin model specs (repo, revision, class, prompt).
* `fetch_qwen_fp8.py`: the downloader.

All runs were on the GB10 build host: CPU with 8 threads and RSS below 10 GiB, plus two short checks on its CUDA GPU.
Nothing ran on TNCA, the FPGA host or the card.

## 1. Results

* **Models.** Both are pinned and sha256-verified (section 2). Both load through the official
  `<Class>.from_pretrained`, and their FP8 linears stay transformers' own `FP8Linear`:
  * Qwen3-VL: 196 FP8 linears;
  * Qwen3.5: 200 FP8 linears.

  Every loaded float8 weight and scale is byte-equal to the checkpoint (`*/ref_fp8/verify_weights.json`).
* **CPU reference, W8A8.** The prompt is "Describe this image in one sentence." with a fixed 256×256 image of a red
  circle, a blue square and a green triangle. Decoding is greedy, up to 32 new tokens, with a 256-slot StaticCache.
  * Qwen3-VL output: "A red circle, a blue square, and a green triangle are arranged on a white background." (20 tokens).
  * Qwen3.5 output: "A simple geometric composition featuring a red circle, a blue square, and a green triangle
    arranged against a white background." (26 tokens).
  * **The staged runner is bitwise identical to the official code, for both models.**
    * The vision stage equals `get_image_features`.
    * The prefill logits and all hidden states equal `model.forward`.
    * Every greedy token and every step's logits equal `model.generate()` with the same StaticCache.

    These are the `checks` in `*/ref_fp8/run.json`.
* **The reference is validated against the official Hub kernel itself.** The GB10 has a CUDA GPU (sm_121). There,
  transformers' `fp8_linear` dispatches to `kernels-community/finegrained-fp8` v4 (`matmul_2d`), bit for bit.
  * With the activation scale computed as `amax*fl32(1/448)`, the reference reproduces the kernel's activation scales
    bitwise. Its codes differ in 1 of about 800 k, and its bf16 outputs match in 99.99 % of elements.
  * The fp32 accumulator agrees to rel-L2 7e-8.
  * With IEEE `amax/448`, which is the literal source text and what the brief specified, about 0.1 % of activation codes
    flip, and the output moves by up to 5e-3 rel-L2 (section 3).
* **Dequantized BF16 against W8A8 is inside the W8A8 noise floor.**
  * Teacher-forced prefill hidden states: rel-L2 0.073 (Qwen3-VL) and 0.106 (Qwen3.5).
  * Any rounding-level change to the W8A8 reference moves them by 0.087–0.150, because dynamic FP8 activation
    quantization amplifies ulp changes into code flips.
  * So model-level comparisons cannot prove a device FP8 GEMM correct. The proof is `layer-check`: every FP8Linear
    alone, on its real input. There the CPU reference sits exactly at bf16 rounding of the exact W8A8 value
    (≤ 2.06e-3).
* **TNCA.** Every stage of both models captures with `torch.compile(fullgraph=True)`: vision, prefill and a
  fixed-bucket decode step. A real CPU Inductor compile runs the whole pipeline.
  * **Qwen3-VL fits every window**, including 4 GiB: peak 2.03 GiB.
  * **Qwen3.5 fits 6 GiB with 0.023 GiB to spare and 8 GiB comfortably; it does not fit 4 GiB.** Peak is 4.73 GiB
    (4.78 GiB as an upper bound) against the 4.80 GiB the 6 GiB window allows.
  * Blockers are in section 6.

## 2. Provenance and environment

| repo | revision | files (sha256 rechecked on GB10 after the restart; `environment/sha256_recheck.txt`) |
|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct-FP8 | `46485250d8854c0a9be4f1adbc67ca47e5bb6fa5` (2025-10-20) | `model-00001-of-00001.safetensors` 3,468,553,776 B `da14428e…e57c` |
| lovedheart/Qwen3.5-4B-FP8 | `752f65fd155dd34323cfc22d6143df1d48ce3c28` (2026-03-03) | shard 1: 3,300,663,624 B `35b7391d…a46b`; shard 2: 2,347,002,232 B `f2428aa0…fd5a`; `tokenizer.json` `5f9e4d49…cb42` |

The two repos are not gated, and their library is transformers.
* `environment/qwen_fp8_manifest.json` lists every file. The download used parallel range requests on the resolve
  URLs. All small files match the Hub's git blob ids.
* Software: torch 2.14.0+cu130 (torchenv, unmodified; the `pip freeze` md5 `4614c543…` is unchanged), transformers
  5.20.2, accelerate 1.15.0, kernels 0.17.2.
* The Hub kernel is `kernels-community/finegrained-fp8` v4, snapshot `0e5764aa`, kernel commit `fe2bb66`, cuda build
  only.

Two changes to `~/flux-venv` only (freeze diffs are in `environment/`):
1. **The earlier `kernels` install had broken transformers in flux-venv.** It pulled cryptography 50.0.2. The system
   pyOpenSSL 23.2 then failed to import (`GEN_EMAIL`). That broke `boto3` → `accelerate` → every
   `from_pretrained`, FLUX's runner included. Fix: pyOpenSSL 26.4.0 (requires cryptography>=49,<51), installed with
   `--no-deps`.
2. **torchvision 0.29.0+cu130**, the torch 2.14.0 pair. `AutoProcessor` for both models needs it: Qwen3VLProcessor
   requires the video processor even for images. Without it only the PIL image processor loads.

## 3. FP8 handling (generic, keyed on `quantization_config.quant_method == "fp8"`)

**What transformers' quantizer does on non-CUDA devices** (`quantizers/quantizer_finegrained_fp8.py`,
`validate_environment`; probe in `*/probe_quantizer.json`):
* **It forces `dequantize=True` when no CUDA/XPU is available.** If `dequantize` is false and both
  `torch.cuda.is_available()` and `is_torch_xpu_available()` are false, then:
  * a pre-quantized checkpoint gets the warning "Using FP8 quantized models requires a GPU or XPU, we will default to
    dequantizing the model to bf16…" and `quantization_config.dequantize = True`;
  * otherwise it raises `RuntimeError`.
* **The decision is keyed on the host, not on the target device.**
  * The GB10 has CUDA, so the quantizer keeps `FP8Linear` there (196 / 201) even for a CPU model. It only warns
    "loaded an FP8 model on CPU…".
  * On that host, an FP8Linear forward on CPU tensors fails inside the Triton launch.
  * With CUDA unavailable (patched in the probe only), every quantized layer comes back as a BF16 `nn.Linear`. Those
    weights are bitwise equal to `fp8_blockwise_ref.dequant_weight`, i.e. `(q.float()*s).to(bf16)`; all 196 and 200
    layers were checked.
* A GPU below compute capability 8.9 also forces dequantize. DeepGEMM is used only on SM90/SM100; on sm_121 it is not
  loadable, so `fp8_linear` goes to the Triton kernel.

**What the runner does, without editing transformers:**
* *Loading:*
  * `--weights fp8` sets the documented loading attribute `dequantize=False`; `--weights bf16` sets `True`.
  * During `from_pretrained`, the quantizer registry entries for `fp8` (and `mxfp8`) are a subclass whose only change
    is that the requested value survives `validate_environment` (`pinned_fp8_quantizer`). The result is therefore the
    same on any host.
* *Compute:* `FP8Linear.forward → fp8_linear → finegrained_fp8_linear → load_finegrained_fp8_kernel().matmul`.
  `--fp8-kernel` selects the object behind it:
  * `official`: transformers loads the Hub kernel itself; this is route (a), which the kernels agent makes run on tnca;
  * `reference`: `fp8_blockwise_ref.ReferenceKernel` is placed in transformers' own `_FINEGRAINED_FP8` slot; this is
    route (b).

  Neither choice involves model or layer names.

**The reference** (`fp8_blockwise_ref.py`; it never imports the Hub kernel):
* It mirrors `matmul_2d`'s dispatch: block-wise dynamic, block-wise static, tensor-wide, and MX refused.
* Per token and per 128-group: `s = amax/448`, then `q = (x/max(s,1e-12)).to(e4m3)` with RNE (torch's cast).
* Per K tile: a dot product of e4m3 values, then `*a_s`, then `*b_s`, accumulated in fp32 in tile order. The output
  is cast to the input dtype (bf16).
* Options:
  * `dot=exact`: the tile dot is computed exactly in float64 and rounded once. Every e4m3 value is a multiple of 2^-9,
    so a 128-term sum is exact in 53 bits. This is the default and is order-independent.
  * `dot=fp32`: a plain fp32 bmm. It needs no float64; this is the variant tnca can run.
  * `fma`: CUDA's contracted `acc + t*b_s`.
  * `scale=recip`: `amax*fl32(1/448)`, which is what Triton emits for `/448.0` on CUDA.
* The dequantize alternative: `dequant_weight`, then `F.linear`.

**Validation** (`fp8_ref/validate_{cpu,cuda}.json`):
1. The vectorized reference equals a literal transcription of `w8a8_block_dynamic_fp8_matmul_kernel`'s tile loop
   bit for bit in 12/12 cases. The transcription covers wrapped and masked tiles, adaptive BLOCK_M, the zero row, and
   ragged N=200/K=328.
2. **Against the Hub kernel on CUDA** (20 cases: synthetic, plus real layers of both checkpoints with fp32 and bf16
   scales):

| quantity vs Hub kernel | scale `ieee` (default) | scale `recip` |
|---|---|---|
| activation scales | 43 % bitwise (1 ulp otherwise) | **100 % bitwise** |
| e4m3 activation codes | ≥ 99.66 % | ≥ 99.9999 % (1 code in ~800 k) |
| fp32 accumulator, rel-L2 | 5.2e-3 | **7.5e-8** (5.3e-8 with `fma`) |
| bf16 output (transformers' call) | ≥ 85.2 % bitwise, rel-L2 ≤ 5.3e-3 | **≥ 99.99 % bitwise**, max 3 ulp |

* Triton lowers `/ 448.0` to a multiply by the rounded reciprocal. An elementwise check (`act_quant_*`) confirmed it,
  and confirmed that the fp32→e4m3 conversion is plain RNE.
* With bf16 data, a quotient often lands exactly on an e4m3 rounding midpoint, so a 1-ulp scale change flips codes.
* The remaining fp32 differences come from tensor-core accumulation inside a tile.
* `fp8_linear` (transformers' entry point) equals the kernel's `matmul_2d` bitwise.

3. **Dequantized BF16 against W8A8, per layer, on real activations** (`*/layer_check_cpu_eager.json`). The dequantized
   layer differs from exact W8A8 by 0.24–3.95 % rel-L2 (Qwen3-VL) and 0.44–3.30 % (Qwen3.5). It is within
   2.9e-3 / 2.9e-3 of the unquantized-activation product. In other words, the BF16 route is weight-only quantization;
   W8A8 adds the activation-quantization error.

## 4. CPU references and tolerance floors

Stage modules (`qwen_fp8_lib.py`) call the model's own code:
* **vision:** the vision tower. The grid tensors are precomputed by transformers' `vision_utils` helpers on the host.
* **prefill:** the whole `ForConditionalGeneration`, with image features passed through the official
  `mm_encoder_outputs` input. Positions come from the model's own `_prepare_position_ids_for_generation`. It writes a
  `StaticCache(bucket=256)`; Qwen3.5's 24 linear-attention layers get conv and recurrent states.
* **decode:** one token on the same cache: shape-static, one graph for all steps.

Argmax and stopping run on the host.

| model / weights | vision | prefill (tokens) | decode step | peak RSS | tokens |
|---|---|---|---|---|---|
| Qwen3-VL W8A8 (ref, `dot=exact`) | 1.08 s | 4.80 s (81) | 1.00 s | 4.8 GiB | 20 |
| Qwen3-VL BF16 dequant | 0.81 s | 0.95 s | 0.47 s | 8.0 GiB | 20 (same text) |
| Qwen3.5 W8A8 (ref) | 1.43 s | 23.7 s (85) | 3.9–6.9 s | 7.4 GiB | 26 |
| Qwen3.5 BF16 dequant | 0.67 s | 3.86 s | 0.90 s | 14.4 GiB | 30 ("…arranged in a triangular formation against…") |

* CPU time says nothing about TNCA; float64 tile dots make the W8A8 reference slow.
* Peak RSS includes the mmap'd checkpoint. The BF16 runs hold the dequantized weights.
* GPU runs of the official path (`*/official_cuda`, unmodified `generate()` with the Hub kernel): Qwen3-VL "A simple
  geometric illustration features a red circle, a blue square, and a green triangle arranged on a white background.",
  Qwen3.5 the same text as the reference except "on" for "against". The CUDA peaks were 3.0 GiB and 5.4 GiB.

**Teacher-forced comparisons against `ref_fp8`.** Each stage takes the reference's inputs: decode starts from the
reference prefill cache and is fed the reference tokens. Files: `*/compare/*.json`, summary in `*/compare/summary.json`.

| change | Qwen3-VL hidden (max over layers) / last-token logits / decode logits (max) / decode argmax agree | Qwen3.5, same columns |
|---|---|---|
| 4 threads instead of 8 | **0 / 0 / 0 / 19 of 19** (bitwise) | **0 / 0 / 0 / 25 of 25** |
| `dot=fp32` instead of exact | 0.087 / 0.048 / 0.062 / 19 of 19 | 0.128 / 0.107 / 0.100 / 24 of 25 |
| scale `recip` (the CUDA rounding) | 0.095 / 0.054 / 0.058 / 19 of 19 | 0.150 / 0.143 / 0.139 / 24 of 25 |
| `fma` | 0.092 / 0.056 / 0.070 / 19 of 19 | – |
| CPU Inductor, fullgraph (vision 0.043 / 0.045) | 0.096 / 0.066 / 0.079 / 19 of 19 | 0.151 / 0.153 / 0.152 / 24 of 25 |
| Inductor + `emulate_precision_casts` | 0.090 / 0.056 / 0.060 / 19 of 19 | – |
| **BF16 dequantized weights** | **0.073 / 0.049 / 0.048 / 19 of 19** | **0.106 / 0.086 / 0.094 / 24 of 25** |
| official kernel on CUDA (chained) | 0.116 / 0.074 / – / first token equal | 0.230 / 0.160 / – / 18 tokens equal |

* The reference is deterministic: thread count changes nothing.
* **Every rounding-level change lands at 0.09–0.15** (hidden) and 0.05–0.15 (logits). The FP8 activation
  quantization amplifies ulp changes. The official `generate()` with a DynamicCache instead of the StaticCache shows
  this too:
  * 3–6 % logits rel-L2 for W8A8 against 1–2 % for BF16;
  * Qwen3.5 changes a word at token 19.
* **BF16 dequantization sits inside that floor**, so on these prompts W8A8 against BF16 is not distinguishable from
  rounding noise.
* Qwen3.5's single 24/25 step is a near-tie ("against / on / in") at step 18. Chained, every variant still gives the
  same scene.

**Acceptance on TNCA** (same three levels as FLUX):
1. **Op level, which carries the proof.** Run `layer-check --device tnca`. Every FP8Linear runs alone on its captured
   reference input and is compared with the exact W8A8 value.
   * CPU eager: 196 and 200 layers bitwise equal to the reference. Their error equals bf16 rounding of exact
     (max 2.06e-3 / 2.01e-3; ratio 1.0000000).
   * CPU Inductor with `dot=fp32`: rel-L2 ≤ 3.0e-5 against eager.
   * The Hub kernel on CUDA: ≤ 4.5e-3 / 3.5e-3 against exact-ieee. That is exactly the scale-rounding effect
     (exact-recip vs exact-ieee is 4.2e-3 / 3.1e-3).
   * Tolerance: about 1.1× bf16 rounding of exact against the matching scale rounding, or about 2.5× against the
     other.
2. **Stage level, teacher-forced:**
   * vision rel-L2 ≤ 0.05;
   * hidden states ≤ 0.12 (Qwen3-VL) / ≤ 0.18 (Qwen3.5);
   * last-token logits ≤ 0.08 / ≤ 0.17;
   * decode-step argmax agreement ≥ 95 %.
3. **End to end:** the same description of the image. Bitwise equality is not expected.

Artifacts:
* `*/ref_fp8/tensors.safetensors` (12.9 MB and 17.3 MB) holds:
  * `input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values`, `image_grid_thw`, `position_ids_prefill`;
  * `vision_embeds` (and Qwen3-VL's three `vision_deepstack_*`);
  * `prefill_hidden_00..` (29 / 33 layers);
  * `prefill_logits_last`, `generated_tokens`, `decode_logits_top16_*`, `decode_logits_step0`.
* `run.json` has sha256, mean, std and finiteness of every tensor and of the full logits, plus settings, versions,
  per-stage seconds and peak RSS (VmHWM, reset per stage), and the checks.
* On GB10 only (`~/qwen-work/out/{vl,q35}/`; too large to commit, needed for `--teacher-force`):
  * `logits.safetensors`: full prefill and decode logits, 30 MB / 55 MB;
  * `cache_prefill.safetensors`: the cache state after prefill, 9 MB / 55 MB.

## 5. Per-stage TNCA requirements (`*/plan.json`, `plan --ref-dot fp32`)

`plan` runs `torch.compile(fullgraph=True)` with a recording backend: dynamo, then AOT with Inductor's decomposition
table. It executes the graph on CPU.
* Weights are what the captured graph actually reads; the LM stages leave the vision tower off the device.
* KV/state is the StaticCache allocation for the bucket of 256.
* Activations are the eager peak of storages created during the stage. The bound uses the unfused graph peak.
* Windows: usable ≥ 1.25 × peak, i.e. peak ≤ 3.2 / **4.8 / 6.4** GiB for 4 / **6 / 8** GiB.

| stage | weights GiB | KV/state GiB | act. GiB (graph) | **peak GiB** (bound) | 6 GiB | 8 GiB | 4 GiB (ref.) | FLOPs (FP8 GEMM share) |
|---|---|---|---|---|---|---|---|---|
| VL vision (256 patches → 64 tokens) | 0.758 (bf16) | – | 0.011 (0.013) | **0.77** (0.77) | yes | yes | yes | 174.8 G (0) |
| VL prefill (81 tokens) | 1.893 (1.31 fp8 + 0.58 tied embedding/lm_head) | 0.027 | 0.083 (0.106) | **2.00** (2.03) | yes | yes | yes | 230.4 G (228.3 G) |
| VL decode step (ctx 81, bucket 256) | 1.893 | 0.027 | 0.048 (0.075) | **1.97** (2.00) | yes | yes | yes | 3.50 G (2.82 G) |
| Q3.5 vision | 0.621 | – | 0.010 (0.012) | **0.63** (0.64) | yes | yes | yes | 165.4 G (0) |
| Q3.5 prefill (85 tokens) | 4.514 (3.31 fp8 + 1.18 tied embedding + bf16 rest) | 0.056 (attention KV + 24×2 MiB fp32 recurrent state) | 0.154 (0.206) | **4.73** (4.78) | **yes, 0.02 GiB margin** | yes | no | 623.6 G (606.1 G) |
| Q3.5 decode step | 4.514 | 0.056 | 0.089 (0.145) | **4.66** (4.72) | yes, 0.08 GiB margin | yes | no | 8.4 G (7.1 G) |

Code, descriptors and reserved ranges are not included, so the 6 GiB verdict for Qwen3.5 is optimistic: it has
almost no margin. Per generated token, on top of one prefill, a decode step costs 3.5 GFLOP for Qwen3-VL and 8.4 GFLOP
for Qwen3.5.

**Ops and dtypes per stage** (the ATen graph after Inductor decompositions; counts in `plan.json` `ops`):
* **dtypes:** vision bf16, fp32, int32, int64. Prefill and decode bf16, fp32, **float8_e4m3fn**, int64, bool. There is
  **no float64** with `--ref-dot fp32`; float64 appears only in the CPU-only `exact` mode.
* **FP8 linear, route (b), per layer:**
  * activation quantization: `abs`, `amax`, `div` (2), `clamp_min`, `convert fp32→float8_e4m3fn`;
  * `convert float8→fp32` (activations and weights), `view`/`permute` of float8;
  * **one fp32 `bmm`** of shape [K/128, M, 128] × [K/128, 128, N];
  * a K-tile chain of `select`/`mul`/`add` in fp32, then `convert fp32→bf16`.

  Counts: 196 (Qwen3-VL) and 200 (Qwen3.5) per prefill and per decode graph. Route (a) replaces this subgraph with one
  Triton kernel per layer (the kernels agent): an fp8 `tl.dot` with fp32 scaling.
* **Qwen3-VL text, and Qwen3.5 full attention:**
  * `_scaled_dot_product_flash_attention` (28 / 8), bf16 in, fp32 logsumexp; causal over the 256-slot bucket with a
    bool mask and `where`;
  * KV update as `index_put` (bf16, int64 index);
  * RMSNorm (`pow`, `mean`, `rsqrt` in fp32), M-RoPE `cos`/`sin` fp32 from `iota`;
  * SiLU MLP, `embedding` (bf16, int64), tied lm_head `mm` bf16;
  * Qwen3.5 adds an output gate (`sigmoid`).
* **Qwen3.5 Gated DeltaNet** (24 layers; transformers' torch fallbacks, since fla and causal_conv1d are not
  installed):
  * depthwise `convolution` bf16 (kernel 4) and `constant_pad_nd`;
  * `cumsum`, `exp`, `softplus`/`sigmoid`, l2norm;
  * chunked `bmm` fp32;
  * **`linalg_solve_triangular` fp32 ×48 per prefill**;
  * the recurrent state update in fp32 on decode;
  * the bf16 `in_proj_a/b` stay unquantized, as `mm`.
* **Vision (both models):**
  * `convolution` (Conv3d patch embedding) bf16, `addmm` bf16 ×104 / ×98;
  * SDPA ×24, LayerNorm `var_mean`, GELU-tanh;
  * bilinear position-embedding gather (`embedding` + weights).

## 6. Blockers and requirements for the TNCA stack

| # | item | status, options |
|---|---|---|
| B1 | **Qwen3.5 capacity.** Peak 4.73 GiB (bound 4.78) against 4.80 allowed in 6 GiB: about 0.5 % margin before code and descriptors. 4 GiB fails. | 8 GiB is safe. For 6 GiB: keep the 1.18 GiB embedding table off the device (gather on the host; the lm_head needs it, so this only works with a separate lm_head copy or a split stage), or shrink the bucket. Qwen3-VL fits all three windows. |
| B2 | **The FP8 GEMM itself.** | Route (a): the Hub Triton kernel on tnca (kernels agent). Route (b) runs today as fp8→fp32 converts plus an fp32 bmm (no FP8 MMA, about 2× the flops in fp32, activation transient [K/128, M, N] fp32). The packed FP8 MAC is not assumed. |
| B3 | **Activation-scale rounding must be specified.** IEEE `amax/448` and CUDA's `amax*fl(1/448)` differ by up to 4.5e-3 rel-L2 per layer, because bf16 data hits e4m3 midpoints. | Whatever tnca's Triton lowers `/448.0` to, judge it with `layer-check` against the reference run with the matching `--ref-scale`. Both are implemented. |
| B4 | **Host syncs and data-dependent shapes**, captured by dynamo as unbacked symbols. | Vision: `cu_seqlens.tolist()` in the SDPA path, giving 72 `_local_scalar_dense` + `split_with_sizes` + 216 `_assert_scalar` per image. Prefill: `masked_scatter` of image features, a `sum(bool)` / `_local_scalar_dense` check, and Qwen3-VL deepstack `index.Tensor` / `index_put` with a bool mask (data-dependent row count) ×3. **Decode has none.** tnca needs unbacked-symint support or host round trips here. A one-image fixed shape could constant-fold them, but that is a model-code change and was not done. |
| B5 | **Qwen3.5 linear attention needs `linalg_solve_triangular` (fp32, 64×64 chunks, ×48), `cumsum`, depthwise `conv1d`, `exp`.** | Not expressible on tnca today unless lowered. transformers' export path replaces the triangular solve with forward substitution (`is_torchdynamo_exporting`), but that path is not taken under `torch.compile`. |
| B6 | **Acceptance cannot be model-level only.** The W8A8 noise floor (0.09–0.15 hidden) is larger than W8A8 against BF16. | Use the three levels in section 4. The op-level `layer-check` is the proof. |
| B7 | **Checkpoint quirks.** | Qwen3.5's `model.safetensors.index.json` has no `metadata`, so transformers 5.20 `from_pretrained` raises `KeyError: 'metadata'`. The runner loads from a symlink view whose index adds `metadata.total_size` (weights untouched; `load.snapshot_view` in run.json). Qwen3.5's tied `lm_head` becomes an `FP8Linear` shell holding the BF16 embedding, with `weight_scale_inv` reported MISSING. That is harmless: `FP8Linear.forward` takes `F.linear` for 2-byte weights, and the runner counts it as BF16. The 7 `mtp.*` FP8 tensors (multi-token prediction) are not loaded by this class. |
| B8 | **Environment.** | flux-venv repaired (pyOpenSSL) and torchvision added (section 2). The tnca runner path (`--*-device tnca`, `use_tnca(--tnca-tree)`) was not run here. |

## 7. Reproduce (GB10: `HF_HOME=~/flux-hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 ~/flux-venv/bin/python -I`)

```sh
python tools/models/fetch_qwen_fp8.py
python tools/models/fp8_blockwise_ref.py validate --out validate_cpu.json
HF_HUB_OFFLINE=0 python tools/models/fp8_blockwise_ref.py validate --cuda --triton-pythonpath TREE/python TREE/compiler/python \
    --checkpoint VL.safetensors Q35-shard2.safetensors --layers <prefixes> <prefixes> --out validate_cuda.json
R=tools/models/qwen3vl_run.py        # or qwen35_run.py
python $R run --check-official --verify-weights --out ref_fp8           # W8A8 CPU reference
python $R run --weights bf16 --verify-weights --out ref_bf16            # dequantized BF16
python $R run --teacher-force ref_fp8 [--threads 4|--ref-dot fp32|--ref-scale recip|--ref-fma|--weights bf16|--cpu-compile inductor [--emulate-precision-casts]] --out tf_X
python $R compare tf_X ref_fp8 --out compare/tf_X_vs_ref_fp8.json
python $R layer-check --teacher-force ref_fp8 [--cpu-compile inductor --ref-dot fp32] --out layer_check.json
python $R plan --ref-dot fp32 --out plan.json
HF_HUB_OFFLINE=0 python $R probe-quantizer --tnca-tree TREE --out probe_quantizer.json
HF_HUB_OFFLINE=0 python $R layer-check --device cuda --fp8-kernel official --tnca-tree TREE --teacher-force ref_fp8 --out l.json
HF_HUB_OFFLINE=0 python $R official --device cuda --fp8-kernel official --static-cache --tnca-tree TREE --out official_cuda
# on TNCA (not run): each stage alone against the reference, then every FP8Linear alone
python $R run --teacher-force ref_fp8 --vision-device tnca --prefill-device tnca --decode-device tnca \
    --fp8-kernel official|reference --ref-dot fp32 --tnca-tree TREE --out tnca
python $R layer-check --device tnca --fp8-kernel official --teacher-force ref_fp8 --tnca-tree TREE --out tnca_layers.json
```
`HF_HUB_OFFLINE=0` is needed only where the Hub kernel is loaded. `kernels` resolves `version=4` to a revision online,
and the cache has no `refs/` for it.
