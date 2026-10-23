# FLUX.2 [klein] 4B FP8: runner, CPU reference, TNCA stage requirements

R41 FP8 work, branch `r41-fp8`, 2026-10-21. Code: `tools/models/` (`fp8_checkpoint.py`,
`test_fp8_checkpoint.py`, `flux2_klein_run.py`, `fetch_flux2_klein.py`). Everything here ran on
the GB10 build host's CPU (8 threads, RSS < 40 GB). Nothing ran on TNCA, the FPGA host or any card.

## 1. Results

* **Loader decision: diffusers cannot load this checkpoint, so a generic format-keyed loader is used.**
  diffusers 0.41.0 `Flux2Transformer2DModel.from_single_file` *raises* on the file
  (its FLUX.2 key converter `torch.chunk`s the 0-d `qkv.weight_scale`). No code in diffusers or
  transformers reads `_quantization_metadata`, and no diffusers quantizer covers this format.
  Bypassing the crash makes it worse: the float8 codes are cast to BF16 without `weight_scale`, so
  the weights come out ~1700x too large. diffusers neither keeps FP8 nor dequantizes it.
  `tools/models/fp8_checkpoint.py` reads only the metadata. It maps keys only through diffusers' own
  single-file converter and keeps float8 weights in an `FP8Linear` that calls `torch._scaled_mm`.
  It has no model or layer names.
* **CPU reference (256x256, 4 steps, guidance 1.0, seed 0, official prompt) is done for FP8
  (W8A8, CPU `torch._scaled_mm`) and BF16.** Both produce the expected image. On both paths the
  staged runner is **bitwise identical to the unmodified `Flux2KleinPipeline.__call__`**: every
  step's latents and the final image match exactly. FP8 vs BF16 image PSNR is 25.3 dB.
* **TNCA: all three stages capture with `torch.compile(fullgraph=True)`** (text-encoder forward,
  one denoising step, VAE decode) at 256 and 1024. A real Inductor compile on CPU found one
  library graph break, in transformers' lazy hook installation. It is fixed generically.
  * Capacity: only the VAE fits the 4 GiB B40 window under the R41 rule `C >= 1.25 H_peak`.
  * The FP8 transformer (3.79 GiB of weights) and the VAE fit an 8 GiB SystemC/TLM window at both
    resolutions, if SDPA stays fused or online.
  * The BF16 text encoder (7.49 GiB) fits neither window as a whole module.
  * Details and the op/dtype list are in section 5.
* **How to judge the TNCA output.** The CPU reference is bitwise reproducible, but a denoise step
  amplifies rounding-level changes to 5-12 % of its noise prediction. That is larger than the effect
  of a 10 % fault in one FP8 layer. Section 4.1 therefore sets per-stage teacher-forced tolerances
  plus an op-level `layer-check`; on CPU that check is exact to bf16 rounding.

![FP8](reference/fp8/image.png) ![BF16](reference/bf16/image.png)
(left: FP8 W8A8; right: BF16 transformer; same text encoder, VAE, noise and schedule)

## 2. Environment and provenance

| | |
|---|---|
| host | GB10 `gx10-6646`, aarch64, `torch.backends.cpu` capability SVE128, 8 threads, CPU only |
| torch | 2.14.0+cu130 from `~/tc_env/torchenv`, **unmodified** (`pip freeze` md5 `4614c543...` before and after) |
| venv | `~/flux-venv` = `~/tc_env/torchenv/bin/python -m venv --system-site-packages ~/flux-venv` |
| python | 3.12.3 |
| diffusers / transformers | **0.41.0 / 5.20.2** |
| accelerate / safetensors / huggingface_hub / tokenizers | 1.15.0 / 0.8.0 / 1.33.0 / 0.23.3 |
| numpy | 2.5.3 (from torchenv) |

The venv layering: torchenv is itself a venv, so `--system-site-packages` points at `/usr` and not at
torchenv. One `zz_torchenv.pth` line in flux-venv's site-packages adds torchenv's site-packages.
Installs were pinned with constraints `torch==2.14.0, numpy==2.5.3`, so pip changed nothing of
torchenv's. The full package list is in `environment/flux_venv_packages.txt`. torchenv's editable `tnca` install is a
`.pth` finder that is not active inside flux-venv. For a TNCA run the runner puts the tree's
`python/` (and `python/_staged`, `compiler/python`) first on `sys.path`, before anything imports
triton. torchenv's triton loads a `tnca` backend, and torch caches a failed triton probe.

| repo | revision (last modified) | files used |
|---|---|---|
| black-forest-labs/FLUX.2-klein-4b-fp8 | `5b4408e59397a4a37ccb46afe426d8ed86379441` (2026-02-24) | `flux-2-klein-4b-fp8.safetensors`, 4,070,624,520 B, sha256 `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6` |
| black-forest-labs/FLUX.2-klein-4B | `e7b7dc27f91deacad38e78976d1f2b499d76a294` (2026-02-24) | model_index, scheduler, text_encoder (2 shards `8c0506e7...`, `82f2bd83...`), tokenizer, transformer (BF16, `9f29f9ed...`, 7,751,109,744 B), vae (`ca70d220...`, 168,120,878 B) |

Every file, with its size and sha256 (LFS) or git blob id, is in `environment/hf_download_manifest.json`.
Not downloaded: the BFL-layout BF16 single file `flux-2-klein-4b.safetensors` and the sample images.
Both repos are Apache-2.0 and not gated.

How the download was done:
* The default cache `~/.cache/huggingface` on GB10 is root-owned (from an earlier docker run) and
  there is no sudo, so `HF_HOME=~/flux-hf` was used, on the same disk.
* Xet stalled at ~1 KB/s, and one HTTPS stream gave ~1 MB/s.
* `tools/models/fetch_flux2_klein.py` therefore fetches the official `huggingface.co/<repo>/resolve/<rev>/`
  URLs with 16 parallel range requests (~11 MB/s). It writes the standard cache layout and checks
  every LFS file against the sha256 the Hub API reports for the pinned revision.

## 3. Loading the FP8 checkpoint

### 3.1 The format, as read from the file

* 309 tensors:
  * 80 `F8_E4M3` weights (3.43 GiB);
  * 80 `weight_scale` and 80 `input_scale` tensors, 0-d F32 (static per-tensor W8A8);
  * 69 BF16 tensors (QK-norm scales, embedders, modulation, final layer; 0.36 GiB).
* `__metadata__._quantization_metadata` =
  `{"format_version": "1.0", "layers": {<name>: {"format": "float8_e4m3fn"}}}` for the same 80 layers.
  These are every double-block `img/txt_attn.qkv`, `*_attn.proj`, `*_mlp.0`, `*_mlp.2` and every
  single-block `linear1`, `linear2`.
* There are no biases. Input scales range from 0.0226 to 11.57.

### 3.2 What the official code does (`loader/official_loader_probe.json`, `flux2_klein_run.py probe-official-loader`)

1. `Flux2Transformer2DModel.from_single_file(<fp8 file>, config=<main repo>, subfolder="transformer")`
   raises `RuntimeError: chunk expects at least a 1-dimensional tensor`. The failure is at
   `single_file_utils.py:3946`, in `convert_flux2_double_stream_blocks`, which chunks every key
   containing `qkv` into three, and that includes the 0-d scales. The same converter's
   `.weight/.bias/.scale` filter would leave every `*.input_scale` key unconverted.
2. No `.py` file in diffusers 0.41.0 or transformers 5.20.2 mentions `_quantization_metadata`. The
   diffusers quantizers are auto-round, bitsandbytes 4/8-bit, gguf, modelopt, nunchaku_lite,
   quanto, sdnq and torchao. All of them need a `quantization_config` and their own layout.
3. Without a quantizer, `load_model_dict_into_meta` keeps float8 only when `hf_quantizer is not None`;
   otherwise it casts to `torch_dtype`. With the scale tensors removed (so the converter does not
   crash), `from_single_file` loads `transformer_blocks.0.attn.to_out.0.weight` as BF16 with
   absmax **448.0**, against 0.2656 in the official BF16 checkpoint: an RMS ratio of 1691, which is
   1/`weight_scale`.

So diffusers does not support the format natively. It neither keeps FP8 with `_scaled_mm` nor
dequantizes correctly: it refuses, and without the crash it would load wrong values silently. The
model card names ComfyUI and diffusers. This metadata layout looks like ComfyUI's mixed-precision
format; that was not checked here, and ComfyUI is not the official diffusers/transformers path.

### 3.3 Decision: `tools/models/fp8_checkpoint.py` (generic, keyed on the format)

* **Input:** only the metadata: its layers and formats (`float8_e4m3fn`, `float8_e5m2`) and
  `format_version` 1.0. Anything else is refused with `UnsupportedQuantizedCheckpoint`.
  Float8 tensors not described by the metadata are refused, and so are scales without a layer.
* **Key mapping:** only the model's own diffusers single-file converter,
  `SINGLE_FILE_LOADABLE_CLASSES[cls]["checkpoint_mapping_fn"]`, the same function
  `from_single_file` uses.
  * It runs once, on a probe copy in which each float8 weight is replaced by a zero-copy int64
    "row tag" view (`magic | source index | row`) of the same shape.
  * The tags show which rows of which source each converted tensor holds. The float8 rows and their
    scales follow the tags. For example, the fused QKV becomes `to_q/to_k/to_v`, each taking the
    source's per-tensor weight and input scale.
  * It refuses a converter that changes tag values (arithmetic or casts), mixes rows of two scaled
    sources, mixes columns, or does not place every row exactly once.
* **Module:** each converted `X.weight` in float8 must land on an `nn.Linear` named `X`, which is
  replaced by `FP8Linear`:
  * state: float8 `weight` [out, in], `weight_scale` fp32 (0-d, or per row), `input_scale` fp32 0-d;
  * forward: `xq = clamp(x.float() / input_scale, ±448).to(float8_e4m3fn)`, then
    `torch._scaled_mm(xq, W.t(), scale_a=input_scale, scale_b=weight_scale, bias=bias, out_dtype=x.dtype)`;
  * `mode="emulate"` does the same math in fp32 ops (dequantize, then GEMM), for devices without
    `_scaled_mm`. Without an input scale the layer is weight-only.
* **Loading:** strict (no missing or unexpected keys, nothing left on meta). Unquantized tensors are
  cast to `torch_dtype` as diffusers does. The model is only ever moved with `.to(device)`; a dtype
  `.to()` would convert the float8 weights.
* The only model-specific inputs are the class (from `model_index.json`) and the config directory,
  exactly what `from_single_file` takes.

### 3.4 Validation

* **Toy tests** (`tools/models/test_fp8_checkpoint.py`, log `loader/toy_tests.log`; 5/5 pass):
  * a plain `nn.Module` with no converter: metadata parsing, placement, and `scaled_mm` and
    `emulate` against an fp32 fake-quant reference;
  * per-row weight scales and weight-only layers;
  * an arbitrary non-FLUX converter that splits, swaps halves (a row permutation), renames and
    strips a prefix: the exact float8 bytes follow;
  * refusals: bad version, unknown format, a missing scale, a casting converter, mixed sources,
    dropped rows, undeclared float8;
  * **a tiny `Flux2Transformer2DModel` through diffusers' official converter.** diffusers'
    `from_single_file` raises the same `chunk` error on it. The loader's 16 FP8Linears equal
    official `from_single_file` on the dequantized checkpoint bit for bit, and the forward pass
    matches a fake-quant reference (relative error 0).
* **Real checkpoint** (`loader/fp8_load_report.json`, `loader/verify_weights.json`):
  * 309 keys give **100 FP8Linear** modules (80 layers; the 10 fused QKVs become 30 linears).
    `model.dtype` is bf16.
  * Every dequantized weight against the official BF16 transformer's parameter of the same name:
    relative L2 **0.0264-0.0267** and cosine >= 0.99964 for all 100. That is the expected e4m3
    per-tensor quantization error; a q/k/v swap or a wrong split would be about 1.4.
  * The 69 unquantized tensors are bitwise equal to the BF16 checkpoint.

## 4. CPU reference (no TNCA)

Settings: prompt `"A cat holding a sign that says hello world"`, 256x256, **4 steps** (the official
example for the distilled model on the model card: `num_inference_steps=4, guidance_scale=1.0`;
`model_index.json` has `is_distilled: true`), seed 0 (`torch.Generator("cpu")`), max sequence 512,
text-encoder layers (9, 18, 27).
* Schedule: `mu = 1.96557`; timesteps 1000, 955.39, 877.13, 704.11.
* Stage modules (`flux2_klein_run.py`) call the pipeline's own helpers:
  * `TextEncodeStage`: the device half of `_get_qwen3_prompt_embeds`;
  * `DenoiseStage`: one loop iteration, with `FlowMatchEulerDiscreteScheduler.step` written
    functionally;
  * `DecodeStage`: unpack, BN de-normalisation, unpatchify, `vae.decode`.

Checks (all true on both runs; `reference/*/run.json` `checks`, `compare/staged_vs_official_*.json`):
* the prompt embeddings are bitwise equal to the official `_get_qwen3_prompt_embeds`;
* each of the 4 functional Euler steps is bitwise equal to `scheduler.step`;
* **`official` (the unmodified `Flux2KleinPipeline.__call__`, FP8 transformer from the loader passed in)
  is bitwise equal to the staged run: all 4 step latents and the final image**, for FP8 and BF16 alike.

| stage (wall s / peak RSS GiB) | FP8 W8A8 | BF16 |
|---|---|---|
| load VAE | 0.38 / 0.69 | 0.39 / 0.69 |
| load text encoder (mmap, lazy) | 0.38 / 0.82 | 0.39 / 0.82 |
| text encode (512 tokens, incl. first-touch page-in) | 7.64 / 9.30 | 7.64 / 9.30 |
| load transformer | 2.08 / 1.78 | 0.08 / 1.77 (mmap) |
| denoise step 0 / 1 / 2 / 3 | 15.41 / 18.02 / 15.80 / 16.58, peak 5.58 | 8.43 / 8.06 / 8.63 / 9.55, peak 8.31 |
| VAE decode | 0.79 / 1.41 | 0.89 / 1.40 |
| total (incl. one official TE check) | 87.3 | 52.8 |

* Peak RSS is per stage: VmHWM is reset through `/proc/self/clear_refs` and includes mmap-resident
  weights. Because of those resets, `/usr/bin/time`'s whole-process maximum is not meaningful.
* CPU `_scaled_mm` is slower than the BF16 GEMM here. That says nothing about TNCA.
* The official `__call__` took 63.4 s (FP8) and 49.6 s (BF16).

FP8 vs BF16 (`compare/fp8_vs_bf16.json`; same TE, noise and schedule):

| | step-0 noise pred | step-0 latents | final latents | image |
|---|---|---|---|---|
| rel-L2 | 0.057 | 0.0047 | 0.141 | 0.072 (PSNR **25.3 dB**, cos 0.9974) |

**The reference is sensitive to sub-ulp changes.** Computing RoPE in fp32 instead of the fp64
diffusers uses on CPU (`--fp64-unsupported-devices cpu`; section 5.6) changes the step-0 noise
prediction by rel-L2 0.050, about as much as FP8 does. It moves the final image to 18.0 dB PSNR,
visually the same (`reference/fp8_rope_fp32/`). End-to-end trajectories amplify any rounding
difference, so TNCA output cannot be judged by bitwise or end-to-end equality. Use per-stage
teacher forcing and the op-level layer check, with the floors measured in 4.1.

### 4.1 Noise floors for judging a non-bitwise run (teacher-forced: each stage gets the reference's inputs)

These runs use `run --teacher-force reference/fp8`: each stage gets the reference's own inputs,
and its output is compared with the reference's (`compare/teacher_forced/*.json`). The numbers are
rel-L2, with the minimum cosine over steps in brackets.

| change (teacher-forced, against `reference/fp8`) | text encoder `prompt_embeds` | one denoise step: `noise_pred`, steps 0 / 1 / 2 / 3 | VAE `image_decoded` (image PSNR) |
|---|---|---|---|
| 4 threads instead of 8 | 0 (bitwise) | 0 (bitwise) | 0 (bitwise) |
| FP8 `emulate` (dequantize + fp32 GEMM) instead of `_scaled_mm` | unchanged | 0.054 / 0.120 / 0.081 / 0.060 (0.9928) | unchanged |
| RoPE in fp32 instead of fp64 | unchanged | 0.050 / 0.081 / 0.078 / 0.061 (0.9967) | unchanged |
| Inductor-compiled stages (cpu) | 0.018 (0.99983) | 0.056 / 0.120 / 0.089 / 0.068 (0.9928) | 0.0036 (56.1 dB) |
| Inductor + `emulate_precision_casts` | 0.0091 (0.99996) | 0.049 / 0.085 / 0.070 / 0.063 (0.9964) | 0.0022 (60.0 dB) |
| for scale: BF16 instead of FP8 transformer (chained, step 0) | 0 | 0.057 (0.9984) | |

* **The reference is deterministic.** It is bitwise identical across thread counts, also end to end
  (`compare/fp8_threads4_vs_threads8.json`).
* **But a denoise step amplifies any rounding-level change** to 5-12 % rel-L2 of the noise
  prediction. That is as much as replacing all FP8 weights with BF16.
* **The step-level floor also hides real faults** (`compare/teacher_forced/sensitivity.json`,
  `sensitivity.py`). One FP8Linear (`single_transformer_blocks.10.attn.to_out`, step 0) was
  perturbed:
  * 1 % multiplicative noise on its output: rel-L2 0.023;
  * 10 % multiplicative noise: rel-L2 0.029, still below the 0.05 rounding floor;
  * output zeroed: rel-L2 0.20 (cos 0.979), only about 4x the floor.

So correctness on TNCA has to be shown at three levels:
1. **Op level, which carries the proof:**
   `flux2_klein_run.py layer-check --device tnca --teacher-force reference/fp8`. It runs each of the
   100 FP8Linears alone on its captured reference input and compares the output with the exact W8A8
   result in float64 and with CPU eager. A correct device GEMM sits at bf16 output rounding.
   * Dry run on CPU (`layer_check/cpu_eager.json`, `layer_check/cpu_inductor.json`): eager and
     Inductor-compiled per layer, all 100 layers are bitwise equal to the captured outputs.
   * Their error against the exact result, at most **1.69e-3**, equals the bf16 rounding of the
     exact result to about 1e-11, so CPU `_scaled_mm` is correctly rounded.
   * A TNCA layer should therefore land at rel-L2 of about 1.7e-3 against the exact result, and
     within a few bf16 ulps of CPU. A 1 % error in one layer is then 6x the tolerance, where the
     step-level comparison above could not see it.
2. **Stage level, teacher-forced:**
   * text encoder rel-L2 within about 0.02-0.03;
   * every denoise step's `noise_pred` rel-L2 within about 0.15 and cos >= 0.99;
   * VAE image >= 50 dB.

   Use the fp64-RoPE reference, or one made with `--fp64-unsupported-devices cpu` if tnca is
   declared fp64-unsupported.
3. **End to end:** the same scene and composition. Bitwise or near-bitwise equality is not expected:
   * FP8 vs BF16: 25.3 dB;
   * RoPE fp32: 18.0 dB;
   * Inductor on CPU: 21.3 dB, with the sign reading "helllo world".

### 4.2 Artifacts (the TNCA comparison reference)

`reference/fp8/` and `reference/bf16/` each hold `run.json`, `image.png` and `tensors.safetensors`
(9.4 MB). `run.json` has settings, versions, schedule, per-stage time and memory, and a sha256,
mean, std and finiteness for every tensor. `tensors.safetensors` holds:
* `input_ids`, `attention_mask`;
* `prompt_embeds` [1,512,7680] bf16, `text_ids`;
* `latents_init` [1,256,128], `latent_ids`;
* `step{0..3}_noise_pred` and `step{0..3}_latents` [1,256,128] bf16;
* `final_latents_packed`, `final_latents_vae` [1,32,32,32];
* `image_decoded` [1,3,256,256] (VAE output, [-1,1]) and `image_pt` (post-processed, [0,1]).

`official/{fp8,bf16}/` holds the official call's step latents and image.

| tensor (FP8 run) | sha256 (first 16) |
|---|---|
| prompt_embeds (same in both runs) | `d21cf85102a961f2` |
| latents_init (same in both runs) | `608316187cd4156e` |
| step3_latents | `265ed8b7a631f993` (BF16 `d93bbb478035c781`) |
| image_pt | `c2b83524c3b67d73` (BF16 `08718f9ee437da80`) |
| image.png | `80011e0c20a83451...` (BF16 `9d3d92afdfe1abb5...`) |

## 5. TNCA plan

### 5.1 The runner on TNCA (not run yet)

```sh
# all three stages on tnca, each stage compared alone against the cpu reference
flux2_klein_run.py run --transformer fp8 --te-device tnca --dit-device tnca --vae-device tnca \
    --teacher-force evidence/r41/fp8/flux/reference/fp8 --out OUT
flux2_klein_run.py compare OUT evidence/r41/fp8/flux/reference/fp8
# every FP8Linear alone on tnca against the exact W8A8 result (op-level proof; 4.1)
flux2_klein_run.py layer-check --device tnca --teacher-force evidence/r41/fp8/flux/reference/fp8 --out L.json
# end to end (chained), and with the text encoder kept on the host (5.4)
flux2_klein_run.py run --transformer fp8 --te-device cpu --dit-device tnca --vae-device tnca --out OUT2
```

What a tnca stage does:
1. `use_tnca()` registers the device from this tree, before triton is imported (`TNCA_BACKEND=systemc-tlm*`
   registers the TLM backend first, as the harness does).
2. The stage module is moved with `.to("tnca")`.
3. transformers' output-capture hooks are installed up front (`prepare_for_compile`).
4. The stage is wrapped in `torch.compile(stage, fullgraph=True)`, i.e. Inductor with the tnca device.
5. Inputs are copied in, outputs copied back. `torch.tnca.max_memory_allocated()` is recorded for
   the stage.

Stages are loaded and freed one after another, so they never share the device window. Across
stages only the prompt embeddings (7.5 MiB) and the latents (64 KiB at 256) move. The four steps
reuse one compiled module: `t` and `dt` are 0-d tensor inputs, so steps do not recompile. If the
device has no float64, add `--fp64-unsupported-devices tnca` and compare against a CPU reference
made with `--fp64-unsupported-devices cpu` (5.6, B4).

### 5.2 Compile-only / dry pass (`plan/plan.json`, `flux2_klein_run.py plan`)

`torch.compile(stage, fullgraph=True)` was run with a recording backend: dynamo, then AOT with
**Inductor's decomposition table**, which gives the graph Inductor lowers for any device. The graph
was executed only on meta or CPU tensors. Every stage captured as **one graph, no breaks**:
* text encode: 3308 nodes;
* denoise step: 5199 (FP8) and 4499 (BF16) on meta, 4434 (FP8) on CPU;
* VAE decode: 1099 on meta, 1159 on CPU;
* at both 256 and 1024.

There are two passes:
* **meta**: no data; every stage, resolution and transformer. On meta, SDPA decomposes to the math
  path, so activation figures there include S×S scores.
* **cpu**: real tensors (FP8 transformer and VAE at 256 and 1024; text encoder with the real
  tokens). Here SDPA stays the fused `_scaled_dot_product_flash_attention_for_cpu`, which is what
  an online/flash SDPA on TNCA would keep.

**A real compile as well** (`run --cpu-compile inductor`, `reference/fp8_inductor/`): every stage
went through `torch.compile(fullgraph=True)` with Inductor's CPU code generation, so every op above was
really lowered, including `_scaled_mm`, the float8 casts, the float64 RoPE and SDPA. The pipeline then
ran end to end.

* **The first attempt found a graph break the recording pass had missed.** transformers 5.20
  installs its `output_hidden_states` capture hooks lazily, on the first call, inside
  `with threading.Lock()`. dynamo stops there with `Unsupported context manager`, so the first call
  of a fullgraph-compiled text encoder fails. The recording pass had run eager first, which had
  already installed the hooks.
* The fix is generic: `prepare_for_compile()` calls transformers' own
  `maybe_install_capturing_hooks` for every `PreTrainedModel` in a stage before compiling. The tnca
  path does the same; an eager warm-up on tnca would be refused anyway.
* After the fix, the stage times were (wall s, peak RSS GiB):
  * text encode 24.3 / **6.26**, including compile; eager is 9.30, because the layers that dead-code
    elimination removed are never paged in;
  * step 0: 42.2, including compile;
  * steps 1-3: 24.6-24.9;
  * VAE: 8.2, including compile.
* Against the eager reference (chained, `compare/fp8_inductor_vs_eager.json`):
  * prompt embeddings rel-L2 0.018;
  * step-0 noise prediction 0.098;
  * final latents 0.206;
  * image PSNR 21.3 dB.
* **The sign reads "helllo world"**: compiled fusion and rounding alone, same weights and inputs, change
  a glyph. The teacher-forced numbers in 4.1 separate each stage's own error from this accumulation.

### 5.3 Ops and dtypes each stage needs

From the captured ATen graphs (`plan/plan.json` `ops`, counts per graph), after Inductor's decompositions:

**Text encode** (Qwen3 36L, 2560 hidden, 32 q / 8 kv heads × 128, 512 tokens). dtypes: bf16, fp32, int64, bool.
* `mm` bf16 ×189 (27 layers × 7; the graph returns hidden states 9/18/27, so AOT dead-code
  elimination drops layers 28-35, the final norm and the lm_head).
* `_scaled_dot_product_flash_attention_for_cpu` (on tnca: its SDPA lowering) ×27: causal plus
  padding mask (bool, `where`), GQA expanded by `expand`/`clone`, bf16 in, fp32 logsumexp.
* `embedding` (bf16 table, int64 ids); RMSNorm in fp32 (`pow`, `mean`, `rsqrt`, `mul`);
  SiLU-gated MLP (`neg`, `exp`, `div`, `mul`).
* Rotary: `cos`/`sin` fp32 from `iota` int64.
* Mask construction: `le` (int64), `bitwise_and` (bool), `index.Tensor` (bool, int64), `full`.
* bf16↔fp32 `convert_element_type`, `cat`, views (`view`, `permute`, `slice`, `unsqueeze`, `expand`).

**Denoise step, FP8** (5 double + 20 single blocks, 24 heads × 128, S = 768 at 256, 4608 at 1024).
dtypes: bf16, fp32, **float8_e4m3fn**, **float64**, int64 (+ bool with math SDPA).
* **`_scaled_mm` ×100**: float8_e4m3fn [M,K] row-major × float8_e4m3fn [K,N] column-major (`weight.t()`),
  `scale_a`/`scale_b` 0-d fp32, no bias, `out_dtype` bf16.
  * K ∈ {3072, 9216, 12288}; N ∈ {3072, 18432, 27648}.
  * M = image tokens (256 / 4096) or text tokens (512) in the double blocks, S in the single blocks.
* Static input quantization ×100: `div` fp32 by the 0-d scale, `clamp_min`/`clamp_max` ±448,
  `convert_element_type` fp32→**float8_e4m3fn** (round to nearest even).
* `permute` of float8 weights (the `.t()` view).
* SDPA ×25 (fused on CPU): non-causal, no mask, bf16. On meta it decomposes into `bmm` fp32,
  `amax`, `exp`, `sum`, `div`, plus `any`/`logical_not`/`where`/`eq` for fully-masked rows.
* `mm` bf16 ×9 (embedders, modulation; unquantized BF16 linears).
* **RoPE in float64**: `iota`→int64→**float64**, `pow.Scalar`, `reciprocal`, `mul`, `div`, `cos`,
  `sin` in float64, then `convert` to fp32 (`Flux2PosEmbed` asks
  `maybe_adjust_dtype_for_device(float64)`).
* LayerNorm `var_mean` fp32; RMS QK-norm fp32; timestep embedding `exp`/`sin`/`cos` fp32;
  SwiGLU/SiLU fp32.
* bf16↔fp32 converts, `cat`, `split`, `split_with_sizes`, `unbind`, `select`, views; int64
  `add`/`mul` (position ids).
* The BF16 transformer needs the same minus the FP8 ops: `mm` bf16 ×109 instead of `_scaled_mm`.

**VAE decode** (AutoencoderKLFlux2 decoder, BF16). dtypes: bf16, fp32, int64.
* `convolution` bf16 ×36 (3×3 and 1×1, up to 512 channels).
* GroupNorm as `var_mean` fp32 ×30.
* Nearest 2× upsampling as `_unsafe_index` (fp32 data, int64 indices) ×3.
* Mid-block attention: SDPA ×1, one head of 512, at 1024 (256) or 16384 (1024) tokens; plus `mm`,
  `addmm` bf16.
* SiLU fp32.
* Latent unpack: `scatter.src` (bf16, int64 index), `iota`, int64 `add`/`mul`; BN de-norm `sqrt` bf16.

### 5.4 Device memory per stage

All figures are bytes held on the device during the stage:
* weights as moved (`stage.to()`);
* inputs;
* the activation peak, i.e. storages created during the stage. Measured eager with fused SDPA
  (cpu pass). The upper bound in brackets is the larger of the unfused captured graph run node
  by node and, where it matters, the math-SDPA meta pass. Inductor fusion lowers the graph figure.
* Windows: B40 board **4 GiB**; SystemC/TLM **8 GiB** (`--memory-bytes 8589934592`, as in
  `tools/performance_harness/systemc/README.md`).
* Rule (doc/r41_tasks_and_acceptance.md §D): `C_usable >= 1.25 H_peak`, i.e. H_peak <= 3.2 GiB
  and <= 6.4 GiB. Code, descriptors, launch reserve and reserved ranges are not included, so these
  verdicts are optimistic.

| stage | weights (GiB) | activations (GiB) | peak H (GiB) | 4 GiB (H<=3.2) | 8 GiB (H<=6.4) |
|---|---|---|---|---|---|
| text encode, whole module | 7.492 (bf16, tied embedding once) | 0.250 [graph 0.066] | 7.74 | no | no |
| text encode, only weights the compiled graph reads | 5.800 (27 of 36 layers + embedding) | 0.066-0.250 | 5.87-6.05 | no | yes, if only those are moved (not implemented) |
| denoise step FP8 @256 | 3.791 (3.428 fp8 + 0.363 bf16) | 0.163 [graph 0.128; math 0.21] | 3.96 | **no** | yes |
| denoise step FP8 @1024 | 3.791 | 0.961 [graph 0.769; math SDPA 4.36-6.02] | 4.76 | no | yes, with fused/online SDPA (math SDPA: 8.2-9.8, no) |
| denoise step BF16 @256 / @1024 | 7.219 | 0.163 / 0.961 | 7.39 / 8.19 | no | no |
| VAE decode @256 | 0.157 (0.092 read: decoder) | 0.094 [graph 0.219] | 0.25 [0.38] | yes | yes |
| VAE decode @1024 | 0.157 | 1.502 [graph 3.500; math 2.22] | 1.66 [3.66] | yes (bound: no) | yes |

* Inputs are small: 8 KiB for the text encoder; 7.6 MiB (256) and 8.6 MiB (1024) for a denoise
  step; 72 KiB and 1.1 MiB for the VAE.
* The text-encoder activations include the 37 hidden states that `output_hidden_states` keeps and,
  in eager, the 512×151936 bf16 logits (156 MB).
* To run every stage as written, including the BF16 transformer, a window of at least ~10.2 GiB is
  needed (1.25 × 8.19). The TLM memory size is a parameter; whether its DRAM address map takes more
  than 8 GiB was not checked.

### 5.5 Compute (FLOPs; `torch.utils.flop_counter`, 2 per MAC)

| stage | 256x256 (S = 768) | 1024x1024 (S = 4608) |
|---|---|---|
| text encode, eager (36 layers + lm_head) | 4.273 T | 4.273 T |
| text encode, compiled graph (27 layers, no lm_head) | 2.906 T | 2.906 T |
| one denoise step | **4.917 T** (FP8 GEMM 4.711, attention 0.181, bf16 mm 0.025) | **34.820 T** (FP8 GEMM 28.266, attention 6.523, bf16 mm 0.031) |
| 4 denoise steps | 19.67 T | 139.28 T |
| VAE decode | 0.622 T (conv 0.618) | 10.475 T (conv 9.891, attention 0.55) |
| per image (compiled TE + 4 steps + VAE) | **23.2 T** | **152.7 T** |

These agree with the analytic count. Each block is 122.7 M MAC per token × 25 blocks, plus
2·S²·3072 MAC of attention per block (4.92 T and 34.8 T per step). Qwen3-4B is 3.63 G MAC per
token × 512 tokens, plus lm_head and attention (4.27 T).

### 5.6 Blockers and requirements for the TNCA stack

| # | requirement / blocker | status, options |
|---|---|---|
| B1 | **4 GiB B40 window: only the VAE stage fits.** FP8 transformer weights alone are 3.79 GiB (95 % of the window); H = 3.96 GiB at 256 against the 3.2 GiB allowed. The text encoder is 7.49 GiB. | Blocker for the board image. Needs a larger window, or weights streamed per block (no longer one graph per stage; not done). |
| B2 | **Text encoder: BF16 Qwen3-4B (8.04 GB), no FP8 version published.** As a whole module it fits neither window (needs 9.7 GiB). | Generic options: (a) keep it on the host, `--te-device cpu` (its output is 7.5 MiB); (b) put only the parameters the compiled graph consumes on the device (5.80 GiB, H 5.9-6.05: fits 8 GiB; needs a runner/stack feature, not implemented); (c) a TLM window of 10 GiB or more. |
| B3 | **FP8 GEMM**: `aten._scaled_mm` float8_e4m3fn × float8_e4m3fn → bf16, fp32 0-d scales, operand B column-major (`weight.t()`), no bias, 100 per step. Also fp32 → float8_e4m3fn conversion (round to nearest even) behind a ±448 clamp. | This is what the generic-FP8 TNCA work has to provide. Without `_scaled_mm`, `--fp8-mode emulate` runs the same W8A8 math as float8→fp32 converts plus an fp32 GEMM. The model is unchanged; numerics are in 4.1. |
| B4 | **float64**: `Flux2PosEmbed` computes RoPE in float64 (`pow`, `reciprocal`, `mul`, `div`, `cos`, `sin`, `convert`) on every device diffusers does not list as fp64-unsupported (mps, npu, neuron). | Either fp64 elementwise on tnca (small tables: [S, 16] per axis, 8 cos and 8 sin per step), or declare the device: `--fp64-unsupported-devices tnca` adds tnca to diffusers' own table, a generic device statement that gives fp32 RoPE. In that case compare against a reference made the same way (`reference/fp8_rope_fp32`), because fp32 RoPE alone moves step-0 noise by 5 % and the image to 18 dB. |
| B5 | **SDPA must stay fused/online** (the TNCA online SDPA needs fullgraph, which all stages have). Math SDPA at 1024 materializes 24 × 4608² scores per block: H 8.2-9.8 GiB, over 6.4. | Shapes: DiT 24 heads × 128, S 768 / 4608, no mask. TE causal plus padding mask, GQA 32/8, S 512. VAE 1 head × 512, 1024 / 16384 tokens. |
| B6 | Library graph break in transformers' lazy hook installation (found by the Inductor compile). | Fixed generically (`prepare_for_compile`). |
| B7 | The other ops in 5.3: bf16 GEMM (TE ×189, K and N up to 9728), bf16 conv2d ×36, fp32 reductions (`var_mean`, `mean`, `rsqrt`), embedding gather, `scatter`, `_unsafe_index`, int64 `iota`/arithmetic, bool masks and `where`, bf16↔fp32 converts. | Check against the stack; not assessed here (python/tnca is owned elsewhere). |
| B8 | **Acceptance cannot be bitwise or end to end.** The reference itself moves under sub-ulp changes, and Inductor on CPU already changes a glyph. A 10 % fault in one FP8 layer stays under the step-level floor. | Three levels (4.1): `layer-check --device tnca` per FP8Linear against exact W8A8 (tolerance about 1.7e-3); teacher-forced stage tolerances; a visual and PSNR check of the end-to-end image. |

## 6. Reproduce (GB10, `HF_HOME=~/flux-hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8`, `~/flux-venv/bin/python -I`)

```sh
python tools/models/fetch_flux2_klein.py                                  # pinned files, sha256-checked
python tools/models/test_fp8_checkpoint.py                                # toy tests
python tools/models/flux2_klein_run.py probe-official-loader --out loader/official_loader_probe.json
python tools/models/flux2_klein_run.py verify-weights --out loader/verify_weights.json
python tools/models/flux2_klein_run.py run --transformer fp8  --check-official --out reference/fp8
python tools/models/flux2_klein_run.py run --transformer bf16 --check-official --out reference/bf16
python tools/models/flux2_klein_run.py official --transformer fp8 --out official/fp8
python tools/models/flux2_klein_run.py compare reference/fp8 official/fp8
python tools/models/flux2_klein_run.py plan --out plan/plan.json          # ~11 min, 8 threads
python tools/models/flux2_klein_run.py run --transformer fp8 --cpu-compile inductor --out reference/fp8_inductor
python tools/models/flux2_klein_run.py run --transformer fp8 --fp64-unsupported-devices cpu --out reference/fp8_rope_fp32
# noise floors (4.1): teacher-forced variants, then compare each against reference/fp8
python tools/models/flux2_klein_run.py run --transformer fp8 --fp8-mode emulate --teacher-force reference/fp8 --out TF
python tools/models/flux2_klein_run.py run --transformer fp8 --cpu-compile inductor [--emulate-precision-casts] --teacher-force reference/fp8 --out TF
python tools/models/flux2_klein_run.py layer-check --teacher-force reference/fp8 [--cpu-compile inductor] --out layer_check/cpu_eager.json
python evidence/r41/fp8/flux/compare/teacher_forced/sensitivity.py reference/fp8 compare/teacher_forced/sensitivity.json
```

On GB10 the venv, the HF cache (`~/flux-hf`, 20 GB) and every run directory (`~/flux-work/out`)
are still in place for the TNCA runs.
