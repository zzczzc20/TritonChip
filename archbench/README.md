# ArchBench

ArchBench is a PyTorch-level benchmark for comparing the TNCA architecture with
GPUs on the workloads of large-language-model inference. Every benchmark point
is defined once, as a PyTorch module and its inputs; each platform compiles it
with its own `torch.compile` backend, runs it under a common timing protocol,
and is scored against the same strict lower bound.

```text
            ┌─ GPU:  torch.compile ─► Inductor ─► Triton (CUDA) / cuBLAS / cuDNN ─► GPU
point ──────┤
            └─ TNCA: torch.compile ─► Inductor ─► Triton ─► tnca backend ─► .tnca ─► Session ─► RTL / U50
```

The measurement and scoring rules are specified in [PROTOCOL.md](PROTOCOL.md).
This document describes the contents and how to run them.

## Contents

| Group | Families | Notes |
|---|---|---|
| LLM operators | `add`, `silu`, `gelu_tanh`, `swiglu`, `softmax`, `rmsnorm`, `layernorm`, `argmax`, `matmul`, `linear`, `gemv`, `bmm`, `attention_prefill`, `attention_decode`, `rope`, `embedding`, `copy` | FP16 unless noted; `matmul` also in FP32; `copy` in HBM tier |
| General operators | `transpose`, `cumsum`, `conv2d`; FP32 `add`, `softmax` (row length not a multiple of 16), `matmul` | Patterns outside decoder inference |
| LLM models | `llama_prefill`, `llama_decode` | LLaMA-2/3 architecture: RMSNorm, grouped-query attention with RoPE, SwiGLU MLP. Prefill returns the logits of the last position; decode generates one token against a *P*-entry KV cache and returns the new K/V entries |
| Other models | `gpt2_block`, `mlp`, `resnet_block` | GPT-2 block (LayerNorm, biased QKV, GELU); two-layer MLP; ResNet basic block (general track) |

Points are defined across five tiers ([PROTOCOL.md §1.1](PROTOCOL.md#11-points-tiers-tracks)):

| Tier | Sized for | Example (`llama_decode`) |
|---|---|---|
| `tiny` | The RTL simulator (~500 target cycles per wall-clock second) | dim 64, 2 layers, 16-entry cache |
| `small` | The board: every kernel's arguments fit the 384 KiB device heap of the U50 image's 512 KiB memory | dim 128, 4 layers, 128-entry cache |
| `medium` | Moderately bigger load for the board (2–8× `small`), staying within the same device heap (≤352 KiB arguments) | dim 192, 4 layers, 256-entry cache |
| `large` | GPU-realistic shapes; Llama-3.2-1B for the LLaMA models (capacity-exceeded on TNCA without external memory) | dim 2048, 16 layers, 512-entry cache |
| `hbm` | Working sets ≥ 32 MiB (streaming, GEMM, transpose, gather, and 4-layer 211 MiB decoder) requiring external memory | dim 1024, 4 layers, 512-entry cache |

Weights are drawn from the point's seed. Neither timing nor agreement with
the reference depends on trained weights, and no model download is needed.
Every platform computes on identical values.

## Layout

```text
archbench/
  registry.py      Point and Demand definitions, tolerances, the registry
  ops.py           operator families
  models.py        model families
  bounds.py        strict lower bounds: TNCA in cycles, GPU in seconds
  generate.py      multi-step greedy LLaMA generation chaining prefill and decode
  hbm.py           HBM tier points (working sets >= 32 MiB)
  specs/           GPU peak-rate specifications with their sources
  platforms/       tnca.py (Inductor → tnca → Session), gpu.py (CUPTI timing), common.py
  provenance.py    record provenance (software tree digest, per-point bundles)
  run.py           runs points and appends one JSON record per point
  report.py        scoring (η, E, A) and cross-platform comparison
  suite.py         freezes and checks suites/*.json
  selftest.py      validates definitions and scoring invariants
  suites/          frozen point definitions per tier (content hashes)
```

## Requirements

- PyTorch 2.14.0 and Triton 3.8.0 ([`spec/r38/torch.lock.json`](../spec/r38/torch.lock.json)).
- The `tnca` package installed into the same environment (it registers the
  `tnca` Triton backend), and Icarus Verilog 12 for simulation.
- For GPU measurements, a CUDA build of the same PyTorch; the Linux PyPI wheels
  include CUDA. One environment can serve both platforms: TNCA runs hide CUDA
  from their worker processes, so only the TNCA Triton driver is active there.

The Makefile targets use `TORCH_PYTHON` as the interpreter.

## Usage

```sh
# List points
python -m archbench.run --list --tier tiny

# TNCA on the RTL simulator (one worker process per point)
python -m archbench.run --platform tnca --tier tiny --jobs 12 --out results/tnca_tiny.jsonl

# Compile every kernel without simulating (coverage sweep, seconds per point)
python -m archbench.run --platform tnca --tier tiny small --compile-only --jobs 8 --out /tmp/coverage.jsonl

# Through the U50 register protocol on the board's RTL top level
python -m archbench.run --platform tnca --backend fpga-u50-sim --profile core-vrf-sync-v1 \
    --tier tiny --jobs 6 --out results/u50sim_tiny.jsonl

# On a board (see doc/r38_board_bringup.md): fpga-u50 or fpga-u280; a board is used serially
python -m archbench.run --platform tnca --backend fpga-u280 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "/path/to/tnca_u280.xclbin"}' --tier tiny small --out results/u280.jsonl

# Protocol 2 timed board measurement (warm-up calls, timed samples, provenance bundle)
python -m archbench.run --platform tnca --backend fpga-u280 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "/path/to/tnca_u280.xclbin"}' --tier tiny small \
    --warmup 5 --timed 30 --bundle-dir results/bundles --out results/u280.jsonl

# Multi-step greedy generation
python -m archbench.generate --tier small --new 8 --profile core-vrf-sync-v1 \
    --backend fpga-u280 --backend-args '{"xclbin": "/path/to/tnca_u280.xclbin"}' \
    --out results/u280_generate.json

# GPU (sequential; the primary variant is inductor-cudagraph)
python -m archbench.run --platform gpu --tier tiny small large --out results/gpu.jsonl
python -m archbench.run --platform gpu --variant inductor-triton --tier small --out results/gpu_triton.jsonl

# Scoring and comparison
python -m archbench.report results/tnca_tiny.jsonl results/gpu.jsonl --tier tiny
python -m archbench.report results/gpu.jsonl --tier tiny small \
    --freeze-weights archbench/suites/gpu-reference-weights.json

# Self-checks
python -m archbench.selftest
python -m archbench.suite --check
```

Equivalent Makefile targets: `archbench-selftest`, `archbench-coverage`,
`archbench-tiny`, `archbench-u50sim`, `archbench-gpu` (records under
`sim/build/archbench/`).

### Debugging a TNCA failure

```sh
TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=/tmp/dump \
    python -m archbench.run --platform tnca --point <id> --compile-only
python tools/r38/ttir_lower.py /tmp/dump/<kernel>/<kernel>.ttir --asm
```

The plugin writes every stage of every kernel, including one that fails to
lower; `ttir_lower.py` reproduces the lowering outside Inductor with a full
traceback.

## Adding a point

1. Register a `Point` in `ops.py`, `models.py`, or `hbm.py`: a `make(seed)` returning
   `(module, inputs)`, the dtype, and the semantic `Demand` per [PROTOCOL.md §4](PROTOCOL.md#4-lower-bound-and-efficiency).
2. Run `python -m archbench.selftest`. It checks that declared MACs never
   exceed PyTorch's FLOP counter, that dense families match it exactly, and
   that declared bytes match the actual tensors.
3. Run `python -m archbench.suite --freeze` and commit the updated
   `suites/*.json` with the definition. Changing a definition changes its
   hash.

## Record format

One JSON object per line (`archbench-torch-record/1` or `archbench-torch-record/2` per [PROTOCOL.md §3](PROTOCOL.md#3-timing) and [§5](PROTOCOL.md#5-aggregation)):

| Field | Content |
|---|---|
| `point_id`, `point_sha256` | The point and the hash of its definition |
| `platform` | TNCA: backend, profile, optimization level, Inductor configuration. GPU: device, SM count, L2, driver, variant |
| `evidence_level` | `measured-rtl`, `measured-board` or `measured-gpu` |
| `status`, `problems` | Outcome ([PROTOCOL.md §6](PROTOCOL.md#6-outcomes)) and its reasons |
| `fallbacks`, `kernels` | Host-side computation found in the generated code; the generated kernels |
| `correctness` | Maximum relative error and tolerance |
| `time` | TNCA: cycles per launch and their sum (`serial-grid`); seconds only with a board clock. GPU: CUPTI device time and event latency statistics, cold and warm |
| `timings` | Protocol 2: per-timing-kind sample summaries and raw samples |
| `demand`, `bound`, `eta` | Semantic demand, the strict bound (value, unit, binding resource, terms), η = L / T |
| `host` | Wall-clock time and bytes transferred between host and device |
| `counters` | TNCA device counters (`--counters`) |
| `provenance` | Protocol 2: software tree digest, artifact hashes, image, clock, launch map, optional bundle |
| `env` | PyTorch, Triton and Python versions; host |
