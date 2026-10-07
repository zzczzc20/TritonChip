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
| LLM operators | `add`, `silu`, `gelu_tanh`, `swiglu`, `softmax`, `rmsnorm`, `layernorm`, `argmax`, `matmul`, `linear`, `gemv`, `bmm`, `attention_prefill`, `attention_decode`, `rope`, `embedding` | FP16 unless noted; `matmul` also in FP32 |
| General operators | `transpose`, `cumsum`, `conv2d`; FP32 `add`, `softmax` (row length not a multiple of 16), `matmul` | Patterns outside decoder inference |
| LLM models | `llama_prefill`, `llama_decode` | LLaMA-2/3 architecture: RMSNorm, grouped-query attention with RoPE, SwiGLU MLP. Prefill returns the logits of the last position; decode generates one token against a *P*-entry KV cache and returns the new K/V entries |
| Other models | `gpt2_block`, `mlp`, `resnet_block` | GPT-2 block (LayerNorm, biased QKV, GELU); two-layer MLP; ResNet basic block (general track) |

Each family is defined in three tiers:

| Tier | Sized for | Example (`llama_decode`) |
|---|---|---|
| `tiny` | The RTL simulator (~500 target cycles per wall-clock second) | dim 64, 2 layers, 16-entry cache |
| `small` | The U50 board: every kernel's arguments fit the 384 KiB device heap | dim 128, 4 layers, 128-entry cache |
| `large` | GPU-realistic shapes; Llama-3.2-1B for the LLaMA models | dim 2048, 16 layers, 512-entry cache |

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
  specs/           GPU peak-rate specifications with their sources
  platforms/       tnca.py (Inductor → tnca → Session), gpu.py (CUPTI timing), common.py
  run.py           runs points and appends one JSON record per point
  report.py        scoring (η, E, A) and cross-platform comparison
  suite.py         freezes and checks suites/*.json
  selftest.py      validates definitions and scoring invariants
  suites/          frozen point definitions per tier (content hashes)
```

## Requirements

- PyTorch 2.14.0 and Triton 3.8.0 (`spec/r38/torch.lock.json`).
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

# On the U50 board (see doc/r38_board_bringup.md); the board is used serially
python -m archbench.run --platform tnca --backend fpga-u50 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "/path/to/tnca_u50.xclbin"}' --tier tiny small --out results/u50.jsonl

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

1. Register a `Point` in `ops.py` or `models.py`: a `make(seed)` returning
   `(module, inputs)`, the dtype, and the semantic `Demand` per PROTOCOL §4.
2. Run `python -m archbench.selftest`. It checks that declared MACs never
   exceed PyTorch's FLOP counter, that dense families match it exactly, and
   that declared bytes match the actual tensors.
3. Run `python -m archbench.suite --freeze` and commit the updated
   `suites/*.json` with the definition. Changing a definition changes its
   hash.

## Record format

One JSON object per line (`archbench-torch-record/1`):

| Field | Content |
|---|---|
| `point_id`, `point_sha256` | The point and the hash of its definition |
| `platform` | TNCA: backend, profile, optimization level, Inductor configuration. GPU: device, SM count, L2, driver, variant |
| `evidence_level` | `measured-rtl`, `measured-board` or `measured-gpu` |
| `status`, `problems` | Outcome (PROTOCOL §6) and its reasons |
| `fallbacks`, `kernels` | Host-side computation found in the generated code; the generated kernels |
| `correctness` | Maximum relative error and tolerance |
| `time` | TNCA: cycles per launch and their sum (`serial-grid`); seconds only with a board clock. GPU: CUPTI device time and event latency statistics, cold and warm |
| `demand`, `bound`, `eta` | Semantic demand, the strict bound (value, unit, binding resource, terms), η = L / T |
| `host` | Wall-clock time and bytes transferred between host and device |
| `counters` | TNCA device counters (`--counters`) |
| `env` | PyTorch, Triton and Python versions; host |
