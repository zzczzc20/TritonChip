# R41 Qwen FP8 chat driver

`tools/models/chat.py` provides one-shot and interactive text chat with the pinned
Qwen3-VL-2B-Instruct-FP8 and Qwen3.5-4B-FP8 checkpoints. It uses the official
Hugging Face processor, tokenizer, chat template, model classes, FP8Linear and
Hub-kernel route. The model is loaded once per process. Text prompts and one
optional image are supported; the board preflight is text-only.

## Usage

The Hugging Face cache must contain the pinned snapshots. Set `HF_HOME` and
`HF_HUB_OFFLINE=1`, and use the Torch/Transformers environment already used by
`tools/models/qwen_tnca_layers.py`.

```sh
python tools/models/chat.py --model vl --device tnca \
  --prompt "Hello. Introduce yourself briefly." --new-tokens 32 --bucket 256 \
  --out OUT
```

Use `--model q35` for Qwen3.5-4B-FP8. `--sample --temperature 0.8 --top-k 40
--top-p 0.95 --seed 7` enables seeded sampling; omitting `--sample` selects
greedy generation. Token selection, EOS handling and control flow run on the
host. Model computation on TNCA uses `torch.compile(fullgraph=True)` and does
not fall back to CPU numerical execution.

Interactive mode retains the loaded model:

```sh
python tools/models/chat.py --model vl --device tnca --interactive
```

The REPL commands are `/greedy`, `/sample TEMPERATURE [TOP_K TOP_P SEED]`,
`/image PATH|none`, `/help` and `/quit`. Each prompt has its own static cache;
each generation compiles at most one new decode graph, and recompilation is
refused after that graph. A changed prompt shape can require a new prefill
graph, which is separate from token-by-token decode.

`--layers N` truncates decoder depth for integration proofs while retaining the
official checkpoint, tokenizer and remaining weights. Its report says
`reduced-depth-proof`; it is never a full-model result. `--check-tokens N` first
runs the CPU W8A8 reference in `tools/models/fp8_blockwise_ref.py`, feeds its
first N greedy tokens to the target, and compares target logits and argmax
choices at every forced step. Copying the reference token IDs is not itself the
check. `--max-rel-l2` sets the numerical comparison threshold.

## Measurements and the memory-bandwidth ceiling

Every report records actual wall time for prefill and each decode call, graph
growth, host transfers inside each call, token IDs and output text. Decode
seconds per token and tokens/s are derived from those logged step times. Model
or RTL times are not physical-U280 measurements.

Decode rereads the language-model weights for each token. The existing core-port
interface is 8 B/cycle per core at 100 MHz, or a theoretical aggregate 2.4 GB/s
across three cores (`doc/r40_tasks_and_acceptance.md`). The driver divides this
interface rate by the model's actual language-parameter bytes to report a
weight-stream ceiling. It is an upper bound, not achieved bandwidth: HBM,
cache, descriptor, launch and host overheads can only lower it, and reuse or
compression can change the traffic assumption. Measured tokens/s must come from
the generation report.

## Operation coverage and remaining blockers

The following statements are scoped to the logged layer, two-layer decode and
vision-tower evidence in `evidence/r41/fp8/models_tnca/README.md` and the generic
operator evidence in `evidence/r41/fp8/ops/README.md`.

- **Working in that evidence:** official FP8Linear Hub kernels; Qwen3-VL full
  attention, norms, M-RoPE, MLP and residuals; its single-image BF16 vision
  tower; Qwen3.5 full attention and gated-DeltaNet linear attention, including
  conv1d, float cumsum, softplus, triangular-solve decomposition, FP32 recurrent
  state and static KV/state decode. Linear attention is not still blocked.
- **Host metadata is not numerical fallback:** single-image vision attention
  keeps `grid_thw` and `cu_seqlens` as host integer metadata. Its host reads do
  not execute vision arithmetic on the CPU. The documented multimodal prefill
  also has device scalar reads for masks/counts; text decode has none.
- **Still refused:** two or more image/video attention segments with symbolic
  per-segment lengths. The driver accepts one optional image and does not claim
  multi-image or video support.
- **Not established by layer evidence:** full-depth end-to-end prefill/decode,
  long-context or image-to-decode sessions, and physical-board generation.
  Those require the chat driver's own logged runs. The `torch._scaled_mm`
  blockwise gap is unrelated to this driver, which uses the official FP8Linear
  Hub route instead.

The run manifests and archived chat reports live under `spec/r41/chat/` and
`evidence/r41/chat/`. Board jobs are lead-only under the shared-U280 rules.
