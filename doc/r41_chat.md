# R41 Qwen FP8 chat driver

`tools/models/chat.py` provides one-shot and interactive text chat with the pinned
Qwen3-VL-2B-Instruct-FP8 and Qwen3.5-4B-FP8 checkpoints. It uses the official
Hugging Face processor, tokenizer, chat template, model classes, FP8Linear and
Hub-kernel route. The model is loaded once per process. Text prompts and one
optional image are supported; the board preflight is text-only.

## Usage

The Hugging Face cache must contain the pinned model and Hub-kernel snapshots.
Set `HF_HOME`, and use the Torch/Transformers environment already used by
`tools/models/qwen_tnca_layers.py`. The current GB10 launcher uses
`HF_HUB_OFFLINE=0` to resolve the staged Hub-kernel version; strictly offline
resolution was refused and that failure is preserved. The FPGA preparation uses
the previously validated offline environment.

```sh
TNCA_PROFILE=core-vrf-sync-v5 python tools/models/chat.py --model vl --device tnca \
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
check. `--max-rel-l2` (default 0.05) is the fixed per-token gate; it is set
before the run and never derived from the run's output. The driver also runs
the CPU reference with reciprocal scale rounding and logs that numerical floor
at each forced step, as a diagnostic only: it does not change the gate. (A
floor-based allowance, `max(0.05, 2 × floor)`, was briefly used after the
q35-v1 failure. It was withdrawn on 2026-10-28 because it moved the tolerance
after the result was seen; see `evidence/r41/hbm_bw/AUDIT.md`, R6.)

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

### Archived reduced-depth TLM results

`evidence/r41/chat/tlm-summary.json` is derived from the raw archives by
`chat_tlm_closeout.py`. These are uncalibrated SystemC/TLM wall times for small
prompts and reduced decoder depth, not full-model results or physical-U280
measurements. The generated text is not a model-quality claim.

| Run | Layers | Generated tokens | Prefill (s) | Mean decode (s/token) | Decode (tokens/s) | Reduced-model weight-stream ceiling (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VL greedy | 1 | 3 | 327.4231004854664 | 312.1245263405144 | 0.003203849475478396 | 1.8532575820474197 |
| VL sampled | 1 | 2 | 305.28765878267586 | 307.43000791873783 | 0.0032527729051886416 | 1.8532575820474197 |
| Qwen3.5 greedy (v2; failed the fixed 0.05 check) | 4 | 3 | 1093.5643128491938 | 837.9598828395829 | 0.0011933745522653353 | 0.802693368029665 |

The VL sampled run uses temperature 0.8, top-k 40, top-p 0.95 and seed 7. The
VL greedy teacher check covers two forced tokens: relative L2 values are
0.024659919041966807 and 0.02574394384689021, and both argmax choices match.
The Qwen3.5 teacher check also covers two forced tokens: relative L2 values are
0.035197324952002645 and 0.060020090237703445, and both argmax choices match.
The second value exceeds the fixed 0.05 limit, so **the Qwen3.5 run fails its
numerical check**; its timing row above is kept as a record, not as a passing
result. Its logged CPU reciprocal-scale floors are 0.03199982662928769 and
0.05941731822088129 (diagnostic only).

In the steady second decode, both greedy runs create zero new graphs, perform
zero host reads, and perform two host writes totaling 40 bytes. The sampled run
exercises sampling but has only one decode step and therefore does not establish
steady decode behavior.

The first Qwen3.5 attempt is retained in `tlm-q35-v1.tgz` with its nonzero exit:
its second relative L2 exceeded the fixed 0.05 limit. The second attempt
(`tlm-q35-v2.tgz`) produced bit-identical logits and fails the same limit; its
archived report says "pass" only because it applied the withdrawn floor-based
allowance. The independent `q35-cpu-floor-v1.tgz` diagnostic shows that CPU
scale rounding alone accounts for 0.05941731822088129 at that step. That is a
reason to propose a different rule for future runs, in writing and before
they start; it does not turn either recorded run into a pass.

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
- **Current X8 small-M prefill blocker:** fresh chat and standalone official-FP8
  controls on this tree fault in `t.ld.async.b.t` with `DESCRIPTOR_FAULT` for the
  small-M Hub kernel, on both current and pre-P3 sources. The failed reports and
  assembly are preserved in `evidence/r41/chat/`. The chat proof and prepared
  board manifest therefore use the portable `core-vrf-sync-v5` gather lowering;
  the approved X8 image executes those v5 artifacts. This is not an X8 result and
  does not close the transposed-load fault.
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
