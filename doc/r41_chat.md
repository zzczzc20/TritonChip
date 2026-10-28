# R41 Qwen FP8 chat driver

This document follows `doc/engineering_discipline.md`. Every number below comes
from an archived run and is labelled by domain: **board** (the U280 with the
run-1 100 MHz image), **TLM** (the SystemC/TLM model, uncalibrated), **RTL**
(Icarus simulation) or **CPU** (host-only reference or smoke runs).

`tools/models/chat.py` provides one-shot and interactive text chat with the pinned
Qwen3-VL-2B-Instruct-FP8 and Qwen3.5-4B-FP8 checkpoints. It uses the official
Hugging Face processor, tokenizer, chat template, model classes, FP8Linear and
Hub-kernel route. The model is loaded once per process. Text prompts and any
number of images are supported.

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

Interactive mode retains the loaded model and one generation session:

```sh
python tools/models/chat.py --model vl --device tnca --interactive
```

The REPL commands are `/greedy`, `/sample TEMPERATURE [TOP_K TOP_P SEED]`,
`/image PATH [PATH...]|none`, `/reset`, `/help` and `/quit`. `--script FILE`
runs the same session from a file (one prompt or command per line) and writes
every turn's report to `OUT/transcript.json`; this is how a chat is queued as a
board step.

**Session.** The process creates the compiled prefill and decode stages and one
static cache once, and every prompt reuses them. A new conversation resets the
cache in place (the same device tensors, zeroed). A follow-up turn whose
rendered conversation text extends the cached transcript keeps the generated
token IDs exactly as generated, tokenizes only the new template text, and
prefills only the new tokens; otherwise the session resets and prefills the
whole conversation. Qwen3.5's chat template drops the thinking block from
earlier turns, so its follow-up turns take the reset path (CPU run
`q35-full-repl.log`); Qwen3-VL text turns reuse the cache (21 cached tokens
reused in `vl-full-repl.log` and `vl-script`). After the first two decode calls
of a prompt, recompilation is refused.

**Chunked prefill (optional).** `--prefill-chunk C` prefills text in chunks of
C tokens and sends the remainder through the decode graph, so the number of
compiled graphs no longer depends on prompt length. The default (0) prefills
each prompt whole, which compiles one prefill graph per distinct prompt length.
Image prompts are always prefilled whole, because an image's placeholder tokens
must stay in one call.

**Images.** `--image` is repeatable. The vision tower runs once per image, so
each compiled call has exactly one static attention segment, and the per-image
features are passed to the language model as the per-image tuples that
`get_image_features` returns. `tools/models/vision_split_check.py` compares this
with one joint call over all images on CPU: for three images of two sizes all
four outputs (pooled features and three deepstack levels) are bitwise equal
(`evidence/r41/chat/p3/cpu-and-x8ctl-tlm.tgz`, `cpu-smoke/vision-split.json`).
The compiler still refuses a single compiled call with two or more symbolic
segments (`evidence/r41/fp8/ops/README.md`, V4); that refusal is not fixed, the
driver does not need it.

**Persistent-session check.** `--session-check PROMPT` runs, in the same
process, session and cache, a second conversation and then the first prompt
again with the first run's tokens forced. It requires bitwise-identical logits
and candidate tokens, and records new graphs and code-store evictions (modules
the runtime unloads to make room in the 128 KiB code store). Every step also
records its code-store evictions.

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

### Session, image and default-settings runs on the TLM model (P3)

Domain: **TLM** (uncalibrated SystemC/TLM model built from the current tree,
sha256 `276a8f12…`), reduced depth, `core-vrf-sync-v5-x8` profile. Commands,
launchers and source states are in `evidence/r41/chat/p3/runs.json`; raw output
is in `p3/tlm-vl-session-default.tgz` and `p3/tlm-q35-and-vl-images.tgz`. These
sources still applied the withdrawn floor-based allowance, so their reports
print "pass"; the judgments below apply the fixed 0.05 gate.

| Run | Settings | Teacher rel L2 (token 1, 2) | Gate 0.05 | Other result |
| --- | --- | --- | --- | --- |
| VL 1 layer, chunk 5, session check | non-default | 0.024659919041966807, 0.02574394384689021 | pass | session repeat bitwise equal; 3 graphs in total, none after the first prompt; 18 code-store evictions per decode token |
| VL 1 layer | default | 0.024659919041966807, 0.02574394384689021 | pass | — |
| VL 1 layer (current source, fixed gate in the report) | default | 0.024659919041966807, 0.02574394384689021 | pass | steady decode: 0 host reads, 40 bytes written per token; the 303872-byte logits download is reported separately as `logits_download_bytes` |
| VL 1 layer, two images | default | 0.021313732649428156, 0.023404515994576682 | pass | vision tower once per image (5615.917901336215 s and 5505.577816525474 s of model wall time) |
| Qwen3.5 4 layers, chunk 7, session check | non-default | 0.035197324952002645, 0.060020090237703445 | **fail** | session repeat bitwise equal; the post-reset prompt compiled one new graph, the repeat none |
| Qwen3.5 4 layers | default | 0.035197324952002645, 0.060020090237703445 | **fail** | 74 code-store evictions per steady decode token |

Both argmax choices match in every run. The few small host reads in the first
prefill and decode calls come from Inductor's compile-time constant folding
(`joint_graph.py`), not from model execution. The Qwen3.5 second-token value is the
same in all four Qwen3.5 TLM runs (default and non-default settings, with and
without chunked prefill), and the CPU reference's own reciprocal-versus-IEEE
scale-rounding difference at that token is 0.05941731822088129. A full-depth
Qwen3.5 board run is therefore expected to be at risk under the 0.05 gate; it
will be recorded as it comes out. Any different rule needs a written proposal
approved before the run that uses it.

### Teacher-forced rule registered on 2026-10-28, and the chat-quality check

Source for both: the author's decision in the lead's chat on 2026-10-28, relayed by the
coordinator. The acceptance criterion is whether the final model can hold a normal
conversation and show normal intelligence.

- **Rule (from v3 on).** `--cpu-floor-multiplier 2 --require-argmax`. For each forced
  step: relative L2 at most max(0.05, 2 × the CPU reference's own reciprocal-versus-IEEE
  scale-rounding floor at that step), and the device argmax equal to the reference argmax.
  The rule, its date and its justification are written into `s2-qwen-chat-v3-*.json`
  (`tolerance_rule`) before any run that uses it. The default of `chat.py` is still the
  fixed 0.05 limit, and every earlier run keeps its judgment under that limit: the
  Qwen3.5 TLM runs remain failed.
- **Quality check.** `spec/r41/chat/quality-v1.json` registers six prompts: factual QA,
  arithmetic, reasoning, instruction following, a two-turn follow-up, and an image
  description for VL. `tools/models/chat_quality.py` first records the CPU W8A8 reference
  model's own greedy answers (CPU, full depth, `evidence/r41/chat/quality-ref/`; both
  models answer "Paris", "42", "Bob", "OK" and "Alice", and VL lists circle, square and
  triangle). A device answer then passes a turn when it contains every a-priori keyword
  the CPU answer contains, and when its normalized text contains the CPU answer's text or
  the reverse. Token-by-token agreement with the CPU answer is reported for every turn.

### Board environment

The author approved the board interpreter (`~/r41s2_fp8env`, the torchenv whose `.pth`
sets `torch.version.cuda`) as the default chat environment (lead's chat, 2026-10-28).
Non-default compiler settings such as `TNCA_IEEE_DIV=1` are a separate matter: the
v2/v3 manifests use default TNCA settings, and the 51aef19 run that set it stays
labelled non-default.

### Qwen3.5 output projection (`lm_head.weight_scale_inv | MISSING`)

Qwen3.5-4B-FP8 ties `lm_head` to `embed_tokens`. Its checkpoint holds neither
`lm_head.weight` nor a scale, and its `modules_to_not_convert` lists `embed_tokens` but
not `lm_head`. transformers therefore built `lm_head` as an FP8Linear that holds the tied
bf16 embedding and a newly allocated, uninitialized `weight_scale_inv` (all zeros in the
GB10 load, `p3/lmhead-diag`). The scale is never read: `FP8Linear.forward` runs
`F.linear` when its weight is not 8-bit (transformers 5.20.2, both environments). So
`lm_head` was the intended unquantized bf16 projection on the CPU reference and on the
device alike, and earlier logits are not affected. Qwen3-VL lists `lm_head` in
`ignored_layers` and is not affected.

`qwen_fp8_lib.load_model` now leaves unconverted any module that the quantization config
would convert but whose block scale the checkpoint lacks (or whose weight is not FP8). It
reloads with those names added to the config's own `modules_to_not_convert`, and the load
report lists them (`unscaled_modules_not_converted`). This is keyed on the checkpoint's
contents, not on a model or module name. CPU logits of Qwen3.5 (4 layers) are bit
identical before and after the change.

### Host memory and the v2 SIGKILL

Board queue 0a8d8db: `q35-session` received SIGKILL after 410 s, with a 17,489,168 KiB
peak child RSS and no output (`evidence/r41/chat/board-0a8d8db.tgz`). The harness did
not send it: `experiment.py` kills only on a timeout, which it records as `timeout`.
acf3019 (62 GB, shared) runs `systemd-oomd`, and the user slice's `memory.events`
reports `oom_kill 1` (`board-0a8d8db-host-memory-events.txt`). That counter has no
timestamp, so a kernel OOM kill is the likely cause, but this is not proven.

The CPU part of the same step peaks at 7,192,212 KiB on GB10 (full-depth Qwen3.5,
two-token teacher check), so most of the board peak came from the device phase. There,
every host-to-device copy went through `transport.upload`, which turned the payload into
a Python list of 64-bit integers, one per word (about 44 bytes of host memory per 8
payload bytes). `_put_payload` then joined that list back into bytes. Further full-size
copies came from `tobytes`, padding and `bytes()`. The Qwen3-VL board step peaked at
17,685,692 KiB too.

The upload path now keeps the payload as one bytes-like view from `from_host` down to
the DMA. `Transport._put_payload_bytes` keeps the per-word path as the
medium-independent fallback; the MMIO backends override it, sending the register-window
words one at a time as before and the DMA part as bytes in 64 MiB chunks. On the RTL
stand-in (`tools/r41/upload_path_tests.py`, RTL simulation), a 256 KiB upload across both
windows reads back bit for bit on both paths. Its Python allocation peak is 896,215 bytes
on the new path and 5,900,871 bytes on the word-list path. The DMA, handoff,
host-copy, MMIO, HBM-layout, backend and hbm8 regression tests pass
(`p3/upload-tests.tgz`). The board peak with this change is not measured yet.

### TLM proofs of the v3 changes (2026-10-28)

Domain: **TLM** (and one **RTL** run), default TNCA settings, `core-vrf-sync-v5-x8`.
Commands and sources are in `evidence/r41/chat/p3/runs.json`.

- *Qwen3.5, 4 layers, registered rule, loader fix* (`tlm-q35-rule-v1`): passes the rule.
  Relative L2 is 0.035197324952002645 and 0.060020090237703445 against allowances of
  0.06399965325857539 and 0.11883463644176258, with argmax equal at both steps. `lm_head`
  is left unconverted on the device and no key is missing. Steady decode does 0 host reads,
  with the 496,640-byte logits download counted separately, and 74 code-store evictions per
  token.
- *Quality runner, Qwen3-VL, 1 layer* (`tlm-vl-quality-v1`): **failed**. The first prompt
  ran. The second prompt, of another length, made `torch.compile`'s automatic dynamic
  shapes recompile the prefill graph with a symbolic length, which the TNCA lowering
  refuses (`[UNSUPPORTED] scf.if whose branch contains arith.remsi`). The earlier session
  checks used two prompts of equal length and could not show this. Fix: the stages compile
  with `dynamic=False` (one static graph per shape; the first compile is static in either
  mode), and the v3 quality steps prefill text in chunks of 8 tokens, so the number of
  shapes stays bounded. The failure is preserved, and its reruns are below.
- *Static shapes, Qwen3-VL, 1 layer* (`tlm-vl-shapes-v1`): after an 11-token prompt, a
  20-token prompt in the same session compiled one new static prefill graph and ran.
  Repeating the first prompt added no graph and reproduced every logit and candidate
  token bit for bit (80 code-store evictions in between). The teacher check passes the
  registered rule (0.024659919041966807, 0.02574394384689021, argmax equal).
- *X8 Hub-kernel control on RTL* (`x8rtl-v2`, Icarus, `core-vrf-sync-v5-x8`): the first
  case (eager, x(1,7,384) → 320) passes bit for bit. The compiled case of the same shape hit
  the 14,400-second host timeout of the simulation, and the 1×16×2048 case exceeded the
  simulated device memory. Those two are limits of the simulation setup, not X8 results,
  and remain errors.

### Full-depth board results (queue from immutable source 51aef19)

Domain: **board**, U280 with the run-1 image (sha256 `30e79769…`, UUID
`49964b17-e0b4-5bd2-dd40-7fc37b1205d3`); the experiment's read-only probes show
the same UUID and a 100 MHz clock counter before and after each step. Raw
archive: `evidence/r41/chat/board-51aef19.tgz`; the summary
`board-51aef19-summary.json` is derived from it by `chat_board_closeout.py`.
The manifest (`spec/r41/chat/s2-qwen-chat.json`) set the non-default
`TNCA_IEEE_DIV=1` and `TNCA_TORCH_DEVICE=native` and compiled for the
`core-vrf-sync-v5` profile, so these results are not default-settings results
(`doc/engineering_discipline.md` §2.2).

**Qwen3-VL-2B-Instruct-FP8, full checkpoint (28 decoder layers), text prompt
"Say hi.":** the step completed and passed. The model generated
`Hello! How can` (tokens 9707, 0, 2585, 646). The two teacher-forced tokens
match the CPU W8A8 argmax; their relative L2 errors are 0.03492688958696281
and 0.03500718871823851, both below the fixed 0.05 gate. (The 51aef19 source
still applied the since-withdrawn floor-based allowance, so its report lists
allowances of 0.08446516185846147 and 0.0671607935159835; the result does not
depend on them. The logged CPU reciprocal-scale floors, a diagnostic, are
0.04223258092923073 and 0.03358039675799175.)

| Call | Wall (s) | New graphs | Device cycles | Device time at 100 MHz (s) | Launches | Host reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill (11 tokens) | 413.42981309397146 | 1 | 4238154739 | 42.38154739 | 701 | 84 |
| decode 1 | 346.60041622119024 | 1 | 4065580434 | 40.65580434 | 563 | 84 |
| decode 2 | 63.094938225112855 | 0 | 4065511739 | 40.65511739 | 563 | 0 |
| decode 3 | 64.12581400899217 | 0 | 4065552515 | 40.65552515 | 563 | 0 |

The first prefill and decode calls include compilation. In steady decode
(calls 2 and 3) the measured rate is 0.015720705662231143 tokens/s (about 64 s
per token): 40.7 s of device time and the rest host time for 563 launches per
token. Steady decode adds no graph and performs no host read; it writes 40 bytes
per token. The device peak was 2662758040 bytes. The run reports
`mean_decode_seconds_per_token` 157.94038948509842 over all three decode
calls, which includes the decode compilation.

**Qwen3.5-4B-FP8:** the step failed in 20 s before loading the model: the pinned
checkpoint was not in the FPGA host's input cache (`r41s2_fp8_inputs/hf/hub`).
That failure is preserved in the archive. The checkpoint has since been staged
(see `evidence/r41/acceptance/STATUS.md`); no Qwen3.5 board generation exists yet.

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
- **X8 transposed loads (corrected diagnosis):** the `DESCRIPTOR_FAULT` that
  the earlier chat and official-FP8 controls hit in `t.ld.async.b.t`
  (`evidence/r41/chat/fp8-x8-controls.tgz`) came from a stale SystemC/TLM
  binary. Both controls ran `~/r41-fp8-ops-build/tnca-chip`, built on
  2026-10-22 from a tree whose `tnca_tensor_model.hpp` admitted only 2-byte
  transposed loads; the X8 change that admits 1-byte ones was not in it. With a
  model built from the current tree (sha256 `276a8f12…`), the same control
  (`fp8_kernels_cases.py --group transformers`, profile
  `core-vrf-sync-v5-x8`) passes 3/3 bit for bit, with 32 `t.ld.async.b.t` in the
  block-dynamic kernel and no faulting launch (TLM; `p3/cpu-and-x8ctl-tlm.tgz`,
  `x8ctl-v1`). On the board, the P8 campaign (commit 63acb94; `evidence/r41/acceptance/STATUS.md`, P8 milestone) already ran the Hub kernels
  compiled for `core-vrf-sync-v5-x8` 35/35 bitwise on the run-1 image
  (`evidence/r41/acceptance/p8/`). The chat manifests from v2 on therefore
  compile for the image's own `core-vrf-sync-v5-x8` profile.
- **Still refused by the compiler:** one compiled call with two or more
  image/video attention segments of symbolic length. The driver does not issue
  such a call: it runs the vision tower once per image (see Images above).
  Video input is not supported by the driver.
- **Established on the board:** full-depth Qwen3-VL text generation (above).
  **Not yet on the board:** Qwen3.5 generation, image prompts, the persistent
  session check, scripted multi-turn chat and default-settings runs; they are
  registered in `spec/r41/chat/s2-qwen-chat-v2.json`. The `torch._scaled_mm`
  blockwise gap is unrelated to this driver, which uses the official FP8Linear
  Hub route instead.

The run manifests and archived chat reports live under `spec/r41/chat/` and
`evidence/r41/chat/`. Board jobs are lead-only under the shared-U280 rules.
