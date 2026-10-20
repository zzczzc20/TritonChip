# Language-model feasibility evidence

This is Stage 1 preparation, not complete-model execution or Stage 2 signoff.
The official checkpoint is pinned at `15852e8c16360a2fea060d615a32b45270f8a8fc`;
the reference Transformers sources are pinned at `4321355eeae7934bba83fce43269345c5f17feda`.
URLs and hashes are in [the source manifest](kickoff/transformers-reference/sources.json).

## Storage, operators and state

[The audited headers](kickoff/qwen-storage/inventory.json) contain 632 tensor storages overall,
4,548,144,832 bytes of payload. The language branch has 320 distinct storages totaling
3,763,655,360 bytes: BF16 3,763,644,992 bytes and F32 10,368 bytes.
The index and every header tensor, dtype/shape byte count, overlapping range and shard size were independently
checked by [the header audit](kickoff/qwen-storage/header-audit.json).
The tied embedding declaration must remain a single runtime storage; the vision and MTP branches are excluded
by a recorded prefix, without modifying the official configuration or execution graph.

| Component in the pinned reference | Necessary standard operations | Storage/arithmetic semantics |
|---|---|---|
| Input embedding and tied output projection | embedding/gather, linear, output token selection | BF16 weights/activations; preserve the tied storage |
| All 24 decoder layers | residual add, RMSNorm, projections, gated SiLU MLP | hidden 2048, MLP 6144; RMS reductions use FP32 arithmetic |
| Six full-attention layers | Q/K norm, partial RoPE, split/transpose, causal GQA, FP32 softmax, sigmoid output gate | Hq=8, Hkv=2, D=256; Q projection also carries the gate; BF16 KV |
| Eighteen linear-attention layers | four input projections, grouped causal conv1d K=4, SiLU, split/reshape, sigmoid beta, softplus/exp decay, q/k L2 normalization, gated RMSNorm and output projection | 16 key and value heads, key/value D=128; conv channels=6144 |
| Linear-attention prefill | FP32 casts, padding, cumulative sums, triangular masks/solves, chunked matmuls, recurrent-state update | official generic chunk rule contains additional intermediates; full prefill composition has not been compiled |
| Linear-attention decode | FP32 decay, predicted value reduction, outer-product delta update, query/state product | recurrent state `[B,16,128,128]` FP32; no FP16/BF16 state narrowing |
| Stateful cache | cat, slice, pad, copy_, persistent storage bookkeeping | conv cache `[B,6144,4]` BF16 per linear layer; full-attention K/V caches plus FP32 recurrent state |

The official decode rule explicitly converts q/k/v and decay arithmetic to FP32.
The checkpoint's small F32 tensors include normalization/decay parameters; they are not evidence that runtime
cache storage is FP32 or correct. Cache dtype/lifetime still needs execution verification.

## Representative compilation and capacity

`kickoff/depthwise-regression-v2-evidence.tgz` retains the three final representative graphs and source identities:
causal BF16 D256 GQA; BF16 depthwise conv; FP32 128x128 delta-rule state transition.
All use `torch.compile(backend="inductor", fullgraph=True)` and common native Session execution,
with independent CPU references, no host numerical fallback, no steady tensor staging and preserved FP32 state.
The report records cold compile-plus-execute time, machine-code sizes, predicted cycles and cumulative host RSS.
These graphs cover primitive semantics; the recurrence uses one representative head, not the full 18-layer state.
The earlier failed combined report remains at `kickoff/primitives/report.json` and is not counted as a final PASS.
Actual compiler-stage IR inventory and complete-model peak/code working-set estimates still need completion.

The analytical [capacity calculation](kickoff/qwen-core-capacity.json) for B=1, prefill128+decode16 gives:

- recurrent state: 18 × 16 × 128 × 128 × 4 = 18,874,368 bytes;
- conv state: 18 × 6144 × 4 × 2 = 884,736 bytes;
- full-attention KV: 6 × 2 × 2 × 144 × 256 × 2 = 1,769,472 bytes;
- distinct weights plus these states: 3,785,183,936 bytes, leaving 509,783,360 bytes before runtime and activations.

This is not a runtime peak bound. Argument tables, code, alignment, address holes, repack/cast temporaries,
simultaneously live activations and optional cache history must also fit.
Duplicating the largest weight storage would already exceed 4 GiB with the above base live set.
`model_capacity_estimate.py` records the formulas, individual buffer estimates and exclusions; the tail estimate
uses 129+4. These are feasibility inputs, not a Stage 2 freeze or capacity PASS.

## Public interface and cache-adapter differences

The pinned public model consumes token IDs, masks, positions and a `Cache` object. Its text branch returns
hidden states and cache; the tied output projection produces logits. The conditional-generation checkpoint
also contains vision and MTP weights; text-only
execution must select the official language branch while retaining its complete 24-layer configuration.
No adapter may invoke a TNCA-specific numeric kernel or substitute benchmark/model-named code.

The required cache protocol includes ordinary attention `update(k,v,layer_idx)`, linear-state getters,
`update_conv_state`, `update_recurrent_state` and `is_recording_past`. A simple KV-only tuple adapter does not
implement this hybrid interface. Conv prefill returns the concatenated history/current input but retains only
the last K entries when history recording is off. Recurrent updates use `copy_` into persistent storage.
Decode's in-place conv-state update and prefill's concatenation therefore need separate alias/lifetime checks.

The native device supplies standard tensor storage/views and copy operations. A model-facing adapter must
preserve the public cache method semantics, static storage addresses and state dtype across prefill/decode;
reset, repeated prompts and reset epochs must invalidate stale state explicitly. Recording history, rollback,
offloading and speculative decoding are not established capabilities. CUDA stream/offload machinery cannot
be used for TNCA state. Optional hub kernels must resolve to the official generic Torch implementations
unless a supported standard backend exists. Official eager attention additionally exercises norm, RoPE and
the sigmoid gate beyond the tested SDPA primitive. These differences remain inputs to Stage 2 readiness;
they are not covered by the representative primitive PASS.
