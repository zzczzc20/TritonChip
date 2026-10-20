# Frozen full-model validation readiness

This checkpoint is **reference-only**. R40-08.decode, R40-09.model, all model
performance results, and final-image execution remain **not-measured**.
The frozen contract, model, shapes, seed and tolerances are unchanged.

`tools/r40/native_model.py` prepares a standard Torch fullgraph validation
path for the frozen four-layer model: dim 1024, 16 query heads, 4 KV heads,
FFN 2816, vocabulary 32000, FP16, batch 1, seed 0, prompt 512, and
221,267,968 bytes of weights. No reduced-tier or shape override is exposed.
Prefill and decode share the original parameter objects; generated tokens
and device KV state feed subsequent compiled steps. Weight upload is audited
once, and implicit tensor staging inside resident calls is refused.

Three CPU executions completed successfully on the jump host:

1. The initial state-wrapper runner generated the frozen reference prefix.
2. An independent runner executed the original `archbench/generate.py` CPU
   reference block, using its actual classes/functions, without hardware stubs.
3. The current validation runner independently regenerated the same reference.

All 64 original tokens match exactly; weight hashes and harness/source hashes
are checked in `summary.json`. The original public `--new 64` sequence has
one prefill-generated token and 63 decode calls. The runner preserves that
entire prefix and labels one additional decode separately to check the
contract's literal 64-decode-call requirement. The additional token is not
silently included in frozen performance statistics.

The unexecuted native branch checks numerical logits and KV state against
IEEE FP32 references, using the registry's frozen tolerance. It separately
checks the independent token sequence and retains any near-tie mismatch for
review, without an automatic exemption. Generated wrapper source must be
captured for the fallback audit; an empty capture cannot establish no fallback.

Native steps retain completion histories, device endpoints, DRAINING and
quiesce evidence, explicit validation transfers and host ownership-handoff
histories. Validation downloads are outside resident execution auditing.
They perturb subsequent cache state, so this runner deliberately cannot
certify TTFT, decode throughput, scaling, or any performance gate.
Source identity is recomputed before and after execution, including when
running against the frozen 8abf527 library, which predates refresh=True.

The next native experiment is conditional on a qualified final image and the
original-size vertical checkpoint. On the FPGA server, use an immutable
8abf527 library archive with this exact harness copied into tools/r40:

```sh
PYTHONPATH=python:compiler/python:tools/asm CUDA_VISIBLE_DEVICES= \
  ~/tc_env/torchenv/bin/python tools/r40/native_model.py \
  --software-tree-sha256 "$R40_FINAL_SOFTWARE_TREE" \
  --xclbin "$R40_FINAL_XCLBIN" --image-sha256 "$R40_FINAL_IMAGE_SHA256" \
  --out "$R40_MODEL_VALIDATION_OUT"
```

Bind those identities and the fresh output directory in the run manifest
before execution. The currently loaded legacy single-core image is not a
valid substitute. Success unlocks the separate frozen 1/3-core model timing
experiment. Failure retains the first failed native step and its raw audit;
fix that cause without shrinking the model or changing the contract.

`raw.tar.gz` retains original reports, scripts, logs, exit files and both
hashed state-wrapper source versions. The verified source version matches
the committed runner. The native branch has undergone source review only;
the CPU reference results must not be represented as native execution proof.
