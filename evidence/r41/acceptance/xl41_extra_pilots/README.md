# XL gather, attention and recurrence pilot preparation

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../AUTHOR_DIRECTIVE_20261031.md).

These are bounded standard-Torch pilots, not full XL-core acceptance. Their
inputs, tolerances and scope were registered before any candidate-board result
in [`r3-xl41-extra-queue.json`](../../../../spec/r41/stage2/board/r3-xl41-extra-queue.json),
SHA-256 `7b7172000ee8c00ee3388d68295f1d01df2570249c03537834e19b4232dca545`.
The queue waits for verified transpose, GEMM and projection board pilots,
checks the same idle card, image, clock and queue lock, and never loads an
image. Five predecessor/scope tests passed ([`waiter_tests.log`](waiter_tests.log)).

The frozen gather formula produces 8,193 indices but does not include rows
0, 32,767 or 65,535, which the same specification also requires. The pilot
preserves the exact formula and runs a separate four-row boundary probe; it
does **not** claim that one vector satisfies both requirements. On GB10 CPU,
all 16,779,264 formula output elements and 8,192 boundary output elements
passed ([`../xl41_gather_pilot/cpu_preflight.tgz`](../xl41_gather_pilot/cpu_preflight.tgz)).

The attention pilot exercises BF16 D256 causal ragged GQA with zero output for
masked query tails. All 528,384 outputs passed against an independent CPU FP32
reference, with maximum relative error 0.00287
([`../xl41_attention_pilot/cpu_preflight.tgz`](../xl41_attention_pilot/cpu_preflight.tgz)).

The recurrence specification does not define an update equation. The pilot
therefore records the exploratory equation `state * exp(-softplus(gate)) +
sigmoid(gate) * input`, not an asserted official-model recurrence. Four FP32
updates across 32/32/1 chunks passed on GB10 CPU against an independent FP64
reference (maximum absolute error 6.51e-8)
([`../xl41_recurrence_pilot/cpu_preflight.tgz`](../xl41_recurrence_pilot/cpu_preflight.tgz)).
No candidate-board result or resolution of the two specification gaps is
claimed by these CPU records.
