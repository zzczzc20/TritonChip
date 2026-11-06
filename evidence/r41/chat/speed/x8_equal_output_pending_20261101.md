# Equal-output candidate handoff (2026-11-01 UTC)

The candidate is unmerged. Its source and prior TLM results are preserved in
`x8_equal_output_probe_20261101.patch.gz` and adjacent archives; the separate
local worktree is `/home/ubuntu/TritonChip-x8-equal-output-20261101`. The
board's fixed source archive has not been modified.

The immutable runtime code snapshot on the GB10 is
`~/r41_current_x8_equal_output_20261101`. The Qwen3.5 four-layer TLM control
and candidate use `r41-x8-equal-output-q35-{0,1}-20261101.service`; their
automatic comparator is `r41-x8-equal-output-q35-compare-20261101.service`
and writes `~/r41_x8_equal_output_q35_compare_20261101.json`. The completed Qwen3-VL
one-layer control and candidate used
`r41-x8-equal-output-vl-l1-{0,1}-20261101.service`; their comparator is
`r41-x8-equal-output-vl-l1-compare-20261101.service` and writes
`~/r41_x8_equal_output_vl_l1_compare_20261101.json`. Its exact logits and TLM counters are archived in `x8_equal_output_vl_l1_tlm_20261101.tgz`. The profiles and logits
are in the matching `~/r41_x8_equal_output_{q35,vl_l1}_{0,1}_20261101`
directories. The comparison reads actual completed launch counters and code
loads. Check the JSON and service logs, and preserve failures if either run or
comparator fails; no report is a board result or independent full-model quality
pass. Source, launch metadata, logits, counters and logs must be archived before
any decision to integrate.

Default-code audit is 113/113 unchanged. The FP8 suite with `TNCA_IEEE_DIV=1`
is 16/16 exact; with default division, both control and candidate are 12/16,
with the same four standalone activation-quantization failures. The 12 matmul
and related graph tests and the separately registered default-division fused
FP8 benchmarks matched exact references. Masked-row experiments that produced
identical code did not exercise the candidate. None of these results erases the
Qwen3.5 board multi-turn quality failure.

Only after the author's exactness and performance gates are actually satisfied
may the isolated change be integrated and a separately registered board
comparison use the existing verified image under `candidate_queue.lock` and
idle/identity checks. Do not program an image, modify a live board archive,
mark quality passed, or merge PRs on the strength of this pending work.

The candidate and a board-queue draft are committed only in the isolated
branch `r41-x8-equal-output-20261101` through `14592dfa`. The queue checks the
recorded exact-reference and default-audit evidence, requires the completed
Qwen3.5 comparison (VL-only evidence is rejected), bitwise model logits, zero
steady decode host reads and improved TLM decode cycles, and checks the source
hash before using the existing locked/idle verified-image stage. A small local
gate check confirmed that VL-only evidence and a synthetic host-read violation
are rejected. It has not been integrated, deployed, or run on the board. The
four pre-existing default quantizer failures remain explicit in its gate,
which requires their comparison records and standalone assembly to match the
control.

The Q35 model comparison completed and is archived as `x8_equal_output_q35_tlm_20261101.tgz`. The opt-in change and queue were subsequently integrated on R41, and the independently registered board performance session is waiting under `r41-equal-output-board-20261101.service`; the quality diagnostic ahead of it uses `r41-q35-precision-b32-whole-v3-20261101.service`. Check board status no more often than every 10 minutes. Neither session has a final result yet.
