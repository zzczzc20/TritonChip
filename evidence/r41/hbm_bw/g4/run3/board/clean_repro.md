# Clean-archive independent reproduction (pending)

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../../../AUTHOR_DIRECTIVE_20261031.md).

A cache-free archive of the board's frozen runtime source is staged at
`/home/zzhouch/r41run3/clean_repro_source_20261031/source.tar`, SHA-256
`5525f3509ac86a8be5d07b0b08c6c82f921129e5682065e11ed69b62971e41cf`.
Its freshly extracted source has the same recorded software-tree identity as
the candidate board runs. The official model code was copied separately
without Python caches and its three registered source hashes match. The model
will be reconstructed in a fresh process from the existing hash-verified
checkpoint, using new per-stage caches; checkpoint and independent reference
files are checked again by the model script.
The preparation script, exit status and source/model hash checks are preserved
in [`clean_repro_preparation.tgz`](clean_repro_preparation.tgz), SHA-256
`05fc7805ac2e8ea5dd97778087f47ff54fb526b45b4a848905985a76c24beaa3`.

Board-host PID 1145662 waits for all existing candidate and XL queues. It then
uses the extracted source to rerun representative legacy identity points,
HBM8 high-address checks and the full 17-step Q41-core session. The exact
[registration](../../../../../../spec/r41/stage2/board/r3-clean-repro-queue.json)
fixes the image, source and input hashes and the pilot-derived time bound.
The queue does not program the card. A failed stage stops it with evidence
preserved. Its output is `/home/zzhouch/r41run3/clean_repro_queue`.

The source staging check passed; no independent board result exists yet.
Actual results and the breadth of this reconstruction must be reviewed against
S2-06 before any reproduction or overall acceptance is claimed.
