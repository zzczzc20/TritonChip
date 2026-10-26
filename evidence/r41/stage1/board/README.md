# Stage 1 board evidence

Read-only snapshot from `acf3019:/home/zzhouch/r41s1_board_25094197f6e6`, retrieved 2026-10-26 UTC.
Product commit: `25094197f6e6`; software tree: `files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
Image: B40 `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, 90 MHz, three cores, 4 GiB.

`finished.tgz` contains the six completed manifests' reports, state, batches, formal records and comparison outputs under `out/board-*`.
`index.json` records every archive member's size and SHA256 plus the archive digest.
`host-raw.sha256` preserves hashes of the host queue log and raw logs/JSONL; paths there are relative to the host root above. The line immediately after the queue digest records its byte size. Large raw logs, caches and compiler dumps remain on the host and are excluded from git. The archive is a selected evidence bundle, not a complete execution backup.

Extract locally with `tar xzf evidence/r41/stage1/board/finished.tgz -C /path/to/evidence`.
A completed experiment records command completion; its hypothesis and contract gates require separate judgement.

`board-original-suite` is **not-measured**. The interrupted first attempt is excluded from this bundle; the lead schedules its rerun. No queue, script or device state was changed during collection.
