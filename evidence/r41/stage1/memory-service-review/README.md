# Memory-service evidence review, 2026-10-24

This is an offline review of the preserved memory-service case and comparator.
It adds no model, RTL, or board execution samples. No FPGA host or device was
accessed. The Stage 1 product and existing indexed evidence remain unchanged.

## Reproduction

From the repository root:

```sh
python3 evidence/r41/stage1/memory-service-review/verify.py
python3 -m tools.docmaint check
make r41-stage1-acceptance
```

`verify.py` reads `kickoff/memory-service-v1-evidence.tgz`, validates the saved
input bytes against every successful report's hashes, and invokes the current
comparator on the three historical model/RTL cohorts. It uses temporary JSON
files and never invokes the device case. The generated reports retain completed
launch cycles, outlet traffic, accumulated model service counters, bindings,
and scope limitations. `verification.json` binds this review to the archive,
case, and comparator hashes.

All 12 checks passed: three complete cohorts were accepted; missing cases,
failed numerical status, mixed product identities, zero cycles, wrong backend,
changed input hash, changed shader, mixed host policy, and an invalid domain
were rejected. All 13 successful archived reports have matching saved input
hashes. The two original failed setup attempts remain failures. This checks
saved evidence consistency and comparator behavior; it does not independently
rerun numerical execution or verify physical timing.

## Identity and acceptance

`frozen-acceptance.log` and `frozen-acceptance.json` were produced by extracting
`git archive 25094197f6e6` into a temporary directory and running
`make r41-stage1-acceptance` there. The result reports the expected identity:

```text
files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a
```

That archived commit is **not accepted**: 38 pass, 0 fail, 21 not measured,
2 blocked, exit status 2 from make. Its evidence index is the one saved at that
commit; this result is not a review of any later board campaign.

`merged-acceptance.json` records acceptance on merge commit `51ab03f`. That
tree matches `origin/r41-main` as fetched for the merge. Its identity is
`files:a86af4ae352dd47d9910b35fbe411fcaef920c483bd6dcc7db330972584e1303`.
Stage 2 software and documentation changes in the integrated branch differ
from the archived Stage 1 tree. The merged check is **not accepted**: 37 pass,
2 fail, 21 not measured, 1 blocked. The direct failure is the rewritten
`doc/r41_tasks_and_acceptance.md` no longer matching the indexed document hash;
the other failure is its dependent all-required item. No hashes or verdicts
were rewritten to conceal this mismatch. `python3 -m tools.docmaint check`
passes for all 72 registered documents; `python` is unavailable on this machine.

## Decision and remaining work

The case and comparator were retained byte-for-byte from `origin/r41-main`;
both also match `25094197f6e6`. Further changes in this milestone touch evidence
only. No product defaults changed.

The MMIO RTL stand-in's completed launch cycles are 37,429 (1/1), 19,132
(1/8), and 21,623 (3/8). The three-port case is slower than the one-port case
at depth 8. This synthetic-slave observation does not establish physical HBM
performance or whole-call latency. The saved model cohorts remain uncalibrated.

All historical service cases bind
`files:d35b4197c409549d4ce4af32721cca3f1ca47248f09f5519b7c3415b5d6bf0b9`,
which is different from the final frozen Stage 1 product. They cannot establish
final-source board closure. The lead must supply any authorized board results
and matching frozen-source model cases before a physical residual or final
Stage 1 memory-service pass can be claimed. The board queue remains with the
lead. Document-index reconciliation and remaining Required evidence are also
outstanding; neither model predictions nor this review substitutes for them.
