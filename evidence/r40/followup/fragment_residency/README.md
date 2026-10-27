# R40 follow-up: resident accumulator fragments

**Required gate delta: none.** `R40-03.gemm-tile` remains
`no-significant-gain`; `R40-10.acceptance` remains failed. This is an isolated
compiler candidate, not a replacement for the delivered fe1b software or its
frozen board records.

## Change and evidence

Backport the projected SSA-use graph and dot/add fusion from R41 commit
`d1feb89cfcfde3db385c4a885cb9e360407ac58f`. Only the two compiler functions
needed for fragment residency are changed. The regression fixture is copied
from the same commit. The broader R41 product changes are not included.

GB10's copied baseline was checked against delivered product tree
`files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`
before applying the patch. The candidate, including its regression fixture, is
`files:4bfc9ca1fa8565bb6f8e4d38ef8a43c5845646634ae3921d60aee1af1bc1e1f1`;
the local candidate independently produces the same digest.

The original code fails the resident-fragment regression; the candidate passes.
The counterexample retaining a second reader of the old accumulator continues
to block destructive accumulation. Insufficient physical accumulators are
still refused. `baseline.log`, `candidate.log`, and both static summaries
preserve these results. The complete compiler-unit suite passes with strict
skip checking (`compiler-unit.log`).

| Static operations per K-loop iteration, N=32 | Baseline | Candidate |
|---|---:|---:|
| Accumulator reads | 128 | 0 |
| Accumulator writes | 96 | 0 |
| Accumulator zero operations | 2 | 0 |
| MMA commands | 2 | 2 |
| Input DMA commands | 3 | 3 |

The entire directed N=16 IR hash is unchanged. The entire N=32 counterexample
IR hash is also unchanged. These are directed comparisons, not a claim that
every default kernel is byte-identical.

The full frozen nine-point GEMM subset compiles offline on the native-stub
path for both 16,16,16 and 16,32,16: all 18 configurations succeed. The original
preflight command, item selection and complete report are retained. This
establishes compile coverage only, including the original HBM shapes; no
FPGA device was opened by preflight.

## Reproduce the compiler checks

Run in the Torch/Triton environment on GB10:

```sh
R1_STRICT=1 PYTHONPATH=compiler/python python compiler/test/run_tests.py --no-rtl
python tools/r40/followup_fragment_summary.py \
  --source-root "$R40_CANDIDATE" --tests-root "$R40_CANDIDATE"
```

`source_manifest.json` binds the retrieved raw files to their GB10 paths and
hashes. The recorded runners specify their complete commands. The initial
numerical launch failed because the runner incorrectly selected backend `rtl`,
which does not exist. That failure is preserved. Its replacement uses
`sim-file`, a private simulator cache and fresh output directories.

## Open validation and next decision

Update, 2026-10-27: standard Torch mm/addmm numerical simulation for FP16 and
BF16 **passed all eight directed cases**. Both controller return codes are zero;
the final state and complete dtype logs are archived here. This proves the
listed directed cases on candidate `4bfc9ca1…`, not the full frozen suites or
the later `60cf7062…` candidate. No RTL simulation was placed on the FPGA host.

This fix does not yet establish a competitive wide tile. Inspection of the
new linear/medium assembly still finds 1,886 non-directive operation lines versus 518
for the narrow tile, including 96 versus 16 `a.vread` instructions and 12
versus 2 fences. These totals include epilogues, exclude assembly directives
and pseudo-constant materializations, and are not machine-instruction or dynamic cycle
predictions. They identify further lowering work before a board campaign:
projected users and original effect-order facts must also reach transpose
selection and deferred epilogues. R41 commit `867620e` contains related fixes
that require a separate port and validation on this R40 candidate.

If the current numerical batch fails, preserve the failure and fix the
compiler before further measurement. If it succeeds, retain its identity-bound
evidence and investigate the remaining lowering overhead before spending
board time. A future accepted delivery requires full native77/HBM14/model
correctness and remeasurement of the frozen performance subset on an
explicitly allocated frozen image. Board performance and gate improvement
remain **not-measured**.
