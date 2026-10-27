# R40 follow-up: preserve chunk users and effect order

**Required gate delta: none.** The immutable R40 v3 acceptance remains
25/32 Required gates passed. `R40-03.gemm-tile` is still not passed;
none of these model results replaces a frozen board measurement.

## Candidate and safety checks

Candidate product identity:
`files:60cf7062521a3e5d84677016836f48bf331270c389aaf4166f1f8d3d5818ef4c`.
The source tree is unchanged before and after the remote checks. It extends
the resident-fragment candidate with two fixes adapted from R41 `867620e`:

- Use the original TTIR operation and effect order to decide whether a chunk
  expression can be deferred, preserving the original single-user check.
- Project the original users for transposed-load selection, so later column
  fragments can use the existing DMA transpose path.

Deferral explicitly stops at `ttg.barrier` and `tt.descriptor_reduce` as well
as the existing store, atomic, dot and control-flow boundaries. The regression
tests cover safe deferral, intervening effects, multiple users, different
blocks, DMA transpose, masks and live non-matrix readers. A mutation removing
the two new ordering guards fails on both affected effects; its expected
failure log is preserved.

The strict compiler-unit suite and all 18 native-stub compile configurations
for the original frozen nine-point GEMM subset pass. This is compile evidence,
not execution of the large HBM points. The previous candidate's eight Torch/RTL
cases pass; the new candidate's independent Torch/RTL batch is still running
at this checkpoint. Its controller is GB10 PID 574831 and state file is
`~/r40_followup_fragment_20261026/lowering-v2-numeric-state.json`; verify the
process before treating it as live. Its numerical results must not be inferred
from the previous candidate.

## Model screen on an original frozen point

The complete original `op.linear/medium/fp16-m128-k320-n256` point runs through
the native Torch/Inductor/Triton path and the M2 SystemC model. All four runs
match the numerical reference, with no fallback. No dimension is reduced.

| Software / tile | Predicted device cycles |
|---|---:|
| Delivered fe1b / 16,16,16 | 1,826,446 |
| Resident-fragment candidate / 16,32,16 | 9,318,383 |
| This candidate / 16,16,16 | 1,826,446 |
| This candidate / 16,32,16 | 4,934,039 |

The latest wide candidate improves by 1.889x over the first follow-up candidate,
but remains 2.701x slower than its narrow control in the model. This is a reason
to withhold a board performance campaign, not a performance pass. Only this
single original point was screened; no frozen-subset geometric mean is claimed.

Identity: M2 binary SHA-256
`2402d46d9375558c5fc80d2689b9b03fc235817902d2bf38852492f4d6a3c98d`,
base configuration SHA-256
`dc2c1763eafb7d022304ee820651465c4df04135a68381c0d3833ea7195eb796`.
Effective configuration is preserved in every raw record: model clock 90 MHz,
4 GiB window, one active core/outlet, one outstanding request, pipeline 1.
Model cycles are **uncalibrated predictions**, not board measurements. The
record's simulated device identity is not a loaded FPGA image identity.

The initial model runner failed before launching a graph because it pointed at
a source tree without the M2 configuration. Its failure is archived. The corrected
runner uses the exact binary and configuration from the previous R40 GEMM
diagnosis, after verifying both hashes.

## Reproduce and interpret

```sh
python3 tools/r40/followup_fragment_model_audit.py
```

This checks raw-file hashes, product identities, the original point identity,
effective environment, identical model configuration, the binary identity and
successful numerical status. It compares cycles only after those checks.

The old model harness's `results.json` copies pre-override defaults into
`knobs`: pipeline 2 and tile 16 may appear even when `--env` selected pipeline 1
and tile 32. **Do not use that summary field as configuration evidence.** The
unaltered raw `record.json` and `env.json` agree on the actual configuration.
The audit reports the stale summary fields and uses only the checked raw records.
Both summaries and raw records are retained in this directory.

## Remaining blocker and next experiment

The current blocker is remaining wide-fragment lowering overhead. The archived
linear assembly drops from the prior candidate's 1,886 non-directive operation
lines to 1,114, but the narrow control has 518. Wide accumulator epilogue reads
fall from 96 to 32 and writes from 64 to zero; five fences remain versus two in
the narrow tile. Assembly lines exclude directives and pseudo-constant
materializations and are not dynamic instruction counts.

Next, investigate lost affine/alignment facts in later fragments before another
model comparison. After a candidate passes the effect tests and full compile
preflight, run the same original point with its fresh source identity:

```sh
"$TORCH_PYTHON" "$M2_HARNESS/run_graphs.py" \
  --source-root "$NEXT_CANDIDATE" --binary "$M2_BINARY" \
  --config "$M2_CONFIG" --out "$FRESH_OUTPUT" \
  --point op.linear/medium/fp16-m128-k320-n256 \
  --backend systemc-tlm-grid --active-cores 1 --memory-bytes 4294967296 \
  --env TNCA_GEMM_TILE=16,32,16 --env TNCA_PIPELINE=1 --jobs 1
```

Use the verified M2 paths and private extension/cache directories on GB10.
Numerical or identity failure rejects the candidate. A slower prediction keeps
board testing withheld and directs further lowering analysis. A numerically
correct candidate predicted faster than its narrow control unlocks the full
frozen-subset screen, followed by authorized board scheduling and complete
native77/HBM14/model validation; it does not itself pass the gate.

Board performance, full-subset speedup and gate improvement remain
**not-measured**. The frozen contract and historical failures are unchanged.
