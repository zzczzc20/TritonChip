# Bounded execution of real Torch candidates

`tools/r40/autotune_candidate_worker.py` connects the existing isolated
`tnca.autotune.run_measurement` supervisor to standard Torch compilation and
execution. It does not select a winner, activate `TNCA_TUNE` in the standard
path, or certify an R40 gate.

The worker uses the predeclared block widths 16 and 32, original frozen point
definitions, seed 0, one warmup and one steady sample. Physical requests must
name the frozen HBM or holdout suite, an immutable software-tree identity,
the complete image SHA-256, three cores/outlets, and a requested HBM window
of at least 1 GiB. Results must show numerical correctness, no host fallback,
zero tensor staging, an actual multi-PC allocation and complete device-clock
endpoints. Compiler bundles retain the graph, wrapper, IR and compiled
artifacts; their recorded hashes are checked before publishing events.

Always invoke the worker through the existing graph budget and isolated
supervisor. All candidates for a graph share the same maximum 600-second
budget; host time enforces that deadline and never ranks candidates. A fresh
`--events` path is required. The corresponding `.raw.json`, `.info.json`,
`.error.json` and `.bundle` paths preserve failures and provenance. Internal
candidate execution disables recursive tuning and optional counter reads.
Reserved controls cannot be overwritten through `base_knobs`.

The compiler, runtime, benchmark package and scheduler model must resolve
inside the same immutable checkout. Their actual origins and hashes are
recorded. This prevents an inherited `PYTHONPATH` or installed compiler from
silently invalidating the software-tree label.

## Validation limits

The final worker was exercised on the unchanged original tiny add point with
final-candidate RTL geometry. Width 16 passed numerical, artifact and endpoint
validation. Width 32 did not complete before the shared 600-second deadline;
its process group was terminated and no completed result is claimed. The
supervisor's nonzero exit and partial summary remain intact. Current-source
and foreign-compiler checks, reserved-control refusals and timing/source/image
mutations are retained alongside the actual run.

Earlier draft failures are also retained: a stale compiler import, an expected
mutation refusal escaping the validation supervisor, and an initial supervisor
import-order error. See `report.json` and `raw.tar.gz` for exact identities.

These are plumbing checks, not the full native77, HBM14, frozen holdout or model
acceptance runs. Functional RTL mode cannot establish a physical clock or image
and must not supply board rankings. After physical qualification and the
unscaled Torch checkpoint, evaluate the fixed search policy on the full frozen
inventories and replay the chosen configurations using the contract's formal
sampling and comparison protocol. Do not relabel internal fixed-candidate
records as standard-path `TNCA_TUNE=1` results.
