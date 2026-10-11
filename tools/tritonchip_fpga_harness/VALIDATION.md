# Harness validation record

This record describes what the first deliverable proves. It does not turn an
existing FPGA artifact into a new implementation run.

## Harness hardening

Timing cases are generic fixtures; no commit-specific behavior is encoded in
the Harness. The executable contract now checks profile/adapter compatibility,
kernel/device clock arithmetic, expected artifact names, and manifest fields.
Each run records a host/tool/platform snapshot and a resolved run specification.

Full-flow validation requires the routed timing report and DRC report in
addition to XO/XCLBIN, link markers, and clock readback. Results distinguish
`PASS`, `TIMING_FAIL`, and `BUILD_FAIL`. A lost controller leaves durable
process identity metadata; `reconcile --terminate` can recover the vendor
process group after checking PID start time and boot ID. `export` copies the
complete run tree and emits `evidence-manifest.json` with hashes for every
file. OOC capacity is no longer reserved in target profiles; a real OOC run
must provide an explicit manifest-level command and declared artifacts.

## Local checks

From the TritonChip repository root:

```text
PYTHONPATH=tools python3 -m unittest discover \
  -s tools/tritonchip_fpga_harness/tests -p 'test_*.py'
```

The current suite passes **36 tests**. It covers manifest/source identity,
full and mock OOC lifecycle, token reservation/settlement, owned-process
resource sampling, package/link command planning, U50 shell syntax, missing-XO
preflight, 150 MHz AP clock / divide-by-four device-clock contract, artifact
hashes, the `VPP_RC=0` plus missing-XCLBIN failure case, and the U280 PR5
platform/part/clock/divider/artifact contract. Python compilation and
`bash -n flow/fpga/link.sh` also pass.

On the development Mac, `doctor --adapter u50` returns a blocking result because
Vivado, Vitis, `v++`, `xclbinutil` and the U50 platform are absent. No vendor
process is started by that check.

## U50 host checks

The same new directory was copied temporarily to the U50 host `RSCL` and run
against the existing checkout at commit
`3ea50a8ba4a1f9d81861418a12193ebe749d26f9`:

- The tool/platform checks passed: Vivado/Vitis 2022.2, `v++`, `xclbinutil`
  and `xilinx_u50_gen3x16_xdma_5_202210_1` were found. The final doctor result
  correctly **blocked full admission** because the same user still had an old
  Vivado process with a deleted working directory. Harness does not kill that
  unowned process; it reports `existing_vendor_processes` and waits for a
  clean host.
- adapter dry-run: **PASS**; it resolved the existing `package_xo.tcl` and
  `link.sh` commands, created no build directory and launched no vendor tool.
- the public `validate` CLI on the existing
  `u50_build/pkg150/tnca_u50.xclbin`: **PASS**;
  mapped clock `DATA_CLK` was read as **150 MHz** and the artifact hash was
  `04fcf5f81748f6f78b9f53afc6276606cfd569367cd54803637131fc71a868`.

No new full implementation was launched in this validation pass. The existing
baseline artifact is evidence that the adapter can parse the real U50 output;
it is not claimed as a build produced by this Harness. A fresh full run should
be submitted only from a clean candidate snapshot and within the one-full-job
resource policy.

The real U50 adapter currently exposes the full package/link endpoint. A
module-level OOC runner is reserved in the profile but deliberately rejected
until an explicit module OOC flow and its constraints are added.

## PR5 review and compatibility check

The remote PR5 head was fetched as
`74f841cee06d7f509716f9e192950e687b26e17d`.  A detached clean worktree was
used for all checks; the existing dirty `fpga-dev` checkout was not used as the
candidate.

The `u50-r38-pr5` profile records the flow contract visible in that snapshot:
100 MHz kernel clock, 100 MHz device clock, and `DEV_DIV=1`.  The adapter
cross-checks `link.sh`, `package_xo.tcl`, `ooc.xdc`, and the wrapper before
admission.  The old `u50-legacy` profile is rejected against the same tree
before any vendor tool is started because it expects `EXPECT_MHZ=150` and
divider 4.  With the PR5 profile, local plan and adapter dry-run pass, record both package/link commands,
and create no build artifacts.  The PR5 worktree also passed the Harness mock
full lifecycle with the exact source SHA; this is a lifecycle test only.

The U280 evidence directory in PR5 is deliberately rejected by the U50
`validate` endpoint: it contains summary logs and clock text, not a U50 XO and
XCLBIN.  This prevents the U280 100 MHz result from being mislabeled as a U50
implementation.  On `RSCL`, after sourcing Vitis 2022.2, the profile checks,
tools and U50 platform were found, but the real full submission was blocked by
the pre-existing same-user Vivado/loader processes.  No new vendor process or
XCLBIN was produced.

The PR5 U280 result itself is therefore review evidence, not a Harness-produced
head artifact.  Its README says the loaded image came from `8d8e4e9` (timing-011)
while the PR5 head adds timing-012 afterward; a fresh head build is required
before attributing the measured U280 image to the full PR5 SHA.  The new
`u280-r38-pr5` adapter/profile now validates the U280 platform, Vitis 2023.2
environment, the common `boards.sh` U280 case and its literal
`--connectivity.slr tnca_u50_1:SLR1` option, `tnca_u280.xclbin`, and the allowed
clock-rule set; raw routed timing reports are still required for a complete
timing verdict.

For the exact clean PR5 head, the local Harness checks were run against a
detached worktree at `74f841cee06d7f509716f9e192950e687b26e17d`:

- `plan --adapter u280` passed with a clean source tree and the exact SHA;
- U280 `doctor` passed every repository contract check (100 MHz flow/package/
  OOC clock, divider 1, board case, SLR1 option, and U280 platform selection), then blocked
  only because this Mac has no Vivado/Vitis/U280 platform;
- U280 dry-run emitted the expected package command with the U280 part and the
  board-selectable link command with `TNCA_BOARD=u280`/`VPP_JOBS=8`;
- the mock full lifecycle produced `tnca_u50.xo`, `tnca_u280.xclbin`, and a
  timing summary while retaining the exact source identity.

The real U280 host named by PR5 (`acf3019.ece.ust.hk`) was not reachable in
this session: SSH accepted the connection endpoint and then closed it before a
shell was created. Consequently no real Vivado/Vitis process was started and
no new PR5 XCLBIN or routed timing report was produced. Full acceptance still
requires a clean U280 host reservation and the raw implementation artifacts.

The board evidence stage is now implemented after the link stage. A full
manifest can request programming, `tools/r36/conformance.py --require-measured`,
and ArchBench with counters. The stage retains the exact XCLBIN hash and raw
JSONL, rejects simulated evidence or records without `done_cycles`, and
preserves the stall counters for attribution. A separate `measure RUN_ID`
command allows implementation and board windows to be scheduled independently
while using the XCLBIN recorded by the original run.

The reusable SSH worker boundary is also covered by the Harness: `dispatch`
sends a validated request over stdin, and the remote `worker --stdin` endpoint
invokes the same local `doctor`/`submit` implementation on the selected U50 or
U280 host. The response includes transport status, a remote result, and hashes
plus tails for worker logs; SSH reachability alone is never reported as board
success.
