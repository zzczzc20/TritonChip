# Harness validation record

## On r38, 2026-10-11 (branch `fpga-harness`)

The harness as committed on top of `4b7767e`. What was run, and what was not.

**Offline suite.** 82 tests (85 since) pass on the development Mac (Python 3.9) and on
the U280 host `acf3019` (Python 3.10.12), from a clean checkout in both
places, and leave the checkout clean:

```text
PYTHONPATH=tools python3 -m unittest discover \
  -s tools/tritonchip_fpga_harness/tests -p 'test_*.py'
```

**On the U280 host, read-only.** Nothing below built, loaded or ran anything
on the card.

- `doctor --target u280-r38-pr5 --adapter u280`: every repository contract and
  vendor check passes (Vivado and Vitis 2023.2, the U280 platform, `v++`,
  `xclbinutil`, no vendor process of this account), from a bare SSH shell as
  well as after sourcing `settings64.sh`.
- `doctor --adapter u280 --manifest ...` for a board stage with ArchBench: the
  programmer, `xbutil`, `xclbinutil`, XRT's library, the conformance tool and
  the ArchBench runner are found. `board_archbench_python` **fails**: this
  account's `python3` had no PyTorch at first; after `torch 2.14.0+cpu` and
  `triton 3.8.0` appeared in its user site the check reports them and still
  fails, correctly, because the `tnca` package is not installed there and
  Triton has no `cpu` backend. ArchBench cannot run under this account until
  it is.
- `flow/fpga/program.sh --check` (the programmer the profile now names, with
  `TNCA_BOARD=u280`) on the XCLBIN linked from `b5c606d`: the file parses, the
  platform and interface UUID match, the card is present at `0000:04:00.1` and
  idle, the temperatures are in range. Nothing was loaded.
- `validate` on three real U280 links kept under `~/.tritonchip-fpga-harness/runs`:

  | Run | Result | Design WNS / WHS (ns) | Failing endpoints | DRC |
  |---|---|---|---|---|
  | `pr5-latest-b5c606d-u280-100mhz` | PASS, 100 MHz | +0.016 / +0.006 | 0 of 584 460 | 0 errors, 5 checks |
  | `timing011-u280-full-20261010` | PASS, 100 MHz | +0.016 / +0.006 | 0 of 566 734 | 0 errors |
  | `pr5-u280-full-20261009` | FAIL: `v++` rc 1, no XCLBIN | −0.079 / +0.006 | 304 of 572 249 | — |

  `b5c606d` and `4b7767e` have the same tree, so the first row is the image of
  this branch's base. Before the parser was corrected the same command called
  that run FAIL for a missing DRC report and gave its hold slack as 167 090 ns.

## On the board, 2026-10-12

The first board stage run through this harness's own command, on `acf3019`,
from a clean checkout of this branch (`5e7ad2a`):

```text
cli.py measure pr5-latest-b5c606d-u280-100mhz --manifest board-tiny.json
```

`board-tiny.json` gave the run its board block at measure time
(`fpga-u280`, `core-vrf-sync-v2`, `program: false`, conformance, ArchBench
tiny with counters). Nothing was loaded: the card already held this run's
XCLBIN, left there by an earlier session, and the harness recorded that.

| | |
|---|---|
| Result | `PASS`; `implementation_status` `PASS` |
| Stages | `loaded_image` 0, `conformance` 0, `archbench` 0 |
| Image | card xclbin UUID `78f04922-980e-d3cb-ee78-cf732d6ec13f`, the run's XCLBIN (sha256 `a2f3a930…ab79f1a`) |
| Software | this checkout against the implemented `b5c606d`: `harness-only-diff`, clean |
| Conformance | 5 of 5 cases, `--require-measured` |
| ArchBench tiny | 27 of 27 pass, `measured-board`, 100 MHz |
| Cycles | 4 459 740 (`done_cycles_total` the same); 1 541 692 instructions retired |

`doctor --manifest` passed all 25 checks beforehand, including the ArchBench
interpreter (`/usr/bin/python3`: torch 2.14.0+cpu, triton 3.8.0, `tnca` as
Triton's `cpu` backend). The card held the same image afterwards and the
checkout was still clean.

The same image had been measured the day before by the earlier, uncommitted
copy of the harness, driven as `core-vrf-sync-v1`: 4 434 372 cycles, and 22
of the 27 points differ from the run above (−2.2 % to +2.7 %). The image is
`core-vrf-sync-v2`. The profile a card is driven as changes what the compiler
emits, which is why `board.profile` is required and checked.

### The load

Later the same day the run was measured again with `program: true`, from
`c9f34b6`. Every stage returned 0 and ArchBench gave the same 4 459 740
cycles -- **and nothing had been loaded.** The card already held that XCLBIN,
and XRT does not download an image the device has: it returns success and the
device keeps its state. The device says so itself. Its request counter stood
at 62336 at the conformance run that followed; a device that has just been
configured starts again at 2. That run is a second unprogrammed measurement,
and the harness had recorded it as programmed. An earlier version of this
record called it the load, and was wrong.

The harness now records what a load replaced and whether the image changed
(`replaced_xclbin_uuid`, `loaded_xclbin_uuid`, `reconfigured`), and the load
was tested as what it is, a change of image, in both directions, each through
`measure` from `8d0026d`:

| Run | Image on the card | `reconfigured` | Device counter after | Then |
|---|---|---|---|---|
| `timing011-u280-full-20261010`, with `--allow-source-mismatch` | `78f04922…` → `c4637d43…` | true | 2 | conformance 5 of 5 as `core-vrf-sync-v1` |
| `pr5-latest-b5c606d-u280-100mhz` | `c4637d43…` → `78f04922…` | true | 2 | conformance 5 of 5, ArchBench tiny 27 of 27 `measured-board`, 4 459 740 cycles |

The second row is the board stage whole, in one command: the programmer's
gates, a load that replaced another image, conformance, ArchBench, cycles.
All 27 points have the cycle counts of both earlier measurements of this
image. The first row drove an image built from another commit (`8d8e4e9`)
with this checkout's runtime; the measurement says so (`software.relation:
differs`). The card was left holding the image it held before.

Two things the first attempt at this showed:

- The flow's programmer refuses to load while any process's command line
  matches `(xbutil|xbmgmt).*program`, and the SSH command that started the
  harness did. The harness stopped at `program_check` with `BOARD_FAIL` and
  the card was not touched. Start the harness from a script file or a short
  command line.
- `measure` replaces the board records of the previous attempt in the run
  directory. Copies of the two earlier measurements are kept on the host
  beside the harness's work files.

**Still not done.** No link has been started from this branch: every board
run above used a link the earlier copy built. So `submit` has not
run end to end on a real host, only its two halves -- the build half on a
played flow and `validate` on real output, the board half on the card. The
small tier has not been run, and nothing has been run on the U50 host from
this branch.

The sections below are the record of the harness as first written, against
the `fpga-dev` checkout (`3ea50a8`) and the PR5 head of the time. They are
kept as written; where they name `flow/fpga/u280/program.sh`, a test count,
or an unreachable U280 host, the section above supersedes them.

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
