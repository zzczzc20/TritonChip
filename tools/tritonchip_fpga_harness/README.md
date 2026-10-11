# TritonChip FPGA Harness

This directory is the complete, self-contained harness.  It is an execution
platform for RTL candidates: a candidate manifest selects a target/profile and
the harness records the source identity, allocates a full or OOC slot, runs the
existing U50 flow, validates outputs, and exports a small evidence package.

The harness does not change RTL or the existing `flow/fpga` scripts.  The
`mock` adapter makes the lifecycle testable on a machine without Vivado.  The
`u50` adapter invokes the repository's existing `package_xo.tcl` and
`link.sh`; the target host supplies `XILINX_VITIS` and `XILINX_PLATFORMS`.

For handing a new PR to another FPGA worker, start with
[`HANDOFF.md`](HANDOFF.md). It is the operational checklist for real-board
programming and measured ArchBench cycles.

## Quick start

From the TritonChip repository root:

```bash
python3 tools/tritonchip_fpga_harness/cli.py doctor --target u50-legacy --json
python3 tools/tritonchip_fpga_harness/cli.py plan candidate.json
python3 tools/tritonchip_fpga_harness/cli.py submit candidate.json --adapter mock
python3 tools/tritonchip_fpga_harness/cli.py status RUN_ID --json
python3 tools/tritonchip_fpga_harness/cli.py export RUN_ID --output evidence
python3 tools/tritonchip_fpga_harness/cli.py validate \
  --build-dir /scratch/u50_build/pkg150 --json
```

`submit` is synchronous and leaves `run.json`, `manifest.json`,
`environment.json`, stage logs, artifacts, and `result.json` under a sibling
`.tritonchip-fpga-harness/runs/RUN_ID` directory by default. Pass
`--state-dir` to choose another external state directory. Pass `--allow-dirty`
only when intentionally testing an uncommitted tree; the run records its diff
digest and a source snapshot. A normal candidate must be committed so that
`source_commit` is reproducible.

The harness is target-agnostic: a target profile describes clocks, board,
platform, scripts, expected artifacts, and acceptance policy. The adapter and
target must match (`u50` with a U50 profile, `u280` with a U280 profile), and
the profile's kernel/device clock/divider arithmetic is checked before a run.
Timing-011 and timing-012 are regression fixtures only; they are not special
cases in the Harness implementation.

Clock and divider settings belong to the target profile.  The original
`u50-legacy` profile is the 150 MHz kernel / 37.5 MHz device configuration
(divider 4).  `u50-r38-pr5` is the PR5 flow contract: 100 MHz kernel and device
clock (divider 1).  The adapter checks both `flow/fpga/link.sh` and
`rtl/board/tnca_u50_xo.v` before admitting a run, so a profile cannot silently
be applied to a mismatched RTL snapshot.

`u280-r38-pr5` selects the PR5 common flow (`flow/fpga/package_xo.tcl` plus
`flow/fpga/link.sh`), the U280 platform, Vitis 2023.2, `TNCA_BOARD=u280`,
and the literal SLR1 placement option in `boards.sh`
(`--connectivity.slr tnca_u50_1:SLR1`). The package artifact remains
`tnca_u50.xo` for compatibility, while the linked artifact is
`tnca_u280.xclbin`. Its profile permits only the documented
`AUTO-FREQ-SCALING-08` advisory; any actual frequency change fails validation.
The adapter passes the U280 part argument required by PR5's board-selectable
package script and rejects a source snapshot without the common board case.

On the U280 host, run the real flow from a clean detached candidate tree:

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
python3 tools/tritonchip_fpga_harness/cli.py doctor \
  --target u280-r38-pr5 --adapter u280 --json
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" submit \
  candidate-u280.json --adapter u280 \
  --state-dir "$HOME/.tritonchip-fpga-harness" --timeout 28800 --json
```

The candidate manifest must name the exact commit under test. The submit gate
requires a clean source tree, one available full slot, and all vendor/platform
checks before it starts Vivado. It writes raw logs and artifacts outside the
checkout; a PASS still requires a non-empty XO/XCLBIN, `VPP_RC=0`, `LINK_OK`,
clock readback, a routed timing report with clean WNS/TNS/WHS/THS and zero
failing endpoints, and a DRC report with zero errors. Results distinguish
`PASS`, `TIMING_FAIL`, `BUILD_FAIL`, and `OOC_FAIL`. A link that produced a
routed report saying the design did not close is `TIMING_FAIL` even though
`v++` exited non-zero; `BUILD_FAIL` is a link that did not get that far.

If an SSH session or controller is lost during a vendor run, the worker leaves
an `active_process.json` record with PID start time, process group, and boot ID.
Inspect or terminate it from a new session with:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" reconcile RUN_ID \
  --state-dir "$HOME/.tritonchip-fpga-harness" --terminate --json
```

The input manifest is intentionally small:

```json
{
  "schema_version": 1,
  "candidate_id": "fma-retime-01",
  "target": "u50-legacy",
  "mode": "full",
  "clock_mhz": 150,
  "source_commit": "HEAD"
}
```

`mode` is `full` or `ooc`. Full mode uses the target profile and the checked-in
package/link flow. OOC is intentionally **not** reserved in the target profile:
the user or optimization agent decides the module mapping for each experiment.
A real OOC manifest supplies an argv-only command and output artifacts, for
example:

```json
{
  "schema_version": 1,
  "candidate_id": "fma-ooc-01",
  "target": "u280-r38-pr5",
  "mode": "ooc",
  "clock_mhz": 100,
  "ooc": {
    "top": "u_mac",
    "script": "flow/fpga/ooc_fma.tcl",
    "command": ["vivado", "-mode", "batch", "-source", "{repo}/flow/fpga/ooc_fma.tcl", "-tclargs", "{repo}", "{run_dir}", "{top}", "{clock_mhz}"],
    "artifacts": ["artifacts/fma_ooc.dcp", "artifacts/fma_timing.rpt"],
    "result_file": "ooc_result.json"
  }
}
```

The command runs without a shell and must write `ooc_result.json` with
`{"status":"PASS"}` or `{"status":"FAIL"}`. The Harness checks the command,
result file, and declared artifacts, then exports their hashes. OOC jobs do
not consume a phantom profile slot; a coordinator may apply its own OOC
concurrency policy. The mock adapter retains a small lifecycle fixture that
produces only an XO and never an XCLBIN.

For a managed worker, token reservations use the persistent ledger outside the
checkout:

```bash
python3 -m tools.tritonchip_fpga_harness budget reserve work-item-01 2000 \
  --ledger /scratch/tritonchip-fpga/token-ledger.json --limit 10000
python3 -m tools.tritonchip_fpga_harness budget settle work-item-01 --actual 1432 \
  --ledger /scratch/tritonchip-fpga/token-ledger.json --limit 10000
```

An unknown provider usage stays reserved instead of being counted as zero. This
ledger controls requests issued by the managed worker; it does not claim hard
control over an external Codex conversation.

`validate` only reads an existing build directory. By default it requires a
neighboring Harness `run.json`, so an old or unrelated XCLBIN cannot be
reported as the current candidate. Use `--historical` when intentionally
checking legacy artifacts; the result is marked `HISTORICAL` rather than
treated as a fresh reproducible run.

`export` copies the complete run tree (logs, reports, source snapshot, XO,
XCLBIN, metadata) and writes `evidence-manifest.json` containing a SHA256 for
every exported file.

## Board execution and ArchBench

Full implementation and board execution are separate gates. Add a `board`
block to a full candidate manifest to program the exact XCLBIN, run the shared
hardware conformance suite, and collect raw ArchBench records:

```json
{
  "board": {
    "backend": "fpga-u280",
    "profile": "core-vrf-sync-v2",
    "program": true,
    "conformance": true,
    "archbench": {"tiers": ["tiny", "small"], "jobs": 1, "counters": true}
  }
}
```

`submit` validates the routed build and then, under the same full lease, runs
the programmer's `--check` and `--go`, conformance with `--require-measured`,
and ArchBench. Both boards use the flow's own programmer,
`flow/fpga/program.sh`, which takes its card from `TNCA_BOARD` and
`flow/fpga/boards.sh` and checks the shell's interface UUID, the device being
idle, and the temperatures before it loads anything. The harness sets
`TNCA_BOARD` from the target profile. `board.bdf` is optional: it overrides the
card `boards.sh` records for that board (exported to the programmer as `BDF`),
for a host whose card sits at another address. A worker may replace any step
with an explicit argv-only command.

`board.profile` is the device profile conformance and ArchBench drive the card
as. The flow takes no profile argument -- the image is whatever the RTL
parameter defaults elaborate to -- so the harness compares the named profile's
parameters (`spec/r1_profiles.json`) with those defaults before it touches the
card:

- a parameter the profile declares and the RTL contradicts stops the run
  (`BOARD_FAIL`, `failed_stage: image_profile`);
- a profile that matches while another profile matches *and* pins down more
  switches is reported as `image_profile.more_specific` in `plan` and in the
  measurement. That is the sign of an older name on a newer image: at this
  revision the default build is `core-vrf-sync-v2`, and `core-vrf-sync-v1`
  would label its cycles as a machine that was not on the card.

ArchBench needs PyTorch, Triton and the installed `tnca` package
(`spec/r38/torch.lock.json`), which this repository keeps in an environment of
its own; the harness needs only the standard library. The ArchBench stage
therefore runs under `board.archbench.python` if the manifest names one, else
`$TORCH_PYTHON` (the Makefile's variable), else the interpreter running the
harness. Conformance needs none of that and runs under the harness's own.

Before a build that is to be followed by a board stage, the harness checks
that the stage can start on this host: the programmer and `xbutil`, XRT's
library, the conformance tool, and that the ArchBench interpreter imports
PyTorch and Triton and has `tnca` as Triton's `cpu` backend. A failure is
`ENV_FAIL` and no vendor tool is started -- the link takes hours and would end
in the same place. `doctor --manifest candidate.json` runs the same checks on
their own, and `measure` runs them before it takes the lease.

The ArchBench stage writes the untouched JSONL plus `archbench_summary.json`.
The summary retains total and per-tier cycles, `done_cycles_total`,
`done_retired_total`, and every counter total. A record is accepted only when
its backend is the requested board, its evidence level is `measured-board`,
its timing unit is cycles, and the requested counters are present.
`done_cycles` is the completion latency used by ArchBench; `perf_*` counters
are retained for stall attribution and are never substituted for it.

Every board stage, passed or failed, leaves `board_measurement.json`: the
stages that ran, the XCLBIN hash, the profile check, and the identity of the
checkout whose runtime drove the card. A failure names its `failed_stage`.
Programming or conformance failing is `BOARD_FAIL`; an ArchBench point that is
missing, simulated, failed, or without its completion counters is
`ARCHBENCH_FAIL`.

If implementation and board access are separate windows, submit the board
manifest with `--skip-board` during the implementation window; this keeps the
board configuration and the exact XCLBIN without loading the card. Run the
second gate later:

```bash
python3 tools/tritonchip_fpga_harness/cli.py submit candidate-u280.json \
  --adapter u280 --skip-board --timeout 28800 --json
python3 tools/tritonchip_fpga_harness/cli.py measure RUN_ID \
  --state-dir "$HOME/.tritonchip-fpga-harness" --timeout 21600 --json
```

A run that was submitted with no `board` block can be given one at that point
with `measure RUN_ID --manifest candidate-u280.json`. The manifest must name
the same candidate, target and mode, and no commit other than the one the run
was built from; the run's own `manifest.json` is left as submitted and the
late block is kept beside it as `board_manifest.json`.

`measure` drives the card with the runtime, conformance suite and ArchBench of
the checkout it is run from, which may have moved on since the build. It
refuses a dirty checkout, and one that differs from the implemented commit
anywhere outside `tools/tritonchip_fpga_harness/`; `--allow-dirty` and
`--allow-source-mismatch` override that and are recorded in
`measurement.software` (`relation`: `same-commit`, `same-tree`,
`harness-only-diff`, `differs` or `unknown`).

The result and `export` packet contain the exact source commit, XCLBIN hash,
board/profile, command logs, conformance JSON, raw ArchBench JSONL, and the
cycle/counter summary.

## SSH worker dispatch

The coordinator does not need Vivado, XRT, or a board account. It can send a
manifest to a prepared U50/U280 checkout and let that host run the same local
Harness, including its environment snapshot, full-board lease, programmer,
conformance and ArchBench stages:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" dispatch \
  u280-worker candidate-u280.json \
  --remote-repo /srv/tritonchip-pr6 \
  --adapter u280 --state-dir /srv/tritonchip-harness \
  --timeout 21600 --output dispatch.json --json
```

Only the manifest crosses SSH over stdin; the remote checkout and state
directory are explicit. The remote worker owns the board account and returns
the Harness JSON result. `dispatch.json` records the destination, exact argv,
return code and remote result, while the full logs and XCLBIN remain in the
remote evidence directory until exported by that worker.
