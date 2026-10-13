# TritonChip FPGA Harness handoff

This document is for the collaborator who will validate a new RTL/PR on a
real U50 or U280. The goal of the handoff is a fresh, reproducible board
measurement: a new implementation produces an XCLBIN, that exact XCLBIN is
loaded onto the selected card, ArchBench runs through the physical-board
backend, and the returned JSONL contains real completion cycles and counters.

What each stage does, every check and how to read the results are in
[`README.md`](README.md); this is the checklist for running one.

The Harness is an orchestration and evidence layer. It does not decide
whether a change is timing-011, timing-012, FMA, LSU, reduction, or another
optimization. Those are candidate labels in the manifest. The board profile
and the measured evidence determine whether the result is accepted.

## What must be available on the worker

Run the Harness on the FPGA server that owns the card. The coordinator Mac
does not need Vivado or XRT when using SSH dispatch.

For a U50 worker, the profile currently expects:

- a Linux checkout at the exact candidate commit;
- the U50 platform and the profile's Vitis environment;
- Vivado/Vitis, XRT, `xclbinutil`, `xbutil`;
- a Python environment with PyTorch, Triton and the `tnca` package installed
  (`spec/r38/torch.lock.json`), named by `TORCH_PYTHON` or by
  `board.archbench.python` in the manifest. The system `python3` that runs the
  harness does not need them and on the FPGA hosts does not have them;
- the U50 card visible through XRT and an explicit BDF in the programmer
  environment when the host has more than one card.

For a U280 worker, also verify:

- Vitis 2023.2 and the U280 platform
  `xilinx_u280_gen3x16_xdma_1_202211_1`;
- the U280 XRT device, at the PCIe BDF `flow/fpga/boards.sh` records for
  `u280` -- or the manifest's `board.bdf` if this host's card is elsewhere;
- the checkout contains the board-selectable flow (`flow/fpga/boards.sh` with
  a `u280)` case, and `flow/fpga/program.sh`).

The card is never guessed. The flow's programmer loads only when the device
at that BDF reports the shell interface UUID `boards.sh` expects for the
board, so a program operation cannot land on a different card and still look
like a successful command.

The card is shared. Loading an image replaces whatever another user had
loaded; agree a window with them before a board stage, and do not kill
another user's process.

Before a real run, stop or reconcile stale Vivado/Vitis/XRT processes owned by
the worker account. Do not kill another user's process. The Harness acquires
one full implementation/board lease within its state directory and records
the host/tool/platform snapshot in `environment.json`.

## Candidate preparation

The candidate should be committed before dispatch. From the candidate checkout:

```bash
git status --short
git rev-parse HEAD
git show --stat --oneline HEAD
```

The manifest should name the exact commit instead of relying on `HEAD` when a
result will be compared with another PR. A minimal implementation-only
manifest is:

```json
{
  "schema_version": 1,
  "candidate_id": "pr-new-fma-01",
  "target": "u280-r38-pr5",
  "mode": "full",
  "clock_mhz": 100,
  "source_commit": "<FULL_COMMIT_SHA>"
}
```

For a complete board run, add the board block. Use `fpga-u50` with a U50
profile and `fpga-u280` with a U280 profile; the Harness rejects a mismatch.

```json
{
  "schema_version": 1,
  "candidate_id": "pr-new-fma-01-u280",
  "target": "u280-r38-pr5",
  "mode": "full",
  "clock_mhz": 100,
  "source_commit": "<FULL_COMMIT_SHA>",
  "board": {
    "backend": "fpga-u280",
    "profile": "core-vrf-sync-v2",
    "program": true,
    "conformance": true,
    "archbench": {
      "tiers": ["tiny", "small"],
      "jobs": 1,
      "counters": true,
      "timeout_s": 21600
    }
  }
}
```

`board.profile` must be the device profile of the image the candidate builds.
`plan` prints the check (`board_measurement.image_profile`): `mismatches` must
be empty, and a non-empty `more_specific` means a newer profile describes the
image better than the one named. At this revision the default build is
`core-vrf-sync-v2`.

Use `program: false` only when the worker has already loaded the exact XCLBIN
being tested. It still runs conformance and ArchBench, and it records the
identity of what it measured: the programmer's read-only `--status` report is
kept as the `loaded_image` stage and the card's xclbin UUID as
`loaded_xclbin_uuid`. The runtime refuses a card whose image is not the run's
XCLBIN, so a different image fails conformance rather than being measured. Do
not use this option to bypass programming for a different candidate image.

## Preflight and implementation

On the worker, source the toolchain and run the target doctor first:

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh   # U280 example
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  doctor --target u280-r38-pr5 --adapter u280 --json
```

The doctor must pass the repository flow checks, profile clocks/divider,
platform paths, and vendor-tool checks. A doctor failure is an environment or
flow-contract failure; do not start a long implementation until it is fixed.

Give the doctor the manifest and it also checks the board stage that manifest
asks for -- the programmer, XRT, and the ArchBench interpreter:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  doctor --adapter u280 --manifest candidate-u280.json --json
```

`submit` makes the same checks before it starts Vivado and ends in `ENV_FAIL`
if one fails, rather than linking for hours and failing at the card.

Inspect the resolved plan before submission:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  plan candidate-u280.json --adapter u280 --json
```

Run the complete implementation and board stages locally on the worker:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  submit candidate-u280.json \
  --adapter u280 \
  --state-dir "$HOME/.tritonchip-fpga-harness" \
  --timeout 28800 \
  --json
```

The implementation gate must produce a non-empty XO and XCLBIN, successful
package/link return codes, the expected clock, clean routed timing and DRC
reports, and the exact artifact hashes. The board gate then runs programmer,
conformance and ArchBench while holding the full board lease.

## SSH dispatch from a coordinator

The coordinator can submit the same manifest without having FPGA tools:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" dispatch \
  u280-worker candidate-u280.json \
  --remote-repo /srv/tritonchip-pr6 \
  --adapter u280 \
  --state-dir /srv/tritonchip-harness \
  --timeout 28800 \
  --ssh-timeout 300 \
  --output dispatch.json \
  --json
```

The dispatch request contains the manifest and operation envelope on stdin.
The remote endpoint is:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo /srv/tritonchip-pr6 \
  worker --stdin
```

The remote worker invokes the normal local `doctor`/`submit` code. It returns
transport status, the Harness result, and hashes/tails for worker logs. SSH
reachability or a zero SSH exit code is never sufficient evidence of a board
measurement.

The remote checkout must already contain the requested commit. Dispatch does
not silently fetch, merge, or modify the candidate checkout. If the checkout
is dirty, use `--allow-dirty` only for an explicitly documented experiment;
the Harness then records a diff and source snapshot.

## How to obtain real cycles

After a successful implementation and programming stage, ArchBench is invoked
with the physical backend (`fpga-u50` or `fpga-u280`) and `--counters`. The
TNCA backend obtains completion information from the board protocol for each
kernel launch. The raw record must contain:

- `evidence_level: "measured-board"`;
- `status: "pass"`;
- `time.unit: "cycles"` and positive `time.device`;
- a measured `time.clock_hz` matching the target profile;
- `counters.done_cycles` and `counters.done_retired`;
- the requested `perf_*` counters when counter collection is enabled.

The Harness checks `done_cycles == time.device` for each ArchBench point. The
`done_cycles` value is the completion latency used for ArchBench comparisons.
`perf_cycles` and `perf_*` values are attribution counters; they are retained
for bottleneck analysis and are not substituted for completion latency.

The following are not new board measurements:

- `mock` adapter output;
- `measured-rtl` records from a simulator;
- a historical PR JSONL copied into the run directory;
- routed timing alone without an XCLBIN loaded and a kernel completed.

## Separate implementation and board windows

If the implementation window and card-access window cannot overlap, submit a
full implementation manifest containing the board configuration with
`--skip-board`. The run records the manifest and exact XCLBIN without
programming a card. Then run:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  submit candidate-u280.json --adapter u280 \
  --state-dir "$HOME/.tritonchip-fpga-harness" \
  --timeout 28800 --skip-board --json
```

After the implementation finishes, run the board stage:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  measure RUN_ID \
  --state-dir "$HOME/.tritonchip-fpga-harness" \
  --timeout 21600 \
  --json
```

`measure` reuses the recorded XCLBIN and manifest. It refuses to measure if
the profile changed after implementation, so a later board run cannot silently
use a different flow contract. It also refuses a dirty checkout, and a
checkout that differs from the implemented commit outside
`tools/tritonchip_fpga_harness/`: the runtime and ArchBench that drive the
card come from the checkout `measure` is run in, and the measurement records
which one that was (`measurement.software`).

A run built from a manifest with no `board` block is measured by giving the
board block then:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  measure RUN_ID --manifest candidate-u280.json \
  --state-dir "$HOME/.tritonchip-fpga-harness" \
  --timeout 21600 \
  --json
```

## Evidence and acceptance

The run directory is normally:

```text
<state-dir>/runs/<RUN_ID>/
  run.json
  manifest.json
  environment.json
  result.json
  artifact_validation.json
  artifacts/<exact-xo-and-xclbin>
  board_measurement.json
  board/conformance.json
  board/archbench.jsonl
  archbench_summary.json
  logs/
```

Export the self-contained packet after the run:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  export RUN_ID \
  --state-dir "$HOME/.tritonchip-fpga-harness" \
  --output ./evidence \
  --json
```

A complete real-board handoff requires all of the following:

1. candidate commit and profile hash are recorded;
2. implementation status is `PASS` with a non-empty exact XCLBIN;
3. programming check and programming command both pass, and the measurement
   says `reconfigured: true` -- or `false` with the reason understood: the
   card already held this exact image, XRT downloaded nothing, and the device
   was not reset;
4. conformance passes with `--require-measured`;
5. every requested ArchBench point passes with `measured-board` evidence;
6. `done_cycles`, `done_retired`, clock and `perf_*` counters are present;
7. `board_measurement.json`, raw JSONL, logs and evidence manifest are
   exported;
8. the report identifies board, BDF, platform, Vitis/XRT versions and the
   worker hostname.

Report the following fields back to the optimization agent:

```text
candidate_id:
source_commit:
board/profile/backend:
worker_hostname:
run_id:
result_status:
xclbin_sha256:
archbench_summary:
cycles_by_tier:
done_cycles_total:
done_retired_total:
counter_totals:
evidence_directory:
failure_or_limitations:
```

## Failure handling

Use the result status rather than inferring from a log fragment. A failed
board stage still writes `board_measurement.json`, with `failed_stage` and the
stages that did run:

- `ENV_FAIL`/doctor failure: the checkout's flow does not match the target,
  or the worker's toolchain, platform, XRT or board setup is not ready --
  including an ArchBench interpreter without PyTorch, Triton or `tnca`.
  Nothing was built. `result.json` lists the failed checks under
  `preflight_failed` (flow and host) or `board_preflight` (board stage);
- `BUILD_FAIL`: package/link/artifact failure;
- `TIMING_FAIL`: routed timing or DRC acceptance failure;
- `BOARD_FAIL`: the board profile contradicts the image, or programming or
  conformance failed;
- `ARCHBENCH_FAIL`: one or more measured points/counters invalid;
- `ORPHANED`: controller disappeared during a vendor process.

For an interrupted vendor run, inspect first:

```bash
python3 tools/tritonchip_fpga_harness/cli.py --repo "$PWD" \
  reconcile RUN_ID --state-dir "$HOME/.tritonchip-fpga-harness" --json
```

Only use `--terminate` for a process owned by the current worker account and
after confirming that it belongs to this run. Preserve the run directory and
logs even when the result is a failure; they are part of the review evidence.

## Harness self-check before handoff

On a machine without FPGA tools, verify the Harness implementation itself:

```bash
PYTHONPATH=tools python3 -m unittest discover \
  -s tools/tritonchip_fpga_harness/tests -p 'test_*.py'
python3 -m py_compile tools/tritonchip_fpga_harness/*.py \
  tools/tritonchip_fpga_harness/adapters/*.py
bash -n flow/fpga/program.sh
```

These checks prove the Harness lifecycle and validators. They do not prove that
a new PR has been implemented, programmed, or measured on a real FPGA. That
claim requires the worker-side flow and the exported `measured-board` evidence
described above.
