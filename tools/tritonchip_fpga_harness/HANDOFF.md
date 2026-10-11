# TritonChip FPGA Harness handoff

This document is for the collaborator who will validate a new RTL/PR on a
real U50 or U280. The goal of the handoff is a fresh, reproducible board
measurement: a new implementation produces an XCLBIN, that exact XCLBIN is
loaded onto the selected card, ArchBench runs through the physical-board
backend, and the returned JSONL contains real completion cycles and counters.

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
- Vivado/Vitis, XRT, `xclbinutil`, `xbutil`, Python and the ArchBench/TNCA
  dependencies;
- the U50 card visible through XRT and an explicit BDF in the programmer
  environment when the host has more than one card.

For a U280 worker, also verify:

- Vitis 2023.2 and the U280 platform
  `xilinx_u280_gen3x16_xdma_1_202211_1`;
- the U280 XRT device and its PCIe BDF;
- `TNCA_BDF` is set, or the manifest contains `board.bdf`;
- the checkout contains the U280 board-selectable flow and
  `flow/fpga/u280/program.sh`.

The programmer refuses to guess a U280 card. This is intentional: a successful
program operation on a different card would invalidate the cycles while still
looking like a successful command.

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
    "bdf": "0000:65:00.1",
    "profile": "core-vrf-sync-v1",
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

Use `program: false` only when the worker has already loaded the exact XCLBIN
being tested and that loaded-image identity is documented. It still runs
conformance and ArchBench. Do not use this option to bypass programming for a
different candidate image.

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
use a different flow contract.

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
3. programming check and programming command both pass;
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

Use the result status rather than inferring from a log fragment:

- `ENV_FAIL`/doctor failure: worker toolchain, platform, XRT or board setup;
- `BUILD_FAIL`: package/link/artifact failure;
- `TIMING_FAIL`: routed timing or DRC acceptance failure;
- `BOARD_FAIL`: programming or conformance failure;
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
bash -n flow/fpga/u280/program.sh
```

These checks prove the Harness lifecycle and validators. They do not prove that
a new PR has been implemented, programmed, or measured on a real FPGA. That
claim requires the worker-side flow and the exported `measured-board` evidence
described above.
