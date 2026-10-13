# TritonChip FPGA Harness

Give the harness a commit and a target. It runs the repository's own FPGA flow
on that commit, judges the result from what the tools left behind, loads the
image on the card, measures ArchBench on it, and keeps all of it in a run
directory outside the checkout.

It is an execution and evidence layer. It adds no RTL and changes no flow
script: packaging is `flow/fpga/package_xo.tcl`, linking is `flow/fpga/link.sh`,
loading is `flow/fpga/program.sh`, the measurements are `tools/r36/conformance.py`
and `archbench.run`. What the harness adds is the order, the checks before and
after each step, one lease so two runs cannot share a card, and a record that
says exactly which commit, image and software a number came from.

It needs only the Python standard library (3.9 or later), so the same code
plans a run on a laptop and executes it on an FPGA host.

## Contents

1. [At a glance](#1-at-a-glance)
2. [Quick start](#2-quick-start)
3. [The FPGA flow it drives](#3-the-fpga-flow-it-drives)
4. [Judging a build](#4-judging-a-build)
5. [On the card: load, conformance, ArchBench](#5-on-the-card-load-conformance-archbench)
6. [Reading the ArchBench result](#6-reading-the-archbench-result)
7. [The run directory and the evidence packet](#7-the-run-directory-and-the-evidence-packet)
8. [Commands](#8-commands)
9. [The manifest](#9-the-manifest)
10. [Out-of-context runs](#10-out-of-context-runs)
11. [SSH worker dispatch](#11-ssh-worker-dispatch)
12. [Running a real candidate](#12-running-a-real-candidate)
13. [What has been run](#13-what-has-been-run)
14. [What the harness does not do](#14-what-the-harness-does-not-do)

## 1. At a glance

```text
 candidate.json
      │
      ▼
   doctor ── is this host, and this checkout's flow, ready for the target?    (starts nothing)
      │
      ▼
    plan ─── what would run, on which commit, as which image                  (starts nothing)
      │
      ▼
   submit    one lease for the whole run
      │
      │   IMPLEMENTATION                            BOARD  (only with a `board` block)
      │   ┌─────────┐  ┌──────┐  ┌─────────────┐    ┌─────────┐  ┌─────────────┐  ┌───────────┐
      │   │ package │─►│ link │─►│ judge the   │───►│ program │─►│ conformance │─►│ ArchBench │
      │   │ vivado  │  │ v++  │  │ build       │    │ --check │  │ 5 cases on  │  │ tiers,    │
      │   │         │  │      │  │ clock,      │    │ --go    │  │ the card    │  │ counters  │
      │   │         │  │      │  │ timing, DRC │    │         │  │             │  │           │
      │   └─────────┘  └──────┘  └─────────────┘    └─────────┘  └─────────────┘  └───────────┘
      │    .xo          .xclbin   PASS, or           the card     conformance      archbench.jsonl
      │                           TIMING_FAIL /      holds the    .json            archbench_summary
      │                           BUILD_FAIL         image                         .json
      ▼
  <state-dir>/runs/<RUN_ID>/     run.json  result.json  artifacts/  logs/  board/
      │
      ▼
   export ── a copy of the run with a SHA-256 for every file
```

`submit --skip-board` stops after the build is judged; `measure RUN_ID` runs the
board half later, on the image that run built. `validate` judges a build that
already exists without starting any tool.

## 2. Quick start

From the repository root. None of this needs an FPGA tool:

```bash
H=tools/tritonchip_fpga_harness
python3 $H/cli.py doctor --target u280 --json                          # what this host lacks
python3 $H/cli.py plan $H/examples/candidate-u280.json                 # what would run
python3 $H/cli.py submit $H/examples/candidate-u280.json --adapter mock --skip-board
PYTHONPATH=tools python3 -m unittest discover -s $H/tests -p 'test_*.py'
```

`--adapter mock` plays the build without Vivado, to exercise the lifecycle;
without it a target is built by its own adapter. The tests need no vendor
tool either: most run in throwaway git repositories with each vendor step
played by a command, and a few read this checkout's flow scripts, profile
table and recorded board results, skipping where a checkout has none.

On the U280 host the real thing is one command:

```bash
python3 $H/cli.py --repo "$PWD" submit candidate-u280.json \
  --state-dir "$HOME/.tritonchip-fpga-harness" --timeout 28800 --json
```

Start it from a script file, or keep the command line short: the flow's
programmer treats any process whose command line matches
`(xbutil|xbmgmt).*program` as a concurrent load and refuses.

A run prints its result as JSON and exits 0 only for `PASS`. Its state is in
`<state-dir>/runs/<RUN_ID>/`; the default state directory is
`.tritonchip-fpga-harness` beside the checkout, never inside it.

## 3. The FPGA flow it drives

### 3.1 Targets

A target is one card, as this checkout's flow builds for it: the part and
platform, the kernel and device clocks, the scripts to run, what a build must
produce and what it must prove. It is a file in [`targets/`](targets/), and a
manifest names it.

| Target | Card | Platform | Kernel clock → device clock | Vitis |
|---|---|---|---|---|
| `u280` | Alveo U280, kernel in SLR1 | `xilinx_u280_gen3x16_xdma_1_202211_1` | 100 MHz → 100 MHz (÷1) | 2023.2 |
| `u50` | Alveo U50 | `xilinx_u50_gen3x16_xdma_5_202210_1` | 100 MHz → 100 MHz (÷1) | 2022.2 |

A target is named for its card and nothing else, because its file lives
beside the flow it describes. A commit that moves the flow to another clock
changes `link.sh`, `package_xo.tcl`, `ooc.xdc` and the target file together;
if it forgets the target, the checks of §3.3 fail and nothing starts. There is
never a second target for "the flow as it used to be".

### 3.2 What runs

The implementation is two commands of the repository's own flow, run with the
checkout as working directory and the run's `artifacts/` as build directory:

```text
package   vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs <repo> <artifacts> [<part>]
          → artifacts/tnca_u50.xo                                    log: artifacts/package.log

link      flow/fpga/link.sh <artifacts>
          → artifacts/tnca_<board>.xclbin, link.log, xclbin.info     log: artifacts/adapter-link.log
```

The U280 target passes the card's part to the packager and sets
`TNCA_BOARD=u280`, `XILINX_VITIS`, `XILINX_PLATFORMS` and `VPP_JOBS` for the
link, which is how `flow/fpga/boards.sh` selects the platform and the SLR1
placement. The kernel and its name (`tnca_u50`) are the same for both cards,
so the XO is always `tnca_u50.xo`; only the XCLBIN is named for the board.

The link is not attempted on a missing or empty XO. A command that exceeds
`--timeout` has its whole process group terminated and is recorded with
return code 124.

### 3.3 Before anything runs

A target is a claim about the checkout's flow. `doctor` checks the claim, and
`submit` checks it again before it starts Vivado, so a target cannot be
applied to a tree it does not describe. A run refused here ends `ENV_FAIL`
with the failed checks in `preflight_failed` and in `preflight.json`.

| Check | Reads | Must hold |
|---|---|---|
| `flow_clock_contract` | `EXPECT_MHZ=` in `flow/fpga/link.sh` | the target's kernel clock |
| `package_clock_contract` | `set AP_CLK_HZ` in `flow/fpga/package_xo.tcl` | the same, in Hz |
| `ooc_clock_contract` | `create_clock -period` in `flow/fpga/ooc.xdc` | that clock's period |
| `device_divider_contract` | `DEV_DIV` in `rtl/board/tnca_u50_xo.v` | the target's divider |
| `board_contract` (U280) | `flow/fpga/boards.sh` | has a `u280)` case |
| `board_options_contract` (U280) | `flow/fpga/boards.sh` | carries `--connectivity.slr tnca_u50_1:SLR1` |

Then the host:

| Check | Must hold |
|---|---|
| `vivado`, `v++`, `xclbinutil` | on the `PATH` of the shell that starts the harness |
| `vitis_settings`, `platform` | `<vitis_root>/settings64.sh` and the platform directory exist |
| `existing_vendor_processes` | this account has no `vivado`, `vpl`, `v++` or `loader` already running |
| `build_outside_repo`, `build_parent_writable` | the build directory is outside the checkout and can be written |

A leftover vendor process is reported, never killed. Setting
`FPGA_HARNESS_ALLOW_EXISTING_VENDOR_PROCS=1` turns that one check into a warning.

### 3.4 Which source, and one run at a time

- **Source identity.** A run records the commit, a digest of the tracked tree,
  and whether the working tree is dirty. `source_commit` in the manifest must
  be the checkout's `HEAD`. A dirty tree is refused; with `--allow-dirty` the
  run keeps `source.diff` and a copy of every untracked file.
- **The lease.** A full run holds `leases/full.lock` in the state directory
  from the first vendor command to the last board command. A second `submit`
  or `measure` on the same state directory is refused, so one run's image
  cannot be replaced under another's measurement. The lease is per state
  directory: it coordinates the harness with itself, not with someone loading
  the card by hand.
- **Losing the controller.** Each vendor command leaves `active_process.json`
  (PID, start time, process group, boot ID). If the SSH session dies, `status`
  reports the run `ORPHANED` and `reconcile RUN_ID [--terminate]` finds or
  ends the process group, and only that one.
- **Resources.** Every command's record carries the peak resident memory of
  its process tree and the lowest available memory seen, sampled every two
  seconds; the run records free disk before and after.

## 4. Judging a build

A link that exits 0 has not thereby passed. The harness reads the build's own
output, and the verdict is the conjunction of these checks:

| Check | Evidence | Passes when |
|---|---|---|
| `xo`, `xclbin` | the two artifacts | each exists and is not empty |
| `vpp_rc` | `VPP_RC=` in `link.log` | 0 |
| `link_ok_marker` | `link.log` | carries `LINK_OK` |
| `no_auto_frequency_scaling` | `AUTO-FREQ-SCALING` rules in `link.log` | none, or only those the target allows (U280: rule 08, which reports that the requested clock was kept) |
| `clock_info`, `expected_clock` | the clock-mapping line in `link.log`, and `xclbin.info` | the XCLBIN's entry for kernel clock 0 is read back and is the target's kernel clock |
| `routed_timing_report` | a routed `*timing_summary*.rpt` under `artifacts/` | one is found |
| `timing_clean` | that report's Design Timing Summary | WNS, TNS, WHS, THS and WPWS are all ≥ 0 and no endpoint fails |
| `drc_report`, `drc_clean` | `DRC finished with N Errors` in the implementation run's `runme.log` | found, and N is 0 every time |

**Timing** is read from the routed report's *Design Timing Summary*: the whole
design, every clock and crossing, setup, hold and pulse width. The kernel
clock's own row is not used -- it says nothing about a path into the shell.
`failing_endpoints` is the sum of the setup, hold and pulse-width failing
endpoints.

**DRC.** A Vitis link writes no DRC report. Vivado runs DRC as a precondition
to `opt_design`, `place_design`, `route_design` and `write_bitstream` and logs
each verdict; those lines are the evidence. A link with no verdict anywhere is
not accepted.

### 4.1 The verdict

| `status` | Meaning |
|---|---|
| `PASS` | every stage returned 0, both artifacts exist, every check passed -- and, if a board stage was asked for, it passed too |
| `TIMING_FAIL` | a routed report says the design did not close, or DRC found errors. Also a link that succeeded but left no timing or DRC evidence |
| `BUILD_FAIL` | a stage failed or an artifact is missing for any other reason |
| `OOC_FAIL` | an out-of-context run's own result was not `PASS` (§10) |
| `ENV_FAIL` | this host or this checkout's flow cannot run the target (§3.3), or the board stage asked for could not start here (§5.2). Nothing was built; `preflight_failed` or `board_preflight` lists the checks |
| `BOARD_FAIL` | the image is not the profile named, or loading or conformance failed |
| `ARCHBENCH_FAIL` | ArchBench produced no record, or one that is not an accepted board measurement |
| `FAIL` | the harness itself raised; `error` says what |

A missed timing is `TIMING_FAIL` even though `v++` exits non-zero for it;
`BUILD_FAIL` is a link that never got as far as a routed report.

### 4.2 Where the numbers are

`result.json` carries the whole judgement under `artifact_validation` (also
written alone as `artifact_validation.json`):

```json
"artifact_validation": {
  "ok": true,
  "observed_clock_mhz": 100,
  "mapped_clock_name": "DATA_CLK",
  "checks": [{"name": "timing_clean", "ok": true, "detail": "..."}],
  "timing": {
    "report": ".../artifacts/tmp/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt",
    "scope": "design",
    "metrics": {
      "wns_ns": 0.016, "tns_ns": 0.0, "tns_failing_endpoints": 0, "tns_total_endpoints": 584460,
      "whs_ns": 0.006, "ths_ns": 0.0, "ths_failing_endpoints": 0, "ths_total_endpoints": 582791,
      "wpws_ns": 0.0, "tpws_ns": 0.0, "tpws_failing_endpoints": 0,
      "failing_endpoints": 0
    }
  },
  "artifacts": {"xclbin": {"path": "...", "size": 56396523, "sha256": "a2f3a930..."}}
}
```

Those are the numbers of the U280 link of `b5c606d`: at a 10 ns kernel clock
the design closes with 16 ps of setup slack and 6 ps of hold slack over
584 460 endpoints. `scope` is `design` for a Vivado report.

To judge a build that already exists, without starting a tool:

```bash
python3 $H/cli.py validate --target u280 --build-dir <state-dir>/runs/<RUN_ID>/artifacts --json
```

`validate` needs the build's `run.json` beside it, so an old or unrelated
XCLBIN cannot be reported as a candidate's. `--historical` inspects one
without lineage and marks the result `HISTORICAL`, which is not a pass: the
command exits 1 for it.

### 4.3 What is kept but not judged

The harness accepts or rejects a build; it does not rank builds. These are in
the run directory, and in the evidence packet, for whoever compares candidates:

| | Where |
|---|---|
| Utilisation after routing | `artifacts/tmp/reports/link/imp/impl_1_full_util_routed.rpt`, and `impl_1_slr_util_routed.rpt` per SLR |
| Every clock, path group and crossing | the routed timing report named in `timing.report` |
| `v++`'s own account of the link | `artifacts/link.log`, `artifacts/vpp_logs/` |

## 5. On the card: load, conformance, ArchBench

A `board` block in a full manifest adds the second half:

```json
"board": {
  "backend": "fpga-u280",
  "profile": "core-vrf-sync-v2",
  "program": true,
  "conformance": true,
  "archbench": {"tiers": ["tiny", "small"], "jobs": 1, "counters": true, "timeout_s": 21600}
}
```

It runs only after the build is judged `PASS`, under the lease the build took.

### 5.1 Which machine is on the card

`board.profile` is the device profile conformance and ArchBench drive the card
as (`spec/r1_profiles.json`). It is required, and never defaulted: the profile
decides what the compiler emits, so the same image measured under two profile
names gives two sets of cycle counts.

The flow takes no profile argument. The image is whatever the RTL's parameter
defaults elaborate to -- on `main`, `core-vrf-sync-v2` -- so the harness
compares the named profile with those defaults before it touches the card:

- a parameter the profile declares and the RTL contradicts stops the run
  (`BOARD_FAIL`, `failed_stage: image_profile`);
- a profile that matches while another matches *and* pins down more switches
  is reported as `image_profile.more_specific`, in `plan` and in the
  measurement. That is an older name on a newer image.

### 5.2 Can the board stage start at all

A link takes hours, and the board stage follows it. Before the build, the
harness checks what that stage needs on this host; a failure is `ENV_FAIL`
and no vendor tool starts. `doctor --manifest candidate.json` runs the same
checks alone, and `measure` runs them before it takes the lease.

| Check | Must hold |
|---|---|
| `board_program_script` | the target's programmer exists and is executable |
| `board_xbutil`, `board_xclbinutil` | XRT's `xbutil`, and an `xclbinutil` from Vitis, XRT or `PATH` |
| `board_xrt_library` | `$XILINX_XRT/lib/libxrt_core.so` (default `/opt/xilinx/xrt`) |
| `board_conformance_tool`, `board_archbench_runner` | `tools/r36/conformance.py`, `archbench/run.py` |
| `board_archbench_python` | the ArchBench interpreter imports PyTorch and Triton and has `tnca` as Triton's `cpu` backend |

ArchBench needs PyTorch, Triton and the installed `tnca` package
(`spec/r38/torch.lock.json`), which this repository keeps in an environment of
its own. So ArchBench runs under `board.archbench.python` if the manifest
names one, else `$TORCH_PYTHON` (the Makefile's variable), else the
interpreter running the harness. Conformance needs none of that.

A step the manifest replaces with its own command is not checked.

### 5.3 Loading

```text
program_check   flow/fpga/program.sh --check <xclbin>     reads only; loads nothing
program         flow/fpga/program.sh --go    <xclbin>
```

The flow's programmer takes its card from `TNCA_BOARD` and
`flow/fpga/boards.sh`, which the harness sets from the target. `board.bdf`
overrides the address `boards.sh` records (exported as `BDF`). Its gates are
the flow's, not the harness's: the file parses as an XCLBIN for this platform
and shell interface, the device is present and reports that interface, no
context or client has it open, no other load is running, and the FPGA and HBM
temperatures are in range. After the load it requires the device to report the
file's UUID.

**A load that succeeds is not always a load that happened.** XRT does not
download an image the card already holds; it returns success and the device
keeps its state. The measurement therefore says what the load did:

| Field | |
|---|---|
| `programmed` | the manifest asked for a load |
| `replaced_xclbin_uuid`, `loaded_xclbin_uuid` | what the card held before, and holds after |
| `reconfigured` | `true`: another image was put on the card. `false`: it was the one already there, nothing was written, the device was not reset. `null`: the programmer's log does not say |

With `"program": false` nothing is loaded and the card is measured as found.
The harness then records the programmer's read-only `--status` report (stage
`loaded_image`) and the card's UUID; the runtime refuses a card whose image is
not the run's XCLBIN, so a different image fails conformance rather than being
measured.

### 5.4 Conformance

```text
python3 tools/r36/conformance.py --backends fpga-<board> --profile <board.profile>
        --xclbin <xclbin> --require-measured --json board/conformance.json
```

Five cases on the card: session lifecycle, transfers, compatibility, a launch
that completes with a cycle count and the oracle's value, and a launch that
traps and leaves the session usable. `--require-measured` makes "the hardware
could not be reached" a failure instead of a not-applicable. A failure here is
`BOARD_FAIL`, and ArchBench does not run.

### 5.5 ArchBench

```text
<python> -m archbench.run --platform tnca --backend fpga-<board> --profile <board.profile>
         --backend-args {"xclbin": "<xclbin>"} --tier <tiers...> --jobs <jobs>
         --out board/archbench.jsonl [--counters]
```

ArchBench compiles each point from PyTorch through Inductor and Triton to TNCA
and launches it on the card. The tiers that fit the board are `tiny`
(27 points, about five minutes on the U280) and `small` (25 points). The file
it writes is kept untouched; `board/archbench_logs/` has one log per point.

A record is accepted as a board measurement only if all of this holds:

- `platform.backend` is the board the target names;
- `evidence_level` is `measured-board` -- a simulator's `measured-rtl` is not;
- `status` is `pass`, which includes the numerical check against the reference;
- `time.unit` is `cycles` and `time.device` is a positive integer;
- `time.clock_hz` is the target's device clock, within 1 %;
- with counters: `done_cycles` and `done_retired` are present, and
  `done_cycles` equals `time.device`.

One record failing any of them is `ARCHBENCH_FAIL`, with the point and the
reason in `archbench_summary.json` under `failures`.

### 5.6 Build now, measure later

```bash
python3 $H/cli.py submit candidate-u280.json --skip-board --timeout 28800 --json
python3 $H/cli.py measure RUN_ID --state-dir "$HOME/.tritonchip-fpga-harness" --timeout 21600 --json
```

`--skip-board` keeps the board block and the exact XCLBIN without loading the
card. `measure` then runs the board half on that XCLBIN and no other.

- A run submitted with no `board` block can be given one at that point:
  `measure RUN_ID --manifest candidate-u280.json`. The manifest must name the
  same candidate, target and mode, and no commit but the one the run was built
  from. It is kept as `board_manifest.json`; the run's own manifest is not
  rewritten.
- `measure` drives the card with the runtime, conformance suite and ArchBench
  of the checkout it is run from, which may have moved on since the build. It
  refuses a dirty checkout, and one that differs from the implemented commit
  anywhere outside `tools/tritonchip_fpga_harness/`. `--allow-dirty` and
  `--allow-source-mismatch` override that and are recorded: `software.relation`
  is `same-commit`, `same-tree`, `harness-only-diff`, `differs` or `unknown`.
- `measure` replaces the board records of an earlier attempt on the same run.
  The verdict of the link is kept as `implementation_status`.

## 6. Reading the ArchBench result

Three files, from raw to digested:

| File | What it is |
|---|---|
| `board/archbench.jsonl` | ArchBench's own records, one line per point, untouched |
| `archbench_summary.json` | the harness's check of them, and the totals |
| `board_measurement.json` | the whole board stage: image, profile, software, stages, conformance, and that summary |

`archbench_summary.json`, from the U280 run of §13:

```json
{
  "status": "PASS", "evidence_level": "measured-board", "backend": "fpga-u280",
  "records": 27, "passed": 27, "failed": 0,
  "cycles_total": 4459740, "cycles_by_tier": {"tiny": 4459740},
  "done_cycles_total": 4459740, "done_retired_total": 1541692,
  "counter_totals": {"perf_cycles": 4445561, "perf_issued": 1541692, "perf_stall_total": 2102292,
                     "perf_x_hazard": 991887, "perf_x_frontend": 801577, "perf_x_lsu": 624309,
                     "perf_x_vrf": 195648, "perf_x_txf": 150591, "...": "..."},
  "points": [
    {"point_id": "model.llama_decode/tiny/fp16-b1-p16", "family": "llama_decode", "tier": "tiny",
     "status": "pass", "device_cycles": 1244739, "device_seconds": 0.01244739, "launches": 42,
     "done_cycles": 1244739, "done_retired": 376394,
     "per_launch": [{"kernel": "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0",
                     "grid": [1, 1, 1], "programs": 1, "cycles": 6842}, "..."]}
  ],
  "failures": []
}
```

- **`device_cycles`** is the point's latency: the sum, over its kernel
  launches, of the cycles from the device accepting a launch to the core going
  idle. It is what ArchBench compares, and it has no host time, upload or
  download in it. `per_launch` breaks it down by kernel.
- **`done_cycles`** is the same quantity as the device's completion counter
  reported it; the harness requires the two to agree. **`done_retired`** is
  instructions retired.
- **`perf_*`** are the hardware counters (`spec/r38/counters.json`): where the
  cycles went -- issued, or stalled on a register hazard, the front end, the
  LSU, the VRF. They explain a latency and are never substituted for it.
- **`device_seconds`** is `device_cycles / clock_hz`. A change that keeps the
  cycles and raises the clock shows here and nowhere else.

To print the cycles per point:

```bash
python3 -c "
import json, sys
for p in json.load(open(sys.argv[1]))['points']:
    print(f\"{p['device_cycles']:>10}  {p['point_id']}\")
" <state-dir>/runs/<RUN_ID>/archbench_summary.json
```

**Comparing two candidates** is not done by the harness. Give the two runs'
`board/archbench.jsonl` to the repository's scorer,
`python -m archbench.report A.jsonl B.jsonl --tier tiny`, under the ArchBench
interpreter. A comparison means something only between runs of the same
tiers, the same `board.profile` and the same software commit; each run's
`board_measurement.json` records all three.

In the runs of §13, one image under one profile and one software commit gave
the same cycle count for every point across three measurements and a reload. So a point that differs between two runs is
something to explain, not to average.

## 7. The run directory and the evidence packet

```text
<state-dir>/runs/<RUN_ID>/
  run.json                    manifest, source identity, the target and its hash, host, status
  preflight.json              the flow-contract and host checks, as run before the build
  manifest.json               the manifest as submitted
  environment.json            host, tool versions, platform and script fingerprints
  result.json                 status, stages, artifact_validation, measurement, artifact hashes
  artifact_validation.json    the judgement of the build (§4.2)
  artifacts/
    tnca_u50.xo  tnca_<board>.xclbin  xclbin.info
    package.log  adapter-link.log  link.log
    tmp/  vpp_logs/           v++'s working tree: the Vivado runs and the routed reports
  board_manifest.json         a board block given at measure time, if one was
  board_measurement.json      the board stage (§5, §6)
  archbench_summary.json
  board/
    conformance.json  archbench.jsonl  archbench_logs/
  logs/board/
    program_check.log  program.log  loaded_image.log  conformance.log  archbench.log
  source.diff  source_snapshot/     only for an --allow-dirty run
```

Each stage in `result.json` has its argv, return code, start and finish
times, and log. A failed board stage still writes `board_measurement.json`,
with `failed_stage` and the stages that did run.

```bash
python3 $H/cli.py export RUN_ID --state-dir "$HOME/.tritonchip-fpga-harness" --output ./evidence --json
```

`export` copies the run and writes `evidence-manifest.json` with a SHA-256
for every file. That directory is the thing to attach to a claim.

## 8. Commands

`cli.py [--repo <checkout>] <command> ...`. Every command prints JSON; the
exit status is 0 for success or `PASS`, 1 for a failed check or run, 2 for a
harness or manifest error.

| Command | Does | Options |
|---|---|---|
| `doctor` | checks the flow contract and the host for a target; with `--manifest`, that manifest's board stage too | `--target` or `--manifest` |
| `plan <manifest>` | resolves a manifest: source identity, clocks, artifacts, board stage, profile check. Starts nothing | |
| `submit <manifest>` | runs it | `--state-dir`, `--timeout`, `--run-id`, `--skip-board`, `--allow-dirty`, `--adapter mock` to play the build |
| `measure <run>` | the board stage of an existing run | `--state-dir`, `--timeout`, `--manifest`, `--allow-dirty`, `--allow-source-mismatch` |
| `status <run>` | a run's record and result; marks a run whose controller is gone `ORPHANED` | `--state-dir` |
| `reconcile <run>` | finds the vendor process a lost run left; `--terminate` ends its process group | `--state-dir`, `--terminate` |
| `validate` | judges an existing build directory | `--target`, `--build-dir`, `--historical`, `--output` |
| `export <run>` | writes the evidence packet | `--state-dir`, `--output` |
| `dispatch <host> <manifest>` | sends a manifest to a worker over SSH (§11) | `--remote-repo`, `--state-dir`, `--timeout`, `--ssh-timeout`, `--skip-board`, `--output` |
| `worker --stdin` | the remote end of `dispatch` | |
| `budget reserve\|settle\|status` | a token ledger for a managed worker; a usage that is unknown stays reserved rather than counting as zero | `--ledger`, `--limit` |

## 9. The manifest

```json
{
  "schema_version": 1,
  "candidate_id": "fma-retime-01",
  "target": "u280",
  "mode": "full",
  "clock_mhz": 100,
  "source_commit": "<full SHA, or HEAD>",
  "hypothesis": "what this candidate is meant to show",
  "board": { }
}
```

| Field | |
|---|---|
| `candidate_id`, `target`, `mode` | required. `target` is a file in `targets/` (§3.1); `mode` is `full` or `ooc` |
| `clock_mhz` | if given, must be the target's kernel clock |
| `source_commit` | if given, must be the checkout's `HEAD`. Name the full SHA for anything that will be compared |
| `hypothesis` | free text, kept with the run |
| `board.backend` | `fpga-u50` or `fpga-u280`; must be the target's card |
| `board.profile` | required: the device profile of the image (§5.1) |
| `board.bdf` | the card's PCI address, if not the one `boards.sh` records |
| `board.program` | `true` (default), `false`, or `{"check_command": [...], "command": [...]}` |
| `board.program_script` | another programmer, relative to the checkout, in place of the target's |
| `board.conformance` | `true` (default), `false`, or `{"command": [...], "result_file": "..."}` |
| `board.archbench` | `{"tiers": [...], "jobs": 1, "counters": true, "timeout_s": ..., "python": "..."}`, or `{"command": [...]}`. Absent: ArchBench does not run |

A command given in the manifest is an argv list, run without a shell. In it
`{repo}`, `{run_dir}`, `{xclbin}`, `{backend}`, `{profile}`, `{board}` and
`{clock_mhz}` are substituted, and `TNCA_HARNESS_RUN_DIR`,
`TNCA_HARNESS_XCLBIN` and `TNCA_BOARD` are in its environment. An unknown
field anywhere in a manifest is an error.

## 10. Out-of-context runs

`"mode": "ooc"` is for a synthesis-only experiment on part of the design. The
module, its constraints and its script are the experiment's, so the manifest
says how to run it:

```json
{
  "candidate_id": "fma-ooc-01", "target": "u280", "mode": "ooc", "clock_mhz": 100,
  "ooc": {
    "top": "u_mac",
    "command": ["vivado", "-mode", "batch", "-source", "{repo}/flow/fpga/ooc_fma.tcl",
                "-tclargs", "{repo}", "{run_dir}", "{top}", "{clock_mhz}"],
    "artifacts": ["artifacts/fma_ooc.dcp", "artifacts/fma_timing.rpt"],
    "result_file": "ooc_result.json"
  }
}
```

The command must write `ooc_result.json` with `{"status": "PASS"}` or
`{"status": "FAIL"}`. The harness checks the command, that file and the
declared artifacts, and records their hashes. An OOC run takes no lease and
never produces an XCLBIN; `OOC_FAIL` is its failing status.

## 11. SSH worker dispatch

A coordinator without Vivado, XRT or a board account can send a manifest to a
prepared checkout on an FPGA host, which runs the same harness there:

```bash
python3 $H/cli.py --repo "$PWD" dispatch u280-worker candidate-u280.json \
  --remote-repo /srv/tritonchip --state-dir /srv/tritonchip-harness \
  --timeout 21600 --output dispatch.json --json
```

Only the manifest crosses SSH, on stdin. The remote checkout must already be
at the commit; dispatch does not fetch, merge or modify it. `dispatch.json`
records the destination, the exact argv, the return code, the remote result,
and a hash and tail of each remote log. The logs, the XCLBIN and the evidence
stay on the worker until exported there. A zero SSH exit status is not a
result: read `remote.result.status`.

## 12. Running a real candidate

1. **Check out the candidate's commit on the FPGA host**, in a clean tree.
   The harness does not fetch.
2. **Agree a window for the card.** It is shared: a load replaces whatever
   anyone had on it, and the lease (§3.4) covers the harness only. Never end
   another user's process.
3. **Write the manifest** (§9) with the full SHA and a `board` block whose
   `profile` is the image the checkout builds (§5.1). `plan` shows the check
   under `board_measurement.image_profile`: `mismatches` and `more_specific`
   must both be empty.
4. **`doctor --manifest candidate.json`.** Every check must pass (§3.3, §5.2).
   The usual gap on a host is the ArchBench interpreter: set `TORCH_PYTHON` or
   `board.archbench.python`.
5. **`submit`**, from a script file (§2), with a `--timeout` that allows a
   link of several hours. Or `submit --skip-board` now and `measure` in the
   card's window (§5.6).
6. **Read the result** (§4.1). For a complete one: `status` is `PASS`;
   `artifact_validation.timing.metrics` has the slack; in
   `board_measurement.json`, `reconfigured` is `true`, conformance passed and
   every ArchBench record is `measured-board`.
7. **`export`** the run (§7), and quote the run id, the commit and the
   XCLBIN's SHA-256 with any number taken from it.

If the session is lost during a link, `status RUN_ID` and then `reconcile`
(§3.4); do not start a second run on top of it.

## 13. What has been run

On `acf3019` (Alveo U280, Vitis 2023.2, XRT 2.16), from clean checkouts, on
2026-10-11 and 12. The image is the link of `b5c606d`, whose tree is that of
the commit this harness was added on.

| What | Result |
|---|---|
| The suite, 93 tests | passes on macOS (Python 3.9) and on the host (Python 3.10) |
| `doctor` for the U280 target, alone and with a board manifest | every check passes |
| `validate` on three real links | two `PASS`: design WNS +0.016 ns, WHS +0.006 ns, no failing endpoint of 584 460 and of 566 734, DRC 0 errors. One, a link that missed timing, rejected with its numbers: WNS −0.079 ns, 304 endpoints |
| `measure` without a load, the card already holding the image | conformance 5 of 5; ArchBench `tiny` 27 of 27 `measured-board`; 4 459 740 cycles |
| `measure` with a load of that same image | `xbutil` reported success and nothing was written: the device's request counter read 62336 afterwards, not 2. Such a load is now recorded as `reconfigured: false` (§5.3) |
| `measure` with a load that changes the image, both ways between two links | `reconfigured: true` and a request counter of 2 each time; conformance 5 of 5 after each; ArchBench `tiny` 27 of 27, 4 459 740 cycles, every point equal to the earlier measurements |

Not run:

- **a link started by this harness.** Every board run above used a link built
  earlier, so `submit` has run end to end only with a played build; its build
  half is covered by `validate` on real output, its board half by `measure`;
- **the `small` tier**;
- **anything on the U50 host.**

## 14. What the harness does not do

- **It does not rank or compare candidates.** It accepts or rejects one build
  and one measurement. Slack, utilisation and cycles are recorded for a
  comparison someone else makes (§4.3, §6).
- **It does not collect utilisation or power.** The routed utilisation
  reports are kept; nothing parses them.
- **It does not choose the image.** It builds what the checkout's flow builds
  by default, and checks that the profile named for the board is that image.
- **It does not coordinate with other users of the card.** The lease is per
  state directory. Someone loading the card by hand is outside it.
- **It cannot force a reload of the image already on the card**, and says so
  (`reconfigured: false`) instead of reporting a load.
- **It does not fetch code.** A run is of the checkout as it stands.
