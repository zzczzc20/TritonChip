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
plans a run on a laptop and executes it on an FPGA host. The Vivado flow and
the card need not be on one host: `pipeline` builds on one server and measures
on another (§11.2).

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
11. [Remote servers: dispatch, and a pipeline over two](#11-remote-servers-dispatch-and-a-pipeline-over-two)
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
| `u280` | Alveo U280, kernel in SLR1 | `xilinx_u280_gen3x16_xdma_1_202211_1` | 100 MHz → 100 MHz (÷1) | 2023.2 by default; 2022.2 or 2023.2 accepted |
| `u280-multicore-r40` | Alveo U280, R40's three-core machine: one core per SLR, three HBM outlets (`package_xo.tcl` variant `multicore-hbm`) | `xilinx_u280_gen3x16_xdma_1_202211_1` | 90 MHz → 90 MHz (÷1) | 2023.2 |
| `u50` | Alveo U50 | `xilinx_u50_gen3x16_xdma_5_202210_1` | 100 MHz → 100 MHz (÷1) | 2022.2 |

Where Vitis is installed is the host's business, not the card's. The target's
`vitis_root` and `platforms_root` are the card host's paths; a host that keeps
them elsewhere sets `FPGA_HARNESS_VITIS_ROOT` and `FPGA_HARNESS_PLATFORMS_ROOT`
(a hosts file does it for a pipeline, §11.2). `vitis_versions` lists the
versions the card accepts images from, and a root outside it is refused
(`vitis_version`, §3.3). A run records the roots it used and whether the host
or the target named them (`environment.json`, `tool_roots`).

A target is named for its card (and, on the U280, for the machine the flow
builds for it: one core, or a generation's three-core machine) and nothing
else, because its file lives beside the flow it describes. A commit that moves the flow to
another clock changes `link.sh`, `package_xo.tcl`, `ooc.xdc` and the target
file together; if it forgets the target, the checks of §3.3 fail and nothing
starts. There is never a second target for "the flow as it used to be".

`u280-multicore-r40` builds only described machines (`image.device`, §5.1):
the three-core top's own defaults are no machine anyone measured. Its
`device_platform` is what the top and the runtime fix -- three cores, 256-bit
HBM outlets, 4 GiB of HBM, the memory controls and cache maintenance -- and a
description that differs from it, or from the target's board, variant or
kernel clock, is refused in `plan`. Each generation's machine fixes its own
clock (R40's closes at 90 MHz; R41's, at 100, gets a target of its own).

### 3.2 What runs

The implementation is two commands of the repository's own flow, run with the
checkout as working directory and the run's `artifacts/` as build directory:

```text
package   vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs <repo> <artifacts> [<part> [<variant>]]
          → artifacts/tnca_u50.xo                                    log: artifacts/package.log

link      flow/fpga/link.sh <artifacts>
          → artifacts/tnca_<board>.xclbin, link.log, xclbin.info     log: artifacts/adapter-link.log
```

The U280 target passes the card's part to the packager and sets
`TNCA_BOARD=u280`, `XILINX_VITIS`, `XILINX_PLATFORMS` and `VPP_JOBS` for the
link, which is how `flow/fpga/boards.sh` selects the platform and the SLR1
placement; `u280-multicore-r40` also passes the variant (`multicore-hbm`), which
the packager records beside the XO and `link.sh` links with the three-core
connectivity and placement. Every command is given the target's kernel clock
as `TNCA_KERNEL_MHZ`. The kernel and its name (`tnca_u50`) are the same for both cards,
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
| `vitis_version` | the root's version (its last path component) is one the target accepts |
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

**A named profile as the image (`image.profile`).** With
`"image": {"profile": "<name>"}` the build is that profile. `package_xo.tcl`
sources the Tcl file `TNCA_IMAGE_CONFIG` names (r40's mechanism, since
`d1a00dd`) and sets each parameter on the packaged core, printing
`TNCA_IMAGE_PARAM name=value`; the wrapper and the U50/U280 top pass the
switches through to `tnca_fpga_top`. The harness writes that file
(`image.py`) the way simulation reads the profile: the values of
`rtl/common/tnca_uarch_defaults.vh`, then the profile's `elaboration` and
`implementation.rtl_parameters`, every switch the wrapper exposes set
explicitly. It refuses, in `plan`, a profile with more than one core or a
parameter the wrapper cannot carry, rather than build another machine than the
one simulated; a build whose package log does not report every parameter is
`BUILD_FAIL` (`image_unapplied`). The run keeps `image_config.tcl` and
`image_config.json` (the parameters and the hashes of the files they came
from), and the board stage takes the image's profile from there:
`board.profile` must be `image.profile`.

**A described machine as the image (`image.device`).** With
`"image": {"device": "<name>"}` the build is the machine
`spec/devices/<name>.json` describes: the per-core profile, the uncore (the
NUCA geometry, its pipelines, the outlets) and the platform. The parameters
are generated by the checkout's own `python/tnca/devices.py` -- loaded from the
checkout, never from an installed `tnca` -- the same generator the agent
harness's board-top simulation and the SystemC model's configuration use, so
the image, the simulation and the model are one machine. A checkout whose
staged profile copy (`python/tnca/data`, which git ignores) differs from
`spec/r1_profiles.json` is refused rather than built from the stale copy.
`image_config.json` records the device, its platform, every parameter, the
description's fingerprint and the hashes of the files they came from;
`board.profile` is the per-core profile.

Without `image`, the flow takes no profile. The image is whatever the RTL's
parameter defaults elaborate to -- on `main`, `core-vrf-sync-v2` -- so the
harness compares the named profile with those defaults before it touches the
card:

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

**Which compiler compiles the points.** ArchBench finds TNCA through the
installed `tnca` package's Triton entry point, but the runtime and the
compiler it imports must be the checkout's: conformance and ArchBench run with
the checkout's `python/` and `compiler/python/` first on `PYTHONPATH`
(recorded as `pythonpath` in `board_measurement.json`), and the board
preflight refuses an interpreter that still imports `tnca` or `tnca_cc` from
elsewhere (`board_archbench_source`). Until `969201b` it did not: on the U280
host the package is an editable install of an older checkout, and every board
stage there compiled with that tree whatever the checkout was (§13).

`board.archbench.points` names points instead of tiers (`--point <ids...>`):
an evaluator's hidden set, compared with its own simulation of the same
points. The manifest takes one or the other.

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
| `export <run>` | writes the evidence packet; `--light` leaves out v++'s working tree | `--state-dir`, `--output`, `--light` |
| `import` | installs a run built on another host from its light packet, every file's hash checked (§11.2) | `--state-dir`, `--from-tar FILE\|-` |
| `pipeline <manifest>` | builds on one server of a hosts file and measures on another (§11.2) | `--hosts`, `--build-host`, `--board-host`, `--timeout`, `--poll`, `--skip-board`, `--build-run`, `--no-sync`, `--adapter mock`, `--output` |
| `sync <host>` | puts a clean checkout of a commit on a server, from a git bundle | `--hosts`, `--commit` |
| `dispatch <host> <manifest>` | sends a manifest to a worker over SSH (§11.1) | `--remote-repo`, `--state-dir`, `--timeout`, `--ssh-timeout`, `--skip-board`, `--output` |
| `worker --stdin` | the remote end of `dispatch` and `pipeline`: `doctor`, `submit`, `status`, `export`, `measure` | |
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
| `image` | `{"profile": "<name>"}`: build that profile instead of the RTL defaults (§5.1). Full mode only |
| `board.archbench` | `{"tiers": [...]` or `"points": [...]`, `"jobs": 1, "counters": true, "timeout_s": ..., "python": "..."}`, or `{"command": [...]}`. Absent: ArchBench does not run |

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

## 11. Remote servers: dispatch, and a pipeline over two

### 11.1 One server: `dispatch`

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

### 11.2 Two servers: `pipeline`

The Vivado flow wants a fast CPU and memory to spare; the card is wherever the
card is. The same commit linked in 103 minutes on a build server with Vitis
2022.2 (2026-10-17) and in 213 on the card host with 2023.2 (2026-10-10), and
the 2022.2 image loaded on the card (§13). That is one link on each side, with
both the machine and the Vitis version different. So the flow can run on one server and the
board stage on another, with a controller -- any machine whose SSH reaches
both -- doing what one `submit` does:

```text
 controller                     build server                    card server
 ──────────                     ────────────                    ───────────
 sync ── git bundle ──────────► mirror.git → checkouts/<sha>
      ── git bundle ────────────────────────────────────────►  mirror.git → checkouts/<sha>
 doctor ─────────────────────►  flow + Vivado/Vitis checks
        ──────────────────────────────────────────────────────►  board-stage checks
 build ──────────────────────►  submit --skip-board, detached
 (polls status) ◄────────────   PASS / TIMING_FAIL / BUILD_FAIL
 export ─────────────────────►  light packet: image, records, routed reports
 transfer   ◄── tar ──────────  packet
            ── tar ────────────────────────────────────────────►  import: every hash checked
 measure ────────────────────────────────────────────────────►  measure RUN_ID, detached
 (polls status) ◄──────────────────────────────────────────────  PASS / BOARD_FAIL / ARCHBENCH_FAIL
```

```bash
python3 $H/cli.py --repo "$PWD" pipeline candidate-u280.json \
  --hosts ~/.config/tritonchip-fpga-harness/hosts.json \
  --build-host build --board-host card --timeout 28800 --output pipeline.json --json
```

**The hosts file** is the operator's and stays outside the repository
([`examples/hosts.example.json`](examples/hosts.example.json), `hosts.py`). Per
server: an `ssh` destination that the controller's own SSH configuration
resolves (user, key, jump host: the harness adds none), its `roles` (`build`,
`board`), the `python` (3.9 or later) that runs the harness there, a
`workspace`, a `state_dir`, and for a build server the `vitis_root` it keeps.
`environment` passes further variables to every command the controller runs
there: on a build server a `PATH` with Vivado's and Vitis's `bin` first (a
remote command does not read the login shell's profile), on a card server the
ArchBench interpreter. Every value is one shell-safe token that does not start
with `-`; anything else is refused when the file is read. The harness passes
`--` before the destination, and quotes for the remote shell every word it did
not take from the hosts file.

**What each step does.**

| Step | Where | |
|---|---|---|
| `sync` | each server | a clean checkout of the commit at `<workspace>/checkouts/<sha>`, a worktree of the bare `mirror.git`. The controller sends the commit as a git bundle, so the servers need no access to the repository's remote. A checkout already there is used if it is at the commit and clean, and refused otherwise; nothing on a server is deleted |
| `doctor` | both | the build server's flow contract and vendor tools, with its own Vitis root (§3.3); the card server's board-stage checks only (§5.2), so it needs no Vivado. Both checkouts must be the commit, clean |
| `build` | build server | `submit --skip-board`, started in a session of its own: the link does not depend on the controller's SSH connection, and the controller polls `status` |
| `export` | build server | a light packet: the run's records and logs, the XO and XCLBIN, the routed text reports and the implementation logs the DRC verdict is read from, without the rest of v++'s working tree (about 1 GB a link) |
| `transfer` | controller | the packet as a tar stream from one server into `import` on the other; the servers never connect to each other |
| `import` | card server | accepts only plain files under one run directory, every file's hash as the packet lists it, and an image that is the one the build's own result recorded; a run already present is accepted only if it is the same packet |
| `measure` | card server | `measure RUN_ID`, detached, polled: §5 on the image the build server built, driven by the card server's checkout of the same commit, with the board block of the manifest given to `pipeline` (which must name the same candidate, target, mode and commit). A server with both roles and one state directory measures the build in place, with no packet |

**What it leaves.** The build's run directory on the build server (with v++'s
working tree), the imported run with its board records on the card server
(`imported.json` says where it came from), and on the controller the record
`pipeline` prints: the commit, the servers, each step's timing and outcome,
the build's verdict and slack, the image's hash on both sides, and the board
stage's result. Its `status` is the board stage's, or the build's when that
did not pass; `PIPELINE_FAIL` is a step that could not be done, with the
reason in `error`.

**Picking up.** A build or measurement survives the controller. Each
detached step is an attempt with its own record and log on the server
(`<state_dir>/requests/<run>.<submit|measure>.<n>`), and a measurement writes
its attempt's name into the run's record when it starts, on that server's
clock. `status` reports `STARTING`, `RUNNING`, `MEASURING`, a verdict,
`ORPHANED` (the process is gone), or `FAILED_TO_START` with the tail of the
child's log; a measurement's verdict is reported only once the record names
the latest attempt, so the imported build's `PASS` never answers for the
board. `pipeline --build-run RUN_ID` attaches to a build that is running or
done (checking only the build server's checkout, since a full doctor would
count that build's own Vivado against it), refuses a run that is not this
manifest's candidate, target, mode and commit, and carries on from there: a
failed transfer or board stage does not cost a link, and a board stage still
running from an earlier controller is attached to rather than started again.
`--skip-board` stops after the build; `--no-sync` uses the checkouts as they
are; `--adapter mock` plays the build (and its board stage measures a
placeholder image), to exercise the servers without Vivado.

**How long the controller waits.** `--timeout` is each remote command's limit
on the server (package, link, load, conformance; ArchBench has its own
`board.archbench.timeout_s`). The controller waits for the build what its two
commands may take, and for the board stage what its three commands and
ArchBench may take, each plus an hour; then it stops waiting and says so, and
the stage can be attached to again.

**One run at a time, per server.** Each server's harness takes its own lease
(§3.4): a build on the build server and a board stage on the card server can
overlap, two links on one build server cannot.

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

**A U280 image built on another server** (2026-10-17, by hand with the flow's
own scripts, before `pipeline` existed). `b5c606d` packaged and linked on a
build server (Core i5-11400, Vivado/Vitis 2022.2, no licence file: 2022.2
grants xcu280 implementation itself): 103 minutes, against 213 for the same
commit on the card host with 2023.2. Place 22 min (68), route 34 (67), kernel
synthesis 12 (26). Routed ap_clk WNS +0.138 ns (2023.2: +0.177), design WNS
+0.016 ns on both (the platform's own path), clock 100 MHz. On the card host,
`program.sh --check` passed every gate with that XCLBIN (its interface UUID is
the shell's) and `--go` loaded it: the device reports its UUID, `DATA_CLK`
100 MHz. No kernel was run on it.

**`pipeline` between the two servers** (2026-10-17, commit `8082597`). With a
played build (`--adapter mock`) and a board stage that loaded nothing and ran
no conformance: sync created a checkout on each server from an 11 MB bundle,
the build server's doctor used its own Vitis 2022.2 (`vitis_source: host`),
the packet's 9 files were checked on the card server, and the board stage read
the card as it was (the image above) from a checkout of the same commit; 3
minutes in all. The real U280 doctor on the build server passed all 22 checks
with Vivado, `v++` and `xclbinutil` from 2022.2.

**A real candidate through `pipeline`** (2026-10-17, commit `ef4c4fc`, whose
`rtl/` and `flow/` are the agent task's base: the RTL defaults,
core-vrf-sync-v2, at 100 MHz). `PASS` end to end in 113 minutes. The link on
the build server took 105 minutes (synthesis 25, place 23, route 34): design
WNS +0.016 ns, ap_clk WNS +0.138 ns, no failing endpoint of 584 497, clock
100 MHz. Export, transfer and import took under a minute (the hashes held),
and the board stage 5 minutes: the load replaced the image on the card
(`reconfigured: true`, now `8c9c8fca…`), conformance 5 of 5, ArchBench `tiny`
27 of 27 `measured-board`, 4 459 740 cycles -- every point equal to the
measurement of the same design linked on the card host with 2023.2
(`pr5-latest-b5c606d-u280-100mhz`). Two SSH handshakes to the card host timed
out on the way and were retried. The review that followed (four lenses, each
finding re-checked) found that a board stage could not be repeated, that an
imported build's PASS could answer for the board, that a refused measure
damaged the run's record, that `--build-run` was not checked against the
manifest, and that run ids and option-like values were not refused; all are
fixed and tested (`tests/test_split_hosts.py`).

**The board stage compiled with an installed tree, not the checkout**
(found 2026-10-18). On the U280 host, `/usr/bin/python3` has an editable
install of `tnca` 0.2.0 pointing at an older checkout (`4b7767e`, with local
changes), and ArchBench imported `tnca` and `tnca_cc` from it: every board
stage on that host compiled its points with that tree's compiler while
`software.relation` said `same-commit` -- the real candidate above, the 2023.2
measurement it was compared with, and the measurements in the table below.
Those agree with one another because they all used it. It showed when
the agent harness certified core-vrf-sync-v5 (its README §19): the card
counted, point by point, the simulated cycles of a compiler with the m.zero
lowering the task base removes. On the same image, with the checkout's own
packages first, every point equals the base's simulation. Fixed in `969201b`
(§5.5).

**core-vrf-sync-v5 as the image, through `pipeline`** (2026-10-18, twice,
`image.profile`, for the agent harness's certification of its task base). Each
link took about two hours on the build server; the package log reported all 17
parameters; ap_clk WNS +0.085 ns, design WNS +0.016 ns, no failing endpoint of
590 869, 336 716 LUTs and 367 BRAM tiles, identical both times. On the card:
`reconfigured: true`, conformance passed, ArchBench five named points
(`board.archbench.points`) `measured-board`. The first board stage compiled
with the installed tree (above); the second, with `969201b`, counted exactly
the cycles the evaluator's simulation of the same commit counted, point by
point.

**A named profile packaged** (2026-10-17, package only, on the build server,
the image build of `d1a00dd` before it was committed). Without
`TNCA_IMAGE_CONFIG` the packaged core is what it was: v2's values (`NUM_RCTX`
2, `RCTX_SWITCH` 0, `MTX_ABUF` 0). With the file `image.py` writes for
core-vrf-sync-v5, the log reports all 17 parameters and the packaged
`component.xml` carries them: `NUM_RCTX` 4, `RCTX_SWITCH` 2, `MTX_ABUF` 1,
`LSU_INFLIGHT` 2, `FPU_OVERLAP` 1, `HAS_TENSOR_XPOSE` 1 among them -- r40's
values for the signed-off image. Both packages exit 0.

On the card host (Alveo U280, Vitis 2023.2, XRT 2.16), from clean checkouts, on
2026-10-11 and 12. The image is the link of `b5c606d`, whose tree is that of
the commit this harness was added on.

| What | Result |
|---|---|
| The suite, 93 tests (131 with the split-server, image and compiler-source tests of 2026-10-17 and 18, run on macOS) | passes on macOS (Python 3.9) and on the host (Python 3.10) |
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
- **It does not fetch code.** A run is of the checkout as it stands. `sync`
  puts a checkout on a server from a bundle the controller sends, never from a
  remote, and never deletes or rewrites one that is there.
