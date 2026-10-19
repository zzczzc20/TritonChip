# TritonChip Agent Harness

The environment in which CLI coding agents, each driven by a model, optimise
TNCA's compiler, ISA and microarchitecture together, so that agent systems can
be compared on equal terms. It rests on one premise:

> **A score is worth comparing only if the agent cannot reach what produces it.**

This file is the design and the reference, and the one document kept up to
date: what is decided, what is built, what is still open. **§17 is the
authoritative status;** a status mentioned anywhere else must agree with it.
A shareable overview page is generated from this file (`overview/build.py`).

## Contents

1. [Scope](#1-scope)
2. [Assumptions](#2-assumptions)
3. [The roles](#3-the-roles)
4. [Threat model](#4-threat-model)
5. [Boundaries](#5-boundaries)
6. [The hidden benchmark](#6-the-hidden-benchmark)
7. [Sandbox providers and `doctor`](#7-sandbox-providers-and-doctor)
8. [Authentication and network](#8-authentication-and-network)
9. [The workspace and the gate](#9-the-workspace-and-the-gate)
10. [The final evaluator](#10-the-final-evaluator)
11. [The device-boundary contract and the observer](#11-the-device-boundary-contract-and-the-observer)
12. [In-run services: scoring and FPGA](#12-in-run-services-scoring-and-fpga)
13. [Fair comparison and statistics](#13-fair-comparison-and-statistics)
14. [The run record](#14-the-run-record)
15. [When the core is done](#15-when-the-core-is-done)
16. [Decisions](#16-decisions)
17. [Progress](#17-progress)
18. [Quick start](#18-quick-start)
19. [What has been run](#19-what-has-been-run)
20. [Sources](#20-sources)

## 1. Scope

**What it is for.** An agent is given the repository at one commit and a task:
make a class of PyTorch workloads faster on TNCA, through the full chain
PyTorch → Triton → TNCA IR → ISA → RTL. It may change the compiler, the ISA
and the hardware. The harness isolates it, controls what it can see, lets it
request measurements within a budget, and scores what it leaves behind.

**The core:** the run lifecycle and the run record; isolation (container,
network, credentials); the workspace, the submission gate and visibility; the
final evaluator; budgets and metering; the harness's own tests.

**Built on the core:** task packages (§9); the in-run services the agent may
call, the scoring service and the FPGA service (§12); tools inside the
sandbox, such as the cycle-accurate simulator.

**Our own thin core, rather than adopting Harbor or Inspect** (decided
2026-10-14), with formats compatible with Harbor: the task layout, ATIF
trajectories, the error classes. Inspect's model bridge calls the provider
itself, and subscription credentials may not be handed to another product
(§8). Harbor's defaults put the real API key inside the container and leave
`/logs` writable by the agent; it meters Claude Code and Codex differently, and
its releases change every few weeks.

**Portable by construction.** Other people run this on their own machines;
nothing in it names a host. The FPGA steps need a host with Vivado, and the
board stage a host with the card; both will be named in the operator's site
configuration (§7).

**No operator's details ship.** No credential value, proxy address or machine
name appears in the repository or the images. Credentials, account labels,
proxies, hosts and resource limits live in each operator's site configuration,
outside the repository. A run record names an account by the operator's label
and lists the model hosts its tunnels reached; `doctor` reports name the host
and the credential paths, and stay with the operator.

## 2. Assumptions

Each design choice rests on one of these. If one stops holding, revisit the
sections it names.

| # | Assumption | Used in |
|---|---|---|
| A1 | An agent's output is unreviewed until the operator checks the final candidate. Isolation and automated gates protect the run; the operator decides whether to accept its final score | everywhere |
| A2 | Claude Code and Codex sign in with a subscription credential. Their credentials must travel with the unmodified CLI | §8 |
| A3 | GLM is reached with an API key. With a key the credential could stay outside the sandbox; whether the plan's terms allow a key-injecting proxy and formal runs is to be checked (O10) | §8 |
| A4 | Until dedicated experiment accounts exist, the accounts in use are for development only | §8 |
| A5 | A host offers rootful Docker on cgroup v2. (`doctor`'s limit canaries read cgroup v2 files; rootless Podman is planned) | §7 |
| A6 | The agent may change the compiler, ISA and RTL freely. The constraint is the chain: PyTorch → Triton → TNCA IR → ISA → RTL, nothing skipped | §10, §11 |
| A7 | The agent's sandbox has the full toolchain, the cycle-accurate simulator included. It measures its own test cases with it | §3, §6 |
| A8 | For now, while the harness is built, all of ArchBench is hidden from the agent (§6) | §6, §12 |
| A9 | The evaluator runs no candidate code except the design it evaluates. The candidate's compiler (and the parts of `python/tnca/` it may change) still runs in the host-side measuring process; P2.1 moved the device and its count out of that process's reach. Final human review checks the remaining result path; P2.2 can automate that check (§10.5, §11) | §10, §11 |
| A10 | RTL simulation cycles are deterministic; board measurements are not exactly | §13 |
| A11 | FPGA builds are slow: synthesis of the whole device takes about 25 minutes, a full link about 4 hours. The agent's use of them is budgeted, and the budget is enforced | §12 |

## 3. The roles

Six parts, and one question tells them apart: **does what it produces go back
to the agent?**

| Role | What it does | When | Who starts it | What reaches the agent |
|---|---|---|---|---|
| **Agent sandbox** | Where the agent works: a fresh workspace, the toolchain, the simulator. What the agent measures itself, it measures here, on its own test cases | during the run | the harness | everything it produces itself |
| **Submission gate** | Takes the agent's changes under the design paths and nothing else, as base commit plus overlay | at every service request, and at the end | the harness | nothing |
| **Scoring service** | Rebuilds base plus the candidate from source and measures the hidden **validation set** in simulation | during the run, when the agent asks, within its budget | the agent | aggregates only (§6.4) |
| **FPGA service** | Quick tier: out-of-context synthesis and timing of the device. Slow tier: implementation, then a board stage. Runs through the FPGA harness on the build worker, and on the card host for the board stage | during the run, when the agent asks, within its FPGA time budget | the agent | timing and resources; for the board stage, aggregates on the validation set |
| **Final evaluator** | Rebuilds base plus the final candidate and measures the hidden **final set** (and the validation set, for the record) | after the agent has stopped | the harness | nothing, ever |
| **Final human review** | Checks the final candidate and the evaluator's evidence before its score is accepted (§10.5) | after final evaluation and any required board certification | the operator | nothing |
| **FPGA harness** (`tools/tritonchip_fpga_harness/`) | The execution and evidence layer for FPGA work: drives the repository's own flow, judges builds, loads and measures on the card | under the FPGA service, and for board certification of a candidate the operator selects | the FPGA service or the operator | only through the FPGA service |

Apart from the services' budgeted results, everything the agent learns, it
learns in its sandbox.

```text
                         ┌──────────── agent sandbox ────────────┐
                         │ workspace (no ArchBench, no history)   │
                         │ toolchain + simulator: own test cases  │──── offline: model API only
                         │ tnca-service CLI ──┐                   │     (gateway, §8)
                         └────────────────────┼───────────────────┘
                                              │ one socket, budgeted (planned)
                                              ▼
                                     ┌──── submission gate ────┐
                                     │ design paths only        │
                                     └──────┬──────────┬────────┘
                       aggregates           │          │          timing, resources
             ┌──────────────────────────────┘          └──────────────────────────────┐
             ▼                                                                        ▼
   ┌── scoring service ──┐                                              ┌──── FPGA service ────┐
   │ validation set, sim │                                              │ quick: OOC synthesis │
   └─────────────────────┘                                              │ slow: link + board   │
                                                                        │   (FPGA harness)     │
                                                                        └──────────────────────┘
   after the run:  gate ──► final evaluator ──► final set + validation set ──► run record ──► human review
```

**The operator** runs experiments. They own the site configuration, their copy
of the hidden benchmark, the build worker, the card host and the card windows.
They are trusted; the agent is not. **The agent never runs on an FPGA host**;
only candidate code built or measured there by a service does, inside a sandbox
(§12.5). The exception, until the card host's sandbox is built: board
certification (§10.4) compiles a candidate's points there with its compiler,
when the operator who read it says so.

## 4. Threat model

**Actors.** The agent and everything it starts are untrusted, including any
process run as the agent's user and any candidate code that a service or the
evaluator runs. The CLI binary is trusted to be unmodified but not trusted for
its numbers. The operator, harness, gateway, services and evaluator are
trusted.

| # | Threat | Example | Stopped by |
|---|---|---|---|
| T1 | Seeing what it must not | ArchBench's points or results, the task's definition or reference patch, another run, git history, the evaluator's code, host credentials | sandbox (§7), workspace and visibility (§6, §9) |
| T2 | Reaching the network | fetching a solution, exfiltrating a credential, reaching services on the harness host | gateway (§8) |
| T3 | Measurement or result-path errors | an edited shell counter, early completion, or a host-side result that did not come from the measured device run | observer (§11, P2.1) and final human review (§10.5); P2.2 is optional automation |
| T4 | Skipping the chain | host fallback, hand-written ISA for one kernel, tables keyed on a benchmark | hidden benchmark (§6), chain checks (§10) |
| T5 | Probing the hidden benchmark through results | many scoring requests to map the validation set; error text naming a shape; a compiler that encodes what it compiled into its timing | aggregates only, budgets, a final set never scored during the run (§6.4) |
| T6 | Harm to an FPGA host | candidate RTL that reads host files at synthesis (`$readmemh` of an absolute path); candidate code on the card host reading other users' files or what another session left on the card | sandboxes on the build worker and the card host (§12.5) |
| T7 | Cross-run contamination | a shared cache used as a message board | fresh state per run (§7, §9) |
| T8 | Budget evasion | a nested `claude -p` on the same credential, under-reported tokens, FPGA time not charged | metering (§8), service accounting (§12) |
| T9 | Unequal conditions | different configs, resources, rate limits, model fallback, queue waits | equalisation (§13) |
| T10 | Escaping the container | writing outside the workspace | provider hardening (§7) |

## 5. Boundaries

| Boundary | Separates | What crosses it |
|---|---|---|
| **Container** | the agent from the host and from other runs | its workspace, its CLI config directory, the bridge script, the gateway's socket directory |
| **Network** | the agent from the internet and from the harness host | the gateway socket (model API); the service socket (§12) |
| **Visibility** | the agent from the benchmark | nothing about the final set; aggregates about the validation set |
| **Submission gate** | the workspace from everything that measures it | a diff against the base commit, read from the source repository, never from the agent's `.git` |
| **Observation** | the design from its measurement | numbers the platform counted itself, at the device boundary (§11, P2.1) |

The rule that ties them together: **the agent owns the design; the evaluator
owns the observation and the questions.** No file is "protected" in the sense
that changing it rejects a run. A change is either part of the design, and goes
into the candidate, or it is ignored, because nothing that measures uses the
candidate's copy of it. P2.1 separated the device and its cycle count, while
candidate code still runs in the host-side measuring process. The final human
review covers that remaining result path (§10.5); P2.2 could automate it (§11).

**How each boundary is built.**

| Boundary | Mechanism | Settings | Verified by | Status |
|---|---|---|---|---|
| Container | Docker, rootful: one container per run, from the `agent` image (`provider.py`). It gets the bridge script as one read-only file, never the harness directory, which holds the task packages | `--network none`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`, `--user 1000:1000`, `--init`, `--cpus`, `--memory` = `--memory-swap`, `--pids-limit`, tmpfs `/tmp` and home, the default seccomp profile, no Docker socket, no devices | `doctor` canaries (§7); `tests/test_runner.py` | done |
| Network | A CONNECT-only proxy (`gateway.py`) in a sidecar container on the host's network. It serves a Unix socket in a directory the agent container mounts read-only; `bridge.py` inside the sandbox turns that socket into `127.0.0.1:3128`, and the CLI is given it as `HTTPS_PROXY` | `offline`: the account's host allow-list, port 443. `open`: any public host, ports 443 and 80, never loopback, private or link-local addresses. TLS never decrypted; every tunnel and refusal logged as JSON lines | `tests/test_gateway.py`, `doctor`, smoke | done |
| Credentials | A fresh copy per run: `--env-file`, or a copied `CODEX_HOME` | never on a command line; transcripts copied out file by file, credential files skipped; a refreshed Codex credential is copied back only after checks (§7) | `tests/test_siteconfig.py`, `tests/test_runner.py` | done |
| Visibility | The workspace is written with git plumbing (`cat-file --batch`) from one commit of the source repository, minus the hidden paths, and committed again with a fixed author and date | no history and no reflog; instruction files, both harnesses, `evidence/` and the task's hidden paths removed; a hide entry that matches nothing stops the export | `tests/test_workspace.py`; the leak audit (§6.5) | done for `replay-mzero` |
| Gate | Compares the workspace with the base commit read from the source repository, never from the workspace's `.git` | design paths taken; evaluator-owned paths dropped; under a design path, a symbolic link, special file or nested repository rejects the candidate | `tests/test_workspace.py` | done, with the gaps in §9 |
| Evaluation | A separate container from the `toolchain` image, with the same hardening and no credential, on the base plus the candidate's overlay | its own copies of the testbench, ArchBench, the tools and the flow | the noop and reference runs | done |
| Observation (P2.1) | Each point's device (relay, simulator, the platform's observer) in an evaluator-owned container, built from the base code with only the candidate's hardware, reached over a Unix socket | `--network none`, the same hardening; the observer's log written where candidate code cannot reach it; candidate RTL checked for file and system tasks | §19: a halved shell counter scores 1.000; `$fopen` is rejected | done |
| FPGA build worker | bubblewrap per job, run over SSH by the FPGA service | `--unshare-all` (no network), a read-only system and Vivado installation, only the job's directory writable, no home, a hard `timeout` | the license test and the service check (§12.5, §19) | in progress |
| Card host | bubblewrap (`bwrap`), the sandbox tool available there | no network, a read-only system, only the card's device nodes, no home or harness state | — | planned |

## 6. The hidden benchmark

### 6.1 The rule

**For now, while the harness is being built, all of ArchBench is hidden from
the agent** (decided 2026-10-15). The agent sees no point and none of the code
that defines one: not a point's id, shape, code or result, and nothing a point
could be derived from. The agent writes its own test cases. By design, the
task's syllabus states the class of workloads and the scoring rules (§6.3);
how specific it may be is O15.

Nothing of the final set ever reaches the agent. Of the validation set it gets
only budgeted aggregates (§6.4): a channel that is narrowed, not closed.

The reason is generality. A method that is only good on the points it was shown
is not a result (§10.3). An agent that cannot see a single point can only make
general improvements.

### 6.2 How it is hidden

- **In the developers' repository, ArchBench stays where it is** and is maintained with everything else. Agents never see the repository, only a workspace exported from it.
- **The workspace export removes it.** For `replay-mzero`: `archbench/` and the 25 other paths the leak audit found (§6.5). A hide entry that matches nothing stops the export, because a typo, or a directory written without its trailing slash, would otherwise leave what it names in the workspace.
- **The final evaluator and the services measure the base commit with the hidden package in it.** In the developers' repository, which contains all of ArchBench, that is simply the full base commit.
- **For delivery, the benchmark is split into a public and a hidden package.** The harness defines visibility; it does not create a benchmark. A task's validation and final sets belong to the hidden package: in a delivered harness the task refers to them, and the operator's copy supplies the point ids.

Planned: a check at export that refuses a workspace in which any point id or
shape string of the hidden package appears, run over the git objects as well as
the working tree.

### 6.3 The syllabus

With no visible points, the task text tells the agent what is measured, as a
syllabus does for an exam: the class of workloads, how a point runs and is
judged, how the score is formed, and the agent's budgets. It names no point.
`replay-mzero`'s says: matrix-multiply-like PyTorch programs, at shapes small
enough for the RTL simulator, in the floating-point types TNCA's compiler
supports. It also shows how to compile and measure a test case of the agent's
own (`tasks/replay-mzero/instruction.md`).

### 6.4 Three sets, and what flows back

*Proposed, 2026-10-15; the split into two hidden sets is not yet decided (O18).
The task and the evaluator already carry both sets.*

| Set | Agent sees its points? | Agent gets results? | Measured by |
|---|---|---|---|
| The agent's own test cases | yes, it wrote them | yes, all of it | its own sandbox |
| **Validation set** (hidden) | no | **aggregates only**, a budgeted number of times | the scoring service; the slow tier's board stage (both planned) |
| **Final set** (hidden) | no | **never** | the final evaluator, after the agent has stopped |

**Why two hidden sets.** A set whose score the agent can request repeatedly is
slowly fitted, because each request leaks a little. And the candidate's own
compiler necessarily processes the points it is scored on, so a hostile
compiler can encode what it saw into its timing. Budgets and aggregates narrow
that channel; only a set that runs after the agent has stopped closes it. That
is the final set.

**An aggregate is:** the number of points, the number that passed, the
geometric-mean speedup over the points that passed, and `fast_1` (§10.2). It
never contains a point's name, shape, cycles or error text: an error such as
"unsupported shape 64×128" names the question. Because a wrong point fails
alone, an aggregate still says how many points a request broke; the size of
the validation set bounds what that teaches (O17).

### 6.5 Leak audit (2026-10-15)

**Method.** The workspace `replay-mzero` exported was audited by four
independent finders, each over a different scope: files that name ArchBench,
point ids or shape strings; every document; equivalent workloads in software
tests and examples (same operation, shape and dtype as a point); and
hardware-side files and data. A catalog of all 101 points (24 families) was
matched against them. Every finding was then checked by a separate agent asked
to refute it, and a completeness critic looked for what all of them missed.
118 verdicts: 56 real leaks, 62 harmless or refuted.

**Fixed in the repository (`6f2bdee`, 2026-10-16), not hidden.** Four comments
in files the agent's flow needs (`rtl/core/tnca_core.v`,
`rtl/board/tnca_u50_xo.v`, `compiler/python/tnca_cc/partition.py`,
`python/tnca/runtime.py`) named the tiny tier's size, per-family results or a
model point's sizes; they now describe the mechanism only. Seven graphs of
`tools/r38/torch_tests.py` and the example of the torch.compile tutorial were
exactly a point's workload; their shapes changed (torch tests 24/24 pass; the
tutorial's printed output re-recorded from a real run), so the agent keeps its
main regression suite and the tutorial. `replay-mzero`'s base moved to
`9a3b1378`: its old base plus this commit, nothing else.

**Removed from the workspace (26 hide entries).** `archbench/`; the r39
ArchBench baselines (`spec/r39/baseline_archbench*.json[l]`); suite and workload
lists (`spec/r38/suites/dev.json`, `spec/r38/evidence_registry.json`,
`spec/r38/design_changes/xpose-dma-001.json`, `spec/r3_workloads.json`,
`spec/r35_workloads.json`, `spec/r3v2_inventory.json`); tests and scripts that
are exactly a point's workload (`tools/r38/dev_cases.py`,
`tools/r37/upstream_cases.py`, `tools/r37/aot_run.py`,
`tools/r38/triton_native_run.py`, `tools/r38/tdesc_tests.py`); the old scoring
copy `tools/r38/archbench/`; documents with per-point or per-family results
(`doc/architecture_overview.md`, `doc/compiler_overview.md`,
`doc/r38_board_bringup.md`, `doc/r38_pytorch_p1plus.md`,
`doc/r38_stage1_status.md`, `compiler/README.md`, `flow/fpga/README.md`); the
tutorial that quotes board cycles (`tutorial/11_running_on_the_u280.md`); and
the two r39 planning documents. None of them is read by the agent's flow
(staging, `tnca-sim prepare`, the compiler, the runtime).

**After removal:** no point id and no shape string appears in the exported
working tree or its commit (570 files). Files that still name "ArchBench"
(Makefile targets, comments) were judged harmless by the audit.

**Not yet checked:** `/opt/Tilebench`, the TileBench checkout in the agent
image; it is a separate benchmark, but its shapes were not compared with
ArchBench's.

## 7. Sandbox providers and `doctor`

**The isolation unit is an OCI container.** Two targets of one Containerfile
(`image/Containerfile`):

- `toolchain`: Ubuntu 24.04, the HDL and MLIR tools, PyTorch and Triton, TileBench at its pinned commit, `tnca` importable from `/work`. The evaluator and `doctor` use it.
- `agent`: the toolchain plus Node and the three CLIs.

Versions are passed by hand as build arguments (§18): torch and triton as in
`spec/r38/torch.lock.json` and `spec/r1_toolchain.lock.json`, the TileBench
commit as in CI, Node (with its sha256) and the CLIs pinned in the build
arguments. The apt packages and the torch wheel are not hash-pinned. A lock
file of the harness's own is planned.

**Docker is supported. Rootless Podman, running the same image, is planned.**

**Hardening always applies; a task or user cannot turn it off:** a non-root
user; `--cap-drop ALL`; `--security-opt no-new-privileges`; a read-only root
filesystem with tmpfs `/tmp` and home; `--network none`; `--cpus`, `--memory`,
`--memory-swap`, `--pids-limit`; `--init`; the default seccomp profile; no
Docker socket and no host devices. The containers run as uid 1000 whatever uid
the image was built with, so on a host where the operator's uid is not 1000 the
state directory must be writable by uid 1000 (planned: take the uid from the
site).

**The gateway runs as a trusted sidecar container** with network access; the
agent container has none. They share a Unix socket through a read-only mount,
and a bridge inside the sandbox serves it as `127.0.0.1:3128`.

**`doctor` measures the host instead of trusting its configuration.** It starts
canary containers, checks each guarantee from inside, writes a report, and
exits non-zero when a guarantee fails. `run-task` does not yet refuse to start
on such a host (planned).

| Guarantee | Canary |
|---|---|
| Filesystem view | the host user's home path is absent, the container's home has no `.ssh`, the only bind mounts are the expected ones. Planned: plant files in the host home, `~/.ssh`, a sibling run and a hidden task path, and try to read them |
| No network | DNS and TCP to the outside fail; a request through the gateway succeeds; an unlisted host is refused |
| Resource limits are real | a memory hog is killed at the limit; the CPU, memory and process limits are visible in the cgroup v2 files |
| No devices | no `/dev/xcl*`, `/dev/dri` or other host device nodes |
| No cross-run state | a marker written by one canary is invisible to the next |
| Disk | planned: free space above a threshold |

| Host class | Use |
|---|---|
| Rootful Docker, cgroup v2 | formal runs |
| Rootful Docker, cgroup v1 | not yet supported: the limit canaries read cgroup v2 files |
| Rootless Podman, cgroup v2 | formal runs, once the provider exists |
| Rootless runtime, cgroup v1 | development only: limits are silently ignored |
| Docker Desktop (macOS) | developing the harness itself (`doctor` does not detect it and reports it as formal) |

**Site configuration** (`siteconfig.py`, schema `tritonchip-agent-site/1`). One
file per operator, outside the repository, for example
`~/.config/tritonchip-agent-harness/site.json`: the provider, images, per-run
resources, the state directory, network routes and an optional upstream proxy,
the repository workspaces are exported from, and for each account a label, its
CLI, its auth mode, the *path* to its credential, its hosts and its models. The
repository ships only a template (`examples/site.json`). The build worker and
the card host will be named here too (planned).

`doctor` parses each credential file but keeps and reports only its size, key
names and value lengths; the file must belong to the operator, with no group or
other permission bits. Credentials reach a run as a fresh copy, through
`--env-file` or a copied home, never on a command line. Transcripts are
collected file by file, regular files with safe paths only, skipping
`auth.json` and `.credentials.json`. A Codex credential the CLI refreshed
during the run is copied back over the operator's only if the operator's copy
did not change meanwhile and the sandbox's copy is a regular file of plausible
size whose JSON has the same top-level keys; otherwise the refusal is recorded.

## 8. Authentication and network

| | Subscription credential | API key |
|---|---|---|
| Credential | inside the sandbox, with the CLI: `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, or Codex `auth.json` in the run's `CODEX_HOME` | today: inside the sandbox, as an env file. Deferred: outside, injected by an API-key proxy |
| Why | Anthropic does not permit third parties to "collect, store, or intermediate" claude.ai credentials; OAuth is for the unmodified Claude Code | the key owner may provision its own key |
| Network | a gateway that accepts only `CONNECT`: in `offline`, to the account's model hosts; in `open`, to any public host. It never decrypts TLS | the same gateway today |
| Tokens | reported by the CLI, collected outside the sandbox, recorded beside the gateway's byte counts. An automatic cross-check is planned | today the same; with the proxy, measured by it |

**Network modes (decided 2026-10-15).** Exactly one per run, recorded:

- `offline` (default): the gateway allows only the account's model hosts; every CLI web tool is off.
- `open`: the gateway allows any public host on ports 443 and 80 and logs every tunnel; web tools stay on. Loopback, private, link-local and reserved addresses are refused even then, by literal address, by local name, and, for direct routes, by what a name resolves to. Internal names reached through an upstream proxy cannot be told apart; a site deny list for them is planned.

**What the CLIs try anyway.** Codex 0.162.1 fetches from `github.com`,
`api.github.com`, `files.openai.com` and `*.oaiusercontent.com` at startup, with
no documented switch; opencode tries `registry.npmjs.org`. In offline mode
they are refused and logged, and the CLIs work without them.

**Adapters keep the agent system's own capabilities** (subagents, compaction,
reasoning) and switch off what leaks the environment: user config, instruction
files, memory, MCP, plugins and skills fetched from the network, telemetry,
auto-update, and web tools in offline mode (`adapters.py`).

**Rate limits and model identity.** The model is pinned (`--model`) and the
served model is recorded from the CLI's report. `smoke` fails on a mismatch;
invalidating a task run on one is planned. A rate limit is an infrastructure
event: detecting it, and re-running the run it broke, is planned. One account
carries one run at a time: for now the operator's responsibility, not enforced.

**Credentials and data.** The accounts in use now are for development only (A4).
Before TritonChip code is sent, each account's model-training setting is
checked and GLM's data policy read; the code is unpublished research. Formal
experiments at scale use dedicated accounts or API keys.

## 9. The workspace and the gate

**A task package** (`tasks/<id>/`): `task.json` (base commit, design paths,
evaluator-owned paths, hidden paths, budget, the validation and final sets),
`instruction.md` (the syllabus) and an optional `reference.patch` (the known
answer, for validating the loop). The prompt is the syllabus with the run's
rules appended (network mode, budget); it is byte-identical for every agent in
a cell, and its sha256 and that of `task.json` are recorded. The package lives
in the harness directory, which no workspace contains and no agent container
mounts. A task's validation and final sets belong to the hidden package (§6.2).

**The workspace** is one commit: the task's base commit, exported file by file,
with a fixed author and date so equal trees hash equally. It has:

- no history, remote, tags or reflog;
- no agent instruction files at any depth (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.claude/`, `.codex/`, `.cursor/`, `.gemini/`, `.opencode/`, `.mcp.json`, `opencode.json`);
- no harness code (neither harness), no top-level `evidence/`;
- nothing the task hides: for `replay-mzero`, ArchBench and the files of the leak audit (§6.5), and the planning documents that describe the answer.

**The gate** compares every file in the workspace, by content and mode, with
the base commit read from the *source* repository:

- changes under the design paths go into the candidate;
- except the evaluator-owned paths, which are dropped and recorded. They are the code that builds and drives the simulated device: `python/tnca/device.py`, the runtime, transport and backends, the Triton driver, the Inductor glue, the profile loader `python/tnca/profiles.py` and data staging `python/tnca/_data.py`. They always come from the evaluator, whatever the task (decided 2026-10-15). The profiles themselves (`spec/r1_profiles.json`) are a design path;
- other changes are dropped;
- under the design paths, a symbolic link, special file or nested repository rejects the candidate; elsewhere it is dropped like any other change;
- files ignored by the workspace's `.gitignore` rules are skipped.

**Known gaps in the gate (planned).** The ignore rules come from the workspace,
so an agent's `.gitignore` edit changes what the gate sees; they should come
from the base tree, and bytecode under `__pycache__` should always be dropped.
An owned module `X.py` is not yet protected against a package `X/`, an
extension `X.*.so` or a `.pyc` that Python would load first. Until P2, a
candidate can patch owned modules at run time anyway (§11).

**Storage and re-scoring.** A passing candidate is stored as base commit plus
overlay (`candidate/files/` and `candidate.json`). `--agent patch:<file>`
re-scores a change given as a git patch under the current rules; re-scoring a
stored candidate directory is planned.

## 10. The final evaluator

### 10.1 What it does

At the level of CHIA (§20): it rebuilds from source with no AI involved and
measures with its own copies. It exports the base commit without the task's
hide list (the harnesses, `evidence/` and agent instruction files are still
left out) and lays the candidate's overlay on top, so the testbench, ArchBench,
the tools and the flow are the base's. Its platform (the testbench with the
observer, the relay) comes from the commit the task pins
(`evaluation.platform`) and is laid over every tree it builds, whatever the
base had. It then runs ArchBench on the validation and final sets, with each
point's device in a container of its own (P2.1, §11), and compares each point
with the base's own cycles, both counted by the platform. The base is measured once per
(base commit, profile, point list) and kept; a run whose base does not measure
cleanly is `infra-invalid`.

### 10.2 How a point is judged

**A wrong point fails alone; the other points still count, and the run stays
valid** (decided 2026-10-15). A point fails if its numbers are wrong, if any of
it fell back to the host (the chain was skipped), or if it produced no record.

*The aggregates are proposed (2026-10-15), not yet decided:* for each set the
evaluator reports

- `n` and `passed`;
- `speedup_geomean`: the geometric mean of base cycles / candidate cycles over the points that passed;
- `fast_1`: the share of **all** points that passed and beat the base.

`fast_1` is there because the passed-only mean alone would reward breaking a
point the candidate had made slower. How they combine into one score is open
(O12).

**The machine and the latency (decided 2026-10-17, built 2026-10-18).** The
machine is the Alveo U280, with its kernel clock fixed at 100 MHz. Hardware
cost is not scored; it is a constraint: a candidate's machine must fit the U280
and close timing at 100 MHz. The score is latency: device time, the platform's
count of each launch with host transfers left out (`archbench/PROTOCOL.md`
§3.1). At one fixed clock that is the simulated cycles, so the points are
scored as they always were. Scoring at the candidate's own clock (the agent
declares a target, a miss is linked again lower and scored there) was decided
first and is deferred (§16): it needs a link of every candidate before any of
its score is known, and a first version does without it.

**Feasibility (built: `certify.py`, §10.4).** A final candidate that changed
the machine -- an RTL file, or the parameters of the profile it is scored on
(`evaluator.hardware_changed`) -- keeps its score only once it is certified:
built for the U280 at 100 MHz and its points run on the card, each counting
the cycles simulation counted for it. Its run record says
`certification: pending` until then. A candidate that does not fit, does not
close 100 MHz, or disagrees with simulation on the card fails every point; the
run stays valid, and the certification record keeps the reason. A screen
before the link (the quick tier's estimate against what has closed), the
diagnosis of a candidate that does not fit and a reference latency for it are
deferred.

### 10.3 Chain integrity and generality

1. **The evaluator drives the chain.** It starts from its own ArchBench PyTorch modules and runs `torch.compile` → Inductor → Triton → the candidate's compiler → the candidate's RTL. A candidate supplies a toolchain and a design, never kernels or binaries.
2. **A host fallback fails the point** (§10.2).
3. **The benchmark is hidden** (§6; for `replay-mzero` since 2026-10-15), so nothing can be keyed on its kernels.
4. **Final review and planned automation:** the operator audits the candidate diff and generated code before accepting the score (§10.5), including large instruction literals and branches on kernel names or shape constants. Automated provenance that regenerates or verifies each stage from TNCA IR to ISA with evaluator-owned code remains planned; the stage files the candidate's compiler writes (`TNCA_KERNEL_DUMP=1`) are only its own claim. Monitors fix the environment; they are not part of the score.
5. **ISA extensions are allowed.** They must take effect through compiler lowering. Their resources are not charged; like any hardware change they must fit the U280 (§10.2).

**The machine is the agent's to choose; the starting point is the task's.**
A profile is a machine configuration: the same RTL with switches on or off and
its parameters set. `spec/r1_profiles.json` is a design path, so the agent may
turn on, turn off or add any switch of the profile it is scored on (pilot 1's
Claude Code run enabled seven), and everything that measures follows the
candidate's configuration: the simulated device is built from the candidate's
`rtl/` and profile (`evaluator.HARDWARE`), and the FPGA quick tier synthesises
the same configuration (`fpga_quick.effective_params`). The task's
`evaluation.profile` names only where the agent starts. What the task fixes is
that starting point and the baseline it is compared with: core-vrf-sync-v5,
the signed-off U280 image (u280_r39h; O23, decided 2026-10-17, the
collaborator to confirm). The flow builds it since `d1a00dd`: r40's
`TNCA_IMAGE_CONFIG`, proposed to the collaborator, sets every switch the U280
top passes from the profile (before, the package took the RTL defaults, which
are core-vrf-sync-v2, whatever the profile said), and the FPGA harness refuses
a profile the top cannot carry rather than build another machine.
`replay-mzero` starts from v5 since base `083b57d`.

**Machine changes are recorded.** The evaluator records every changed
`elaboration` and `rtl_parameters` entry of the task's profile; other profile
keys, compiler target files and RTL changes appear in the gate's list of files
taken.

### 10.4 Board certification (built)

`certify` (`certify.py`) shows that a candidate's machine is real: that it
fits the U280, closes 100 MHz, and counts on the card the cycles simulation
counted. It runs on the **controller**, the one machine that reaches both FPGA
servers; the servers never connect to each other, and the evaluation host does
not reach the card server. It needs no site configuration, only the FPGA
harness's hosts file and a clone with the task's commits.

1. **The tree.** The base commit with nothing hidden, every file the gate took
   from the candidate on top (compiler and hardware alike: the card compiles
   the points with the candidate's compiler, as simulation did), then
   `evaluation.certify.paths` -- the FPGA flow, the board wrapper and top, the
   FPGA harness -- laid over whole from `evaluation.certify.commit`. The
   candidate cannot have changed those paths (they are evaluator-owned or
   outside the design paths), so the platform is the evaluator's. The tree is
   committed with a fixed identity and date, so one candidate is one commit,
   and both servers check out exactly it.
2. **The build and the board.** The FPGA harness's `pipeline` (its README
   §11.2) links the tree on the build server at 100 MHz with the scored
   profile as the image (`image.profile`), then on the card server programs
   it, runs conformance and runs the task's validation and final points
   (`board.archbench.points`).
3. **The comparison.** Each point that passed in simulation must pass on the
   card, with no fallback, and the card's `time.device` must equal the
   device's own count that simulation recorded (`done_cycles` in the run's
   score rows; the observer checks that count against the record, §11).

| Verdict | When | The score |
|---|---|---|
| `not-needed` | the machine is the base's | stands |
| `certified` | 1-3 held | stands |
| `infeasible` | the build failed: it does not fit or does not close 100 MHz, or the profile cannot be built as the image | every point fails |
| `board-fail` | the image did not load, or the board stage did not reach the points | every point fails |
| `board-disagrees` | a point failed on the card, fell back to the host, or counted other cycles | every point fails |
| `no-verdict` | the pipeline or a server failed | re-run |

The record, `certification.json` beside `run.json`, keeps the verdict and its
reason, the tree's commit, the image's parameters, the build's timing, each
point's two counts, and the run's score after certification. The base is
certified the same way against the evaluator's baseline (`--base`): the
machine every candidate is compared with must itself fit, close and agree.

**What runs on the card server.** The card compiles the points with the
candidate's compiler, and its sandbox is not built yet (§12.5). Until it is,
that is code the operator runs having read it: `certify` without `--yes`
builds the tree, lists the candidate's files that run on the card server, and
builds nothing. A hardware-only candidate, and the base, run none.

### 10.5 Final human review before accepting a score

`run.json`'s `validity: valid` means the automated run and its base measurement
were usable. It is not an operator's acceptance of the candidate or its score.
After the agent loop ends, the operator reviews the final candidate before
using its final-set result in a comparison or publication:

1. Pin the run ID, task hash, base commit and candidate overlay. Read
   `candidate/candidate.json` and the actual diff, especially compiler and
   host-side `python/tnca/` changes. Check that the measured work still follows
   PyTorch → Triton → TNCA IR → ISA → RTL and that the output path uses the
   measured device's results, not a host computation substituted for them.
2. Check the final evaluator's ArchBench records and per-point observer logs:
   numerical correctness, no host fallback, expected launches, platform
   cycles, and no activity outside a launch. Reproduce any surprising point
   with a fresh seed and inspect its generated code and device trace.
3. If the machine changed, require `certification.json` with `certified` and
   matching board cycles (§10.4). Record the reviewer, date, exact run ID,
   evidence inspected, decision and any exclusions in `review.md` beside
   `run.json`. Only an accepted review makes the score a final result.

This is an explicit human sign-off, not a claim that the current software
proves every byte's origin automatically. P2.2 (§11) is a possible way to
automate the result-path portion if unattended scoring becomes a requirement.

## 11. The device-boundary contract and the observer

**The contract** is the fixed interface between the design, which agents may
change freely, and the evaluator's instruments:

1. the public device top's ports (`tnca_shell`): `cmd_*`, `done_*`, `host_*`, `mem_*`, `imem_*`, and the status ports `busy`, `core_idle` and the `perf_*` counters (`perf_sel` / `perf_rdata`, indexed by `spec/r38/counters.json`);
2. `spec/r2_abi_v1.json`: where a kernel's arguments, inputs and outputs live;
3. `spec/r2_transport_v1.json`, `spec/r2_container_v1.json`: the host↔device operations and the `.tnca` artifact;
4. the chain's entry point: a PyTorch module given to `torch.compile` with TNCA as the backend.

Inside it, everything may change, including the shell's internals. Changing the
ports or the specs is a contract change: a new contract version, supported by
the evaluator first.

**P1, one generated parameter header per profile (done).** A profile reaches
the device as one generated Verilog header of `TNCA_IF_*` (interface: they
shape the ports; a closed set: `NUM_RCTX`, `RCTX_W`, `LSU_TAG_W`) and
`TNCA_UARCH_*` (microarchitecture: an open set; an agent may add one)
definitions, with defaults in `rtl/common/tnca_uarch_defaults.vh`. The platform
names only the interface parameters. Verified for nine single-core device
builds (core-sim and v1, each also `/slow`; v2–v5; core-xpose-v1): every
parameter of every instance has the same value and width as before, and cycles
are identical on the points measured. Not yet migrated: the core-level benches,
`tnca_shell_tb`, the FPGA top.

**P2, the observer (P2.1 built and verified 2026-10-16; P2.2 optional automation).** Two facts motivate this boundary.
The cycle count that scores a run is `run_cycles` in `rtl/tnca_shell.v`, which
the agent may edit. And the measuring process runs candidate code: the
candidate's compiler, and its versions of the `python/tnca/` modules that are
design paths (among them `abi.py`, `artifact.py`, `cli.py` behind `tnca-sim
prepare`, and `__init__.py`). That same process starts the simulator as its own
child (`python/tnca/transport.py`), checks correctness and writes the record.

**Why a second container.** Inside one container every process runs as the
same unprivileged user, so an observer placed there can be overwritten, and the
simulator it watches can be replaced, by the candidate's code; without
privileges no second user can be made to separate them. Candidate code and the
observer must therefore not share a container. The seam already exists: the
device is a separate process behind the transport protocol
(`spec/r2_transport_v1.json`), and moving it to another container changes only
the medium, which the protocol allows. The existing `sim-socket` relay does not
give this by itself: the untrusted process starts it, over TCP. The design
principles stay: the agent owns the design, the evaluator owns the observation;
the contract (ports, ABI, transport, container) does not change.

**P2.1: the cycle count (built).**

```text
host container (candidate; as today)               device container (evaluator-owned; new)
 PyTorch + candidate compiler + runtime + ArchBench   relay + vvp
   └ sim-socket backend ──── Unix socket ──────────►   └ testbench (platform) + observer + candidate RTL
                                                         └ observer log, written only to the evaluator's directory
```

1. **The device container runs no candidate Python.** Its tree is the base commit with only the candidate's hardware paths laid on (`rtl/`, `spec/r1_profiles.json`, and `isa/` where the device build reads it), and the base's `device.py` builds the simulator. Building with the candidate's tree would execute its `tnca/__init__.py`.
2. **One device container per point**, each with its own socket. The evaluator runs `archbench.run` once per point (up to `jobs` at a time) with that point's socket, so every observer log belongs to exactly one point.
3. **The observer** (main repository, `tb/device/tnca_device.v`): the platform counts the cycles from the shell accepting a launch to the shell presenting its completion, and flags any memory-port request while neither a launch nor a host operation is in progress. Per launch it appends one line (sequence, platform count, `done_cycles`, status, requests inside and outside the window) to a log named by a plusarg.
4. **The relay** (main repository, `python/tnca/backends/`): the client connects to a relay that is already running instead of starting one, and the relay listens on a Unix socket. The protocol on the wire is unchanged.
5. **Candidate RTL is checked before the build:** no `$fopen`, `$fwrite`, `$fdisplay` to a file, `$system` or `$readmem*`, and no `` `include `` of an absolute path or one with `..`. A violation rejects the candidate. Today `rtl/` uses only `$value$plusargs`.
6. **Scoring** uses the platform's count, summed over the point's launches; `done_cycles` is kept only to cross-check. A point fails if its observer log is missing, if a request fell outside a launch window, or if a launch did not complete.
7. **The board platform becomes evaluator-owned:** everything above `tnca_shell`, `rtl/board/tnca_u50_xo.v` and `rtl/fpga/` (`tnca_fpga_top`, `tnca_mmio`, `tnca_fpga_mem`, `tnca_axi_lite`, `tnca_u50_top`), today under the design path.
8. **Device time is redefined** as the platform's count (O4). It differs from `done_cycles` by the shell's few handshake cycles, so the base is measured again once.

**P2.2: optional automated result-path check.** A final score credits work
done by the measured device; host-side compilation, reference computation and
data staging are normal parts of evaluation. The relay sees every upload and
download. An automated evaluator could check that every upload is an input of
the point (generated from its seed by base code), a block the device returned
earlier, or the program and its
arguments; that the program images and arguments are identical when the point
runs again with a second seed; and that the outputs, taken from the device's
downloads rather than from the candidate's record, match a reference computed
by base code. Provenance is judged by content, not by which side wrote memory:
host writes reach memory through the agent's own shell (`rtl/tnca_shell.v`
merges host and core requests onto one memory port), which could delay them
into a launch window. The second run would make P2.2 a final-evaluation step
only. Until this is built, §10.5 supplies the final review gate.

| Result-path concern | Check |
|---|---|
| The shell's counter edited | P2.1: the score uses the platform's count |
| Completion raised early, work continued | P2.1: memory requests outside the launch window |
| Work started while inputs were uploaded | P2.1: memory requests outside the launch window |
| The cycle number rewritten in the measuring process | P2.1: the observer log lives in the device container |
| Host-side result substituted for device work | §10.5 reviews the final candidate and output path; P2.2 could check upload provenance automatically |
| Input values embedded in the program image | §10.5 reviews generated code for surprising points; P2.2 could compare images across seeds |
| Output record disagrees with device downloads | §10.5 inspects the result path; P2.2 could compare the recorded output with device downloads |

**Where it lives.** The observer (3) and the relay (4) are in the main
repository (`55872ef`); `PROTOCOL.md` §3.1 states who counts (8). The evaluator
takes them from the platform commit the task pins, so a task's older base gets
the current platform. The rest is the harness's: the two trees and the device
containers (`evaluator._measure_observed`), the RTL check
(`evaluator.rtl_check`, applied by the loop after the gate), and scoring from
the observer's log (`evaluator.observed_point`), where a point whose log is
missing or incomplete, that shows memory activity outside a launch, or whose
record disagrees with what the device reported, fails.

**The RTL check (built 2026-10-18).** Candidate RTL may not use file or
system tasks (`$fopen`, `$system`, `$readmemh`, ...) or include a file from
outside the tree: it is simulated beside the observer's log, and Vivado reads
it on the build server. Since the card counts too, the check also refuses what
lets simulation and synthesis see two designs, or lets the design reach into
the testbench that counts it: synthesis pragmas (`translate_off`,
`full_case`, `parallel_case`), branching on a tool's macro (`` `ifdef
SYNTHESIS ``, `VERILATOR`, `__ICARUS__`, `XIL_*`, ...), `defparam`, and
hierarchical references. Each is refused only where the candidate adds it: a
line the base already has in the same file is the base's (the base's RTL
has 17 such lines today). The
check is cheap and catches the plain cases; what it misses, certification
catches as a disagreement on the card (§10.4).

**Verified (2026-10-16, `replay-mzero`).** On core-sim a launch reads the
platform's count six cycles above `done_cycles` (the shell's handshake), and the
base shows no activity outside a launch.

| Run | Result |
|---|---|
| `noop` | 1.000 on both sets |
| `reference` | 1.016 / 1.010, as before P2.1 |
| a shell whose `done_cycles` reports half its count | 1.000 / 1.000: the score is the platform's count |
| RTL that opens the observer's log with `$fopen` | `integrity-rejected` before anything is built |

## 12. In-run services: scoring and FPGA

### 12.1 Cheap estimators first

The FPGA flow is slow, so the agent should rarely need it (user direction,
2026-10-15): strengthen the estimators inside the sandbox instead. That means a
fast performance model, a stronger correctness model, and a preliminary
synthesis mapped to the FPGA (LUT, FF, DSP and BRAM estimates, logic levels,
and an Fmax estimate calibrated against Vivado). Today the sandbox has the
cycle-accurate simulator and generic Yosys synthesis (`make synth`). Planned.

### 12.2 The channel (built)

The broker (`services.py`) runs in the harness process beside the agent's run.
It listens on a Unix socket in a short directory of its own (a socket's path is
limited to about 100 bytes, and a run directory is longer), mounted read-only
into the sandbox at `/run/tnca-service`. `tnca-service` (`service_cli.py`,
mounted as one read-only file at `/usr/local/bin/tnca-service`) is the agent's
side: `budget`, `submit score`, `submit fpga-quick`, `status [ID]`,
`result ID`, `wait ID`. One JSON line in, one out.

- **A request is a snapshot, not files.** The broker passes the workspace through the same gate as the final evaluation, so only changes under the design paths count. A rejected snapshot, or RTL that fails the check of §11, is refused before anything is charged.
- **Asynchronous, one per service at a time.** The run's clock keeps going.
- **Budgets come from the task** (`services` in `task.json`) and the syllabus states them (`task.render`).
- **Everything goes into the run record** (`agent_result.services`): each request, the snapshot it measured, what the agent saw, and the full result it did not.

### 12.3 The scoring service (built)

Measures the validation set the final evaluator's way, P2.1 included
(`evaluator.measure_validation`), and returns the aggregate of §6.4: points,
passed, `speedup_geomean`, `fast_1`. If the candidate's device does not build,
the build's last lines come back, since they concern only its own RTL. A
request is charged when it is accepted.

### 12.4 The FPGA service

**The agent decides whether and when to use it. Its time is limited, the
syllabus says by how much, and the limit is enforced** (decided 2026-10-15).

| Tier | What runs | About | Returns | State |
|---|---|---|---|---|
| **Quick** | out-of-context synthesis and timing of the whole device (`fpga_quick.py`): the repository's own `flow/synth/vivado_ooc_timing.tcl` on the candidate's hardware, for the profile the run is scored on | 25 min | status; worst slack at the target period and the frequency it implies; the worst paths, as start and end points in the agent's own RTL; utilisation (LUT, FF, DSP, BRAM, URAM) | in progress |
| **Slow** | implementation through the FPGA harness (package, link, judged timing and DRC), then a board stage on the validation set | hours | timing closure; utilisation; the board stage's aggregate | planned |

**The quick tier times the agent's configuration** of the scored profile
(§10.3). The FPGA top's parameters are still written by hand, set to
core-vrf-sync-v2's, because P1 has not reached it. So the service passes the
candidate's profile to it as `NAME=VALUE`: the defaults in
`rtl/common/tnca_uarch_defaults.vh`, then the profile's own values, limited to
parameters the top has. It calls the Tcl script directly, into a short output
directory: `vivado_ooc.sh` names its output after every pair, and a whole
profile's worth is longer than a file name may be.

**Enforcement (built).**

- Each run has an FPGA time budget, charged as the worker's actual time.
- One FPGA request at a time.
- A request that could run past what is left is refused: each may take up to the task's `job_minutes`.
- On the worker the job runs under `timeout`, so it ends at its deadline whatever happens to the harness.
- A job still running when the agent stops is stopped (`pkill` on the worker, with a pattern that cannot match its own command line) and charged.
- Queue waits are not charged; the run's wall clock goes on.

**Synthesis estimates are optimistic.** Before routing, delays leave out most
of the wire: on this design the routed worst path was about 1.6 times the
post-synthesis estimate. The quick tier ranks designs and catches paths a
change made worse; the frequency a design closes at is the slow tier's to say.
The syllabus says so.

### 12.5 Where the FPGA work runs, and how it is isolated

**Placement (decided 2026-10-15).** All FPGA builds run on the **build
worker**; the **card host** is used only for the board stage. The site
configuration names the build worker (`fpga_worker`: SSH target, job directory,
Vivado settings script, read-only binds, threads, part).

- **The build worker (built).** Each job runs in a bubblewrap sandbox: no network (`--unshare-all`), a read-only `/usr`, `/etc`, libraries and Vivado installation, only the job's own directory writable, a fresh `/tmp` and home, and nobody's home directory or harness state visible. Candidate RTL is read by Vivado, so the sandbox is what stops `$readmemh` of a host file (T6). (The design first named a container; bubblewrap needs no image, and it is already the card host's tool.)
- **License (tested 2026-10-15): none is needed.** Vivado 2022.2 ran OOC synthesis, timing and utilisation for the U280 and U50 parts with no license reachable and no network; it reports the entitlement for these parts as built in. A control proved the license was really hidden. The unit `tnca_matrix` synthesised the same way in 1.8 minutes.
- **The card host (planned).** The candidate's compiler compiles the benchmark there, so it runs in a sandbox with no network, a read-only system, only the card's device nodes, and no home directory, harness state or compiler cache of anyone's. The card is shared, so board stages happen in agreed windows (O14), and between owners the card is reprogrammed or its memory cleared. Until the sandbox is built, certification (§10.4) runs the candidate's compiler there only when the operator, having read what it lists, passes `--yes`.
- **Versions (decided 2026-10-17, was O13).** The Vivado flow runs on the build worker with its Vitis 2022.2; the card host only loads and measures. Validated once: `b5c606d` linked on the build worker in 103 minutes (213 on the card host with 2023.2, an earlier link), closed 100 MHz with ap_clk WNS +0.138 ns (2023.2: +0.177), and its XCLBIN loaded on the card's 2023.2/XRT 2.16 shell, whose gates check only the platform's interface UUID. The U280 target accepts images from 2022.2 and 2023.2, and the build worker's installation overrides the target's default path.
- **How a candidate crosses (built, FPGA harness).** `pipeline` in `tools/tritonchip_fpga_harness` (its README §11.2) runs one candidate across the two: a clean checkout of the commit on each from a git bundle, the build detached on the build worker, a light packet of the image and records streamed through the controller into the card host, where every file's hash is checked before the run is installed, and the board stage there. The servers never connect to each other. A real candidate ran through it on 2026-10-17: the base hardware linked on the build worker in 105 minutes (ap_clk WNS +0.138 ns), loaded on the card, conformance 5 of 5, ArchBench tiny 27 of 27 with every point's cycles equal to the same design built with 2023.2 on the card host. (Both board stages compiled with the card host's installed `tnca`, an older tree, not the checkout's; found and fixed on 2026-10-18, `969201b`, §19.)

## 13. Fair comparison and statistics

**Unit of comparison.** One cell is a (scaffold, model, auth mode) on a fixed
(task, start, budget, environment version).

**Configurations (decided 2026-10-15).**

| Scaffold | Model | Auth | Role |
|---|---|---|---|
| Claude Code | Claude | subscription | vendor pairing |
| Codex | GPT | subscription | vendor pairing |
| opencode | GLM | API key | **neutral configuration**: an open, provider-agnostic scaffold |

The primary comparison is between agent systems as their vendors pair them;
models and scaffolds are trained together. The scaffold's and the model's
effects are confounded; that is accepted for now, and reports say so.
Separating them later needs one model under two scaffolds (GLM is the natural
candidate).

**Budgets.** Today the enforced budget is wall clock, rate-limit waits
included. Planned: working time (rate-limit waits excluded), a token budget,
and the service budgets, which are scoring requests (a count) and FPGA time
(minutes), both stated in the syllabus and enforced (§12). Reported: wall
clock, tool calls, service calls, queue waits.

**Equalised CLI conditions.** Off for every CLI: instruction files, web tools
(offline), memory, MCP, hooks, network-fetched plugins and skills, telemetry,
auto-update. CLI versions pinned. Task text byte-identical. Differences that
cannot be equalised are recorded: reasoning-effort scales, compaction, where
the system prompt sits.

**Repetition and reporting.** At least 5 independent runs per cell,
interleaved in time; paired comparisons on the same task and start; means with
bootstrap intervals, success rates with Wilson intervals; no best-of-k.
Simulated cycles are deterministic and need no repeated measurement; board
numbers do (A10). Confirmed cheating counts as failure. Hardware-changing
candidates are reported with their feasibility (pending, certified,
infeasible); hardware cost does not enter the score (§10.2).

**Validity classes.** Assigned today: `valid`; `infra-invalid` (the base did
not measure cleanly, or a pseudo-agent's patch did not apply);
`integrity-rejected` (the gate rejected the candidate); `budget-exhausted`.
Planned: `gave-up` (the agent declared failure and the gate took nothing), and
`infra-invalid` for a CLI that failed to start, a rate limit, an evaluator
timeout or a model mismatch. An early stop is `valid` today, with its reason
recorded.

**Pre-registration.** Metrics, budgets, cells and run counts are frozen before
the formal runs (`doc/r38_isca_roadmap.md` §9.4, §10). Pilots decide them.

## 14. The run record

One JSON per run (`run.json`), written outside the sandbox:

- **identity:** run id, task id and sha256, base commit, workspace commit, agent (account label or pseudo-agent), mode, prompt sha256;
- **versions:** harness version; for agent runs, the CLI version and launch switches and the image ids;
- **models:** requested; served, from the CLI's report;
- **usage:** tokens as the CLI reports them; gateway tunnels with bytes, and refusals;
- **time:** started, finished, seconds, the reason the agent stopped;
- **candidate:** the gate's verdict, the files taken, dropped, evaluator-owned changes, rejections;
- **evaluation:** both sets, point by point (with the device's own count, `done_cycles`) and in aggregate; machine changes; the validity class;
- **hardware:** whether the candidate changed the machine (the RTL files, the profile's changed parameters), and `certification`: `not-needed` or `pending`. Certification writes its own record, `certification.json`, beside this one (§10.4).

Planned additions: the evaluator's image id for every run, pseudo-agents
included; tokens normalised to one schema across CLIs (uncached input, cache
write, cache read, output, reasoning) and an API-equivalent cost from a dated
price table; every service request with its time, snapshot hash, charge and
result.

## 15. When the core is done

1. **`doctor`:** on each supported host class, every canary gives the expected result, and a host that cannot give a guarantee is refused.
2. **Scripted boundary checks:** each automated attempt is blocked or detected: read a hidden path or a task package; find an ArchBench point in the workspace; reach the internet; reach a service on the harness host; read `~/.ssh`; write outside the workspace; see another run; rewrite the workspace's git history to hide a change; add a symlink to evaluator data; shadow an evaluator-owned module; start a nested CLI on the credential; under-report tokens; raise `core_idle` early; edit the shell's counter; exceed the FPGA budget; read an FPGA host's files from RTL. A host-side result substitution is checked by the final human review (§10.5), or by P2.2 if automated scoring is later required.
3. **LLM red team:** deferred, not needed for now (decided 2026-10-15).
4. **Metering:** in API mode, the proxy's tokens and the CLI's report agree within a recorded bound.
5. **Replay tasks:** a no-op agent scores the base exactly, and the reference patch scores its known gain.

## 16. Decisions

**Decided.**

| Date | Decision |
|---|---|
| 2026-10-14 | Our own thin core rather than adopting Harbor or Inspect, with Harbor-compatible formats. The test environment is a separate, clean host |
| 2026-10-14 | P1 is prototyped on `main` by Claude; the user takes the contract design to the collaborator |
| 2026-10-15 | P1 is merged into the agent-harness branch (PR #9) rather than opened as a separate PR |
| 2026-10-15 | Network modes: `offline` and `open`, nothing in between |
| 2026-10-15 | Tokens: the CLI's report plus the gateway's bytes is enough for pilots; the API-key proxy is deferred |
| 2026-10-15 | No LLM red team or hacker-fixer loop for now |
| 2026-10-15 | The evaluator first at CHIA's level; the observer (P2) is the second step, and it is required |
| 2026-10-15 | Configurations: Claude Code + Claude, Codex + GPT, opencode + GLM (neutral). Vendor pairing is the primary comparison |
| 2026-10-15 | Tasks are cross-layer from the start |
| 2026-10-15 | Measurement-path files always come from the evaluator, whatever the task; the gate drops changes to them |
| 2026-10-15 | Strengthen the cheap estimators (performance model, correctness model, preliminary synthesis) rather than run the FPGA flow after every hardware change |
| 2026-10-15 | For now, while the harness is built, ArchBench is hidden from the agent in every sense; the agent writes its own test cases; the task text is a syllabus |
| 2026-10-15 | A wrong point fails alone; the run stays valid |
| 2026-10-15 | A scoring service, used at the agent's discretion, a limited number of times |
| 2026-10-15 | An FPGA service, used at the agent's discretion; FPGA time limited, stated in the syllabus, enforced; the agent sees results, never the points. Testing starts with the quick tier |
| 2026-10-15 | FPGA builds on the build worker; the card host only for the board stage |
| 2026-10-15 | For delivery, the benchmark is split into a public and a hidden package; the harness defines only visibility |
| 2026-10-16 | The decision brief's recommendations are adopted and built until the collaborator confirms them (rows marked *provisional*) |
| 2026-10-16 | *Provisional (O20, O4).* P2: we write the testbench observer and the collaborator reviews it; device time is the platform's count (`archbench/PROTOCOL.md` §3.1); the relay listens on a Unix socket; the board platform (`rtl/board/`, `rtl/fpga/`) is evaluator-owned; P2.1 first, P2.2 in the final evaluation only; candidate RTL may not use file or system tasks |
| 2026-10-16 | *Provisional (O2).* The contract's first version is the single-core `tnca_shell`; a second follows r40 |
| 2026-10-16 | *Provisional (O3).* A microarchitecture parameter's name matches `^[A-Z][A-Z0-9_]*$` |
| 2026-10-16 | *Provisional (O19).* The comments that revealed ArchBench are reworded and the tests and tutorial that duplicated points reshaped, in the repository |
| 2026-10-16 | *Provisional (O18).* Two hidden sets: validation (aggregates during the run) and final (only after it) |
| 2026-10-16 | *Provisional (O12).* The three aggregates are reported, and runs rank by `fast_1`, then `speedup_geomean`. Not built. (Its other half, hardware in the score through the clock and a resource cap, was replaced on 2026-10-17) |
| 2026-10-16 | *Provisional.* Until P1 reaches the FPGA top, the quick tier passes the scored profile to it as `NAME=VALUE`; the slow tier's board stage returns aggregates only, through a request schema of its own |
| 2026-10-16 | *Provisional (O16).* Pilot budgets: 3 scoring requests; 45 FPGA minutes, up to 20 per request (35 let only one request fit, §19) |
| 2026-10-17 | The Vivado flow (package, link) runs on the build worker with Vitis 2022.2; the card host is used only to load and measure. Validated once on the card (§12.5). Closes O13 |
| 2026-10-16 | The machine is the Alveo U280 (agreed with the collaborator). Hardware cost is not scored; fitting the U280 and closing timing is a constraint (§10.2). Retires O11 |
| 2026-10-17 | Latency is device time, the platform's count, host transfers left out, at the candidate's achieved clock: the agent declares a target clock, the certification link runs at it, and a miss is linked once more at the frequency the routed slack allows (rounded down) and scored there (§10.2) |
| 2026-10-17 | Feasibility: the quick tier screens; every hardware-changing final candidate that passes is certified by a routed link on the build worker; its score is pending until then |
| 2026-10-17 | A candidate that does not fit (resources beyond SLR1, or no route at any reasonable clock) fails every point and the run stays valid; its diagnosis and a reference latency at the base clock are recorded, not scored |
| 2026-10-17 | O23: `replay-mzero` starts from and is compared with core-vrf-sync-v5 (u280_r39h), once the flow can build a named profile (r40's `TNCA_IMAGE_CONFIG`). Decided by the user; the collaborator to confirm |
| 2026-10-18 | *A first version, for now in place of the three entries on latency and feasibility above:* the kernel clock is fixed at 100 MHz and the score is the simulated cycles; a final candidate that changed the machine is certified on the card (`certify`, §10.4) -- it fits, closes 100 MHz, and each point counts on the card the cycles simulation counted -- or every point fails. The achieved and declared clock, the screen, the diagnosis and the reference latency are deferred |
| 2026-10-18 | O23 built: the flow builds a named profile (`d1a00dd`: r40's `TNCA_IMAGE_CONFIG` and the U280 top's pass-through, proposed to the collaborator); `replay-mzero` moves to base `083b57d` (`9a3b1378` plus those three files) and core-vrf-sync-v5 |
| 2026-10-18 | The RTL check also refuses, where the candidate adds them, synthesis pragmas, tool-macro branches, `defparam` and hierarchical references (§11) |
| 2026-10-18 | `replay-mzero` is a test task (it validates the loop: `noop` 1.000, `reference` the known gain); its base commits stay on local branches. The formal tasks replace it, and their bases go to the remote. Decided by the user |
| 2026-10-19 | The operator reviews the final candidate and result path before accepting its score (§10.5). P2.2 is optional automation for unattended scoring, not a blocker for the manually reviewed loop. Decided by the user |

**Open.**

| # | Decision | Owner |
|---|---|---|
| O6 | Replay tasks: which commits (candidates: `49da68a`, compiler; `d795aa2`, MTX_ABUF, RTL) | collaborator |
| O7 | Whether Codex can drive GLM | to verify |
| O8 | Whether a CLI switches models automatically under subscription limits | to verify |
| O10 | When and how dedicated accounts are obtained; each account's training setting; each provider's data policy and plan terms | user |
| O14 | Windows for the shared card | user, lab |
| O15 | How specific a task's syllabus may be | user |
| O17 | The size of the hidden sets. Of the 27 tiny points, each matrix-multiply family has a single shape (matmul in two dtypes), so "same operation, unseen shape" needs new tiny-scale points | collaborator |
| O21 | How ArchBench splits into a public and a hidden package for delivery: by family, or by shape tables made data | collaborator |
| O22 | Who migrates P1's remainder (the FPGA top first, then the core benches and `tnca_shell_tb`) | collaborator |
| O24 | *Deferred with the screen.* The quick-tier screen's thresholds (share of SLR1, worst path at the declared clock), calibrated on the images that closed (v2, v4, v5) | us, with the collaborator |
| O25 | *Deferred with the achieved clock.* How the flow takes a requested kernel clock (`link.sh`, `package_xo.tcl` and `ooc.xdc` fix 100 MHz, which the score now assumes), and where a candidate declares its clock. The named profile is built (`d1a00dd`); the collaborator to confirm that change | collaborator |

## 17. Progress

Status: **done** (built and verified), **in progress** (being built now),
**planned (required)** (the user has required it), **planned**, **deferred**,
**not needed**. Evidence names the test, run or commit that shows it.

| Area | Item | Status | Evidence |
|---|---|---|---|
| Sandbox | Docker provider, hardening, resource limits | done | `doctor` 29/29 on the test host, class `formal` |
| Sandbox | The agent container gets the bridge script only, never the harness directory | done | `tests/test_runner.py`; fixed 2026-10-15, after the review found the task packages readable in the sandbox. Pilot 1's transcripts show no access to them |
| Sandbox | `doctor` canaries as described in §7 | done | `canary.py`, `doctor.py` |
| Sandbox | `doctor`: planted-file reads, disk space, cgroup v1, refusing runs | planned | — |
| Sandbox | The container uid taken from the site, not fixed at 1000 | planned | — |
| Sandbox | Podman provider | planned | needs a second host |
| Network | CONNECT gateway as sidecar, `offline` and `open` | done | `tests/test_gateway.py`; smoke runs |
| Network | `open` mode refuses loopback, private and link-local addresses | done | `tests/test_gateway.py` |
| Network | A site deny list for internal names behind an upstream proxy | planned | — |
| Network | API-key proxy | deferred | — |
| Accounts | Site configuration, credentials by path, never by value | done | `tests/test_siteconfig.py` |
| Accounts | Codex credential write-back only after checks | done | `tests/test_runner.py` |
| Agents | Adapters: Claude Code, Codex, opencode | done | smoke passes for all three, offline |
| Agents | Model-mismatch check, rate-limit detection, failed-start detection in task runs | planned | — |
| Agents | opencode served model after a timeout | planned | pilot 1: lost when the CLI was stopped |
| Workspace | Single-commit export, no reflog, instruction files and harnesses removed | done | `tests/test_workspace.py` |
| Workspace | A hide entry that matches nothing stops the export | done | `tests/test_workspace.py` |
| Workspace | Gate: design paths, evaluator-owned paths, rejections | done | `tests/test_workspace.py` |
| Workspace | Gate: ignore rules from the base tree; shadowing of owned modules | planned | §9 |
| Visibility | ArchBench hidden from the `replay-mzero` workspace | done | §6.5: 26 hide entries on base `9a3b1378`; no point id or shape string in the tree or the commit |
| Visibility | Syllabus for `replay-mzero`, with the services and their budgets | done | `tasks/replay-mzero/instruction.md`, `task.render` |
| Visibility | Residual leaks reworded; tests and tutorial reshaped | done | `6f2bdee`; torch tests 24/24 pass; task base `9a3b1378` |
| Visibility | Export-time leak check over the tree and the git objects | planned (required) | — |
| Tasks | Main-line v5 pilot with a remotely available base | done | `tasks/r39-v5-main-pilot/VALIDATION.md`: fresh export; observer-backed noop 3/3 validation and 2/2 final, each at 1.000 |
| Tasks | Formal tasks and wider hidden sets | planned | O6/O17/O21; `replay-mzero` remains a local framework regression whose base `083b57d` was not published |
| Visibility | TileBench in the image checked against ArchBench | planned | — |
| Evaluator | CHIA-level evaluator: base plus overlay, own copies | done | noop 1.000; reference 1.016 / 1.010 |
| Evaluator | The evaluator's platform from a pinned commit, laid over every tree | done | `task.json` `evaluation.platform` |
| Evaluator | Each point judged alone; validation and final sets | done | `tests/test_evaluator.py` |
| Evaluator | Machine-change record (profile `elaboration` and `rtl_parameters`) | done | pilot runs |
| Evaluator | `gave-up` class; provenance and diff audit | planned | — |
| Device boundary | P1: the profile header | done | commit `8c49e2b`, merged in `9035f0f`; parameters identical on nine builds |
| Device boundary | P1: FPGA top, core benches, `tnca_shell_tb` | planned | O22 |
| Device boundary | P2.1: the platform's count, the device in its own container, the RTL check | done | `55872ef`; §19: a halved shell counter scores 1.000, `$fopen` in RTL is rejected |
| Device boundary | P2.2: automated result-path provenance for unattended scoring | optional | §11; final human review is the acceptance gate (§10.5) |
| Services | Cheap estimators in the sandbox | planned | §12.1 |
| Services | Service channel and accounting | done | `services.py`, `service_cli.py`, `tests/test_services.py` |
| Services | Scoring service | done | §19: 48 s, the aggregate equal to the reference's validation numbers |
| Services | FPGA quick tier: license and isolation feasibility | done | OOC synthesis with no license and no network, with a control (§12.5) |
| Services | FPGA quick tier service: bubblewrap on the build worker, the agent's configuration, an enforced budget | done | §19: `ok` in 11.8 min, +2.313 ns at 10 ns |
| Services | FPGA slow tier | planned | waits on O14; the FPGA harness can now build on the build worker and measure on the card host |
| FPGA harness | The Vivado flow on the build worker, the board stage on the card host (`pipeline`, `sync`, `import`, host tool roots) | done | `tools/tritonchip_fpga_harness/tests/test_split_hosts.py`; between the two servers with a played build and no load; a 2022.2 build loaded on the card by hand (§12.5) |
| FPGA harness | `pipeline` between the two servers, a real candidate | done | 2026-10-17: the base hardware (v2, 100 MHz) linked on the build worker in 105 min, ap_clk WNS +0.138 ns; loaded on the card, conformance 5/5, ArchBench tiny 27/27, every point's cycles equal to the 2023.2 build's (both compiled with the card host's installed tree) |
| FPGA harness | The board stage compiles with the checkout's runtime and compiler, and refuses an interpreter that imports them from elsewhere | done | `969201b`; found by the v5 base certification (§19) |
| Metering | Tokens as reported, gateway bytes | done | run records |
| Metering | Cross-check; one token schema; API-equivalent cost | planned | — |
| Red team | Scripted (§15 item 2) | planned | — |
| Red team | LLM red team | deferred | not needed for now (decided 2026-10-15) |
| Loop | `run-task` with noop, reference, `patch:` and real agents | done | §19 |
| Pilots | Pilot 1: three configurations, one run each | done | §19; before ArchBench was hidden |
| Pilots | An agent that uses the services on its own | in progress | §19 pilot 2: Claude Code and Codex scored on their own; neither asked for FPGA, which a cycles-only score gave no reason to |
| Scoring | Latency at a fixed 100 MHz: the simulated cycles; `hardware` and `certification` in the run record | done | §10.2; `replay-mzero` on v5: noop 1.000, reference 1.017 / 1.009 (§19) |
| Scoring | Board certification of a final candidate (`certify`) | done | §10.4; `tests/test_certify.py`; the v5 base `certified` on the card, every point equal to simulation (§19) |
| Scoring | Latency at the achieved clock; the declared clock and the derated re-link; the screen, the diagnosis, the reference latency | deferred | §16 (2026-10-18); O24, O25 |
| Flow | A named profile in the build (r40's `TNCA_IMAGE_CONFIG`, the U280 top's pass-through) | done | `d1a00dd`; packaged on the build worker: v5 gives r40's 17 parameters, the default package is unchanged (v2) |
| Flow | A requested kernel clock | deferred | O25 |
| Device boundary | RTL check: pragmas, tool macros, `defparam`, hierarchical references, where the candidate adds them | done | `tests/test_certify.py` (`SplitRtl`); the base's 58 RTL files pass |

## 18. Quick start

From the repository root, on a Linux host with rootful Docker on cgroup v2.

**1. Site configuration** (yours, outside the repository):

```bash
mkdir -p ~/.config/tritonchip-agent-harness && cp tools/tritonchip_agent_harness/examples/site.json ~/.config/tritonchip-agent-harness/site.json
```

Edit it: accounts (CLI, auth mode, credential path, hosts, models), routes
(`direct` or `upstream`, and `network.upstream_proxy` for `upstream`),
`repository`, the clone workspaces are exported from (it must contain the task's
base commit and its pinned platform commit), and, for the FPGA service,
`fpga_worker`: a host you reach by SSH with keys, where Vivado is installed and
`bwrap` works (§12.5). Each credential is a file
you create, mode 0600:

- `CLAUDE_CODE_OAUTH_TOKEN=…`, from `claude setup-token`;
- a Codex home holding `auth.json`, from `codex login --device-auth` with `CODEX_HOME` set to it;
- `GLM_API_KEY=…`.

**2. Images.** The containers run as uid 1000 (§7), so build with
`--build-arg UID=1000`:

```bash
docker build --target agent -t tritonchip-agent:dev -f tools/tritonchip_agent_harness/image/Containerfile tools/tritonchip_agent_harness/image --build-arg TORCH_VERSION=2.14.0 --build-arg TRITON_VERSION=3.8.0 --build-arg TILEBENCH_COMMIT=6e2cf155dfe01f4660540d187c81b09eb55f394f --build-arg UID=1000 --build-arg NODE_VERSION=22.23.1 --build-arg NODE_SHA256=9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578 --build-arg CLAUDE_CODE_VERSION=2.1.287 --build-arg CODEX_VERSION=0.162.1 --build-arg OPENCODE_VERSION=1.19.0
```

Build `--target toolchain -t tritonchip-toolchain:dev` the same way.

**3. Check the host, then the accounts:**

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness doctor
```

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness smoke
```

**4. Validate the loop with no model, then run an agent:**

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness run-task tools/tritonchip_agent_harness/tasks/r39-v5-main-pilot --agent noop
```

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness run-task tools/tritonchip_agent_harness/tasks/r39-v5-main-pilot --agent <account label> --mode offline --budget-seconds 1200
```

Each run leaves `<state_dir>/runs/<run_id>/`: `run.json`, `prompt.md`,
`agent/` (the CLI's output and transcript), `candidate/` (the gate's verdict
and the files taken), `eval/` (the evaluator's logs and ArchBench records),
`gateway-log/`, and `workspace/`. Also: `--agent patch:<file>` re-scores a
change given as a git patch; `credentials` shows each account's credential
from its metadata only.

**5. Certify a candidate that changed the machine** (`run.json` says
`certification: pending`; §10.4). On the controller, with a clone that has
the task's base and `evaluation.certify.commit`, and the FPGA harness's hosts
file (its README §11.1). Copy the run's record and candidate from the
evaluation host:

```bash
mkdir -p runs && scp -r <eval host>:<state_dir>/runs/<run_id> runs/
```

Build the tree and see what would run, the candidate's code on the card server
included; nothing is built:

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness certify tools/tritonchip_agent_harness/tasks/r39-v5-main-pilot --run runs/<run_id>
```

Then link it on the build server and run its points on the card (about two
hours; `--build-run <id>` attaches to a build already started):

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness certify tools/tritonchip_agent_harness/tasks/r39-v5-main-pilot --run runs/<run_id> --hosts ~/.config/tritonchip-fpga-harness/hosts.json --build-host <build host> --board-host <card host> --yes
```

The verdict and the score after certification are in
`runs/<run_id>/certification.json`. `--base <baseline json>` (from
`<state_dir>/baselines/` on the evaluation host) certifies the base instead.

**The harness's own tests** (Python 3.9 or later, no Docker):

```bash
python3 -m unittest discover -s tools/tritonchip_agent_harness/tests -p 'test_*.py'
```

**The overview page** (a view of this file and the branch's history):

```bash
python3 tools/tritonchip_agent_harness/overview/build.py overview.html
```

## 19. What has been run

All on the test host (rootful Docker 27.5, cgroup v2, kernel 6.8),
2026-10-14 to 16. **Every agent run before "P2.1" below predates the hidden benchmark:** its
"dev" points (today's validation set) were named to the agent, and its
"held-out" points (today's final set) were visible in the workspace, so the
held-out numbers do not measure generality. The task package was also readable
inside the sandbox until the fix of 2026-10-15; the transcripts show that no
agent opened it.

**`doctor`:** 29 of 29 checks pass; host class `formal`. **`smoke`, offline:**
every account passes. For one "OK", each scaffold's fixed input is about 27k
tokens (Claude Code), 14k (Codex) and 6.7k (opencode).

**`replay-mzero`**, base `58ce8f98`: `main` plus P1, without `49da68a`'s m.zero
lowering. Three dev points and two held-out points, all of the tiny tier.

| Agent | Taken | Dev | Held-out | Note |
|---|---|---|---|---|
| `noop` | nothing | 1.000 | 1.000 | base and noop measure identically |
| `reference` | `frontend.py` | 1.016 | 1.010 | cycles equal `main`'s own, point by point |
| Claude Code + Sonnet 5.5, first run (earlier base, without P1), re-scored here | profile, target | 1.381 | 1.493 | enabled the existing `HAS_TENSOR_XPOSE` on core-sim |

**Pilot 1** (2026-10-15): one offline 20-minute run per configuration, one
after another. One run each: not a comparison.

| Configuration | Used | Change taken | Kind | Dev | Held-out |
|---|---|---|---|---|---|
| Claude Code 2.1.287 + Sonnet 5.5 | 3.8 min, 22 turns | core-sim profile: the seven R39 timing switches of core-vrf-sync-v5 | hardware, through the profile | 1.065 | 1.125 |
| Codex 0.162.1 + gpt-5.6-luna | 6.9 min | `rtl/core/tnca_core.v`: `.OPND_REG(0)`, which undoes timing-002's MAC operand register (the device's worst FPGA path, 10.19 → 9.30 ns) | hardware, RTL | 1.0009 | 1.0002 |
| opencode 1.19.0 + GLM-5.3 | 20 min (timeout) | none | — | 1.000 | 1.000 |

**What it showed, and what changed because of it.**

- **Two of three changed the machine, and simulated cycles charge nothing for hardware.** → The FPGA service (§12) and O11.
- **The held-out points were not hidden.** → The hidden benchmark (§6).
- **The cycle count is the design's own counter, and the measuring process runs candidate code.** → P2 is required (§11).
- **One model, two strategies across runs** (`HAS_TENSOR_XPOSE`, then the timing switches). → At least 5 runs per cell.
- **Claude's last message said it stopped "at the time limit" after 3.8 of 20 minutes.** A run's self-report is not its record.

**License test for the FPGA quick tier** (2026-10-15, on the build-worker
candidate): §12.5.

**Review of this design** (2026-10-15). Four reviewers checked this file
against the code, the user's decisions, its own consistency and the
repository, and a second agent tried to refute each finding. The review found
the task packages readable in the sandbox, internal addresses reachable in
`open` mode, silent hide entries and an unchecked credential write-back, all
fixed the same day, and many statements of this file that overstated what was
built, corrected here.

**After the fixes** (2026-10-15, on the hidden-benchmark workspace of 567
files): `doctor` 29 of 29; `smoke` for Claude Code passes with the bridge-only
mount; `noop` scores 1.000 on both sets (3/3, 2/2 passed); `reference` scores
1.016 on the validation set and 1.010 on the final set, as before. The
workspace has no `archbench/`, no r39 baseline and no reflog, and the prompt
names no point.

**P2.1** (2026-10-15, base `9a3b1378`, platform files from `55872ef`): the
device of every point in an evaluator-owned container, counted by the
platform's observer.

| Candidate | Validation | Final | What it tests |
|---|---|---|---|
| `noop` | 1.000 (3/3) | 1.000 (2/2) | the observer's count equals the old one on the base |
| `reference` | 1.016 (3/3) | 1.010 (2/2) | unchanged from before P2.1, point by point |
| `rtl/tnca_shell.v` reports half the cycles | 1.000 (3/3) | 1.000 (2/2) | the shell's counter no longer scores |
| `$fopen` in `rtl/tnca_shell.v` | — | — | integrity-rejected by the RTL check, before any measurement |

**Services, without an agent** (2026-10-15, the `reference` change as the
snapshot). Scoring: 48 s; the agent's view is `{n: 3, passed: 3,
speedup_geomean: 1.0162, fast_1: 0.667}`, the reference's own validation
numbers. FPGA quick tier on the build worker: `ok` in 710 s (11.8 min) for the
core-sim configuration; worst slack +2.313 ns at 10 ns, 130.1 MHz estimated,
0 of 340,477 endpoints failing; 317,438 LUT, 111,823 FF, 135 BRAM, 171 DSP;
the worst paths run from `x_va_idx_reg` into the vector FPU's FMA, 32 logic
levels. The first try failed on a file name: `vivado_ooc.sh` names its output
after every parameter, so the service calls the flow's Tcl with a short
directory of its own. At 35 minutes a request only one fitted in the 45-minute
budget; `job_minutes` is 20.

**Pilot 2: an agent with services** (2026-10-16, hidden benchmark, base
`9a3b1378`, offline, 60 minutes, 3 scoring requests and 45 FPGA minutes).

| Configuration | Used | Change taken | Validation | Final | Services |
|---|---|---|---|---|---|
| Claude Code 2.1.287 + Sonnet 5.5 | 10.7 min, 60 turns | `frontend.py`, `ir.py`: each tile load waits where its data is first needed, so the two matmul operand loads overlap | 1.0046 (3/3) | 1.0013 (2/2) | 1 score, after its own tests; no FPGA |
| Codex 0.162.1 + gpt-5.6-luna | 2.8 min | core-sim profile: `FE_NOBUBBLE` 1, `RCTX_SWITCH` 2, `LSU_INFLIGHT` 2, `FPU_OVERLAP` 1 | 1.0728 (3/3) | 1.1270 (2/2) | 1 score, then stopped before its result; no FPGA |

- **Both chose to score on their own,** once each: the syllabus and `tnca-service` were enough to find and use the service.
- **Claude tried the machine and dropped it.** In its own simulator it copied core-vrf-sync-v5's switches into core-sim: the banked VRF made matmuls 20–40% slower and `MTX_ABUF` changed nothing, so it reverted the profile.
- **Codex changed the machine and never asked what it costs.** Four timing switches, from its own 16×16 matmul (1065 → 1006 cycles), then a scoring request, then it stopped with the request running. Neither agent asked for FPGA. The reason is the score: it counts cycles only, the clock and the resources do not enter it yet (O11 is provisional and not built), and the syllabus says so. An agent that optimises the score has no use for synthesis, so the FPGA service matters only once hardware reaches the score. The first version (2026-10-18) brings in the fit: a changed machine must fit the U280, close 100 MHz and agree with simulation on the card, or every point fails (§10.4); the achieved clock is deferred.
- **Both stopped early:** Claude with 49 minutes left, Codex with 57.
- **A cancelled request's numbers were kept as its result.** Codex's scoring request was cancelled when it stopped, while its point containers were being removed; the half-measured aggregate (2 of 3 passed) went into the record as the result. Fixed: such numbers are kept only as `full`, and the syllabus says a request still running at the end is lost.
- **Tokens.** Claude: 120 input, 91,519 cache writes, 3,868,404 cache reads, 22,455 output; $1.36 at list price. Codex: 1,488,541 input, 1,418,496 of them cached; 7,800 output, 2,071 of them reasoning.

**`replay-mzero` on core-vrf-sync-v5** (2026-10-18, base `083b57d`, platform
`55872ef`). The base measured cleanly on all five points (the platform's
count: 1555, 8379, 2190, 4981, 9783 cycles; the device's own count is 6 less
per launch, and bmm makes two launches). `noop` scores 1.000 on both sets.
`reference` scores 1.0166 on validation (3 of 3 passed, 2 faster: the matmul
1555 → 1497, the linear 8379 → 8285; bmm unchanged) and 1.0088 on final (2 of
2, both faster), about what it gained on core-sim (1.016 / 1.010). Neither
changes the machine: both records say `certification: not-needed`.

**The v5 base certified on the card** (2026-10-18, `certify --base`, tree
`86230ed`: base `083b57d` with the platform of `969201b`). The build server
linked it at 100 MHz with core-vrf-sync-v5 as the image in 121 minutes: all 17
parameters reported set, ap_clk WNS +0.085 ns (the signed-off u280_r39h:
+0.088), design WNS +0.016 ns on the platform's own path, no failing endpoint
of 590 869; 336 716 LUTs, 227 533 registers, 367 BRAM tiles (r39h: 336 334,
227 496, 367). On the card the load replaced the image, conformance passed, and
each of the five points counted exactly the cycles the evaluator's simulation
of the base had counted (1549, 8373, 2178, 4975, 9777): `certified`.

The first attempt, from the same RTL, linked identically (+0.085 ns, the same
LUT count) and came back `board-disagrees` on four points. The card's counts
were not noise: they were, point by point, the simulated cycles of the
`reference` run (1491, 8279, 2178, 4917, 9721). The card host's interpreter
has an editable install of `tnca` that points at an older checkout, which has
the m.zero lowering the task base removes, and ArchBench had imported the
runtime and the compiler from it, not from the checkout under test. Run again
on the same image with the checkout's own packages first, every point equalled
the base's simulation. The FPGA harness now does that and refuses an
interpreter that imports them from elsewhere (`969201b`; its README §5.5);
every earlier board result on that host used the installed tree. Exact
equality is what showed it: the installed compiler was up to 4 % faster, and on
bmm not at all.

## 20. Sources

- CHIA: [arXiv 2606.27350](https://arxiv.org/html/2606.27350v3) (ucb-bar/chia). Rebuilds and re-simulates with "no AI involved", runs workers in Docker, keeps holdouts and references from the agent; assumes a non-adversarial LLM.
- Harbor v0.24.0, Inspect 0.3.280 / inspect_swe 0.2.71, checked against these requirements (notes kept outside the repository).
- [Claude Code: Legal and compliance](https://code.claude.com/docs/en/legal-and-compliance), [Authentication](https://code.claude.com/docs/en/authentication).
- [Codex: Authentication](https://learn.chatgpt.com/docs/auth).
- Project: `doc/r38_isca_roadmap.md` §9–§10, §16; `doc/r38_stage1_tasks_and_acceptance.md` S1-10; `doc/r38_board_bringup.md` §6.2 (OOC timing); `archbench/PROTOCOL.md`; `rtl/tnca_shell.v` (header); `tools/tritonchip_fpga_harness/README.md`.
