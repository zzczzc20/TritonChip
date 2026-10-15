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
| A1 | Agents are untrusted and may be adversarial. Reward hacking is the expected behaviour, not an edge case | everywhere |
| A2 | Claude Code (Claude Max) and Codex (ChatGPT Pro) run on subscriptions. Their credentials must travel with the unmodified CLI | §8 |
| A3 | GLM runs on a Coding Plan, reached with an API key. With a key the credential could stay outside the sandbox; whether the plan's terms allow a key-injecting proxy and formal runs is to be checked (O10) | §8 |
| A4 | Until dedicated experiment accounts exist, the subscriptions are personal accounts, used for development only | §8 |
| A5 | A host offers rootful Docker on cgroup v2. (`doctor`'s limit canaries read cgroup v2 files; rootless Podman is planned) | §7 |
| A6 | The agent may change the compiler, ISA and RTL freely. The constraint is the chain: PyTorch → Triton → TNCA IR → ISA → RTL, nothing skipped | §10, §11 |
| A7 | The agent's sandbox has the full toolchain, the cycle-accurate simulator included. It measures its own test cases with it | §3, §6 |
| A8 | For now, while the harness is built, all of ArchBench is hidden from the agent (§6) | §6, §12 |
| A9 | The evaluator runs no candidate code except the design it evaluates. Today that design code (the candidate's compiler, and the parts of `python/tnca/` it may change) runs in the measuring process; P2 removes that (§11) | §10, §11 |
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
   after the run:  gate ──► final evaluator ──► final set + validation set ──► run record (never to the agent)
```

**The operator** runs experiments. They own the site configuration, their copy
of the hidden benchmark, the build worker, the card host and the card windows.
They are trusted; the agent is not. **The agent never runs on an FPGA host**;
only candidate code built or measured there by a service does, inside a sandbox
(§12.5).

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
| T3 | Tampering with measurement | editing the shell's counter, finishing a launch early, faking outputs in the runtime, computing on the host | observer (§11, P2) |
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
| **Network** | the agent from the internet and from the harness host | the gateway socket (model API); the service socket (planned) |
| **Visibility** | the agent from the benchmark | nothing about the final set; aggregates about the validation set (the services are planned) |
| **Submission gate** | the workspace from everything that measures it | a diff against the base commit, read from the source repository, never from the agent's `.git` |
| **Observation** | the design from its measurement | numbers the evaluator measured itself, at the device boundary (§11, planned) |

The rule that ties them together: **the agent owns the design; the evaluator
owns the observation and the questions.** No file is "protected" in the sense
that changing it rejects a run. A change is either part of the design, and goes
into the candidate, or it is ignored, because nothing that measures uses the
candidate's copy of it. Until P2, one exception stands: candidate code that
runs in the measuring process can still patch evaluator-owned modules at run
time (§11).

**How each boundary is built.**

| Boundary | Mechanism | Settings | Verified by | Status |
|---|---|---|---|---|
| Container | Docker, rootful: one container per run, from the `agent` image (`provider.py`). It gets the bridge script as one read-only file, never the harness directory, which holds the task packages | `--network none`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`, `--user 1000:1000`, `--init`, `--cpus`, `--memory` = `--memory-swap`, `--pids-limit`, tmpfs `/tmp` and home, the default seccomp profile, no Docker socket, no devices | `doctor` canaries (§7); `tests/test_runner.py` | done |
| Network | A CONNECT-only proxy (`gateway.py`) in a sidecar container on the host's network. It serves a Unix socket in a directory the agent container mounts read-only; `bridge.py` inside the sandbox turns that socket into `127.0.0.1:3128`, and the CLI is given it as `HTTPS_PROXY` | `offline`: the account's host allow-list, port 443. `open`: any public host, ports 443 and 80, never loopback, private or link-local addresses. TLS never decrypted; every tunnel and refusal logged as JSON lines | `tests/test_gateway.py`, `doctor`, smoke | done |
| Credentials | A fresh copy per run: `--env-file`, or a copied `CODEX_HOME` | never on a command line; transcripts copied out file by file, credential files skipped; a refreshed Codex credential is copied back only after checks (§7) | `tests/test_siteconfig.py`, `tests/test_runner.py` | done |
| Visibility | The workspace is written with git plumbing (`cat-file --batch`) from one commit of the source repository, minus the hidden paths, and committed again with a fixed author and date | no history and no reflog; instruction files, both harnesses, `evidence/` and the task's hidden paths removed; a hide entry that matches nothing stops the export | `tests/test_workspace.py`; the leak audit (§6.5) | done for `replay-mzero` |
| Gate | Compares the workspace with the base commit read from the source repository, never from the workspace's `.git` | design paths taken; evaluator-owned paths dropped; under a design path, a symbolic link, special file or nested repository rejects the candidate | `tests/test_workspace.py` | done, with the gaps in §9 |
| Evaluation | A separate container from the `toolchain` image, with the same hardening and no credential, on the base plus the candidate's overlay | its own copies of the testbench, ArchBench, the tools and the flow | the noop and reference runs | done |
| Observation (P2) | The simulator in its own container behind a socket transport; a counter and write monitor in the evaluator's platform | — | — | planned |
| FPGA build worker | A container per job | no network, Vivado mounted read-only, only the job's tree writable, CPU and memory limits, no home | the license test (§12.5): synthesis needs no license or network | planned |
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
- **The workspace export removes it.** For `replay-mzero`: `archbench/` and the 28 other paths the leak audit found (§6.5). A hide entry that matches nothing stops the export, because a typo, or a directory written without its trailing slash, would otherwise leave what it names in the workspace.
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

**Removed from the workspace (29 hide entries).** `archbench/`; the r39
ArchBench baselines (`spec/r39/baseline_archbench*.json[l]`); suite and workload
lists (`spec/r38/suites/dev.json`, `spec/r38/evidence_registry.json`,
`spec/r38/design_changes/xpose-dma-001.json`, `spec/r3_workloads.json`,
`spec/r35_workloads.json`, `spec/r3v2_inventory.json`); tests that are exactly
a point's workload (`tools/r38/torch_tests.py`, `tools/r38/dev_cases.py`,
`tools/r37/upstream_cases.py`, `tools/r37/aot_run.py`,
`tools/r38/triton_native_run.py`, `tools/r38/tdesc_tests.py`); the old scoring
copy `tools/r38/archbench/`; documents with per-point or per-family results
(`doc/architecture_overview.md`, `doc/compiler_overview.md`,
`doc/r38_board_bringup.md`, `doc/r38_pytorch_p1plus.md`,
`doc/r38_stage1_status.md`, `compiler/README.md`, `flow/fpga/README.md`); the
tutorials whose example computes a final-set point
(`tutorial/10_torch_compile.md`, `tutorial/examples/10_torch_compile.py`) or
quotes board cycles (`tutorial/11_running_on_the_u280.md`); and, as before,
the two r39 planning documents. None of them is read by the agent's flow
(staging, `tnca-sim prepare`, the compiler, the runtime); `make r38-torch`
and `make tutorial` lose their inputs.

**After removal:** no point id and no shape string appears in the exported
working tree or its commit. 21 files still name "ArchBench" (Makefile targets,
comments), which the audit judged harmless.

**Residual leaks the export cannot remove** (comments in files the flow needs;
for the main author to reword in the repository, O19):

| File | What it says | Rewording the audit proposes |
|---|---|---|
| `rtl/core/tnca_core.v` L252–253, L2297–2298 | the tiny tier has 27 points; 14 % of ArchBench's cycles were unnamed | "a few points"; "a sizeable share of measured cycles" |
| `rtl/board/tnca_u50_xo.v` L93, L105–106 | "ArchBench tiny 27/27" | "passed conformance and the profiler with every cycle count identical to RTL simulation" |
| `compiler/python/tnca_cc/partition.py` L1418–1423 | per-family results (linear, mlp, matmul, gemv, llama_decode at small and medium) | describe the mechanism, without families, tiers or numbers |
| `python/tnca/runtime.py` L169–172 | that `llama_decode/small` exists, and its output-head size | a generic case with no model, tier or size |

The audit also recommends reshaping, rather than hiding, the tests that are the
agent's main regression suite (`tools/r38/torch_tests.py`: change five shapes)
and the torch.compile tutorial (change its example's shape), both in the main
repository.

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

| | Subscription (Claude Max, ChatGPT Pro) | API key (GLM Coding Plan) |
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

**Credentials and data.** Personal subscriptions are for development only (A4).
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
the tools and the flow are the base's. It then runs ArchBench on the validation
and final sets, in a hardened container with no network and no credential, and
compares each point with the base's own cycles. The base is measured once per
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

### 10.3 Chain integrity and generality

1. **The evaluator drives the chain.** It starts from its own ArchBench PyTorch modules and runs `torch.compile` → Inductor → Triton → the candidate's compiler → the candidate's RTL. A candidate supplies a toolchain and a design, never kernels or binaries.
2. **A host fallback fails the point** (§10.2).
3. **The benchmark is hidden** (§6; for `replay-mzero` since 2026-10-15), so nothing can be keyed on its kernels.
4. **Planned:** provenance and a diff audit. Provenance must regenerate or verify each stage from TNCA IR to ISA with evaluator-owned code; the stage files the candidate's compiler writes (`TNCA_KERNEL_DUMP=1`) are only its own claim. The diff audit looks for large instruction literals and branches on kernel names or shape constants. Monitors fix the environment; they are not part of the score.
5. **ISA extensions are allowed.** They must take effect through compiler lowering. How their resources are charged is open (O11); until then the FPGA service reports their utilisation (§12) and hardware-changing candidates are reported separately (§13).

**Machine changes are recorded.** A candidate may change the profile it is
measured on. The evaluator records every changed `elaboration` and
`rtl_parameters` entry of the task's profile; other profile keys, compiler
target files and RTL changes appear in the gate's list of files taken.

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

**P2, the observer (planned; required).** Two facts make it necessary.
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

**P2.1: the cycle count (the first step).**

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

**P2.2: the results (the second step).** The rule: **the host may move data,
never compute it.** The relay sees every upload and download. The evaluator
checks that every upload is an input of the point (generated from its seed by
base code), a block the device returned earlier, or the program and its
arguments; that the program images and arguments are identical when the point
runs again with a second seed; and that the outputs, taken from the device's
downloads rather than from the candidate's record, match a reference computed
by base code. Provenance is judged by content, not by which side wrote memory:
host writes reach memory through the agent's own shell (`rtl/tnca_shell.v`
merges host and core requests onto one memory port), which could delay them
into a launch window. The second run makes P2.2 a final-evaluation step only.

| Cheat | Caught by |
|---|---|
| The shell's counter edited | P2.1: the score uses the platform's count |
| Completion raised early, work continued | P2.1: memory requests outside the launch window |
| Work started while inputs were uploaded | P2.1: memory requests outside the launch window |
| The cycle number rewritten in the measuring process | P2.1: the observer log lives in the device container |
| Results computed on the host | P2.2: an upload that is neither an input, a returned block nor the program |
| Results baked into the program image | P2.2: the image changes with the seed |
| The runtime fakes outputs | P2.2: outputs come from the device's downloads |

**Owned by the main repository:** the observer in the testbench (3), the relay
(4), and `PROTOCOL.md` §3.1 (8, O4). The rest is the harness's. The open points
are being settled with the main author (P2-a to P2-f in the decision brief).

## 12. In-run services: scoring and FPGA

### 12.1 Cheap estimators first

The FPGA flow is slow, so the agent should rarely need it (user direction,
2026-10-15): strengthen the estimators inside the sandbox instead. That means a
fast performance model, a stronger correctness model, and a preliminary
synthesis mapped to the FPGA (LUT, FF, DSP and BRAM estimates, logic levels,
and an Fmax estimate calibrated against Vivado). Today the sandbox has the
cycle-accurate simulator and generic Yosys synthesis (`make synth`). Planned.

### 12.2 The channel

One Unix socket, mounted into the sandbox like the gateway's, and a small CLI
inside it (`tnca-service`, name provisional): `submit score`,
`submit fpga-quick`, `submit fpga-slow` (waits on O13, O14), `status`,
`result`. A request does not carry files: the harness takes the snapshot
itself, through the same gate as the final evaluation, so a service measures
exactly what the final evaluator would. Requests are asynchronous; the agent
may keep working while one runs. Every request and result is written to the run
record. Planned.

### 12.3 The scoring service (decided; not built)

Measures the validation set in simulation, with the final evaluator's code, and
returns an aggregate (§6.4). The number of requests per run is fixed and stated
in the syllabus (O16). It runs on its own CPU allocation, so it neither slows
the agent nor lets two concurrent runs disturb each other.

### 12.4 The FPGA service (decided; not built; the quick tier's feasibility is verified)

**The agent decides whether and when to use it. Its time is limited, the
syllabus says by how much, and the limit is enforced** (decided 2026-10-15).

| Tier | What runs | About | Returns |
|---|---|---|---|
| **Quick** | out-of-context synthesis and timing of the whole device, `flow/synth/vivado_ooc.sh tnca_fpga_top <period>`, on the profile the run is scored on | 25 min | status; worst slack at the target period and the frequency it implies; the worst paths, as start and end points in the agent's own RTL; utilisation (LUT, FF, DSP, BRAM, URAM) |
| **Slow** | implementation through the FPGA harness (package, link, judged timing and DRC), then a board stage on the validation set | hours | timing closure (WNS, TNS, failing endpoints); utilisation; the board stage's aggregate |

**The quick tier must time the scored profile, with the candidate's changes.**
Until P1 reaches the FPGA top, whose parameters are set by hand to
core-vrf-sync-v2's, the service passes that profile's parameters to
`vivado_ooc.sh` as `NAME=VALUE`.

**The slow tier's board stage returns only an aggregate.** It runs the
validation set from the evaluator's tree, never whole tiers, and never returns
the FPGA harness's per-point records, logs or dispatch tails. The FPGA harness
today runs whole tiers and records per point, so agent-triggered jobs need a
request schema of their own.

**Enforcement.**

- Each run has an FPGA time budget, charged as worker time.
- One FPGA request per run at a time.
- A request whose maximum duration exceeds the remaining budget is refused.
- A job is killed at its deadline; the time it used is charged.
- Queue waits are not charged to the FPGA budget; the run's wall clock goes on.
- Pilots run one agent at a time on the build worker, so no run waits behind another.

**Synthesis estimates are optimistic.** Before routing, delays leave out most
of the wire: on this design the routed worst path was about 1.6 times the
post-synthesis estimate. The quick tier ranks designs and catches paths a
change made worse; the frequency a design closes at is the slow tier's to say.
The syllabus says so.

### 12.5 Where the FPGA work runs, and how it is isolated

**Placement (decided 2026-10-15).** All FPGA builds run on the **build
worker**; the **card host** is used only for the board stage. "FPGA hosts"
means both. Both will be named in the site configuration.

- **The build worker** runs each job in a container: no network, the Vivado installation mounted read-only, only the job's tree writable, CPU and memory limits, no home directory. Candidate RTL is read by Vivado, so the sandbox is what stops `$readmemh` of a host file (T6); RTL is also linted for absolute or `..` paths in `` `include ``, `$readmem*` and `$fopen`.
- **The card host** runs the board stage. The candidate's compiler compiles the benchmark there, so that is candidate code on a shared machine. It runs in a sandbox with no network, a read-only view of the system, only the card's device nodes, and no home directory, harness state or compiler caches (`/tmp/torchinductor_*`, `~/.triton`) of anyone's. The card is shared, so board stages happen in agreed windows (O14), and between owners the card is reprogrammed or its memory cleared, so no session reads what another left in device memory.
- **License (tested 2026-10-15): none is needed.** Vivado 2022.2 ran OOC synthesis, timing and utilisation for the U280 and U50 parts with no license reachable and no network; Vivado reports the entitlement for these parts as built in. A control proved the license was really hidden: on a device that needs one, the same hiding turned a passing synthesis into a license error, and a file watcher saw the license file opened in every visible run and in no hidden one. The real unit `tnca_matrix` synthesised the same way in 1.8 minutes (worst slack +3.9 ns at 10 ns). Still to test: a real container image, which needs the installation's system libraries, a writable home for Vivado's own files, and a non-root user.
- **Versions (open, O13).** The repository's flow pins the U280 to Vitis 2023.2. A build worker that has only 2022.2 needs either a 2023.2 installation or a flow that accepts 2022.2 for the U280, followed by one full validation on the card.

## 13. Fair comparison and statistics

**Unit of comparison.** One cell is a (scaffold, model, auth mode) on a fixed
(task, start, budget, environment version).

**Configurations (decided 2026-10-15).**

| Scaffold | Model | Auth | Role |
|---|---|---|---|
| Claude Code | Claude | subscription | vendor pairing |
| Codex | GPT | subscription | vendor pairing |
| opencode | GLM | GLM Coding Plan (API key) | **neutral configuration**: an open, provider-agnostic scaffold |

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
numbers do (A10). Confirmed cheating counts as failure. Hardware-changing and
software-only candidates are reported separately until hardware cost enters the
score (O11).

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
- **evaluation:** both sets, point by point and in aggregate; machine changes; the validity class.

Planned additions: the evaluator's image id for every run, pseudo-agents
included; tokens normalised to one schema across CLIs (uncached input, cache
write, cache read, output, reasoning) and an API-equivalent cost from a dated
price table; every service request with its time, snapshot hash, charge and
result.

## 15. When the core is done

1. **`doctor`:** on each supported host class, every canary gives the expected result, and a host that cannot give a guarantee is refused.
2. **Scripted red team:** each attempt is blocked or detected, 100%: read a hidden path or a task package; find an ArchBench point in the workspace; reach the internet; reach a service on the harness host; read `~/.ssh`; write outside the workspace; see another run; rewrite the workspace's git history to hide a change; add a symlink to evaluator data; shadow an evaluator-owned module; start a nested CLI on the credential; under-report tokens; raise `core_idle` early; edit the shell's counter; compute on the host and write the result back; exceed the FPGA budget; read an FPGA host's files from RTL.
3. **LLM red team:** deferred, not needed for now (decided 2026-10-15).
4. **Metering:** in API mode, the proxy's tokens and the CLI's report agree within a recorded bound.
5. **Replay tasks:** a no-op agent scores the base exactly, and the reference patch scores its known gain.

## 16. Decisions

**Decided.**

| Date | Decision |
|---|---|
| 2026-10-14 | Our own thin core rather than adopting Harbor or Inspect, with Harbor-compatible formats. The test environment is a separate, clean host |
| 2026-10-14 | P1 is prototyped on `main` by Claude; the user takes the contract design to the main author |
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

**Open.**

| # | Decision | Owner |
|---|---|---|
| O2 | The contract's first version: single-core `tnca_shell`, the r40 multicore shell, or both. r40 (PR #7) rewrites much of the platform side | main author |
| O3 | Naming rules for new microarchitecture parameters | main author |
| O4 | Restating ArchBench device time against the observer (`PROTOCOL.md` §3.1) | ArchBench owner |
| O6 | Replay tasks: which commits (candidates: `49da68a`, compiler; `d795aa2`, MTX_ABUF, RTL) | main author |
| O7 | Whether Codex can drive GLM | to verify |
| O8 | Whether a CLI switches models automatically under subscription limits | to verify |
| O10 | When and how dedicated accounts are obtained; each account's training setting; GLM's data policy and Coding Plan terms | user |
| O11 | The final score's hardware term: device time as cycles × achieved clock period, a resource budget, or both | user, main author |
| O12 | How `passed`, `speedup_geomean` and `fast_1` combine into one score, and whether those aggregates are the ones | user |
| O13 | The slow tier's Vitis version for the U280: install 2023.2 on the build worker, or let the flow accept 2022.2 | main author |
| O14 | Windows for the shared card | user, lab |
| O15 | How specific a task's syllabus may be | user |
| O16 | The scoring-request count and FPGA minutes per run | user (pilots decide) |
| O17 | The size of the validation and final sets. Of the 27 tiny points, each matrix-multiply family has a single shape (matmul in two dtypes), so "same operation, unseen shape" needs new tiny-scale points | main author |
| O18 | Whether the hidden benchmark is split into a validation set (aggregates during the run) and a final set (only after the run), as proposed in §6.4 | user |
| O19 | Rewording the residual leaks in the repository (§6.5) and reshaping the tests and tutorial that duplicate points | main author |
| O20 | P2's details (§11): who writes the testbench observer, the relay change, the board platform's ownership, the RTL check, P2.2's method | main author |

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
| Visibility | ArchBench hidden from the `replay-mzero` workspace | done | §6.5: 29 hide entries; no point id or shape string left |
| Visibility | Syllabus for `replay-mzero` | done | `tasks/replay-mzero/instruction.md` |
| Visibility | Residual leaks reworded in the repository | planned (required) | O19; the user's rule: no point visible in any sense |
| Visibility | Export-time leak check over the tree and the git objects | planned (required) | — |
| Visibility | TileBench in the image checked against ArchBench | planned | — |
| Evaluator | CHIA-level evaluator: base plus overlay, own copies | done | noop scores 1.000, reference 1.016 / 1.010 |
| Evaluator | Each point judged alone; validation and final sets | done | `tests/test_evaluator.py` |
| Evaluator | Machine-change record (profile `elaboration` and `rtl_parameters`) | done | pilot runs |
| Evaluator | `gave-up` class; provenance and diff audit | planned | — |
| Device boundary | P1: the profile header | done | commit `8c49e2b`, merged in `9035f0f`; parameters identical on nine builds |
| Device boundary | P1: core benches, `tnca_shell_tb`, FPGA top | planned | — |
| Device boundary | P2: the observer | planned (required) | design in §11 (P2.1, P2.2); main-repository parts wait on O4, O20 |
| Services | Cheap estimators in the sandbox | planned | §12.1 |
| Services | Service channel and accounting | planned (required) | both services need it |
| Services | Scoring service | planned (required) | decided 2026-10-15 |
| Services | FPGA quick tier: license and isolation feasibility | done | OOC synthesis with no license and no network, with a control (§12.5) |
| Services | FPGA quick tier service: worker container, scored-profile parameters, enforced FPGA budget | planned (required) | decided 2026-10-15 |
| Services | FPGA slow tier | planned | waits on O13, O14 |
| Metering | Tokens as reported, gateway bytes | done | run records |
| Metering | Cross-check; one token schema; API-equivalent cost | planned | — |
| Red team | Scripted (§15 item 2) | planned | — |
| Red team | LLM red team | deferred | not needed for now (decided 2026-10-15) |
| Loop | `run-task` with noop, reference, `patch:` and real agents | done | §19 |
| Pilots | Pilot 1: three configurations, one run each | done | §19; before ArchBench was hidden |

## 18. Quick start

From the repository root, on a Linux host with rootful Docker on cgroup v2.

**1. Site configuration** (yours, outside the repository):

```bash
mkdir -p ~/.config/tritonchip-agent-harness && cp tools/tritonchip_agent_harness/examples/site.json ~/.config/tritonchip-agent-harness/site.json
```

Edit it: accounts (CLI, auth mode, credential path, hosts, models), routes
(`direct` or `upstream`, and `network.upstream_proxy` for `upstream`), and
`repository`, the clone workspaces are exported from. Each credential is a file
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
PYTHONPATH=tools python3 -m tritonchip_agent_harness run-task tools/tritonchip_agent_harness/tasks/replay-mzero --agent noop
```

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness run-task tools/tritonchip_agent_harness/tasks/replay-mzero --agent reference
```

```bash
PYTHONPATH=tools python3 -m tritonchip_agent_harness run-task tools/tritonchip_agent_harness/tasks/replay-mzero --agent <account label> --mode offline --budget-seconds 1200
```

Each run leaves `<state_dir>/runs/<run_id>/`: `run.json`, `prompt.md`,
`agent/` (the CLI's output and transcript), `candidate/` (the gate's verdict
and the files taken), `eval/` (the evaluator's logs and ArchBench records),
`gateway-log/`, and `workspace/`. Also: `--agent patch:<file>` re-scores a
change given as a git patch; `credentials` shows each account's credential
from its metadata only.

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
2026-10-14 and 15. **Every agent run below predates the hidden benchmark:** its
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

## 20. Sources

- CHIA: [arXiv 2606.27350](https://arxiv.org/html/2606.27350v3) (ucb-bar/chia). Rebuilds and re-simulates with "no AI involved", runs workers in Docker, keeps holdouts and references from the agent; assumes a non-adversarial LLM.
- Harbor v0.24.0, Inspect 0.3.280 / inspect_swe 0.2.71, checked against these requirements (notes kept outside the repository).
- [Claude Code: Legal and compliance](https://code.claude.com/docs/en/legal-and-compliance), [Authentication](https://code.claude.com/docs/en/authentication).
- [Codex: Authentication](https://learn.chatgpt.com/docs/auth).
- Project: `doc/r38_isca_roadmap.md` §9–§10, §16; `doc/r38_stage1_tasks_and_acceptance.md` S1-10; `doc/r38_board_bringup.md` §6.2 (OOC timing); `archbench/PROTOCOL.md`; `rtl/tnca_shell.v` (header); `tools/tritonchip_fpga_harness/README.md`.
