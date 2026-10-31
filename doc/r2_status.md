# R2 Status: Actual Progress by Work Package

Contract: [phase2_tasks_and_acceptance.md](phase2_tasks_and_acceptance.md).
Interface freeze: [r2_interfaces.md](r2_interfaces.md).
Prerequisites: [R1 contract](phase1_tasks_and_acceptance.md), [R1 status](r1_status.md),
[R1 sign-off](r1_signoff.md) (**R1 is ACCEPTED**).

This document records only components that have **already been executed with verified evidence archived**. Statuses are strictly classified into three categories: **DELIVERED**, **PARTIAL**, or **NOT STARTED**; there are no completion percentages.

**R2 is NOT signed off.** The table below records seven DELIVERED, five PARTIAL, and two NOT STARTED packages. The final verdict further requires: re-running the full R1 suite on the R2 candidate commit (DEP-02), grid/multi-batch scheduling (R2-09), host transport completion boundaries (R2-10), MLIR main path lowering (R2-04), remaining ABI boundaries (R2-06), end-to-end matrix and mutations (R2-13), and dual-environment release (R2-14).

## Core Properties Established in This Phase

One compilation, one artifact, repeated executions across different parameters, with intermediate results remaining in device memory:

```text
    compiler invocations: 1 for the first compile, 0 for four more
    alpha=2.5   SUCCEEDED  match=True
    alpha=-1.0  SUCCEEDED  match=True
    alpha=0.25  SUCCEEDED  match=True
    32 launches, alternating alpha: 32 correct
    device processes started: 1 (131 requests served in one)
    A->B->C with no host download in between: SUCCEEDED/SUCCEEDED, correct=True
```

Post-installation, repository removed from `sys.path`, without Triton or compiler:

```text
    $ tnca-sim prepare
      building core-sim -> ~/.cache/tnca/devices/core-sim-d4b7392.../tnca_device.vvp
    $ tnca-sim run axpy.tnca --arg n=16 --arg alpha=2.5 ...
      job 1: SUCCEEDED, 113 device cycles
      o_ptr: [6.0, 6.0, 6.0, ...]
```

## Work Package Status

| Work Package | Status | Actual Deliverables | Unmet Acceptance Criteria |
|---|---|---|---|
| R2-01 R1 transition / contract | **PARTIAL** | 84 semantic cases expanded to 170 execution IDs; `tools/r2/baseline.py` implements DEP-01 (rejects missing, forged, incomplete, or PARTIAL R1, reusing R1's own verifier); `doc/r2_interfaces.md` freezes dependency directions and adapter boundaries | DEP-02 requires re-running the full R1 suite on the R2 candidate commit; DEP-03 support matrix remains to be completed |
| R2-02 Installation / API | **DELIVERED** | PKG-01/02/04/05/06 all pass on Linux. Non-editable installation in two clean venvs followed by compile→run→download; probe asserts repository is not on `sys.path` and `tnca` is not imported from checkout before proceeding; handles paths containing spaces and arbitrary cwd; four failure classes each produce stable codes consistent between CLI and API; license, version, and 7 runtime assets complete, reinstallable with `--no-index --no-cache-dir`. **PKG-06 is newly added**: byte-for-byte comparison of 95 files in the wheel against tree sources — a `build/` directory had previously been committed to git, and setuptools only recopies sources that are newer, causing the wheel to package months-old snapshots where the host wrote 6 mailbox fields while the device expected 8 | None |
| R2-03 Compiler driver | **DELIVERED** | CMP-01…05 all pass: real Triton add, mul-add, and dynamic alpha AXPY compiled via production entry points and executed on RTL; six malformed inputs each rejected with corresponding error codes; R1 negative cases remain rejected through production paths under `-O2 --debug` | None |
| R2-04 MLIR main path | **NOT STARTED** | — | MLIR-01…05 |
| R2-05 Artifact / loader | **DELIVERED** | ART-01…06 all pass: single-byte modifications, truncation, section overlaps, duplicate JSON keys, unknown versions, and mixed sections from another artifact are each rejected; profile, ABI, or format mismatches rejected **in zero device requests**; container allows no path traversal | None |
| R2-06 Dynamic ABI | **PARTIAL** | ABI-01 (independent byte-level oracle covering 6 types and 22 values), ABI-02 (three sets of addresses and scalars with unchanged code hash), ABI-03 (nine bad parameter types, none launching a task), ABI-04 (post-submission container mutation has no effect) | ABI-06 runtime non-contiguous gather/scatter out-of-bounds fault (ABI-05 passes: declared in-place aliasing executes correctly, undeclared overlaps rejected in **zero device requests**) |
| R2-07 Cache / determinism | **DELIVERED** | CACHE-01…04: seven dimensions alter cache key, runtime parameters cannot enter the key; identical hashes across two clean environments; corrupted and truncated caches rejected; **compiler invocations counted rather than timed**, 0 compiler invocations across four reuses | None |
| R2-08 Runtime lifecycle | **DELIVERED** | RT-01…07: complete lifecycle, 32 consecutive launches without restarts, eight queued tasks with the 9th returning QUEUE_FULL, wait is idempotent with host timeout not being a verdict, buffer leases block conflicting access, two sessions remain isolated, dependency chains execute while cross-session dependencies are rejected | None |
| R2-09 Grid / admission | **PARTIAL** | GRID-01/02 (nine grid shapes, each PID executed exactly once, NPID reflects logical grid size, verified against independent oracle), GRID-06 (bijective mapping, out-of-bounds rejected). `t.program_id(0)` previously returned linear ID; transport now passes all six grid values | GRID-03/04/05/07 require four-tile persistent device |
| R2-10 Host transport / completion | **PARTIAL** | MEM-03 (A→B→C intermediate values remain in device memory throughout), MEM-05 (static: modules under `python/tnca` contain zero hierarchical references; dynamic: killing device causes download to fail with DEVICE_LOST rather than returning a value), MEM-06 (every request records direction, address, length, status; out-of-bounds rejected; canaries intact). Device bounds checking previously failed for 0xFFFFFFF0 due to 32-bit signed integer | MEM-01/02/04 require dirty cache lines and latency injection hooks on the four-tile profile |
| R2-11 Faults / recovery | **DELIVERED** | ERR-01…07 all pass: traps attributed to code/PC, session reusable after drainable faults, device deadline poisons session and fails queued tasks, recover establishes new epoch, late responses from old epochs discarded, killing device yields DEVICE_LOST with no orphan processes | None |
| R2-12 Reports / diagnostics / examples | **DELIVERED** | OBS-01 (resources measured from emitted code: AXPY uses 4/64 vector registers, unmeasured costs marked `unknown` rather than 0; machine instruction count cross-checked against code segment), OBS-02 (five user error classes each provide stable codes, locations, remedies, and no bare tracebacks), OBS-03 (`Session.capture_failure` emits replayable bundle, `tnca-sim replay` reproduces identical semantic error; control group with normal kernel must not reproduce), OBS-04 (three runnable post-installation examples) | None |
| R2-13 E2E / mutation | **PARTIAL** | E2E-01 (216 cases: 3 kernels × 8 values of N × 3 value seeds × 3 memory seeds, references computed independently in Python, sentinel tail and bounding canaries verified each run), E2E-02 (nine grids + real numerical kernels, empty grid performs no compute, three 3D grids execute), E2E-05 (16 launches plus dependency chain under each of three seeds), R2M-01/02/05/05b/06/08 six mutations all caught by designated tests | Detectors for E2E-03/04/06 and R2M-03/04/07 do not yet exist |
| R2-14 Dual-environment / release | **NOT STARTED** | — | REL-01…05 |

## Frozen Contracts

| Contract | File | Key Rules |
|---|---|---|
| Manifest v1 | `spec/r2_manifest_v1.schema.json` | Reject unknown or higher major **or minor**; reject unknown fields; reject duplicate JSON keys; `optional_metadata` is the only ignorable extension point |
| Container v1 | `spec/r2_container_v1.json` | Single file, no paths, every length is a claim checked against actual file size; atomic writes |
| ABI v1 | `spec/r2_abi_v1.json` | Six types, 8-byte slots, `offset = 8×index`, upper 32 bits must be zero, f32 encoded as bit pattern |
| Transport v1 | `spec/r2_transport_v1.json` | Monotonic sequence request/response mailboxes; late responses discarded by sequence number; device restart is an explicit epoch event |
| Error codes | `python/tnca/errors.py` | Codes are part of public interface: never renamed, never reused |

## A Boundary That Must Remain Documented

Contract §1.1 forbids misrepresenting capabilities not yet migrated to the real Triton frontend as "already supporting Triton GEMM".
R1's dedicated GEMM, multi-role, and rolled pipeline paths enter the production path via **`tnca-ir` inputs**; the manifest's `identity.source_kind` records this distinction, and CMP-05 verifies that it specifies `tnca-ir` rather than `triton`.

