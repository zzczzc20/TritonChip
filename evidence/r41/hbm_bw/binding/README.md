# Packaged 32-PC stream CU and two-CU link flow

> **Quarantined (audit 2026-10-28, `../AUDIT.md` Q1).** The 32-PC stream lanes, bank and CU are a constructive exception under discipline §2.2.4. The engine is a fixed-function copy, fill and uint32-reduce unit that bypasses the cores, the compiler and the NUCA hierarchy. The work is kept as data. It must not be linked into an image, enabled in a runtime, or merged into or enabled on `r41` without the author's written approval. None of its results counts for any target.

## Board binding and packaging

`tnca_stream_hbm_xo.v` wraps the real 32-lane bank with the existing 32-bit AXI-Lite adapter and 32 named 512-bit AXI masters. It shares the 100 MHz clock and uses a synchronized active-low reset. `tools/r41/generate_stream_binding.py` generates the wrapper, kernel XML and combined connectivity file together; `--check` detects stale generated output.

The connectivity instantiates the existing `tnca_u50_1` and new `tnca_stream_1` in one xclbin. The three core masters retain their shared HBM[0:31] allocation group. Each stream master is pinned to its named PC. Kernel memory-argument metadata is used for connectivity; the custom register command interface controls execution, and the stream RTL supplies the fixed PC address tags.

The new `stream-hbm` packaging variant produced a real XO on the FPGA host:

- `~/r41_hbmbw_binding_v1/xo/tnca_stream.xo`; its SHA-256 and completed packaging log are retained in `package.log`.
- `packaged_metadata.tgz` retains the actual component.xml, kernel.xml and variant marker.
- `package_audit.json` verifies all 32 masters, all signal directions/widths and logical-to-physical port maps, 100 MHz clock association for all 33 AXI interfaces, and active-low reset association.
- Three metadata mutations (data width, clock and reset) are refused by the audit. `package_mutations.json` records those checks.
- `packaged_sources.sha256` matches the local RTL and packaging inputs to the actual remote source files.

An XO is a packaged IP container, not a linked FPGA image or bitstream. No target bandwidth or full-image timing is established by this artifact.

## Functional wrapper and flow checks

`python3 tools/r41/stream_binding_tests.py` passes real AXI-Lite writes/reads through the generated top. It programs the full aperture and descriptors, runs pattern writes on PC0 and PC31, checks every write byte and burst geometry, and verifies counters and drained completion. All inactive named masters must remain quiet. PC0 transfers 17 lines; PC31 transfers three lines across a 4 KiB boundary.

`link.sh` accepts an optional `TNCA_STREAM_XO` only alongside the U280 multicore core XO. It requires a distinct stream XO and matching `stream-hbm` variant marker, selects the combined connectivity/placement inputs, records the stream hash, and passes both XOs to v++. The legacy invocation retains its single-XO behavior.

`python3 tools/r41/stream_link_tests.py` verifies the actual shell command arguments using a recording compiler, which deliberately exits before implementation. It checks both-XO versus legacy inputs, the 100 MHz frequency request, stream connectivity/placement, and refusals for missing/wrong/duplicate stream artifacts. It also proves an incompatible full-run configuration is rejected before creating a build directory or invoking vendor tools. This is command construction evidence, not a Vitis link result. The initial test fixture used the same temporary directory name for an XO and a case; its failure is retained in `link_fixture_failure.log` and the corrected checks pass.

`run_r41_hbm_stream.sh` prepares both XOs and links them in a fresh directory. It requires a numbered full-run allocation (2..5), at least 38 GiB available memory and a core configuration with HBM_STRIPED=1, 256-bit legacy masters and three legacy ports. It never programs a board. **It has not been run: lead approval, combined screens and runtime integration remain required.**

## Physical results and current free job

The complete stream-bank OOC route finished successfully in SLR0 at 10 ns: setup WNS +3.516 ns, hold WHS +0.006 ns, zero TNS/THS. Routed area is 54,234 LUTs and 51,518 registers; synthesis used 240 BRAM tiles. Raw reports are retained under `../bank/screen_v1/period10/`. This is a bank-alone result without the core tiles or platform congestion.

A free synthesis screen of the exact core candidate is live on the FPGA host at `~/r41_hbmbw_kernel_v1/`, launcher PID 2618860. It uses `image/core_image.tcl`, derived from the existing integration parameter set with HBM_STRIPED=1, and runs through the shared slot guard with a 20 GiB threshold. `external_state.txt` confirms the process was live at 2026-10-26 08:42:33 UTC. No full-run budget slot has been consumed.

The current source has changes in core/generated RTL relative to the historical c9f257 image source. Therefore the old tile checkpoint is not assumed equivalent; current-source kernel and tile screens are required. Frozen Stage 1 acceptance identities remain unchanged.

The same read-only external-state check found no card context/client at that instant, with loaded UUID 1bada9db-78f3-5bc8-7ba6-cce062e7ade8. A single idle observation is not a 15-minute idle proof. No card was opened, reset or programmed.

## Next required work

Implement the runtime bank driver and an additional CU context on the already-open XRT device handle. It must not reload the image, must verify identity/capabilities and descriptor readback, and must close the peer context before its parent device handle. Serialize against core work and perform cache ownership handoff for the logical ranges.

Connect real native storage/tensor copies, with large-buffer alignment and semantic fallbacks. Add explicit integer read-reduce and bit-pattern write APIs. Verify that combination, then run combined tile/bank placement screens using the exact core synthesis checkpoint. Only after these gates should the lead allocate a numbered full image run. Board loading still requires explicit approval through the lead and the shared-card idle protocol.
