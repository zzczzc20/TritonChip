# MMIO v2: shared board binding, coherent FPGA wrapper and native RTL proof

Parent implementation committed by the user as f4ada06. This batch completes source/engineering checks and records joint functional validation on the jump host snapshot ~/tritonchip-r40-mmio. No matching multicore board image is loaded or accepted.

The new identity 0x00020000544e4341 refuses legacy images before writes. The shared board/RTL host binding negotiates physical cores, role chunks and memory outlets; its actual register driver reaches the persistent coherent mesh through independent AXI masters. Full opcode matching prevents unknown values truncating into launches. Host words go through private code RAM or coherent homes; DMA bypass is explicitly refused. Code/parameters are private per physical core. Global completion waits for AXI/link drain and publishes sequence-bound common-clock endpoints. Legacy launch32 capability/command are refused; physical opcode10 independently advertises the streaming grid budget.

Confirmed results:
- Native resident Torch (a+b)*0.5: 4097 elements, 257 logical programs, three physical cores, numerical PASS, peak concurrency3 and zero measured tensor staging. Each AXI outlet has real traffic: reads355/288/259, writes88/86/84.
- Native FP16 descriptor GEMM: main generated Triton kernel executes 2 descriptor loads, two program contexts per physical batch, nine logical programs, peak3, numerical/residency PASS. Auxiliary copy kernels use audited pointer lowering.
- Native FP16 producer/consumer pair: 4 descriptor loads, two pairs per batch, 512-byte scratch per program, nine logical programs, peak3, numerical/residency PASS.
- Final one-core/one-outlet and three-core/three-outlet register tests: upload/readback of assembled code, two exact-once 455-program 3-D memory grids, all 455 host readbacks, actual outlet traffic, sequence-bound timestamps with submission gaps, unknown/reserved/legacy opcodes, invalid flags/index, stale doorbell, cold host write, zero work and deadline stopping new issue while accepted programs drain. Both pass.
- Ten host identity/opcode/option boundaries pass. Four new modules pass Yosys proc/check/memory inference; eight bad geometries fail Icarus elaboration through explicitly undefined invalid-configuration cells.
- Backend registry passes3/3. Legacy FPGA gate passes8/8, including synthesis, twelve clock ratios, adapter mutations, reset ordering, register protocol and common-clock endpoints.

The engineering gate initially rejected simulator-only initial/$fatal guards and a Verilog-only reader. Guards now fail at elaboration in invalid generate branches, and the FPGA gate reads SystemVerilog. No guard or numerical check was disabled. The simulator-start boundary scan explicitly names the new simulation binding; the shared board module remains scanned.

The native tests ran before these final guard-only edits and the legacy capability8 refusal; valid data paths are unchanged. Final synthesis, invalid-geometry and one/three-core protocol tests ran after those edits. The optional image_profile=None normalization is covered by the final host boundary test. Hashes identify the final batch.

Remaining for R40-05/06: AXI-Lite platform top/kernel packaging, physical three-SLR placement/timing/CDC, matching-image native Torch board execution, coherent bulk DMA/ownership transfer and explicit lost-response recovery, fixed-resource bandwidth/performance and full acceptance. The separately audited common-clock image is single-core and does not contain this MMIO v2 implementation.
