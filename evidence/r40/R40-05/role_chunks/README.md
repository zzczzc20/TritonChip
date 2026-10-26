# Physical program chunks: joint validation

Parent: 9e968f3. Authoritative simulation snapshot: jump host ~/tritonchip-r40-chunks-v2. Compiler, dispatcher, shell and runtime were implemented together; the complete simulation validation batch is recorded here.

Confirmed results:
- Native resident Torch FP32 add/multiply, 4097 elements, 257 programs: 4 logical program roles per physical batch, numerical PASS, peak physical cores 3, completed counts 88/84/85, one host launch, zero tensor staging.
- Native FP16 descriptor GEMM: two program contexts per batch, 2 descriptor loads, peak physical cores 3, 9 logical programs complete, numerical PASS and zero resident-call staging.
- Native FP16 producer/consumer GEMM: two program pairs, 4 descriptor loads, 512-byte scratch per program, peak physical cores 3, 9 programs complete, numerical PASS and zero resident-call staging. Auxiliary device copy kernels use pointer lowering and are audited independently; they need no descriptor or pair split.
- Genuine Triton 13x7x5 grid: width 4, 455 outputs and exactly 455 i32 atomic increments, tail sentinels unchanged, one host launch, isolated per-core parameter tables. The attempted i64 frontend atomic was explicitly refused; existing legacy IR/hardware 64-bit atomic oracle remains separate and unchanged.
- Combined regressions: 40 bounded IR cases, 21 protocol/admission cases, 5 legacy admission cases, registry 3/3, conformance all 3 simulation backends measured, dispatcher 20 configurations (including 2/4-wide chunks and >4096 tails), public automatic Session and persistent shell/deadline cases passed.

A numerical failure before the fix (3264/4097 differing elements) identified all-zero vector bases. The synthesizable shell now assigns private 64-register ranges, with pairing remapped inside the core. Inactive chunk lanes exit before descriptor/scratch side effects.

The earlier non-chunk 8-core native run is now PASS: 257 logical programs, peak physical concurrency 8, numerical correctness and zero resident-call staging. Native 8-core chunk test is now PASS: 257 programs, peak physical concurrency 8, numerical correctness and zero resident-call staging. The chunk artifact grid_dims_supported declaration was corrected locally to [true,true,true] after the real 3-D proof; Final metadata passed the real 3-D oracle in ~/tritonchip-r40-chunks-metadata; the only AOT source difference from v2 is that declaration. All final snapshot source hashes match the working tree. The v2 and final AOT sources differ only in the supported-axis declaration, and the final metadata passed the real three-dimensional oracle. No three-SLR board placement or performance gate is claimed; full R40 remains unaccepted.
