# Automatic physical-grid backend validation

Parent: faed778. Tests ran on the ARM jump host in ~/tritonchip-r40-multibackend with the locked Torch/Triton environment and real Icarus RTL.

- Native resident Torch, 4097 FP32 elements: 3-core and 8-core numerical smoke passed; one whole-grid submission per generated kernel, zero tensor staging during the measured call.
- Automatic public Session: genuine Triton vector tail, 455-program 3-D AOT oracle and exactly-once atomic count passed; unsupported role-SPMD artifact refused.
- Backend registry checks: 3/3 passed.
- Shared conformance: sim-file, sim-socket, sim-multicore all measured, five lifecycle/transfer/compatibility/completion/fault groups each passed, including relaunch after fault. Physical-grid replies contain global makespan, not retired counts; fault code/PC are explicitly unavailable.
- Native smoke ran before adding compiler version to the cache identity; that cache-only change was then exercised by conformance. RTL and native numerical test source were unchanged.

This evidence does not establish simultaneous core activity, board placement, speedup or full R40 acceptance. Hardware concurrency telemetry, role-SPMD/pair support and 1/2/3-core board gates remain open.
