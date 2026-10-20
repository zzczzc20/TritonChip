# Public Session physical-grid integration

`Session(physical_grid=True, bench=...)` negotiates hardware cores before
starting the worker, validates the active mask, and allocates one aligned
parameter/descriptor/scratch block per configured core. Tensor pointers remain
shared; private descriptor and scratch addresses relocate to each core's block.
One whole-grid request executes the task. Event.device_cycles is its global
makespan and program_cycles is unset, rather than inventing serial samples.
Recover renegotiates the image. Legacy sessions retain their serial path.

The public Session/AOT test executes 455 three-dimensional programs on three
physical cores through the persistent mailbox. Every result is correct and an
atomic counter is exactly 455, with one host launch. It checks all private
parameter table copies. A genuine Triton GPU-lowered FP32 vector kernel also
passed 257 elements across 17 programs, including the final partial block,
again with one whole-grid request. A real unpartitioned role-SPMD artifact is
refused at admission without device requests or unfinished events. All seven
legacy runtime tests passed.

Full role-SPMD/pair partitioning, automatic backend/native Torch selection,
per-core counters, structured private scratch/descriptor execution tests,
physical board integration and all frozen performance criteria remain required.
This checkpoint is not complete R40-05 acceptance.
