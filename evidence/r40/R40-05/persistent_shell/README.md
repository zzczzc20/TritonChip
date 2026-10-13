# Persistent multicore shell checkpoint

`rtl/tnca_multicore_shell.v` binds the global dispatcher and coherent host port
to private synchronous instruction RAM in each physical core. Code writes
broadcast to every copy; code reads return core 0's copy. The code window is
[0, CODE_WORDS*8), with data above it. The caller uploads valid code before
execution. Every launch carries a 64-bit ID, grid dimensions, active core mask,
role/slice settings, argument base/stride and vector bases. Completion retains
its ID/status/cycles until acknowledged. Unaligned or out-of-window entry PCs
are refused without execution. Unaligned host addresses receive status 3.

The shell contains no file IO, delays or simulator memory calls. External
line memory is a tagged platform port. Code uploads, downloads and launches
are mutually excluded; data accesses use the coherent host NI. Instruction
fetch beyond the code window produces an explicit trap.

`make r40-multicore-shell` tests 1/3/8 cores using only public ports: 256 code
words uploaded and read back, invalid host/entry addresses, completion hold,
two 455-program grids, per-core issue/drain counts, code overwrite refusal
while a command is active, and download of all 455 results. The program is a
self-checking scalar memory kernel, not a standard Torch acceptance workload.

Global deadline/cancel/recovery, persistent mailbox binding, public runtime,
capability negotiation and role-SPMD partitioning remain to be integrated.
This checkpoint does not mark R40-05 accepted or establish board timing,
placement, bulk-DMA coherence or multicore performance.

The full 1/3/8-core runner passed. Its two-grid execution plus all public-port
code/data transfers finished at 79514 / 32752 / 20251 simulation cycles.
The preceding invalid entry cases returned status 3 with their own launch IDs;
normal grids returned status 0. The existing coherent-host tests and all four
legacy 2x2 system tests passed after adding the slice metadata plumbing.
