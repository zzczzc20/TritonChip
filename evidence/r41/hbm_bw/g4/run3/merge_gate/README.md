# Strict compiler gate after the harness merge

On GB10, an independent `git archive` of merge-tree commit
`f7c3a627e3e466205e1717af3cd86283a9391705` passed `R1_STRICT=1 make compiler`
on 2026-10-30: 170 compiler unit tests and six IR RUN cases, exit 0. The
compiled RTL-backed DP test ran 16 programs; four BF16/FP16 size combinations
were not compilable on that test target because of resource limits. The test
did not claim those four combinations executed. This is a compiler gate, not
a substitute for the remaining board campaign or every RTL regression.

The gate built `tb_core.vvp` and the MAC_DP=1, MAC_CUTS=29 bench from this
snapshot. It reused `tnca-opt` from the earlier GB10 `r41g4_gate4` build,
SHA-256 `5805708133dd159bfc82ef19dd69ead9f0a62658aa5657592b7c9a6dcc51c958`;
the compiler CMake, C++ sources, includes and tools are unchanged between
that gate's source commit `55c71e0e6276a2e6941a0796387136d5aed3c293` and this
snapshot. The job was limited to six CPUs, one thread per numerical library,
and 16 GiB virtual address space per process. The exact script, log and exit
status are in [`compiler_strict.tgz`](compiler_strict.tgz), SHA-256
`d6b971c543cbe9f293edf2ad057e3d849e3304cbe2ea31ece2a714c7f0c73acc`.
