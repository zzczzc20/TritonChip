# Native numerical execution extension

No required R40 gate is promoted. The complete-chip model remains unqualified.
All builds, oracle generation and execution ran on the GB10, with no model
workload added to the FPGA server.

[report.json](report.json) and [raw.tar.gz](raw.tar.gz) bind the immutable source
archive `1ddf8cf1524c16ff91ede05e743ee91d58dfaf43c8851b5b02f40aa754820ab4`,
model and probe binaries, loaded SystemC library, canonical contract lock,
configuration, compiler inputs, exact reference vectors and complete outputs.
The raw archive SHA is
`3d99faa7f72d7307e48a1812b15e1b06be8738ab91e0a799e8171d8a98e93bf3`.

The native executor now implements all five interchange-format conversions,
integer conversion and saturation, multiply-high, FP min/max modes, balanced
reductions, shuffles and SFU operations. Existing unsupported numerical
modifiers still fail explicitly; this is not complete numerical or ISA
qualification. Opcode existence and maskability are generated from the
canonical schema rather than inferred from the ALU's internal function numbers.
Undefined opcode, unsupported descriptor and trailing-command mutations are
retained as refusals.

The independent campaign uses the project's Fraction oracle for all narrow
source patterns and every destination format, structured/random binary32
conversions, Python arbitrary-precision multiply-high expectations, exact
Fraction integer rounding/saturation and the existing 60-digit decimal SFU
tolerance oracle. Results contain no mismatches. NaNs compare by class, as in
the existing RTL validator; no payload guarantee is invented.

Native integration also checks masked halfword memory access with explicit
FP16→FP32→FP16 conversion, integer/FP32 reductions, in-place permutation and
shuffle, masked insertion, predicate selection of both branches, all SFU
special-value paths, untouched tail lanes and the existing two three-core
grids. The SFU resource has finite pass occupancy and result-availability
events. Its timing constants remain uncalibrated model hypotheses.

The failed v8 FP16 fixture is preserved. It supplied halfword bytes to the
historical elementwise constructor, which actually generated 32-bit memory
instructions. The corrected fixture emits explicit narrow-memory TNCA IR;
the compiler's output, rather than the requested dtype label, determines the
reference storage layout. The original failure was not relabeled as a pass.

This extension does not establish original Torch graph execution, native77,
HBM14, the frozen model/holdout, physical clock frequency or a performance
speedup. Multi-role execution, asynchronous DMA/MMA, barriers, RF bank/port
resources, DCO epochs, finite router buffering, the runtime adapter and
end-to-end timing calibration remain required work.
