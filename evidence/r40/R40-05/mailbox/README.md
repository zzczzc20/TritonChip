# Whole-grid mailbox transport

The simulation binding translates file requests to the synthesizable
persistent shell ports. Opcode 10 negotiates physical cores and the streaming
budget. Opcode 11 submits all grid dimensions, core mask and argument stride
in one request, using its sequence as the hardware launch ID. It does not loop
over programs on the host. Code/data transfers use public ports; no testbench
hierarchical memory access performs uploads, downloads or computation.

The real three-core mailbox test uploaded code, launched 455 three-dimensional
programs and downloaded every correct result. The 6145-program run also passed,
with exactly one opcode-11 launch and no legacy per-program opcode-3/9 launches.
Eleven admission/packet tests cover old-image refusal, malformed capabilities,
float/overflow axes, alignment, core masks, reserved role bits and bit 31.

The first binding omitted the sequence-zero boot response; it was corrected
before the successful runs. A deadline diagnostic before full hardware drain
poisons the simulation epoch and refuses further writes/launches. Platform
recovery and public runtime integration remain separate requirements.

This is an assembly-based protocol diagnostic, not standard Torch, routed
board, multicore performance or complete R40-05 acceptance evidence.
