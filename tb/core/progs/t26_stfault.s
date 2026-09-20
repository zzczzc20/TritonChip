// t26_stfault -- `t.st.async` is encodable but this engine cannot perform it.
//
// ISA sec 9E allocates func 0x08 to a window -> global transfer. The tensor
// engine has a vector WRITE port and no read port, so it cannot source one.
// It used to be excluded from the memory-issue condition and nothing else,
// which meant a store entered the transfer state, issued no request, never
// saw a response, and sat there until the testbench timed out: a decodable
// instruction that deadlocked the core.
//
// sec 1.6: a capability the machine does not have is a fault, not an
// approximation and not a hang. The store raises ILLEGAL_INSTRUCTION, the
// barrier is signalled with its error bit so the waiter is released rather
// than left hanging, and the issuing context traps.
//
// The load below is the positive twin. It uses THE SAME descriptor and the
// same window, so if it completes the descriptor geometry is valid and the
// only thing left to explain the fault is the direction.
.org 0
start:
    w.def     w0, #32, #8
    .li       s1, #24576
    td.load   d0, [s1]

    // ---- positive twin: the same descriptor loads cleanly ----
    mb.init   b0
    mb.arm    s3, b0, #0, #1
    .li       s10, #0
    .li       s11, #0
    t.ld.async.h w0, d0, s10, s3
    mb.wait   s3

    // ---- the store: must raise ILLEGAL_INSTRUCTION (0x01) ----
    mb.arm    s4, b0, #0, #1
    t.st.async.h w0, d0, s10, s4
    mb.wait   s4                    // released by the error completion
    trap      #99                   // must be unreachable

.org 0x4000
src:
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000

.org 0x6000
desc_a:
    .dword 0x0000000000004000, 0x0000000000000010, 0x0000000000000020, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000040, 0x0000000000000002
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000001000000010
    .dword 0x0000000000000000, 0x0000000200000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    .dword 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
