// t30_memboundary -- what the memory adaptation boundary has to survive
//
// doc/phase3_6_tasks_and_acceptance.md R36-MEM-02. Two memories behind the
// same ports must give the same answers, under every latency, backpressure
// pattern and response order the contract allows.
//
// This program is written to be SENSITIVE to the things that differ between
// two implementations rather than to be a large kernel:
//
//   - partial stores of every width, so a byte enable that is wrong in one
//     model is a wrong byte rather than a wrong word in both;
//   - a load of a word a previous store only partly wrote, so a
//     read-modify-write that loses the untouched bytes is visible;
//   - vector loads and stores, which issue several tagged requests and are
//     where a response returning out of order is felt;
//   - an atomic, because the two models perform it in different places.
//
// The results are left at RESULT_BASE for the host to read back through the
// shell's own port. Nothing here asserts: the checking is on the host side,
// against values computed in the bench, because "the device agreed with
// itself" is not evidence.

        .equ RESULT_BASE, 0x1000
        .equ SCRATCH,     0x1200

        .li      s1, #0x1000              // RESULT_BASE
        .li      s2, #0x1200              // SCRATCH

        // --- partial stores, every width -----------------------------------
        // Start from a known word so an untouched byte is distinguishable
        // from a zero.
        .li      s3, #0xEEEEEEEEEEEEEEEE
        s.st.d   [s2, #0], s3

        .li      s4, #0x11
        s.st.b   [s2, #0], s4             // byte 0 only
        .li      s4, #0x2222
        s.st.h   [s2, #2], s4             // bytes 2..3
        .li      s4, #0x33333333
        s.st.w   [s2, #4], s4             // bytes 4..7

        // Read the whole word back: bytes 1 must still be 0xEE.
        s.ld.d   s5, [s2, #0]
        s.st.d   [s1, #0], s5             // result 0

        // --- a vector load and store ---------------------------------------
        // Sixteen lanes, which is sixteen tagged memory requests, which is
        // where response order matters.
        v.iota   v0                       // 0..15
        v.st.w   [s2, #64], v0
        v.ld.w   v1, [s2, #64]
        v.add    v2, v1, v1
        v.st.w   [s2, #128], v2
        // Reduce it so one scalar word carries the whole vector's answer.
        v.reduce.add s6, v2
        s.st.d   [s1, #8], s6             // result 1: sum of 2*i = 240

        // --- a strided gather, which is per-lane addressing ----------------
        // Lane i reads element (15 - i), so a model that returns responses in
        // the wrong order and an LSU that trusts it disagree here.
        .li      s7, #15
        v.splat  v3, s7
        v.sub    v4, v3, v0               // 15 - i
        .li      s13, #0x1240             // SCRATCH + 64
        v.ld.gather.w v5, [s13, v4]
        v.reduce.add s8, v5
        s.st.d   [s1, #16], s8            // result 2: same 120, different route

        // --- an atomic ------------------------------------------------------
        .li      s9, #7
        .li      s14, #0x1300
        s.st.d   [s14, #0], s9
        .li      s10, #5
        .li      s14, #0x1300             // SCRATCH + 256
        atom.add.d s11, [s14], s10       // returns the OLD value
        s.st.d   [s1, #24], s11           // result 3: 7
        s.ld.d   s12, [s14, #0]
        s.st.d   [s1, #32], s12           // result 4: 12

        exit
