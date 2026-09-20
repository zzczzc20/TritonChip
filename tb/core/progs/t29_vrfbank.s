// t29_vrfbank -- directed bank contention for the banked VRF
//
// doc/phase3_tasks_and_acceptance.md R3V2-CORE-02 and -03. Three sources in
// one bank must serialise; three across banks must go together; and every
// read client has to be seen really waiting, not inferred to have waited.
//
// The mapping is phys = vbase + idx, bank = phys % 4. With role 0's base at
// 0 that makes v0, v4, v8, v12 bank 0, and v1, v2, v3 banks 1, 2 and 3.
//
// On the REFERENCE this program is an ordinary FP kernel: four read ports,
// no contention, and the counters all read zero. That is the point -- the
// same program on both implementations, and the difference is the file.
//
// Every register holds a DIFFERENT value, and the program reads each one
// back and checks it. The first version set them all with `v.iota`, so
// every bank held the same sixteen numbers -- and a register file that
// returned the wrong bank's data returned the right answer anyway. A
// mutation that crossed the bank-to-requester wiring ran this program to
// completion and passed. Distinct values are what make a wrong bank
// observable at all.
.org 0
start:
    // ---- set up ---------------------------------------------------------
    // bank 0: v0, v4, v8, v12      banks 1, 2, 3: v1, v2, v3
    v.movi    v0, #0x11
    v.movi    v1, #0x22
    v.movi    v2, #0x33
    v.movi    v3, #0x44
    v.movi    v4, #0x55
    v.movi    v8, #0x66

    // ---- three sources, all bank 0: v0, v4, v8 --------------------------
    // One read port on that bank, so the three reads take three turns.
    vf.fma    v12, v0, v4, v8
    vf.fma    v12, v0, v4, v8
    vf.fma    v12, v0, v4, v8
    vf.fma    v12, v0, v4, v8

    // ---- three sources, three banks: v1, v2, v3 -------------------------
    // Different banks, so all three are accepted in the same cycle.
    vf.fma    v13, v1, v2, v3
    vf.fma    v13, v1, v2, v3
    vf.fma    v13, v1, v2, v3
    vf.fma    v13, v1, v2, v3

    // ---- the LSU reading a vector source --------------------------------
    // A vector store reads its data through the same ports the ALU uses,
    // so it contends for the same banks.
    .li       s1, #8192
    v.st.w    [s1, #0], v0
    v.st.w    [s1, #64], v4
    v.st.w    [s1, #128], v8

    // ---- every register read back through the same ports ----------------
    // Six reads, one per register, each through the operand collector.
    // A bank that answers for another bank shows up here as the wrong
    // number, in the register whose bank was crossed.
    .assert_lane v0, #0, #0x11, #90
    .assert_lane v1, #0, #0x22, #91
    .assert_lane v2, #0, #0x33, #92
    .assert_lane v3, #0, #0x44, #93
    .assert_lane v4, #0, #0x55, #94
    .assert_lane v8, #0, #0x66, #95
    // ...and again in the last lane, so a file that answers only lane 0
    // correctly does not get away with it.
    .assert_lane v0, #15, #0x11, #96
    .assert_lane v8, #15, #0x66, #97

    // ---- the matrix unit ------------------------------------------------
    // Its reads go to port 3 and wait a cycle each, whatever the banks do,
    // and they contend with the execute ports for the same four banks.
    m.zero    a0
    mb.init   b0
    mb.arm    s10, b0, #0, #1
    m.mma.async a0, v0, v4, s10
    mb.wait   s10
    exit
