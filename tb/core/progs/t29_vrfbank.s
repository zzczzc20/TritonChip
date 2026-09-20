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
.org 0
start:
    // ---- set up ---------------------------------------------------------
    v.iota    v0
    v.iota    v1
    v.iota    v2
    v.iota    v3
    v.iota    v4
    v.iota    v8

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

    // ---- the matrix unit ------------------------------------------------
    // Its reads go to port 3 and wait a cycle each, whatever the banks do,
    // and they contend with the execute ports for the same four banks.
    m.zero    a0
    mb.init   b0
    mb.arm    s10, b0, #0, #1
    m.mma.async a0, v0, v4, s10
    mb.wait   s10
    exit
