// t39_switch -- two Role Contexts, each spending most of its time waiting on
// its own long-latency results (R39-05 RCTX_SWITCH).
//
// Every context runs this code at once (+allroles). Every value each
// computes depends on its role, so a result written into the other
// context's register, a dropped instruction that is never refetched, or one
// that executes twice, changes an asserted value:
//   A  a scalar pointer chase: every load's address is the last load's
//      result, and the sum of what was loaded plus the role is kept;
//   B  vector loads feeding a multiply chain (the MUL pipe) and an add of
//      role + 1, sixteen links;
//   C  an SFU chain: exp2(0 * x), eight links, each waiting the SFU's
//      latency on the last;
//   E  ordinary FP and a reduction issued while an SFU pass runs;
//   D  the results stored to the role's own words and read back.
// With RCTX_SWITCH an instruction held in R by one of these hazards gives
// way to the other context's; without it, the same program checks the same
// values, so the test also runs on every bench that does not have the switch.
.org 0
start:
    s.rdspecial s1, role_id
    s.addi      s20, s1, #1              // role + 1

// ---------------------------------------------------------------- A
    .li         s3, #chain
    .li         s7, #0                   // sum
    s.ld.d      s5, [s3, #0]             // offset of the next link
    .li         s9, #24
chase:
    s.add       s6, s3, s5
    s.ld.d      s5, [s6, #0]
    s.add       s7, s7, s5
    s.add       s7, s7, s1               // + role, every link
    s.addi      s9, s9, #-1
    br.nz       s9, chase
    // link k = 1..24 loads word k, which holds 8 * (k + 1), so the sum is
    // 8 * (2 + ... + 25) + 24 * role = 2592 + 24 * role
    s.muli      s10, s1, #24
    s.addi      s10, s10, #2592
    s.sub       s11, s7, s10
    .assert_eq  s11, #0, #10

// ---------------------------------------------------------------- B
    .li         s12, #vdata
    v.ld.w      v1, [s12, #0]            // i in lane i
    v.ld.w      v2, [s12, #64]           // 1 in every lane
    v.splat     v3, s20                  // role + 1
    v.movi      v8, #0
    v.add       v4, v1, v8
    .li         s9, #16
mulchain:
    v.mul       v4, v4, v2               // MUL pipe: x * 1
    v.add       v4, v4, v3               // + role + 1
    v.ld.w      v5, [s12, #64]           // a load between links, result unused
    v.add       v4, v4, v8
    s.addi      s9, s9, #-1
    br.nz       s9, mulchain
    // lane i: i + 16 * (role + 1)
    s.muli      s13, s20, #16
    v.splat     v6, s13
    v.add       v6, v6, v1
    v.cmp.eq    p1, v4, v6
    p.ballot    s14, p1
    .assert_eq  s14, #0xffff, #20

// ---------------------------------------------------------------- C
    v.ld.w      v9, [s12, #128]          // 0.0f in every lane
    v.ld.w      v10, [s12, #192]         // 1.0f in every lane
    v.add       v11, v1, v8              // integer i, about to be zeroed
    .li         s9, #8
sfuchain:
    vf.mul      v11, v11, v9             // 0 * x = 0 (x is finite)
    v.exp2.approx v11, v11               // 2^0 = 1.0
    s.addi      s9, s9, #-1
    br.nz       s9, sfuchain
    vf.cmp.eq   p2, v11, v10
    p.ballot    s15, p2
    .assert_eq  s15, #0xffff, #30

// ---------------------------------------------------------------- E
// Ordinary FP and a reduction issued while an SFU pass runs (FPU_OVERLAP
// lets the first through; the reduction still waits for the pass).
    v.ld.w      v13, [s12, #192]         // 1.0f
    v.ld.w      v14, [s12, #128]         // 0.0f
    v.exp2.approx v15, v14               // 2^0 = 1.0, sixteen lanes through one slice
    vf.add      v16, v13, v13            // 2.0
    vf.mul      v17, v16, v16            // 4.0
    vf.add      v18, v17, v13            // 5.0
    vf.reduce.add s23, v18               // 16 * 5.0 = 80.0
    vf.add      v19, v15, v18            // 1.0 + 5.0 = 6.0
    .assert_lane v15, #0, #0x3f800000, #60
    .assert_lane v15, #15, #0x3f800000, #61
    .assert_lane v18, #0, #0x40a00000, #62
    .assert_lane v18, #9, #0x40a00000, #63
    .assert_eq  s23, #0x42a00000, #64
    .assert_lane v19, #0, #0x40c00000, #65
    .assert_lane v19, #15, #0x40c00000, #66

// ---------------------------------------------------------------- D
    .li         s16, #results
    s.shli      s17, s1, #7              // 128 bytes per role
    s.add       s16, s16, s17
    s.st.d      [s16, #0], s7
    v.st.w      [s16, #64], v4
    s.ld.d      s18, [s16, #0]
    s.sub       s19, s18, s10
    .assert_eq  s19, #0, #40
    v.ld.w      v12, [s16, #64]
    v.cmp.eq    p3, v12, v6
    p.ballot    s21, p3
    .assert_eq  s21, #0xffff, #41

    s.rdspecial s22, role_id
    s.sub       s22, s22, s1
    .assert_eq  s22, #0, #50
    exit

.org 0x4000
// chain: word k (at byte 8k) holds 8 * (k + 1), so the chase walks forward
// one word per link: the first load reads word 0, the 24 links words 1..24.
chain:
    .dword 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128
    .dword 136, 144, 152, 160, 168, 176, 184, 192, 200
.org 0x4400
vdata:
    .word 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    .word 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    .word 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
.org 0x4800
results:                                 // 128 bytes per role, up to four
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
