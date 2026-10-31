// t41_boundary -- the R41 Stage 2 core timing boundaries (F01, F02, F04, F05).
//
// Every context runs this program at once (+allroles, two or four of them),
// and every value depends on the role, so an operand forwarded from another
// context's write, a dropped or repeated instruction, or a stale operand
// changes an asserted value. Each section aims at one new register boundary;
// the core's own bench assertions (tnca_core, R41 F02/F04/F05) compare the
// registered operands, busy bits and physical registers with what the old
// combinational reads returned, on every cycle of every section.
//   A  scalar operands A, B and C forwarded from the instruction just ahead
//      (F02): X results, a MUL-pipe result, a load, S0, a branch condition;
//   B  predicate operands and the lane mask forwarded (F04): masked moves,
//      a negated mask, p.and / p.ballot, masked reductions;
//   C  RAW through the scoreboard with D's prefetch (F05): dependent vector
//      chains, a WAW, an FP-pipe chain;
//   D  an instruction stalled in R holding its collected operands while D
//      prefetches (F01: the operand registers are no longer stall-enabled);
//   E  taken branches squashing younger vector work (F01/F05);
//   H  tight loops whose load-use hazard sits right behind a taken branch,
//      so that a context giving way in R (RCTX_SWITCH) meets a redirect in
//      X, of its own context or another's (F01);
//   G  mb.wait / mb.arrive ping-pong between the roles of a pair, the wait
//      token built by the instruction right before it (F01/F02: the barrier
//      test now starts from a forwarded register). A role whose partner was
//      not launched skips it, so the program also runs with +roles=1, where
//      every instruction follows its producer directly and forwards.
.org 0
start:
    s.rdspecial s1, role_id
    s.addi      s20, s1, #1              // role + 1

// ---------------------------------------------------------------- A
    s.addi      s2, s20, #100            // r + 101
    s.addi      s3, s20, #200            // r + 201
    s.sel       s4, s2, s3, s3           // (s3 != 0) ? s2 : s3 = r + 101; B, C from X
    s.sel       s5, s3, s4, s0           // (0 != 0)  ? s3 : s4 = r + 101; B from X
    s.add       s6, s5, s4               // A from X, B from W
    s.subi      s7, s6, #202             // 2r + 202 - 202 = 2r
    s.shli      s8, s1, #1
    s.sub       s9, s7, s8
    .assert_eq  s9, #0, #1
    s.mul       s10, s20, s2             // (r + 1)(r + 101), MUL pipe
    s.add       s11, s10, s10            // waits for the MUL result
    s.muli      s12, s20, #2
    s.mul       s12, s12, s2
    s.sub       s13, s11, s12
    .assert_eq  s13, #0, #2
    .li         s14, #sdata
    s.shli      s15, s1, #3
    s.add       s14, s14, s15            // &sdata[role]
    s.ld.d      s16, [s14, #0]           // 1000 + role
    s.add       s17, s16, s16            // load result through W
    s.subi      s17, s17, #2000
    s.sub       s17, s17, s8
    .assert_eq  s17, #0, #3
    s.addi      s0, s2, #5               // S0 discards the write ...
    s.add       s18, s0, s2              // ... and still reads zero in X
    s.sub       s18, s18, s2
    .assert_eq  s18, #0, #4
    s.addi      s19, s20, #1
    s.addi      s19, s19, #1
    s.addi      s19, s19, #1             // r + 4, three back-to-back writers
    s.subi      s19, s19, #4
    s.sub       s19, s19, s1
    .assert_eq  s19, #0, #5
    s.addi      s21, s0, #3
    s.add       s22, s0, s0
loopa:
    s.addi      s22, s22, #1
    s.addi      s21, s21, #-1
    br.nz       s21, loopa               // condition forwarded from X
    s.subi      s22, s22, #3
    .assert_eq  s22, #0, #6

// ---------------------------------------------------------------- B
    v.iota      v1                       // i in lane i
    v.splat     v2, s20                  // r + 1
    v.movi      v3, #7
    v.movi      v4, #9
    v.cmp.lt    p1, v1, v2               // lanes 0..r
    @p1  v.mov  v3, v1                   // lane mask forwarded from X
    @!p1 v.mov  v4, v1                   // negated mask, p1 now in W
    p.and       p2, p1, p1               // A and B
    p.ballot    s23, p2                  // A forwarded from X
    .li         s24, #1
    s.shl       s24, s24, s20
    s.subi      s24, s24, #1             // (1 << (r + 1)) - 1
    s.sub       s25, s23, s24
    .assert_eq  s25, #0, #10
    .assert_lane v3, #0, #0, #11
    .assert_lane v3, #15, #7, #12
    .assert_lane v4, #0, #9, #13
    .assert_lane v4, #15, #15, #14
    v.cmp.lt    p3, v1, v2
    @p3  v.reduce.add s26, v1            // 0 + ... + r, mask forwarded from X
    @!p3 v.reduce.add s27, v1            // (r + 1) + ... + 15
    s.mul       s28, s1, s20
    s.lshri     s28, s28, #1             // r (r + 1) / 2
    s.sub       s29, s26, s28
    .assert_eq  s29, #0, #15
    s.add       s29, s26, s27
    .assert_eq  s29, #120, #16
    p.not       p4, p3
    @p4  v.reduce.add s30, v1            // the same lanes as @!p3
    s.sub       s30, s30, s27
    .assert_eq  s30, #0, #17

// ---------------------------------------------------------------- C
    v.add       v5, v1, v2               // i + r + 1
    v.add       v6, v5, v5               // 2 (i + r + 1)
    v.add       v7, v6, v5               // 3 (i + r + 1)
    v.add       v5, v7, v6               // WAW: 5 (i + r + 1)
    v.add       v8, v5, v5               // 10 (i + r + 1)
    s.muli      s31, s20, #10
    s.addi      s31, s31, #30            // lane 3: 10 (3 + r + 1)
    v.splat     v9, s31
    v.cmp.eq    p5, v8, v9
    p.ballot    s32, p5
    .assert_eq  s32, #0x0008, #20
    .li         s33, #fdata
    v.ld.w      v10, [s33, #0]           // 1.0f
    vf.add      v11, v10, v10            // 2.0, FP pipe
    vf.mul      v12, v11, v11            // 4.0, waits on the pipe
    vf.add      v13, v12, v11            // 6.0
    vf.add      v13, v13, v10            // 7.0, WAW behind the pipe
    .assert_lane v13, #0, #0x40e00000, #21
    .assert_lane v13, #15, #0x40e00000, #22

// ---------------------------------------------------------------- D
    .li         s34, #vbuf
    s.shli      s35, s1, #8              // 256 bytes per role
    s.add       s34, s34, s35
    v.ld.w      v14, [s34, #0]           // keep the load/store unit busy
    v.ld.w      v15, [s34, #64]
    v.ld.w      v16, [s34, #128]
    v.add       v17, v1, v2              // i + r + 1
    v.st.w      [s34, #192], v17         // waits for v17, then for the unit
    v.add       v18, v1, v1              // D prefetches while the store waits
    v.add       v19, v18, v17            // 3i + r + 1
    v.ld.w      v20, [s34, #192]         // what the store wrote
    v.sub       v21, v20, v17
    v.cmp.eq    p6, v21, v0              // v0: zero, never written
    p.ballot    s36, p6
    .assert_eq  s36, #0xffff, #30
    v.add       v22, v1, v1
    v.add       v22, v22, v1
    v.add       v22, v22, v2             // 3i + r + 1
    v.cmp.eq    p7, v19, v22
    p.ballot    s37, p7
    .assert_eq  s37, #0xffff, #31

// ---------------------------------------------------------------- E
    v.movi      v23, #0
    v.movi      v24, #1
    .li         s38, #5
loope:
    v.add       v23, v23, v24
    s.addi      s38, s38, #-1
    br.nz       s38, loope
    v.add       v23, v23, v24            // squashed while the loop runs
    v.add       v23, v23, v2             // 6 + r + 1
    s.addi      s39, s20, #6
    v.splat     v25, s39
    v.cmp.eq    p8, v23, v25
    p.ballot    s40, p8
    .assert_eq  s40, #0xffff, #40

// ---------------------------------------------------------------- H
    .li         s57, #sdata
    s.add       s57, s57, s15            // &sdata[role]
    .li         s58, #12
    s.add       s59, s0, s0
looph:
    s.ld.d      s5, [s57, #0]            // 1000 + role
    s.add       s59, s59, s5             // load-use: waits in R
    s.addi      s58, s58, #-1
    br.nz       s58, looph
    s.add       s60, s5, s5              // the fall-through also uses the load
    s.muli      s6, s16, #12
    s.sub       s6, s59, s6
    .assert_eq  s6, #0, #45
    s.sub       s60, s60, s17
    s.subi      s60, s60, #2000
    s.sub       s60, s60, s8
    .assert_eq  s60, #0, #46

// ---------------------------------------------------------------- G
// Roles 0/1 are a pair on barriers b0 (producer -> consumer) and b1 (back);
// roles 2/3 on b2 and b3. Six rounds.
    .li         s41, #pbuf
    s.andi      s42, s1, #2
    s.shli      s43, s42, #2             // 8 bytes per pair, at pbuf + 8*(r & 2)
    s.add       s41, s41, s43
    .li         s44, #1                  // round k
    .li         s45, #6
    s.andi      s46, s1, #1
    s.rdspecial s9, num_roles
    s.sub       s9, s9, s1
    s.subi      s9, s9, #1               // roles launched after this one
    s.or        s10, s9, s46             // 0: an even role without a partner
    br.z        s10, done0
    br.nz       s42, pair1
    br.nz       s46, cons0
prod0:
    mb.arm      s47, b0, #1, #0          // generation k
    mb.test     s48, s47                 // the arm's token, forwarded from X
    .assert_eq  s48, #0, #50
    s.ld.d      s49, [s41, #0]           // a little delay before publishing
    s.ld.d      s49, [s41, #0]
    s.muli      s50, s44, #1000
    s.add       s50, s50, s1
    s.st.d      [s41, #0], s50
    mb.arrive   b0
    s.shli      s51, s44, #16
    s.ori       s51, s51, #1             // {k, b1}, built right before the wait
    mb.wait     s51
    s.addi      s44, s44, #1
    s.addi      s45, s45, #-1
    br.nz       s45, prod0
    br          done
cons0:
    s.shli      s51, s44, #16
    s.ori       s51, s51, #0             // {k, b0}
    mb.wait     s51
    s.ld.d      s52, [s41, #0]
    s.muli      s53, s44, #1000
    s.add       s53, s53, s1
    s.subi      s53, s53, #1             // the producer is role - 1
    s.sub       s54, s52, s53
    .assert_eq  s54, #0, #51
    mb.arm      s55, b1, #1, #0
    mb.arrive   b1
    s.addi      s44, s44, #1
    s.addi      s45, s45, #-1
    br.nz       s45, cons0
    br          done
pair1:
    br.nz       s46, cons1
prod1:
    mb.arm      s47, b2, #1, #0
    mb.test     s48, s47
    .assert_eq  s48, #0, #52
    s.ld.d      s49, [s41, #0]
    s.ld.d      s49, [s41, #0]
    s.muli      s50, s44, #1000
    s.add       s50, s50, s1
    s.st.d      [s41, #0], s50
    mb.arrive   b2
    s.shli      s51, s44, #16
    s.ori       s51, s51, #3
    mb.wait     s51
    s.addi      s44, s44, #1
    s.addi      s45, s45, #-1
    br.nz       s45, prod1
    br          done
cons1:
    s.shli      s51, s44, #16
    s.ori       s51, s51, #2
    mb.wait     s51
    s.ld.d      s52, [s41, #0]
    s.muli      s53, s44, #1000
    s.add       s53, s53, s1
    s.subi      s53, s53, #1
    s.sub       s54, s52, s53
    .assert_eq  s54, #0, #53
    mb.arm      s55, b3, #1, #0
    mb.arrive   b3
    s.addi      s44, s44, #1
    s.addi      s45, s45, #-1
    br.nz       s45, cons1
done:
    .assert_eq  s45, #0, #54
done0:
    s.rdspecial s56, role_id
    s.sub       s56, s56, s1
    .assert_eq  s56, #0, #55
    exit

.org 0x4000
sdata:
    .dword 1000, 1001, 1002, 1003
.org 0x4100
fdata:
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
.org 0x4200
pbuf:                                    // one dword per pair
    .dword 0, 0, 0, 0
.org 0x4400
vbuf:                                    // 256 bytes per role, up to four
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .dword 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
