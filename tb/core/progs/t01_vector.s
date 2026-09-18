// t01_vector -- vector integer, predicate, shuffle and reduction
// (ISA sec 8.1, 8.5, 8.6, 8.7)
.org 0
start:
    // --- iota and splat -----------------------------------------------------
    v.iota   v1
    .assert_lane v1, #0,  #0,  #1
    .assert_lane v1, #15, #15, #2

    .li      s1, #7
    v.splat  v2, s1
    .assert_lane v2, #0,  #7, #3
    .assert_lane v2, #11, #7, #4

    // --- arithmetic ---------------------------------------------------------
    v.add    v3, v1, v2              // lane i = i + 7
    .assert_lane v3, #0, #7,  #10
    .assert_lane v3, #9, #16, #11

    v.sub    v3, v1, v2              // lane i = i - 7
    .assert_lane v3, #0, #0xfffffff9, #12
    .assert_lane v3, #7, #0,          #13

    v.mul    v3, v1, v1
    .assert_lane v3, #12, #144, #14

    v.muli   v3, v1, #-3
    .assert_lane v3, #4, #0xfffffff4, #15

    // 32-bit high multiply
    v.movi   v4, #-1                 // 0xffffffff in every lane
    v.mulh.u v5, v4, v4
    .assert_lane v5, #0, #0xfffffffe, #16
    v.mulh.s v5, v4, v4
    .assert_lane v5, #0, #0, #17

    // --- shifts and logic ---------------------------------------------------
    v.shli   v6, v1, #28
    .assert_lane v6, #1, #0x10000000, #20
    .assert_lane v6, #8, #0x80000000, #21
    v.ashri  v7, v6, #28
    .assert_lane v7, #8, #0xfffffff8, #22
    v.lshri  v7, v6, #28
    .assert_lane v7, #8, #8, #23

    v.andi   v8, v1, #3
    .assert_lane v8, #13, #1, #24
    v.xor    v8, v1, v1
    .assert_lane v8, #5, #0, #25

    // --- min / max ----------------------------------------------------------
    v.movi   v9, #8
    v.min.s  v10, v1, v9
    .assert_lane v10, #3,  #3, #26
    .assert_lane v10, #12, #8, #27
    v.max.s  v10, v1, v9
    .assert_lane v10, #3,  #8,  #28
    .assert_lane v10, #12, #12, #29

    // --- clz / popc ---------------------------------------------------------
    v.movi   v11, #1
    v.clz    v12, v11
    .assert_lane v12, #0, #31, #30
    v.movi   v11, #-1
    v.popc   v12, v11
    .assert_lane v12, #0, #32, #31

    // --- compare produces a predicate ---------------------------------------
    v.movi   v13, #8
    v.cmp.lt p1, v1, v13             // lanes 0..7 true
    p.ballot s2, p1
    .assert_eq s2, #0x00ff, #40
    p.any    s2, p1
    .assert_eq s2, #1, #41
    p.all    s2, p1
    .assert_eq s2, #0, #42
    p.popc   s2, p1
    .assert_eq s2, #8, #43

    v.cmpi.ge p2, v1, #16            // no lane true
    p.any    s2, p2
    .assert_eq s2, #0, #44

    // --- predicate logic ----------------------------------------------------
    p.not    p3, p1
    p.ballot s2, p3
    .assert_eq s2, #0xff00, #45
    p.and    p4, p1, p3
    p.ballot s2, p4
    .assert_eq s2, #0, #46
    p.or     p4, p1, p3
    p.ballot s2, p4
    .assert_eq s2, #0xffff, #47

    .li      s3, #0x0f0f
    p.frombits p5, s3
    p.ballot s2, p5
    .assert_eq s2, #0x0f0f, #48

    // --- masked write preserves inactive lanes (ISA sec 2.2) ----------------
    v.movi   v14, #99
    @p1 v.mov v14, v1                // lanes 0..7 take i, 8..15 keep 99
    .assert_lane v14, #3,  #3,  #50
    .assert_lane v14, #11, #99, #51

    // --- inverted mask ------------------------------------------------------
    v.movi   v15, #55
    @!p1 v.mov v15, v1
    .assert_lane v15, #3,  #55, #52
    .assert_lane v15, #11, #11, #53

    // --- v.sel writes every lane regardless of the mask ---------------------
    v.movi   v16, #1
    v.movi   v17, #2
    @p1 v.sel v18, v16, v17
    .assert_lane v18, #3,  #1, #54
    .assert_lane v18, #11, #2, #55

    // --- a masked compare zeroes the inactive lanes -------------------------
    @p1 v.cmp.ge p6, v1, v13
    p.ballot s2, p6
    .assert_eq s2, #0, #56

    // --- shuffles -----------------------------------------------------------
    .li      s4, #1
    v.shuffle.xor v19, v1, s4        // swap neighbours
    .assert_lane v19, #0, #1, #60
    .assert_lane v19, #1, #0, #61
    .assert_lane v19, #14, #15, #62

    .li      s4, #2
    v.shuffle.up v20, v1, s4
    .assert_lane v20, #0, #0, #63
    .assert_lane v20, #5, #3, #64
    v.shuffle.down v20, v1, s4
    .assert_lane v20, #5,  #7,  #65
    .assert_lane v20, #15, #15, #66
    v.rotate v20, v1, s4
    .assert_lane v20, #15, #1, #67

    v.movi   v21, #3
    v.permute v22, v1, v21
    .assert_lane v22, #9, #3, #68

    .li      s5, #42
    v.insert v23, s5, #6
    .assert_lane v23, #6, #42, #69

    // --- reductions ---------------------------------------------------------
    v.reduce.add   s6, v1
    .assert_eq s6, #120, #70             // 0+1+...+15
    v.reduce.max.s s6, v1
    .assert_eq s6, #15, #71
    v.reduce.min.s s6, v1
    .assert_eq s6, #0, #72
    v.reduce.or    s6, v1
    .assert_eq s6, #15, #73
    v.reduce.and   s6, v1
    .assert_eq s6, #0, #74
    v.reduce.xor   s6, v1
    .assert_eq s6, #0, #75               // 0^1^...^15 == 0

    // broadcast form
    v.reduce.add.b v24, v1
    .assert_lane v24, #0,  #120, #76
    .assert_lane v24, #15, #120, #77

    // masked reduction: inactive lanes contribute the identity
    @p1 v.reduce.add s6, v1
    .assert_eq s6, #28, #78              // 0+1+...+7
    @p1 v.reduce.min.s s6, v1
    .assert_eq s6, #0, #79
    @p1 v.reduce.max.s s6, v1
    .assert_eq s6, #7, #80

    exit
