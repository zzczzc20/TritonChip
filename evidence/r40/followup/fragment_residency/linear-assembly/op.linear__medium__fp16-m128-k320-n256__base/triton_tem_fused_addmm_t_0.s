    s.rdspecial  s1, role_id
    s.rdspecial  s2, pid.x
    s.rdspecial  s3, npid.x
    s.add        s4, s2, s1
    s.cmp.ge     s1, s4, s3
    br.nz        s1, __rctx_next
__rctx_program:
    s.rdspecial  s1, arg_base
    s.rdspecial  s2, arg_base
    s.ld.d       s3, [s1, #32]                   // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s1, role_id
    s.movz       s5, #512
    .li       s6, #2147483647
    s.mul        s7, s1, s5
    .li       s1, #2147483647
    s.movz       s5, #640
    s.movz       s8, #2
    .li       s9, #68719476752
    .li       s10, #8589934592
    s.rdspecial  s11, arg_base
    s.ld.d       s12, [s2, #8]                   // arg_A: device_ptr64 at arg_base+8
    s.add        s13, s3, s7                     // this context's own scratch
    s.rdspecial  s3, role_id
    s.addi       s7, s13, #0                     // TensorMap 0: a 2-D tt.load (built once, at entry)
    s.movz       s13, #512
    .li       s14, #2147483647
    s.mul        s15, s3, s13
    .li       s3, #2147483647
    s.movz       s13, #640
    s.movz       s16, #2
    .li       s17, #68719476752
    s.ld.d       s18, [s2, #0]                   // in_ptr0: device_ptr64 at arg_base+0
    .li       s19, #8589934592
    s.movz       s20, #0
    s.movz       s20, #0
    s.ld.d       s20, [s2, #16]                  // arg_B: device_ptr64 at arg_base+16
    s.movz       s21, #16
    s.ld.d       s22, [s2, #24]                  // out_ptr0: device_ptr64 at arg_base+24
    s.movz       s2, #0
    s.st.d       [s7, #0], s12
    s.movz       s12, #0
    s.st.d       [s7, #8], s6
    s.movz       s6, #0
    s.cmp.lt     s23, s21, s6
    s.st.d       [s7, #16], s1
    s.movz       s1, #0
    s.sub        s6, s1, s23
    s.movz       s1, #8
    s.st.d       [s7, #48], s5
    s.movz       s5, #1
    s.st.d       [s7, #56], s8
    s.cmp.lt     s8, s4, s2
    s.movz       s2, #0
    s.sub        s23, s12, s8
    s.movz       s8, #0
    s.st.d       [s7, #88], s9
    s.xor        s9, s4, s23
    s.sub        s12, s9, s23
    s.st.d       [s7, #104], s10
    s.lshri      s9, s12, #4
    fence.release.device
    td.load      d0, [s7]                        // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.xor        s7, s23, s6
    s.movz       s10, #0
    s.ld.d       s23, [s11, #32]                 // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.add        s11, s23, s15                   // this context's own scratch
    s.xor        s15, s9, s7
    s.cmp.lt     s23, s4, s8
    s.sub        s8, s15, s7
    s.movz       s7, #0
    s.sub        s15, s1, s8
    s.muli       s1, s9, #16
    s.min.s      s9, s15, s5
    s.addi       s15, s11, #256                  // TensorMap 1: a tile contiguous down its columns (built once, at entry)
    s.cmp.lt     s11, s9, s2
    s.sub        s2, s7, s23
    s.movz       s7, #0
    s.st.d       [s15, #0], s20
    s.sub        s20, s10, s11
    s.movz       s10, #1
    s.xor        s11, s4, s2
    s.sub        s23, s12, s1
    s.xor        s1, s21, s6
    s.movz       s12, #0
    s.sub        s23, s1, s6
    s.st.d       [s15, #8], s14
    s.xor        s1, s9, s20
    s.sub        s6, s11, s2
    s.st.d       [s15, #16], s3
    s.sub        s3, s1, s20
    s.addi       s1, s7, #0
    s.st.d       [s15, #48], s13
    s.cmp.eq     s7, s3, s10
    s.addi       s10, s12, #0
    s.st.d       [s15, #56], s16
    s.st.d       [s15, #88], s17
    s.st.d       [s15, #104], s19
    fence.release.device
    td.load      d1, [s15]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.mov        s1, s6
    br.nz        s7, sdiv_done_1
    s.movz       s7, #0
    s.movz       s11, #31
    s.mov        s1, s7
    s.addi       s7, s11, #0
sdiv_2:
    s.lshr       s11, s6, s7
    s.shli       s12, s10, #1
    s.andi       s13, s11, #1
    s.movz       s11, #0
    s.or         s14, s12, s13
    s.movz       s12, #1
    s.cmp.geu    s13, s14, s3
    s.shl        s15, s12, s7
    s.sub        s12, s11, s13
    s.subi       s11, s7, #1
    s.and        s13, s3, s12
    s.and        s16, s15, s12
    s.sub        s12, s14, s13
    s.or         s13, s1, s16
    s.mov        s10, s12
    s.mov        s1, s13
    s.mov        s7, s11
    s.movz       s11, #0
    s.cmp.ge     s12, s7, s11
    br.nz        s12, sdiv_2
sdiv_done_1:
    s.movz       s1, #0
    s.movz       s3, #0
    s.cmp.lt     s6, s4, s1
    s.movz       s1, #0
    s.sub        s7, s3, s6
    s.movz       s3, #0
    s.xor        s6, s4, s7
    s.movz       s4, #0
    s.sub        s11, s6, s7
    s.cmp.lt     s6, s9, s1
    s.lshri      s1, s11, #4
    s.movz       s12, #0
    s.muli       s13, s1, #16
    s.movz       s1, #0
    s.sub        s14, s4, s6
    s.movz       s4, #0
    s.cmp.lt     s6, s21, s12
    s.sub        s12, s11, s13
    s.xor        s11, s9, s14
    s.xor        s9, s12, s7
    s.xor        s12, s10, s2
    s.sub        s10, s9, s7
    s.sub        s7, s4, s6
    s.cmp.lt     s4, s10, s3
    s.sub        s3, s11, s14
    s.sub        s6, s1, s4
    s.movz       s1, #0
    s.xor        s4, s10, s6
    s.movz       s9, #1
    s.sub        s10, s12, s2
    s.xor        s2, s21, s7
    s.sub        s11, s4, s6
    s.movz       s4, #0
    s.addi       s12, s1, #0
    s.cmp.eq     s1, s3, s9
    s.add        s9, s8, s10
    s.sub        s8, s2, s7
    s.addi       s2, s4, #0
    s.mov        s12, s11
    br.nz        s1, sdiv_done_3
    s.movz       s1, #0
    s.movz       s4, #31
    s.mov        s12, s1
    s.addi       s1, s4, #0
sdiv_4:
    s.lshr       s4, s11, s1
    s.shli       s7, s2, #1
    s.andi       s8, s4, #1
    s.movz       s4, #0
    s.or         s10, s7, s8
    s.movz       s7, #1
    s.cmp.geu    s8, s10, s3
    s.shl        s13, s7, s1
    s.sub        s7, s4, s8
    s.subi       s4, s1, #1
    s.and        s8, s3, s7
    s.and        s15, s13, s7
    s.sub        s7, s10, s8
    s.or         s8, s12, s15
    s.mov        s2, s7
    s.mov        s12, s8
    s.mov        s1, s4
    s.movz       s4, #0
    s.cmp.ge     s7, s1, s4
    br.nz        s7, sdiv_4
sdiv_done_3:
    s.xor        s1, s6, s14
    s.mul        s2, s9, s21
    s.xor        s3, s12, s1
    s.movz       s4, #0
    s.sub        s6, s3, s1
    v.splat      v0, s9
    s.mul        s1, s6, s21
    s.andi       s3, s2, #127
    v.splat      v1, s4
    v.splat      v2, s6
    v.splat      v3, s4
    s.andi       s6, s1, #255
    v.iota       v4                              // arange(0, VLEN)
    v.splat      v5, s2
    v.cmp.ge     p1, v0, v1
    v.splat      v0, s1
    v.splat      v1, s3
    v.cmp.ge     p1, v2, v3
    v.iota       v2                              // arange(0, VLEN)
    v.splat      v3, s6
    v.add        v6, v5, v4
    v.iota       v5                              // arange(0, VLEN)
    v.add        v6, v0, v4
    v.add        v0, v1, v2
    v.add        v0, v3, v5
    s.movz       s7, #20
    s.addi       s8, s4, #0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a0
for_5:
    s.cmp.ge     s4, s8, s7
    br.nz        s4, endfor_6
    s.mul        s4, s8, s21
    w.def        w0, #56, #8                     // v56..v63
    mb.arm       s9, b0, #0, #1                  // 0 arrival(s), 1 operation(s)
    s.add        s10, s8, s5
    s.movz       s11, #1
    // a 2-D tt.load: one tile into a window (ISA sec 9E)
    s.mov        s12, s3                         // coordinates must be consecutive (ISA sec 9E)
    s.mov        s13, s4                         // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w0, d0, s12, s9
    mb.wait      s9
    w.def        w1, #48, #8                     // v48..v55
    mb.arm       s12, b1, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a tile contiguous down its columns: its transpose is a plain tile, written transposed by the engine (ISA sec 9E mod[5])
    s.mov        s13, s6                         // coordinates must be consecutive (ISA sec 9E)
    s.mov        s14, s4                         // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h.t w1, d1, s13, s12
    mb.wait      s12
    mb.arm       s4, b2, #0, #1                  // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.async.k16.f16 a0, v56, v48, s4
    mb.wait      s4
    s.mov        s8, s10
    br.nz        s11, for_5
endfor_6:
    s.movz       s3, #0
    s.movz       s5, #0
    s.add        s6, s3, s1                      // the tile origin
    v.iota       v0                              // arange(0, VLEN)
    v.splat      v1, s6
    s.add        s3, s5, s2                      // the tile origin
    s.movz       s5, #128
    s.movz       s6, #256
    s.muli       s7, s1, #2                      // a runtime tile origin, x2
    v.splat      v2, s3
    v.splat      v3, s5
    v.add        v4, v0, v1
    v.splat      v0, s6
    v.cmp.lt     p1, v2, v3
    s.add        s5, s18, s7                     // the tile origin
    v.cmp.lt     p2, v4, v0
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.movz       s6, #0
    p.and        p3, p1, p2
    a.vread      v0, a0, s6
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s5, #0]
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.muli       s6, s1, #2                      // a runtime tile origin, x2
    s.addi       s1, s3, #1
    s.muli       s7, s2, #512                    // a runtime tile origin, x512
    v.splat      v2, s3
    v.splat      v4, s1
    s.add        s1, s22, s6                     // the tile origin
    v.cmp.lt     p1, v2, v3
    v.cmp.lt     p3, v4, v3
    s.add        s2, s1, s7                      // the tile origin
    p.and        p4, p1, p2
    s.movz       s1, #1
    p.and        p1, p3, p2
    s.addi       s6, s3, #1
    vf.cvt.f16.f32 v2, v1
    s.addi       s7, s3, #2
    v.splat      v1, s6
    v.splat      v4, s7
    s.addi       s6, s2, #512                    // row 1
    vf.add       v5, v0, v2
    s.movz       s7, #2
    s.addi       s8, s3, #2
    s.addi       s10, s3, #3
    v.splat      v0, s8
    vf.cvt.f32.f16 v2, v5
    v.splat      v5, s10
    s.addi       s8, s2, #1024                   // row 2
    s.movz       s10, #3
    s.addi       s11, s3, #3
    @p4 v.st.h       [s2, #0], v2
    a.vread      v2, a0, s1
    v.cmp.lt     p3, v1, v3
    p.and        p4, p3, p2
    s.addi       s1, s3, #4
    v.splat      v1, s11
    v.splat      v6, s1
    s.addi       s1, s2, #1536                   // row 3
    s.movz       s11, #4
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p1 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v2, v4
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s6, #0], v2
    a.vread      v2, a0, s7
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s6, s3, #4
    s.addi       s7, s3, #5
    v.splat      v0, s6
    v.splat      v4, s7
    s.addi       s6, s2, #2048                   // row 4
    s.movz       s7, #5
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v2, v5
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s8, #0], v2
    a.vread      v2, a0, s10
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s8, s3, #5
    s.addi       s10, s3, #6
    v.splat      v1, s8
    v.splat      v5, s10
    s.addi       s8, s2, #2560                   // row 5
    s.movz       s10, #6
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v2, v6
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s1, #0], v2
    a.vread      v2, a0, s11
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s1, s3, #6
    s.addi       s11, s3, #7
    v.splat      v0, s1
    v.splat      v6, s11
    s.addi       s1, s2, #3072                   // row 6
    s.movz       s11, #7
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v2, v4
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s6, #0], v2
    a.vread      v2, a0, s7
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s6, s3, #7
    s.addi       s7, s3, #8
    v.splat      v1, s6
    v.splat      v4, s7
    s.addi       s6, s2, #3584                   // row 7
    s.movz       s7, #8
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v2, v5
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s8, #0], v2
    a.vread      v2, a0, s10
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s8, s3, #8
    s.addi       s10, s3, #9
    v.splat      v0, s8
    v.splat      v5, s10
    s.addi       s8, s2, #4096                   // row 8
    s.movz       s10, #9
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v2, v6
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s1, #0], v2
    a.vread      v2, a0, s11
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s1, s3, #9
    s.addi       s11, s3, #10
    v.splat      v1, s1
    v.splat      v6, s11
    s.addi       s1, s2, #4608                   // row 9
    s.movz       s11, #10
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v2, v4
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s6, #0], v2
    a.vread      v2, a0, s7
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s6, s3, #10
    s.addi       s7, s3, #11
    v.splat      v0, s6
    v.splat      v4, s7
    s.addi       s6, s2, #5120                   // row 10
    s.movz       s7, #11
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v2, v5
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s8, #0], v2
    a.vread      v2, a0, s10
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s8, s3, #11
    s.addi       s10, s3, #12
    v.splat      v1, s8
    v.splat      v5, s10
    s.addi       s8, s2, #5632                   // row 11
    s.movz       s10, #12
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v2, v6
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s1, #0], v2
    a.vread      v2, a0, s11
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s1, s3, #12
    s.addi       s11, s3, #13
    v.splat      v0, s1
    v.splat      v6, s11
    s.addi       s1, s2, #6144                   // row 12
    s.movz       s11, #13
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v2, v4
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s6, #0], v2
    a.vread      v2, a0, s7
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s6, s3, #13
    s.addi       s7, s3, #14
    v.splat      v1, s6
    v.splat      v4, s7
    s.addi       s6, s2, #6656                   // row 13
    s.movz       s7, #14
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v2, v5
    vf.cvt.f32.f16 v2, v7
    @p4 v.st.h       [s8, #0], v2
    a.vread      v2, a0, s10
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s8, s3, #14
    s.addi       s10, s3, #15
    v.splat      v0, s8
    v.splat      v5, s10
    s.addi       s8, s2, #7168                   // row 14
    s.movz       s10, #15
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s5, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v2, v6
    vf.cvt.f32.f16 v2, v7
    s.addi       s13, s3, #15
    s.addi       s3, s2, #7680                   // row 15
    v.splat      v6, s13
    @p4 v.st.h       [s1, #0], v2
    a.vread      v2, a0, s11
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s5, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v1
    vf.add       v1, v2, v4
    vf.cvt.f32.f16 v2, v1
    @p4 v.st.h       [s6, #0], v2
    a.vread      v1, a0, s7
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v0, [s5, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v0
    vf.add       v0, v1, v2
    vf.cvt.f32.f16 v1, v0
    @p4 v.st.h       [s8, #0], v1
    a.vread      v0, a0, s10
    v.cmp.lt     p1, v6, v3
    p.and        p4, p1, p2
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s5, #0]
    vf.cvt.f16.f32 v2, v1
    vf.add       v1, v0, v2
    vf.cvt.f32.f16 v0, v1
    @p4 v.st.h       [s3, #0], v0
__rctx_next:
    exit
