    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    s.ld.d       s3, [s1, #8]                    // arg_A: device_ptr64 at arg_base+8
    s.ld.d       s4, [s1, #16]                   // arg_B: device_ptr64 at arg_base+16
    s.ld.d       s5, [s1, #24]                   // out_ptr0: device_ptr64 at arg_base+24
    s.rdspecial  s1, arg_base
    s.ld.d       s6, [s1, #32]                   // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s1, role_id
    s.movz       s7, #512
    s.mul        s8, s1, s7
    s.add        s1, s6, s8                      // this context's own scratch
    s.addi       s6, s1, #0                      // TensorMap 0: a 2-D tt.load (built once, at entry)
    s.movz       s1, #0
    s.st.d       [s6, #0], s3
    .li       s3, #2147483647
    s.st.d       [s6, #8], s3
    s.st.d       [s6, #16], s3
    s.movz       s7, #128
    s.st.d       [s6, #48], s7
    s.movz       s8, #2
    s.st.d       [s6, #56], s8
    .li       s9, #68719476752
    s.st.d       [s6, #88], s9
    .li       s10, #8589934592
    s.st.d       [s6, #104], s10
    fence.release.device
    td.load      d0, [s6]                        // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.rdspecial  s6, arg_base
    s.ld.d       s11, [s6, #32]                  // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s6, role_id
    s.movz       s12, #512
    s.mul        s13, s6, s12
    s.add        s6, s11, s13                    // this context's own scratch
    s.addi       s11, s6, #256                   // TensorMap 1: a tile contiguous down its columns (built once, at entry)
    s.st.d       [s11, #0], s4
    s.st.d       [s11, #8], s3
    s.st.d       [s11, #16], s3
    s.st.d       [s11, #48], s7
    s.st.d       [s11, #56], s8
    s.st.d       [s11, #88], s9
    s.st.d       [s11, #104], s10
    fence.release.device
    td.load      d1, [s11]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s3, #4
    s.cmp.lt     s4, s3, s1
    s.sub        s6, s1, s4
    s.xor        s4, s3, s6
    s.sub        s7, s4, s6
    s.movz       s4, #1
    s.movz       s7, #0
    s.rdspecial  s8, role_id
    s.rdspecial  s9, npid.x
__rctx_program:
    s.cmp.lt     s10, s8, s1
    s.addi       s11, s7, #0
    s.sub        s12, s1, s10
    s.addi       s10, s7, #0
    s.xor        s13, s8, s12
    s.xor        s14, s12, s6
    s.sub        s15, s13, s12
    s.sub        s16, s13, s12
    s.lshri      s13, s15, #2
    s.mov        s11, s16
    s.xor        s17, s13, s14
    s.muli       s18, s13, #4
    s.sub        s13, s17, s14
    s.sub        s14, s4, s13
    s.min.s      s17, s14, s4
    s.sub        s14, s15, s18
    s.cmp.lt     s14, s17, s1
    s.sub        s15, s1, s14
    s.xor        s14, s17, s15
    s.sub        s18, s14, s15
    s.cmp.eq     s14, s18, s4
    br.nz        s14, sdiv_done_1
    s.movz       s14, #0
    s.movz       s15, #31
    s.mov        s11, s14
    s.addi       s14, s15, #0
sdiv_2:
    s.lshr       s15, s16, s14
    s.shli       s19, s10, #1
    s.andi       s20, s15, #1
    s.shl        s15, s4, s14
    s.or         s21, s19, s20
    s.subi       s19, s14, #1
    s.cmp.geu    s20, s21, s18
    s.sub        s22, s7, s20
    s.and        s20, s18, s22
    s.and        s23, s15, s22
    s.sub        s15, s21, s20
    s.or         s20, s11, s23
    s.mov        s10, s15
    s.mov        s11, s20
    s.mov        s14, s19
    s.cmp.ge     s15, s14, s7
    br.nz        s15, sdiv_2
sdiv_done_1:
    s.cmp.lt     s11, s8, s7
    s.cmp.lt     s14, s17, s7
    s.sub        s15, s7, s11
    s.sub        s11, s7, s14
    s.xor        s14, s8, s15
    s.cmp.lt     s16, s3, s7
    s.sub        s18, s14, s15
    s.xor        s14, s17, s11
    s.lshri      s17, s18, #2
    s.xor        s19, s10, s12
    s.muli       s10, s17, #4
    s.sub        s17, s7, s16
    s.sub        s16, s14, s11
    s.movz       s14, #0
    s.movz       s20, #1
    s.sub        s21, s18, s10
    s.sub        s10, s19, s12
    s.xor        s12, s21, s15
    s.xor        s18, s3, s17
    s.sub        s19, s12, s15
    s.addi       s12, s14, #0
    s.cmp.lt     s15, s19, s7
    s.cmp.eq     s21, s16, s20
    s.sub        s22, s7, s15
    s.add        s15, s13, s10
    s.xor        s10, s19, s22
    s.sub        s13, s18, s17
    s.sub        s13, s10, s22
    s.addi       s10, s14, #0
    s.mov        s12, s13
    br.nz        s21, sdiv_done_3
    s.movz       s17, #0
    s.movz       s18, #31
    s.mov        s12, s17
    s.addi       s17, s18, #0
sdiv_4:
    s.lshr       s18, s13, s17
    s.shli       s19, s10, #1
    s.andi       s21, s18, #1
    s.shl        s18, s20, s17
    s.or         s23, s19, s21
    s.subi       s19, s17, #1
    s.cmp.geu    s21, s23, s16
    s.sub        s24, s14, s21
    s.and        s21, s16, s24
    s.and        s25, s18, s24
    s.sub        s18, s23, s21
    s.or         s21, s12, s25
    s.mov        s10, s18
    s.mov        s12, s21
    s.mov        s17, s19
    s.cmp.ge     s18, s17, s14
    br.nz        s18, sdiv_4
sdiv_done_3:
    s.xor        s10, s22, s11
    s.movz       s11, #16
    s.xor        s13, s12, s10
    s.mul        s12, s15, s11
    s.sub        s16, s13, s10
    v.splat      v0, s15
    s.mul        s10, s16, s11
    v.splat      v1, s14
    s.andi       s13, s12, #15
    v.splat      v2, s16
    v.iota       v3                              // arange(0, VLEN)
    s.andi       s15, s10, #63
    v.splat      v4, s12
    v.splat      v5, s10
    v.cmp.ge     p1, v0, v1
    v.splat      v0, s13
    v.cmp.ge     p1, v2, v1
    v.splat      v1, s15
    v.iota       v2                              // arange(0, VLEN)
    v.add        v6, v4, v3
    v.add        v4, v5, v3
    v.add        v4, v0, v3
    v.add        v0, v1, v2
    s.addi       s16, s14, #0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a0
for_5:
    s.cmp.ge     s14, s16, s3
    br.nz        s14, endfor_6
    s.mul        s14, s16, s11
    w.def        w0, #56, #8                     // v56..v63
    mb.arm       s17, b0, #0, #1                 // 0 arrival(s), 1 operation(s)
    s.add        s18, s16, s4
    s.movz       s19, #1
    // a 2-D tt.load: one tile into a window (ISA sec 9E)
    s.mov        s20, s13                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s21, s14                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w0, d0, s20, s17
    mb.wait      s17
    w.def        w1, #48, #8                     // v48..v55
    mb.arm       s20, b1, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a tile contiguous down its columns: its transpose is a plain tile, written transposed by the engine (ISA sec 9E mod[5])
    s.mov        s21, s15                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s22, s14                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h.t w1, d1, s21, s20
    mb.wait      s20
    mb.arm       s14, b2, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.dp.async.k16.f16 a0, v56, v48, s14
    mb.wait      s14
    s.mov        s16, s18
    br.nz        s19, for_5
endfor_6:
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.movz       s13, #0
    v.iota       v0                              // arange(0, VLEN)
    s.add        s15, s13, s10                   // the tile origin
    s.add        s16, s13, s12                   // the tile origin
    v.splat      v1, s15
    s.movz       s15, #64
    s.muli       s18, s10, #2                    // a runtime tile origin, x2
    v.splat      v2, s16
    v.splat      v3, s11
    v.add        v4, v0, v1
    v.splat      v0, s15
    v.cmp.lt     p1, v2, v3
    s.add        s10, s2, s18                    // the tile origin
    v.cmp.lt     p2, v4, v0
    p.and        p3, p1, p2
    a.vread      v0, a0, s13
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s10, #0]
    s.addi       s11, s16, #1
    s.muli       s13, s12, #128                  // a runtime tile origin, x128
    v.splat      v2, s16
    v.splat      v4, s11
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.add        s12, s5, s18                    // the tile origin
    v.cmp.lt     p1, v2, v3
    v.cmp.lt     p3, v4, v3
    s.add        s15, s12, s13                   // the tile origin
    p.and        p4, p1, p2
    s.movz       s12, #1
    p.and        p1, p3, p2
    s.addi       s13, s16, #2
    v.splat      v2, s11
    vf.cvt.f16.f32 v4, v1
    v.splat      v1, s13
    s.addi       s11, s15, #128                  // row 1
    s.movz       s18, #2
    s.addi       s19, s16, #3
    vf.add       v5, v0, v4
    v.splat      v0, s13
    v.splat      v4, s19
    s.addi       s13, s15, #256                  // row 2
    s.movz       s21, #3
    vf.cvt.f32.f16 v6, v5
    v.splat      v5, s19
    s.addi       s19, s15, #384                  // row 3
    s.movz       s22, #4
    @p4 v.st.h       [s15, #0], v6
    a.vread      v6, a0, s12
    v.cmp.lt     p3, v2, v3
    p.and        p4, p3, p2
    s.addi       s12, s16, #4
    v.splat      v2, s12
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p1 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v1, v7
    vf.add       v7, v6, v1
    v.splat      v1, s12
    s.addi       s12, s15, #512                  // row 4
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s11, #0], v6
    a.vread      v6, a0, s18
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s11, s16, #5
    s.movz       s18, #5
    v.splat      v0, s11
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v6, v4
    v.splat      v4, s11
    s.addi       s11, s15, #640                  // row 5
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s13, #0], v6
    a.vread      v6, a0, s21
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    s.addi       s13, s16, #6
    s.movz       s21, #6
    v.splat      v5, s13
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v7
    vf.add       v7, v6, v2
    v.splat      v2, s13
    s.addi       s13, s15, #768                  // row 6
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s19, #0], v6
    a.vread      v6, a0, s22
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s19, s16, #7
    s.movz       s22, #7
    v.splat      v1, s19
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v0, v7
    vf.add       v7, v6, v0
    v.splat      v0, s19
    s.addi       s19, s15, #896                  // row 7
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s12, #0], v6
    a.vread      v6, a0, s18
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s12, s16, #8
    s.movz       s18, #8
    v.splat      v4, s12
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v6, v5
    v.splat      v5, s12
    s.addi       s12, s15, #1024                 // row 8
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s11, #0], v6
    a.vread      v6, a0, s21
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s11, s16, #9
    s.movz       s21, #9
    v.splat      v2, s11
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v1, v7
    vf.add       v7, v6, v1
    v.splat      v1, s11
    s.addi       s11, s15, #1152                 // row 9
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s13, #0], v6
    a.vread      v6, a0, s22
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s13, s16, #10
    s.movz       s22, #10
    v.splat      v0, s13
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v6, v4
    v.splat      v4, s13
    s.addi       s13, s15, #1280                 // row 10
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s19, #0], v6
    a.vread      v6, a0, s18
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    s.addi       s18, s16, #11
    s.movz       s19, #11
    v.splat      v5, s18
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v7
    vf.add       v7, v6, v2
    v.splat      v2, s18
    s.addi       s18, s15, #1408                 // row 11
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s12, #0], v6
    a.vread      v6, a0, s21
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s12, s16, #12
    s.movz       s21, #12
    v.splat      v1, s12
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v0, v7
    vf.add       v7, v6, v0
    v.splat      v0, s12
    s.addi       s12, s15, #1536                 // row 12
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s11, #0], v6
    a.vread      v6, a0, s22
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s11, s16, #13
    s.movz       s22, #13
    v.splat      v4, s11
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v6, v5
    v.splat      v5, s11
    s.addi       s11, s15, #1664                 // row 13
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s13, #0], v6
    a.vread      v6, a0, s19
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s13, s16, #14
    s.movz       s19, #14
    v.splat      v2, s13
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v1, v7
    vf.add       v7, v6, v1
    v.splat      v1, s13
    s.addi       s13, s15, #1792                 // row 14
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s18, #0], v6
    a.vread      v6, a0, s21
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s18, s16, #15
    s.movz       s16, #15
    v.splat      v0, s18
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    s.addi       s21, s15, #1920                 // row 15
    vf.add       v7, v6, v4
    v.splat      v4, s18
    vf.cvt.f32.f16 v6, v7
    @p4 v.st.h       [s12, #0], v6
    a.vread      v6, a0, s22
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    v.movi       v5, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v5, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v5
    vf.add       v5, v6, v2
    vf.cvt.f32.f16 v2, v5
    @p4 v.st.h       [s11, #0], v2
    a.vread      v2, a0, s19
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s10, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v0, v1
    vf.add       v1, v2, v0
    vf.cvt.f32.f16 v0, v1
    @p4 v.st.h       [s13, #0], v0
    a.vread      v0, a0, s16
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s10, #0]
    vf.cvt.f16.f32 v2, v1
    vf.add       v1, v0, v2
    vf.cvt.f32.f16 v0, v1
    @p4 v.st.h       [s21, #0], v0
__rctx_next:
    s.addi       s10, s8, #2                     // the next program of this context: + 2
    s.cmp.lt     s11, s10, s9
    s.mov        s8, s10
    br.nz        s11, __rctx_program
    exit
