    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]                    // arg_A: device_ptr64 at arg_base+0
    s.ld.d       s3, [s1, #8]                    // arg_B: device_ptr64 at arg_base+8
    s.ld.d       s4, [s1, #16]                   // out_ptr0: device_ptr64 at arg_base+16
    s.rdspecial  s1, arg_base
    s.ld.d       s5, [s1, #24]                   // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s1, role_id
    s.movz       s6, #1024
    s.mul        s7, s1, s6
    s.add        s1, s5, s7                      // this context's own scratch
    s.addi       s5, s1, #0                      // TensorMap 0: a tile contiguous down its columns (built once, at entry)
    s.movz       s1, #0
    s.st.d       [s5, #0], s3
    .li       s3, #2147483647
    s.st.d       [s5, #8], s3
    s.st.d       [s5, #16], s3
    s.movz       s3, #128
    s.st.d       [s5, #48], s3
    s.movz       s3, #2
    s.st.d       [s5, #56], s3
    .li       s3, #68719476752
    s.st.d       [s5, #88], s3
    .li       s3, #8589934592
    s.st.d       [s5, #104], s3
    fence.release.device
    td.load      d0, [s5]                        // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s3, #4
    s.cmp.lt     s5, s3, s1
    s.sub        s6, s1, s5
    s.xor        s5, s3, s6
    s.sub        s7, s5, s6
    s.movz       s5, #1
    s.rdspecial  s7, role_id
    s.rdspecial  s8, npid.x
__rctx_program:
    s.cmp.lt     s9, s7, s1
    s.addi       s10, s1, #0
    s.sub        s11, s1, s9
    s.addi       s9, s1, #0
    s.xor        s12, s7, s11
    s.xor        s13, s11, s6
    s.sub        s14, s12, s11
    s.sub        s15, s12, s11
    s.lshri      s12, s14, #2
    s.mov        s10, s15
    s.xor        s16, s12, s13
    s.muli       s17, s12, #4
    s.sub        s12, s16, s13
    s.sub        s13, s5, s12
    s.min.s      s16, s13, s5
    s.sub        s13, s14, s17
    s.cmp.lt     s13, s16, s1
    s.sub        s14, s1, s13
    s.xor        s13, s16, s14
    s.sub        s17, s13, s14
    s.cmp.eq     s13, s17, s5
    br.nz        s13, sdiv_done_1
    s.movz       s13, #0
    s.movz       s14, #31
    s.mov        s10, s13
    s.addi       s13, s14, #0
sdiv_2:
    s.lshr       s14, s15, s13
    s.shli       s18, s9, #1
    s.andi       s19, s14, #1
    s.movz       s14, #0
    s.or         s20, s18, s19
    s.shl        s18, s5, s13
    s.cmp.geu    s19, s20, s17
    s.subi       s21, s13, #1
    s.sub        s22, s14, s19
    s.and        s19, s17, s22
    s.and        s23, s18, s22
    s.sub        s18, s20, s19
    s.or         s19, s10, s23
    s.mov        s9, s18
    s.mov        s10, s19
    s.mov        s13, s21
    s.cmp.ge     s18, s13, s14
    br.nz        s18, sdiv_2
sdiv_done_1:
    s.movz       s10, #0
    s.xor        s13, s9, s11
    s.cmp.lt     s9, s7, s10
    s.cmp.lt     s14, s16, s10
    s.sub        s15, s10, s9
    s.sub        s9, s10, s14
    s.xor        s14, s7, s15
    s.cmp.lt     s17, s3, s10
    s.sub        s18, s14, s15
    s.xor        s14, s16, s9
    s.lshri      s16, s18, #2
    s.sub        s19, s10, s17
    s.muli       s17, s16, #4
    s.sub        s16, s14, s9
    s.movz       s14, #1
    s.sub        s20, s13, s11
    s.xor        s11, s3, s19
    s.sub        s13, s18, s17
    s.addi       s17, s10, #0
    s.xor        s18, s13, s15
    s.cmp.eq     s13, s16, s14
    s.sub        s21, s18, s15
    s.add        s15, s12, s20
    s.cmp.lt     s12, s21, s10
    s.sub        s18, s11, s19
    s.sub        s11, s10, s12
    s.addi       s12, s10, #0
    s.xor        s18, s21, s11
    s.sub        s19, s18, s11
    s.mov        s17, s19
    br.nz        s13, sdiv_done_3
    s.movz       s13, #0
    s.movz       s18, #31
    s.mov        s17, s13
    s.addi       s13, s18, #0
sdiv_4:
    s.lshr       s18, s19, s13
    s.shli       s20, s12, #1
    s.andi       s21, s18, #1
    s.shl        s18, s14, s13
    s.or         s22, s20, s21
    s.subi       s20, s13, #1
    s.cmp.geu    s21, s22, s16
    s.sub        s23, s10, s21
    s.and        s21, s16, s23
    s.and        s24, s18, s23
    s.sub        s18, s22, s21
    s.or         s21, s17, s24
    s.mov        s12, s18
    s.mov        s17, s21
    s.mov        s13, s20
    s.cmp.ge     s18, s13, s10
    br.nz        s18, sdiv_4
sdiv_done_3:
    s.xor        s10, s11, s9
    s.movz       s9, #16
    s.xor        s11, s17, s10
    s.movz       s12, #0
    s.sub        s13, s11, s10
    v.splat      v0, s15
    s.mul        s10, s13, s9
    v.splat      v1, s12
    v.splat      v2, s13
    v.iota       v3                              // arange(0, VLEN)
    v.cmp.ge     p1, v0, v1
    s.andi       s11, s10, #63
    v.splat      v0, s10
    v.splat      v4, s11
    v.cmp.ge     p1, v2, v1
    v.add        v1, v0, v3
    v.add        v0, v4, v3
    s.addi       s13, s12, #0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a0
for_5:
    s.cmp.ge     s12, s13, s3
    br.nz        s12, endfor_6
    s.mul        s12, s13, s9
    s.muli       s14, s12, #2                    // a runtime tile origin, x2
    s.add        s16, s2, s14                    // the tile origin
    v.ld.h       v0, [s16, #0]
    w.def        w0, #56, #8                     // v56..v63
    mb.arm       s14, b0, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a tile contiguous down its columns: its transpose is a plain tile, written transposed by the engine (ISA sec 9E mod[5])
    s.mov        s16, s11                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s17, s12                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h.t w0, d0, s16, s14
    mb.wait      s14
    s.rdspecial  s12, arg_base
    s.ld.d       s16, [s12, #24]                 // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s12, role_id
    s.movz       s17, #1024
    s.mul        s18, s12, s17
    s.add        s12, s16, s18                   // this context's own scratch
    s.addi       s16, s12, #256                  // 512 bytes of repack scratch
    // repack 16x16 f16 through memory: there is no v.pack (ISA sec 8.4)
    v.st.h       [s16, #0], v0
    s.addi       s17, s16, #32                   // row 1
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #64                   // row 2
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #96                   // row 3
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #128                  // row 4
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #160                  // row 5
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #192                  // row 6
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #224                  // row 7
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #256                  // row 8
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #288                  // row 9
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #320                  // row 10
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #352                  // row 11
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #384                  // row 12
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #416                  // row 13
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #448                  // row 14
    v.st.h       [s17, #0], v0
    s.addi       s17, s16, #480                  // row 15
    v.st.h       [s17, #0], v0
    s.addi       s17, s12, #768                  // TensorMap 1: a repacked tile
    s.movz       s12, #0
    s.st.d       [s17, #0], s16
    s.st.d       [s17, #8], s9
    s.st.d       [s17, #16], s9
    s.movz       s16, #32
    s.st.d       [s17, #48], s16
    s.movz       s16, #2
    s.st.d       [s17, #56], s16
    .li       s16, #68719476752
    s.st.d       [s17, #88], s16
    .li       s16, #8589934592
    s.st.d       [s17, #104], s16
    fence.release.device
    td.load      d1, [s17]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    w.def        w1, #48, #8                     // v48..v55
    mb.arm       s16, b1, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a repacked tile: one tile into a window (ISA sec 9E)
    s.mov        s17, s12                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s18, s12                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w1, d1, s17, s16
    mb.wait      s16
    mb.arm       s12, b2, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.dp.async.k16.f16 a0, v48, v56, s12
    mb.wait      s12
    s.add        s17, s13, s5
    s.mov        s13, s17
    s.movz       s17, #1
    br.nz        s17, for_5
endfor_6:
    s.mul        s11, s15, s9
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.movz       s9, #0
    v.iota       v0                              // arange(0, VLEN)
    s.add        s13, s9, s10                    // the tile origin
    a.vread      v1, a0, s9
    v.splat      v2, s13
    s.add        s13, s9, s11                    // the tile origin
    s.movz       s9, #64
    s.muli       s15, s10, #2                    // a runtime tile origin, x2
    v.splat      v4, s13
    v.splat      v5, s5
    v.add        v6, v0, v2
    v.splat      v0, s9
    vf.cvt.f32.f16 v2, v1
    v.cmp.lt     p1, v4, v5
    v.cmp.lt     p2, v6, v0
    s.add        s9, s4, s15                     // the tile origin
    p.and        p3, p1, p2
    s.movz       s10, #1
    @p3 v.st.h       [s9, #0], v2
    a.vread      v0, a0, s10
    s.addi       s10, s13, #1
    s.movz       s15, #2
    v.splat      v1, s10
    vf.cvt.f32.f16 v2, v0
    s.addi       s10, s13, #2
    v.cmp.lt     p1, v1, v5
    p.and        p3, p1, p2
    v.splat      v0, s10
    s.movz       s10, #3
    s.addi       s17, s13, #3
    v.cmp.lt     p1, v0, v5
    p.and        p4, p1, p2
    @p3 v.st.h       [s9, #0], v2
    a.vread      v0, a0, s15
    v.splat      v1, s17
    s.movz       s15, #4
    vf.cvt.f32.f16 v2, v0
    v.cmp.lt     p1, v1, v5
    p.and        p3, p1, p2
    s.addi       s17, s13, #4
    v.splat      v0, s11
    v.splat      v1, s17
    v.add        v4, v0, v3
    s.movz       s11, #5
    s.addi       s17, s13, #5
    @p4 v.st.h       [s9, #0], v2
    a.vread      v0, a0, s10
    v.cmp.lt     p1, v1, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v1, v0
    v.splat      v0, s17
    s.movz       s10, #6
    s.addi       s17, s13, #6
    s.movz       s18, #7
    v.splat      v2, s17
    s.addi       s17, s13, #7
    s.movz       s19, #8
    v.splat      v3, s17
    s.addi       s17, s13, #8
    @p3 v.st.h       [s9, #0], v1
    a.vread      v1, a0, s15
    v.cmp.lt     p1, v0, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v0, v1
    v.splat      v1, s17
    s.movz       s15, #9
    s.addi       s17, s13, #9
    v.splat      v4, s17
    s.movz       s17, #10
    @p4 v.st.h       [s9, #0], v0
    a.vread      v0, a0, s11
    v.cmp.lt     p1, v2, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v2, v0
    s.addi       s11, s13, #10
    v.splat      v0, s11
    s.movz       s11, #11
    @p3 v.st.h       [s9, #0], v2
    a.vread      v2, a0, s10
    v.cmp.lt     p1, v3, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v3, v2
    s.addi       s10, s13, #11
    v.splat      v2, s10
    s.movz       s10, #12
    @p4 v.st.h       [s9, #0], v3
    a.vread      v3, a0, s18
    v.cmp.lt     p1, v1, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v1, v3
    s.addi       s18, s13, #12
    v.splat      v3, s18
    s.movz       s18, #13
    @p3 v.st.h       [s9, #0], v1
    a.vread      v1, a0, s19
    v.cmp.lt     p1, v4, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v4, v1
    s.addi       s19, s13, #13
    v.splat      v1, s19
    s.movz       s19, #14
    @p4 v.st.h       [s9, #0], v4
    a.vread      v4, a0, s15
    v.cmp.lt     p1, v0, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v0, v4
    s.addi       s15, s13, #14
    s.addi       s20, s13, #15
    v.splat      v4, s15
    s.movz       s13, #15
    v.splat      v6, s20
    @p3 v.st.h       [s9, #0], v0
    a.vread      v0, a0, s17
    v.cmp.lt     p1, v2, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v2, v0
    @p4 v.st.h       [s9, #0], v2
    a.vread      v0, a0, s11
    v.cmp.lt     p1, v3, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v2, v0
    @p3 v.st.h       [s9, #0], v2
    a.vread      v0, a0, s10
    v.cmp.lt     p1, v1, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v1, v0
    @p4 v.st.h       [s9, #0], v1
    a.vread      v0, a0, s18
    v.cmp.lt     p1, v4, v5
    p.and        p4, p1, p2
    vf.cvt.f32.f16 v1, v0
    @p3 v.st.h       [s9, #0], v1
    a.vread      v0, a0, s19
    v.cmp.lt     p1, v6, v5
    p.and        p3, p1, p2
    vf.cvt.f32.f16 v1, v0
    @p4 v.st.h       [s9, #0], v1
    a.vread      v0, a0, s13
    vf.cvt.f32.f16 v1, v0
    @p3 v.st.h       [s9, #0], v1
__rctx_next:
    s.addi       s9, s7, #2                      // the next program of this context: + 2
    s.cmp.lt     s10, s9, s8
    s.mov        s7, s9
    br.nz        s10, __rctx_program
    exit
