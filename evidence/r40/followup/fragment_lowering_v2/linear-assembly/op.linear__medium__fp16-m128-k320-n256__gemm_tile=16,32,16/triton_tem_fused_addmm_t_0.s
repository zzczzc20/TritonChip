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
    s.movz       s5, #3328
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
    s.movz       s13, #3328
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
    s.movz       s21, #8
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
    s.movz       s1, #1
    s.st.d       [s7, #48], s5
    s.cmp.lt     s5, s4, s2
    s.movz       s2, #0
    s.st.d       [s7, #56], s8
    s.sub        s8, s12, s5
    s.movz       s5, #0
    s.xor        s12, s4, s8
    s.st.d       [s7, #88], s9
    s.sub        s9, s12, s8
    s.movz       s12, #0
    s.st.d       [s7, #104], s10
    s.lshri      s10, s9, #3
    fence.release.device
    td.load      d0, [s7]                        // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.xor        s7, s8, s6
    s.movz       s8, #0
    s.ld.d       s23, [s11, #32]                 // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.cmp.lt     s11, s4, s5
    s.add        s5, s23, s15                    // this context's own scratch
    s.xor        s15, s10, s7
    s.sub        s23, s8, s11
    s.sub        s8, s15, s7
    s.muli       s7, s10, #8
    s.sub        s10, s21, s8
    s.movz       s11, #0
    s.min.s      s15, s10, s1
    s.addi       s10, s5, #1280                  // TensorMap 1: a tile contiguous down its columns (built once, at entry)
    s.cmp.lt     s5, s15, s2
    s.st.d       [s10, #0], s20
    s.sub        s2, s12, s5
    s.movz       s5, #1
    s.sub        s12, s9, s7
    s.xor        s7, s4, s23
    s.xor        s9, s21, s6
    s.sub        s12, s7, s23
    s.sub        s7, s9, s6
    s.movz       s6, #0
    s.st.d       [s10, #8], s14
    s.xor        s7, s15, s2
    s.sub        s9, s7, s2
    s.st.d       [s10, #16], s3
    s.addi       s2, s11, #0
    s.addi       s3, s6, #0
    s.st.d       [s10, #48], s13
    s.cmp.eq     s6, s9, s5
    s.st.d       [s10, #56], s16
    s.st.d       [s10, #88], s17
    s.st.d       [s10, #104], s19
    fence.release.device
    td.load      d1, [s10]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.mov        s2, s12
    br.nz        s6, sdiv_done_1
    s.movz       s5, #0
    s.movz       s6, #31
    s.mov        s2, s5
    s.addi       s5, s6, #0
sdiv_2:
    s.lshr       s6, s12, s5
    s.shli       s7, s3, #1
    s.andi       s10, s6, #1
    s.movz       s6, #0
    s.or         s11, s7, s10
    s.movz       s7, #1
    s.cmp.geu    s10, s11, s9
    s.shl        s13, s7, s5
    s.sub        s7, s6, s10
    s.subi       s6, s5, #1
    s.and        s10, s9, s7
    s.and        s14, s13, s7
    s.sub        s7, s11, s10
    s.or         s10, s2, s14
    s.mov        s3, s7
    s.mov        s2, s10
    s.mov        s5, s6
    s.movz       s6, #0
    s.cmp.ge     s7, s5, s6
    br.nz        s7, sdiv_2
sdiv_done_1:
    s.movz       s2, #0
    s.movz       s5, #0
    s.cmp.lt     s6, s4, s2
    s.movz       s2, #0
    s.sub        s7, s5, s6
    s.movz       s5, #0
    s.xor        s6, s4, s7
    s.movz       s4, #0
    s.sub        s9, s6, s7
    s.cmp.lt     s6, s15, s2
    s.lshri      s2, s9, #3
    s.movz       s10, #0
    s.muli       s11, s2, #8
    s.sub        s2, s4, s6
    s.movz       s4, #0
    s.cmp.lt     s6, s21, s10
    s.xor        s10, s15, s2
    s.sub        s12, s9, s11
    s.movz       s9, #0
    s.xor        s11, s12, s7
    s.xor        s12, s3, s23
    s.sub        s3, s11, s7
    s.sub        s7, s9, s6
    s.cmp.lt     s6, s3, s5
    s.sub        s5, s10, s2
    s.sub        s9, s4, s6
    s.movz       s4, #0
    s.xor        s6, s3, s9
    s.movz       s3, #1
    s.sub        s10, s12, s23
    s.xor        s11, s21, s7
    s.sub        s12, s6, s9
    s.movz       s6, #0
    s.addi       s13, s4, #0
    s.cmp.eq     s4, s5, s3
    s.add        s3, s8, s10
    s.sub        s8, s11, s7
    s.addi       s7, s6, #0
    s.mov        s13, s12
    br.nz        s4, sdiv_done_3
    s.movz       s4, #0
    s.movz       s6, #31
    s.mov        s13, s4
    s.addi       s4, s6, #0
sdiv_4:
    s.lshr       s6, s12, s4
    s.shli       s8, s7, #1
    s.andi       s10, s6, #1
    s.movz       s6, #0
    s.or         s11, s8, s10
    s.movz       s8, #1
    s.cmp.geu    s10, s11, s5
    s.shl        s14, s8, s4
    s.sub        s8, s6, s10
    s.subi       s6, s4, #1
    s.and        s10, s5, s8
    s.and        s15, s14, s8
    s.sub        s8, s11, s10
    s.or         s10, s13, s15
    s.mov        s7, s8
    s.mov        s13, s10
    s.mov        s4, s6
    s.movz       s6, #0
    s.cmp.ge     s8, s4, s6
    br.nz        s8, sdiv_4
sdiv_done_3:
    s.xor        s4, s9, s2
    s.movz       s2, #32
    s.xor        s5, s13, s4
    v.iota       v0                              // arange(0, VLEN)
    s.sub        s6, s5, s4
    v.movi       v1, #16
    s.mul        s4, s6, s2
    v.movi       v2, #0
    v.add        v3, v0, v1
    s.movz       s2, #16
    v.movi       v0, #255
    v.splat      v1, s4
    s.mul        s5, s3, s2
    v.movi       v4, #256
    v.add        v5, v1, v3
    s.movz       s7, #0
    s.andi       s8, s4, #255
    v.sub        v1, v2, v5
    v.cmp.lt     p1, v5, v2
    s.andi       s9, s5, #127
    @p1 v.sel        v3, v1, v5
    v.splat      v1, s3
    v.splat      v5, s7
    v.and        v6, v3, v0
    v.splat      v0, s6
    v.splat      v3, s7
    v.iota       v7                              // arange(0, VLEN)
    v.cmp.lt     p2, v4, v2
    v.cmp.ge     p3, v1, v5
    v.splat      v1, s5
    v.iota       v5                              // arange(0, VLEN)
    v.cmp.ge     p3, v0, v3
    v.splat      v0, s4
    v.splat      v3, s9
    v.add        v8, v1, v7
    v.iota       v1                              // arange(0, VLEN)
    v.splat      v7, s8
    v.add        v8, v0, v5
    v.iota       v0                              // arange(0, VLEN)
    v.sub        v5, v2, v4
    v.sub        v8, v2, v6
    s.add        s3, s2, s4
    v.add        v2, v3, v1
    v.add        v1, v7, v0
    v.movi       v0, #1
    @p2 v.sel        v0, v5, v4
    @p1 v.sel        v0, v8, v6
    s.movz       s6, #20
    s.addi       s10, s7, #0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a1
for_5:
    s.cmp.ge     s11, s10, s6
    br.nz        s11, endfor_6
    s.mul        s11, s10, s2
    w.def        w0, #56, #8                     // v56..v63
    mb.arm       s12, b0, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a 2-D tt.load: one tile into a window (ISA sec 9E)
    s.mov        s13, s9                         // coordinates must be consecutive (ISA sec 9E)
    s.mov        s14, s11                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w0, d0, s13, s12
    mb.wait      s12
    s.rdspecial  s13, arg_base
    s.ld.d       s14, [s13, #32]                 // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s13, role_id
    s.movz       s15, #3328
    s.mul        s16, s13, s15
    s.add        s13, s14, s16                   // this context's own scratch
    s.addi       s14, s13, #256
    v.splat      v1, s7
    s.movz       s15, #0
    s.add        s16, s15, s11                   // the tile origin
    v.splat      v2, s16
    s.movz       s15, #320
    v.splat      v3, s15
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s14, #0], v2
    s.addi       s15, s14, #64
    s.addi       s17, s16, #1
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #128
    s.addi       s17, s16, #2
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #192
    s.addi       s17, s16, #3
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #256
    s.addi       s17, s16, #4
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #320
    s.addi       s17, s16, #5
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #384
    s.addi       s17, s16, #6
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #448
    s.addi       s17, s16, #7
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #512
    s.addi       s17, s16, #8
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #576
    s.addi       s17, s16, #9
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #640
    s.addi       s17, s16, #10
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #704
    s.addi       s17, s16, #11
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #768
    s.addi       s17, s16, #12
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #832
    s.addi       s17, s16, #13
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #896
    s.addi       s17, s16, #14
    v.splat      v2, s17
    v.mul        v4, v0, v3
    v.add        v5, v2, v4
    v.add        v2, v1, v5
    v.st.w       [s15, #0], v2
    s.addi       s15, s14, #960
    s.addi       s14, s16, #15
    v.splat      v2, s14
    v.mul        v4, v0, v3
    v.add        v3, v2, v4
    v.add        v2, v1, v3
    v.st.w       [s15, #0], v2
    fence.acq_rel.device
    w.def        w1, #48, #8                     // v48..v55
    mb.arm       s14, b1, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a tile contiguous down its columns: its transpose is a plain tile, written transposed by the engine (ISA sec 9E mod[5])
    s.mov        s15, s8                         // coordinates must be consecutive (ISA sec 9E)
    s.mov        s16, s11                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h.t w1, d1, s15, s14
    mb.wait      s14
    s.addi       s11, s13, #1536
    s.addi       s15, s13, #256
    v.ld.w       v1, [s15, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s11, #0], v2
    s.addi       s15, s11, #64
    s.addi       s16, s13, #320
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #128
    s.addi       s16, s13, #384
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #192
    s.addi       s16, s13, #448
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #256
    s.addi       s16, s13, #512
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #320
    s.addi       s16, s13, #576
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #384
    s.addi       s16, s13, #640
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #448
    s.addi       s16, s13, #704
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #512
    s.addi       s16, s13, #768
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #576
    s.addi       s16, s13, #832
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #640
    s.addi       s16, s13, #896
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #704
    s.addi       s16, s13, #960
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #768
    s.addi       s16, s13, #1024
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #832
    s.addi       s16, s13, #1088
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #896
    s.addi       s16, s13, #1152
    v.ld.w       v1, [s16, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    s.addi       s15, s11, #960
    s.addi       s11, s13, #1216
    v.ld.w       v1, [s11, #0]
    v.ld.gather.h v2, [s20, v1]                  // index is in elements (ISA sec 9.1)
    v.st.w       [s15, #0], v2
    fence.acq_rel.device
    mb.arm       s11, b2, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.async.k16.f16 a0, v56, v48, s11
    mb.wait      s11
    s.addi       s15, s13, #2560                 // 512 bytes of repack scratch
    // repack 16x16 f16 through memory: there is no v.pack (ISA sec 8.4)
    s.addi       s16, s13, #1536
    v.ld.w       v1, [s16, #0]
    v.st.h       [s15, #0], v1
    s.addi       s16, s13, #1600
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #32                   // row 1
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1664
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #64                   // row 2
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1728
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #96                   // row 3
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1792
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #128                  // row 4
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1856
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #160                  // row 5
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1920
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #192                  // row 6
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #1984
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #224                  // row 7
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2048
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #256                  // row 8
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2112
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #288                  // row 9
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2176
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #320                  // row 10
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2240
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #352                  // row 11
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2304
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #384                  // row 12
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2368
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #416                  // row 13
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2432
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #448                  // row 14
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #2496
    v.ld.w       v1, [s16, #0]
    s.addi       s16, s15, #480                  // row 15
    v.st.h       [s16, #0], v1
    s.addi       s16, s13, #3072                 // TensorMap 2: a repacked tile
    s.movz       s13, #0
    s.st.d       [s16, #0], s15
    s.movz       s13, #16
    s.st.d       [s16, #8], s13
    s.movz       s13, #16
    s.st.d       [s16, #16], s13
    s.movz       s13, #32
    s.st.d       [s16, #48], s13
    s.movz       s13, #2
    s.st.d       [s16, #56], s13
    .li       s13, #68719476752
    s.st.d       [s16, #88], s13
    .li       s13, #8589934592
    s.st.d       [s16, #104], s13
    fence.release.device
    td.load      d2, [s16]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s13, #0
    w.def        w2, #40, #8                     // v40..v47
    mb.arm       s15, b3, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a repacked tile: one tile into a window (ISA sec 9E)
    s.mov        s16, s13                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s17, s13                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w2, d2, s16, s15
    mb.wait      s15
    mb.arm       s13, b4, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.async.k16.f16 a1, v56, v40, s13
    mb.wait      s13
    s.add        s16, s10, s1
    s.mov        s10, s16
    s.movz       s16, #1
    br.nz        s16, for_5
endfor_6:
    s.movz       s1, #0
    s.movz       s2, #0
    s.add        s6, s1, s4                      // the tile origin
    v.iota       v0                              // arange(0, VLEN)
    v.splat      v1, s6
    s.add        s1, s2, s5                      // the tile origin
    s.movz       s2, #128
    s.movz       s6, #256
    s.muli       s7, s4, #2                      // a runtime tile origin, x2
    v.splat      v2, s1
    v.splat      v3, s2
    v.add        v4, v0, v1
    v.splat      v1, s6
    v.cmp.lt     p1, v2, v3
    s.add        s2, s18, s7                     // the tile origin
    v.cmp.lt     p2, v4, v1
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.movz       s7, #0
    p.and        p3, p1, p2
    a.vread      v1, a0, s7
    v.movi       v2, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v2, [s2, #0]
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.muli       s7, s4, #2                      // a runtime tile origin, x2
    s.addi       s4, s1, #1
    s.muli       s8, s5, #512                    // a runtime tile origin, x512
    v.splat      v4, s1
    v.splat      v5, s4
    s.add        s4, s22, s7                     // the tile origin
    v.cmp.lt     p1, v4, v3
    v.cmp.lt     p3, v5, v3
    s.add        s7, s4, s8                      // the tile origin
    p.and        p4, p1, p2
    s.movz       s4, #1
    p.and        p1, p3, p2
    s.addi       s8, s1, #1
    vf.cvt.f16.f32 v4, v2
    v.splat      v2, s8
    s.addi       s8, s7, #512                    // row 1
    s.movz       s9, #2
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.muli       s10, s3, #2                     // a runtime tile origin, x2
    vf.add       v5, v1, v4
    s.add        s16, s18, s10                   // the tile origin
    vf.cvt.f32.f16 v1, v5
    s.addi       s10, s1, #2
    s.addi       s17, s1, #2
    v.splat      v4, s10
    s.addi       s10, s1, #3
    @p4 v.st.h       [s7, #0], v1
    a.vread      v1, a0, s4
    v.cmp.lt     p3, v2, v3
    p.and        p4, p3, p2
    v.splat      v2, s17
    v.splat      v5, s10
    s.addi       s4, s7, #1024                   // row 2
    s.movz       s10, #3
    s.addi       s17, s1, #3
    s.addi       s18, s1, #4
    v.splat      v6, s17
    v.splat      v7, s18
    s.addi       s17, s7, #1536                  // row 3
    s.movz       s18, #4
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p1 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v8
    vf.add       v8, v1, v4
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a0, s9
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s8, s1, #4
    s.addi       s9, s1, #5
    v.splat      v2, s8
    v.splat      v4, s9
    s.addi       s8, s7, #2048                   // row 4
    s.movz       s9, #5
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v8
    vf.add       v8, v1, v5
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a0, s10
    v.cmp.lt     p1, v6, v3
    p.and        p4, p1, p2
    s.addi       s4, s1, #5
    s.addi       s10, s1, #6
    v.splat      v5, s4
    v.splat      v6, s10
    s.addi       s4, s7, #2560                   // row 5
    s.movz       s10, #6
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v7, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v7, v8
    vf.add       v8, v1, v7
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s17, #0], v1
    a.vread      v1, a0, s18
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s17, s1, #6
    s.addi       s18, s1, #7
    v.splat      v2, s17
    v.splat      v7, s18
    s.addi       s17, s7, #3072                  // row 6
    s.movz       s18, #7
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v8
    vf.add       v8, v1, v4
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a0, s9
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    s.addi       s8, s1, #7
    s.addi       s9, s1, #8
    v.splat      v4, s8
    v.splat      v5, s9
    s.addi       s8, s7, #3584                   // row 7
    s.movz       s9, #8
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v8
    vf.add       v8, v1, v6
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a0, s10
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s4, s1, #8
    s.addi       s10, s1, #9
    v.splat      v2, s4
    v.splat      v6, s10
    s.addi       s4, s7, #4096                   // row 8
    s.movz       s10, #9
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v7, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v7, v8
    vf.add       v8, v1, v7
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s17, #0], v1
    a.vread      v1, a0, s18
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s17, s1, #9
    s.addi       s18, s1, #10
    v.splat      v4, s17
    v.splat      v7, s18
    s.addi       s17, s7, #4608                  // row 9
    s.movz       s18, #10
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v8
    vf.add       v8, v1, v5
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a0, s9
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s8, s1, #10
    s.addi       s9, s1, #11
    v.splat      v2, s8
    v.splat      v5, s9
    s.addi       s8, s7, #5120                   // row 10
    s.movz       s9, #11
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v8
    vf.add       v8, v1, v6
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a0, s10
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s4, s1, #11
    s.addi       s10, s1, #12
    v.splat      v4, s4
    v.splat      v6, s10
    s.addi       s4, s7, #5632                   // row 11
    s.movz       s10, #12
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v7, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v7, v8
    vf.add       v8, v1, v7
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s17, #0], v1
    a.vread      v1, a0, s18
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s17, s1, #12
    s.addi       s18, s1, #13
    v.splat      v2, s17
    v.splat      v7, s18
    s.addi       s17, s7, #6144                  // row 12
    s.movz       s18, #13
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v8
    vf.add       v8, v1, v5
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a0, s9
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s8, s1, #13
    s.addi       s9, s1, #14
    v.splat      v4, s8
    v.splat      v5, s9
    s.addi       s8, s7, #6656                   // row 13
    s.movz       s9, #14
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v8
    vf.add       v8, v1, v6
    vf.cvt.f32.f16 v1, v8
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a0, s10
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s4, s1, #14
    s.addi       s10, s1, #15
    v.splat      v2, s4
    v.splat      v6, s10
    s.addi       s4, s7, #7168                   // row 14
    s.movz       s10, #15
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v7, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v7, v8
    vf.add       v8, v1, v7
    vf.cvt.f32.f16 v1, v8
    s.addi       s19, s7, #7680                  // row 15
    @p4 v.st.h       [s17, #0], v1
    a.vread      v1, a0, s18
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s7, s1, #15
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.movz       s17, #0
    v.splat      v4, s7
    s.muli       s7, s3, #2                      // a runtime tile origin, x2
    s.muli       s18, s5, #512                   // a runtime tile origin, x512
    s.add        s5, s22, s7                     // the tile origin
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s2, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    v.splat      v5, s1
    s.add        s7, s5, s18                     // the tile origin
    vf.cvt.f16.f32 v8, v7
    vf.add       v7, v1, v8
    s.movz       s5, #0
    vf.cvt.f32.f16 v1, v7
    s.add        s18, s5, s3                     // the tile origin
    v.splat      v7, s6
    v.splat      v8, s18
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a0, s9
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    v.add        v2, v0, v8
    v.splat      v0, s1
    v.movi       v8, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v8, [s2, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v8
    s.movz       s3, #1
    vf.add       v8, v1, v6
    vf.cvt.f32.f16 v1, v8
    s.addi       s5, s1, #1
    v.splat      v6, s5
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a0, s10
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    v.cmp.lt     p1, v5, v3
    s.addi       s4, s1, #1
    s.addi       s5, s7, #512                    // row 1
    v.splat      v4, s4
    v.movi       v5, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v5, [s2, #0]
    v.cmp.lt     p2, v2, v7
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v5
    s.movz       s2, #2
    vf.add       v5, v1, v2
    vf.cvt.f32.f16 v1, v5
    s.addi       s4, s1, #2
    v.splat      v2, s4
    @p4 v.st.h       [s19, #0], v1
    a.vread      v1, a1, s17
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s4, s1, #2
    s.addi       s6, s1, #3
    v.splat      v0, s4
    v.splat      v5, s6
    s.addi       s4, s7, #1024                   // row 2
    s.movz       s6, #3
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v1, v6
    s.addi       s8, s7, #1536                   // row 3
    vf.cvt.f32.f16 v1, v7
    s.addi       s9, s1, #3
    v.splat      v6, s9
    @p4 v.st.h       [s7, #0], v1
    a.vread      v1, a1, s3
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s3, s1, #4
    s.movz       s9, #4
    v.splat      v4, s3
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v7
    vf.add       v7, v1, v2
    vf.cvt.f32.f16 v1, v7
    s.addi       s3, s1, #4
    s.addi       s10, s7, #2048                  // row 4
    v.splat      v2, s3
    @p4 v.st.h       [s5, #0], v1
    a.vread      v1, a1, s2
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #5
    s.movz       s3, #5
    v.splat      v0, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v1, v5
    vf.cvt.f32.f16 v1, v7
    s.addi       s2, s1, #5
    s.addi       s5, s7, #2560                   // row 5
    v.splat      v5, s2
    @p4 v.st.h       [s4, #0], v1
    a.vread      v1, a1, s6
    v.cmp.lt     p1, v6, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #6
    s.movz       s4, #6
    v.splat      v6, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v1, v4
    vf.cvt.f32.f16 v1, v7
    s.addi       s2, s1, #6
    s.addi       s6, s7, #3072                   // row 6
    v.splat      v4, s2
    @p4 v.st.h       [s8, #0], v1
    a.vread      v1, a1, s9
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #7
    s.movz       s8, #7
    v.splat      v2, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v0, v7
    vf.add       v7, v1, v0
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #7
    s.addi       s9, s7, #3584                   // row 7
    v.splat      v1, s2
    @p4 v.st.h       [s10, #0], v0
    a.vread      v0, a1, s3
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #8
    s.movz       s3, #8
    v.splat      v5, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v0, v6
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #8
    s.addi       s10, s7, #4096                  // row 8
    v.splat      v6, s2
    @p4 v.st.h       [s5, #0], v0
    a.vread      v0, a1, s4
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #9
    s.movz       s4, #9
    v.splat      v4, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v7
    vf.add       v7, v0, v2
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #9
    s.addi       s5, s7, #4608                   // row 9
    v.splat      v2, s2
    @p4 v.st.h       [s6, #0], v0
    a.vread      v0, a1, s8
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #10
    s.movz       s6, #10
    v.splat      v1, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v0, v5
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #10
    s.addi       s8, s7, #5120                   // row 10
    v.splat      v5, s2
    @p4 v.st.h       [s9, #0], v0
    a.vread      v0, a1, s3
    v.cmp.lt     p1, v6, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #11
    s.movz       s3, #11
    v.splat      v6, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v7
    vf.add       v7, v0, v4
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #11
    s.addi       s9, s7, #5632                   // row 11
    v.splat      v4, s2
    @p4 v.st.h       [s10, #0], v0
    a.vread      v0, a1, s4
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #12
    s.movz       s4, #12
    v.splat      v2, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v1, v7
    vf.add       v7, v0, v1
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #12
    s.addi       s10, s7, #6144                  // row 12
    v.splat      v1, s2
    @p4 v.st.h       [s5, #0], v0
    a.vread      v0, a1, s6
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #13
    s.movz       s5, #13
    v.splat      v5, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v6, v7
    vf.add       v7, v0, v6
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #13
    s.addi       s6, s7, #6656                   // row 13
    v.splat      v6, s2
    @p4 v.st.h       [s8, #0], v0
    a.vread      v0, a1, s3
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #14
    s.movz       s3, #14
    v.splat      v4, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v2, v7
    vf.add       v7, v0, v2
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #14
    s.addi       s8, s7, #7168                   // row 14
    v.splat      v2, s2
    @p4 v.st.h       [s9, #0], v0
    a.vread      v0, a1, s4
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s2, s1, #15
    s.movz       s4, #15
    v.splat      v1, s2
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v7, [s16, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v5, v7
    vf.add       v7, v0, v5
    vf.cvt.f32.f16 v0, v7
    s.addi       s2, s1, #15
    s.addi       s1, s7, #7680                   // row 15
    v.splat      v5, s2
    @p4 v.st.h       [s10, #0], v0
    a.vread      v0, a1, s5
    v.cmp.lt     p1, v6, v3
    p.and        p4, p1, p2
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v6, [s16, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v4, v6
    vf.add       v6, v0, v4
    vf.cvt.f32.f16 v0, v6
    @p4 v.st.h       [s6, #0], v0
    a.vread      v0, a1, s3
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    v.movi       v2, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v2, [s16, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    vf.cvt.f16.f32 v1, v2
    vf.add       v2, v0, v1
    vf.cvt.f32.f16 v0, v2
    @p4 v.st.h       [s8, #0], v0
    a.vread      v0, a1, s4
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v1, [s16, #0]
    vf.cvt.f16.f32 v2, v1
    vf.add       v1, v0, v2
    vf.cvt.f32.f16 v0, v1
    @p4 v.st.h       [s1, #0], v0
__rctx_next:
    exit
