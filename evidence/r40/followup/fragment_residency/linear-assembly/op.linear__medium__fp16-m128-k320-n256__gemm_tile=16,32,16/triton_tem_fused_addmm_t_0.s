    s.rdspecial  s1, role_id
    s.rdspecial  s2, pid.x
    s.rdspecial  s3, npid.x
    s.add        s4, s2, s1
    s.cmp.ge     s1, s4, s3
    br.nz        s1, __rctx_next
__rctx_program:
    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    s.ld.d       s3, [s1, #8]                    // arg_A: device_ptr64 at arg_base+8
    s.ld.d       s5, [s1, #16]                   // arg_B: device_ptr64 at arg_base+16
    s.ld.d       s6, [s1, #24]                   // out_ptr0: device_ptr64 at arg_base+24
    s.rdspecial  s1, arg_base
    s.ld.d       s7, [s1, #32]                   // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s1, role_id
    s.movz       s8, #10048
    s.mul        s9, s1, s8
    s.add        s1, s7, s9                      // this context's own scratch
    s.addi       s7, s1, #0                      // TensorMap 0: a 2-D tt.load (built once, at entry)
    s.movz       s1, #0
    s.st.d       [s7, #0], s3
    .li       s1, #2147483647
    s.st.d       [s7, #8], s1
    .li       s1, #2147483647
    s.st.d       [s7, #16], s1
    s.movz       s1, #640
    s.st.d       [s7, #48], s1
    s.movz       s1, #2
    s.st.d       [s7, #56], s1
    .li       s1, #68719476752
    s.st.d       [s7, #88], s1
    .li       s1, #8589934592
    s.st.d       [s7, #104], s1
    fence.release.device
    td.load      d0, [s7]                        // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s1, #8
    s.movz       s3, #0
    s.movz       s7, #0
    s.movz       s8, #0
    s.movz       s9, #0
    s.cmp.lt     s10, s1, s9
    s.sub        s9, s8, s10
    s.xor        s8, s1, s9
    s.sub        s10, s8, s9
    s.movz       s8, #1
    s.movz       s10, #0
    s.movz       s11, #0
    s.movz       s12, #0
    s.movz       s13, #0
    s.movz       s14, #0
    s.movz       s15, #0
    s.movz       s16, #1
    s.cmp.lt     s17, s4, s7
    s.sub        s7, s3, s17
    s.xor        s3, s4, s7
    s.sub        s17, s3, s7
    s.lshri      s3, s17, #3
    s.muli       s18, s3, #8
    s.sub        s19, s17, s18
    s.xor        s17, s7, s9
    s.xor        s7, s3, s17
    s.sub        s3, s7, s17
    s.sub        s7, s1, s3
    s.min.s      s9, s7, s8
    s.cmp.lt     s7, s4, s11
    s.sub        s11, s10, s7
    s.cmp.lt     s7, s9, s13
    s.sub        s10, s12, s7
    s.xor        s7, s4, s11
    s.sub        s12, s7, s11
    s.xor        s7, s9, s10
    s.sub        s13, s7, s10
    s.addi       s7, s14, #0
    s.addi       s10, s15, #0
    s.cmp.eq     s14, s13, s16
    s.mov        s7, s12
    br.nz        s14, sdiv_done_1
    s.movz       s14, #0
    s.movz       s15, #31
    s.mov        s7, s14
    s.addi       s14, s15, #0
sdiv_2:
    s.lshr       s15, s12, s14
    s.shli       s16, s10, #1
    s.andi       s17, s15, #1
    s.movz       s15, #0
    s.or         s18, s16, s17
    s.movz       s16, #1
    s.cmp.geu    s17, s18, s13
    s.shl        s19, s16, s14
    s.sub        s16, s15, s17
    s.subi       s15, s14, #1
    s.and        s17, s13, s16
    s.and        s20, s19, s16
    s.sub        s16, s18, s17
    s.or         s17, s7, s20
    s.mov        s10, s16
    s.mov        s7, s17
    s.mov        s14, s15
    s.movz       s15, #0
    s.cmp.ge     s16, s14, s15
    br.nz        s16, sdiv_2
sdiv_done_1:
    s.movz       s7, #0
    s.movz       s12, #0
    s.cmp.lt     s13, s4, s7
    s.movz       s7, #0
    s.sub        s14, s12, s13
    s.movz       s12, #0
    s.xor        s13, s4, s14
    s.movz       s4, #0
    s.sub        s15, s13, s14
    s.cmp.lt     s13, s9, s7
    s.lshri      s7, s15, #3
    s.movz       s16, #0
    s.muli       s17, s7, #8
    s.sub        s7, s4, s13
    s.movz       s4, #0
    s.cmp.lt     s13, s1, s16
    s.xor        s16, s9, s7
    s.sub        s9, s15, s17
    s.movz       s15, #0
    s.xor        s17, s9, s14
    s.xor        s9, s10, s11
    s.sub        s10, s17, s14
    s.sub        s14, s15, s13
    s.cmp.lt     s13, s10, s12
    s.sub        s12, s16, s7
    s.sub        s15, s4, s13
    s.movz       s4, #0
    s.xor        s13, s10, s15
    s.movz       s10, #1
    s.sub        s16, s9, s11
    s.xor        s9, s1, s14
    s.sub        s1, s13, s15
    s.movz       s11, #0
    s.addi       s13, s4, #0
    s.cmp.eq     s4, s12, s10
    s.add        s10, s3, s16
    s.sub        s3, s9, s14
    s.addi       s3, s11, #0
    s.mov        s13, s1
    br.nz        s4, sdiv_done_3
    s.movz       s4, #0
    s.movz       s9, #31
    s.mov        s13, s4
    s.addi       s4, s9, #0
sdiv_4:
    s.lshr       s9, s1, s4
    s.shli       s11, s3, #1
    s.andi       s14, s9, #1
    s.movz       s9, #0
    s.or         s16, s11, s14
    s.movz       s11, #1
    s.cmp.geu    s14, s16, s12
    s.shl        s17, s11, s4
    s.sub        s11, s9, s14
    s.subi       s9, s4, #1
    s.and        s14, s12, s11
    s.and        s18, s17, s11
    s.sub        s11, s16, s14
    s.or         s14, s13, s18
    s.mov        s3, s11
    s.mov        s13, s14
    s.mov        s4, s9
    s.movz       s9, #0
    s.cmp.ge     s11, s4, s9
    br.nz        s11, sdiv_4
sdiv_done_3:
    s.xor        s1, s15, s7
    s.movz       s3, #32
    s.xor        s4, s13, s1
    v.iota       v0                              // arange(0, VLEN)
    s.sub        s7, s4, s1
    v.movi       v1, #16
    s.mul        s1, s7, s3
    v.movi       v2, #0
    v.add        v3, v0, v1
    s.movz       s3, #16
    v.movi       v0, #255
    v.splat      v1, s1
    s.mul        s4, s10, s3
    v.movi       v4, #256
    v.add        v5, v1, v3
    s.movz       s9, #0
    s.andi       s11, s1, #255
    v.sub        v1, v2, v5
    v.cmp.lt     p1, v5, v2
    s.andi       s12, s4, #127
    @p1 v.sel        v3, v1, v5
    v.splat      v1, s10
    v.splat      v5, s9
    v.and        v6, v3, v0
    v.splat      v0, s7
    v.splat      v3, s9
    v.iota       v7                              // arange(0, VLEN)
    v.cmp.lt     p2, v4, v2
    v.cmp.ge     p3, v1, v5
    v.splat      v1, s4
    v.iota       v5                              // arange(0, VLEN)
    v.cmp.ge     p3, v0, v3
    v.splat      v0, s1
    v.splat      v3, s12
    v.add        v8, v1, v7
    v.iota       v1                              // arange(0, VLEN)
    v.splat      v7, s11
    v.add        v8, v0, v5
    v.iota       v0                              // arange(0, VLEN)
    v.sub        v5, v2, v4
    v.sub        v8, v2, v6
    s.add        s7, s3, s1
    v.add        v2, v3, v1
    v.add        v1, v7, v0
    v.movi       v0, #1
    @p2 v.sel        v0, v5, v4
    @p1 v.sel        v0, v8, v6
    s.movz       s10, #20
    s.addi       s13, s9, #0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a0
    // a zero tile carried round the loop: one m.zero (ISA sec 9C)
    m.zero       a1
for_5:
    s.cmp.ge     s14, s13, s10
    br.nz        s14, endfor_6
    s.mul        s14, s13, s3
    w.def        w0, #56, #8                     // v56..v63
    mb.arm       s15, b0, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a 2-D tt.load: one tile into a window (ISA sec 9E)
    s.mov        s16, s12                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s17, s14                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w0, d0, s16, s15
    mb.wait      s15
    s.rdspecial  s16, arg_base
    s.ld.d       s17, [s16, #32]                 // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s16, role_id
    s.movz       s18, #10048
    s.mul        s19, s16, s18
    s.add        s16, s17, s19                   // this context's own scratch
    s.addi       s17, s16, #256
    s.movz       s18, #320
    v.splat      v1, s18
    v.mul        v2, v0, v1
    v.st.w       [s17, #0], v2
    fence.acq_rel.device
    s.addi       s17, s16, #320
    s.movz       s18, #0
    s.add        s19, s18, s14                   // the tile origin
    v.splat      v1, s19
    s.addi       s18, s16, #256
    v.ld.w       v2, [s18, #0]
    v.add        v3, v1, v2
    v.st.w       [s17, #0], v3
    s.addi       s18, s17, #64
    s.addi       s20, s19, #1
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #128
    s.addi       s20, s19, #2
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #192
    s.addi       s20, s19, #3
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #256
    s.addi       s20, s19, #4
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #320
    s.addi       s20, s19, #5
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #384
    s.addi       s20, s19, #6
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #448
    s.addi       s20, s19, #7
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #512
    s.addi       s20, s19, #8
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #576
    s.addi       s20, s19, #9
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #640
    s.addi       s20, s19, #10
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #704
    s.addi       s20, s19, #11
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #768
    s.addi       s20, s19, #12
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #832
    s.addi       s20, s19, #13
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #896
    s.addi       s20, s19, #14
    v.splat      v1, s20
    s.addi       s20, s16, #256
    v.ld.w       v2, [s20, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #960
    s.addi       s17, s19, #15
    v.splat      v1, s17
    s.addi       s17, s16, #256
    v.ld.w       v2, [s17, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    fence.acq_rel.device
    s.addi       s17, s16, #1344
    v.splat      v1, s9
    s.addi       s18, s16, #320
    v.ld.w       v2, [s18, #0]
    v.add        v3, v1, v2
    v.st.w       [s17, #0], v3
    s.addi       s18, s17, #64
    s.addi       s19, s16, #384
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #128
    s.addi       s19, s16, #448
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #192
    s.addi       s19, s16, #512
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #256
    s.addi       s19, s16, #576
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #320
    s.addi       s19, s16, #640
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #384
    s.addi       s19, s16, #704
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #448
    s.addi       s19, s16, #768
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #512
    s.addi       s19, s16, #832
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #576
    s.addi       s19, s16, #896
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #640
    s.addi       s19, s16, #960
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #704
    s.addi       s19, s16, #1024
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #768
    s.addi       s19, s16, #1088
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #832
    s.addi       s19, s16, #1152
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #896
    s.addi       s19, s16, #1216
    v.ld.w       v2, [s19, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    s.addi       s18, s17, #960
    s.addi       s17, s16, #1280
    v.ld.w       v2, [s17, #0]
    v.add        v3, v1, v2
    v.st.w       [s18, #0], v3
    fence.acq_rel.device
    s.addi       s17, s16, #2368
    s.movz       s18, #0
    s.add        s19, s18, s14                   // the tile origin
    s.muli       s14, s11, #320                  // a runtime tile origin, x320
    s.add        s18, s19, s14                   // the tile origin
    v.iota       v1                              // arange(0, VLEN)
    s.movz       s14, #320
    v.splat      v2, s14
    v.mul        v3, v1, v2
    v.splat      v1, s18
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v1
    s.addi       s14, s17, #64
    s.addi       s19, s18, #1
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #128
    s.addi       s19, s18, #2
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #192
    s.addi       s19, s18, #3
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #256
    s.addi       s19, s18, #4
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #320
    s.addi       s19, s18, #5
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #384
    s.addi       s19, s18, #6
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #448
    s.addi       s19, s18, #7
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #512
    s.addi       s19, s18, #8
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #576
    s.addi       s19, s18, #9
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #640
    s.addi       s19, s18, #10
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #704
    s.addi       s19, s18, #11
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #768
    s.addi       s19, s18, #12
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #832
    s.addi       s19, s18, #13
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #896
    s.addi       s19, s18, #14
    v.splat      v1, s19
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    s.addi       s14, s17, #960
    s.addi       s17, s18, #15
    v.splat      v1, s17
    v.add        v2, v3, v1
    v.ld.gather.h v1, [s5, v2]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v1
    fence.acq_rel.device
    s.addi       s14, s16, #3392
    s.addi       s17, s16, #1344
    v.ld.w       v1, [s17, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s14, #0], v2
    s.addi       s17, s14, #64
    s.addi       s18, s16, #1408
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #128
    s.addi       s18, s16, #1472
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #192
    s.addi       s18, s16, #1536
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #256
    s.addi       s18, s16, #1600
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #320
    s.addi       s18, s16, #1664
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #384
    s.addi       s18, s16, #1728
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #448
    s.addi       s18, s16, #1792
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #512
    s.addi       s18, s16, #1856
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #576
    s.addi       s18, s16, #1920
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #640
    s.addi       s18, s16, #1984
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #704
    s.addi       s18, s16, #2048
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #768
    s.addi       s18, s16, #2112
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #832
    s.addi       s18, s16, #2176
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #896
    s.addi       s18, s16, #2240
    v.ld.w       v1, [s18, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    s.addi       s17, s14, #960
    s.addi       s14, s16, #2304
    v.ld.w       v1, [s14, #0]
    v.ld.gather.h v2, [s5, v1]                   // index is in elements (ISA sec 9.1)
    v.st.w       [s17, #0], v2
    fence.acq_rel.device
    s.addi       s14, s16, #4416                 // 512 bytes of repack scratch
    // repack 16x16 f16 through memory: there is no v.pack (ISA sec 8.4)
    s.addi       s17, s16, #2368
    v.ld.w       v1, [s17, #0]
    v.st.h       [s14, #0], v1
    s.addi       s17, s16, #2432
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #32                   // row 1
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2496
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #64                   // row 2
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2560
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #96                   // row 3
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2624
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #128                  // row 4
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2688
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #160                  // row 5
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2752
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #192                  // row 6
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2816
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #224                  // row 7
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2880
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #256                  // row 8
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #2944
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #288                  // row 9
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3008
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #320                  // row 10
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3072
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #352                  // row 11
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3136
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #384                  // row 12
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3200
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #416                  // row 13
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3264
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #448                  // row 14
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #3328
    v.ld.w       v1, [s17, #0]
    s.addi       s17, s14, #480                  // row 15
    v.st.h       [s17, #0], v1
    s.addi       s17, s16, #4928                 // TensorMap 1: a repacked tile
    s.movz       s18, #0
    s.st.d       [s17, #0], s14
    s.movz       s14, #16
    s.st.d       [s17, #8], s14
    s.movz       s14, #16
    s.st.d       [s17, #16], s14
    s.movz       s14, #32
    s.st.d       [s17, #48], s14
    s.movz       s14, #2
    s.st.d       [s17, #56], s14
    .li       s14, #68719476752
    s.st.d       [s17, #88], s14
    .li       s14, #8589934592
    s.st.d       [s17, #104], s14
    fence.release.device
    td.load      d1, [s17]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s14, #0
    w.def        w1, #48, #8                     // v48..v55
    mb.arm       s17, b1, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a repacked tile: one tile into a window (ISA sec 9E)
    s.mov        s18, s14                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s19, s14                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w1, d1, s18, s17
    mb.wait      s17
    mb.arm       s14, b2, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.async.k16.f16 a0, v56, v48, s14
    mb.wait      s14
    s.addi       s18, s16, #5184                 // 512 bytes of repack scratch
    // repack 16x16 f16 through memory: there is no v.pack (ISA sec 8.4)
    s.addi       s19, s16, #3392
    v.ld.w       v1, [s19, #0]
    v.st.h       [s18, #0], v1
    s.addi       s19, s16, #3456
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #32                   // row 1
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3520
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #64                   // row 2
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3584
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #96                   // row 3
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3648
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #128                  // row 4
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3712
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #160                  // row 5
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3776
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #192                  // row 6
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3840
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #224                  // row 7
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3904
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #256                  // row 8
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #3968
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #288                  // row 9
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4032
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #320                  // row 10
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4096
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #352                  // row 11
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4160
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #384                  // row 12
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4224
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #416                  // row 13
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4288
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #448                  // row 14
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #4352
    v.ld.w       v1, [s19, #0]
    s.addi       s19, s18, #480                  // row 15
    v.st.h       [s19, #0], v1
    s.addi       s19, s16, #5696                 // TensorMap 2: a repacked tile
    s.movz       s16, #0
    s.st.d       [s19, #0], s18
    s.movz       s16, #16
    s.st.d       [s19, #8], s16
    s.movz       s16, #16
    s.st.d       [s19, #16], s16
    s.movz       s16, #32
    s.st.d       [s19, #48], s16
    s.movz       s16, #2
    s.st.d       [s19, #56], s16
    .li       s16, #68719476752
    s.st.d       [s19, #88], s16
    .li       s16, #8589934592
    s.st.d       [s19, #104], s16
    fence.release.device
    td.load      d2, [s19]                       // a versioned snapshot, not a pointer (ISA sec 9D.1)
    s.movz       s16, #0
    w.def        w2, #40, #8                     // v40..v47
    mb.arm       s18, b3, #0, #1                 // 0 arrival(s), 1 operation(s)
    // a repacked tile: one tile into a window (ISA sec 9E)
    s.mov        s19, s16                        // coordinates must be consecutive (ISA sec 9E)
    s.mov        s20, s16                        // coordinates must be consecutive (ISA sec 9E)
    t.ld.async.h w2, d2, s19, s18
    mb.wait      s18
    mb.arm       s16, b4, #0, #1                 // 0 arrival(s), 1 operation(s)
    // acc + tl.dot(a, b): accumulated in place (ISA sec 9C)
    m.mma.async.k16.f16 a1, v56, v40, s16
    mb.wait      s16
    s.add        s19, s13, s8
    s.mov        s13, s19
    s.movz       s19, #1
    br.nz        s19, for_5
endfor_6:
    s.movz       s3, #0
    s.movz       s5, #0
    s.add        s8, s3, s1                      // the tile origin
    v.iota       v0                              // arange(0, VLEN)
    v.splat      v1, s8
    s.add        s3, s5, s4                      // the tile origin
    s.movz       s5, #128
    s.movz       s8, #256
    s.muli       s9, s1, #2                      // a runtime tile origin, x2
    v.splat      v2, s3
    v.splat      v3, s5
    v.add        v4, v0, v1
    v.splat      v1, s8
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.rdspecial  s5, arg_base
    v.cmp.lt     p1, v2, v3
    v.cmp.lt     p2, v4, v1
    s.add        s10, s2, s9                     // the tile origin
    p.and        p3, p1, p2
    s.ld.d       s9, [s5, #32]                   // descriptor scratch: one slot past the declared arguments (ABI v1 offsets are unchanged)
    s.rdspecial  s5, role_id
    s.movz       s11, #10048
    s.addi       s12, s3, #1
    s.mul        s13, s5, s11
    v.splat      v1, s12
    s.addi       s5, s3, #2
    s.addi       s11, s3, #3
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    v.splat      v1, s5
    v.movi       v2, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v2, [s10, #0]
    s.add        s5, s9, s13                     // this context's own scratch
    v.cmp.lt     p1, v1, v3
    s.addi       s9, s5, #5952
    p.and        p3, p1, p2
    s.addi       s12, s9, #64
    v.splat      v1, s11
    s.addi       s11, s9, #128
    s.addi       s13, s3, #4
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p2
    v.splat      v1, s13
    s.addi       s13, s9, #192
    s.addi       s19, s3, #5
    v.st.w       [s9, #0], v2
    v.splat      v2, s19
    s.addi       s19, s9, #256
    s.addi       s20, s3, #6
    s.addi       s21, s9, #320
    v.splat      v4, s20
    s.addi       s20, s3, #7
    v.splat      v5, s20
    s.addi       s20, s9, #384
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    v.st.w       [s12, #0], v6
    s.addi       s12, s3, #8
    v.splat      v1, s12
    s.addi       s12, s9, #448
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p3, p1, p2
    v.st.w       [s11, #0], v6
    s.addi       s11, s3, #9
    v.splat      v2, s11
    s.addi       s11, s9, #512
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p5, p1, p2
    v.st.w       [s13, #0], v6
    s.addi       s13, s3, #10
    v.splat      v4, s13
    s.addi       s13, s9, #576
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p2
    v.st.w       [s19, #0], v6
    s.addi       s19, s3, #11
    v.splat      v5, s19
    s.addi       s19, s9, #640
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p3, p1, p2
    v.st.w       [s21, #0], v6
    s.addi       s21, s3, #12
    v.splat      v1, s21
    s.addi       s21, s9, #704
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p5, p1, p2
    v.st.w       [s20, #0], v6
    s.addi       s20, s3, #13
    v.splat      v2, s20
    s.addi       s20, s9, #768
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p2
    v.st.w       [s12, #0], v6
    s.addi       s12, s3, #14
    v.splat      v4, s12
    s.addi       s12, s9, #832
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p3, p1, p2
    v.st.w       [s11, #0], v6
    s.addi       s11, s3, #15
    v.splat      v5, s11
    s.addi       s11, s9, #896
    s.addi       s22, s9, #960
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p2
    v.splat      v1, s3
    v.st.w       [s13, #0], v6
    s.movz       s9, #0
    v.splat      v6, s8
    s.add        s8, s9, s7                      // the tile origin
    s.muli       s9, s7, #2                      // a runtime tile origin, x2
    v.splat      v7, s8
    // a masked 16x16 tile with contiguous rows: one predicated load a row
    s.addi       s8, s5, #6976
    v.add        v8, v0, v7
    s.add        s13, s2, s9                     // the tile origin
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v0, [s10, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    v.st.w       [s19, #0], v0
    s.addi       s2, s3, #1
    s.addi       s9, s8, #64
    v.splat      v0, s2
    v.movi       v2, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v2, [s10, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p3, p1, p2
    v.st.w       [s21, #0], v2
    s.addi       s2, s3, #2
    s.addi       s19, s8, #128
    v.splat      v2, s2
    v.movi       v4, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v4, [s10, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p5, p1, p2
    v.st.w       [s20, #0], v4
    s.addi       s2, s3, #3
    s.addi       s20, s8, #192
    v.splat      v4, s2
    v.movi       v5, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v5, [s10, #0]
    v.cmp.lt     p1, v1, v3
    v.st.w       [s12, #0], v5
    s.addi       s2, s3, #4
    s.addi       s12, s8, #256
    v.splat      v1, s2
    v.movi       v5, #0                          // other=, materialised (sec 12)
    @p3 v.ld.h       v5, [s10, #0]
    v.cmp.lt     p3, v8, v6
    p.and        p4, p1, p3
    v.st.w       [s11, #0], v5
    s.addi       s2, s3, #5
    s.addi       s11, s8, #320
    v.splat      v5, s2
    v.movi       v6, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v6, [s10, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p5, p1, p3
    v.st.w       [s22, #0], v6
    fence.acq_rel.device
    s.addi       s2, s3, #6
    s.addi       s10, s3, #7
    v.splat      v0, s2
    v.splat      v6, s10
    s.addi       s2, s8, #384
    s.addi       s10, s8, #448
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p3
    v.st.w       [s8, #0], v7
    s.addi       s21, s3, #8
    v.splat      v2, s21
    s.addi       s21, s8, #512
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p5, p1, p3
    v.st.w       [s9, #0], v7
    s.addi       s9, s3, #9
    v.splat      v4, s9
    s.addi       s9, s8, #576
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p3
    v.st.w       [s19, #0], v7
    s.addi       s19, s3, #10
    v.splat      v1, s19
    s.addi       s19, s8, #640
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p5, p1, p3
    v.st.w       [s20, #0], v7
    s.addi       s20, s3, #11
    v.splat      v5, s20
    s.addi       s20, s8, #704
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p3
    v.st.w       [s12, #0], v7
    s.addi       s12, s3, #12
    v.splat      v0, s12
    s.addi       s12, s8, #768
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p5, p1, p3
    v.st.w       [s11, #0], v7
    s.addi       s11, s3, #13
    v.splat      v6, s11
    s.addi       s11, s8, #832
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p3
    v.st.w       [s2, #0], v7
    s.addi       s2, s3, #14
    v.splat      v2, s2
    s.addi       s2, s8, #896
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p5, p1, p3
    v.st.w       [s10, #0], v7
    s.addi       s10, s3, #15
    s.addi       s22, s8, #960
    v.splat      v4, s10
    v.movi       v7, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v7, [s13, #0]
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p3
    v.st.w       [s21, #0], v7
    s.addi       s8, s5, #5952
    // a computed 16x16 tile, into an accumulator (ISA sec 9B)
    s.movz       s10, #0
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v1, [s13, #0]
    v.cmp.lt     p1, v5, v3
    p.and        p5, p1, p3
    v.st.w       [s9, #0], v1
    s.addi       s9, s5, #6016
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v1, [s13, #0]
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p3
    v.st.w       [s19, #0], v1
    s.movz       s19, #1
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v0, [s13, #0]
    v.cmp.lt     p1, v6, v3
    p.and        p5, p1, p3
    v.st.w       [s20, #0], v0
    s.addi       s20, s5, #6080
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v0, [s13, #0]
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p3
    v.st.w       [s12, #0], v0
    s.movz       s12, #2
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v0, [s13, #0]
    v.cmp.lt     p1, v4, v3
    p.and        p5, p1, p3
    v.st.w       [s11, #0], v0
    s.addi       s11, s5, #6144
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p4 v.ld.h       v0, [s13, #0]
    v.st.w       [s2, #0], v0
    s.movz       s2, #3
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p5 v.ld.h       v0, [s13, #0]
    s.addi       s13, s5, #6208
    v.st.w       [s22, #0], v0
    fence.acq_rel.device
    s.movz       s21, #4
    v.ld.w       v0, [s8, #0]
    s.addi       s8, s5, #6272
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s10, v1
    v.ld.w       v0, [s9, #0]
    s.movz       s9, #5
    s.addi       s10, s5, #6336
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s19, v1
    v.ld.w       v0, [s20, #0]
    s.movz       s19, #6
    s.addi       s20, s5, #6400
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s12, v1
    v.ld.w       v0, [s11, #0]
    s.movz       s11, #7
    s.addi       s12, s5, #6464
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #8
    s.addi       s13, s5, #6528
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s21, v1
    v.ld.w       v0, [s8, #0]
    s.movz       s8, #9
    s.addi       s21, s5, #6592
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #10
    s.addi       s10, s5, #6656
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s19, v1
    v.ld.w       v0, [s20, #0]
    s.movz       s19, #11
    s.addi       s20, s5, #6720
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s11, v1
    v.ld.w       v0, [s12, #0]
    s.movz       s11, #12
    s.addi       s12, s5, #6784
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #13
    s.addi       s13, s5, #6848
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s8, v1
    v.ld.w       v0, [s21, #0]
    s.movz       s8, #14
    s.addi       s21, s5, #6912
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #15
    s.addi       s10, s5, #6976
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s19, v1
    v.ld.w       v0, [s20, #0]
    // a computed 16x16 tile, into an accumulator (ISA sec 9B)
    s.movz       s19, #0
    s.addi       s20, s5, #7040
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s11, v1
    v.ld.w       v0, [s12, #0]
    s.movz       s11, #1
    s.addi       s12, s5, #7104
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #2
    s.addi       s13, s5, #7168
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s8, v1
    v.ld.w       v0, [s21, #0]
    s.movz       s8, #3
    s.addi       s21, s5, #7232
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a2, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #4
    s.addi       s10, s5, #7296
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s19, v1
    v.ld.w       v0, [s20, #0]
    s.movz       s19, #5
    s.addi       s20, s5, #7360
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s11, v1
    v.ld.w       v0, [s12, #0]
    s.movz       s11, #6
    s.addi       s12, s5, #7424
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #7
    s.addi       s13, s5, #7488
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s8, v1
    v.ld.w       v0, [s21, #0]
    s.movz       s8, #8
    s.addi       s21, s5, #7552
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #9
    s.addi       s10, s5, #7616
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s19, v1
    v.ld.w       v0, [s20, #0]
    s.movz       s19, #10
    s.addi       s20, s5, #7680
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s11, v1
    v.ld.w       v0, [s12, #0]
    s.movz       s11, #11
    s.addi       s12, s5, #7744
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #12
    s.addi       s13, s5, #7808
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s8, v1
    v.ld.w       v0, [s21, #0]
    s.movz       s8, #13
    s.addi       s21, s5, #7872
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #14
    s.addi       s10, s5, #7936
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s19, v1
    v.ld.w       v0, [s20, #0]
    s.movz       s19, #15
    s.movz       s20, #0
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s11, v1
    v.ld.w       v0, [s12, #0]
    s.movz       s11, #0
    // a computed 16x16 tile, into an accumulator (ISA sec 9B)
    s.movz       s12, #0
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s2, v1
    v.ld.w       v0, [s13, #0]
    s.movz       s2, #1
    s.movz       s13, #1
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s8, v1
    v.ld.w       v0, [s21, #0]
    s.movz       s8, #1
    s.movz       s21, #2
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s9, v1
    v.ld.w       v0, [s10, #0]
    s.movz       s9, #2
    s.movz       s10, #2
    vf.cvt.f16.f32 v1, v0
    a.vwrite     a3, s19, v1
    a.vread      v0, a0, s20
    a.vread      v1, a2, s11
    s.movz       s11, #3
    s.movz       s19, #3
    vf.add       v2, v0, v1
    s.movz       s20, #3
    a.vwrite     a4, s12, v2
    a.vread      v0, a0, s2
    a.vread      v1, a2, s13
    s.movz       s2, #4
    s.movz       s12, #4
    vf.add       v2, v0, v1
    s.movz       s13, #4
    a.vwrite     a4, s8, v2
    a.vread      v0, a0, s21
    a.vread      v1, a2, s9
    s.movz       s8, #5
    s.movz       s9, #5
    vf.add       v2, v0, v1
    s.movz       s21, #5
    a.vwrite     a4, s10, v2
    a.vread      v0, a0, s11
    a.vread      v1, a2, s19
    s.movz       s10, #6
    s.movz       s11, #6
    vf.add       v2, v0, v1
    s.movz       s19, #6
    a.vwrite     a4, s20, v2
    a.vread      v0, a0, s2
    a.vread      v1, a2, s12
    s.movz       s2, #7
    s.movz       s12, #7
    vf.add       v2, v0, v1
    s.movz       s20, #7
    a.vwrite     a4, s13, v2
    a.vread      v0, a0, s8
    a.vread      v1, a2, s9
    s.movz       s8, #8
    s.movz       s9, #8
    vf.add       v2, v0, v1
    s.movz       s13, #8
    a.vwrite     a4, s21, v2
    a.vread      v0, a0, s10
    a.vread      v1, a2, s11
    s.movz       s10, #9
    s.movz       s11, #9
    vf.add       v2, v0, v1
    s.movz       s21, #9
    a.vwrite     a4, s19, v2
    a.vread      v0, a0, s2
    a.vread      v1, a2, s12
    s.movz       s2, #10
    s.movz       s12, #10
    vf.add       v2, v0, v1
    s.movz       s19, #10
    a.vwrite     a4, s20, v2
    a.vread      v0, a0, s8
    a.vread      v1, a2, s9
    s.movz       s8, #11
    s.movz       s9, #11
    vf.add       v2, v0, v1
    s.movz       s20, #11
    a.vwrite     a4, s13, v2
    a.vread      v0, a0, s10
    a.vread      v1, a2, s11
    s.movz       s10, #12
    s.movz       s11, #12
    vf.add       v2, v0, v1
    s.movz       s13, #12
    a.vwrite     a4, s21, v2
    a.vread      v0, a0, s2
    a.vread      v1, a2, s12
    s.movz       s2, #13
    s.movz       s12, #13
    vf.add       v2, v0, v1
    s.movz       s21, #13
    a.vwrite     a4, s19, v2
    a.vread      v0, a0, s8
    a.vread      v1, a2, s9
    s.movz       s8, #14
    s.movz       s9, #14
    vf.add       v2, v0, v1
    s.movz       s19, #14
    a.vwrite     a4, s20, v2
    a.vread      v0, a0, s10
    a.vread      v1, a2, s11
    s.movz       s10, #15
    s.movz       s11, #15
    vf.add       v2, v0, v1
    s.movz       s20, #15
    a.vwrite     a4, s13, v2
    a.vread      v0, a0, s2
    a.vread      v1, a2, s12
    s.movz       s2, #0
    s.movz       s12, #0
    vf.add       v2, v0, v1
    // a computed 16x16 tile, into an accumulator (ISA sec 9B)
    s.movz       s13, #0
    a.vwrite     a4, s21, v2
    a.vread      v0, a0, s8
    a.vread      v1, a2, s9
    s.movz       s8, #1
    s.movz       s9, #1
    vf.add       v2, v0, v1
    s.movz       s21, #1
    a.vwrite     a4, s19, v2
    a.vread      v0, a0, s10
    a.vread      v1, a2, s11
    s.movz       s10, #2
    s.movz       s11, #2
    vf.add       v2, v0, v1
    s.movz       s19, #2
    a.vwrite     a4, s20, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #3
    s.movz       s12, #3
    vf.add       v2, v0, v1
    s.movz       s20, #3
    a.vwrite     a5, s13, v2
    a.vread      v0, a1, s8
    a.vread      v1, a3, s9
    s.movz       s8, #4
    s.movz       s9, #4
    vf.add       v2, v0, v1
    s.movz       s13, #4
    a.vwrite     a5, s21, v2
    a.vread      v0, a1, s10
    a.vread      v1, a3, s11
    s.movz       s10, #5
    s.movz       s11, #5
    vf.add       v2, v0, v1
    s.movz       s21, #5
    a.vwrite     a5, s19, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #6
    s.movz       s12, #6
    vf.add       v2, v0, v1
    s.movz       s19, #6
    a.vwrite     a5, s20, v2
    a.vread      v0, a1, s8
    a.vread      v1, a3, s9
    s.movz       s8, #7
    s.movz       s9, #7
    vf.add       v2, v0, v1
    s.movz       s20, #7
    a.vwrite     a5, s13, v2
    a.vread      v0, a1, s10
    a.vread      v1, a3, s11
    s.movz       s10, #8
    s.movz       s11, #8
    vf.add       v2, v0, v1
    s.movz       s13, #8
    a.vwrite     a5, s21, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #9
    s.movz       s12, #9
    vf.add       v2, v0, v1
    s.movz       s21, #9
    a.vwrite     a5, s19, v2
    a.vread      v0, a1, s8
    a.vread      v1, a3, s9
    s.movz       s8, #10
    s.movz       s9, #10
    vf.add       v2, v0, v1
    s.movz       s19, #10
    a.vwrite     a5, s20, v2
    a.vread      v0, a1, s10
    a.vread      v1, a3, s11
    s.movz       s10, #11
    s.movz       s11, #11
    vf.add       v2, v0, v1
    s.movz       s20, #11
    a.vwrite     a5, s13, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #12
    s.movz       s12, #12
    vf.add       v2, v0, v1
    s.movz       s13, #12
    a.vwrite     a5, s21, v2
    a.vread      v0, a1, s8
    a.vread      v1, a3, s9
    s.movz       s8, #13
    s.movz       s9, #13
    vf.add       v2, v0, v1
    s.movz       s21, #13
    a.vwrite     a5, s19, v2
    a.vread      v0, a1, s10
    a.vread      v1, a3, s11
    s.movz       s10, #14
    s.movz       s11, #14
    vf.add       v2, v0, v1
    s.movz       s19, #14
    a.vwrite     a5, s20, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #15
    s.movz       s12, #15
    vf.add       v2, v0, v1
    s.movz       s20, #15
    a.vwrite     a5, s13, v2
    a.vread      v0, a1, s8
    a.vread      v1, a3, s9
    s.movz       s8, #0
    s.addi       s9, s5, #8000
    vf.add       v2, v0, v1
    s.movz       s13, #1
    a.vwrite     a5, s21, v2
    a.vread      v0, a1, s10
    a.vread      v1, a3, s11
    s.addi       s10, s9, #64
    s.movz       s11, #2
    vf.add       v2, v0, v1
    s.addi       s21, s9, #128
    a.vwrite     a5, s19, v2
    a.vread      v0, a1, s2
    a.vread      v1, a3, s12
    s.movz       s2, #3
    s.addi       s12, s9, #192
    vf.add       v2, v0, v1
    s.movz       s19, #4
    a.vwrite     a5, s20, v2
    a.vread      v0, a4, s8
    s.addi       s8, s9, #256
    s.movz       s20, #5
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s9, #0], v1
    a.vread      v0, a4, s13
    s.addi       s13, s9, #320
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s10, #0], v1
    a.vread      v0, a4, s11
    s.movz       s10, #6
    s.addi       s11, s9, #384
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s21, #0], v1
    a.vread      v0, a4, s2
    s.movz       s2, #7
    s.addi       s21, s9, #448
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s12, #0], v1
    a.vread      v0, a4, s19
    s.movz       s12, #8
    s.addi       s19, s9, #512
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s8, #0], v1
    a.vread      v0, a4, s20
    s.movz       s8, #9
    s.addi       s20, s9, #576
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s13, #0], v1
    a.vread      v0, a4, s10
    s.movz       s10, #10
    s.addi       s13, s9, #640
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s11, #0], v1
    a.vread      v0, a4, s2
    s.movz       s2, #11
    s.addi       s11, s9, #704
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s21, #0], v1
    a.vread      v0, a4, s12
    s.movz       s12, #12
    s.addi       s21, s9, #768
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s19, #0], v1
    a.vread      v0, a4, s8
    s.movz       s8, #13
    s.addi       s19, s9, #832
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s20, #0], v1
    a.vread      v0, a4, s10
    s.movz       s10, #14
    s.addi       s20, s9, #896
    vf.cvt.f32.f16 v1, v0
    s.addi       s22, s9, #960
    v.st.w       [s13, #0], v1
    a.vread      v0, a4, s2
    s.movz       s2, #15
    s.movz       s9, #0
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s11, #0], v1
    a.vread      v0, a4, s12
    s.addi       s11, s5, #9024
    s.movz       s12, #1
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s21, #0], v1
    a.vread      v0, a4, s8
    s.addi       s8, s11, #64
    s.movz       s13, #2
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s19, #0], v1
    a.vread      v0, a4, s10
    s.addi       s10, s11, #128
    s.movz       s19, #3
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s20, #0], v1
    a.vread      v0, a4, s2
    s.addi       s2, s11, #192
    s.movz       s20, #4
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s22, #0], v1
    fence.acq_rel.device
    a.vread      v0, a5, s9
    s.addi       s9, s11, #256
    s.movz       s21, #5
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s11, #0], v1
    a.vread      v0, a5, s12
    s.addi       s12, s11, #320
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s8, #0], v1
    a.vread      v0, a5, s13
    s.movz       s8, #6
    s.addi       s13, s11, #384
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s10, #0], v1
    a.vread      v0, a5, s19
    s.movz       s10, #7
    s.addi       s19, s11, #448
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s2, #0], v1
    a.vread      v0, a5, s20
    s.movz       s2, #8
    s.addi       s20, s11, #512
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s9, #0], v1
    a.vread      v0, a5, s21
    s.movz       s9, #9
    s.addi       s21, s11, #576
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s12, #0], v1
    a.vread      v0, a5, s8
    s.movz       s8, #10
    s.addi       s12, s11, #640
    vf.cvt.f32.f16 v1, v0
    s.muli       s22, s1, #2                     // a runtime tile origin, x2
    v.st.w       [s13, #0], v1
    a.vread      v0, a5, s10
    s.movz       s1, #11
    s.addi       s10, s11, #704
    vf.cvt.f32.f16 v1, v0
    s.add        s13, s6, s22                    // the tile origin
    v.st.w       [s19, #0], v1
    a.vread      v0, a5, s2
    s.movz       s2, #12
    s.addi       s19, s11, #768
    vf.cvt.f32.f16 v1, v0
    v.splat      v0, s3
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    v.st.w       [s20, #0], v1
    a.vread      v0, a5, s9
    s.movz       s9, #13
    s.addi       s20, s11, #832
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s21, #0], v1
    a.vread      v0, a5, s8
    s.movz       s8, #14
    s.addi       s21, s11, #896
    vf.cvt.f32.f16 v1, v0
    s.addi       s22, s11, #960
    v.st.w       [s12, #0], v1
    a.vread      v0, a5, s1
    s.movz       s1, #15
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.addi       s11, s5, #8000
    vf.cvt.f32.f16 v1, v0
    v.st.w       [s10, #0], v1
    a.vread      v0, a5, s2
    s.muli       s2, s4, #512                    // a runtime tile origin, x512
    s.addi       s10, s5, #8064
    vf.cvt.f32.f16 v1, v0
    s.add        s12, s13, s2                    // the tile origin
    s.addi       s2, s3, #1
    v.splat      v0, s2
    s.addi       s2, s12, #512                   // row 1
    v.cmp.lt     p1, v0, v3
    p.and        p5, p1, p2
    v.st.w       [s19, #0], v1
    a.vread      v0, a5, s9
    s.addi       s9, s5, #8128
    s.addi       s13, s3, #2
    vf.cvt.f32.f16 v1, v0
    v.splat      v0, s13
    s.addi       s13, s12, #1024                 // row 2
    v.st.w       [s20, #0], v1
    a.vread      v1, a5, s8
    s.addi       s8, s5, #8192
    s.addi       s19, s3, #3
    vf.cvt.f32.f16 v2, v1
    v.splat      v1, s19
    s.addi       s19, s12, #1536                 // row 3
    v.st.w       [s21, #0], v2
    a.vread      v2, a5, s1
    s.addi       s1, s5, #8256
    s.addi       s20, s3, #4
    vf.cvt.f32.f16 v4, v2
    v.splat      v2, s20
    s.addi       s20, s12, #2048                 // row 4
    v.st.w       [s22, #0], v4
    fence.acq_rel.device
    s.addi       s21, s5, #8320
    v.ld.w       v4, [s11, #0]
    s.addi       s11, s3, #5
    v.splat      v5, s11
    s.addi       s11, s12, #2560                 // row 5
    @p4 v.st.h       [s12, #0], v4
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    v.ld.w       v0, [s10, #0]
    s.addi       s10, s5, #8384
    @p5 v.st.h       [s2, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p2
    s.addi       s2, s3, #6
    v.splat      v0, s2
    s.addi       s2, s12, #3072                  // row 6
    v.ld.w       v1, [s9, #0]
    s.addi       s9, s5, #8448
    @p4 v.st.h       [s13, #0], v1
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s13, s3, #7
    v.splat      v1, s13
    s.addi       s13, s12, #3584                 // row 7
    v.ld.w       v2, [s8, #0]
    s.addi       s8, s5, #8512
    @p5 v.st.h       [s19, #0], v2
    v.cmp.lt     p1, v5, v3
    p.and        p5, p1, p2
    s.addi       s19, s3, #8
    v.splat      v2, s19
    s.addi       s19, s12, #4096                 // row 8
    v.ld.w       v4, [s1, #0]
    s.addi       s1, s5, #8576
    @p4 v.st.h       [s20, #0], v4
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    s.addi       s20, s3, #9
    v.splat      v0, s20
    s.addi       s20, s12, #4608                 // row 9
    v.ld.w       v4, [s21, #0]
    s.addi       s21, s5, #8640
    @p5 v.st.h       [s11, #0], v4
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p2
    s.addi       s11, s3, #10
    v.splat      v1, s11
    s.addi       s11, s12, #5120                 // row 10
    v.ld.w       v4, [s10, #0]
    s.addi       s10, s5, #8704
    @p4 v.st.h       [s2, #0], v4
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    s.addi       s2, s3, #11
    v.splat      v2, s2
    s.addi       s2, s12, #5632                  // row 11
    v.ld.w       v4, [s9, #0]
    s.addi       s9, s5, #8768
    @p5 v.st.h       [s13, #0], v4
    v.cmp.lt     p1, v0, v3
    p.and        p5, p1, p2
    s.addi       s13, s3, #12
    v.splat      v0, s13
    s.addi       s13, s12, #6144                 // row 12
    v.ld.w       v4, [s8, #0]
    s.addi       s8, s5, #8832
    @p4 v.st.h       [s19, #0], v4
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p2
    s.addi       s19, s3, #13
    v.splat      v1, s19
    s.addi       s19, s12, #6656                 // row 13
    v.ld.w       v4, [s1, #0]
    s.addi       s1, s5, #8896
    @p5 v.st.h       [s20, #0], v4
    v.cmp.lt     p1, v2, v3
    p.and        p5, p1, p2
    s.addi       s20, s3, #14
    s.muli       s22, s7, #2                     // a runtime tile origin, x2
    v.splat      v2, s20
    s.addi       s7, s12, #7168                  // row 14
    s.addi       s20, s12, #7680                 // row 15
    v.ld.w       v4, [s21, #0]
    s.addi       s12, s5, #8960
    s.muli       s21, s4, #512                   // a runtime tile origin, x512
    v.splat      v5, s3
    s.add        s4, s6, s22                     // the tile origin
    s.addi       s6, s3, #15
    v.splat      v6, s6
    // a masked 16x16 tile with contiguous rows: one predicated store a row
    s.addi       s6, s5, #9024
    s.add        s22, s4, s21                    // the tile origin
    @p4 v.st.h       [s11, #0], v4
    v.cmp.lt     p1, v0, v3
    p.and        p4, p1, p2
    v.ld.w       v0, [s10, #0]
    s.addi       s4, s5, #9088
    s.addi       s10, s3, #1
    s.addi       s11, s22, #512                  // row 1
    v.splat      v4, s10
    @p5 v.st.h       [s2, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p2
    v.ld.w       v0, [s9, #0]
    s.addi       s2, s5, #9152
    s.addi       s9, s3, #2
    s.addi       s10, s22, #1024                 // row 2
    v.splat      v1, s9
    @p4 v.st.h       [s13, #0], v0
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p2
    v.ld.w       v0, [s8, #0]
    s.addi       s8, s5, #9216
    s.addi       s9, s3, #3
    s.addi       s13, s22, #1536                 // row 3
    v.splat      v2, s9
    @p5 v.st.h       [s19, #0], v0
    v.cmp.lt     p1, v6, v3
    p.and        p5, p1, p2
    v.cmp.lt     p1, v5, v3
    p.and        p2, p1, p3
    v.ld.w       v0, [s1, #0]
    s.addi       s1, s5, #9280
    s.addi       s9, s3, #4
    s.addi       s19, s22, #2048                 // row 4
    v.splat      v5, s9
    @p4 v.st.h       [s7, #0], v0
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p3
    v.ld.w       v0, [s12, #0]
    s.addi       s7, s5, #9344
    s.addi       s9, s3, #5
    s.addi       s12, s22, #2560                 // row 5
    v.splat      v4, s9
    @p5 v.st.h       [s20, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p3
    s.addi       s9, s5, #9408
    s.addi       s20, s22, #3072                 // row 6
    v.ld.w       v0, [s6, #0]
    s.addi       s6, s3, #6
    v.splat      v1, s6
    @p2 v.st.h       [s22, #0], v0
    v.cmp.lt     p1, v2, v3
    p.and        p2, p1, p3
    s.addi       s6, s5, #9472
    v.ld.w       v0, [s4, #0]
    s.addi       s4, s3, #7
    v.splat      v2, s4
    s.addi       s4, s22, #3584                  // row 7
    @p4 v.st.h       [s11, #0], v0
    v.cmp.lt     p1, v5, v3
    p.and        p4, p1, p3
    s.addi       s11, s5, #9536
    v.ld.w       v0, [s2, #0]
    s.addi       s2, s3, #8
    v.splat      v5, s2
    s.addi       s2, s22, #4096                  // row 8
    @p5 v.st.h       [s10, #0], v0
    v.cmp.lt     p1, v4, v3
    p.and        p5, p1, p3
    s.addi       s10, s5, #9600
    v.ld.w       v0, [s8, #0]
    s.addi       s8, s3, #9
    v.splat      v4, s8
    s.addi       s8, s22, #4608                  // row 9
    @p2 v.st.h       [s13, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p2, p1, p3
    s.addi       s13, s5, #9664
    v.ld.w       v0, [s1, #0]
    s.addi       s1, s3, #10
    v.splat      v1, s1
    s.addi       s1, s22, #5120                  // row 10
    @p4 v.st.h       [s19, #0], v0
    v.cmp.lt     p1, v2, v3
    p.and        p4, p1, p3
    s.addi       s19, s5, #9728
    v.ld.w       v0, [s7, #0]
    s.addi       s7, s3, #11
    v.splat      v2, s7
    s.addi       s7, s22, #5632                  // row 11
    @p5 v.st.h       [s12, #0], v0
    v.cmp.lt     p1, v5, v3
    p.and        p5, p1, p3
    s.addi       s12, s5, #9792
    v.ld.w       v0, [s9, #0]
    s.addi       s9, s3, #12
    v.splat      v5, s9
    s.addi       s9, s22, #6144                  // row 12
    @p2 v.st.h       [s20, #0], v0
    v.cmp.lt     p1, v4, v3
    p.and        p2, p1, p3
    s.addi       s20, s5, #9856
    v.ld.w       v0, [s6, #0]
    s.addi       s6, s3, #13
    v.splat      v4, s6
    s.addi       s6, s22, #6656                  // row 13
    @p4 v.st.h       [s4, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p4, p1, p3
    s.addi       s4, s5, #9920
    v.ld.w       v0, [s11, #0]
    s.addi       s11, s3, #14
    v.splat      v1, s11
    s.addi       s11, s22, #7168                 // row 14
    @p5 v.st.h       [s2, #0], v0
    v.cmp.lt     p1, v2, v3
    p.and        p5, p1, p3
    s.addi       s2, s5, #9984
    v.ld.w       v0, [s10, #0]
    s.addi       s5, s3, #15
    s.addi       s3, s22, #7680                  // row 15
    v.splat      v2, s5
    @p2 v.st.h       [s8, #0], v0
    v.cmp.lt     p1, v5, v3
    p.and        p2, p1, p3
    v.ld.w       v0, [s13, #0]
    @p4 v.st.h       [s1, #0], v0
    v.cmp.lt     p1, v4, v3
    p.and        p4, p1, p3
    v.ld.w       v0, [s19, #0]
    @p5 v.st.h       [s7, #0], v0
    v.cmp.lt     p1, v1, v3
    p.and        p5, p1, p3
    v.ld.w       v0, [s12, #0]
    @p2 v.st.h       [s9, #0], v0
    v.cmp.lt     p1, v2, v3
    p.and        p2, p1, p3
    v.ld.w       v0, [s20, #0]
    @p4 v.st.h       [s6, #0], v0
    v.ld.w       v0, [s4, #0]
    @p5 v.st.h       [s11, #0], v0
    v.ld.w       v0, [s2, #0]
    @p2 v.st.h       [s3, #0], v0
__rctx_next:
    exit
