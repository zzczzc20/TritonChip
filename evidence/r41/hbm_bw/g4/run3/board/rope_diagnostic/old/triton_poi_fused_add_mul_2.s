    s.rdspecial  s1, role_id
    s.rdspecial  s2, pid.x
    s.rdspecial  s3, npid.x
    s.add        s4, s2, s1
    s.cmp.ge     s1, s4, s3
    br.nz        s1, __rctx_next
__rctx_program:
    s.shli       s1, s4, #4                      // x16 as a shift
    v.iota       v0                              // arange(0, VLEN)
    v.splat      v1, s1
    v.movi       v2, #0
    v.add        v3, v1, v0
    v.movi       v0, #255
    v.sub        v1, v2, v3
    v.cmp.lt     p1, v3, v2
    @p1 v.sel        v4, v1, v3
    s.rdspecial  s2, arg_base
    v.and        v1, v4, v0
    s.ld.d       s3, [s2, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    v.sub        v0, v2, v1
    s.shli       s4, s1, #1                      // element offset -> bytes (x2)
    s.ld.d       s1, [s2, #16]                   // in_ptr2: device_ptr64 at arg_base+16
    v.movi       v4, #512
    @p1 v.sel        v5, v0, v1
    s.ld.d       s5, [s2, #8]                    // in_ptr1: device_ptr64 at arg_base+8
    s.add        s6, s3, s4                      // &in_ptr0[...]
    s.add        s3, s1, s4                      // &in_ptr2[...]
    s.ld.d       s1, [s2, #24]                   // in_ptr3: device_ptr64 at arg_base+24
    v.cmp.lt     p1, v3, v4
    v.movi       v0, #256
    @p1 v.ld.h       v1, [s6, #0]
    v.sub        v3, v2, v0
    v.cmp.lt     p2, v0, v2
    @p2 v.sel        v2, v3, v0
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p1 v.ld.gather.h v0, [s5, v5]               // index is in elements (ISA sec 9.1)
    vf.cvt.f16.f32 v2, v1
    @p1 v.ld.h       v1, [s3, #0]
    vf.cvt.f16.f32 v3, v0
    vf.mul       v0, v2, v3
    v.movi       v2, #0                          // other=, materialised (sec 12)
    @p1 v.ld.gather.h v2, [s1, v5]               // index is in elements (ISA sec 9.1)
    vf.cvt.f16.f32 v3, v1
    vf.cvt.f16.f32 v1, v2
    s.ld.d       s1, [s2, #32]                   // out_ptr0: device_ptr64 at arg_base+32
    vf.mul       v2, v3, v1
    s.add        s2, s1, s4                      // &out_ptr0[...]
    vf.add       v1, v0, v2
    vf.cvt.f32.f16 v0, v1
    @p1 v.st.h       [s2, #0], v0
__rctx_next:
    exit
