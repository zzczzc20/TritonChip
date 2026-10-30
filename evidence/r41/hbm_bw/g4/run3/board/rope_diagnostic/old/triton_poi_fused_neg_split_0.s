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
    v.movi       v0, #0
    v.sub        v1, v2, v3
    v.sub        v4, v0, v3
    v.cmp.lt     p1, v3, v2
    v.cmp.lt     p2, v3, v0
    @p1 v.sel        v5, v1, v3
    v.movi       v1, #7
    @p2 v.sel        v6, v4, v3
    v.movi       v4, #3
    v.and        v7, v5, v1
    v.lshr       v1, v6, v4
    v.sub        v4, v2, v7
    v.sub        v5, v0, v1
    v.movi       v6, #8
    @p1 v.sel        v8, v4, v7
    @p2 v.sel        v4, v5, v1
    v.movi       v1, #16
    s.rdspecial  s1, arg_base
    v.add        v5, v8, v6
    v.mul        v7, v4, v1
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    v.movi       v1, #256
    v.add        v4, v5, v7
    v.cmp.lt     p1, v3, v1
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p1 v.ld.gather.h v1, [s2, v4]               // index is in elements (ISA sec 9.1)
    v.sub        v3, v2, v6
    v.sub        v4, v0, v6
    v.add        v5, v8, v7
    v.cmp.lt     p2, v6, v2
    v.cmp.lt     p3, v6, v0
    vf.cvt.f16.f32 v0, v1
    s.ld.d       s2, [s1, #8]                    // out_ptr0: device_ptr64 at arg_base+8
    @p2 v.sel        v1, v3, v6
    @p3 v.sel        v1, v4, v6
    vf.neg       v1, v0
    vf.cvt.f32.f16 v0, v1
    @p1 v.st.scatter.h [s2, v5], v0
__rctx_next:
    exit
