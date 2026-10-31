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
    v.movi       v3, #0
    v.add        v4, v1, v0
    v.movi       v0, #7
    v.movi       v1, #3
    v.sub        v5, v2, v4
    v.sub        v6, v3, v4
    v.cmp.lt     p1, v4, v2
    v.cmp.lt     p2, v4, v3
    @p1 v.sel        v7, v5, v4
    @p2 v.sel        v5, v6, v4
    s.rdspecial  s1, arg_base
    v.and        v6, v7, v0
    v.lshr       v0, v5, v1
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    v.sub        v1, v2, v6
    v.sub        v5, v3, v0
    v.movi       v7, #8
    @p1 v.sel        v8, v1, v6
    @p2 v.sel        v1, v5, v0
    v.movi       v0, #16
    v.add        v5, v8, v7
    v.movi       v6, #256
    v.mul        v9, v1, v0
    v.sub        v0, v2, v7
    v.cmp.lt     p1, v4, v6
    v.add        v1, v5, v9
    s.ld.d       s3, [s1, #8]                    // out_ptr0: device_ptr64 at arg_base+8
    v.sub        v4, v3, v7
    v.add        v5, v8, v9
    v.cmp.lt     p2, v7, v2
    v.cmp.lt     p3, v7, v3
    @p2 v.sel        v2, v0, v7
    @p3 v.sel        v0, v4, v7
    v.movi       v0, #0                          // other=, materialised (sec 12)
    @p1 v.ld.gather.h v0, [s2, v1]               // index is in elements (ISA sec 9.1)
    vf.cvt.f16.f32 v1, v0
    vf.neg       v0, v1
    vf.cvt.f32.f16 v1, v0
    @p1 v.st.scatter.h [s3, v5], v1
__rctx_next:
    exit
