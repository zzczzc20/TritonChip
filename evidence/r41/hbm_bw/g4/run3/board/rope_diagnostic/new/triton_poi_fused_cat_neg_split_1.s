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
    v.movi       v0, #3
    v.movi       v1, #7
    v.sub        v5, v2, v4
    v.cmp.lt     p1, v4, v2
    v.sub        v6, v3, v4
    @p1 v.sel        v7, v5, v4
    v.cmp.lt     p2, v4, v3
    @p2 v.sel        v5, v6, v4
    v.lshr       v6, v7, v0
    s.rdspecial  s1, arg_base
    v.and        v0, v5, v1
    v.sub        v1, v2, v6
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    v.sub        v5, v3, v0
    @p1 v.sel        v7, v1, v6
    v.movi       v1, #16
    @p2 v.sel        v6, v5, v0
    v.movi       v0, #256
    v.mul        v5, v7, v1
    v.movi       v1, #8
    v.cmp.lt     p1, v4, v0
    v.add        v0, v6, v5
    v.sub        v4, v3, v1
    s.ld.d       s3, [s1, #8]                    // out_ptr0: device_ptr64 at arg_base+8
    v.sub        v5, v2, v1
    v.cmp.lt     p2, v1, v3
    v.cmp.lt     p3, v1, v2
    @p2 v.sel        v2, v4, v1
    @p3 v.sel        v2, v5, v1
    v.movi       v1, #0                          // other=, materialised (sec 12)
    @p1 v.ld.gather.h v1, [s2, v0]               // index is in elements (ISA sec 9.1)
    @p1 v.st.scatter.h [s3, v0], v1
__rctx_next:
    exit
