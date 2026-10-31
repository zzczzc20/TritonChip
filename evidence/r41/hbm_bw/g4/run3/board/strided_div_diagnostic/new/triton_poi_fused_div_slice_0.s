    s.rdspecial  s1, role_id
    s.rdspecial  s2, pid.x
    s.rdspecial  s3, npid.x
    s.add        s4, s2, s1
    s.cmp.ge     s1, s4, s3
    br.nz        s1, __rctx_next
__rctx_program:
    s.rdspecial  s1, arg_base
    s.shli       s2, s4, #4                      // x16 as a shift
    s.ld.d       s3, [s1, #8]                    // in_ptr1: device_ptr64 at arg_base+8
    v.iota       v0                              // arange(0, VLEN)
    v.splat      v1, s2
    v.movi       v2, #2
    .li       s4, #1065353216
    v.splat      v3, s4                          // a 32-bit constant does not fit v.movi (ISA 0x40)
    v.add        v4, v1, v0
    s.shli       s4, s2, #1                      // element offset -> bytes (x2)
    v.mul        v0, v4, v2
    v.ld.gather.h v1, [s3, v0]                   // index is in elements (ISA sec 9.1)
    vf.cvt.bf16.f32 v2, v1
    s.ld.d       s2, [s1, #0]                    // in_ptr0: device_ptr64 at arg_base+0
    v.rcp.approx v1, v2                          // ISA 8.8: <= 2**-22 relative
    v.ld.gather.h v4, [s2, v0]                   // index is in elements (ISA sec 9.1)
    vf.cvt.bf16.f32 v0, v4
    s.ld.d       s2, [s1, #16]                   // out_ptr0: device_ptr64 at arg_base+16
    vf.fms       v4, v2, v1, v3
    vf.fms       v2, v1, v4, v1
    s.add        s1, s2, s4                      // &out_ptr0[...]
    vf.neg       v3, v2                          // Newton step: r0*(2 - x*r0)
    vf.cmp.o     p1, v3, v3
    @p1 v.sel        v2, v3, v1
    vf.mul       v1, v0, v2                      // a/y as a refined reciprocal (ISA has no divide)
    vf.cvt.f32.bf16 v0, v1
    v.st.h       [s1, #0], v0
__rctx_next:
    exit
