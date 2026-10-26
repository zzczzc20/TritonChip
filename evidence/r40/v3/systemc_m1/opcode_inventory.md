Scanned 228 (point, config) runs, 1235 kernel images, 836008 instruction words.

| source | (point, config) runs | every word executable before M1 | ... and RTL-exact before M1 | every word implemented now |
|---|---:|---:|---:|---:|
| A | 113 | 40 | 18 | 113 |
| B | 115 | 29 | 14 | 115 |

| opcode | major.func | static | kernels | points | before M1 | semantics | timing |
|---|---|---:|---:|---:|---|---|---|
| `s.movz` | 0x01.0x40 | 84933 | 744 | 107 | implemented | implemented | issue+scalar_latency |
| `s.addi` | 0x01.0x20 | 62613 | 657 | 89 | implemented | implemented | issue+scalar_latency |
| `v.extract` | 0x14.0x08 | 62030 | 360 | 61 | implemented | implemented | issue+shuffle_latency |
| `v.splat` | 0x10.0x1a | 59702 | 898 | 113 | implemented | implemented | issue+vector_int_latency |
| `v.movi` | 0x10.0x40 | 43384 | 730 | 113 | implemented | implemented | issue+vector_int_latency |
| `v.cmp` | 0x10.0x18 | 37846 | 871 | 113 | implemented | implemented | issue+compare_latency |
| `a.vread` | 0x41.0x00 | 35568 | 342 | 58 | refused | implemented | issue+matrix_engine_free+acc_read_latency |
| `v.insert` | 0x14.0x09 | 30292 | 133 | 34 | implemented | implemented | issue+shuffle_latency |
| `v.st` | 0x20.0x18 | 28670 | 770 | 109 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `v.ld` | 0x20.0x10 | 27185 | 640 | 96 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `v.add` | 0x10.0x01 | 24183 | 898 | 113 | implemented | implemented | issue+vector_int_latency |
| `v.sel` | 0x10.0x19 | 23897 | 606 | 113 | implemented | implemented | issue+vector_int_latency |
| `vf.fma` | 0x11.0x06 | 23042 | 180 | 57 | implemented | implemented | issue+vector_fp_latency |
| `s.movk` | 0x01.0x41 | 20961 | 671 | 94 | implemented | implemented | issue+scalar_latency |
| `p.and` | 0x13.0x01 | 20326 | 605 | 74 | implemented | implemented | issue+predicate_latency |
| `v.sub` | 0x10.0x02 | 18813 | 331 | 113 | implemented | implemented | issue+vector_int_latency |
| `s.shli` | 0x01.0x2c | 17062 | 871 | 113 | implemented | implemented | issue+scalar_latency |
| `s.add` | 0x01.0x01 | 16100 | 898 | 113 | implemented | implemented | issue+scalar_latency |
| `s.lshri` | 0x01.0x2d | 14676 | 453 | 63 | implemented | implemented | issue+scalar_latency |
| `s.or` | 0x01.0x09 | 12652 | 425 | 54 | implemented | implemented | issue+scalar_latency |
| `vf.cvt` | 0x12.0x04 | 11975 | 644 | 85 | implemented | implemented | issue+convert_latency |
| `s.cmp` | 0x01.0x18 | 11082 | 898 | 113 | implemented | implemented | issue+scalar_latency |
| `vf.mul` | 0x11.0x03 | 9823 | 395 | 62 | implemented | implemented | issue+vector_fp_latency |
| `s.sub` | 0x01.0x02 | 9609 | 336 | 57 | implemented | implemented | issue+scalar_latency |
| `a.vwrite` | 0x41.0x01 | 9520 | 146 | 46 | refused | implemented | issue+matrix_engine_occupancy |
| `vf.add` | 0x11.0x01 | 8079 | 494 | 90 | implemented | implemented | issue+vector_fp_latency |
| `s.st` | 0x20.0x08 | 7553 | 483 | 60 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `s.and` | 0x01.0x08 | 7430 | 306 | 45 | implemented | implemented | issue+scalar_latency |
| `v.ld.gather` | 0x20.0x20 | 7182 | 419 | 113 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `s.rdspecial` | 0x00.0x01 | 6675 | 898 | 113 | implemented | implemented | issue+scalar_latency |
| `s.mov` | 0x01.0x00 | 5760 | 611 | 78 | implemented | implemented | issue+scalar_latency |
| `v.mul` | 0x10.0x03 | 5161 | 336 | 113 | implemented | implemented | issue+vector_int_latency |
| `v.iota` | 0x10.0x1b | 5106 | 898 | 113 | implemented | implemented | issue+vector_int_latency |
| `br.nz` | 0x02.0x01 | 5016 | 898 | 113 | implemented | implemented | issue |
| `s.ld` | 0x20.0x00 | 4898 | 898 | 113 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `s.xor` | 0x01.0x0a | 4794 | 336 | 57 | implemented | implemented | issue+scalar_latency |
| `s.muli` | 0x01.0x22 | 4328 | 582 | 67 | implemented | implemented | issue+scalar_mul_latency |
| `s.mul` | 0x01.0x03 | 3752 | 601 | 84 | implemented | implemented | issue+scalar_mul_latency |
| `s.ashri` | 0x01.0x2e | 3350 | 113 | 27 | implemented | implemented | issue+scalar_latency |
| `vf.fms` | 0x11.0x07 | 3291 | 325 | 58 | implemented | implemented | issue+vector_fp_latency |
| `v.lshr` | 0x10.0x0d | 3247 | 286 | 113 | implemented | implemented | issue+vector_int_latency |
| `v.and` | 0x10.0x08 | 3216 | 326 | 113 | implemented | implemented | issue+vector_int_latency |
| `vf.cmp` | 0x11.0x18 | 3116 | 330 | 62 | implemented | implemented | issue+compare_latency |
| `v.st.scatter` | 0x20.0x28 | 2865 | 192 | 113 | implemented | implemented | issue+lsu_slots+noc+nuca+hbm(+load_latency) |
| `fence` | 0x21.0x20 | 2241 | 500 | 63 | refused | implemented | issue+own_scoreboard_drain |
| `v.cmpi` | 0x10.0x38 | 1905 | 57 | 24 | implemented | implemented | issue+compare_latency |
| `v.mov` | 0x10.0x00 | 1803 | 288 | 49 | implemented | implemented | issue+vector_int_latency |
| `vf.neg` | 0x11.0x14 | 1786 | 333 | 58 | implemented | implemented | issue+vector_fp_latency |
| `exit` | 0x02.0x1f | 1356 | 898 | 113 | implemented | implemented | issue |
| `td.load` | 0x30.0x00 | 1129 | 483 | 60 | implemented | implemented | issue+tensor_engine_admission(+descriptor_snapshot_via_noc_nuca) |
| `s.andi` | 0x01.0x28 | 1123 | 303 | 45 | implemented | implemented | issue+scalar_latency |
| `v.rcp.approx` | 0x16.0x01 | 1026 | 295 | 54 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `s.subi` | 0x01.0x21 | 950 | 329 | 49 | implemented | implemented | issue+scalar_latency |
| `mb.arm` | 0x50.0x01 | 937 | 274 | 43 | implemented | implemented | issue+barrier_latency+event_wake |
| `mb.wait` | 0x50.0x04 | 937 | 274 | 43 | implemented | implemented | issue+barrier_latency+event_wake |
| `v.exp2.approx` | 0x16.0x00 | 770 | 103 | 45 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `vf.sub` | 0x11.0x02 | 716 | 110 | 44 | implemented | implemented | issue+vector_fp_latency |
| `s.lshr` | 0x01.0x0d | 656 | 303 | 45 | implemented | implemented | issue+scalar_latency |
| `p.or` | 0x13.0x02 | 655 | 76 | 34 | implemented | implemented | issue+predicate_latency |
| `v.shl` | 0x10.0x0c | 643 | 87 | 28 | implemented | implemented | issue+vector_int_latency |
| `s.shl` | 0x01.0x0c | 630 | 303 | 45 | implemented | implemented | issue+scalar_latency |
| `w.def` | 0x30.0x10 | 623 | 274 | 43 | implemented | implemented | issue+tensor_engine_admission(+descriptor_snapshot_via_noc_nuca) |
| `t.ld.async` | 0x31.0x00 | 623 | 274 | 43 | implemented | implemented | issue+tensor_engine(dma_units,noc,nuca,hbm)+barrier_completion |
| `v.or` | 0x10.0x09 | 576 | 32 | 15 | implemented | implemented | issue+vector_int_latency |
| `vf.reduce.add` | 0x15.0x08 | 529 | 270 | 36 | implemented | implemented | issue+fp_reduction_latency |
| `v.rsqrt.approx` | 0x16.0x02 | 469 | 208 | 27 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `v.reduce.or` | 0x15.0x04 | 420 | 132 | 33 | implemented | implemented | issue+reduction_latency |
| `m.zero` | 0x40.0x00 | 419 | 320 | 52 | refused | implemented | issue+matrix_engine_occupancy |
| `s.min.s` | 0x01.0x10 | 388 | 303 | 45 | implemented | implemented | issue+scalar_latency |
| `vf.abs` | 0x11.0x15 | 329 | 20 | 12 | implemented | implemented | issue+vector_fp_latency |
| `trap` | 0x02.0x18 | 318 | 82 | 15 | refused | implemented | issue+fault |
| `vf.max` | 0x11.0x11 | 316 | 61 | 28 | implemented | implemented | issue+vector_fp_latency |
| `m.mma.async` | 0x40.0x01 | 296 | 249 | 37 | refused | implemented | issue+matrix_engine_occupancy+barrier_completion |
| `p.xor` | 0x13.0x03 | 221 | 30 | 16 | implemented | implemented | issue+predicate_latency |
| `v.ashr` | 0x10.0x0e | 159 | 82 | 15 | implemented | implemented | issue+vector_int_latency |
| `p.mov` | 0x13.0x00 | 105 | 32 | 18 | implemented | implemented | issue+predicate_latency |
| `vf.reduce.max` | 0x15.0x09 | 62 | 53 | 28 | implemented-inexact | implemented | issue+fp_reduction_latency |
| `s.ori` | 0x01.0x29 | 48 | 16 | 8 | implemented | implemented | issue+scalar_latency |
| `p.not` | 0x13.0x04 | 43 | 40 | 22 | implemented | implemented | issue+predicate_latency |
| `mb.arrive` | 0x50.0x02 | 33 | 16 | 8 | implemented | implemented | issue+barrier_latency+event_wake |
| `v.reduce.add` | 0x15.0x00 | 5 | 3 | 3 | implemented | implemented | issue+reduction_latency |

| variant | static | kernels | points | before M1 | semantics | reason |
|---|---:|---:|---:|---|---|---|
| `s.movz` | 84933 | 744 | 107 | implemented | implemented | salu  |
| `s.addi` | 62613 | 657 | 89 | implemented | implemented | salu  |
| `v.extract` | 62030 | 360 | 61 | implemented | implemented | extract  |
| `v.splat` | 59702 | 898 | 113 | implemented | implemented | vint  |
| `v.movi` | 43384 | 730 | 113 | implemented | implemented | vint  |
| `a.vread` | 35568 | 342 | 58 | refused | implemented | a_vread  |
| `v.insert` | 30292 | 133 | 34 | implemented | implemented | insert  |
| `v.cmp.lt` | 27779 | 855 | 113 | implemented | implemented | vint  |
| `v.add` | 24183 | 898 | 113 | implemented | implemented | vint  |
| `v.sel (predicated)` | 23897 | 606 | 113 | implemented | implemented | vint  |
| `vf.fma` | 23042 | 180 | 57 | implemented | implemented | vfp  |
| `v.ld.w` | 21824 | 124 | 32 | implemented | implemented | lsu  |
| `s.movk` | 20961 | 671 | 94 | implemented | implemented | salu  |
| `p.and` | 20326 | 605 | 74 | implemented | implemented | pred  |
| `v.st.w` | 19058 | 157 | 34 | implemented | implemented | lsu  |
| `v.sub` | 18813 | 331 | 113 | implemented | implemented | vint  |
| `s.shli` | 17062 | 871 | 113 | implemented | implemented | salu  |
| `s.add` | 16100 | 898 | 113 | implemented | implemented | salu  |
| `s.lshri` | 14676 | 453 | 63 | implemented | implemented | salu  |
| `s.or` | 12652 | 425 | 54 | implemented | implemented | salu  |
| `vf.mul` | 9823 | 395 | 62 | implemented | implemented | vfp  |
| `s.sub` | 9609 | 336 | 57 | implemented | implemented | salu  |
| `a.vwrite` | 9520 | 146 | 46 | refused | implemented | a_row  |
| `vf.add` | 8079 | 494 | 90 | implemented | implemented | vfp  |
| `s.st.d` | 7551 | 483 | 60 | implemented | implemented | lsu  |
| `s.and` | 7430 | 306 | 45 | implemented | implemented | salu  |
| `v.cmp.ne` | 7227 | 174 | 45 | implemented | implemented | vint  |
| `s.rdspecial` | 6675 | 898 | 113 | implemented | implemented | special  |
| `vf.cvt.f16.f32` | 6109 | 456 | 71 | implemented | implemented | cvt  |
| `vf.cvt.f32.f16` | 5866 | 565 | 81 | implemented | implemented | cvt  |
| `s.mov` | 5760 | 611 | 78 | implemented | implemented | salu  |
| `v.mul` | 5161 | 336 | 113 | implemented | implemented | vint  |
| `v.iota` | 5106 | 898 | 113 | implemented | implemented | vint  |
| `br.nz` | 5016 | 898 | 113 | implemented | implemented | branch  |
| `v.st.h (predicated)` | 4940 | 502 | 63 | implemented | implemented | lsu  |
| `s.ld.d` | 4898 | 898 | 113 | implemented | implemented | lsu  |
| `s.xor` | 4794 | 336 | 57 | implemented | implemented | salu  |
| `s.muli` | 4328 | 582 | 67 | implemented | implemented | salu  |
| `s.cmp.ge` | 4140 | 898 | 113 | implemented | implemented | salu  |
| `s.cmp.lt` | 3818 | 418 | 62 | implemented | implemented | salu  |
| `s.mul` | 3752 | 601 | 84 | implemented | implemented | salu  |
| `v.ld.h (predicated)` | 3418 | 361 | 51 | implemented | implemented | lsu  |
| `s.ashri` | 3350 | 113 | 27 | implemented | implemented | salu  |
| `vf.fms` | 3291 | 325 | 58 | implemented | implemented | vfp  |
| `v.lshr` | 3247 | 286 | 113 | implemented | implemented | vint  |
| `v.and` | 3216 | 326 | 113 | implemented | implemented | vint  |
| `v.st.h` | 2809 | 166 | 52 | implemented | implemented | lsu  |
| `v.cmp.ge` | 2452 | 530 | 62 | implemented | implemented | vint  |
| `vf.cmp.o` | 2265 | 325 | 58 | implemented | implemented | vfp  |
| `v.cmpi.lt` | 1905 | 57 | 24 | implemented | implemented | vint  |
| `v.st.w (predicated)` | 1850 | 171 | 45 | implemented | implemented | lsu  |
| `v.ld.gather.sb (predicated)` | 1839 | 16 | 113 | implemented | implemented | lsu  |
| `v.st.scatter.b (predicated)` | 1824 | 2 | 113 | implemented | implemented | lsu  |
| `v.mov` | 1803 | 288 | 49 | implemented | implemented | vint  |
| `vf.neg` | 1786 | 333 | 58 | implemented | implemented | vfp  |
| `v.ld.gather.w` | 1740 | 74 | 23 | implemented | implemented | lsu  |
| `v.ld.w (predicated)` | 1604 | 97 | 40 | implemented | implemented | lsu  |
| `v.ld.gather.h (predicated)` | 1377 | 193 | 22 | implemented | implemented | lsu  |
| `exit` | 1356 | 898 | 113 | implemented | implemented | exit  |
| `fence.acq_rel.device` | 1316 | 204 | 43 | refused | implemented | fence  |
| `v.ld.gather.h` | 1169 | 119 | 30 | implemented | implemented | lsu  |
| `td.load` | 1129 | 483 | 60 | implemented | implemented | desc  |
| `s.andi` | 1123 | 303 | 45 | implemented | implemented | salu  |
| `v.ld.gather.w (predicated)` | 1057 | 95 | 29 | implemented | implemented | lsu  |
| `v.rcp.approx` | 1026 | 295 | 54 | implemented-inexact | implemented | sfu  |
| `s.cmp.eq` | 980 | 385 | 50 | implemented | implemented | salu  |
| `s.subi` | 950 | 329 | 49 | implemented | implemented | salu  |
| `mb.arm` | 937 | 274 | 43 | implemented | implemented | barrier  |
| `mb.wait` | 937 | 274 | 43 | implemented | implemented | barrier  |
| `fence.release.device` | 925 | 392 | 60 | refused | implemented | fence  |
| `v.st.scatter.w (predicated)` | 860 | 50 | 17 | implemented | implemented | lsu  |
| `s.cmp.le` | 800 | 26 | 13 | implemented | implemented | salu  |
| `v.exp2.approx` | 770 | 103 | 45 | implemented-inexact | implemented | sfu  |
| `vf.sub` | 716 | 110 | 44 | implemented | implemented | vfp  |
| `s.cmp.ne` | 688 | 40 | 22 | implemented | implemented | salu  |
| `s.lshr` | 656 | 303 | 45 | implemented | implemented | salu  |
| `s.cmp.geu` | 656 | 303 | 45 | implemented | implemented | salu  |
| `p.or` | 655 | 76 | 34 | implemented | implemented | pred  |
| `v.shl` | 643 | 87 | 28 | implemented | implemented | vint  |
| `s.shl` | 630 | 303 | 45 | implemented | implemented | salu  |
| `w.def` | 623 | 274 | 43 | implemented | implemented | desc  |
| `v.or` | 576 | 32 | 15 | implemented | implemented | vint  |
| `vf.reduce.add.b` | 529 | 270 | 36 | implemented | implemented | reduce  |
| `v.rsqrt.approx` | 469 | 208 | 27 | implemented-inexact | implemented | sfu  |
| `m.zero` | 419 | 320 | 52 | refused | implemented | m_zero  |
| `s.min.s` | 388 | 303 | 45 | implemented | implemented | salu  |
| `t.ld.async.h` | 365 | 260 | 38 | implemented | implemented | xfer  |
| `v.ld.h` | 339 | 140 | 40 | implemented | implemented | lsu  |
| `vf.abs` | 329 | 20 | 12 | implemented | implemented | vfp  |
| `vf.cmp.lt` | 329 | 20 | 12 | implemented | implemented | vfp  |
| `trap` | 318 | 82 | 15 | refused | implemented | trap  |
| `m.mma.async` | 296 | 249 | 37 | refused | implemented | mma  |
| `v.cmp.geu` | 288 | 32 | 15 | implemented | implemented | vint  |
| `vf.max.prop` | 286 | 33 | 15 | implemented | implemented | vfp  |
| `v.reduce.or` | 263 | 122 | 27 | implemented | implemented | reduce  |
| `t.ld.async.h.t` | 226 | 215 | 28 | implemented | implemented | xfer  |
| `p.xor` | 221 | 30 | 16 | implemented | implemented | pred  |
| `vf.cmp.ne` | 192 | 5 | 4 | implemented | implemented | vfp  |
| `vf.cmp.eq` | 172 | 49 | 27 | implemented | implemented | vfp  |
| `v.ashr` | 159 | 82 | 15 | implemented | implemented | vint  |
| `p.mov` | 105 | 32 | 18 | implemented | implemented | pred  |
| `vf.cmp.gt` | 99 | 8 | 7 | implemented | implemented | vfp  |
| `v.reduce.or (predicated)` | 98 | 46 | 5 | implemented | implemented | reduce  |
| `v.cmp.eq` | 92 | 58 | 26 | implemented | implemented | vint  |
| `v.st.scatter.h (predicated)` | 91 | 77 | 12 | implemented | implemented | lsu  |
| `v.st.scatter.h` | 90 | 63 | 10 | implemented | implemented | lsu  |
| `vf.reduce.max.b` | 62 | 53 | 28 | implemented-inexact | implemented | reduce  |
| `vf.cmp.uo` | 59 | 50 | 28 | implemented | implemented | vfp  |
| `v.reduce.or.b` | 59 | 50 | 28 | implemented | implemented | reduce  |
| `s.ori` | 48 | 16 | 8 | implemented | implemented | salu  |
| `p.not` | 43 | 40 | 22 | implemented | implemented | pred  |
| `mb.arrive` | 33 | 16 | 8 | implemented | implemented | barrier  |
| `t.ld.async.w` | 32 | 14 | 7 | implemented | implemented | xfer  |
| `vf.max.ieee` | 30 | 28 | 13 | implemented | implemented | vfp  |
| `v.st.b (predicated)` | 13 | 12 | 8 | implemented | implemented | lsu  |
| `v.reduce.add` | 5 | 3 | 3 | implemented | implemented | reduce  |
| `v.cmp.gt` | 5 | 3 | 3 | implemented | implemented | vint  |
| `v.cmp.le` | 3 | 3 | 3 | implemented | implemented | vint  |
| `s.st.b` | 2 | 2 | 1 | implemented | implemented | lsu  |
