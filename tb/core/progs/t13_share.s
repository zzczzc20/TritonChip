// t13_share -- two role contexts hand a tile through a SHARED RF window,
// with no copy and no trip to memory.
//
// This is the communication path doc/compiler_overview.md sec 7.3 puts above
// every other option for tensors, and sec 13.1 asks whether the hardware can
// actually do it. It can only work if the two contexts' logical vector names
// map to the SAME physical registers, which is what the programmable per-RCTX
// base exists for. Run with +allroles +aliasvrf.
//
// Note what is NOT shared: the scalar file is private per RCTX (sec 7.3), so
// the consumer cannot read the producer's barrier token out of a register. It
// derives the generation from the iteration number instead, exactly as that
// section describes for a static pipeline.
.org 0

.equ SRC,  0x4000
.equ FLAG, 0x5000

start:
    s.rdspecial s1, role_id
    br.nz    s1, consumer

// ---------------------------------------------------------------- producer
producer:
    // fill v8..v9 with a pattern the consumer will read directly
    .li      s2, #SRC
    v.ld.w   v8, [s2, #0]
    v.ld.w   v9, [s2, #64]

    // publish: the flag is the only thing that goes through memory
    .li      s3, #FLAG
    .li      s4, #1
    s.st.d   [s3, #0], s4
    exit

// ---------------------------------------------------------------- consumer
consumer:
    .li      s3, #FLAG
    .li      s7, #200000
spin:
    s.ld.d   s8, [s3, #0]
    br.nz    s8, ready
    s.addi   s7, s7, #-1
    br.nz    s7, spin
    trap     #90
ready:
    // v8/v9 were never written by THIS context. With aliased bases they are
    // the producer's registers; without aliasing they are a private zero copy,
    // and these checks fail -- which is the point.
    .assert_lane v8, #0,  #0x11111111, #1
    .assert_lane v8, #15, #0x1f1f1f1f, #2
    .assert_lane v9, #0,  #0x22222222, #3
    .assert_lane v9, #15, #0x2f2f2f2f, #4

    // and the consumer can compute on them in place
    v.add    v10, v8, v9
    .assert_lane v10, #0, #0x33333333, #5
    exit

.org 0x4000
src:
    .word 0x11111111, 0x11111111, 0x11111111, 0x11111111
    .word 0x11111111, 0x11111111, 0x11111111, 0x11111111
    .word 0x11111111, 0x11111111, 0x11111111, 0x11111111
    .word 0x11111111, 0x11111111, 0x11111111, 0x1f1f1f1f
    .word 0x22222222, 0x22222222, 0x22222222, 0x22222222
    .word 0x22222222, 0x22222222, 0x22222222, 0x22222222
    .word 0x22222222, 0x22222222, 0x22222222, 0x22222222
    .word 0x22222222, 0x22222222, 0x22222222, 0x2f2f2f2f
.org 0x5000
flag:
    .dword 0
