// t39_isolate -- one Role Context waiting on a long-latency result must not
// stall the others (R39-05).
//
// Context 0 runs a chain of 24 dependent SFU passes, each waiting the SFU's
// latency on the last, so it spends nearly all its time waiting on a
// register. Every other context runs a loop of independent integer and
// vector work that never touches the SFU or memory. Each checks its own
// result. Run with +allroles +ctxtrace: the exit cycles show how long the
// computing contexts took next to the waiting one --
// on a core whose stalled context holds the front end they wait out its
// stalls; with RCTX_SWITCH they issue through them. Measured, two contexts:
// the computing one exits at cycle 1794 without the switch and 1392 with it.
.org 0
start:
    s.rdspecial s1, role_id
    br.nz       s1, compute

// ---------------------------------------------------------------- context 0
    .li         s3, #fdata
    v.ld.w      v3, [s3, #0]             // 0.0f
    v.ld.w      v4, [s3, #64]            // 1.0f
    v.add       v5, v4, v3               // x = 1.0
    .li         s9, #24
sfu:
    vf.mul      v5, v5, v3               // 0 * x
    v.exp2.approx v5, v5                 // 2^0 = 1.0, waiting on the last link
    s.addi      s9, s9, #-1
    br.nz       s9, sfu
    vf.cmp.eq   p1, v5, v4
    p.ballot    s10, p1
    .assert_eq  s10, #0xffff, #10
    exit

// ---------------------------------------------------------------- the others
compute:
    .li         s7, #0
    v.movi      v1, #1
    v.movi      v2, #0
    .li         s9, #100
loop:
    s.addi      s7, s7, #3
    v.add       v2, v2, v1
    s.addi      s7, s7, #-1
    v.add       v2, v2, v1
    s.addi      s9, s9, #-1
    br.nz       s9, loop
    .assert_eq  s7, #200, #20
    .assert_lane v2, #0, #200, #21
    .assert_lane v2, #15, #200, #22
    exit

.org 0x4000
fdata:
    .word 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
    .word 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
