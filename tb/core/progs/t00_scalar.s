// t00_scalar -- scalar integer and control flow (ISA sec 6, sec 7)
// Self-checking: exits cleanly, or traps with the failing check's id.
.org 0
start:
    // --- immediate construction --------------------------------------------
    s.movz   s1, #0x1234
    .assert_eq s1, #0x1234, #1

    s.movk   s1, #0xabcd, lsl #16
    .assert_eq s1, #0xabcd1234, #2

    s.movn   s2, #0
    .assert_eq s2, #0xffffffffffffffff, #3

    // --- s0 is hardwired zero ----------------------------------------------
    s.addi   s0, s0, #5
    .assert_eq s0, #0, #4

    // --- arithmetic ---------------------------------------------------------
    .li      s3, #100
    .li      s4, #7
    s.add    s5, s3, s4
    .assert_eq s5, #107, #10
    s.sub    s5, s3, s4
    .assert_eq s5, #93, #11
    s.mul    s5, s3, s4
    .assert_eq s5, #700, #12
    s.addi   s5, s3, #-150
    .assert_eq s5, #0xffffffffffffffce, #13

    // 64-bit multiply high, signed and unsigned
    .li      s6, #0xffffffffffffffff
    s.mulh.u s7, s6, s6
    .assert_eq s7, #0xfffffffffffffffe, #14
    s.mulh.s s7, s6, s6
    .assert_eq s7, #0, #15

    // --- shifts -------------------------------------------------------------
    .li      s8, #0xf000000000000000
    s.ashri  s9, s8, #4
    .assert_eq s9, #0xff00000000000000, #20
    s.lshri  s9, s8, #4
    .assert_eq s9, #0x0f00000000000000, #21
    s.shli   s9, s8, #4
    .assert_eq s9, #0, #22

    // --- bit counting -------------------------------------------------------
    .li      s10, #0x0000000000000001
    s.clz    s11, s10
    .assert_eq s11, #63, #23
    s.clz    s11, s0
    .assert_eq s11, #64, #24
    .li      s10, #0xff00ff00ff00ff00
    s.popc   s11, s10
    .assert_eq s11, #32, #25

    // --- compares -----------------------------------------------------------
    .li      s12, #-5
    .li      s13, #3
    s.cmp.lt  s14, s12, s13
    .assert_eq s14, #1, #30
    s.cmp.ltu s14, s12, s13
    .assert_eq s14, #0, #31
    s.cmp.ge  s14, s12, s13
    .assert_eq s14, #0, #32
    s.cmpi.eq s14, s13, #3
    .assert_eq s14, #1, #33

    // --- select -------------------------------------------------------------
    s.sel    s15, s12, s13, s14
    .assert_eq s15, #0xfffffffffffffffb, #34

    // --- min / max ----------------------------------------------------------
    s.min.s  s16, s12, s13
    .assert_eq s16, #0xfffffffffffffffb, #35
    s.max.u  s16, s12, s13
    .assert_eq s16, #0xfffffffffffffffb, #36

    // --- sign extension -----------------------------------------------------
    .li      s17, #0x00000000ffffffff
    s.sext.w s18, s17
    .assert_eq s18, #0xffffffffffffffff, #37
    s.zext.w  s18, s17
    .assert_eq s18, #0x00000000ffffffff, #38

    // --- a counted loop -----------------------------------------------------
    .li      s20, #10
    .li      s21, #0
loop:
    s.add    s21, s21, s20
    s.addi   s20, s20, #-1
    br.nz    s20, loop
    .assert_eq s21, #55, #40

    // --- call / ret ---------------------------------------------------------
    .li      s22, #3
    call     s30, double
    .assert_eq s22, #6, #41

    // --- special registers --------------------------------------------------
    s.rdspecial s23, num_lanes
    .assert_eq  s23, #16, #50
    s.rdspecial s23, isa_version
    .assert_eq  s23, #1, #51
    s.rdspecial s23, core_id
    .assert_eq  s23, #0, #52

    exit

double:
    s.add    s22, s22, s22
    ret      s30
