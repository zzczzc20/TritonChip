// t02_mem -- load/store unit: sizes, extension, alignment, masking,
// gather/scatter (ISA sec 9.1)
.org 0
start:
    .li      s1, #d_bytes
    .li      s2, #d_dword
    .li      s3, #d_vec
    .li      s4, #d_out

    // --- scalar loads, all sizes -------------------------------------------
    s.ld.b   s5, [s1, #0]
    .assert_eq s5, #0x01, #1
    s.ld.b   s5, [s1, #1]
    .assert_eq s5, #0xff, #2
    s.ld.sb  s5, [s1, #1]
    .assert_eq s5, #0xffffffffffffffff, #3
    s.ld.h   s5, [s1, #0]
    .assert_eq s5, #0xff01, #4
    s.ld.sh  s5, [s1, #2]
    .assert_eq s5, #0xffffffffffff8040, #5
    s.ld.w   s5, [s1, #0]
    .assert_eq s5, #0x8040ff01, #6
    s.ld.sw  s5, [s1, #0]
    .assert_eq s5, #0xffffffff8040ff01, #7
    s.ld.d   s5, [s2, #0]
    .assert_eq s5, #0x0123456789abcdef, #8

    // --- an unaligned 64-bit load that straddles two memory words -----------
    s.ld.d   s5, [s1, #1]
    .assert_eq s5, #0xef000000008040ff, #9

    // --- scalar stores ------------------------------------------------------
    .li      s6, #0xaabbccdd11223344
    s.st.d   [s4, #0], s6
    s.ld.d   s5, [s4, #0]
    .assert_eq s5, #0xaabbccdd11223344, #10

    // a byte store must leave the other seven bytes alone
    .li      s6, #0x99
    s.st.b   [s4, #3], s6
    s.ld.d   s5, [s4, #0]
    .assert_eq s5, #0xaabbccdd99223344, #11

    // an unaligned 32-bit store spanning the word boundary
    .li      s6, #0xdeadbeef
    s.st.w   [s4, #6], s6
    s.ld.w   s5, [s4, #6]
    .assert_eq s5, #0xdeadbeef, #12
    s.ld.b   s5, [s4, #5]        // byte 5 of ..aabbccdd99223344 is 0xcc
    .assert_eq s5, #0xcc, #13

    // --- contiguous vector load ---------------------------------------------
    v.ld.w   v1, [s3, #0]
    .assert_lane v1, #0,  #100, #20
    .assert_lane v1, #7,  #107, #21
    .assert_lane v1, #15, #115, #22

    // --- contiguous vector store, then read it back -------------------------
    .li      s7, #d_out2
    v.st.w   [s7, #0], v1
    v.ld.w   v2, [s7, #0]
    .assert_lane v2, #0,  #100, #23
    .assert_lane v2, #15, #115, #24

    // --- narrow vector elements with extension ------------------------------
    .li      s8, #d_sbytes
    v.ld.b   v3, [s8, #0]
    .assert_lane v3, #0, #0x7f, #25
    .assert_lane v3, #1, #0x80, #26
    v.ld.sb  v3, [s8, #0]
    .assert_lane v3, #0, #0x7f,       #27
    .assert_lane v3, #1, #0xffffff80, #28

    v.ld.h   v4, [s8, #0]
    .assert_lane v4, #0, #0x807f, #29
    v.ld.sh  v4, [s8, #0]
    .assert_lane v4, #0, #0xffff807f, #30

    // --- masked vector load: inactive lanes keep their old value -------------
    v.movi   v5, #77
    v.iota   v6
    v.cmpi.lt p1, v6, #8
    @p1 v.ld.w v5, [s3, #0]
    .assert_lane v5, #3,  #103, #40
    .assert_lane v5, #11, #77,  #41

    // --- masked vector store: inactive lanes must not write memory ----------
    .li      s9, #d_out3
    v.movi   v7, #-1
    v.st.w   [s9, #0], v7             // fill with 0xffffffff
    v.movi   v8, #5
    @p1 v.st.w [s9, #0], v8           // only lanes 0..7
    v.ld.w   v9, [s9, #0]
    .assert_lane v9, #0,  #5,          #42
    .assert_lane v9, #7,  #5,          #43
    .assert_lane v9, #8,  #0xffffffff, #44
    .assert_lane v9, #15, #0xffffffff, #45

    // --- an unaligned contiguous vector load --------------------------------
    // base+2 : each lane reads a 32-bit window shifted two bytes up, so lane i
    // is (v[i] >> 16) | (v[i+1] << 16).
    v.ld.w   v10, [s3, #2]
    .assert_lane v10, #0, #0x00650000, #46   // (101<<16) | (100>>16)
    .assert_lane v10, #1, #0x00660000, #47

    // --- gather -------------------------------------------------------------
    // indices 15, 14, ... 0 : reverse the vector
    v.iota   v11
    v.movi   v12, #15
    v.sub    v13, v12, v11
    v.ld.gather.w v14, [s3, v13]
    .assert_lane v14, #0,  #115, #50
    .assert_lane v14, #15, #100, #51

    // --- scatter ------------------------------------------------------------
    .li      s10, #d_out4
    v.st.scatter.w [s10, v13], v11    // lane i writes value i at slot 15-i
    v.ld.w   v15, [s10, #0]
    .assert_lane v15, #0,  #15, #52
    .assert_lane v15, #15, #0,  #53

    // --- masked gather: inactive lanes neither read nor write ---------------
    v.movi   v16, #7
    @p1 v.ld.gather.w v16, [s3, v13]
    .assert_lane v16, #0,  #115, #54
    .assert_lane v16, #11, #7,   #55

    // --- a fully masked-off access is a no-op, not an out-of-range read -----
    v.cmpi.lt p2, v6, #0              // no lane active
    v.movi    v17, #33
    .li       s11, #0                 // a null base: must never be touched
    @p2 v.ld.w v17, [s11, #0]
    .assert_lane v17, #0, #33, #56

    exit

// ---------------------------------------------------------------------------
.org 0x4000
d_bytes:
    .word 0x8040ff01
.align 3
d_dword:
    .dword 0x0123456789abcdef
d_vec:
    .word 100, 101, 102, 103, 104, 105, 106, 107
    .word 108, 109, 110, 111, 112, 113, 114, 115
    .word 116, 117                      // spill for the unaligned read
d_sbytes:
    .word 0x0201807f
    .word 0x06050403
    .word 0x0a090807
    .word 0x0e0d0c0b
    .word 0x1211100f
d_out:
    .space 64
d_out2:
    .space 64
d_out3:
    .space 64
d_out4:
    .space 64
