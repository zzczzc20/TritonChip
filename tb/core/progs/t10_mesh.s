// t10_mesh -- runs on all four tiles of the 2x2 system at once.
//
// Every tile does the same work on its own slice of the address space. Because
// the home slice is a function of the line address (ISA/arch sec 10.1) and the
// interleave is 64 bytes, a contiguous per-tile buffer is spread across all
// four slices: roughly three quarters of every tile's accesses leave the tile
// and cross the mesh. That is what this test is for -- it is not a
// single-core test that happens to be replicated.
//
// Phase 1  write a per-tile pattern, read it back
// Phase 2  read a shared read-only table every tile touches, so all four
//          hammer the same home slices at once
// Phase 3  stream through a range much larger than the slice, forcing dirty
//          eviction and MSHR pressure while the other tiles do the same
.org 0

.equ  PRIV_BASE,   0x10000
.equ  PRIV_STRIDE, 0x4000
.equ  SHARED_BASE, 0x8000
.equ  STREAM_BASE, 0x30000
.equ  N_WORDS,     64
.equ  N_STREAM,    128

start:
    s.rdspecial s1, core_id            // 0..3
    .li      s2, #PRIV_STRIDE
    s.mul    s3, s1, s2
    .li      s4, #PRIV_BASE
    s.add    s3, s3, s4                // s3 = this tile's private base

    // ---- phase 1: write a per-tile pattern -------------------------------
    .li      s5, #N_WORDS
    .li      s6, #0                    // index
    s.mov    s7, s3                    // cursor
p1_write:
    .li      s8, #0x1000
    s.mul    s9, s1, s8                // core_id * 0x1000
    s.add    s9, s9, s6                // + index
    s.st.d   [s7, #0], s9
    s.addi   s7, s7, #8
    s.addi   s6, s6, #1
    s.cmp.lt s10, s6, s5
    br.nz    s10, p1_write

    // ---- read back and check ---------------------------------------------
    .li      s6, #0
    s.mov    s7, s3
p1_read:
    s.ld.d   s11, [s7, #0]
    .li      s8, #0x1000
    s.mul    s9, s1, s8
    s.add    s9, s9, s6
    s.cmp.ne s10, s11, s9
    br.z     s10, p1_ok
    trap     #1
p1_ok:
    s.addi   s7, s7, #8
    s.addi   s6, s6, #1
    s.cmp.lt s10, s6, s5
    br.nz    s10, p1_read

    // ---- phase 2: the shared table, read by every tile at once -----------
    .li      s12, #SHARED_BASE
    .li      s6, #0
    .li      s13, #0                   // running sum
p2_loop:
    s.shli   s14, s6, #3
    s.add    s14, s14, s12
    s.ld.d   s11, [s14, #0]
    s.add    s13, s13, s11
    s.addi   s6, s6, #1
    s.cmp.lt s10, s6, s5
    br.nz    s10, p2_loop
    // 0+1+...+63 = 2016
    .assert_eq s13, #2016, #2

    // ---- phase 3: stream, to force eviction under cross-tile pressure ----
    .li      s15, #STREAM_BASE
    .li      s2, #PRIV_STRIDE
    s.mul    s16, s1, s2
    s.add    s15, s15, s16             // per-tile stream window
    .li      s5, #N_STREAM
    .li      s6, #0
p3_loop:
    s.shli   s14, s6, #6               // one word per 64-byte line
    s.add    s14, s14, s15
    s.st.d   [s14, #0], s6
    s.ld.d   s11, [s14, #0]
    s.cmp.ne s10, s11, s6
    br.z     s10, p3_ok
    trap     #3
p3_ok:
    s.addi   s6, s6, #1
    s.cmp.lt s10, s6, s5
    br.nz    s10, p3_loop

    // ---- phase 4: re-read phase 1 data; it must have survived ------------
    .li      s5, #N_WORDS
    .li      s6, #0
    s.mov    s7, s3
p4_loop:
    s.ld.d   s11, [s7, #0]
    .li      s8, #0x1000
    s.mul    s9, s1, s8
    s.add    s9, s9, s6
    s.cmp.ne s10, s11, s9
    br.z     s10, p4_ok
    trap     #4
p4_ok:
    s.addi   s7, s7, #8
    s.addi   s6, s6, #1
    s.cmp.lt s10, s6, s5
    br.nz    s10, p4_loop

    exit
