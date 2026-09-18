// t05_fault -- a fault must stop exactly one Role Context and must never turn
// into a silent hang (ISA sec 10).
//
// Role 0 executes an unallocated encoding. Role 1 keeps working and exits
// normally, and the core must still reach idle. Run with +expectfault=01.
.org 0
start:
    s.rdspecial s1, role_id
    br.nz    s1, keep_working

// ---------------------------------------------------------------- role 0
bad_role:
    // major 0x03 is not allocated by ISA sec 3
    .dword   0x0300000000000000
    trap     #99                      // must be unreachable

// ---------------------------------------------------------------- role 1
keep_working:
    .li      s2, #64
    .li      s3, #0
loop:
    s.add    s3, s3, s2
    s.addi   s2, s2, #-1
    br.nz    s2, loop
    .assert_eq s3, #2080, #1          // 1+2+...+64

    .li      s4, #scratch
    .li      s5, #0x55aa55aa55aa55aa
    s.st.d   [s4, #0], s5
    s.ld.d   s6, [s4, #0]
    .assert_eq s6, #0x55aa55aa55aa55aa, #2
    exit

.org 0x4000
scratch:
    .dword 0
