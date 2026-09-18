// t04_roles -- two Role Contexts sharing one datapath.
//
// Role 0 produces, role 1 consumes through memory. There are no barriers in
// Stage 1 (arch overview sec 13 puts mbarrier in Stage 2), so the handshake is
// a flag word plus a bounded spin: the bound turns a lost update into a trap
// instead of a hang.
//
// Ordering note: role 0 writes the payload before the flag. This relies on the
// Stage-1 LSU retiring one access at a time in program order. Once multiple
// outstanding transactions arrive in Stage 3 this needs an explicit release
// fence (ISA sec 9.2) -- the test is marked so it is revisited then.
.org 0
start:
    s.rdspecial s1, role_id
    s.rdspecial s2, num_roles
    .assert_eq  s2, #2, #1

    br.nz    s1, consumer

// ---------------------------------------------------------------- role 0
producer:
    .li      s3, #payload
    .li      s4, #0x0123456789abcdef
    s.st.d   [s3, #0], s4

    .li      s5, #flag
    .li      s6, #1
    s.st.d   [s5, #0], s6
    exit

// ---------------------------------------------------------------- role 1
consumer:
    .li      s5, #flag
    .li      s7, #200000              // spin bound
spin:
    s.ld.d   s8, [s5, #0]
    br.nz    s8, got_it
    s.addi   s7, s7, #-1
    br.nz    s7, spin
    trap     #90                      // flag never arrived

got_it:
    .li      s3, #payload
    s.ld.d   s9, [s3, #0]
    .assert_eq s9, #0x0123456789abcdef, #91

    // the consumer's own scalar state must be private to this role
    s.rdspecial s10, role_id
    .assert_eq  s10, #1, #92
    exit

.org 0x4000
flag:
    .dword 0
payload:
    .dword 0
