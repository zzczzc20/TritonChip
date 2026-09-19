// t23_amofault -- a misaligned atomic is a fault, not a slow path.
//
// An unaligned ordinary access is legal here: the LSU splits it across two
// words (ISA sec 9.1). That split is exactly what an atomic cannot do, so the
// alternative to faulting would be two separate read-modify-writes that no
// other core could tell apart from one. ISA sec 9.2 makes it ACCESS_FAULT.
//
// The aligned case runs first, so a build that faulted on every atomic would
// fail here rather than passing for the wrong reason. Run with
// +expectfault=03.
.org 0
start:
    .li      s1, #scratch
    .li      s3, #1
    atom.add.w s4, [s1], s3           // aligned: must succeed
    .assert_eq s4, #0, #1
    atom.add.d s5, [s1], s3           // aligned 8-byte: must succeed
    .assert_eq s5, #1, #2

    .li      s2, #scratch
    s.addi   s2, s2, #2                // 2 past an 8-byte boundary
    atom.add.w s6, [s2], s3           // must fault
    trap     #99                       // must be unreachable

.org 0x4000
scratch:
    .dword 0
