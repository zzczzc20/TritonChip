// t28_count -- a program whose executed instruction count is a number you
// can read off the page.
//
// doc/phase3_tasks_and_acceptance.md R3-06 COUNT-01 wants counts matched
// against a hand-constructed trace, so this one is built to be countable:
// straight line, no branches, no loops, and nothing that expands.
//
// The first version used `.li` and `.assert_eq`, and neither is one
// instruction. `.li` is movz plus three movk whatever the immediate, and
// `.assert_eq` is a .li plus a compare plus a branch plus a trap -- a
// fourteen-line program that encoded to thirty words, with one of them
// (the trap) not executed. Every instruction below encodes to exactly one
// word, so the static count, the encoded count and the executed count are
// the same number, and that number is twelve.
//
//   s.movz x3   +   s.add x8   +   exit x1   =   12
//
// s4=1+2=3, s5=3+3=6, s6=6+1=7, s7=7+2=9, s8=9+3=12, s9=12+1=13,
// s10=13+2=15, s11=15+3=18. The values are not what this program is for,
// but a program whose result nobody can state is a program nobody is
// reading.
.org 0
start:
    s.movz    s1, #1          //  1
    s.movz    s2, #2          //  2
    s.movz    s3, #3          //  3
    s.add     s4, s1, s2      //  4
    s.add     s5, s4, s3      //  5
    s.add     s6, s5, s1      //  6
    s.add     s7, s6, s2      //  7
    s.add     s8, s7, s3      //  8
    s.add     s9, s8, s1      //  9
    s.add     s10, s9, s2     // 10
    s.add     s11, s10, s3    // 11
    exit                      // 12
