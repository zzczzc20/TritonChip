// Full-lane store/load/copy plus masked stores through the real mesh/cache/AXI.
// One dispatched program owns one word, sharing lines across physical cores.
.org 0
    s.rdspecial s1, pid.x
    s.rdspecial s2, pid.y
    s.rdspecial s3, pid.z
    .li s4, #7
    s.mul s3, s3, s4
    s.add s2, s2, s3
    .li s4, #13
    s.mul s2, s2, s4
    s.add s1, s1, s2
    .li s4, #8
    s.mul s5, s1, s4
    .li s4, #0x10000
    s.add s5, s5, s4
    .li s4, #64
    s.mul s8, s1, s4
    .li s4, #0x4000
    s.add s10, s8, s4
    .li s4, #0x8000
    s.add s9, s8, s4
    .li s4, #0x18000
    s.add s8, s8, s4
    s.addi s1, s1, #4096
    s.st.d [s5, #0], s1
    s.cmp.eq s7, s1, s1
    p.frombits p1, s7
    v.splat v0, s1
    @p1 v.st.w [s5, #0], v0
    v.ld.w v2, [s8, #0]
    v.st.w [s10, #0], v2
    v.st.w [s8, #0], v0
    v.ld.w v1, [s8, #0]
    v.st.w [s9, #0], v1
    s.ld.d s6, [s5, #0]
    s.cmp.ne s7, s1, s6
    br.z s7, okay
    trap #1
okay:
    exit

// A second grid checks every word after global completion, without rewriting.
.org 0x200
    s.rdspecial s1, pid.x
    s.rdspecial s2, pid.y
    s.rdspecial s3, pid.z
    .li s4, #7
    s.mul s3, s3, s4
    s.add s2, s2, s3
    .li s4, #13
    s.mul s2, s2, s4
    s.add s1, s1, s2
    .li s4, #8
    s.mul s5, s1, s4
    .li s4, #0x10000
    s.add s5, s5, s4
    s.addi s1, s1, #4096
    s.ld.d s6, [s5, #0]
    s.cmp.ne s7, s1, s6
    br.z s7, checked
    trap #1
checked:
    exit
