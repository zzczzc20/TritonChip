// Leading-zero count as a balanced tree (R41 G2, area).
//
// XLS's clz() of a wide value is lowered to a priority structure that Vivado
// builds with carry chains: 768 LUTs and 95 CARRY8 for the FMA's 89-bit count
// (evidence/r41/hbm_bw/fma_dslx/). The same function as a balanced tree
// (Oklobdzija): each node holds "some one below me" and the count within it;
// a parent takes its left child's count if the left child has a one, else
// the left width plus the right child's count. lzc128 is proved equal to
// clz for every input (#[quickcheck], prove_quickcheck_main).

// level 1: 64 two-bit nodes, bit 127 leftmost
fn lzc_l1(x: uN[128]) -> (u1[64], uN[1][64]) {
    for (i, (v, p)): (u32, (u1[64], uN[1][64])) in u32:0..u32:64 {
        let hi = x[(u32:127 - u32:2 * i)+:u1];
        let lo = x[(u32:126 - u32:2 * i)+:u1];
        (update(v, i, hi | lo), update(p, i, (!hi) as uN[1]))
    }((u1[64]:[u1:0, ...], uN[1][64]:[uN[1]:0, ...]))
}

fn lzc_comb<N: u32, M: u32, W: u32, W1: u32 = {W + u32:1}>
    (v: u1[N], p: uN[W][N]) -> (u1[M], uN[W1][M]) {
    for (i, (vo, po)): (u32, (u1[M], uN[W1][M])) in u32:0..M {
        let vl = v[u32:2 * i];
        let vr = v[u32:2 * i + u32:1];
        let pl = p[u32:2 * i];
        let pr = p[u32:2 * i + u32:1];
        (update(vo, i, vl | vr), update(po, i, if vl == u1:1 { u1:0 ++ pl } else { u1:1 ++ pr }))
    }((u1[M]:[u1:0, ...], uN[W1][M]:[uN[W1]:0, ...]))
}

// clz of 128 bits: 0..128
pub fn lzc128(x: uN[128]) -> u8 {
    let (v1, p1) = lzc_l1(x);
    let (v2, p2) = lzc_comb<u32:64, u32:32, u32:1>(v1, p1);
    let (v3, p3) = lzc_comb<u32:32, u32:16, u32:2>(v2, p2);
    let (v4, p4) = lzc_comb<u32:16, u32:8, u32:3>(v3, p3);
    let (v5, p5) = lzc_comb<u32:8, u32:4, u32:4>(v4, p4);
    let (v6, p6) = lzc_comb<u32:4, u32:2, u32:5>(v5, p5);
    let (v7, p7) = lzc_comb<u32:2, u32:1, u32:6>(v6, p6);
    if v7[0] == u1:1 { p7[0] as u8 } else { u8:128 }
}

// clz of N <= 128 bits: the value padded below with ones, so a nonzero
// value's count is unchanged and zero counts N.
pub fn lzc<N: u32, PAD: u32 = {u32:128 - N}>(x: uN[N]) -> u8 {
    let ones = !(all_ones!<uN[128]>() << PAD);
    lzc128(((x as uN[128]) << PAD) | ones)
}

#[test]
fn lzc_edges() {
    assert_eq(lzc(uN[89]:0), u8:89);
    assert_eq(lzc(uN[89]:1), u8:88);
    assert_eq(lzc(uN[89]:1 << u32:88), u8:0);
    assert_eq(lzc(u48:0), u8:48);
    assert_eq(lzc(u48:3), u8:46);
}

#[quickcheck]
fn lzc89_is_clz(m: uN[89]) -> bool { lzc(m) == clz(m) as u8 }
#[quickcheck]
fn lzc64_is_clz(m: u64) -> bool { lzc(m) == clz(m) as u8 }
#[quickcheck]
fn lzc48_is_clz(m: u48) -> bool { lzc(m) == clz(m) as u8 }
#[quickcheck]
fn lzc128_is_clz(m: uN[128]) -> bool { lzc128(m) == clz(m) as u8 }
