// mac_miter.cpp -- Verilator driver: the MAC lane at BE against BE = 0, every mode, structured stimulus.
//   mac_miter N SEED : N vectors per mode and per class.
#include "Vmac_miter.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static uint64_t s[2] = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
static inline uint64_t rnd() { uint64_t x = s[0], y = s[1]; s[0] = y; x ^= x << 23; s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return s[1] + y; }
static inline uint32_t r32() { return (uint32_t)rnd(); }
static uint32_t frac(int w) {              // a fraction of w bits, structured
    uint32_t m = (w >= 32) ? 0xffffffffu : ((1u << w) - 1u);
    switch (rnd() % 8) {
    case 0: return 0;
    case 1: return m;
    case 2: return 1u << (rnd() % w);
    case 3: return m ^ (1u << (rnd() % w));
    case 4: return (m >> (rnd() % w));
    case 5: return (m << (rnd() % w)) & m;
    default: return r32() & m;
    }
}
static int pickexp(int emax) {
    switch (rnd() % 10) {
    case 0: return 0;
    case 1: return 1 + rnd() % 2;
    case 2: return emax - rnd() % 2;
    case 3: return emax;
    case 4: return emax / 2 + (int)(rnd() % 3) - 1;
    default: return 1 + rnd() % emax;
    }
}
// one element of format code fc in a 16-bit slot
static uint32_t elem(int fc) {
    uint32_t sg = rnd() & 1, e, v;
    switch (fc) {
    case 0: e = pickexp(31);  v = (sg << 15) | (e << 10) | frac(10); break;     // FP16
    case 1: e = pickexp(255); v = (sg << 15) | (e << 7) | frac(7); break;       // BF16
    case 2: e = pickexp(31);  v = (sg << 7) | (e << 2) | frac(2); break;        // E5M2 in the low byte
    default: e = pickexp(15); v = (sg << 7) | (e << 3) | frac(3); break;        // E4M3
    }
    if (rnd() % 12 == 0) v = r32() & 0xffff;
    return v;
}
static uint32_t f32s() { return ((rnd() & 1) << 31) | ((uint32_t)pickexp(255) << 23) | frac(23); }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    if (argc > 2) { s[0] ^= strtoull(argv[2], 0, 0) * 0x9e3779b97f4a7c15ull; s[1] ^= strtoull(argv[2], 0, 0) * 3; for (int i = 0; i < 8; ++i) rnd(); }
    Vmac_miter m;
    m.clk = 0; m.rst_n = 0; m.mode = 0; m.fmt = 0; m.fa = m.fb = 0; m.a = m.b = m.c = 0; m.sub_c = m.mul_only = m.ftz = 0;
    for (int i = 0; i < 4; ++i) { m.clk = 1; m.eval(); m.clk = 0; m.eval(); }
    m.rst_n = 1;
    const int LAT = 5;
    long fails = 0, nprint = 0;
    uint32_t ha[8], hb[8], hc[8], hk[8];
    long cyc = 0;
    for (int md = 0; md < 3; ++md) {
        for (int cls = 0; cls < 3; ++cls) {          // class: 0 structured, 1 mixed with random words, 2 aligned c
            long cf = 0;
            for (long i = 0; i < n + LAT + 2; ++i, ++cyc) {
                uint32_t a = 0, b = 0, c = 0, k = 0;
                int fa = rnd() & 3, fb = rnd() & 3, fmt = rnd() % 5;
                if (i < n) {
                    k = (rnd() & 1) | ((rnd() % 8 == 0) << 1) | ((rnd() % 4 == 0) << 2);
                    if (md == 0) {
                        a = f32s(); b = f32s();
                        switch (fmt) {
                        case 0: break;
                        case 1: a = elem(1); b = elem(1); break;
                        case 2: a = elem(0); b = elem(0); break;
                        case 3: a = elem(2) & 0xff; b = elem(2) & 0xff; break;
                        default: a = elem(3) & 0xff; b = elem(3) & 0xff; break;
                        }
                        if (rnd() % 6 == 0) { a = r32(); b = r32(); }
                    } else if (md == 1) {
                        a = (elem(fa) << 16) | elem(fa); b = (elem(fb) << 16) | elem(fb);
                        if (rnd() % 3 == 0) { a = (elem(rnd() & 3) << 16) | (a & 0xffff); b = (elem(rnd() & 3) << 16) | (b & 0xffff); }
                        if (rnd() % 3 == 0) { a = (a & 0xffff0000u) | elem(rnd() & 3); b = (b & 0xffff0000u) | elem(rnd() & 3); }
                        if (rnd() % 10 == 0) { a = r32(); b = r32(); }
                        k = 0;
                    } else {
                        for (int j = 0; j < 4; ++j) { a |= (elem(3) & 0xff) << (8 * j); b |= (elem(3) & 0xff) << (8 * j); }
                        if (rnd() % 10 == 0) { a = r32(); b = r32(); }
                        k = 0;
                    }
                    switch (cls == 2 ? 7 : (int)(rnd() % 6)) {
                    case 0: c = r32(); break;
                    case 1: c = f32s(); break;
                    case 2: c = ((rnd() & 1) << 31) | ((100 + rnd() % 60) << 23) | frac(23); break;
                    case 3: c = ((rnd() & 1) << 31) | ((87 + rnd() % 60) << 23) | frac(23); break;
                    case 4: c = (rnd() & 1) << 31; break;
                    case 7: { // c near the exponent range of a product of the two operands' leading elements (alignment edge)
                        int ex = 127 + (int)(rnd() % 90) - 45; c = ((rnd() & 1) << 31) | ((uint32_t)ex << 23) | frac(23); break; }
                    default: c = ((rnd() & 1) << 31) | ((uint32_t)(137 - rnd() % 30) << 23) | frac(23); break;
                    }
                }
                m.mode = md; m.fa = fa; m.fb = fb; m.fmt = fmt; m.a = a; m.b = b; m.c = c;
                m.sub_c = k & 1; m.mul_only = (k >> 1) & 1; m.ftz = (k >> 2) & 1;
                ha[cyc & 7] = a; hb[cyc & 7] = b; hc[cyc & 7] = c; hk[cyc & 7] = (md << 8) | (fa << 6) | (fb << 4) | k;
                m.eval();
                bool sbad = false;
                for (int w = 0; w < 5; ++w) if (m.al_ref[w] != m.al_new[w]) sbad = true;
                for (int w = 0; w < 4; ++w) if (m.nm_ref[w] != m.nm_new[w]) sbad = true;
                if (sbad) { ++fails; ++cf; if (nprint++ < 20) printf("STRUCT MISMATCH mode %d a %08x b %08x c %08x\n", md, a, b, c); }
                if (i >= LAT && i - LAT < n && m.y_ref != m.y_new) {
                    ++cf; ++fails;
                    if (nprint++ < 20) { long j = (cyc - LAT) & 7; printf("MISMATCH mode %d a %08x b %08x c %08x k %x ref %08x new %08x\n", md, ha[j], hb[j], hc[j], hk[j], m.y_ref, m.y_new); }
                }
                m.clk = 1; m.eval(); m.clk = 0; m.eval();
            }
            printf("mode %d class %d: %ld vectors, %ld mismatches\n", md, cls, n, cf); fflush(stdout);
        }
    }
    printf("mac_miter: %ld mismatches\n%s\n", fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
