// Verilator driver for fma_miter: N vectors per class, one per cycle, and the
// candidate's outputs compared with the previous generated RTL's at both
// latencies. Classes stress what the restructured arithmetic changed:
// alignment (exponent differences around 0..90), massive cancellation with a
// sticky borrow (c within a few ulps of -(a*b)), subnormal inputs/outputs,
// rounding carries (all-ones significands), overflow, ftz/mul_only/sub_c.
#include "Vfma_miter.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

static uint64_t s[2] = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
static inline uint64_t rnd() {            // xorshift128+
    uint64_t x = s[0], y = s[1]; s[0] = y; x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return s[1] + y;
}
static inline uint32_t r32() { return (uint32_t)rnd(); }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint32_t frac() {
    switch (rnd() % 8) {
    case 0: return 0;
    case 1: return 0x7fffff;
    case 2: return 1u << (rnd() % 23);
    case 3: return 0x7fffff ^ (1u << (rnd() % 23));
    case 4: return (0x7fffffu >> (rnd() % 23));
    case 5: return (0x7fffffu << (rnd() % 23)) & 0x7fffff;
    default: return r32() & 0x7fffff;
    }
}
static uint32_t mk(uint32_t sgn, int e, uint32_t f) {
    if (e < 0) e = 0; if (e > 255) e = 255;
    return (sgn << 31) | ((uint32_t)e << 23) | (f & 0x7fffff);
}
static int expo() {
    switch (rnd() % 10) {
    case 0: return 0;
    case 1: return 1 + rnd() % 3;
    case 2: return 254 - rnd() % 3;
    case 3: return 255;
    default: return 1 + rnd() % 254;
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    if (argc > 2) { s[0] ^= strtoull(argv[2], 0, 0); s[1] ^= strtoull(argv[2], 0, 0) * 3; }
    Vfma_miter* m = new Vfma_miter;
    const int CLASSES = 7;
    long fails = 0, total = 0;
    // input history, to name a failing vector
    uint32_t ha[8], hb[8], hc[8], hk[8];
    m->rst = 1; m->clk = 0; m->eval(); m->clk = 1; m->eval(); m->clk = 0; m->eval(); m->rst = 0;
    long cyc = 0;
    for (int cls = 0; cls < CLASSES; ++cls) {
        long cf = 0;
        for (long i = 0; i < n + 4; ++i, ++cyc) {
            uint32_t a, b, c, k = 0;
            uint32_t ctl = rnd();
            k = (ctl & 7) == 0 ? 2 : 0;                       // mul_only 1/8
            k |= (ctl >> 3) & 1;                              // sub_c 1/2
            k |= ((ctl >> 4) & 3) == 0 ? 4 : 0;               // ftz 1/4
            switch (cls) {
            case 0:                                            // anything
                a = r32(); b = r32(); c = r32(); break;
            case 1: {                                          // structured fields
                a = mk(rnd() & 1, expo(), frac());
                b = mk(rnd() & 1, expo(), frac());
                c = mk(rnd() & 1, expo(), frac()); break; }
            case 2: {                                          // alignment sweep
                int ea = 1 + rnd() % 254, eb = 1 + rnd() % 254;
                int d = (int)(rnd() % 200) - 100;
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, frac());
                c = mk(rnd() & 1, ea + eb - 127 + d, frac()); break; }
            case 3: {                                          // cancellation, c ~ -(a*b)
                int ea = 20 + rnd() % 215, eb = 127 + (int)(rnd() % 41) - 20;
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, r32());
                float p = u2f(a) * u2f(b);
                uint32_t pu = f2u(p);
                int dl = (int)(rnd() % 9) - 4;
                c = pu + dl;
                if (rnd() & 1) c ^= 0x80000000u;               // either sign; sub_c random
                k &= ~2u; break; }
            case 4: {                                          // subnormal results and inputs
                a = mk(rnd() & 1, rnd() % 40, frac());
                b = mk(rnd() & 1, 127 - 60 + rnd() % 60, frac());
                c = mk(rnd() & 1, rnd() % 4, frac());
                if (rnd() % 3 == 0) { float p = u2f(a) * u2f(b); c = f2u(p) + ((int)(rnd() % 5) - 2); c ^= (rnd() & 1) << 31; }
                k &= ~2u; break; }
            case 5: {                                          // rounding carries / overflow
                a = mk(rnd() & 1, 100 + rnd() % 155, 0x7fffff ^ (rnd() % 4));
                b = mk(rnd() & 1, 127 + (int)(rnd() % 3) - 1, (rnd() & 1) ? 0 : (0x7fffff ^ (rnd() % 4)));
                c = mk(rnd() & 1, expo(), frac());
                if (rnd() & 1) { int ec = ((a >> 23) & 255) + ((b >> 23) & 255) - 127 - 24 - rnd() % 4; c = mk(rnd() & 1, ec, frac()); }
                break; }
            default: {                                         // exact-product cancellation
                // a*b exact in binary32 (short significands), c = -(a*b) +- tiny
                a = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12);
                b = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12);
                uint32_t pu = f2u(u2f(a) * u2f(b));
                c = pu ^ 0x80000000u;
                if (rnd() & 1) c = mk((pu >> 31) ^ 1, ((pu >> 23) & 255) - (rnd() % 30), frac());
                k &= ~3u; k |= rnd() & 1; break; }
            }
            if (i >= n) { a = b = c = 0; k = 0; }
            m->a = a; m->b = b; m->c = c;
            m->sub_c = k & 1; m->mul_only = (k >> 1) & 1; m->ftz = (k >> 2) & 1;
            ha[cyc & 7] = a; hb[cyc & 7] = b; hc[cyc & 7] = c; hk[cyc & 7] = k;
            m->eval();
            // y for the vector presented LAT cycles ago, combinational from the last stage
            if (i >= 2 && i - 2 < n && m->y_ref2 != m->y_new2) {
                long j = (cyc - 2) & 7;
                if (fails < 20) printf("L2 MISMATCH cls %d a %08x b %08x c %08x k %x ref %08x new %08x\n",
                                       cls, ha[j], hb[j], hc[j], hk[j], m->y_ref2, m->y_new2);
                ++fails; ++cf;
            }
            if (i >= 3 && i - 3 < n && m->y_ref3 != m->y_new3) {
                long j = (cyc - 3) & 7;
                if (fails < 20) printf("L3 MISMATCH cls %d a %08x b %08x c %08x k %x ref %08x new %08x\n",
                                       cls, ha[j], hb[j], hc[j], hk[j], m->y_ref3, m->y_new3);
                ++fails; ++cf;
            }
            m->clk = 1; m->eval(); m->clk = 0; m->eval();
        }
        total += n;
        printf("class %d: %ld vectors, %ld mismatches\n", cls, n, cf);
        fflush(stdout);
    }
    printf("fma_miter: %ld vectors x 2 latencies, %ld mismatches\n%s\n", total, fails, fails ? "FAIL" : "PASS");
    delete m;
    return fails ? 1 : 0;
}
