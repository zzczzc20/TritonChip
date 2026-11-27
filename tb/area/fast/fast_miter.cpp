// fast_miter.cpp -- cxxrtl driver for miter_fma and miter_round (R42 area). The stimulus classes are those of
// evidence/r41/stage2/candidates/F03/miter/fma_miter.cpp (alignment sweep, cancellation, subnormals, rounding
// carries, exact-product cancellation), plus corner classes. Usage: fast_miter N SEED
//   N vectors per FMA class, 7+ classes; round stage: 3 struct classes of N.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "Vmiter_all.h"
#include "verilated.h"

static uint64_t s[2] = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
static inline uint64_t rnd() { uint64_t x = s[0], y = s[1]; s[0] = y; x ^= x << 23; s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return s[1] + y; }
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
static uint32_t mk(uint32_t sgn, int e, uint32_t f) { if (e < 0) e = 0; if (e > 255) e = 255; return (sgn << 31) | ((uint32_t)e << 23) | (f & 0x7fffff); }
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
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    if (argc > 2) { s[0] ^= strtoull(argv[2], 0, 0) * 0x9e3779b97f4a7c15ull; s[1] ^= strtoull(argv[2], 0, 0) * 3; for (int i = 0; i < 8; ++i) rnd(); }
    Verilated::commandArgs(argc, argv);
    Vmiter_all top;
    const int CLASSES = 9;
    long fails = 0, total = 0, nfail = 0;
    for (int cls = 0; cls < CLASSES; ++cls) {
        long cf = 0;
        for (long i = 0; i < n; ++i) {
            uint32_t a, b, c, ctl = rnd(), k;
            k = (ctl & 7) == 0 ? 2 : 0; k |= (ctl >> 3) & 1; k |= ((ctl >> 4) & 3) == 0 ? 4 : 0;
            switch (cls) {
            case 0: a = r32(); b = r32(); c = r32(); break;
            case 1: a = mk(rnd() & 1, expo(), frac()); b = mk(rnd() & 1, expo(), frac()); c = mk(rnd() & 1, expo(), frac()); break;
            case 2: { int ea = 1 + rnd() % 254, eb = 1 + rnd() % 254, d = (int)(rnd() % 200) - 100;
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, frac()); c = mk(rnd() & 1, ea + eb - 127 + d, frac()); break; }
            case 3: { int ea = 20 + rnd() % 215, eb = 127 + (int)(rnd() % 41) - 20;
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, r32()); uint32_t pu = f2u(u2f(a) * u2f(b));
                c = pu + ((int)(rnd() % 9) - 4); if (rnd() & 1) c ^= 0x80000000u; k &= ~2u; break; }
            case 4: { a = mk(rnd() & 1, rnd() % 40, frac()); b = mk(rnd() & 1, 127 - 60 + rnd() % 60, frac()); c = mk(rnd() & 1, rnd() % 4, frac());
                if (rnd() % 3 == 0) { c = f2u(u2f(a) * u2f(b)) + ((int)(rnd() % 5) - 2); c ^= (rnd() & 1) << 31; } k &= ~2u; break; }
            case 5: { a = mk(rnd() & 1, 100 + rnd() % 155, 0x7fffff ^ (rnd() % 4));
                b = mk(rnd() & 1, 127 + (int)(rnd() % 3) - 1, (rnd() & 1) ? 0 : (0x7fffff ^ (rnd() % 4)));
                c = mk(rnd() & 1, expo(), frac());
                if (rnd() & 1) { int ec = ((a >> 23) & 255) + ((b >> 23) & 255) - 127 - 24 - rnd() % 4; c = mk(rnd() & 1, ec, frac()); } break; }
            case 6: { a = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12); b = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12);
                uint32_t pu = f2u(u2f(a) * u2f(b)); c = pu ^ 0x80000000u;
                if (rnd() & 1) c = mk((pu >> 31) ^ 1, ((pu >> 23) & 255) - (rnd() % 30), frac()); k &= ~3u; k |= rnd() & 1; break; }
            case 7: { // near-equal exponents, either sign: effective subtraction with a small exponent difference (0..3)
                int ea = 1 + rnd() % 254, eb = 100 + rnd() % 60; int e = ea + eb - 127 + (int)(rnd() % 7) - 3;
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, frac()); c = mk(rnd() & 1, e, frac()); break; }
            default: { // large alignment gaps: product tiny against c, or c tiny against the product, subnormal edge included
                int ea = 1 + rnd() % 254, eb = 1 + rnd() % 254; int sgn = (rnd() & 1) ? 1 : -1; int d = sgn * (int)(22 + rnd() % 90);
                a = mk(rnd() & 1, ea, frac()); b = mk(rnd() & 1, eb, frac()); c = mk(rnd() & 1, ea + eb - 127 + d, frac()); break; }
            }
            top.a = a; top.b = b; top.c = c;
            top.sub_c = k & 1; top.mul_only = (k >> 1) & 1; top.ftz = (k >> 2) & 1;
            top.eval();
            bool bad = false;
            for (int w = 0; w < 4; ++w) if (top.nm_ref[w] != top.nm_new[w]) bad = true;
            for (int w = 0; w < 5; ++w) if (top.al_ref_o[w] != top.al_new_o[w]) bad = true;
            if (top.y_ref != top.y_new) bad = true;
            if (bad) { ++cf; ++fails; if (nfail++ < 20) printf("MISMATCH cls %d a %08x b %08x c %08x k %x ref %08x new %08x\n", cls, a, b, c, k, top.y_ref, top.y_new); }
        }
        total += n;
        printf("fma class %d: %ld vectors, %ld mismatches\n", cls, n, cf); fflush(stdout);
    }
    Vmiter_all& rt = top;
    for (int mode = 0; mode < 4; ++mode) {
        long cf = 0;
        for (long i = 0; i < n; ++i) {
            uint32_t w[4] = {r32(), r32(), r32(), r32() & 0x3ffffff};
            // field offsets: mag[121:41] any[40] subn[39] shn_left[38] shn_mag[37:31] shs_left[30] shs_mag[29:18] eb0[17:6] ...
            auto setf = [&](int lo, int width, uint64_t v) { for (int b2 = 0; b2 < width; ++b2) { int p = lo + b2; if ((v >> b2) & 1) w[p / 32] |= 1u << (p % 32); else w[p / 32] &= ~(1u << (p % 32)); } };
            if (mode == 1) { setf(40, 1, (rnd() % 8) != 0); setf(39, 1, rnd() % 4 == 0); setf(31, 7, rnd() % 30); setf(18, 12, rnd() % 110); setf(6, 12, 1 + rnd() % 300); }
            if (mode >= 2) { setf(40, 1, 1); setf(39, 1, (rnd() % 3) == 0); setf(38, 1, (rnd() % 3) == 0); setf(31, 7, rnd() % 40); setf(18, 12, rnd() % 130);
                setf(30, 1, (rnd() % 4) == 0); setf(6, 12, 250 + rnd() % 10 - 5); int p = 41 + rnd() % 81; w[p / 32] |= 1u << (p % 32);
                if (mode == 3) { // sparse mags: single bit or a few bits, to exercise the sticky mask and the window edges
                    for (int q = 41; q < 122; ++q) w[q / 32] &= ~(1u << (q % 32)); int nb = 1 + rnd() % 3; for (int t = 0; t < nb; ++t) { int pp = 41 + rnd() % 81; w[pp / 32] |= 1u << (pp % 32); } } }
            for (int q = 0; q < 4; ++q) rt.s[q] = w[q]; rt.eval();
            if (rt.yr_ref != rt.yr_new) { ++cf; ++fails; if (nfail++ < 20) printf("ROUND MISMATCH mode %d s %08x%08x%08x%08x ref %08x new %08x\n", mode, w[3], w[2], w[1], w[0], rt.yr_ref, rt.yr_new); }
        }
        total += n;
        printf("round mode %d: %ld structs, %ld mismatches\n", mode, n, cf); fflush(stdout);
    }
    printf("fast_miter: %ld vectors, %ld mismatches\n%s\n", total, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
