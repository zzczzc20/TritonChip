// fma_vectors.h -- stimulus classes of the FMA miters (random generator, structured operands, ten classes).
// The classes are those of evidence/r41/stage2/candidates/F03/miter/fma_miter.cpp (alignment sweep, cancellation, subnormals,
// rounding carries, exact-product cancellation) plus near-equal exponents and large alignment gaps.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
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


static const int FMA_CLASSES = 9;
// one vector of class cls: operands a, b, c and controls k (bit 0 sub_c, bit 1 mul_only, bit 2 ftz)
static inline void gen_fma(int cls, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& k) {
    uint32_t ctl = rnd();
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
}
