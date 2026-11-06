// lane_check -- Verilator driver for lane_tb (every tnca_pmac_lane variant
// beside the current lane, tnca_fpu_multi at r41-pmac's base, renamed *_ref).
//
//   lane_check miter  N SEED   FP32 mode: each variant against the current lane,
//                              N vectors per stimulus class (F03's miter classes
//                              plus the narrow formats' widening). Bit identity.
//   lane_check single8 SEED    DP4, every (a_k, b_k) pair of every FP8 format
//                              pair in every slot k, the other slots zero products
//                              of the same sign: the group sum is the one product,
//                              exact, so y must equal the binary32 FMA on the
//                              exactly widened pair (the current lane, fmt FP32).
//   lane_check single16 N SEED DP2, the same with N random 16-bit pairs per
//                              (format pair, slot).
//   lane_check dump CLASS N SEED   N multi-product groups of a stimulus class, one
//                              line each for check_dp.py (the exact oracle):
//                              mode alt_a alt_b a b c sub_c y1 y2l y2d y3
//
// Variants not defined for a vector are not compared (V1, V3: no DP2; V3: E4M3
// x E4M3 only). Exit status 1 on any mismatch.
#include "Vlane_tb.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>

static uint64_t s[2] = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
static inline uint64_t rnd() {            // xorshift128+
    uint64_t x = s[0], y = s[1]; s[0] = y; x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return s[1] + y;
}
static inline uint32_t r32() { return (uint32_t)rnd(); }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static int LAT_NEW = 3;
static const int LAT_REF = 3;

struct Vec {
    uint8_t mode, fmt, alt_a, alt_b, sub, mo, ftz;
    uint32_t a, b, c;
    uint8_t rfmt, rsub, rmo, rftz;
    uint32_t ra, rb, rc;
    uint32_t y1, y2l, y2d, y3, yref;
    bool chk1, chk2, chk3, chkref;   // which variants this vector checks against yref
    long idx;
};

static Vlane_tb* m;
static Vec ring[64];
static long cyc = 0;

static void reset() {
    m->rst_n = 0; m->clk = 0; m->eval();
    for (int i = 0; i < 4; ++i) { m->clk = 1; m->eval(); m->clk = 0; m->eval(); }
    m->rst_n = 1;
}

// Present v this cycle; returns the vector whose results completed (or null).
static Vec* step(const Vec& v) {
    Vec& r = ring[cyc & 63];
    r = v;
    m->mode = v.mode; m->fmt = v.fmt; m->alt_a = v.alt_a; m->alt_b = v.alt_b;
    m->a = v.a; m->b = v.b; m->c = v.c; m->sub_c = v.sub; m->mul_only = v.mo; m->ftz = v.ftz;
    m->rfmt = v.rfmt; m->ra = v.ra; m->rb = v.rb; m->rc = v.rc;
    m->rsub = v.rsub; m->rmo = v.rmo; m->rftz = v.rftz;
    m->eval();
    Vec* done = nullptr;
    if (cyc >= LAT_NEW) {
        Vec& n = ring[(cyc - LAT_NEW) & 63];
        n.y1 = m->y1; n.y2l = m->y2l; n.y2d = m->y2d; n.y3 = m->y3;
    }
    if (cyc >= LAT_REF) ring[(cyc - LAT_REF) & 63].yref = m->yref;
    int L = LAT_NEW > LAT_REF ? LAT_NEW : LAT_REF;
    if (cyc >= L) done = &ring[(cyc - L) & 63];
    m->clk = 1; m->eval(); m->clk = 0; m->eval();
    ++cyc;
    return done;
}

static long fails = 0;
static void report(const Vec* d, const char* what, uint32_t got, uint32_t want) {
    if (fails < 30)
        printf("MISMATCH %s #%ld mode %d fmt %d alt %d%d a %08x b %08x c %08x sub %d mo %d ftz %d"
               " | ref fmt %d a %08x b %08x c %08x | got %08x want %08x\n",
               what, d->idx, d->mode, d->fmt, d->alt_a, d->alt_b, d->a, d->b, d->c, d->sub, d->mo,
               d->ftz, d->rfmt, d->ra, d->rb, d->rc, got, want);
    ++fails;
}
static long checked[4] = {0, 0, 0, 0};
static void check(const Vec* d) {
    if (!d || !d->chkref) return;
    if (d->chk1) { ++checked[0]; if (d->y1 != d->yref) report(d, "V1", d->y1, d->yref); }
    if (d->chk2) { ++checked[1]; if (d->y2l != d->yref) report(d, "V2L", d->y2l, d->yref);
                   ++checked[2]; if (d->y2d != d->yref) report(d, "V2D", d->y2d, d->yref); }
    if (d->chk3) { ++checked[3]; if (d->y3 != d->yref) report(d, "V3", d->y3, d->yref); }
}
static void flush(void (*sink)(const Vec*)) {
    Vec z; memset(&z, 0, sizeof z); z.idx = -1;
    for (int i = 0; i < 8; ++i) sink(step(z));
}

//---------------------------------------------------------------- formats
// exact widening of a narrow element to binary32 bits (the oracle's, not the RTL's)
static uint32_t widen8(uint32_t x, int alt) {
    uint32_t sg = (x >> 7) & 1; float v;
    if (alt == 0) {                                   // E4M3
        int e = (x >> 3) & 15, f = x & 7;
        if (e == 15 && f == 7) return 0x7fc00000;
        v = e == 0 ? ldexpf((float)f, -9) : ldexpf((float)(8 + f), e - 10);
    } else {                                          // E5M2
        int e = (x >> 2) & 31, f = x & 3;
        if (e == 31) return f ? 0x7fc00000 : (sg << 31) | 0x7f800000;
        v = e == 0 ? ldexpf((float)f, -16) : ldexpf((float)(4 + f), e - 17);
    }
    return f2u(v) | (sg << 31);
}
static uint32_t widen16(uint32_t x, int alt) {
    uint32_t sg = (x >> 15) & 1;
    if (alt == 1) return x << 16;                     // BF16: exact by construction
    int e = (x >> 10) & 31, f = x & 1023;
    if (e == 31) return f ? 0x7fc00000 : (sg << 31) | 0x7f800000;
    float v = e == 0 ? ldexpf((float)f, -24) : ldexpf((float)(1024 + f), e - 25);
    return f2u(v) | (sg << 31);
}

//---------------------------------------------------------------- stimulus
static uint32_t frac23() {
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
// an addend: anything, or near the scale 2^e2 (e2 unbiased)
static uint32_t addend(int e2) {
    switch (rnd() % 12) {
    case 0: return (rnd() & 1) << 31;                              // +-0
    case 1: return mk(rnd() & 1, 0, frac23());                      // subnormal
    case 2: return mk(rnd() & 1, 255, rnd() % 4 ? 0 : frac23());    // inf / NaN
    case 3: return r32();
    case 4: return mk(rnd() & 1, 254, frac23());                    // near overflow
    default: return mk(rnd() & 1, e2 + 127 + (int)(rnd() % 61) - 30, frac23());
    }
}

// the FP32-mode miter: F03's seven classes, plus a narrow-format class
static void run_miter(long n) {
    for (int cls = 0; cls < 8; ++cls) {
        long f0 = fails;
        for (long i = 0; i < n; ++i) {
            Vec v; memset(&v, 0, sizeof v); v.idx = i;
            uint32_t a, b, c, k = 0, ctl = rnd();
            k = (ctl & 7) == 0 ? 2 : 0; k |= (ctl >> 3) & 1; k |= ((ctl >> 4) & 3) == 0 ? 4 : 0;
            uint8_t fmt = 0;
            switch (cls) {
            case 0: a = r32(); b = r32(); c = r32(); break;
            case 1: a = mk(rnd() & 1, expo(), frac23()); b = mk(rnd() & 1, expo(), frac23());
                    c = mk(rnd() & 1, expo(), frac23()); break;
            case 2: { int ea = 1 + rnd() % 254, eb = 1 + rnd() % 254, d = (int)(rnd() % 200) - 100;
                    a = mk(rnd() & 1, ea, frac23()); b = mk(rnd() & 1, eb, frac23());
                    c = mk(rnd() & 1, ea + eb - 127 + d, frac23()); break; }
            case 3: { int ea = 20 + rnd() % 215, eb = 127 + (int)(rnd() % 41) - 20;
                    a = mk(rnd() & 1, ea, frac23()); b = mk(rnd() & 1, eb, r32());
                    c = f2u(u2f(a) * u2f(b)) + ((int)(rnd() % 9) - 4);
                    if (rnd() & 1) c ^= 0x80000000u; k &= ~2u; break; }
            case 4: { a = mk(rnd() & 1, rnd() % 40, frac23()); b = mk(rnd() & 1, 127 - 60 + rnd() % 60, frac23());
                    c = mk(rnd() & 1, rnd() % 4, frac23());
                    if (rnd() % 3 == 0) { c = f2u(u2f(a) * u2f(b)) + ((int)(rnd() % 5) - 2); c ^= (rnd() & 1) << 31; }
                    k &= ~2u; break; }
            case 5: { a = mk(rnd() & 1, 100 + rnd() % 155, 0x7fffff ^ (rnd() % 4));
                    b = mk(rnd() & 1, 127 + (int)(rnd() % 3) - 1, (rnd() & 1) ? 0 : (0x7fffff ^ (rnd() % 4)));
                    c = mk(rnd() & 1, expo(), frac23());
                    if (rnd() & 1) { int ec = ((a >> 23) & 255) + ((b >> 23) & 255) - 127 - 24 - rnd() % 4; c = mk(rnd() & 1, ec, frac23()); }
                    break; }
            case 6: { a = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12);
                    b = mk(rnd() & 1, 1 + rnd() % 254, (r32() & 0x7ff) << 12);
                    uint32_t pu = f2u(u2f(a) * u2f(b)); c = pu ^ 0x80000000u;
                    if (rnd() & 1) c = mk((pu >> 31) ^ 1, ((pu >> 23) & 255) - (rnd() % 30), frac23());
                    k &= ~3u; k |= rnd() & 1; break; }
            default: {                                         // narrow formats, widened
                    fmt = 1 + rnd() % 4;                       // BF16 F16 E5M2 E4M3
                    a = r32(); b = r32();                      // upper bits must be ignored
                    c = rnd() % 3 ? addend((int)(rnd() % 60) - 30) : r32(); break; }
            }
            v.mode = 0; v.fmt = fmt; v.alt_a = rnd() & 1; v.alt_b = rnd() & 1;
            v.a = a; v.b = b; v.c = c; v.sub = k & 1; v.mo = (k >> 1) & 1; v.ftz = (k >> 2) & 1;
            v.rfmt = fmt; v.ra = a; v.rb = b; v.rc = c; v.rsub = v.sub; v.rmo = v.mo; v.rftz = v.ftz;
            v.chk1 = v.chk2 = v.chk3 = v.chkref = true;
            check(step(v));
        }
        flush(check);
        printf("miter class %d: %ld vectors, %ld mismatches\n", cls, n, fails - f0);
        fflush(stdout);
    }
}

// a group with one product, in slot k; the other slots are zero products with
// that product's sign, so an all-zero group keeps the FMA's zero sign
static Vec single_vec(int mode, int alt_a, int alt_b, int k, uint32_t ea, uint32_t eb, uint32_t c, int sub) {
    Vec v; memset(&v, 0, sizeof v);
    int w = mode == 2 ? 8 : 16, g = 32 / w;
    uint32_t sgn = ((ea >> (w - 1)) ^ (eb >> (w - 1))) & 1;
    uint32_t a = 0, b = 0;
    for (int j = 0; j < g; ++j) {
        uint32_t aj = j == k ? ea : (sgn << (w - 1)), bj = j == k ? eb : 0;
        a |= aj << (w * j); b |= bj << (w * j);
    }
    v.mode = mode; v.fmt = 0; v.alt_a = alt_a; v.alt_b = alt_b; v.a = a; v.b = b; v.c = c; v.sub = sub;
    v.rfmt = 0; v.rc = c; v.rsub = sub;
    v.ra = mode == 2 ? widen8(ea, alt_a) : widen16(ea, alt_a);
    v.rb = mode == 2 ? widen8(eb, alt_b) : widen16(eb, alt_b);
    v.chkref = true;
    v.chk1 = mode == 2; v.chk2 = true; v.chk3 = mode == 2 && alt_a == 0 && alt_b == 0;
    return v;
}
// addends for a single product p (binary32 bits): around it, cancelling it, or anything
static uint32_t c_for(uint32_t p, int i) {
    switch (i % 8) {
    case 0: return 0;
    case 1: return 0x80000000;
    case 2: return p ^ 0x80000000u;                                 // exact cancellation
    case 3: return (p ^ 0x80000000u) + ((int)(rnd() % 5) - 2);       // near cancellation
    case 4: return mk(rnd() & 1, ((p >> 23) & 255) - 24 - (int)(rnd() % 8), frac23());
    case 5: return mk(rnd() & 1, ((p >> 23) & 255) + 24 + (int)(rnd() % 8), frac23());
    case 6: return addend((int)((p >> 23) & 255) - 127);
    default: return r32();
    }
}
static void run_single8() {
    long idx = 0;
    for (int alt = 0; alt < 4; ++alt) {
        long f0 = fails, c0 = checked[0] + checked[1] + checked[3];
        for (int k = 0; k < 4; ++k)
            for (uint32_t ea = 0; ea < 256; ++ea)
                for (uint32_t eb = 0; eb < 256; ++eb) {
                    int aa = alt & 1, ab = alt >> 1;
                    uint32_t p = f2u(u2f(widen8(ea, aa)) * u2f(widen8(eb, ab)));   // exact (<= 8 bits)
                    Vec v = single_vec(2, aa, ab, k, ea, eb, c_for(p, (int)(ea * 7 + eb * 3 + k)), rnd() & 1);
                    v.idx = idx++;
                    check(step(v));
                }
        flush(check);
        printf("single8 alt_a %d alt_b %d: 4 slots x 65536 pairs, %ld lane results checked, %ld mismatches\n",
               alt & 1, alt >> 1, checked[0] + checked[1] + checked[3] - c0, fails - f0);
        fflush(stdout);
    }
}
static uint32_t elem16(int alt) {
    switch (rnd() % 8) {
    case 0: return rnd() & 0x8000;                                  // +-0
    case 1: return (rnd() & 0x8000) | (rnd() % 4 ? (r32() & (alt ? 0x7f : 0x3ff)) : 1);   // subnormal
    case 2: return (rnd() & 0x8000) | (alt ? 0x7f80 : 0x7c00) | (rnd() % 3 ? 0 : (r32() & (alt ? 0x7f : 0x3ff)));
    case 3: return (rnd() & 0x8000) | (alt ? 0x7f7f : 0x7bff) ^ (rnd() % 4);   // near max
    default: return r32() & 0xffff;
    }
}
static void run_single16(long n) {
    long idx = 0;
    for (int alt = 0; alt < 4; ++alt) {
        long f0 = fails, c0 = checked[1];
        int aa = alt & 1, ab = alt >> 1;
        for (int k = 0; k < 2; ++k)
            for (long i = 0; i < n; ++i) {
                uint32_t ea = elem16(aa), eb = elem16(ab);
                uint32_t wa = widen16(ea, aa), wb = widen16(eb, ab);
                // the binary32 product may overflow or underflow; the addend
                // generator only needs its rough scale
                float pf = u2f(wa) * u2f(wb);
                Vec v = single_vec(1, aa, ab, k, ea, eb, c_for(f2u(pf), (int)i), rnd() & 1);
                v.idx = idx++;
                check(step(v));
            }
        flush(check);
        printf("single16 alt_a %d alt_b %d: 2 slots x %ld pairs, %ld lane results checked (x2 V2L/V2D), %ld mismatches\n",
               aa, ab, n, checked[1] - c0, fails - f0);
        fflush(stdout);
    }
}

//---------------------------------------------------------------- multi-product groups
static uint32_t e8(int alt, int cls) {
    if (cls == 3) {                                                  // specials-rich
        switch (rnd() % 8) {
        case 0: return alt ? 0x7c | (rnd() % 4) | (rnd() & 0x80) : 0x7f | (rnd() & 0x80); // inf/NaN
        case 1: return rnd() & 0x80;
        case 2: return (rnd() & 0x80) | (alt ? 0x7b : 0x7e);
        case 3: return (rnd() & 0x80) | 1;
        default: return r32() & 0xff;
        }
    }
    if (cls == 4) {                                                  // extremes: very big or very small
        if (rnd() & 1) return (rnd() & 0x80) | (alt ? (0x78 | (rnd() % 4)) : (0x70 | (rnd() % 7)));
        return (rnd() & 0x80) | (alt ? (rnd() % 8) : (rnd() % 16));
    }
    if (cls == 5) {                                                  // clustered exponents, full significands
        int e = alt ? 14 + rnd() % 3 : 6 + rnd() % 3;
        return (rnd() & 0x80) | (alt ? (e << 2) | 3 : (e << 3) | 7 - (rnd() % 2));
    }
    uint32_t x = r32() & 0xff;
    // keep the specials rare outside class 3
    if (alt == 0 && (x & 0x7f) == 0x7f) x ^= 1;
    if (alt == 1 && ((x >> 2) & 31) == 31) x ^= 0x04;
    return x;
}
static uint32_t e16(int alt, int cls) {
    if (cls == 15) return elem16(alt);
    if (cls == 14) {                                                 // extremes
        switch (rnd() % 4) {
        case 0: return (rnd() & 0x8000) | (alt ? 0x7f00 | (r32() & 0x7f) : 0x7800 | (r32() & 0x3ff));
        case 1: return (rnd() & 0x8000) | (r32() & (alt ? 0x7f : 0x3ff));
        case 2: return (rnd() & 0x8000) | (alt ? 0x7f7f : 0x7bff);
        default: return (rnd() & 0x8000) | (alt ? 0x0080 : 0x0400);
        }
    }
    if (cls == 16) {                                                 // tiny: BF16 near 2^-126, FP16 subnormal
        if (alt) return (rnd() & 0x8000) | ((uint32_t)(rnd() % 12) << 7) | (r32() & 0x7f);
        return (rnd() & 0x8000) | (r32() & 0x7ff);
    }
    uint32_t x = r32() & 0xffff;
    if (alt == 0 && ((x >> 10) & 31) == 31) x ^= 0x400;
    if (alt == 1 && ((x >> 7) & 255) == 255) x ^= 0x80;
    return x;
}
static int ulp16(uint32_t x, int d) {                                 // x +- d ulps, same sign
    int m = (int)(x & 0x7fff) + d; if (m < 0) m = 0; if (m > 0x7bff) m = 0x7bff;
    return (x & 0x8000) | m;
}
// sum of the group in a wide float (quad on aarch64): a stimulus only
typedef long double LD;
static LD val8(uint32_t x, int alt) { uint32_t w = widen8(x, alt); return (LD)u2f(w); }
static LD val16(uint32_t x, int alt) { uint32_t w = widen16(x, alt); return (LD)u2f(w); }

static void dump_line(const Vec* d) {
    if (!d || d->idx < 0) return;
    printf("%d %d %d %08x %08x %08x %d %08x %08x %08x %08x\n", d->mode, d->alt_a, d->alt_b,
           d->a, d->b, d->c, d->sub, d->y1, d->y2l, d->y2d, d->y3);
}
static void run_dump(int cls, long n) {
    for (long i = 0; i < n; ++i) {
        Vec v; memset(&v, 0, sizeof v); v.idx = i;
        int dp4 = cls < 10;
        int aa = rnd() & 1, ab = rnd() & 1;
        if (dp4 && rnd() % 4 == 0) aa = ab = 0;                      // more E4M3 x E4M3 for V3
        uint32_t a = 0, b = 0, c = 0;
        LD sum = 0;
        if (dp4) {
            uint32_t ea[4], eb[4];
            for (int k = 0; k < 4; ++k) { ea[k] = e8(aa, cls); eb[k] = e8(ab, cls); }
            if (cls == 1) {                                          // a pair cancels exactly, a tiny term survives
                ea[1] = ea[0] ^ 0x80; eb[1] = eb[0];
                ea[2] = (rnd() & 0x80) | (rnd() % 8); eb[2] = (rnd() & 0x80) | (rnd() % 8);
                if (rnd() & 1) { ea[3] = ea[2] ^ 0x80; eb[3] = eb[2] ^ (rnd() % 2); }
            }
            for (int k = 0; k < 4; ++k) { a |= ea[k] << (8 * k); b |= eb[k] << (8 * k); sum += val8(ea[k], aa) * val8(eb[k], ab); }
            v.mode = 2;
        } else {
            uint32_t ea[2], eb[2];
            for (int k = 0; k < 2; ++k) { ea[k] = e16(aa, cls); eb[k] = e16(ab, cls); }
            if (cls == 11) {                                         // near cancellation of the two products
                ea[1] = ea[0] ^ 0x8000;
                eb[1] = (aa == ab && ab == 0) ? (uint32_t)ulp16(eb[0], (int)(rnd() % 5) - 2) : eb[0] ^ (rnd() % 4);
                if (rnd() & 1) { uint32_t t = ea[1]; ea[1] = eb[1]; eb[1] = t; if (aa != ab) { ea[1] = ea[0] ^ 0x8000; } }
            }
            if (cls == 12) {                                         // a big and a tiny product
                eb[0] = (rnd() & 0x8000) | (ab ? 0x7e00 | (r32() & 0xff) : 0x7800 | (r32() & 0x3ff));
                ea[0] = (rnd() & 0x8000) | (aa ? 0x4000 | (r32() & 0x3fff) : 0x4000 | (r32() & 0x3bff));
                ea[1] = (rnd() & 0x8000) | (r32() & (aa ? 0x0fff : 0x03ff));
                eb[1] = (rnd() & 0x8000) | (r32() & (ab ? 0x0fff : 0x03ff));
            }
            for (int k = 0; k < 2; ++k) { a |= ea[k] << (16 * k); b |= eb[k] << (16 * k); sum += val16(ea[k], aa) * val16(eb[k], ab); }
            v.mode = 1;
        }
        int e2 = 0;
        if (std::isfinite((double)sum) && sum != 0) { int ex; frexpl(sum, &ex); e2 = ex; }
        switch (cls) {
        case 2: case 13: case 12: {                                  // C cancels the sum, nearly or exactly
            float cf = (float)(-sum);
            c = f2u(cf);
            if (std::isfinite(cf) && rnd() % 3) c += (int)(rnd() % 5) - 2;
            if (cls == 12 && rnd() % 2) { LD big = val16((a & 0xffff), aa) * val16((b & 0xffff), ab); c = f2u((float)(-big)); }
            break; }
        case 1: case 16: c = rnd() % 2 ? ((rnd() & 1) << 31) : mk(rnd() & 1, rnd() % 3, frac23()); break;
        default: c = addend(e2); break;
        }
        v.a = a; v.b = b; v.c = c; v.alt_a = aa; v.alt_b = ab; v.sub = (rnd() % 4) == 0;
        dump_line(step(v));
    }
    flush(dump_line);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 2; }
    const char* lat = getenv("LAT_NEW");
    if (lat) LAT_NEW = atoi(lat);
    std::string cmd = argv[1];
    long seed = 1;
    m = new Vlane_tb;
    reset();
    if (cmd == "miter") {
        long n = atol(argv[2]); seed = argc > 3 ? atol(argv[3]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_miter(n);
        printf("miter: %ld per class x 8 classes, checked V1 %ld V2L %ld V2D %ld V3 %ld, %ld mismatches: %s\n",
               n, checked[0], checked[1], checked[2], checked[3], fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "single8") {
        seed = argc > 2 ? atol(argv[2]) : 1; s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_single8();
        printf("single8: checked V1 %ld V2L %ld V2D %ld V3 %ld, %ld mismatches: %s\n",
               checked[0], checked[1], checked[2], checked[3], fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "single16") {
        long n = atol(argv[2]); seed = argc > 3 ? atol(argv[3]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_single16(n);
        printf("single16: checked V2L %ld V2D %ld, %ld mismatches: %s\n",
               checked[1], checked[2], fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "dump") {
        int cls = atoi(argv[2]); long n = atol(argv[3]); seed = argc > 4 ? atol(argv[4]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull ^ (uint64_t)cls * 0x51ull; s[1] ^= seed * 3 + cls;
        run_dump(cls, n);
    }
    delete m;
    return fails ? 1 : 0;
}
