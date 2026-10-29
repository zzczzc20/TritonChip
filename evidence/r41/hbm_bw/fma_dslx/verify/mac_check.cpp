// mac_check -- Verilator driver for mac_tb (tnca_mac_lane beside the current
// matrix lane, tnca_fpu_multi at mkref.sh's revision, renamed *_ref).
//
//   mac_check miter N SEED      FP32 mode against the current lane, N vectors
//                               per stimulus class (F03's miter classes plus
//                               the narrow formats' widening; fa/fb random,
//                               they must be ignored). Bit identity.
//   mac_check single8 SEED      Every pair of FP8 elements in every slot, the
//                               other slots zero products of the same sign, so
//                               the group sum is the one product, exact: y must
//                               equal the binary32 FMA on the exactly widened
//                               pair (the current lane, fmt FP32). DP4 (E4M3 x
//                               E4M3, 4 slots) and DP2 with the four 8-bit code
//                               pairs (2 slots, the slots' high bytes random).
//   mac_check single16 N SEED   DP2, the same with N random pairs per (code
//                               pair, slot), all 16 code pairs.
//   mac_check dump CLASS N SEED N multi-product groups of a stimulus class, one
//                               line each for check_mac.py (the exact oracle,
//                               tools/model/fdp.py lane_mac_ref):
//                               mode fa fb a b c sub_c y
// Exit status 1 on any mismatch.
#include "Vmac_tb.h"
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

static int LAT_NEW = 4;
static const int LAT_REF = 3;

struct Vec {
    uint8_t mode, fmt, fa, fb, sub, mo, ftz;
    uint32_t a, b, c;
    uint8_t rfmt, rsub, rmo, rftz;
    uint32_t ra, rb, rc;
    uint32_t y, yref;
    bool chk;
    long idx;
};

static Vmac_tb* m;
static Vec ring[64];
static long cyc = 0;

static void reset() {
    m->rst_n = 0; m->clk = 0; m->eval();
    for (int i = 0; i < 4; ++i) { m->clk = 1; m->eval(); m->clk = 0; m->eval(); }
    m->rst_n = 1;
}

static Vec* step(const Vec& v) {
    Vec& r = ring[cyc & 63];
    r = v;
    m->mode = v.mode; m->fmt = v.fmt; m->fa = v.fa; m->fb = v.fb;
    m->a = v.a; m->b = v.b; m->c = v.c; m->sub_c = v.sub; m->mul_only = v.mo; m->ftz = v.ftz;
    m->rfmt = v.rfmt; m->ra = v.ra; m->rb = v.rb; m->rc = v.rc;
    m->rsub = v.rsub; m->rmo = v.rmo; m->rftz = v.rftz;
    m->eval();
    Vec* done = nullptr;
    if (cyc >= LAT_NEW) ring[(cyc - LAT_NEW) & 63].y = m->y;
    if (cyc >= LAT_REF) ring[(cyc - LAT_REF) & 63].yref = m->yref;
    int L = LAT_NEW > LAT_REF ? LAT_NEW : LAT_REF;
    if (cyc >= L) done = &ring[(cyc - L) & 63];
    m->clk = 1; m->eval(); m->clk = 0; m->eval();
    ++cyc;
    return done;
}

static long fails = 0, checked = 0;
static void check(const Vec* d) {
    if (!d || !d->chk) return;
    ++checked;
    if (d->y != d->yref) {
        if (fails < 30)
            printf("MISMATCH #%ld mode %d fmt %d f %d%d a %08x b %08x c %08x sub %d mo %d ftz %d"
                   " | ref fmt %d a %08x b %08x c %08x | got %08x want %08x\n",
                   d->idx, d->mode, d->fmt, d->fa, d->fb, d->a, d->b, d->c, d->sub, d->mo,
                   d->ftz, d->rfmt, d->ra, d->rb, d->rc, d->y, d->yref);
        ++fails;
    }
}
static void flush(void (*sink)(const Vec*)) {
    Vec z; memset(&z, 0, sizeof z); z.idx = -1;
    for (int i = 0; i < 10; ++i) sink(step(z));
}

//---------------------------------------------------------------- formats
// exact widening to binary32 bits (the oracle's arithmetic, not the RTL's)
static uint32_t widen8(uint32_t x, int e5m2) {
    uint32_t sg = (x >> 7) & 1; float v;
    if (!e5m2) {
        int e = (x >> 3) & 15, f = x & 7;
        if (e == 15 && f == 7) return 0x7fc00000;
        v = e == 0 ? ldexpf((float)f, -9) : ldexpf((float)(8 + f), e - 10);
    } else {
        int e = (x >> 2) & 31, f = x & 3;
        if (e == 31) return f ? 0x7fc00000 : (sg << 31) | 0x7f800000;
        v = e == 0 ? ldexpf((float)f, -16) : ldexpf((float)(4 + f), e - 17);
    }
    return f2u(v) | (sg << 31);
}
static uint32_t widen16(uint32_t x, int bf16) {
    uint32_t sg = (x >> 15) & 1;
    if (bf16) return x << 16;
    int e = (x >> 10) & 31, f = x & 1023;
    if (e == 31) return f ? 0x7fc00000 : (sg << 31) | 0x7f800000;
    float v = e == 0 ? ldexpf((float)f, -24) : ldexpf((float)(1024 + f), e - 25);
    return f2u(v) | (sg << 31);
}
// DP2 element codes: 0 FP16, 1 BF16, 2 E5M2, 3 E4M3 (8-bit: the slot's low byte)
static uint32_t widenc(uint32_t x, int code) {
    switch (code) {
    case 0: return widen16(x & 0xffff, 0);
    case 1: return widen16(x & 0xffff, 1);
    case 2: return widen8(x & 0xff, 1);
    default: return widen8(x & 0xff, 0);
    }
}
static uint32_t sbit(int code) { return code >= 2 ? 0x80 : 0x8000; }

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
static uint32_t addend(int e2) {
    switch (rnd() % 12) {
    case 0: return (rnd() & 1) << 31;
    case 1: return mk(rnd() & 1, 0, frac23());
    case 2: return mk(rnd() & 1, 255, rnd() % 4 ? 0 : frac23());
    case 3: return r32();
    case 4: return mk(rnd() & 1, 254, frac23());
    default: return mk(rnd() & 1, e2 + 127 + (int)(rnd() % 61) - 30, frac23());
    }
}

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
            default: {
                    fmt = 1 + rnd() % 4;                       // BF16 F16 E5M2 E4M3
                    a = r32(); b = r32();
                    c = rnd() % 3 ? addend((int)(rnd() % 60) - 30) : r32(); break; }
            }
            v.mode = 0; v.fmt = fmt; v.fa = rnd() & 3; v.fb = rnd() & 3;
            v.a = a; v.b = b; v.c = c; v.sub = k & 1; v.mo = (k >> 1) & 1; v.ftz = (k >> 2) & 1;
            v.rfmt = fmt; v.ra = a; v.rb = b; v.rc = c; v.rsub = v.sub; v.rmo = v.mo; v.rftz = v.ftz;
            v.chk = true;
            check(step(v));
        }
        flush(check);
        printf("miter class %d: %ld vectors, %ld mismatches\n", cls, n, fails - f0);
        fflush(stdout);
    }
}

// a group with one product in slot k; the other slots are zero products with
// that product's sign. DP4: E4M3 bytes. DP2: 16-bit slots, codes fa/fb; an
// 8-bit code's slot gets a random high byte (ignored by the lane).
static Vec single_vec(int mode, int fa, int fb, int k, uint32_t ea, uint32_t eb, uint32_t c, int sub) {
    Vec v; memset(&v, 0, sizeof v);
    int w = mode == 2 ? 8 : 16, g = 32 / w;
    uint32_t sa = mode == 2 ? 0x80 : sbit(fa), sb = mode == 2 ? 0x80 : sbit(fb);
    uint32_t sgn = ((ea & sa) != 0) ^ ((eb & sb) != 0);
    uint32_t a = 0, b = 0;
    for (int j = 0; j < g; ++j) {
        uint32_t aj = j == k ? ea : (sgn ? sa : 0), bj = j == k ? eb : 0;
        if (mode == 1 && fa >= 2) aj = (aj & 0xff) | ((r32() & 0xff) << 8);
        if (mode == 1 && fb >= 2) bj = (bj & 0xff) | ((r32() & 0xff) << 8);
        a |= aj << (w * j); b |= bj << (w * j);
    }
    v.mode = mode; v.fmt = 0; v.fa = mode == 1 ? fa : (rnd() & 3); v.fb = mode == 1 ? fb : (rnd() & 3);
    v.a = a; v.b = b; v.c = c; v.sub = sub;
    v.rfmt = 0; v.rc = c; v.rsub = sub;
    v.ra = mode == 2 ? widen8(ea, 0) : widenc(ea, fa);
    v.rb = mode == 2 ? widen8(eb, 0) : widenc(eb, fb);
    v.chk = true;
    return v;
}
static uint32_t c_for(uint32_t p, int i) {
    switch (i % 8) {
    case 0: return 0;
    case 1: return 0x80000000;
    case 2: return p ^ 0x80000000u;
    case 3: return (p ^ 0x80000000u) + ((int)(rnd() % 5) - 2);
    case 4: return mk(rnd() & 1, ((p >> 23) & 255) - 24 - (int)(rnd() % 8), frac23());
    case 5: return mk(rnd() & 1, ((p >> 23) & 255) + 24 + (int)(rnd() % 8), frac23());
    case 6: return addend((int)((p >> 23) & 255) - 127);
    default: return r32();
    }
}
static void run_single8() {
    long idx = 0;
    {   long f0 = fails, c0 = checked;
        for (int k = 0; k < 4; ++k)
            for (uint32_t ea = 0; ea < 256; ++ea)
                for (uint32_t eb = 0; eb < 256; ++eb) {
                    uint32_t p = f2u(u2f(widen8(ea, 0)) * u2f(widen8(eb, 0)));
                    Vec v = single_vec(2, 3, 3, k, ea, eb, c_for(p, (int)(ea * 7 + eb * 3 + k)), rnd() & 1);
                    v.idx = idx++;
                    check(step(v));
                }
        flush(check);
        printf("single8 DP4 E4M3xE4M3: 4 slots x 65536 pairs, %ld checked, %ld mismatches\n", checked - c0, fails - f0);
        fflush(stdout); }
    for (int fa = 2; fa < 4; ++fa)
        for (int fb = 2; fb < 4; ++fb) {
            long f0 = fails, c0 = checked;
            for (int k = 0; k < 2; ++k)
                for (uint32_t ea = 0; ea < 256; ++ea)
                    for (uint32_t eb = 0; eb < 256; ++eb) {
                        uint32_t p = f2u(u2f(widenc(ea, fa)) * u2f(widenc(eb, fb)));
                        Vec v = single_vec(1, fa, fb, k, ea, eb, c_for(p, (int)(ea * 5 + eb * 3 + k)), rnd() & 1);
                        v.idx = idx++;
                        check(step(v));
                    }
            flush(check);
            printf("single8 DP2 codes %d x %d: 2 slots x 65536 pairs, %ld checked, %ld mismatches\n",
                   fa, fb, checked - c0, fails - f0);
            fflush(stdout);
        }
}
static uint32_t elem16(int bf) {
    switch (rnd() % 8) {
    case 0: return rnd() & 0x8000;
    case 1: return (rnd() & 0x8000) | (rnd() % 4 ? (r32() & (bf ? 0x7f : 0x3ff)) : 1);
    case 2: return (rnd() & 0x8000) | (bf ? 0x7f80 : 0x7c00) | (rnd() % 3 ? 0 : (r32() & (bf ? 0x7f : 0x3ff)));
    case 3: return (rnd() & 0x8000) | ((bf ? 0x7f7f : 0x7bff) ^ (rnd() % 4));
    default: return r32() & 0xffff;
    }
}
static uint32_t elem8(int e5m2) {
    switch (rnd() % 8) {
    case 0: return rnd() & 0x80;
    case 1: return (rnd() & 0x80) | (r32() & (e5m2 ? 3 : 7));
    case 2: return (rnd() & 0x80) | (e5m2 ? 0x7c | (rnd() % 4) : 0x7f);
    case 3: return (rnd() & 0x80) | (e5m2 ? 0x7b : 0x7e);
    default: return r32() & 0xff;
    }
}
static uint32_t elemc(int code) { return code < 2 ? elem16(code) : elem8(code == 2); }
static void run_single16(long n) {
    long idx = 0;
    for (int fa = 0; fa < 4; ++fa)
        for (int fb = 0; fb < 4; ++fb) {
            long f0 = fails, c0 = checked;
            for (int k = 0; k < 2; ++k)
                for (long i = 0; i < n; ++i) {
                    uint32_t ea = elemc(fa), eb = elemc(fb);
                    float pf = u2f(widenc(ea, fa)) * u2f(widenc(eb, fb));
                    Vec v = single_vec(1, fa, fb, k, ea, eb, c_for(f2u(pf), (int)i), rnd() & 1);
                    v.idx = idx++;
                    check(step(v));
                }
            flush(check);
            printf("single16 codes %d x %d: 2 slots x %ld pairs, %ld checked, %ld mismatches\n",
                   fa, fb, n, checked - c0, fails - f0);
            fflush(stdout);
        }
}

//---------------------------------------------------------------- multi-product groups
static uint32_t e8(int e5m2, int cls) {
    if (cls == 3) {
        switch (rnd() % 8) {
        case 0: return e5m2 ? 0x7c | (rnd() % 4) | (rnd() & 0x80) : 0x7f | (rnd() & 0x80);
        case 1: return rnd() & 0x80;
        case 2: return (rnd() & 0x80) | (e5m2 ? 0x7b : 0x7e);
        case 3: return (rnd() & 0x80) | 1;
        default: return r32() & 0xff;
        }
    }
    if (cls == 4) {
        if (rnd() & 1) return (rnd() & 0x80) | (e5m2 ? (0x78 | (rnd() % 4)) : (0x70 | (rnd() % 7)));
        return (rnd() & 0x80) | (e5m2 ? (rnd() % 8) : (rnd() % 16));
    }
    if (cls == 5) {
        int e = e5m2 ? 14 + rnd() % 3 : 6 + rnd() % 3;
        return (rnd() & 0x80) | (e5m2 ? (e << 2) | 3 : (e << 3) | (7 - (rnd() % 2)));
    }
    uint32_t x = r32() & 0xff;
    if (!e5m2 && (x & 0x7f) == 0x7f) x ^= 1;
    if (e5m2 && ((x >> 2) & 31) == 31) x ^= 0x04;
    return x;
}
static uint32_t e16(int bf, int cls) {
    if (cls == 15) return elem16(bf);
    if (cls == 14) {
        switch (rnd() % 4) {
        case 0: return (rnd() & 0x8000) | (bf ? 0x7f00 | (r32() & 0x7f) : 0x7800 | (r32() & 0x3ff));
        case 1: return (rnd() & 0x8000) | (r32() & (bf ? 0x7f : 0x3ff));
        case 2: return (rnd() & 0x8000) | (bf ? 0x7f7f : 0x7bff);
        default: return (rnd() & 0x8000) | (bf ? 0x0080 : 0x0400);
        }
    }
    if (cls == 16) {
        if (bf) return (rnd() & 0x8000) | ((uint32_t)(rnd() % 12) << 7) | (r32() & 0x7f);
        return (rnd() & 0x8000) | (r32() & 0x7ff);
    }
    uint32_t x = r32() & 0xffff;
    if (!bf && ((x >> 10) & 31) == 31) x ^= 0x400;
    if (bf && ((x >> 7) & 255) == 255) x ^= 0x80;
    return x;
}
// a DP2 element of a class: 16-bit codes as e16, 8-bit codes as e8 of the
// matching class, with a random (ignored) high byte
static uint32_t ec(int code, int cls) {
    if (code < 2) return e16(code, cls);
    int c8 = cls == 15 ? 3 : (cls == 14 || cls == 16) ? 4 : 0;
    return ((r32() & 0xff) << 8) | e8(code == 2, c8);
}
static uint32_t big(int code) {
    switch (code) {
    case 0: return (rnd() & 0x8000) | 0x7800 | (r32() & 0x3ff);
    case 1: return (rnd() & 0x8000) | 0x7e00 | (r32() & 0xff);
    case 2: return (rnd() & 0x80) | 0x70 | (rnd() % 12);
    default: return (rnd() & 0x80) | 0x70 | (rnd() % 7);
    }
}
static uint32_t mid(int code) {
    switch (code) {
    case 0: case 1: return (rnd() & 0x8000) | 0x4000 | (r32() & 0x3bff);
    default: return (rnd() & 0x80) | 0x40 | (r32() & 0x1f);
    }
}
static uint32_t tiny(int code) {
    switch (code) {
    case 0: return (rnd() & 0x8000) | (r32() & 0x03ff);
    case 1: return (rnd() & 0x8000) | (r32() & 0x0fff);
    case 2: return (rnd() & 0x80) | (r32() & 0x0f);
    default: return (rnd() & 0x80) | (r32() & 0x0f);
    }
}
typedef long double LD;
static LD valc(uint32_t x, int code) { return (LD)u2f(widenc(x, code)); }

static void dump_line(const Vec* d) {
    if (!d || d->idx < 0) return;
    printf("%d %d %d %08x %08x %08x %d %08x\n", d->mode, d->fa, d->fb, d->a, d->b, d->c, d->sub, d->y);
}
static void run_dump(int cls, long n) {
    for (long i = 0; i < n; ++i) {
        Vec v; memset(&v, 0, sizeof v); v.idx = i;
        int dp4 = cls < 10;
        int fa = rnd() & 3, fb = rnd() & 3;
        uint32_t a = 0, b = 0, c = 0;
        LD sum = 0, big_p = 0;
        if (dp4) {
            uint32_t ea[4], eb[4];
            for (int k = 0; k < 4; ++k) { ea[k] = e8(0, cls); eb[k] = e8(0, cls); }
            if (cls == 1) {
                ea[1] = ea[0] ^ 0x80; eb[1] = eb[0];
                ea[2] = (rnd() & 0x80) | (rnd() % 8); eb[2] = (rnd() & 0x80) | (rnd() % 8);
                if (rnd() & 1) { ea[3] = ea[2] ^ 0x80; eb[3] = eb[2] ^ (rnd() % 2); }
            }
            for (int k = 0; k < 4; ++k) { a |= ea[k] << (8 * k); b |= eb[k] << (8 * k); sum += valc(ea[k], 3) * valc(eb[k], 3); }
            v.mode = 2; fa = rnd() & 3; fb = rnd() & 3;    // ignored in DP4
        } else {
            uint32_t ea[2], eb[2];
            for (int k = 0; k < 2; ++k) { ea[k] = ec(fa, cls); eb[k] = ec(fb, cls); }
            if (cls == 11) {                                 // near cancellation
                ea[1] = ea[0] ^ sbit(fa);
                eb[1] = eb[0] ^ (rnd() % 4);
            }
            if (cls == 12) {                                 // a big and a tiny product
                ea[0] = mid(fa); eb[0] = big(fb);
                ea[1] = tiny(fa); eb[1] = tiny(fb);
                if (rnd() & 1) { uint32_t t; t = ea[0]; ea[0] = ea[1]; ea[1] = t; t = eb[0]; eb[0] = eb[1]; eb[1] = t; }
            }
            for (int k = 0; k < 2; ++k) {
                a |= (ea[k] & 0xffff) << (16 * k); b |= (eb[k] & 0xffff) << (16 * k);
                LD p = valc(ea[k], fa) * valc(eb[k], fb);
                sum += p; if (fabsl(p) > fabsl(big_p)) big_p = p;
            }
            v.mode = 1;
        }
        int e2 = 0;
        if (std::isfinite((double)sum) && sum != 0) { int ex; frexpl(sum, &ex); e2 = ex; }
        switch (cls) {
        case 2: case 13: case 12: {
            float cf = (float)(-sum);
            c = f2u(cf);
            if (std::isfinite(cf) && rnd() % 3) c += (int)(rnd() % 5) - 2;
            if (cls == 12 && rnd() % 2) c = f2u((float)(-big_p));
            break; }
        case 1: case 16: c = rnd() % 2 ? ((rnd() & 1) << 31) : mk(rnd() & 1, rnd() % 3, frac23()); break;
        default: c = addend(e2); break;
        }
        v.a = a; v.b = b; v.c = c; v.fa = fa; v.fb = fb; v.sub = (rnd() % 4) == 0;
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
    m = new Vmac_tb;
    reset();
    if (cmd == "miter") {
        long n = atol(argv[2]); seed = argc > 3 ? atol(argv[3]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_miter(n);
        printf("miter: %ld per class x 8 classes, %ld checked, %ld mismatches: %s\n",
               n, checked, fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "single8") {
        seed = argc > 2 ? atol(argv[2]) : 1; s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_single8();
        printf("single8: %ld checked, %ld mismatches: %s\n", checked, fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "single16") {
        long n = atol(argv[2]); seed = argc > 3 ? atol(argv[3]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull; s[1] ^= seed * 3;
        run_single16(n);
        printf("single16: %ld checked, %ld mismatches: %s\n", checked, fails, fails ? "FAIL" : "PASS");
    } else if (cmd == "dump") {
        int cls = atoi(argv[2]); long n = atol(argv[3]); seed = argc > 4 ? atol(argv[4]) : 1;
        s[0] ^= seed * 0x9e3779b97f4a7c15ull ^ (uint64_t)cls * 0x51ull; s[1] ^= seed * 3 + cls;
        run_dump(cls, n);
    }
    delete m;
    return fails ? 1 : 0;
}
