// cvt_int_miter.cpp -- exhaustive comparison of tnca_cvt_int with cvt_int_f: every 32-bit word for each (fn, rm) pair named on
// the command line, plus random coverage of the other function codes and rounding modes.
//   cvt_int_miter fn rm [lo hi]   exhaustive over x in [lo, hi) (default the whole 2^32)
//   cvt_int_miter random N SEED   N random (x, fn, rm) triples, fn over all 256 codes, rm over all 8
#include "Vcvt_int_miter.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
static uint64_t s[2] = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
static inline uint64_t rnd() { uint64_t x = s[0], y = s[1]; s[0] = y; x ^= x << 23; s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return s[1] + y; }
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vcvt_int_miter m;
    long fails = 0, n = 0;
    if (argc > 1 && !strcmp(argv[1], "random")) {
        long N = atol(argv[2]); s[0] ^= strtoull(argv[3], 0, 0) * 0x9e3779b97f4a7c15ull; for (int i = 0; i < 8; ++i) rnd();
        for (long i = 0; i < N; ++i) {
            uint32_t x = (uint32_t)rnd(); if (rnd() % 3 == 0) x = (x & 0x807fffffu) | ((uint32_t)(100 + rnd() % 70) << 23);
            m.x = x; m.fn = (rnd() % 4 == 0) ? (rnd() & 0xff) : (rnd() & 3); m.rm = rnd() & 7; m.eval();
            if (m.y_ref != m.y_new) { if (fails++ < 20) printf("MISMATCH x %08x fn %02x rm %x ref %08x new %08x\n", x, m.fn, m.rm, m.y_ref, m.y_new); }
            ++n;
        }
    } else {
        int fn = atoi(argv[1]), rm = atoi(argv[2]);
        uint64_t lo = argc > 3 ? strtoull(argv[3], 0, 0) : 0, hi = argc > 4 ? strtoull(argv[4], 0, 0) : (1ull << 32);
        m.fn = fn; m.rm = rm;
        for (uint64_t x = lo; x < hi; ++x) {
            m.x = (uint32_t)x; m.eval();
            if (m.y_ref != m.y_new) { if (fails++ < 20) printf("MISMATCH x %08x fn %02x rm %x ref %08x new %08x\n", (uint32_t)x, fn, rm, m.y_ref, m.y_new); }
            ++n;
        }
    }
    printf("cvt_int_miter: %ld vectors, %ld mismatches\n%s\n", n, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
