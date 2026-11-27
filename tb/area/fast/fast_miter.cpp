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

#include "fma_vectors.h"
int main(int argc, char** argv) {
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    if (argc > 2) { s[0] ^= strtoull(argv[2], 0, 0) * 0x9e3779b97f4a7c15ull; s[1] ^= strtoull(argv[2], 0, 0) * 3; for (int i = 0; i < 8; ++i) rnd(); }
    Verilated::commandArgs(argc, argv);
    Vmiter_all top;
    const int CLASSES = FMA_CLASSES;
    long fails = 0, total = 0, nfail = 0;
    for (int cls = 0; cls < CLASSES; ++cls) {
        long cf = 0;
        for (long i = 0; i < n; ++i) {
            uint32_t a, b, c, k;
            gen_fma(cls, a, b, c, k);
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
