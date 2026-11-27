// fma_cyc_miter.cpp -- Verilator driver: tnca_fmax_fma against the generated FMA at equal latency, N vectors per class,
// one per cycle (pipeline full), plus idle bubbles. Usage: fma_cyc_miter N SEED LAT
#include "Vfma_cyc_miter.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include "fma_vectors.h"
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    if (argc > 2) { s[0] ^= strtoull(argv[2], 0, 0) * 0x9e3779b97f4a7c15ull; s[1] ^= strtoull(argv[2], 0, 0) * 3; for (int i = 0; i < 8; ++i) rnd(); }
    int LAT = argc > 3 ? atoi(argv[3]) : 2;
    Vfma_cyc_miter m;
    m.clk = 0; m.rst_n = 0; m.a = m.b = m.c = 0; m.sub_c = m.mul_only = m.ftz = 0;
    for (int i = 0; i < 4; ++i) { m.clk = 1; m.eval(); m.clk = 0; m.eval(); }
    m.rst_n = 1;
    long fails = 0, np = 0;
    uint32_t ha[8], hb[8], hc[8], hk[8];
    long cyc = 0;
    for (int cls = 0; cls < FMA_CLASSES; ++cls) {
        long cf = 0;
        for (long i = 0; i < n + LAT + 2; ++i, ++cyc) {
            uint32_t a = 0, b = 0, c = 0, k = 0;
            if (i < n) gen_fma(cls, a, b, c, k);
            m.a = a; m.b = b; m.c = c; m.sub_c = k & 1; m.mul_only = (k >> 1) & 1; m.ftz = (k >> 2) & 1;
            ha[cyc & 7] = a; hb[cyc & 7] = b; hc[cyc & 7] = c; hk[cyc & 7] = k;
            m.eval();
            if (i >= LAT && i - LAT < n && m.y_ref != m.y_new) {
                ++cf; ++fails; if (np++ < 20) { long j = (cyc - LAT) & 7; printf("MISMATCH cls %d a %08x b %08x c %08x k %x ref %08x new %08x\n", cls, ha[j], hb[j], hc[j], hk[j], m.y_ref, m.y_new); }
            }
            m.clk = 1; m.eval(); m.clk = 0; m.eval();
        }
        printf("class %d: %ld vectors, %ld mismatches\n", cls, n, cf); fflush(stdout);
    }
    printf("fma_cyc_miter LAT %d: %ld mismatches\n%s\n", LAT, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
