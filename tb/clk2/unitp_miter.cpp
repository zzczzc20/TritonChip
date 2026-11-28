// exhaustive: every 32-bit a for ftz 0 and 1 (2^33 vectors)
#include "Vunitp_miter.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vunitp_miter m;
    uint64_t fails = 0, n = 0;
    for (int ftz = 0; ftz < 2; ++ftz) {
        m.ftz = ftz;
        for (uint64_t a = 0; a < (1ull << 32); ++a) {
            m.a = (uint32_t)a; m.eval();
            if (m.y_ref != m.y_new) { if (fails++ < 10) printf("MISMATCH a %08x ftz %d ref %016llx new %016llx\n", (uint32_t)a, ftz, (unsigned long long)m.y_ref, (unsigned long long)m.y_new); }
            ++n;
        }
    }
    printf("unitp_miter: %llu vectors, %llu mismatches\n%s\n", (unsigned long long)n, (unsigned long long)fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
