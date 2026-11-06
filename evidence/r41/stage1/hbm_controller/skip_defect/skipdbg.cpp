#include <cstdio>
#include <cstdint>
#include <vector>
#include "tnca_dram_bridge.hpp"
static std::uint64_t line(int bg, int bank, int row, int col) {
  std::uint64_t burst = (std::uint64_t(col & 31) << 1) | (std::uint64_t(bank) << 6) | ((std::uint64_t(bg) >> 1) << 8) | (std::uint64_t(row) << 9);
  return burst << 5;
}
int main(int, char** argv) {
  using tnca_dram::Request;
  std::vector<Request> q;
  std::uint64_t seed = 12345, at = 1000;
  auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return seed >> 33; };
  for (int i = 0; i < 200000; ++i) {
    std::uint64_t r = rnd();
    at += (r % 7 == 0) ? (r % 5000) : (r % 23);
    int kind = int(r % 4);
    std::uint64_t off;
    if (kind == 0) off = line(0, 0, int(i % 64), int((i / 64) % 32));
    else if (kind == 1) off = line(2, 1, int((i / 32) % 512), int(i % 32));
    else off = (rnd() % (std::uint64_t(256) << 20)) & ~std::uint64_t(63);
    q.push_back({0, off, (r % 3) == 0, at});
  }
  auto a = tnca_dram::replay(argv[1], 1, q, false, false), b = tnca_dram::replay(argv[1], 1, q, true, true);
  int shown = 0;
  for (std::size_t i = 0; i < q.size() && shown < 12; ++i) if (a.completion[i] != b.completion[i]) {
    auto m = tnca_dram::map_address(q[i].offset);
    std::printf("i=%zu arrival=%llu %s off=0x%llx bg=%d ba=%d row=%d col=%d noskip=%llu skip=%llu lat=%lld/%lld\n", i,
      (unsigned long long)q[i].arrival, q[i].write ? "W" : "R", (unsigned long long)q[i].offset, m.bankgroup, m.bank, m.row, m.column,
      (unsigned long long)a.completion[i], (unsigned long long)b.completion[i],
      (long long)(a.completion[i]-q[i].arrival), (long long)(b.completion[i]-q[i].arrival));
    ++shown;
  }
  std::printf("refreshes %llu/%llu conflicts %llu/%llu\n", (unsigned long long)a.pcs[0].refreshes, (unsigned long long)b.pcs[0].refreshes,
    (unsigned long long)a.pcs[0].row_conflicts, (unsigned long long)b.pcs[0].row_conflicts);
}
