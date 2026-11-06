// Hypothetical independent stream-outlet experiment using the existing
// controller bridge. This is not a tnca-chip execution or a board prediction.
#include "tnca_dram_bridge.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using U64 = std::uint64_t;
struct Trace { std::vector<tnca_dram::Request> requests; };
void decode(const void* ctx, std::size_t i, std::uint32_t& pc, U64& offset, bool& write) {
  const auto& r = static_cast<const Trace*>(ctx)->requests.at(i);
  pc = r.pc; offset = r.offset; write = r.write;
}
void need(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
tnca_dram::Result run(const std::string& yaml, unsigned pcs, Trace& trace,
                      const tnca_dram::StreamAdmission* admission, std::vector<std::uint32_t>& latency) {
  std::vector<U64> arrivals;
  for (const auto& r : trace.requests) arrivals.push_back(r.arrival);
  tnca_dram::Batch b{trace.requests.size(), arrivals.data(), &trace, decode};
  auto result = tnca_dram::replay(yaml, pcs, b, latency, true, true, admission);
  return result;
}
void tests(const std::string& yaml) {
  // Sparse one-credit reads admit at the same edges as an independent replay
  // with explicit arrivals. This checks credit return and kernel-edge rounding.
  for (bool write : {false, true}) {
    Trace sparse;
    for (unsigned i = 0; i < 40; ++i) sparse.requests.push_back({0, i * 64, write, i * 900});
    std::vector<std::uint32_t> plain_lat, bounded_lat;
    auto plain = run(yaml, 1, sparse, nullptr, plain_lat);
    tnca_dram::StreamAdmission a{1, 9, 1, 20};
    auto bounded = run(yaml, 1, sparse, &a, bounded_lat);
    need(plain.error.empty() && bounded.error.empty() && plain_lat == bounded_lat,
         "sparse bounded replay differs from explicit-arrival replay");
    need(bounded.pcs[0].peak_line_credits == 1, "single-credit peak");
    const auto sparse_lat = plain_lat;
    // For a dense single-credit trace, arrival of the next line is exactly the
    // kernel-edge-rounded prior completion plus the declared return delay.
    Trace dense;
    for (unsigned i = 0; i < 40; ++i) dense.requests.push_back({0, i * 64, write, 0});
    auto serial = run(yaml, 1, dense, &a, bounded_lat);
    need(serial.error.empty(), "dense serial replay failed");
    for (unsigned i = 1; i < 40; ++i)
      dense.requests[i].arrival = ((U64(bounded_lat[i-1]) + 180 + 8) / 9) * 9;
    plain = run(yaml, 1, dense, nullptr, plain_lat);
    need(plain.error.empty(), "explicit serial replay failed");
    for (unsigned i = 0; i < 40; ++i)
      need(dense.requests[i].arrival + plain_lat[i] == bounded_lat[i], "serial credit timing mismatch");
    // Independent PCs cannot share a credit pool.
    Trace two;
    for (const auto& r : sparse.requests) {
      two.requests.push_back(r);
      auto other = r; other.pc = 1; two.requests.push_back(other);
    }
    auto independent = run(yaml, 2, two, &a, bounded_lat);
    need(independent.error.empty(), "two-PC replay failed");
    for (unsigned i = 0; i < 40; ++i) {
      need(bounded_lat[2*i] == bounded_lat[2*i+1], "PC independence");
      need(bounded_lat[2*i] == sparse_lat[i], "independent-PC timing against sparse baseline");
    }
    need(independent.pcs[0].accepted_bursts == 80 && independent.pcs[1].completed_bursts == 80,
         "two-PC burst accounting");
    tnca_dram::StreamAdmission invalid{0, 9, 1, 20};
    need(!run(yaml, 1, sparse, &invalid, plain_lat).error.empty(), "zero credit window accepted");
  }
}
}

int main(int argc, char** argv) {
  try {
    need(argc == 2, "usage: tnca-stream-probe HBM.yaml");
    const std::string yaml = argv[1];
    auto timing = tnca_dram::timing(yaml);
    need(timing.error.empty(), "HBM timing resolution failed");
    need(timing.tCK_ps == 1111, "experiment requires nominal 900 MHz HBM timing");
    tests(yaml);
    std::cout << "{\"qualification\":\"hypothetical-independent-outlet-model-only\","
      "\"tests\":\"pass\",\"kernel_hz\":100000000,\"dram_hz_nominal\":900000000,"
      "\"return_kernel_cycles\":20,\"yaml_sha256\":\"" << tnca_dram::sha256_file(yaml) << "\",\"cases\":[\n";
    bool first = true;
    for (unsigned lines : {1024U, 16384U})
      for (const std::string mode : {"read", "write", "mixed"})
        for (unsigned interval : {1U, 2U, 3U})
          for (unsigned window : {8U, 32U, 64U}) {
            Trace trace;
            for (unsigned i = 0; i < lines; ++i) {
              bool wr = mode == "write" || (mode == "mixed" && (i & 1));
              U64 offset = mode == "mixed" ? U64(i / 2) * 64 + (wr ? (1ULL << 26) : 0) : U64(i) * 64;
              trace.requests.push_back({0, offset, wr, 0});
            }
            std::vector<std::uint32_t> latency;
            tnca_dram::StreamAdmission a{window, 9, interval, 20};
            auto r = run(yaml, 1, trace, &a, latency);
            need(r.error.empty(), r.error.c_str());
            const auto& s = r.pcs.at(0);
            need(s.accepted_bursts == U64(lines) * 2 && s.completed_bursts == s.accepted_bursts,
                 "burst completion mismatch");
            need(s.reads + s.writes == lines && s.peak_line_credits <= window, "line credit accounting");
            U64 cycles = (U64(*std::max_element(latency.begin(), latency.end())) + 180 + 8) / 9;
            double gbps = double(lines) * 64 * 0.1 / double(cycles);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "{\"lines\":" << lines << ",\"mode\":\"" << mode << "\",\"issue_kernel_cycles\":" << interval
              << ",\"window_lines\":" << window << ",\"kernel_cycles\":" << cycles
              << ",\"per_pc_GBps\":" << gbps << ",\"peak_credits\":" << s.peak_line_credits
              << ",\"queue_refusals\":" << s.queue_refusals << ",\"refreshes\":" << s.refreshes << "}";
          }
    std::cout << "\n]}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "stream probe failed: " << e.what() << '\n'; return 1;
  }
}
