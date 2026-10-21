// Unit tests of the Ramulator2 bridge (the slow DRAM path), printed as JSON.
//   tnca-dram-bridge-test CONFIG.yaml
// Each test sends 64-byte lines (two 32-byte bursts in two bank groups) to one
// pseudo channel and checks the completion times against the HBM2 timing the
// configuration resolves (DRAM clock cycles).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tnca_dram_bridge.hpp"

using tnca_dram::Request;

static int failures = 0;
static std::string json;

static void check(const char* name, bool ok, const std::string& detail) {
  if (!ok) ++failures;
  if (!json.empty()) json += ",\n";
  json += "  {\"test\": \"" + std::string(name) + "\", \"pass\": " + (ok ? "true" : "false") + ", " + detail + "}";
}

// Offset of a line in (bank group, bank, row, column) per the bridge's mapping;
// bg must be even (a line's second burst is in bank group bg+1).
static std::uint64_t line(int bg, int bank, int row, int col) {
  // bursts: [bg 2 bits][bank 2 bits][col 5 bits][row 14 bits]; a line = bursts 2k, 2k+1
  std::uint64_t burst = std::uint64_t(bg) | (std::uint64_t(bank) << 2) | (std::uint64_t(col) << 4) | (std::uint64_t(row) << 9);
  return burst << 5;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s CONFIG.yaml\n", argv[0]);
    return 2;
  }
  std::string cfg = argv[1];
  auto t = tnca_dram::timing(cfg);
  if (!t.error.empty()) {
    std::fprintf(stderr, "%s\n", t.error.c_str());
    return 2;
  }
  auto run = [&](std::vector<Request> reqs) {
    auto r = tnca_dram::replay(cfg, 1, reqs);
    if (!r.error.empty()) {
      std::fprintf(stderr, "replay error: %s\n", r.error.c_str());
      std::exit(2);
    }
    return r;
  };
  auto lat = [](const tnca_dram::Result& r, const std::vector<Request>& q, size_t i) {
    return (long long)(r.completion[i] - q[i].arrival);
  };
  char buf[512];
  const long long idle_min = t.nRCDRD + t.read_latency;  // ACT -> RD -> data (tRCD + tCL + tBL)

  // (1) One line read on an idle pseudo channel: ACT, tRCD, RD, tCL + burst.
  {
    std::vector<Request> q{{0, line(0, 0, 5, 0), false, 1000}};
    auto r = run(q);
    long long l = lat(r, q, 0);
    std::snprintf(buf, sizeof buf, "\"latency\": %lld, \"tRCD+tCL+tBL\": %lld, \"row_misses\": %llu", l, idle_min,
                  (unsigned long long)r.pcs[0].row_misses);
    check("idle_read", l >= idle_min && l <= idle_min + 8 && r.pcs[0].row_misses == 2, buf);
  }
  // (2) Row hit vs row miss vs row conflict on one bank (each request after the previous completes).
  {
    std::vector<Request> q{{0, line(0, 2, 7, 0), false, 1000},
                           {0, line(0, 2, 7, 1), false, 2000},    // same row: hit
                           {0, line(0, 2, 9, 0), false, 3000}};   // other row: conflict
    auto r = run(q);
    long long miss = lat(r, q, 0), hit = lat(r, q, 1), conflict = lat(r, q, 2);
    std::snprintf(buf, sizeof buf,
                  "\"miss\": %lld, \"hit\": %lld, \"conflict\": %lld, \"tRP\": %d, \"row_hits\": %llu, \"row_conflicts\": %llu",
                  miss, hit, conflict, t.nRP, (unsigned long long)r.pcs[0].row_hits,
                  (unsigned long long)r.pcs[0].row_conflicts);
    check("row_hit_miss_conflict",
          hit < miss && conflict >= miss + t.nRP - 2 && hit >= t.read_latency && r.pcs[0].row_hits == 2 &&
              r.pcs[0].row_conflicts == 2,
          buf);
  }
  // (3) Bank parallelism: 16 lines at once to 8 distinct banks vs to one bank (distinct rows).
  {
    std::vector<Request> spread, same;
    for (int i = 0; i < 16; ++i) {
      spread.push_back({0, line(2 * (i % 2), (i / 2) % 4, 100 + i, 0), false, 1000});
      same.push_back({0, line(0, 0, 100 + i, 0), false, 1000});
    }
    auto a = run(spread), b = run(same);
    long long ma = 0, mb = 0;
    for (size_t i = 0; i < 16; ++i) {
      ma = std::max<long long>(ma, lat(a, spread, i));
      mb = std::max<long long>(mb, lat(b, same, i));
    }
    std::snprintf(buf, sizeof buf, "\"makespan_spread\": %lld, \"makespan_one_bank\": %lld, \"tRC\": %d", ma, mb,
                  t.nRAS + t.nRP);
    check("bank_parallelism", mb >= 15 * (t.nRAS + t.nRP) && ma * 4 < mb, buf);
  }
  // (4) Refresh on a long stream: one line every 40 cycles to one open row for 10 tREFI.
  {
    std::vector<Request> q;
    const int n = 10 * t.nREFI / 40;
    for (int i = 0; i < n; ++i) q.push_back({0, line(2, 3, 11, i % 32), false, std::uint64_t(1000 + 40 * i)});
    auto r = run(q);
    long long base = 1 << 30, worst = 0;
    int spikes = 0;
    for (size_t i = 0; i < q.size(); ++i) {
      long long l = lat(r, q, i);
      base = std::min(base, l);
      worst = std::max(worst, l);
      if (l > base + t.nRFC / 2) ++spikes;
    }
    std::snprintf(buf, sizeof buf,
                  "\"requests\": %d, \"refreshes\": %llu, \"min_latency\": %lld, \"max_latency\": %lld, \"tRFC\": %d, "
                  "\"slow_requests\": %d, \"skipped_cycles\": %llu",
                  n, (unsigned long long)r.pcs[0].refreshes, base, worst, t.nRFC, spikes,
                  (unsigned long long)r.pcs[0].skipped);
    check("refresh_on_stream", r.pcs[0].refreshes >= 9 && worst >= base + t.nRFC / 2 && spikes >= 9, buf);
  }
  // (5) Idle gaps are skipped but a refresh due in a gap still closes the row.
  {
    std::vector<Request> q{{0, line(0, 1, 3, 0), false, 1000}, {0, line(0, 1, 3, 1), false, 1000 + 5 * (std::uint64_t)t.nREFI + 7}};
    auto r = run(q);
    std::snprintf(buf, sizeof buf, "\"second_latency\": %lld, \"skipped_cycles\": %llu, \"ticks\": %llu, \"refreshes\": %llu",
                  lat(r, q, 1), (unsigned long long)r.pcs[0].skipped, (unsigned long long)r.pcs[0].ticks,
                  (unsigned long long)r.pcs[0].refreshes);
    check("idle_skip_with_refresh", r.pcs[0].skipped > 4ull * t.nREFI && r.pcs[0].row_hits == 0, buf);
  }
  // (6) Both bursts must complete after WR + WL + burst.
  {
    std::vector<Request> q{{0, line(2, 1, 4, 0), true, 1000}};
    auto r = run(q);
    long long l = lat(r, q, 0);
    std::snprintf(buf, sizeof buf, "\"write_latency\": %lld, \"tRCDWR+tCWL+tBL\": %d", l, t.nRCDWR + t.nCWL + t.nBL);
    const auto& stats = r.pcs[0];
    check("write", l >= t.nRCDWR + t.nCWL + t.nBL && l <= t.nRCDWR + t.nCWL + t.nBL + 10 &&
          stats.accepted_bursts == 2 && stats.completed_bursts == 2 && stats.write_callbacks == 2 &&
          r.completion[0] == stats.last_write_complete &&
          stats.last_write_complete - stats.last_write_issue == std::uint64_t(t.nCWL + t.nBL), buf);
  }
  // Queue pressure includes refusal between the two halves of a line.
  // Every accepted 32-byte burst must complete exactly once; none is lost
  // or duplicated by retries. This profile has a 32-entry write queue.
  {
    std::vector<Request> q;
    for (int i=0;i<96;++i) q.push_back({0,line(0,0,100+i,0),true,1000});
    auto r=run(q);const auto& stats=r.pcs[0];
    bool complete=true;
    for (std::size_t i=0;i<q.size();++i) complete &= r.completion[i]>=q[i].arrival+t.nRCDWR+t.nCWL+t.nBL;
    std::snprintf(buf,sizeof buf,"\"lines\": %zu, \"accepted_bursts\": %llu, \"completed_bursts\": %llu, \"queue_refusals\": %llu, \"peak_rw_queued_bursts\": %llu",
        q.size(),(unsigned long long)stats.accepted_bursts,(unsigned long long)stats.completed_bursts,
        (unsigned long long)stats.queue_refusals,(unsigned long long)stats.peak_rw_queued_bursts);
    check("write_queue_backpressure",complete && stats.writes==q.size() && stats.accepted_bursts==2*q.size() &&
          stats.completed_bursts==2*q.size() && stats.queue_refusals>0 && stats.peak_rw_queued_bursts==32,buf);
  }
  // Distinct columns avoid forwarding. A physical read's command time is
  // its recorded depart minus the configured read latency. Compare the
  // real callback clocks with independent HBM2 turnaround constraints.
  {
    std::vector<Request> read{{0,line(0,1,7,0),false,1000}};
    auto read_first=run(read);
    // Simultaneous arrivals may legally reorder. Establish the first
    // direction's actual command times before injecting the other one.
    auto read_issued=read_first.pcs[0].last_read_complete-std::uint64_t(t.read_latency);
    std::vector<Request> rw{read[0],{0,line(0,1,7,1),true,read_issued+1}};
    auto a=run(rw);const auto& ra=a.pcs[0];
    std::vector<Request> write{{0,line(0,1,7,0),true,1000}};
    auto first=run(write);
    std::vector<Request> wr{write[0],{0,line(0,1,7,1),false,first.pcs[0].last_write_issue+1}};
    auto b=run(wr);const auto& rb=b.pcs[0];
    auto last_read_issue=ra.last_read_complete-std::uint64_t(t.read_latency);
    auto first_read_issue=rb.first_read_complete-std::uint64_t(t.read_latency);
    auto read_to_write=std::int64_t(ra.first_write_issue)-std::int64_t(last_read_issue);
    auto write_to_read=std::int64_t(first_read_issue)-std::int64_t(rb.last_write_issue);
    std::snprintf(buf,sizeof buf,"\"read_to_write_command_gap\": %lld, \"nRTW\": %d, \"write_to_read_command_gap\": %lld, \"nCWL+nBL+nWTRS\": %d",
        (long long)read_to_write,t.nRTW,(long long)write_to_read,t.nCWL+t.nBL+t.nWTRS);
    check("read_write_turnaround",ra.read_callbacks==2 && ra.write_callbacks==2 && rb.read_callbacks==2 && rb.write_callbacks==2 &&
          ra.first_write_issue>=last_read_issue+std::uint64_t(t.nRTW) &&
          first_read_issue>=rb.last_write_issue+std::uint64_t(t.nCWL+t.nBL+t.nWTRS),buf);
  }
  // (7) Skipping the ticks of a busy controller changes nothing: a mixed
  // trace (streams that conflict in one bank, random lines, writes, bursts of
  // simultaneous arrivals, idle gaps across refreshes) replayed both ways.
  {
    std::vector<Request> q;
    std::uint64_t seed = 12345, at = 1000;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return seed >> 33; };
    for (int i = 0; i < 200000; ++i) {
      std::uint64_t r = rnd();
      at += (r % 7 == 0) ? (r % 5000) : (r % 23);
      int kind = int(r % 4);
      std::uint64_t off;
      if (kind == 0) off = line(0, 0, int(i % 64), int((i / 64) % 32));           // stream A: row conflicts in one bank
      else if (kind == 1) off = line(2, 1, int((i / 32) % 512), int(i % 32));        // stream B: sequential rows
      else off = (rnd() % (std::uint64_t(256) << 20)) & ~std::uint64_t(63);          // random lines
      q.push_back({0, off, (r % 3) == 0, at});
    }
    auto a = tnca_dram::replay(cfg, 1, q, false), b = tnca_dram::replay(cfg, 1, q, true);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < q.size(); ++i) differ += a.completion[i] != b.completion[i];
    std::snprintf(buf, sizeof buf,
                  "\"requests\": %zu, \"differing_completions\": %zu, \"ticks_without_skip\": %llu, \"ticks_with_skip\": %llu, "
                  "\"row_conflicts\": %llu, \"refreshes\": %llu",
                  q.size(), differ, (unsigned long long)a.ticks, (unsigned long long)b.ticks,
                  (unsigned long long)b.pcs[0].row_conflicts, (unsigned long long)b.pcs[0].refreshes);
    check("event_skip_identical", a.error.empty() && b.error.empty() && differ == 0 &&
                                      a.pcs[0].row_conflicts == b.pcs[0].row_conflicts && b.ticks < a.ticks, buf);
  }
  std::printf("{\"timing\": {\"nBL\": %d, \"nCL\": %d, \"nCWL\": %d, \"nRCDRD\": %d, \"nRCDWR\": %d, \"nRP\": %d, \"nRAS\": %d, "
              "\"nREFI\": %d, \"nRFC\": %d, \"read_latency\": %d, \"tCK_ps\": %d},\n \"tests\": [\n%s\n ], \"failures\": %d}\n",
              t.nBL, t.nCL, t.nCWL, t.nRCDRD, t.nRCDWR, t.nRP, t.nRAS, t.nREFI, t.nRFC, t.read_latency, t.tCK_ps,
              json.c_str(), failures);
  return failures ? 1 : 0;
}
