#pragma once
// C++17-clean interface to the Ramulator2 bridge (tnca_dram_bridge.cpp is
// C++20 and the only translation unit that includes Ramulator2). The SystemC
// model (C++17) calls replay() between child passes of the slow DRAM path.
#include <cstdint>
#include <string>
#include <vector>

namespace tnca_dram {

// One memory transaction of the model: a 64-byte line read or write that the
// HBM side receives at `arrival` (DRAM clock cycles). `pc` is a dense index
// of the pseudo channels the replay touches; `offset` the byte offset inside
// that 256 MiB pseudo channel. A line is two 32-byte HBM2 bursts (BL4 x 64).
struct Request {
    std::uint32_t pc = 0;
    std::uint64_t offset = 0;
    bool write = false;
    std::uint64_t arrival = 0;
};

struct PcStats {
    std::uint64_t reads = 0, writes = 0;
    std::uint64_t row_hits = 0, row_misses = 0, row_conflicts = 0;
    std::uint64_t read_row_hits = 0, read_row_misses = 0, read_row_conflicts = 0;
    std::uint64_t write_row_hits = 0, write_row_misses = 0, write_row_conflicts = 0;
    std::uint64_t refreshes = 0, ticks = 0, skipped = 0;
    // Burst units are 32 bytes; reads/writes above count 64-byte lines.
    std::uint64_t accepted_bursts = 0, completed_bursts = 0, queue_refusals = 0, peak_rw_queued_bursts = 0;
    std::uint64_t read_callbacks = 0, write_callbacks = 0;
    std::uint64_t first_read_complete = 0, last_read_complete = 0;
    std::uint64_t first_write_issue = 0, last_write_issue = 0, last_write_complete = 0;
    std::uint64_t peak_line_credits = 0;
};

struct Result {
    std::vector<std::uint64_t> completion;   // per request: DRAM cycle the line is complete
    std::vector<PcStats> pcs;
    std::uint64_t ticks = 0, skipped = 0;
    std::string error;                       // empty on success
};

// The resolved DRAM timing the replay uses (DRAM clock cycles).
struct Timing {
    int nBL = 0, nCL = 0, nCWL = 0, nRCDRD = 0, nRCDWR = 0, nRP = 0, nRAS = 0;
    int nREFI = 0, nRFC = 0, read_latency = 0, tCK_ps = 0;
    int nRTW = 0, nWTRS = 0;
    std::string error;
};

// Replays `requests` (any order; ties keep their order) through one fresh
// Ramulator2 memory system per pseudo channel (DRAM state starts idle,
// refresh phase at cycle 0). Idle gaps are skipped, stopping at every
// refresh due inside a gap so that its precharge and tRFC are modelled.
// event_skip: also skip the ticks of a busy controller in which no command
// can become legal. idle_skip=false ticks every cycle of the idle gaps too:
// the reference both skips are checked against (identical completions,
// tnca-dram-bridge-test); it is far slower and only for tests.
Result replay(const std::string& yaml_path, unsigned pcs, const std::vector<Request>& requests,
              bool event_skip = true, bool idle_skip = true);

// The memory-lean form the model uses (tens of millions of requests): the
// caller keeps the requests in its own layout, gives their arrivals (DRAM
// cycles) as an array and decodes the rest on demand; the result is each
// request's latency (completion - arrival, DRAM cycles). `completion` of the
// returned Result stays empty.
struct Batch {
    std::size_t n = 0;
    const std::uint64_t* arrival = nullptr;
    const void* context = nullptr;
    void (*decode)(const void* context, std::size_t i, std::uint32_t& pc, std::uint64_t& offset, bool& write) = nullptr;
};
// Optional hypothetical stream-outlet admission, independently per PC.
// A credit covers one 64-byte line until both bursts complete plus the fixed
// return pipeline. Admission starts only on kernel clock edges and successive
// lines are separated by issue_cycles kernel cycles. This does not model the
// core, NUCA, AXI width conversion, copy data dependencies or arithmetic.
struct StreamAdmission {
    unsigned lines = 0;
    unsigned dram_cycles_per_kernel = 0;
    unsigned issue_cycles = 0;
    unsigned return_kernel_cycles = 0;
};
Result replay(const std::string& yaml_path, unsigned pcs, const Batch& batch, std::vector<std::uint32_t>& latency,
              bool event_skip = true, bool idle_skip = true, const StreamAdmission* admission = nullptr);
Timing timing(const std::string& yaml_path);
// The controller address mapping inside one pseudo channel (see the .cpp).
struct Mapped {int bankgroup, bank, row, column;};
Mapped map_address(std::uint64_t offset);
// sha256 of a file (hex), for provenance; empty if unreadable.
std::string sha256_file(const std::string& path);

}  // namespace tnca_dram
