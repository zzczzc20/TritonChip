// Ramulator2 bridge for the slow DRAM path (C++20; the only file that
// includes Ramulator2). See tnca_dram_bridge.hpp for the interface and
// hbm2_u280_pc.py for the DRAM configuration and its assumptions.
//
// Three Ramulator2 components are subclassed here only to make their state
// observable or skippable; their behaviour is Ramulator2's:
//   TncaHBM12      = HBM12 controller + busy()/skip_to()/stats access;
//   TncaAllBank    = AllBank refresh, firing at the first tick at or after
//                    each multiple of tREFI (Ramulator2 compares for equality,
//                    which a skipped clock would never meet again);
//   TncaPCSystem   = a memory system whose controller is reachable.
#include "tnca_dram_bridge.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/base/request.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/impl/hbm_controller_base.h"
#include "ramulator/controller/refresh/i_refresh_manager.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/dram/node.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace Ramulator {

class TncaAllBankRefresh final : public IRefreshManager, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, TncaAllBankRefresh, "TncaAllBank")

 public:
  ControllerBase* m_ctrl = nullptr;
  int m_cmd_refab = -1, m_ref_level = -1;
  Clk_t m_nrefi = 0, m_next = 0;
  int m_scatter_interval = 0;
  bool m_debug = false;
  std::uint64_t m_issued = 0;
  std::vector<DRAMNode*> m_nodes;

  void init() override {
    m_ctrl = cast_parent<ControllerBase>();
    RAMULATOR_PARSE_PARAM(m_scatter_interval, int, "scatter_interval").default_val(0);
    RAMULATOR_PARSE_PARAM(m_debug, bool, "debug").default_val(false);
    if (m_scatter_interval != 0) throw std::runtime_error("TncaAllBank: scattered refresh is not supported");
    const auto& spec = *m_ctrl->m_device.m_spec;
    m_cmd_refab = spec.get_command_id("REFab");
    m_ref_level = spec.get_level_id("PseudoChannel");  // HBM2 all-bank refresh scope
    m_nrefi = spec.get_timing_value("nREFI");
    m_ctrl->m_device.m_root->for_each_at_level(m_ref_level, [&](DRAMNode* n) { m_nodes.push_back(n); });
    m_next = m_nrefi;
  }

  void tick() override {
    if (m_ctrl->m_clk < m_next) return;
    m_next = (m_ctrl->m_clk / m_nrefi + 1) * m_nrefi;
    for (auto* node : m_nodes) {
      AddrVec_t v(m_ctrl->m_device.m_spec->level_count, -1);
      for (auto* n = node; n != nullptr; n = n->m_parent_node) v[n->m_level] = n->m_node_id;
      Request req(v, Request::Cmd, m_cmd_refab);
      if (!m_ctrl->priority_send(req)) throw std::runtime_error("TncaAllBank: refresh not accepted");
      ++m_issued;
    }
  }
};

class TncaHBM12 final : public HBMControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, TncaHBM12, HBMControllerBase, "TncaHBM12")

 public:
  bool busy() const {
    return m_read_buffer.size() || m_write_buffer.size() || m_priority_buffer.size() || m_active_buffer.size() ||
           !m_pending.empty();
  }
  void skip_to(Clk_t c) {
    if (c > m_clk) m_clk = c;
  }
  IRefreshManager* refresh() const { return m_refresh; }
  // A lower bound on the next clock at which this controller can do anything:
  // a buffered request's next command becoming legal (the maximum of the
  // ready clocks on its node path; rolling windows are folded into those
  // clocks at issue), or a read's data returning. Nothing changes state
  // before it, so the ticks in between are skipped (refresh and arrivals are
  // bounded by the caller).
  static Clk_t ready_at(const DRAMNode* n, int cmd, const AddrVec_t& v) {
    Clk_t r = n->m_cmd_ready_clk[cmd];
    if (n->m_child_nodes.empty()) return r;
    int child = v[n->m_level + 1];
    if (child == -1) {
      for (auto& c : n->m_child_nodes) r = std::max(r, ready_at(c.get(), cmd, v));
    } else {
      r = std::max(r, ready_at(n->m_child_nodes[child].get(), cmd, v));
    }
    return r;
  }
  Clk_t next_event_bound() {
    Clk_t bound = std::numeric_limits<Clk_t>::max();
    for (auto& q : m_pending) bound = std::min(bound, q.depart);
    auto scan = [&](ReqBuffer& b) {
      for (auto& q : b) {
        int cmd = m_device.get_preq_command(q.final_command, q.addr_vec, m_clk);
        bound = std::min(bound, std::max<Clk_t>(ready_at(m_device.m_root.get(), cmd, q.addr_vec), m_clk + 1));
      }
    };
    scan(m_priority_buffer);
    scan(m_active_buffer);
    scan(m_read_buffer);
    scan(m_write_buffer);
    return bound;
  }
  void fill(tnca_dram::PcStats& s) const {
    s.row_hits = s_row_hits; s.row_misses = s_row_misses; s.row_conflicts = s_row_conflicts;
    s.read_row_hits = s_read_row_hits; s.read_row_misses = s_read_row_misses; s.read_row_conflicts = s_read_row_conflicts;
    s.write_row_hits = s_write_row_hits; s.write_row_misses = s_write_row_misses;
    s.write_row_conflicts = s_write_row_conflicts;
  }
};

class TncaPCSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, TncaPCSystem, "TncaPCSystem")

 public:
  IChannelMapper* m_channel_mapper = nullptr;
  std::vector<IController*> m_controllers;
  unsigned int m_clock_ratio = 1;

  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_CREATE_CHILD(m_channel_mapper, IChannelMapper);
    RAMULATOR_CREATE_CHILD_LIST(m_controllers, IController);
    if (m_controllers.size() != 1) throw std::runtime_error("TncaPCSystem: one controller (one pseudo channel) expected");
    m_controllers[0]->set_channel_id(0);
    m_controllers[0]->m_clock_ratio = m_clock_ratio;
  }
  bool send(Request& req) override { return m_controllers[0]->send(req); }
  void tick() override { m_controllers[0]->tick(); }
  int get_clock_ratio() override { return static_cast<int>(m_clock_ratio); }
  float get_tCK() override { return m_controllers[0]->get_tCK(); }
  int get_tx_bytes() override { return m_controllers[0]->get_tx_bytes(); }
};

}  // namespace Ramulator

namespace tnca_dram {
namespace {

using namespace Ramulator;

std::string load_config_text(const std::string& path) {
  static std::mutex lock;
  static std::map<std::string, std::string> cache;
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(path);
  if (it != cache.end()) return it->second;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open Ramulator2 config " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string text = ss.str();
  auto rename = [&](const std::string& from, const std::string& to) {
    std::string a = "impl: " + from + "\n", b = "impl: " + to + "\n";
    size_t n = 0;
    for (size_t p = text.find(a); p != std::string::npos; p = text.find(a, p + b.size())) {
      text.replace(p, a.size(), b);
      ++n;
    }
    if (n != 1) throw std::runtime_error("Ramulator2 config: expected one '" + a.substr(0, a.size() - 1) + "'");
  };
  rename("HBM12", "TncaHBM12");
  rename("GenericDRAM", "TncaPCSystem");
  rename("AllBank", "TncaAllBank");
  cache[path] = text;
  return text;
}

struct Instance {
  IFrontEnd* frontend = nullptr;
  IMemorySystem* memory = nullptr;
  TncaHBM12* ctrl = nullptr;
  TncaAllBankRefresh* refresh = nullptr;
  explicit Instance(const std::string& path) {
    auto config = Config::parse_config_string(load_config_text(path));
    frontend = Factory::create_frontend(config);
    memory = Factory::create_memory_system(config);
    frontend->connect_memory_system(memory);
    memory->connect_frontend(frontend);
    auto* system = dynamic_cast<TncaPCSystem*>(memory);
    if (!system) throw std::runtime_error("Ramulator2 memory system is not TncaPCSystem");
    ctrl = dynamic_cast<TncaHBM12*>(system->m_controllers[0]);
    if (!ctrl) throw std::runtime_error("Ramulator2 controller is not TncaHBM12");
    refresh = dynamic_cast<TncaAllBankRefresh*>(ctrl->refresh());
    if (!refresh) throw std::runtime_error("Ramulator2 refresh manager is not TncaAllBank");
  }
  ~Instance() {
    delete dynamic_cast<Implementation*>(memory);
    delete dynamic_cast<Implementation*>(frontend);
  }
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
};

}  // namespace

// Synthetic ROW_COLUMN_BANK-style mapping inside one pseudo channel; this
// is not established as the installed platform's mapping. PG276 documents
// a different default (Row Bank Column with BG0 interleave). The 32-byte burst
// offset in bits [4:0], then bank group [6:5], bank [8:7], the column (32
// bursts of a 1 KiB row) [13:9] and the row [27:14]. Consecutive bursts walk
// the bank groups first, so a 64-byte line spans two bank groups.
Mapped map_address(std::uint64_t offset) {
  std::uint64_t burst = offset >> 5;
  Mapped m;
  m.bankgroup = static_cast<int>(burst & 3);
  m.bank = static_cast<int>((burst >> 2) & 3);
  m.column = static_cast<int>((burst >> 4) & 31);
  m.row = static_cast<int>((burst >> 9) & 16383);
  return m;
}

std::string sha256_file(const std::string& path) {
  static const std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::ifstream in(path, std::ios::binary);
  if (!in) return "";
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::uint64_t bits = std::uint64_t(data.size()) * 8;
  data.push_back(char(0x80));
  while (data.size() % 64 != 56) data.push_back(0);
  for (int i = 7; i >= 0; --i) data.push_back(char((bits >> (8 * i)) & 0xff));
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (std::size_t off = 0; off < data.size(); off += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(std::uint8_t(data[off + 4 * i])) << 24) | (std::uint32_t(std::uint8_t(data[off + 4 * i + 1])) << 16) |
             (std::uint32_t(std::uint8_t(data[off + 4 * i + 2])) << 8) | std::uint32_t(std::uint8_t(data[off + 4 * i + 3]));
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
      std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
  return std::string(out, 64);
}

Timing timing(const std::string& yaml_path) {
  Timing t;
  try {
    Instance inst(yaml_path);
    const auto& s = *inst.ctrl->m_device.m_spec;
    t.nBL = s.get_timing_value("nBL"); t.nCL = s.get_timing_value("nCL"); t.nCWL = s.get_timing_value("nCWL");
    t.nRCDRD = s.get_timing_value("nRCDRD"); t.nRCDWR = s.get_timing_value("nRCDWR"); t.nRP = s.get_timing_value("nRP");
    t.nRAS = s.get_timing_value("nRAS"); t.nREFI = s.get_timing_value("nREFI"); t.nRFC = s.get_timing_value("nRFC");
    t.read_latency = s.read_latency; t.tCK_ps = s.get_timing_value("tCK_ps");
  } catch (const std::exception& e) {
    t.error = e.what();
  }
  return t;
}

Result replay(const std::string& yaml_path, unsigned pcs, const Batch& batch, std::vector<std::uint32_t>& latency,
              bool event_skip) {
  Result r;
  r.pcs.resize(pcs);
  latency.assign(batch.n, 0);
  try {
    if (batch.n >= (std::size_t(1) << 32)) throw std::runtime_error("more than 2^32 requests in one replay");
    std::vector<std::vector<std::uint32_t>> by_pc(pcs);
    {
      std::vector<std::size_t> count(pcs, 0);
      std::uint32_t pc;
      std::uint64_t off;
      bool wr;
      for (std::size_t i = 0; i < batch.n; ++i) {
        batch.decode(batch.context, i, pc, off, wr);
        if (pc >= pcs) throw std::runtime_error("request pseudo channel out of range");
        ++count[pc];
      }
      for (unsigned p = 0; p < pcs; ++p) by_pc[p].reserve(count[p]);
      for (std::size_t i = 0; i < batch.n; ++i) {
        batch.decode(batch.context, i, pc, off, wr);
        by_pc[pc].push_back(static_cast<std::uint32_t>(i));
      }
    }
    for (unsigned pc = 0; pc < pcs; ++pc) {
      auto& order = by_pc[pc];
      if (order.empty()) continue;
      std::stable_sort(order.begin(), order.end(),
                       [&](std::uint32_t a, std::uint32_t b) { return batch.arrival[a] < batch.arrival[b]; });
      Instance inst(yaml_path);
      TncaHBM12* ctrl = inst.ctrl;
      const auto& spec = *ctrl->m_device.m_spec;
      const Clk_t nrefi = spec.get_timing_value("nREFI");
      const Clk_t write_done = spec.get_timing_value("nCWL") + spec.get_timing_value("nBL");
      auto& st = r.pcs[pc];
      const std::size_t m = order.size();
      // Per request: bits 0-1 bursts sent, bits 2-3 bursts outstanding/done count.
      std::vector<std::uint8_t> state(m, 0);
      std::size_t next = 0, done = 0;
      std::uint64_t outstanding = 0, stall = 0;
      auto arrival_of = [&](std::size_t k) { return batch.arrival[order[k]]; };
      auto make = [&](std::size_t k, unsigned half) {
        std::uint32_t p;
        std::uint64_t off;
        bool wr;
        batch.decode(batch.context, order[k], p, off, wr);
        off = (off & ~std::uint64_t(63)) + 32 * half;
        Mapped a = map_address(off);
        Ramulator::Request req;
        req.addr = static_cast<Addr_t>(off);
        req.type_id = wr ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
        req.source_id = -1;
        req.size_bytes = 32;
        req.addr_vec = {-1, 0, 0, a.bankgroup, a.bank, a.row, a.column << 2};
        req.callback = [&, k](Ramulator::Request& c) {
          Clk_t at = c.type_id == Ramulator::Request::Type::Read ? c.depart : ctrl->m_clk + write_done;
          std::uint64_t lat = static_cast<std::uint64_t>(at) - arrival_of(k);
          auto& l = latency[order[k]];
          l = static_cast<std::uint32_t>(std::max<std::uint64_t>(l, std::min<std::uint64_t>(lat, 0xffffffffu)));
          --outstanding;
          state[k] += 4;
          if ((state[k] >> 2) == 2) ++done;
        };
        return req;
      };
      while (done < m) {
        // Inject every burst that has arrived; a refused burst retries next tick.
        while (next < m && arrival_of(next) <= static_cast<std::uint64_t>(ctrl->m_clk)) {
          bool all = true;
          for (unsigned half = 0; half < 2; ++half) {
            if (state[next] & (1u << half)) continue;
            auto req = make(next, half);
            ++outstanding;
            if (!ctrl->send(req)) {
              --outstanding;
              all = false;
              break;
            }
            state[next] |= static_cast<std::uint8_t>(1u << half);
          }
          if (!all) break;
          {
            std::uint32_t p;
            std::uint64_t off;
            bool wr;
            batch.decode(batch.context, order[next], p, off, wr);
            (wr ? st.writes : st.reads)++;
          }
          ++next;
        }
        // Idle until the next arrival: skip, stopping for a refresh due in the gap.
        if (outstanding == 0 && !ctrl->busy() && next < m) {
          Clk_t clk = ctrl->m_clk, target = static_cast<Clk_t>(arrival_of(next)) - 1;
          if (target > clk) {
            Clk_t last_ref = (target / nrefi) * nrefi;
            Clk_t to = (last_ref > clk && inst.refresh->m_next <= last_ref) ? last_ref - 1 : target;
            if (to > clk) {
              st.skipped += static_cast<std::uint64_t>(to - clk);
              ctrl->skip_to(to);
            }
          }
        }
        // Busy: skip ticks in which no command can become legal.
        if (event_skip && (outstanding > 0 || ctrl->busy())) {
          Clk_t clk = ctrl->m_clk, bound = ctrl->next_event_bound();
          bound = std::min(bound, inst.refresh->m_next);
          if (next < m) bound = std::min(bound, static_cast<Clk_t>(arrival_of(next)));
          if (bound != std::numeric_limits<Clk_t>::max() && bound - 1 > clk) {
            st.skipped += static_cast<std::uint64_t>(bound - 1 - clk);
            ctrl->skip_to(bound - 1);
          }
        }
        std::uint64_t before = done;
        ctrl->tick();
        ++st.ticks;
        stall = (done == before && outstanding > 0) ? stall + 1 : 0;
        if (stall > 50000000) throw std::runtime_error("Ramulator2 replay made no progress for 5e7 DRAM cycles");
      }
      ctrl->fill(st);
      st.refreshes = inst.refresh->m_issued;
      r.ticks += st.ticks;
      r.skipped += st.skipped;
      std::vector<std::uint32_t>().swap(order);
    }
  } catch (const std::exception& e) {
    r.error = e.what();
  }
  return r;
}

namespace {
struct VectorContext {
  const std::vector<Request>* requests;
};
void decode_vector(const void* c, std::size_t i, std::uint32_t& pc, std::uint64_t& offset, bool& write) {
  const auto& q = (*static_cast<const VectorContext*>(c)->requests)[i];
  pc = q.pc;
  offset = q.offset;
  write = q.write;
}
}  // namespace

Result replay(const std::string& yaml_path, unsigned pcs, const std::vector<Request>& requests, bool event_skip) {
  std::vector<std::uint64_t> arrival(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) arrival[i] = requests[i].arrival;
  VectorContext ctx{&requests};
  Batch b;
  b.n = requests.size();
  b.arrival = arrival.data();
  b.context = &ctx;
  b.decode = decode_vector;
  std::vector<std::uint32_t> latency;
  Result r = replay(yaml_path, pcs, b, latency, event_skip);
  r.completion.resize(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) r.completion[i] = requests[i].arrival + latency[i];
  return r;
}

}  // namespace tnca_dram
