#pragma once
// TNCA memory architecture reference. TLM-2.0 blocking transport, concurrent
// SC_THREAD initiators; all resource waits advance kernel time, never host time.
// No DMI or temporal decoupling: neither may bypass shared-resource contention.
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/multi_passthrough_target_socket.h>
#include <array>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace sc_core;
using u64 = std::uint64_t;
using Line = std::array<unsigned char,64>;

static void require(bool b, const std::string& why) {
    if (!b) throw std::runtime_error(why);
}
// The hot paths pass literals: no std::string is built unless the check fails.
static inline void require(bool b, const char* why) {
    if (__builtin_expect(!b, 0)) throw std::runtime_error(why);
}
// An architectural fault (ISA sec 10): the device would stop the raising
// context and report FAULT for the launch. Distinct from a model refusal
// (std::runtime_error), which means the model cannot answer at all.
struct ArchFault : std::runtime_error {
    unsigned code;
    ArchFault(unsigned c,const std::string& why):std::runtime_error(why),code(c){}
};
inline void arch(bool ok,unsigned code,const std::string& why){if(!ok)throw ArchFault(code,why);}
inline void arch(bool ok,unsigned code,const char* why){if(__builtin_expect(!ok,0))throw ArchFault(code,why);}
static u64 integer(const std::string& s) {
    require(!s.empty() && s[0]!='-', "negative or empty integer");
    size_t n=0; auto v=std::stoull(s,&n,0);
    require(n==s.size(),"invalid integer: "+s); return v;
}
struct Config {
    std::map<std::string,u64> v;
    // physical_map: an explicit AXI BO/PC map (TNCA_DRAM_MAP, tnca_dram_map.hpp)
    // translates every DRAM line, so the window is checked against that map's
    // BO (Model::select_dram_path) instead of the contiguous logical PC map.
    explicit Config(const char* path,bool physical_map=false) {
        std::ifstream f(path); require(bool(f),"cannot open configuration");
        std::string s;
        while (std::getline(f,s)) {
            s=s.substr(0,s.find('#')); if(s.find_first_not_of(" \t\r")==std::string::npos)continue;
            auto p=s.find('='); require(p!=s.npos,"expected key=value");
            auto k=s.substr(0,p); require(!v.count(k),"duplicate configuration: "+k);
            v[k]=integer(s.substr(p+1));
        }
        const std::vector<std::string> keys={"nx","ny","sets","ways","full_line_tag","mshrs","targets",
            "outlets","active_outlets","outstanding","core_window","clock_hz","memory_base",
            "memory_bytes","pc_bytes","pc_count","pc_first","lookup_cycles","service_cycles",
            "noc_stages","noc_bytes_per_cycle","hbm_latency_cycles","hbm_bytes_per_cycle",
            "outlet_issue_cycles","trace_events","dma_queue","dma_unit_slots","dma_outstanding","dma_setup_cycles","dma_rf_cycles",
            // Core and engine timing: every value is an UNCALIBRATED hypothesis.
            "issue_cycles","scalar_latency","scalar_mul_latency","vector_int_latency","vector_fp_latency",
            "convert_latency","compare_latency","predicate_latency","shuffle_latency","reduction_latency",
            "fp_reduction_latency","sfu_occupancy","sfu_latency","load_latency","lsu_slots","lsu_beat_slots",
            "barrier_latency","acc_read_latency","mtx_abuf","mtx_rf_read_cycles","mtx_mac_cycles",
            "mtx_mac_unbuffered_cycles","mtx_pipe_cycles","mtx_complete_cycles","mtx_zero_cycles",
            "mtx_row_cycles","launch_cycles","dispatch_cycles",
            // M2: host time between launches (server mode only) and the timing paths.
            "host_launch_cycles","host_transfer_cycles","host_transfer_cycles_per_kib","memory_model",
            // Timeline memory structure (include/tnca_timeline_memory.hpp)
            "tl_noc_inject_cycles","tl_noc_eject_cycles","tl_slice_hit_occupancy","tl_slice_hit_latency",
            "tl_slice_miss_occupancy","tl_slice_mreq_occupancy","tl_slice_fill_occupancy","tl_slice_wake_occupancy",
            "tl_slice_writeback_occupancy","tl_slice_to_outlet_cycles","tl_outlet_to_slice_cycles",
            "tl_outlet_read_issue_cycles","tl_outlet_write_issue_cycles","tl_outlet_beat_cycles","tl_hbm_write_latency_cycles",
            // Core pipeline structure (core_pipeline=1), from tnca_core.v / tnca_lsu_pair.v / tnca_tensor.v
            "core_pipeline","branch_taken_cycles","park_refetch_cycles","vrf_banks","vrf_bank_conflict_cycles",
            "lsu_setup_cycles","lsu_vector_load_extra","lsu_complete_cycles","lsu_writeback_cycles",
            "fp_compare_latency","shuffle_scalar_latency","reduction_vector_extra","dma_store_register_cycles","dma_drain_cycles",
            "vrf_prefetch","fp_reduction_vector_extra","time_quantum",
            // M3: DRAM path (include/tnca_dram.hpp)
            "dram_path","dram_fixed_cycles","dram_clock_ratio","dram_max_passes","dram_tolerance_ppm"};
        // Optional what-if keys, absent from every shipped configuration; an
        // absent key takes its default and leaves the model unchanged.
        //   mtx_dp_group_16 / mtx_dp_group_8 (default 1): a hypothetical packed
        //     MMA (m.mma.dp, not built) sweeping G consecutive k per lane per
        //     step for 16-bit (G<=2) / FP8 (G<=4) fragments; timing only.
        //   mtx_dp_b_prefetch (default 0): the per-step B register read is
        //     hidden behind the previous step's sweep after the first.
        std::size_t optional=0;
        for(auto& k:optional_keys())if(v.count(k))++optional;
        require(v.size()-optional==keys.size(),"configuration key count mismatch (unknown or missing key)");
        for(auto& k:keys)require(v.count(k),"missing configuration: "+k);
        {u64 g16=get_or("mtx_dp_group_16",1),g8=get_or("mtx_dp_group_8",1);
         require((g16==1 || g16==2) && (g8==1 || g8==2 || g8==4) && get_or("mtx_dp_b_prefetch",0)<=1,
                 "mtx_dp_group_16 is 1|2, mtx_dp_group_8 1|2|4 (one 64-byte B register per group), mtx_dp_b_prefetch 0|1");}
        {auto b=get_or("tl_core_beat_bytes",8),n=get_or("tl_noc_payload_bytes",8);
         require((b==8 || b==16 || b==32 || b==64) && (n==8 || n==16 || n==32 || n==64),
                 "exploratory core/NoC payload bytes must be 8|16|32|64");
         require(b==8 || get("memory_model")==1,"wide core hypothesis requires timeline memory");}
        {auto n=get_or("tl_lsu_units",2);
         require(n==2 || n==4 || n==8 || n==16 || n==32 || n==64,"exploratory LSU units must be 2|4|8|16|32|64");
         require(n==2 || (get("memory_model")==1 && get("core_pipeline")==1),"LSU unit hypothesis needs timeline core pipeline");}
        {auto n=get_or("tl_memory_lanes",1);
         require(n==1 || n==2 || n==4 || n==8,"exploratory NoC/home lanes must be 1|2|4|8");
         require(n==1 || get("memory_model")==1,"parallel memory lanes require timeline memory");}
        require(get_or("tl_lsu_shared_issue",1)<=1,"tl_lsu_shared_issue must be 0|1");
        {auto n=get_or("tl_pc_stripe_bytes",0);require(n==0 || (n>=64 && n%64==0),"PC stripe must be zero or whole lines");}
        cache_hot();
        for(auto& k:keys)if(k!="memory_base" && k!="pc_first" && k!="trace_events" && k!="noc_stages" && k!="full_line_tag"
                            && k!="mtx_abuf" && k!="launch_cycles" && k!="dispatch_cycles" && k!="load_latency"
                            && k!="host_launch_cycles" && k!="host_transfer_cycles" && k!="host_transfer_cycles_per_kib" && k!="memory_model"
                            && k.rfind("tl_",0)!=0 && k!="core_pipeline" && k!="branch_taken_cycles" && k!="park_refetch_cycles"
                            && k!="vrf_bank_conflict_cycles" && k!="lsu_setup_cycles" && k!="lsu_vector_load_extra"
                            && k!="lsu_complete_cycles" && k!="lsu_writeback_cycles" && k!="reduction_vector_extra"
                            && k!="dma_store_register_cycles" && k!="dma_drain_cycles" && k!="vrf_prefetch" && k!="fp_reduction_vector_extra" && k!="time_quantum"
                            && k!="dram_path" && k!="dram_fixed_cycles")
            require(v.at(k)>0,"zero configuration: "+k);
        require(get("mtx_abuf")<=1 && get("lsu_slots")<=64 && get("lsu_beat_slots")<=64,"invalid core engine configuration");
        require(get("memory_model")<=1,"memory_model is 0 (detailed TLM) or 1 (timeline)");
        require(get("time_quantum")==0 || get("memory_model")==1,"time_quantum needs the timeline memory (memory_model=1)");
        require(get("nx")*get("ny")<=16,"at most 16 homes");
        require(get("sets")<=65536 && (get("sets")&(get("sets")-1))==0,"sets must be power of two <=65536");
        require(get("ways")<=32 && get("mshrs")<=256 && get("targets")<=4096,"unreasonable cache geometry");
        require(get("outlets")<=get("pc_count"),"outlets exceed pseudo channels");
        require(get("full_line_tag")<=1 && (get("full_line_tag") || (get("nx")==2 && get("ny")==2)),"compressed tags require frozen 2x2 topology");
        require(get("active_outlets")==1 || get("active_outlets")==get("outlets"),"RTL supports one/all outlets");
        require(get("outstanding")<=256 && get("core_window")<=256,"unreasonable outstanding window");
        require(get("dma_queue")<=16 && get("dma_unit_slots")<=64 && get("dma_outstanding")<=128,"unreasonable DMA resources");
        require(get("clock_hz")<=1000000000000ULL,"clock exceeds time resolution");
        require(get("memory_bytes")<=UINT64_MAX-get("memory_base"),"memory window overflow");
        require(get("memory_base")%64==0 && get("memory_bytes")%64==0,"line-aligned memory window required");
        require(get("pc_bytes")%64==0 && get("pc_first")<get("pc_count"),"invalid PC map");
        require(physical_map || (get("memory_bytes")-1)/get("pc_bytes")<get("pc_count")-get("pc_first"),"window exceeds PC map");
    }
    u64 get(const std::string& k)const{return v.at(k);}
    static const std::vector<std::string>& optional_keys(){
        static const std::vector<std::string> k={"mtx_dp_group_16","mtx_dp_group_8","mtx_dp_b_prefetch","tl_core_beat_bytes","tl_noc_payload_bytes","tl_lsu_units","tl_memory_lanes","tl_lsu_shared_issue","tl_pc_stripe_bytes"};return k;}
    u64 get_or(const std::string& k,u64 fallback)const{auto i=v.find(k);return i==v.end()?fallback:i->second;}
    // Hot-path copies of fixed geometry (set once by cache_hot()).
    u64 nx_=0,ny_=0,sets_=1,full_line_=1,base_=0,bytes_=0;unsigned cores_=1;
    void cache_hot(){nx_=get("nx");ny_=get("ny");sets_=get("sets");full_line_=get("full_line_tag");
        base_=get("memory_base");bytes_=get("memory_bytes");cores_=unsigned(nx_*ny_);}
    unsigned cores()const{return cores_;}
    unsigned home(u64 a)const {
        u64 line=a>>6;
        if(nx_==2 && ny_==2)return unsigned((line^(line>>2))&3);
        unsigned h=0;for(unsigned i=0;i<8;++i)h^=unsigned((line>>(i*8))&255);
        return h%cores_;
    }
    // FULL_LINE_TAG=1 selects low line bits for generic topologies. The frozen
    // 2x2 compressed-tag profile drops the reconstructible HOME_W low bits.
    unsigned set(u64 a)const {
        return unsigned((a>>(full_line_ ? 6:8))&(sets_-1));
    }
    void address(u64 a,unsigned n)const {
        require(a>=base_ && n<=bytes_ && a-base_<=bytes_-n,"address outside model memory window");
        // The 4 KiB TMU register page every shipped NUCA image decodes before
        // external memory (python/tnca/runtime.py DEVICE_RESERVED_RANGES).
        require(!(a<0x7fff1000ULL && a+n>0x7fff0000ULL),"TMU MMIO is not ordinary model memory");
    }
};
struct Context {
    const Config& c;
    // Untimed execution (serve command "T 0"): every instruction, transfer
    // and launch has its architectural effect and takes no modeled time.
    // Used for a graph's cold call, whose timing is not a measurement;
    // timing state is not touched, so the timed call starts unperturbed.
    bool functional=false;
    std::map<std::string,u64> counts;
    std::ofstream trace;
    std::string directory;
    u64 emitted=0,dropped=0;
    std::string monitor_mode="aggregate";
    struct SocketStats {
        u64 calls=0,bytes=0,reads=0,writes=0,errors=0,active=0,peak=0;
        u64 latency_sum=0,latency_max=0,busy_cycles=0,last_cycle=0;
        std::array<u64,65> histogram{};
        void integrate(u64 now){if(active)busy_cycles+=now-last_cycle;last_cycle=now;}
    };
    std::map<std::string,SocketStats> sockets;
    u64 unit=0; // time-resolution units per reference cycle
    explicit Context(const Config& cfg,const std::string& dir):c(cfg),trace(dir+"/trace.json"),directory(dir){
        double units=1.0/double(c.get("clock_hz"))/sc_get_time_resolution().to_seconds();
        // Every model time is an integral number of reference cycles. A clock
        // whose period is not a whole number of resolution units (90 MHz at
        // 1 ps) would otherwise round each wait differently, and two events of
        // the same cycle could compare unequal; the period is rounded once
        // instead (11111 ps at 90 MHz: 1e-5 relative in reported seconds only).
        require(units>=1,"clock period below the time resolution");
        unit=u64(std::llround(units));
        if(const char* mode=std::getenv("TNCA_TLM_MONITOR"))monitor_mode=mode;
        require(monitor_mode=="off" || monitor_mode=="aggregate" || monitor_mode=="trace","unknown monitor mode");
        require(bool(trace),"cannot create trace");trace<<"{\"displayTimeUnit\":\"ns\",\"traceEvents\":[";
    }
    ~Context(){
        if(trace.is_open()){
            std::ofstream partial(directory+"/partial.json");
            partial<<"{\"schema\":\"tnca-model-partial/1\",\"status\":\"incomplete\",\"performance_gate\":\"not-measured\",\"cycles\":"<<now()<<",\"counters\":{";
            bool first=true;for(auto& kv:counts){if(!first)partial<<',';first=false;partial<<'"'<<kv.first<<"\":"<<kv.second;}
            partial<<"}}\n";close(false);
        }
    }
    sc_time cycles(u64 n)const{return unit?sc_time::from_value(n*unit):sc_time(double(n)/double(c.get("clock_hz")),SC_SEC);}
    u64 to_cycles(const sc_time& t)const{return unit?(t.value()+unit/2)/unit:u64(t/cycles(1)+0.5);}
    u64 now()const{return to_cycles(sc_time_stamp());}
    void span(const std::string& name,unsigned lane,sc_time start,u64 id) {
        if(monitor_mode!="trace")return;
        if(emitted>=c.get("trace_events")){++dropped;return;}
        if(emitted++)trace<<',';
        trace<<"{\"name\":\""<<name<<"\",\"cat\":\"model-prediction\",\"ph\":\"X\",\"pid\":1,\"tid\":"<<lane
            <<",\"ts\":"<<std::setprecision(15)<<start.to_seconds()*1e6
            <<",\"dur\":"<<(sc_time_stamp()-start).to_seconds()*1e6<<",\"args\":{\"id\":"<<id<<"}}";
    }
    void close(bool complete=true){
        std::ofstream stats(directory+"/sockets.json");
        stats<<"{\"schema\":\"tnca-tlm-sockets/1\",\"mode\":\""<<monitor_mode<<"\",\"complete\":"<<(complete?"true":"false")<<",\"cycles\":"<<now()<<",\"sockets\":{";
        bool first=true;for(auto& kv:sockets){auto& v=kv.second;v.integrate(now());if(!first)stats<<',';first=false;
            stats<<'"'<<kv.first<<"\":{\"calls\":"<<v.calls<<",\"bytes\":"<<v.bytes<<",\"reads\":"<<v.reads<<",\"writes\":"<<v.writes<<",\"errors\":"<<v.errors<<",\"active\":"<<v.active<<",\"peak\":"<<v.peak<<",\"latency_sum_cycles\":"<<v.latency_sum<<",\"latency_max_cycles\":"<<v.latency_max<<",\"busy_cycles\":"<<v.busy_cycles<<",\"latency_log2_histogram\":[";
            for(unsigned i=0;i<v.histogram.size();++i){if(i)stats<<',';stats<<v.histogram[i];}stats<<"]}";
        }stats<<"}}\n";
        trace<<"],\"metadata\":{\"backend\":\"systemc-tlm-prediction\",\"complete\":"<<(complete?"true":"false")<<",\"dropped_events\":"<<dropped<<"}}\n";trace.close();}
};
struct Request : tlm::tlm_extension<Request> {
    u64 id=0, value=0, compare=0;
    unsigned core=0,home=0,op=0,be=255;
    tlm_extension_base* clone()const override{return new Request(*this);}
    void copy_from(const tlm_extension_base& x)override{*this=static_cast<const Request&>(x);}
};
struct Resource {
    sc_time available=SC_ZERO_TIME;
    u64* service=nullptr; u64* queue=nullptr;
    // Counter slots bind on first use: the key a Resource is used with never
    // changes, and the set of reported counters stays exactly as before.
    void use(Context& x,u64 duration,const std::string& key) {
        if(!service)bind(x,key);
        *service+=duration;
        sc_time now=sc_time_stamp(),begin=std::max(available,now);
        *queue+=x.to_cycles(begin-now);
        available=begin+x.cycles(duration);wait(available-now);
    }
    // Same accounting with counter slots bound once (std::map nodes are stable).
    void bind(Context& x,const std::string& key){service=&x.counts[key+"_service_cycles"];queue=&x.counts[key+"_queue_cycles"];}
    void use(const Context& x,const sc_time& duration,u64 cycles){
        *service+=cycles;sc_time now=sc_time_stamp(),begin=std::max(available,now);
        *queue+=x.to_cycles(begin-now);available=begin+duration;wait(available-now);
    }
};
// Byte-addressed backing store in lazily allocated 1 MiB pages, so a model
// window of many GiB costs memory only where the workload touches it.
struct Store {
    static constexpr u64 page_lines=16384;
    u64 base=0;std::vector<std::unique_ptr<Line[]>> pages;
    void configure(u64 window_base,u64 bytes){base=window_base;pages.resize((bytes/64+page_lines-1)/page_lines);}
    Line& operator[](u64 a){
        u64 line=(a-base)>>6;require(a>=base && line/page_lines<pages.size(),"store address outside window");
        auto& page=pages[line/page_lines];if(!page)page.reset(new Line[page_lines]());
        return page[line%page_lines];
    }
    const Line* find(u64 a)const{
        u64 line=(a-base)>>6;if(a<base || line/page_lines>=pages.size() || !pages[line/page_lines])return nullptr;
        return &pages[line/page_lines][line%page_lines];
    }
};
// Persistent worker processes consuming a job queue: concurrency without
// creating a SystemC process per memory beat or transfer unit.
struct Pool {
    std::deque<std::function<void()>> jobs;sc_event work;
    explicit Pool(unsigned n){
        for(unsigned i=0;i<n;++i)sc_spawn([this](){for(;;){while(jobs.empty())wait(work);auto job=std::move(jobs.front());jobs.pop_front();job();}});
    }
    void submit(std::function<void()> job){jobs.push_back(std::move(job));work.notify(SC_ZERO_TIME);}
};
struct Limit {
    unsigned used=0,peak=0,capacity; sc_event changed;
    explicit Limit(unsigned n):capacity(n){}
    u64* waited=nullptr; // bound on first use; one key per Limit
    void acquire(Context& x,const std::string& key) {
        if(!waited)waited=&x.counts[key+"_wait_cycles"];
        auto t=sc_time_stamp();while(used==capacity)wait(changed);
        *waited+=x.to_cycles(sc_time_stamp()-t);
        peak=std::max(peak,++used);
    }
    void release(){require(used>0,"resource underflow");--used;changed.notify(SC_ZERO_TIME);}
};
// Explicit 64-byte line storage and acknowledged writes; independent PC service
// resources, shared outstanding slots and issue bandwidth per actual outlet.
struct HBM : sc_module {
    tlm_utils::multi_passthrough_target_socket<HBM> target;
    Context& x; Store memory;
    std::vector<std::unique_ptr<Limit>> slots;
    std::vector<Resource> ports,pcs;
    std::vector<std::string> port_keys,pc_keys;
    struct PortCounters {u64 *writes=nullptr,*reads=nullptr,*peak=nullptr;};
    std::vector<PortCounters> port_counts;std::vector<u64*> pc_bytes;
    u64 active_outlets,outlets,pc_first,memory_base,pc_size,issue_cycles,transfer,latency;
    HBM(sc_module_name n,Context& ctx):sc_module(n),target("target"),x(ctx),
        ports(x.c.get("outlets")),pcs(x.c.get("pc_count")) {
        target.register_b_transport(this,&HBM::transport);memory.configure(x.c.get("memory_base"),x.c.get("memory_bytes"));
        for(u64 i=0;i<x.c.get("outlets");++i)slots.emplace_back(new Limit(unsigned(x.c.get("outstanding"))));
        for(u64 i=0;i<x.c.get("outlets");++i)port_keys.push_back("outlet_"+std::to_string(i));
        for(u64 i=0;i<x.c.get("pc_count");++i)pc_keys.push_back("pc_"+std::to_string(i));
        port_counts.resize(port_keys.size());pc_bytes.assign(pc_keys.size(),nullptr);
        active_outlets=x.c.get("active_outlets");outlets=x.c.get("outlets");pc_first=x.c.get("pc_first");
        memory_base=x.c.get("memory_base");pc_size=x.c.get("pc_bytes");issue_cycles=x.c.get("outlet_issue_cycles");
        transfer=(64+x.c.get("hbm_bytes_per_cycle")-1)/x.c.get("hbm_bytes_per_cycle");latency=x.c.get("hbm_latency_cycles");
    }
    void transport(int,tlm::tlm_generic_payload& tx,sc_time& delay){
        if(delay!=SC_ZERO_TIME){wait(delay);delay=SC_ZERO_TIME;}
        auto* r=tx.get_extension<Request>();require(r,"missing request identity");
        u64 a=tx.get_address();x.c.address(a,64);
        require(a%64==0 && tx.get_data_length()==64 && tx.get_streaming_width()==64 && !tx.get_byte_enable_ptr(),"invalid HBM line transaction");
        unsigned port=active_outlets==1 ? 0:unsigned(r->home%outlets);
        unsigned pc=unsigned(pc_first+(a-memory_base)/pc_size);
        const auto& key=port_keys[port];auto begin=sc_time_stamp();auto& pcnt=port_counts[port];
        slots[port]->acquire(x,key);ports[port].use(x,issue_cycles,key);
        pcs[pc].use(x,transfer,pc_keys[pc]);
        wait(x.cycles(latency));
        auto& data=memory[a];
        if(tx.is_write()){std::copy_n(tx.get_data_ptr(),64,data.begin());if(!pcnt.writes)pcnt.writes=&x.counts[key+"_writes"];++*pcnt.writes;}
        else {require(tx.is_read(),"invalid HBM command");std::copy(data.begin(),data.end(),tx.get_data_ptr());if(!pcnt.reads)pcnt.reads=&x.counts[key+"_reads"];++*pcnt.reads;}
        if(!pc_bytes[pc])pc_bytes[pc]=&x.counts[pc_keys[pc]+"_bytes"];
        *pc_bytes[pc]+=64;
        if(!pcnt.peak)pcnt.peak=&x.counts[key+"_peak_outstanding"];
        *pcnt.peak=slots[port]->peak;
        slots[port]->release();tx.set_response_status(tlm::TLM_OK_RESPONSE);
        x.span(tx.is_write()?"HBM write acknowledged":"HBM read",200+port,begin,r->id);
    }
    void initialize(u64 a,u64 word,unsigned be){x.c.address(a,8);require(a%8==0,"unaligned initialization");auto& l=memory[a&~63ULL];for(unsigned b=0;b<8;++b)if(be&(1<<b))l[(a&63)+b]=unsigned(word>>(8*b))&255;}
};
struct Done {bool done=false;sc_event event;};
struct CacheLine {u64 address=0;Line data{};bool valid=false,dirty=false;unsigned pins=0;};
struct Fill {CacheLine* line=nullptr;bool ready=false;unsigned targets=0;sc_event event;};
struct Slice : sc_module {
    tlm_utils::simple_target_socket<Slice> target;
    tlm_utils::simple_initiator_socket<Slice> memory;
    Context& x;unsigned id;std::vector<CacheLine> lines;std::vector<unsigned> rr;
    std::map<u64,std::shared_ptr<Fill>> fills;
    std::map<u64,std::shared_ptr<Done>> tails;
    std::map<u64,std::shared_ptr<Done>> evicting;
    unsigned target_count=0;sc_event changed;Resource engine;
    Slice(sc_module_name n,Context& ctx,unsigned i):sc_module(n),target("target"),memory("memory"),x(ctx),id(i),
        lines(x.c.get("sets")*x.c.get("ways")),rr(x.c.get("sets"),0){target.register_b_transport(this,&Slice::transport);}
    std::string key(const std::string& s)const{return "home_"+std::to_string(id)+"_"+s;}
    // Counter slots, bound on first use so the reported set is unchanged:
    // writebacks, target_full_events, merges, hits, mshr_full_events,
    // set_busy_events, misses, peak_mshrs, evictions, resolution_cycles,
    // maintenance_completed.
    std::array<u64*,11> slots{};std::string engine_key=key("engine");unsigned ways=unsigned(x.c.get("ways"));
    u64 service_cycles=x.c.get("service_cycles"),lookup_cycles=x.c.get("lookup_cycles"),targets=x.c.get("targets"),mshrs=x.c.get("mshrs");
    u64& cnt(unsigned i){
        static const char* names[11]={"writebacks","target_full_events","merges","hits","mshr_full_events","set_busy_events",
            "misses","peak_mshrs","evictions","resolution_cycles","maintenance_completed"};
        if(!slots[i])slots[i]=&x.counts[key(names[i])];
        return *slots[i];
    }
    void transfer(CacheLine& l,bool write,Request& r) {
        tlm::tlm_generic_payload tx;Request sub=r;sub.home=id;
        tx.set_extension(&sub);tx.set_address(l.address);tx.set_command(write?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
        tx.set_data_ptr(l.data.data());tx.set_data_length(64);tx.set_streaming_width(64);sc_time d=SC_ZERO_TIME;
        memory->b_transport(tx,d);tx.clear_extension<Request>();require(tx.is_response_ok(),"HBM response failure");
        if(write){l.dirty=false;++cnt(0);}
    }
    CacheLine* find(u64 a) {
        unsigned start=x.c.set(a)*ways;
        for(unsigned w=0;w<ways;++w){auto& l=lines[start+w];if(l.valid && l.address==a)return &l;}
        return nullptr;
    }
    CacheLine* victim(u64 a) {
        unsigned s=x.c.set(a),wN=ways,base=s*wN;
        for(unsigned w=0;w<wN;++w)if(!lines[base+w].valid && !lines[base+w].pins){rr[s]=(w+1)%wN;return &lines[base+w];}
        for(unsigned i=0;i<wN;++i){unsigned w=(rr[s]+i)%wN;if(!lines[base+w].pins){rr[s]=(w+1)%wN;return &lines[base+w];}}
        return nullptr;
    }
    static u64 load(const Line& d,unsigned off){u64 v=0;for(unsigned b=0;b<8;++b)v|=u64(d[off+b])<<(8*b);return v;}
    static u64 atomic(unsigned op,unsigned be,u64 old,u64 arg,u64 cmp) {
        bool wide=be==255;unsigned shift=(!wide && (be&16))?32:0;
        u64 mask=wide?UINT64_MAX:0xffffffffULL,a=(old>>shift)&mask,b=(arg>>shift)&mask,n=0;
        bool less=wide ? std::int64_t(a)<std::int64_t(b):std::int32_t(a)<std::int32_t(b);
        switch(op){case 0:n=a+b;break;case 1:n=a&b;break;case 2:n=a|b;break;case 3:n=a^b;break;
            case 4:case 11:n=b;break;case 5:n=less?a:b;break;case 6:n=less?b:a;break;
            case 7:n=std::min(a,b);break;case 8:n=std::max(a,b);break;
            case 9:n=a==(cmp&mask)?b:a;break;case 10:n=a;break;default:throw std::runtime_error("unsupported atomic");}
        return (n&mask)<<shift;
    }
    void transport(tlm::tlm_generic_payload& tx,sc_time& delay) {
        if(delay!=SC_ZERO_TIME){wait(delay);delay=SC_ZERO_TIME;}
        auto* r=tx.get_extension<Request>();require(r,"missing cache identity");
        require((r->op==0 && tx.is_read()) || (r->op>=1 && tx.is_write()),"operation/command mismatch");
        u64 a=tx.get_address(),base=a&~63ULL;x.c.address(a,8);
        require(a%8==0 && tx.get_data_length()==8 && tx.get_streaming_width()==8,"invalid word transaction");
        require(tx.get_data_ptr() && tx.get_byte_enable_ptr() && tx.get_byte_enable_length()==8,"word transaction needs data and byte enables");
        unsigned be=0;u64 input=0;
        for(unsigned b=0;b<8;++b){
            require(tx.get_byte_enable_ptr()[b]==0 || tx.get_byte_enable_ptr()[b]==255,"invalid TLM byte enable");
            if(tx.get_byte_enable_ptr()[b])be|=1<<b;
            input|=u64(tx.get_data_ptr()[b])<<(b*8);
        }
        require(be==r->be,"byte enable extension/payload mismatch");
        auto start=sc_time_stamp();auto previous=tails[base];auto self=std::make_shared<Done>();tails[base]=self;
        // Only same-line requests wait for their predecessors; a blocked miss
        // never holds the global lookup resource or prevents unrelated hits.
        engine.use(x,service_cycles,engine_key);wait(x.cycles(lookup_cycles));
        CacheLine* line=nullptr;std::shared_ptr<Fill> fill;bool target_owned=false;
        auto blocked=sc_time_stamp();
        for(;;){
            auto e=evicting.find(base);
            if(e!=evicting.end()){auto pending=e->second;while(!pending->done)wait(pending->event);continue;}
            auto f=fills.find(base);
            if(f!=fills.end() && !f->second->ready){
                if(target_count==targets){++cnt(1);wait(changed);continue;}
                fill=f->second;++fill->targets;++target_count;line=fill->line;++line->pins;target_owned=true;
                ++cnt(2);while(!fill->ready)wait(fill->event);break;
            }
            line=find(base);if(line){++line->pins;++cnt(3);break;}
            if(fills.size()>=mshrs){++cnt(4);wait(changed);continue;}
            if(target_count==targets){++cnt(1);wait(changed);continue;}
            line=victim(base);if(!line){++cnt(5);wait(changed);continue;}
            fill=std::make_shared<Fill>();fill->line=line;fill->targets=1;fills[base]=fill;
            ++target_count;++line->pins;target_owned=true;++cnt(6);
            cnt(7)=std::max<u64>(cnt(7),fills.size());
            bool was_valid=line->valid;line->valid=false;
            if(was_valid){
                ++cnt(8);
                if(line->dirty){
                    auto old=line->address;auto completion=std::make_shared<Done>();evicting[old]=completion;
                    transfer(*line,true,*r);completion->done=true;completion->event.notify(SC_ZERO_TIME);evicting.erase(old);
                }
            }
            line->address=base;transfer(*line,false,*r);line->valid=true;line->dirty=false;
            fill->ready=true;fill->event.notify(SC_ZERO_TIME);changed.notify(SC_ZERO_TIME);break;
        }
        cnt(9)+=x.to_cycles(sc_time_stamp()-blocked);
        if(previous)while(!previous->done)wait(previous->event);
        require(line->valid && line->address==base && line->pins,"cache reservation lost");
        u64 old=load(line->data,unsigned(a&63)),next=input;
        if(r->op>=2){require(r->be==255 || r->be==15 || r->be==240,"atomic requires aligned 32/64-bit mask");next=atomic(r->op-2,r->be,old,input,r->compare);}
        bool write=tx.is_write() && !(r->op==12); // AOP_LD is op 2+10
        if(write){for(unsigned b=0;b<8;++b)if(r->be&(1<<b))line->data[(a&63)+b]=unsigned(next>>(b*8))&255;line->dirty=true;}
        for(unsigned b=0;b<8;++b)if(tx.is_write() || (be&(1<<b)))tx.get_data_ptr()[b]=unsigned(old>>(8*b))&255;
        --line->pins;if(target_owned){--target_count;if(--fill->targets==0)fills.erase(base);}
        self->done=true;self->event.notify(SC_ZERO_TIME);if(tails[base]==self)tails.erase(base);
        changed.notify(SC_ZERO_TIME);tx.set_response_status(tlm::TLM_OK_RESPONSE);
        x.span("NUCA word",100+id,start,r->id);
    }
    void flush() {
        require(fills.empty() && tails.empty() && target_count==0,"maintenance started before requester drain");
        Request r;r.home=id;
        for(auto& line:lines)if(line.valid){require(!line.pins,"pinned line at maintenance");if(line.dirty)transfer(line,true,r);line.valid=false;}
        ++cnt(10);
    }
};
struct Network : sc_module {
    tlm_utils::multi_passthrough_target_socket<Network> target;
    sc_vector<tlm_utils::simple_initiator_socket<Network>> homes;
    struct Link {Resource resource;std::string key;u64* bytes=nullptr;};
    Context& x; std::map<std::pair<unsigned,unsigned>,Link> links;
    unsigned nx;u64 width,stages;
    Network(sc_module_name n,Context& ctx):sc_module(n),target("target"),homes("homes",ctx.c.cores()),x(ctx),
        nx(unsigned(ctx.c.get("nx"))),width(ctx.c.get("noc_bytes_per_cycle")),stages(ctx.c.get("noc_stages")){target.register_b_transport(this,&Network::transport);}
    void route(unsigned from,unsigned to,u64 bytes) {
        while(from!=to){unsigned next=from;
            if(from%nx!=to%nx)next=from%nx<to%nx?from+1:from-1;
            else next=from/nx<to/nx?from+nx:from-nx;
            auto& link=links[{from,next}];
            if(link.key.empty())link.key="link_"+std::to_string(from)+"_"+std::to_string(next);
            link.resource.use(x,(bytes+width-1)/width,link.key);
            if(!link.bytes)link.bytes=&x.counts[link.key+"_bytes"];
            *link.bytes+=bytes;wait(x.cycles(stages));from=next;
        }
    }
    void transport(int,tlm::tlm_generic_payload& tx,sc_time& delay) {
        if(delay!=SC_ZERO_TIME){wait(delay);delay=SC_ZERO_TIME;}
        auto* r=tx.get_extension<Request>();require(r && r->core<x.c.cores(),"invalid network source");
        r->home=x.c.home(tx.get_address());route(r->core,r->home,32);homes[r->home]->b_transport(tx,delay);route(r->home,r->core,16);
    }
};

// Passive b_transport interposer. No wait, delay annotation, payload extension,
// or event scheduling. Aggregate memory is bounded by elaborated socket count.
struct SocketMonitor : sc_module {
    tlm_utils::simple_target_socket<SocketMonitor> target;
    tlm_utils::simple_initiator_socket<SocketMonitor> downstream;
    Context& x; std::string label;unsigned lane;Context::SocketStats* stats=nullptr;
    SocketMonitor(sc_module_name n,Context& ctx,std::string key,unsigned tid):
        sc_module(n),target("target"),downstream("downstream"),x(ctx),label(std::move(key)),lane(tid){
        target.register_b_transport(this,&SocketMonitor::transport);
        if(x.monitor_mode!="off")stats=&x.sockets[label];
    }
    void transport(tlm::tlm_generic_payload& tx,sc_time& delay){
        if(!stats){downstream->b_transport(tx,delay);return;}
        auto& s=*stats;auto start=sc_time_stamp();u64 begin=x.now();
        s.integrate(begin);u64 sequence=++s.calls;s.bytes+=tx.get_data_length();
        s.reads+=tx.is_read();s.writes+=tx.is_write();s.peak=std::max(s.peak,++s.active);
        try {downstream->b_transport(tx,delay);}
        catch(...){finish(begin,true);throw;}
        finish(begin,tx.is_response_error());
        if(x.monitor_mode=="trace"){
            if(x.emitted>=x.c.get("trace_events")){++x.dropped;return;}
            if(x.emitted++)x.trace<<',';
            auto* r=tx.get_extension<Request>();
            x.trace<<"{\"name\":\""<<label<<"\",\"cat\":\"tlm-socket\",\"ph\":\"X\",\"pid\":2,\"tid\":"<<lane
                <<",\"ts\":"<<std::setprecision(15)<<start.to_seconds()*1e6<<",\"dur\":"<<(sc_time_stamp()-start).to_seconds()*1e6
                <<",\"args\":{\"transaction\":"<<sequence<<",\"address\":"<<tx.get_address()<<",\"bytes\":"<<tx.get_data_length()
                <<",\"write\":"<<(tx.is_write()?"true":"false")<<",\"core\":"<<(r?r->core:0)<<",\"home\":"<<(r?r->home:0)<<"}}";
        }
    }
    void finish(u64 begin,bool error){
        auto& s=*stats;u64 end=x.now(),duration=end-begin;s.integrate(end);--s.active;s.errors+=error;
        s.latency_sum+=duration;s.latency_max=std::max(s.latency_max,duration);
        unsigned bucket=0;for(u64 v=duration;v;v>>=1)++bucket;++s.histogram[bucket];
    }
};
