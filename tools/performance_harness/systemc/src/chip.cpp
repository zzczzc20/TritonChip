#include <cstdio>
// Native event-driven execution of compiled TNCA machine words. No RTL runtime.
//
// Every instruction word is decoded once into an `Inst` (the classifier mode
// prints that decision for the opcode inventory) and executed against the
// architectural state of ISA v0.1 with the RTL as the semantic authority.
// Anything the executor does not implement is refused explicitly; it is never
// executed as a no-op. All latencies come from the configuration and are
// UNCALIBRATED hypotheses. Not a complete-chip qualification.
#include "../include/tnca_model.hpp"
#include "../include/tnca_numeric.hpp"
#include "../include/tnca_sfu.hpp"
#include "../include/tnca_vector_ops.hpp"
#include "../include/tnca_matrix_ops.hpp"
#include "../include/tnca_barrier_model.hpp"
#include "../include/tnca_tensor_engine.hpp"
#include "../include/tnca_timeline_memory.hpp"
#include "tnca_isa_generated.hpp"
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unistd.h>
#include <chrono>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#ifdef TNCA_HAVE_RAMULATOR2
#include "tnca_dram_bridge.hpp"
#endif

namespace trapcode {constexpr unsigned illegal=1,descriptor=4,barrier=5,resource=6,user=0x40;}
enum Status : unsigned {STATUS_OK=0,STATUS_FAULT=1,STATUS_TIMEOUT=2,STATUS_REFUSED=3};
struct DeadlineExpired {};

static bool compare(u64 a,u64 b,unsigned cc,unsigned bits=64){
    auto sa=bits==64?std::int64_t(a):std::int64_t(std::int32_t(a));
    auto sb=bits==64?std::int64_t(b):std::int64_t(std::int32_t(b));
    switch(cc){case 0:return a==b;case 1:return a!=b;case 2:return sa<sb;case 3:return sa<=sb;
        case 4:return sa>sb;case 5:return sa>=sb;case 6:return a<b;case 7:return a<=b;
        case 8:return a>b;case 9:return a>=b;default:throw std::runtime_error("uncovered comparison condition");}
}
static u64 alu(unsigned f,u64 a,u64 b,unsigned bits,unsigned cc){
    const u64 mask=bits==64?UINT64_MAX:0xffffffffULL;a&=mask;b&=mask;
    auto sa=bits==64?std::int64_t(a):std::int64_t(std::int32_t(a));
    auto sb=bits==64?std::int64_t(b):std::int64_t(std::int32_t(b));
    switch(f){case 0:return a;case 1:return (a+b)&mask;case 2:return (a-b)&mask;case 3:return (a*b)&mask;
        case 4:return bits==64?tnca_numeric::multiply_high(a,b,true):(u64(std::int64_t(std::int32_t(a))*std::int64_t(std::int32_t(b)))>>32)&mask;
        case 5:return bits==64?tnca_numeric::multiply_high(a,b,false):(a*b)>>32;
        case 8:return a&b;case 9:return a|b;case 10:return a^b;case 11:return a&~b;
        case 12:return (a<<(b&(bits-1)))&mask;case 13:return a>>(b&(bits-1));
        case 14:return u64(sa>>(b&(bits-1)))&mask;
        case 16:return sa<sb?a:b;case 17:return sa>sb?a:b;case 18:return std::min(a,b);case 19:return std::max(a,b);
        case 20:return bits==64?(a?unsigned(__builtin_clzll(a)):64):(a?unsigned(__builtin_clz(unsigned(a))):32);
        case 21:return unsigned(__builtin_popcountll(a));case 22:return u64(std::int64_t(std::int32_t(a)));
        case 23:return a&0xffffffffULL;case 24:return compare(a,b,cc,bits);
        default:throw std::runtime_error("uncovered ALU operation "+std::to_string(f));}
}
static unsigned immediate_alu(unsigned func){return func==32?1:func==33?2:func==34?3:func-32;}
static bool zero_extended_immediate(unsigned func){return func==40 || func==41 || func==42 || func==44 || func==45 || func==46;}

//==============================================================================
// Decode once: legality, register-class checks and the executor's verdict.
//==============================================================================
enum class K : std::uint8_t {Refused,Illegal,Nop,Special,Salu,Branch,Yield,Trap,Exit,Lsu,Atomic,Fence,Vint,Vfp,Cvt,
    Pred,Shuffle,Extract,Insert,Reduce,Sfu,Desc,Xfer,Prefetch,Barrier,MZero,Mma,AVread,ARow,AZero,Count};
static const char* kind_name(K k){
    static const char* names[]={"refused","illegal","nop","special","salu","branch","yield","trap","exit","lsu","atomic",
        "fence","vint","vfp","cvt","pred","shuffle","extract","insert","reduce","sfu","desc","xfer","prefetch","barrier",
        "m_zero","mma","a_vread","a_row","a_zero"};
    return names[unsigned(k)];
}
struct Inst {
    u64 word=0;unsigned opcode=0,major=0,func=0,d=0,a=0,b=0,c=0,pred=0,mod=0,imm=0;bool invert=false;u64 simm=0;
    K kind=K::Refused;std::string why;
};
static const char* timing_class(const Inst& i){
    switch(i.kind){
        case K::Nop:case K::Branch:case K::Yield:case K::Exit:return "issue";
        case K::Special:return "issue+scalar_latency";
        case K::Salu:return (i.func==3 || i.func==4 || i.func==5 || i.func==34)?"issue+scalar_mul_latency":"issue+scalar_latency";
        case K::Trap:return "issue+fault";
        case K::Lsu:return "issue+lsu_slots+noc+nuca+hbm(+load_latency)";
        case K::Atomic:return "issue+lsu_slots+noc+home_slice_rmw";
        case K::Fence:return "issue+own_scoreboard_drain";
        case K::Vint:return (i.func==24 || i.func==56)?"issue+compare_latency":"issue+vector_int_latency";
        case K::Vfp:return i.func==24?"issue+compare_latency":"issue+vector_fp_latency";
        case K::Cvt:return "issue+convert_latency";
        case K::Pred:return "issue+predicate_latency";
        case K::Shuffle:case K::Extract:case K::Insert:return "issue+shuffle_latency";
        case K::Reduce:return i.func>=8?"issue+fp_reduction_latency":"issue+reduction_latency";
        case K::Sfu:return "issue+sfu_occupancy+sfu_latency";
        case K::Desc:return "issue+tensor_engine_admission(+descriptor_snapshot_via_noc_nuca)";
        case K::Xfer:return "issue+tensor_engine(dma_units,noc,nuca,hbm)+barrier_completion";
        case K::Prefetch:return "issue(hint_dropped)";
        case K::Barrier:return "issue+barrier_latency+event_wake";
        case K::MZero:case K::AZero:case K::ARow:return "issue+matrix_engine_occupancy";
        case K::Mma:return "issue+matrix_engine_occupancy+barrier_completion";
        case K::AVread:return "issue+matrix_engine_free+acc_read_latency";
        default:return "none";
    }
}
static Inst decode(u64 w){
    Inst i;i.word=w;i.major=unsigned(w>>56);i.func=unsigned(w>>48)&255;i.d=unsigned(w>>40)&255;i.a=unsigned(w>>32)&255;
    i.b=unsigned(w>>24)&255;i.c=unsigned(w>>16)&255;i.pred=unsigned(w>>12)&15;i.invert=(w>>10)&1;i.mod=unsigned(w)&1023;
    i.imm=(i.b<<8)|i.c;i.simm=u64(std::int64_t(std::int16_t(i.imm)));i.opcode=(i.major<<8)|i.func;
    auto illegal=[&](const std::string& why){i.kind=K::Illegal;i.why="ILLEGAL_INSTRUCTION: "+why;return i;};
    auto refuse=[&](const std::string& why){i.kind=K::Refused;i.why="unimplemented: "+why;return i;};
    auto as=[&](K k){i.kind=k;return i;};
    if(!isa::defined(i.opcode))return illegal("undefined ISA opcode");
    if((w>>11)&1)return illegal("extension word bit (none in v0.1)");
    if(!isa::maskable(i.opcode) && (i.pred || i.invert))return illegal("predicate on unmaskable instruction");
    const unsigned m=i.mod,f=i.func;
    auto vregs=[&](std::initializer_list<unsigned> r){for(unsigned x:r)if(x>=64)return false;return true;};
    switch(i.major){
    case 0:
        if(f==0)return (i.d|i.a|i.b|i.c|m)?illegal("nop fields"):as(K::Nop);
        if(f==1)return i.d>=64?illegal("scalar index"):as(K::Special);
        return illegal("s.wrspecial: v0.1 defines no writable special register");
    case 1:{
        bool imm_form=f>=32,unary=f==0 || f==20 || f==21 || f==22 || f==23;
        if(i.d>=64 || (f<64 && i.a>=64) || (!imm_form && !unary && i.b>=64) || (f==25 && i.c>=64))return illegal("scalar index");
        if(f>=64 && i.a)return illegal("mov-wide source field");
        if((f==24 || f==56) && (m&15)>9)return illegal("integer condition code");
        if(f>=64 && f<=66 && m>3)return illegal("lsl modifier");
        return as(K::Salu);}
    case 2:
        if(f==16)return as(K::Yield);
        if(f==24)return as(K::Trap);
        if(f==31)return as(K::Exit);
        if(((f==1 || f==2 || f==4 || f==9) && i.a>=64) || (f==8 && i.d>=64))return illegal("scalar index");
        return as(K::Branch);
    case 16:
        if((f==24 || f==56)){if(i.d>=16)return illegal("predicate destination");if((m&15)>9)return illegal("integer condition code");
            if(!vregs({i.a}) || (f==24 && i.b>=64))return illegal("vector index");
            return as(K::Vint);}
        if(f==26){if(i.a>=64 || i.d>=64)return illegal("register index");return as(K::Vint);}
        if(f==27 || f==64)return i.d>=64?illegal("vector index"):as(K::Vint);
        if(!vregs({i.d,i.a}) || (!(f>=32 && f<=63) && f!=0 && f!=20 && f!=21 && i.b>=64))return illegal("vector index");
        return as(K::Vint);
    case 17:
        if(!vregs({i.a}) || (f!=20 && f!=21 && i.b>=64) || ((f==6 || f==7) && i.c>=64))return illegal("vector index");
        if(f==24){if(i.d>=16)return illegal("predicate destination");unsigned cc=m&15;
            if(m>>4 || (cc>5 && cc<10) || cc>11)return illegal("floating condition code");
            return as(K::Vfp);}
        if(i.d>=64)return illegal("vector index");
        if(f<=7)return (m&~0x100U)?illegal("FP modifier"):as(K::Vfp);
        if(f==16 || f==17)return (m>2)?illegal("FP min/max mode"):as(K::Vfp);
        return m?illegal("FP modifier"):as(K::Vfp);
    case 18:
        if(!vregs({i.d,i.a}))return illegal("vector index");
        if(f==4)return ((m>>6) || (m&7)>4 || ((m>>3)&7)>4)?illegal("conversion format"):as(K::Cvt);
        if(f<2)return m?illegal("conversion modifier"):as(K::Cvt);
        if(f<=3)return m>3?illegal("conversion rounding mode"):as(K::Cvt);
        return illegal("reserved conversion");
    case 19:
        if(!((i.a<16 || f==19) && (f>=16 || (i.d<16 && i.b<16))))return illegal("predicate index");
        if(f==19 && (i.a>=64 || i.d>=16))return illegal("register index");
        if(f>=16 && f!=19 && i.d>=64)return illegal("scalar index");
        return as(K::Pred);
    case 20:
        if(f==8){if(i.d>=64 || i.a>=64 || (m>>4))return illegal("extract fields");return as(K::Extract);}
        if(f==9){if(i.d>=64 || i.a>=64 || (m>>4))return illegal("insert fields");return as(K::Insert);}
        if(!vregs({i.d,i.a}) || i.b>=64)return illegal("register index");
        return as(K::Shuffle);
    case 21:
        if((m&~1U) || i.a>=64 || i.d>=64)return illegal("reduction fields");
        return as(K::Reduce);
    case 22:
        if(m || !vregs({i.d,i.a}))return illegal("SFU fields");
        return as(K::Sfu);
    case 32:{
        if((m&7)>3 || (m>>6) || ((m>>4)&3)==3)return illegal("reserved memory modifiers");
        bool vector=f>=16;
        if(vector && (m&7)==3)return illegal("vector access width");
        if(i.a>=64 || (vector?i.d>=64:i.d>=64))return illegal("register index");
        if((f==32 || f==40) && i.b>=64)return illegal("index vector");
        return as(K::Lsu);}
    case 33:
        if(((m>>5)&3)==3)return illegal("reserved atomic scope");
        if(((m>>5)&3)==2)return illegal("system scope without CAPABILITY0.SYSTEM_ATOMICS");
        if(f==32)return (m&~0x78U)?illegal("fence modifier"):as(K::Fence);
        if((m&7)!=2 && (m&7)!=3)return illegal("atomic size");
        if(m>>7)return illegal("atomic modifier");
        return refuse("device-scope atomics (not used by the frozen suites; old-value width/extension not yet verified against rtl/core/tnca_lsu.v)");
    case 48:
        if(m)return illegal("descriptor/window modifier");
        if((f==0 || f==2) && i.d>=16)return illegal("descriptor index");
        if((f==16 || f==17) && i.d>=32)return illegal("window index");
        if(f==0 && i.a>=64)return illegal("scalar index");
        return as(K::Desc);
    case 49:
        if((m&~0x27U) || (m&7)>3)return illegal("transfer encoding");
        if(f==16)return as(K::Prefetch);
        if(i.b>=63 || i.c>=64)return illegal("transfer coordinate/token register");
        if(i.d>=32 || i.a>=16)return illegal("window/descriptor index");
        return as(K::Xfer);
    case 64:
        if(f==0)return (i.d>=16 || i.a || i.b || i.c || m)?illegal("m.zero fields"):as(K::MZero);
        if((m&3)==3 || (m>>5))return illegal("reserved MMA K/modifier");
        if(i.d>=16 || i.c>=64 || ((m&16) && (i.c&63)==63))return illegal("MMA accumulator/token register");
        return as(K::Mma);
    case 65:
        if(f==0)return (i.d>=64 || i.a>=16 || i.b>=64 || m)?illegal("a.vread fields"):as(K::AVread);
        if(f==8)return (i.d>=16 || m)?illegal("a.zero fields"):as(K::AZero);
        if(i.d>=16 || i.a>=64 || i.b>=64 || i.c>=64 || m)return illegal("accumulator row fields");
        return as(K::ARow);
    case 80:
        if(m)return illegal("barrier modifier");
        if(f==0 && i.d>=32)return illegal("barrier index");
        if((f==1 || f==2 || f==5) && i.a>=32)return illegal("barrier index");
        return as(K::Barrier);
    default:return refuse("instruction group major="+std::to_string(i.major));
    }
}

// Instruction words decoded once per address. Host writes overlapping a
// cached word invalidate the whole cache (code changes only on module load).
struct CodeCache {
    std::function<u64(u64)> read_word;
    std::unordered_map<u64,Inst> decoded;u64 lo=UINT64_MAX,hi=0;
    const Inst& at(u64 pc){
        auto it=decoded.find(pc);
        if(it!=decoded.end())return it->second;
        lo=std::min(lo,pc);hi=std::max(hi,pc+8);
        return decoded.emplace(pc,decode(read_word(pc))).first->second;
    }
    void invalidate(u64 a,u64 n){if(a<hi && a+n>lo){decoded.clear();lo=UINT64_MAX;hi=0;}}
};

//==============================================================================
// Execution state
//==============================================================================
struct Frame {
    std::shared_ptr<PhysicalRegisters> physical;
    unsigned vbase=0;
    std::array<u64,64> s{};std::array<PhysicalRegisters::Vector,64> v{};
    std::array<unsigned,16> p{};
    std::array<sc_time,64> sr{};std::array<sc_time,64> vr{};std::array<sc_time,16> pr{};
    u64 pc=0,arg=0,retired=0;unsigned px=0,py=0,pz=0,nx=1,ny=1,nz=1,role=0,role_count=1;
    bool sliced=false;unsigned slice_id=0,slice_k=2;
    sc_time fetch_ready=SC_ZERO_TIME; // core_pipeline: refetch after a taken branch
    sc_time ahead=SC_ZERO_TIME;       // time_quantum: this context's local time - simulation time
    Frame(std::shared_ptr<PhysicalRegisters> bank,unsigned base):physical(std::move(bank)),vbase(base){require(base<=192,"vector mapping overflow");p[0]=65535;}
};
// One program's launch on one core: the role mask's context bits, RPAIR
// (bit 28) and RSLICE (bit 31, log2 slices in bits 30:29; 0 means every
// context, RCTX_W = 2) exactly as tb/device/tnca_device.v hands them over.
struct Launch {
    u64 pc=0,arg=0;unsigned px=0,py=0,pz=0,nx=1,ny=1,nz=1,roles=1;bool pair=false,slice=false;unsigned slice_k=2;
    static Launch from_mask(u64 mask){
        Launch l;l.roles=unsigned(mask&15);l.pair=(mask>>28)&1;l.slice=(mask>>31)&1;
        unsigned k=unsigned(mask>>29)&3;l.slice_k=k?k:2;
        require(l.roles && !(mask&0x0ffffff0ULL) && mask<=0xffffffffULL,"role mask outside four contexts and slice flags");
        return l;
    }
};
struct Outcome {unsigned status=STATUS_OK,code=0;u64 pc=0,retired=0;std::string why;};

struct Core : sc_module {
    tlm_utils::simple_initiator_socket<Core> memory;
    Context& x;unsigned id;CodeCache& code;sc_time cycle;Resource issue;Limit lsu;sc_time sfu_available=SC_ZERO_TIME;
    std::array<BarrierModel,32> barriers;
    TensorEngine tensor;Pool beats;
    std::shared_ptr<PhysicalRegisters> vrf=std::make_shared<PhysicalRegisters>();
    using Row=std::array<std::uint32_t,16>;
    std::array<std::array<Row,16>,16> acc{};
    sc_time mtx_free=SC_ZERO_TIME;unsigned mtx_role=0;
    // Timeline path (memory_model=1): the core's single memory port (LSU and
    // transfer engine share it, tnca_mem_mux.v), LSU instructions in flight
    // (LSU_INFLIGHT) and each context's outstanding accesses.
    TimelineMemory* tm=nullptr;Timeline port;std::vector<u64> lsu_inflight;std::array<u64,4> role_lsu_done{};
    // Program-scope fault and deadline state.
    bool aborting=false;Outcome fault;sc_time deadline=SC_ZERO_TIME;
    // Timing constants (configuration, uncalibrated).
    u64 t_issue,t_scalar,t_mul,t_vint,t_vfp,t_cvt,t_cmp,t_pred,t_shuf,t_red,t_fred,t_sfu_occ,t_sfu,t_load,t_bar,t_acc,
        t_rf,t_mac,t_mac_unbuf,t_pipe,t_done,t_zero,t_row,t_launch;bool abuf;
    // What-if packed MMA (m.mma.dp, hypothetical; Config optional keys):
    // G consecutive k per lane per step, timing only (numerics stay per-K).
    u64 dp_g16=1,dp_g8=1;bool dp_pf=false;u64* c_dp=nullptr;
    // Counter slots bound once.
    u64 *c_instr,*c_sfu_queue,*c_sleep,*c_dep_wait,*c_mtx_wait,*c_mtx_busy,*c_mtx_cmds,*c_mma,*c_lsu_wait,*c_arrive_wait,
        *c_fence_wait,*c_programs_issued,*c_programs_completed;
    std::array<u64,unsigned(K::Count)> class_counts{};
    std::array<u64*,4> c_role_issued{},c_role_completed{};
    bool tracing=false;sc_time issue_time;std::string lsu_key;u64 t_lsu_slots=2,t_beat_slots=8;
    // core_pipeline=1: the RTL's single in-order pipeline (tnca_core.v).
    bool pipe=false;u64 t_branch,t_park,n_banks,t_bank,t_lsu_setup,t_vload_extra,t_lsu_complete,t_lsu_wb,t_fcmp,t_shuf_s,t_red_vec,t_fred_vec;
    bool vrf_pf=false,r_reads=false;u64 last_issue=0;
    // The two load/store units of tnca_lsu_pair: a unit is free once its
    // last response is back (+lsu_complete_cycles); the second unit takes
    // an access only after the first has sent all of its beats.
    struct LsuUnit {u64 free=0,drained=0;};std::array<LsuUnit,2> units{};
    std::vector<u64> steals; // LSU completions taking the one writeback port
    unsigned holds=0;sc_event unhold; // a fence/arrive held in R freezes every context
    // time_quantum > 0: conservative lookahead. A context whose next wait
    // ends before any other pending activity of the simulation (and within
    // the quantum) skips it and runs on at a local time ahead of the
    // simulation time: nothing else can happen in between, so the order of
    // every action is unchanged. It waits for real when another process is
    // due first, and synchronises before it touches state that other
    // processes wait on (engines, barriers, transfer commands, exit).
    u64 quantum=0;sc_time quantum_t;unsigned untimed_burst=0;
    u64 *c_freeze,*c_park,*c_bank,*c_branch;
    std::string key(const std::string& s)const{return "core_"+std::to_string(id)+"_"+s;}
    Core(sc_module_name n,Context& ctx,unsigned i,CodeCache& cc):sc_module(n),memory("memory"),x(ctx),id(i),code(cc),
        cycle(ctx.cycles(1)),lsu(unsigned(ctx.c.get("lsu_slots"))),
        tensor("tensor",ctx,i,[this](u64 token){verify_async_token(token);},[this](u64 token,bool error){complete_async(token,error);}),
        beats(unsigned(ctx.c.get("lsu_beat_slots"))){
        auto g=[&](const char* k){return x.c.get(k);};
        t_issue=g("issue_cycles");t_scalar=g("scalar_latency");t_mul=g("scalar_mul_latency");t_vint=g("vector_int_latency");
        t_vfp=g("vector_fp_latency");t_cvt=g("convert_latency");t_cmp=g("compare_latency");t_pred=g("predicate_latency");
        t_shuf=g("shuffle_latency");t_red=g("reduction_latency");t_fred=g("fp_reduction_latency");t_sfu_occ=g("sfu_occupancy");
        t_sfu=g("sfu_latency");t_load=g("load_latency");t_bar=g("barrier_latency");t_acc=g("acc_read_latency");
        t_rf=g("mtx_rf_read_cycles");t_mac=g("mtx_mac_cycles");t_mac_unbuf=g("mtx_mac_unbuffered_cycles");t_pipe=g("mtx_pipe_cycles");
        t_done=g("mtx_complete_cycles");t_zero=g("mtx_zero_cycles");t_row=g("mtx_row_cycles");t_launch=g("launch_cycles");abuf=g("mtx_abuf")!=0;
        dp_g16=x.c.get_or("mtx_dp_group_16",1);dp_g8=x.c.get_or("mtx_dp_group_8",1);dp_pf=x.c.get_or("mtx_dp_b_prefetch",0)!=0;
        // Bound only when a what-if key is active, so default counters are unchanged.
        if(dp_g16>1 || dp_g8>1 || dp_pf)c_dp=&x.counts[key("matrix_dp_whatif_commands")];
        issue_time=x.cycles(t_issue);lsu_key=key("lsu");t_lsu_slots=g("lsu_slots");t_beat_slots=g("lsu_beat_slots");
        pipe=g("core_pipeline")!=0;t_branch=g("branch_taken_cycles");t_park=g("park_refetch_cycles");n_banks=g("vrf_banks");
        t_bank=g("vrf_bank_conflict_cycles");t_lsu_setup=g("lsu_setup_cycles");t_vload_extra=g("lsu_vector_load_extra");
        t_lsu_complete=g("lsu_complete_cycles");t_lsu_wb=g("lsu_writeback_cycles");t_fcmp=g("fp_compare_latency");
        t_shuf_s=g("shuffle_scalar_latency");t_red_vec=g("reduction_vector_extra");t_fred_vec=g("fp_reduction_vector_extra");
        vrf_pf=g("vrf_prefetch")!=0;quantum=g("time_quantum");quantum_t=x.cycles(quantum);
        c_freeze=&x.counts[key("pipeline_freeze_cycles")];c_park=&x.counts[key("context_park_cycles")];
        c_bank=&x.counts[key("vrf_bank_conflict_cycles")];c_branch=&x.counts[key("branch_refetch_cycles")];
        port.service=&x.counts[key("memory_port_service_cycles")];port.queue=&x.counts[key("memory_port_queue_cycles")];
        issue.bind(x,key("issue"));
        c_instr=&x.counts[key("instructions")];c_sfu_queue=&x.counts[key("sfu_queue_cycles")];c_sleep=&x.counts[key("barrier_sleep_cycles")];
        c_dep_wait=&x.counts[key("dependency_wait_cycles")];c_mtx_wait=&x.counts[key("matrix_engine_wait_cycles")];
        c_mtx_busy=&x.counts[key("matrix_engine_busy_cycles")];c_mtx_cmds=&x.counts[key("matrix_engine_commands")];
        c_mma=&x.counts[key("mma_commands")];c_lsu_wait=&x.counts[key("lsu_wait_cycles")];c_arrive_wait=&x.counts[key("arrive_drain_cycles")];
        c_fence_wait=&x.counts[key("fence_drain_cycles")];
        c_programs_issued=&x.counts[key("programs_issued")];c_programs_completed=&x.counts[key("programs_completed")];
        for(unsigned r=0;r<4;++r){c_role_issued[r]=&x.counts[key("role_"+std::to_string(r)+"_issued")];c_role_completed[r]=&x.counts[key("role_"+std::to_string(r)+"_completed")];}
        tracing=x.monitor_mode=="trace";
        start_roles();start_completions();
    }
    u64 cycles_of(sc_time t)const{return x.to_cycles(t);}
    void verify_async_token(u64 token){
        unsigned index=unsigned(token&65535);arch(index<32,trapcode::barrier,"BARRIER_FAULT: asynchronous token index");
        auto& barrier=barriers[index];
        arch(!barrier.error && barrier.armed && barrier.epoch==(token>>16) && barrier.operations>0,trapcode::barrier,"BARRIER_FAULT: asynchronous command generation/count");
    }
    void complete_async(u64 token,bool error){
        unsigned index=unsigned(token&65535);require(index<32,"asynchronous token index");
        if(error){barriers[index].invalidate();return;}
        if(barriers[index].error)return; // the launch has faulted; the fault is what is reported
        require(barriers[index].epoch==(token>>16),"BARRIER_FAULT: command completed in a different generation");
        barriers[index].complete();
    }
    void ready(sc_time t){if(x.functional)return;auto now=sc_time_stamp();if(t>now){*c_dep_wait+=cycles_of(t-now);wait(t-now);}}
    sc_time lt(const Frame& f)const{return sc_time_stamp()+f.ahead;}
    u64 lnow(const Frame& f)const{return x.to_cycles(lt(f));}
    void sync(Frame& f){if(f.ahead!=SC_ZERO_TIME){sc_time d=f.ahead;f.ahead=SC_ZERO_TIME;wait(d);}}
    // Advance this context to time t.
    void go(Frame& f,sc_time t){
        if(x.functional)return;
        sc_time now=sc_time_stamp();
        if(!quantum){if(t>now)wait(t-now);return;}
        if(t<=now+f.ahead)return;
        if(t-now<quantum_t && t-now<sc_time_to_pending_activity()){f.ahead=t-now;return;}
        f.ahead=SC_ZERO_TIME;wait(t-now);
    }
    void ready(Frame& f,sc_time t){auto now=lt(f);if(t>now){*c_dep_wait+=cycles_of(t-now);go(f,t);}}
    // A register hazard parks the context while others run; it refetches
    // park_refetch_cycles after the wake (tnca_core.v rp_go/rctx_wake).
    void hazard(Frame& f,sc_time t){
        auto now=lt(f);if(t<=now)return;
        if(pipe && role_remaining>1){*c_park+=t_park;t+=x.cycles(t_park);}
        *c_dep_wait+=cycles_of(t-now);go(f,t);
    }
    // A stall that does not switch contexts freezes the shared pipeline.
    void freeze(Frame& f,sc_time t){
        auto now=lt(f);if(t<=now)return;
        if(x.functional)return;
        if(pipe){issue.available=std::max(issue.available,t);*c_freeze+=cycles_of(t-now);}
        go(f,t);
    }
    void issue_slot(Frame& f){
        if(x.functional){if(++untimed_burst>=4096){untimed_burst=0;wait(SC_ZERO_TIME);}return;} // let spinning peers run
        if(!quantum){issue.use(x,issue_time,t_issue);return;}
        sc_time t=lt(f),begin=std::max(issue.available,t);
        *issue.service+=t_issue;*issue.queue+=cycles_of(begin-t);
        issue.available=begin+issue_time;go(f,issue.available);
    }
    // A stall of unknown length that does not switch (fence, arrive): the
    // other contexts cannot issue until it ends.
    template<class F> void hold(F&& body){
        if(!pipe){body();return;}
        auto t=sc_time_stamp();++holds;
        try{body();}catch(...){if(--holds==0)unhold.notify(SC_ZERO_TIME);throw;}
        if(--holds==0)unhold.notify(SC_ZERO_TIME);
        *c_freeze+=cycles_of(sc_time_stamp()-t);
    }
    u64 lsu_idle()const{u64 m=0;for(auto v:lsu_inflight)m=std::max(m,v);for(auto& u:units)m=std::max(m,u.free);return m;}
    // RSLICE (tnca_core.v): an index the instruction names gets the slice
    // number ORed into its top bits -- accumulators (ACC_W 4), barriers
    // (BAR_W 5), descriptors and windows (DESC_W 4).
    static unsigned sliced(const Frame& f,unsigned index,unsigned width){return f.sliced?(index|(f.slice_id<<(width-f.slice_k))):index;}
    u64 token_of(const Frame& f,u64 token){
        unsigned index=unsigned(token&65535);arch(index<32,trapcode::barrier,"BARRIER_FAULT: token index");
        return (token&~u64(65535))|sliced(f,index,5);
    }
    void publish(Frame& f,unsigned index,unsigned mask,u64 latency){
        require(index<64,"logical vector register exceeds role allocation");
        f.vr[index]=lt(f)+x.cycles(x.functional?0:latency);
        f.physical->write(f.vbase+index,mask,f.vr[index],f.v[index]);
    }
    u64 word(u64 address,bool store,u64 data,unsigned be,u64 instruction){
        if(!store)be=255; // NUCA returns the complete aligned word; LSU extracts bytes.
        Request ext;ext.id=instruction;ext.core=id;ext.op=store?1:0;ext.be=be;
        std::array<unsigned char,8> bytes{},enables{};
        for(unsigned b=0;b<8;++b){bytes[b]=static_cast<unsigned char>(data>>(b*8));enables[b]=be&(1<<b)?255:0;}
        tlm::tlm_generic_payload tx;tx.set_address(address);tx.set_command(store?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
        tx.set_data_ptr(bytes.data());tx.set_data_length(8);tx.set_streaming_width(8);
        tx.set_byte_enable_ptr(enables.data());tx.set_byte_enable_length(8);tx.set_extension(&ext);
        sc_time delay=SC_ZERO_TIME;memory->b_transport(tx,delay);tx.clear_extension<Request>();
        require(tx.is_response_ok() && delay==SC_ZERO_TIME,"core memory transport failed");
        u64 result=0;for(unsigned b=0;b<8;++b)result|=u64(bytes[b])<<(b*8);return result;
    }
    u64 special(unsigned spec,const Frame& f){
        switch(spec){case 0:return f.px;case 1:return f.py;case 2:return f.pz;case 4:return f.nx;case 5:return f.ny;case 6:return f.nz;
            case 8:return f.role;case 9:return f.role_count;case 10:return id;case 11:return x.c.cores();case 12:return 16;case 13:return 64;
            case 14:return 5;case 15:return 16;case 16:return lnow(f);case 17:return f.retired;case 18:return 32;case 19:return 32;
            case 20:return 16;case 21:return 256;case 32:return f.arg;case 48:return 1;
            case 49:return 0x9efULL; // VECTOR_FP32|SFU|GATHER|ATOM_DEV|MATRIX_F16|MATRIX_BF16|TENSOR_DMA|MBARRIER|MATRIX_FP8
            case 64:case 65:return 0; // the faulting context stops; no surviving reader in this model
            default:return 0;}        // ISA sec 5.1: unimplemented ids read 0
    }
    void drain_own(Frame& f){
        for(auto& t:f.sr)ready(f,t);
        for(auto& t:f.vr)ready(f,t);
        for(auto& t:f.pr)ready(f,t);
        if(tm && role_lsu_done[f.role]>lnow(f))ready(f,x.cycles(role_lsu_done[f.role]));
        f.physical->commit(lt(f));
    }
    //--------------------------------------------------------------------------
    // Matrix engine: one command at a time, core-wide (tnca_matrix.v). Every
    // matrix-FU instruction, a.vread included, waits for it to be free.
    //--------------------------------------------------------------------------
    void engine_wait(){if(x.functional)return;auto now=sc_time_stamp();if(mtx_free>now){*c_mtx_wait+=cycles_of(mtx_free-now);wait(mtx_free-now);}}
    void engine_start(const Frame& f,u64 busy){if(x.functional)busy=0;mtx_free=sc_time_stamp()+x.cycles(busy);mtx_role=f.role;*c_mtx_busy+=busy;++*c_mtx_cmds;}
    void mma(Frame& f,const Inst& in){
        sync(f);
        unsigned slot=sliced(f,in.d,4),K=16U<<(in.mod&3),dt=(in.mod>>2)&3,bytes=tnca_matrix_ops::element_bytes(dt);
        bool source_free=in.mod&16;
        u64 token=token_of(f,f.s[in.c]),sf_token=source_free?token_of(f,f.s[(in.c+1)&63]):0;
        unsigned a_regs=tnca_matrix_ops::fragment_registers(16,K,dt),b_regs=tnca_matrix_ops::fragment_registers(K,16,dt);
        // The RTL reads fragments without a scoreboard (tnca_decoder.v: "no read
        // port and no dependency"). The model waits for this context's own
        // in-flight vector writes to land before reading them at issue.
        for(unsigned r=0;r<a_regs;++r)if(in.a+r<64)ready(f.vr[in.a+r]);
        for(unsigned r=0;r<b_regs;++r)if(in.b+r<64)ready(f.vr[in.b+r]);
        ready(f.sr[in.c]);if(source_free)ready(f.sr[(in.c+1)&63]);
        engine_wait();
        verify_async_token(token);if(source_free)verify_async_token(sf_token);
        std::vector<PhysicalRegisters::Vector> A(a_regs),B(b_regs);
        for(unsigned r=0;r<a_regs;++r){unsigned p=f.vbase+in.a+r;require(p<256,"MMA A fragment beyond the physical file");A[r]=f.physical->read(p,sc_time_stamp());}
        for(unsigned r=0;r<b_regs;++r){unsigned p=f.vbase+in.b+r;require(p<256,"MMA B fragment beyond the physical file");B[r]=f.physical->read(p,sc_time_stamp());}
        {static const bool mlog=std::getenv("TLM_DEBUG_MMA_LOG")!=nullptr;
         if(mlog){u64 ha=1469598103934665603ULL,hb=ha;for(auto& v:A)for(auto w:v)ha=(ha^w)*1099511628211ULL;for(auto& v:B)for(auto w:v)hb=(hb^w)*1099511628211ULL;
            std::fprintf(stderr,"MMA core %u role %u t %llu a %u b %u ha %016llx hb %016llx\n",id,f.role,(unsigned long long)x.now(),in.a,in.b,(unsigned long long)ha,(unsigned long long)hb);
            for(unsigned r=0;r<b_regs;++r){std::fprintf(stderr,"MMAB %u:",r);for(auto w:B[r])std::fprintf(stderr," %08x",w);std::fprintf(stderr,"\n");}
            std::fprintf(stderr,"MMAPENDING %zu\n",f.physical->pending.size());for(auto& w:f.physical->pending)std::fprintf(stderr,"  pend idx %u mask %x due %llu\n",w.index,w.mask,(unsigned long long)x.to_cycles(w.due));}}
        tnca_matrix_ops::mma(acc[slot],A,B,K,dt);
        bool buffered=abuf && (bytes==1?K!=64:K==16);
        u64 sweep=(buffered?a_regs*t_rf:0)+K*(t_rf+16*(buffered?t_mac:t_mac_unbuf));
        unsigned dp_g=unsigned(bytes==1?dp_g8:dp_g16);
        if(dp_g>1 || dp_pf){
            // WHAT-IF timing only (not built, numerics unchanged): K/G steps,
            // one 64-byte B register read per step, B[k..k+G-1][0..15];
            // with prefetch only the first step's read is exposed. The A term
            // and the per-MAC cost (buffered or not) are unchanged.
            u64 steps=K/dp_g,mac=16*(buffered?t_mac:t_mac_unbuf);
            sweep=(buffered?a_regs*t_rf:0)+(dp_pf?t_rf+steps*mac:steps*(t_rf+mac));
            ++*c_dp;
        }
        u64 busy=sweep+t_pipe+1+t_done;
        engine_start(f,busy);++*c_mma;
        sc_time free_at=sc_time_stamp()+x.cycles(x.functional?0:sweep),done_at=mtx_free;
        if(source_free)completions.push_back({free_at,sf_token});
        completions.push_back({done_at,token});
        completion_added.notify(SC_ZERO_TIME);
    }
    // The engine runs one command at a time, so its source-free and result
    // events are produced in time order: one persistent process signals them.
    std::deque<std::pair<sc_time,u64>> completions;sc_event completion_added,completions_drained;
    void start_completions(){
        sc_spawn([this](){for(;;){
            while(completions.empty())wait(completion_added);
            auto next=completions.front();
            if(next.first>sc_time_stamp()){wait(next.first-sc_time_stamp());continue;}
            completions.pop_front();complete_async(next.second,false);
            if(completions.empty())completions_drained.notify(SC_ZERO_TIME);
        }});
    }
    //--------------------------------------------------------------------------
    // Vector sources read from the same VRF bank (bank = register mod
    // vrf_banks with 64-aligned context bases) are read one per cycle.
    void bank_conflicts(Frame& f,const Inst& in){
        // Vector sources per tnca_decoder.v (o_va/vb/vc_en); a masked write
        // does not read its destination.
        unsigned srcs[3];unsigned n=0;auto add=[&](unsigned r){if(n<3)srcs[n++]=r;};
        const unsigned fn=in.func;
        switch(in.kind){
            case K::Vint:if(!(fn>=32 && fn<=63) && fn!=0 && fn!=20 && fn!=21 && fn!=26 && fn!=27 && fn!=64){add(in.a);add(in.b);}break;
            case K::Vfp:if(fn!=20 && fn!=21){add(in.a);add(in.b);if(fn==6 || fn==7)add(in.c);}break;
            case K::Shuffle:if(fn==3){add(in.a);add(in.b);}break;
            case K::Lsu:if(fn==40){add(in.b);add(in.d);}break;
            default:break;
        }
        u64 now=lnow(f);bool slot=vrf_pf && (!r_reads || now>last_issue+1);
        r_reads=false;
        if(n<2)return;
        unsigned count[8]={};unsigned worst=0;
        for(unsigned i=0;i<n;++i){unsigned bk=unsigned((f.vbase+srcs[i])%n_banks)&7;worst=std::max(worst,++count[bk]);}
        if(worst<2)return;
        // VRF_PREFETCH: D reads one cycle early when R is not using the
        // ports, so a second read of one bank costs nothing then.
        r_reads=true;u64 rounds=worst-1;if(slot)--rounds;
        if(rounds){u64 extra=rounds*t_bank;*c_bank+=extra;freeze(f,lt(f)+x.cycles(extra));}
    }
    void execute(Frame& f){
        const std::string role_key=key("role_"+std::to_string(f.role));
        for(;;){
            if(deadline!=SC_ZERO_TIME && lt(f)>deadline)throw DeadlineExpired{};
            const Inst& in=code.at(f.pc);
            if(in.kind==K::Illegal)throw ArchFault(trapcode::illegal,in.why+" at pc "+std::to_string(f.pc));
            if(in.kind==K::Refused)throw std::runtime_error("MODEL_REFUSED "+in.why+" (word "+std::to_string(in.word)+" at pc "+std::to_string(f.pc)+")");
            const unsigned d=in.d,a=in.a,b=in.b,c=in.c,func=in.func,mod=in.mod;u64 next=f.pc+8;auto begin=lt(f);
            auto rs=[&](unsigned r)->u64{hazard(f,f.sr[r]);return r?f.s[r]:0ULL;};
            auto ws=[&](unsigned r,u64 value,u64 latency){if(r){hazard(f,f.sr[r]);f.s[r]=value;f.sr[r]=lt(f)+x.cycles(latency);}};
            auto rv=[&](unsigned r){hazard(f,f.vr[r]);f.v[r]=f.physical->read(f.vbase+r,lt(f));};
            auto wp=[&](unsigned r,unsigned value,u64 latency){if(r){hazard(f,f.pr[r]);f.p[r]=value&65535;f.pr[r]=lt(f)+x.cycles(latency);}};
            unsigned mask=65535;
            if(pipe && f.fetch_ready>lt(f)){*c_branch+=cycles_of(f.fetch_ready-lt(f));go(f,f.fetch_ready);}
            if(in.major>=16){hazard(f,f.pr[in.pred]);mask=f.p[in.pred];if(in.invert)mask^=65535;}
            if(pipe){
                if(holds){sync(f);while(holds)wait(unhold);}
                if(t_bank)bank_conflicts(f,in);
                // An LSU completion uses the writeback port in place of an
                // issued instruction (tnca_core.v wb_steal).
                if(!steals.empty()){u64 now=lnow(f);
                    for(auto it=steals.begin();it!=steals.end();){
                        if(*it<=now){if(*it+1>=last_issue)issue.available=std::max(issue.available,lt(f))+x.cycles(t_lsu_wb);it=steals.erase(it);}
                        else ++it;
                    }}
            }
            // No cycle polling: issue and dependencies reserve their next event.
            issue_slot(f);
            if(pipe)last_issue=lnow(f);
            ++class_counts[unsigned(in.kind)];
            switch(in.kind){
            case K::Exit:drain_own(f);sync(f);++f.retired;++*c_instr;return;
            case K::Nop:case K::Yield:case K::Prefetch:break; // t.prefetch: a hint, dropped (ISA sec 9E.3)
            case K::Trap:throw ArchFault(trapcode::user,"TRAP_USER: code "+std::to_string((a<<16)|(b<<8)|c));
            case K::Special:ws(d,special(mod,f),t_scalar);break;
            case K::Salu:{
                u64 value=0;
                if(func>=64 && func<=66){u64 val=u64(in.imm)<<(16*(mod&3));value=func==64?val:func==66?~val:(rs(d)&~(0xffffULL<<(16*(mod&3))))|val;}
                else if(func==25)value=rs(c)?rs(a):rs(b);
                else {
                    u64 left=rs(a),right=0;unsigned af=func;
                    if(func>=32 && func<=63){right=zero_extended_immediate(func)?in.imm:in.simm;af=immediate_alu(func);}
                    else if(func!=0 && func!=20 && func!=21 && func!=22 && func!=23)right=rs(b);
                    value=alu(af,left,right,64,mod&15);
                }
                ws(d,value,(func==3 || func==4 || func==5 || func==34)?t_mul:t_scalar);break;}
            case K::Branch:{
                auto off24=std::int32_t((unsigned(a)<<24)|(unsigned(b)<<16)|(unsigned(c)<<8))>>8;
                if(func==0)next=f.pc+u64(std::int64_t(off24)*8);
                else if(func==1 || func==2){bool test=rs(a)!=0;if(test==(func==1))next=f.pc+in.simm*8;}
                else if(func==4 || func==9)next=rs(a)&~7ULL;
                else if(func==8){ws(d,f.pc+8,1);next=f.pc+u64(std::int64_t(off24)*8);}
                else throw std::runtime_error("uncovered control operation");
                if(pipe && next!=f.pc+8 && !x.functional)f.fetch_ready=lt(f)+x.cycles(1+t_branch);
                break;}
            case K::Lsu:{
                bool store=func==8 || func==24 || func==40,vector=func>=16,gather=func>=32;
                unsigned size=1U<<(mod&7);
                u64 base=rs(a);if(!gather)base+=in.simm;
                if(vector){rv(d);if(gather)rv(b);}else if(store)rs(d);
                // A scalar access is lane 0 of the mask (tnca_lsu.v).
                unsigned active=vector?mask:(mask&1);
                if(!active)break;
                struct Part {unsigned lane,offset,bytes,shift;};
                if(tm && x.functional){
                    std::array<u64,16> values{};
                    for(unsigned lane=0;lane<(vector?16U:1U);++lane){
                        if(!(active&(1<<lane)))continue;
                        u64 address=base+(vector?(gather?u64(std::int64_t(std::int32_t(f.v[b][lane])))*size:lane*size):0);
                        x.c.address(address,size);u64 value=vector?f.v[d][lane]:f.s[d];unsigned done=0;
                        while(done<size){
                            u64 ap=address+done;unsigned off=unsigned(ap&7),n=std::min(size-done,8-off);
                            u64 data=0;unsigned be=0;
                            for(unsigned byte=0;byte<n;++byte){be|=1U<<(off+byte);data|=((value>>((done+byte)*8))&255)<<((off+byte)*8);}
                            u64 got=tm->access(ap&~7ULL,store,data,store?be:255);
                            if(!store){u64 pm=n==8?UINT64_MAX:((1ULL<<(n*8))-1);values[lane]|=((got>>(off*8))&pm)<<(done*8);}
                            done+=n;
                        }
                    }
                    if(!store){
                        for(unsigned lane=0;lane<(vector?16U:1U);++lane)if(active&(1<<lane)){
                            u64 value=values[lane];if((mod&8) && size<8 && (value&(1ULL<<(size*8-1))))value|=UINT64_MAX<<(size*8);
                            if(vector)f.v[d][lane]=std::uint32_t(value);else ws(d,value,0);
                        }
                        if(vector)publish(f,d,active,0);
                    }
                    break;
                }
                if(tm){
                    // Non-blocking LSU: at most lsu_slots instructions in flight
                    // core-wide; beats leave the core port one per cycle with at
                    // most lsu_beat_slots words outstanding; a load's
                    // destination is scoreboarded until its last word returns.
                    u64 now=lnow(f),gnow=x.now();unsigned unit_index=0;
                    if(pipe){
                        // A busy unit holds the access in R: every context stalls.
                        u64 best=UINT64_MAX;
                        for(unsigned i=0;i<2;++i){u64 at=std::max({now,units[i].free,units[1-i].drained});if(at<best){best=at;unit_index=i;}}
                        if(best>now){*c_lsu_wait+=best-now;freeze(f,x.cycles(best));now=lnow(f);}
                    }else{
                    auto prune=[&](){lsu_inflight.erase(std::remove_if(lsu_inflight.begin(),lsu_inflight.end(),[&](u64 v){return v<=now;}),lsu_inflight.end());};
                    prune();
                    if(lsu_inflight.size()>=t_lsu_slots){u64 m=*std::min_element(lsu_inflight.begin(),lsu_inflight.end());
                        *c_lsu_wait+=m-now;ready(f,x.cycles(m));now=lnow(f);prune();}
                    }
                    gnow=x.now();
                    std::array<u64,16> values{};std::vector<u64> window;u64 t=pipe?now+1+t_lsu_setup:now,complete=now;
                    std::map<u64,std::pair<std::pair<u64,unsigned>,std::vector<Part>>> plan2;
                    for(unsigned lane=0;lane<(vector?16U:1U);++lane){
                        if(!(active&(1<<lane)))continue;
                        u64 address=base+(vector?(gather?u64(std::int64_t(std::int32_t(f.v[b][lane])))*size:lane*size):0);
                        x.c.address(address,size);u64 value=vector?f.v[d][lane]:f.s[d];unsigned done=0;
                        while(done<size){
                            u64 address_part=address+done;unsigned off=unsigned(address_part&7),n=std::min(size-done,8-off);
                            auto& beat=plan2[address_part&~7ULL];beat.second.push_back({lane,off,n,done*8});
                            for(unsigned byte=0;byte<n;++byte){
                                unsigned at=off+byte;beat.first.second|=1<<at;
                                beat.first.first=(beat.first.first&~(255ULL<<(at*8)))|(((value>>((done+byte)*8))&255)<<(at*8));
                            }done+=n;
                        }
                    }
                    // Group only addresses sharing one aligned hypothetical core
                    // beat. Each group still consumes a finite LSU slot, core
                    // issue slot, NoC capacity, home engine and outlet service.
                    const u64 width=x.c.get_or("tl_core_beat_bytes",8);
                    for(auto it=plan2.begin();it!=plan2.end();){
                        auto end=it;u64 region=it->first&~(width-1);
                        std::vector<TimelineMemory::WideWord> words;
                        do{words.push_back({end->first,end->second.first.first,store?end->second.first.second:255,0});++end;}
                        while(end!=plan2.end() && (end->first&~(width-1))==region);
                        while(window.size()>=t_beat_slots){auto m=std::min_element(window.begin(),window.end());t=std::max(t,*m);window.erase(m);}
                        u64 at=port.reserve(t,1,gnow);t=at+1;
                        u64 when=width==8?tm->word(id,words[0].address,store,words[0].data,words[0].be,at,words[0].result):tm->wide(id,words,store,at);
                        window.push_back(when);complete=std::max(complete,when);
                        unsigned wi=0;
                        for(;it!=end;++it,++wi)if(!store)for(auto& part:it->second.second){
                            u64 part_mask=part.bytes==8?UINT64_MAX:((1ULL<<(part.bytes*8))-1);
                            values[part.lane]|=((words[wi].result>>(part.offset*8))&part_mask)<<part.shift;
                        }
                    }
                    if(pipe){units[unit_index].drained=t;units[unit_index].free=complete+t_lsu_complete;
                        if(t_lsu_wb && !store)steals.push_back(complete+t_load-1);}
                    else lsu_inflight.push_back(complete);
                    role_lsu_done[f.role]=std::max(role_lsu_done[f.role],complete);
                    if(!store){
                        u64 latency=complete-now+t_load+(pipe && vector?t_vload_extra:0);
                        for(unsigned lane=0;lane<(vector?16U:1U);++lane)if(active&(1<<lane)){
                            u64 value=values[lane];if((mod&8) && size<8 && (value&(1ULL<<(size*8-1))))value|=UINT64_MAX<<(size*8);
                            if(vector)f.v[d][lane]=std::uint32_t(value);else ws(d,value,latency);
                        }
                        if(vector)publish(f,d,active,latency);
                    }
                    break;
                }
                auto wait_start=sc_time_stamp();lsu.acquire(x,lsu_key);*c_lsu_wait+=cycles_of(sc_time_stamp()-wait_start);
                struct Beat {u64 data=0,value=0;unsigned enables=0;std::vector<Part> parts;};
                std::map<u64,Beat> plan;
                for(unsigned lane=0;lane<(vector?16U:1U);++lane){
                    if(!(active&(1<<lane)))continue;
                    u64 address=base+(vector?(gather?u64(std::int64_t(std::int32_t(f.v[b][lane])))*size:lane*size):0);
                    x.c.address(address,size);u64 value=vector?f.v[d][lane]:f.s[d];unsigned done=0;
                    while(done<size){
                        u64 address_part=address+done;unsigned off=unsigned(address_part&7),n=std::min(size-done,8-off);
                        auto& beat=plan[address_part&~7ULL];beat.parts.push_back({lane,off,n,done*8});
                        for(unsigned byte=0;byte<n;++byte){
                            unsigned at=off+byte;beat.enables|=1<<at;
                            beat.data=(beat.data&~(255ULL<<(at*8)))|(((value>>((done+byte)*8))&255)<<(at*8));
                        }done+=n;
                    }
                }
                unsigned pending=unsigned(plan.size());sc_event beat_done;u64 tag=f.retired;
                for(auto& beat:plan){
                    auto* entry=&beat;
                    beats.submit([this,entry,store,tag,&pending,&beat_done](){entry->second.value=word(entry->first,store,entry->second.data,entry->second.enables,tag);if(--pending==0)beat_done.notify(SC_ZERO_TIME);});
                }
                while(pending)wait(beat_done);
                if(!store){
                    std::array<u64,16> values{};
                    for(auto& beat:plan)for(auto& part:beat.second.parts){
                        u64 part_mask=part.bytes==8?UINT64_MAX:((1ULL<<(part.bytes*8))-1);
                        values[part.lane]|=((beat.second.value>>(part.offset*8))&part_mask)<<part.shift;
                    }
                    for(unsigned lane=0;lane<(vector?16U:1U);++lane)if(active&(1<<lane)){
                        u64 value=values[lane];if((mod&8) && size<8 && (value&(1ULL<<(size*8-1))))value|=UINT64_MAX<<(size*8);
                        if(vector)f.v[d][lane]=std::uint32_t(value);else ws(d,value,t_load);
                    }
                }
                if(vector && !store)publish(f,d,active,t_load);
                lsu.release();break;}
            case K::Fence:{sync(f);auto t=sc_time_stamp();
                hold([&]{drain_own(f);if(pipe)ready(f,x.cycles(lsu_idle()));sync(f);});
                *c_fence_wait+=cycles_of(sc_time_stamp()-t);break;}
            case K::Vint:{
                bool cmp=func==24 || func==56,immediate=func>=32 && func<=63;
                if(!cmp)rv(d);
                if(func!=64 && func!=27 && func!=26)rv(a);
                if(!immediate && func!=0 && func!=20 && func!=21 && func!=26 && func!=27 && func!=64)rv(b);
                unsigned comparison=0;u64 scalar=func==26?rs(a):0;
                for(unsigned lane=0;lane<16;++lane)if(func==25 || (mask&(1<<lane))){
                    u64 value=0;
                    if(func==27)value=lane;else if(func==26)value=scalar;else if(func==64)value=in.simm;
                    else if(func==25)value=(mask&(1<<lane))?f.v[a][lane]:f.v[b][lane];
                    else {u64 right=immediate?(zero_extended_immediate(func)?in.imm:in.simm):f.v[b][lane];
                        value=alu(immediate?immediate_alu(func):func,f.v[a][lane],right,32,mod&15);}
                    if(cmp){if(value)comparison|=1<<lane;}else f.v[d][lane]=std::uint32_t(value);
                }
                if(cmp)wp(d,comparison,t_cmp);
                else publish(f,d,func==25?65535:mask,t_vint);
                break;}
            case K::Vfp:{
                rv(a);if(func!=24)rv(d);
                if(func!=20 && func!=21)rv(b);
                if(func==6 || func==7)rv(c);
                unsigned comparison=0;
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                    if(func==24){if(tnca_vector_ops::vfp_compare(mod,f.v[a][lane],f.v[b][lane]))comparison|=1<<lane;}
                    else f.v[d][lane]=tnca_vector_ops::vfp(func,mod,f.v[a][lane],f.v[b][lane],(func==6 || func==7)?f.v[c][lane]:0);
                }
                if(func==24)wp(d,comparison,pipe?t_fcmp:t_cmp);
                else publish(f,d,mask,t_vfp);
                break;}
            case K::Cvt:{
                rv(a);rv(d);
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=tnca_vector_ops::cvt(func,mod,f.v[a][lane]);
                publish(f,d,mask,t_cvt);break;}
            case K::Pred:{
                unsigned pa=0,pb=0;if(func!=19){ready(f,f.pr[a]);pa=f.p[a];}
                if(func==1 || func==2 || func==3 || func==5){ready(f,f.pr[b]);pb=f.p[b];}
                if(func>=16){
                    if(func==16)ws(d,pa!=0,t_pred);else if(func==17)ws(d,pa==65535,t_pred);
                    else if(func==18)ws(d,pa,t_pred);else if(func==19)wp(d,unsigned(rs(a)&65535),t_pred);
                    else if(func==20)ws(d,unsigned(__builtin_popcount(pa)),t_pred);else throw std::runtime_error("reserved predicate function");
                }else{
                    require(func<=5,"reserved predicate function");
                    unsigned value=func==0?pa:func==1?pa&pb:func==2?pa|pb:func==3?pa^pb:func==4?~pa:pa&~pb;
                    wp(d,value,t_pred);
                }
                break;}
            case K::Shuffle:{
                rv(a);rv(d);auto source=f.v[a];PhysicalRegisters::Vector indices{};unsigned shift=0;
                if(func==3){rv(b);indices=f.v[b];}else shift=unsigned(rs(b))&15;
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                    unsigned from=lane;
                    if(func==0)from=lane^shift;else if(func==1)from=lane>=shift?lane-shift:lane;
                    else if(func==2)from=lane+shift<16?lane+shift:lane;else if(func==3)from=indices[lane]&15;
                    else from=(lane+shift)&15;
                    f.v[d][lane]=source[from];
                }publish(f,d,mask,t_shuf);break;}
            case K::Extract:rv(a);ws(d,f.v[a][mod&15],pipe?t_shuf_s:t_shuf);break;
            case K::Insert:{rv(d);auto value=std::uint32_t(rs(a));if(mask&(1<<(mod&15)))f.v[d][mod&15]=value;publish(f,d,mask&(1<<(mod&15)),t_shuf);break;}
            case K::Reduce:{
                rv(a);
                std::uint32_t reduced=tnca_vector_ops::reduce(func,f.v[a],mask);
                u64 latency=func>=8?t_fred:t_red;
                if(mod&1){if(pipe)latency+=func>=8?t_fred_vec:t_red_vec;rv(d);f.v[d].fill(reduced);publish(f,d,65535,latency);}
                else ws(d,reduced,latency);
                break;}
            case K::Sfu:{
                rv(a);rv(d);
                if(!x.functional){
                auto waiting=lt(f);if(pipe)freeze(f,sfu_available);else ready(f,sfu_available);
                *c_sfu_queue+=cycles_of(lt(f)-waiting);
                sfu_available=lt(f)+x.cycles(t_sfu_occ);}
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=tnca_sfu::evaluate(func,f.v[a][lane]);
                publish(f,d,mask,t_sfu);break;}
            case K::Desc:{
                unsigned slot=sliced(f,d,4);u64 address=func==0?rs(a):0;sync(f);
                tensor.configure(func,slot,address,in.imm);break;}
            case K::Xfer:{
                u64 token=token_of(f,rs(c));sync(f);
                tensor.issue(sliced(f,a,4),sliced(f,d,4),token,1U<<(mod&7),func==8,mod&32,mask!=0,f.physical,f.vbase,f.role,
                    [&](unsigned rank){
                        arch(b+rank<=64,trapcode::descriptor,"DESCRIPTOR_FAULT: coordinates exceed scalar file");
                        std::array<u64,5> coordinates{};for(unsigned dim=0;dim<rank;++dim)coordinates[dim]=rs(b+dim);sync(f);return coordinates;
                    },[&](unsigned base,unsigned length){
                        arch(base<64 && length<=64-base,trapcode::descriptor,"DESCRIPTOR_FAULT: transfer exceeds role mapping");
                        for(unsigned r=base;r<base+length;++r){ready(f.vr[r]);}
                        f.physical->commit(sc_time_stamp());
                    });
                break;}
            case K::Barrier:{
                if(func!=3 && func!=4)sync(f);
                if(func==0)barriers[sliced(f,d,5)].reset();
                else if(func==1){unsigned index=sliced(f,a,5);ws(d,barriers[index].arm(index,b,c),t_bar);}
                else if(func==2){
                    // ISA sec 9A.5 with RPAIR's precision (tnca_core.v stall_arrive):
                    // own register writes landed, own engine commands finished.
                    auto t=sc_time_stamp();
                    hold([&]{drain_own(f);if(pipe)ready(f,x.cycles(lsu_idle()));sync(f);
                        while(tensor.outstanding[f.role])wait(tensor.role_done);
                        if(mtx_role==f.role)engine_wait();});
                    *c_arrive_wait+=cycles_of(sc_time_stamp()-t);
                    barriers[sliced(f,a,5)].arrive();
                }else if(func==3 || func==4){
                    u64 token=token_of(f,rs(a));unsigned index=unsigned(token&65535);sync(f);
                    if(func==3)ws(d,barriers[index].test(token),t_bar);
                    else {auto sleep=sc_time_stamp();barriers[index].sleep(token);*c_sleep+=cycles_of(sc_time_stamp()-sleep);}
                }else if(func==5)barriers[sliced(f,a,5)].invalidate();
                else throw std::runtime_error("reserved barrier function");
                break;}
            case K::MZero:case K::AZero:{
                // m.zero and a.zero are both the engine's zero sweep (cmd_op 0):
                // whole rows, no lane mask (tnca_core.v mtx_cmd_op).
                sync(f);engine_wait();acc[sliced(f,d,4)]={};engine_start(f,t_zero+t_done);break;}
            case K::Mma:mma(f,in);break;
            case K::ARow:{
                // a.vwrite (direct), a.mul.row / a.fma.row (one rounding, ftz=0).
                u64 row=rs(a)&15;std::uint32_t scale=0;
                if(func==2)scale=std::uint32_t(rs(b));else{rv(b);if(func==3)scale=std::uint32_t(rs(c));}
                sync(f);engine_wait();
                tnca_matrix_ops::row_op(acc[sliced(f,d,4)][row],func,scale,func==2?PhysicalRegisters::Vector{}:f.v[b],mask);
                engine_start(f,func==1?t_row+t_done:t_row+t_pipe+1+t_done);break;}
            case K::AVread:{
                u64 row=rs(b)&15;sync(f);engine_wait();rv(d);
#ifdef TNCA_MUTATION_AVREAD_ROW
                row=(row+1)&15; // test-only mutant (separate build): proves end-to-end checks see wrong results
#endif
                auto& r=acc[sliced(f,a,4)][row];
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=r[lane];
                publish(f,d,mask,t_acc);break;}
            default:throw std::runtime_error("MODEL_REFUSED uncovered instruction kind "+std::string(kind_name(in.kind)));
            }
            ++f.retired;++*c_instr;
            if(tracing)x.span("native instruction",id,begin,f.retired);
            f.pc=next;
        }
    }
    void record_fault(unsigned status,unsigned code,u64 pc,const std::string& why){
        if(!aborting){aborting=true;fault.status=status;fault.code=code;fault.pc=pc;fault.why=why;}
        // ISA sec 10: a trap wakes every waiter that could otherwise block forever.
        for(auto& barrier:barriers)if(!barrier.error)barrier.invalidate();
    }
    // Four persistent Role Context processes per core, started once: a
    // program wakes the contexts its mask names (no process per program).
    const Launch* current=nullptr;unsigned role_remaining=0,role_active=0;u64 role_retired=0;
    std::array<sc_event,4> role_start;sc_event role_done;
    void start_roles(){
        for(unsigned role=0;role<4;++role)sc_spawn([this,role](){for(;;){wait(role_start[role]);run_role(role);}});
    }
    void run_role(unsigned role){
        const Launch& l=*current;
        unsigned base=(l.pair?(role&~1U):role)*64;
        Frame f(vrf,base);f.pc=l.pc;f.arg=l.arg;f.px=l.px;f.py=l.py;f.pz=l.pz;f.nx=l.nx;f.ny=l.ny;f.nz=l.nz;
        f.role=role;f.role_count=role_active;f.sliced=l.slice;f.slice_k=l.slice_k;f.slice_id=l.pair?role>>1:role;
        ++*c_role_issued[role];
        try{execute(f);++*c_role_completed[role];}
        catch(const ArchFault& e){record_fault(STATUS_FAULT,e.code,f.pc,e.what());}
        catch(const DeadlineExpired&){record_fault(STATUS_TIMEOUT,0,f.pc,"device deadline expired");}
        role_retired+=f.retired;
        --role_remaining;role_done.notify(SC_ZERO_TIME);
    }
    Outcome program(const Launch& l){
        require(l.roles>0 && l.roles<16,"invalid role mask");
        require(role_remaining==0,"program started on a core with contexts still running");
        tensor.reset();
        for(auto& barrier:barriers)barrier.reset();
        aborting=false;fault=Outcome{};
        current=&l;role_active=unsigned(__builtin_popcount(l.roles));role_remaining=role_active;role_retired=0;
        ++*c_programs_issued;
        for(unsigned role=0;role<4;++role)if(l.roles&(1<<role))role_start[role].notify();
        while(role_remaining)wait(role_done);
        u64 retired=role_retired;
        // ISA sec 7: complete only when every role has exited and every
        // asynchronous command has completed.
        tensor.drain();engine_wait();
        while(!completions.empty())wait(completions_drained);
        if(!aborting)++*c_programs_completed;
        Outcome out=aborting?fault:Outcome{};out.retired=retired;
        return out;
    }
    void report_classes(){for(unsigned k=0;k<unsigned(K::Count);++k)if(class_counts[k])x.counts[key("class_")+kind_name(K(k))]=class_counts[k];}
};

//==============================================================================
// The chip: cores, NoC, NUCA homes, HBM; grid dispatch and host access.
//==============================================================================
struct Chip : sc_module {
    SC_HAS_PROCESS(Chip);
    Context& x;HBM& hbm;std::vector<std::unique_ptr<Slice>>& caches;CodeCache code;
    std::vector<std::unique_ptr<Core>> cores;
    // Host-driven command interface: one pending command, one completion.
    struct Command {bool grid=false;Launch launch;u64 stride=0,core_mask=1,deadline=0;unsigned gx=1,gy=1,gz=1,group=1;};
    std::unique_ptr<Command> pending;bool command_done=false,flush_request=false,shutdown=false;
    Outcome result;u64 start_cycle=0,end_cycle=0,launches=0;sc_event wake;
    u64 *c_grid_launches,*c_single_launches,*c_host_cycles;
    Chip(sc_module_name n,Context& ctx,HBM& mem,Network& net,std::vector<std::unique_ptr<Slice>>& s):
        sc_module(n),x(ctx),hbm(mem),caches(s){
        code.read_word=[this](u64 pc){u64 v=0;unsigned char bytes[8];host_read(pc,bytes,8);for(unsigned b=0;b<8;++b)v|=u64(bytes[b])<<(8*b);return v;};
        for(unsigned i=0;i<x.c.cores();++i){
            cores.emplace_back(new Core(("core"+std::to_string(i)).c_str(),x,i,code));
            cores.back()->memory.bind(net.target);cores.back()->tensor.memory.bind(net.target);
        }
        c_grid_launches=&x.counts["chip_grid_launches"];c_single_launches=&x.counts["chip_single_launches"];
        c_host_cycles=&x.counts["chip_host_gap_cycles"];
        SC_THREAD(run);
    }
    // Functionally coherent host access in zero modeled time: the device is
    // idle, so a line is either valid in its home slice (newest copy) or in
    // HBM. Writes update both copies. Nothing is flushed or perturbed.
    void host_access(u64 address,unsigned char* data,u64 n,bool write){
        if(!n)return;
        if(const char* w=std::getenv("TLM_DEBUG_WATCH")){u64 lo=std::strtoull(w,nullptr,0);const char* c=std::strchr(w,',');u64 hi=lo+(c?std::strtoull(c+1,nullptr,0):0);
            if(address+n>lo && address<hi)std::fprintf(stderr,"WATCH host t %llu %s a %llx n %llu\n",(unsigned long long)x.now(),write?"W":"R",(unsigned long long)address,(unsigned long long)n);}
        u64 base=x.c.get("memory_base"),bytes=x.c.get("memory_bytes");
        require(address>=base && n<=bytes && address-base<=bytes-n,"host access outside model memory window");
        require(!(address<0x7fff1000ULL && address+n>0x7fff0000ULL),"TMU MMIO is not ordinary model memory");
        if(write)code.invalidate(address,n);
        u64 done=0;
        while(done<n){
            u64 a=address+done,line=a&~63ULL;unsigned off=unsigned(a&63),len=unsigned(std::min<u64>(64-off,n-done));
            CacheLine* cached=caches[x.c.home(line)]->find(line);
            if(write){
                auto& backing=hbm.memory[line];std::memcpy(backing.data()+off,data+done,len);
                if(cached)std::memcpy(cached->data.data()+off,data+done,len);
            }else if(cached)std::memcpy(data+done,cached->data.data()+off,len);
            else if(const Line* backing=hbm.memory.find(line))std::memcpy(data+done,backing->data()+off,len);
            else std::memset(data+done,0,len);
            done+=len;
        }
    }
    void host_write(u64 a,const unsigned char* p,u64 n){host_access(a,const_cast<unsigned char*>(p),n,true);}
    void host_read(u64 a,unsigned char* p,u64 n){host_access(a,p,n,false);}
    TimelineMemory* tm=nullptr;
    void flush(){
        if(tm && x.functional)return; // untimed: nothing is cached on the timeline path
        if(tm){u64 now=x.now(),done=tm->flush(now);if(done>now)wait(x.cycles(done-now));return;}
        unsigned remaining=unsigned(caches.size());sc_event done;
        for(auto& c:caches){auto* ptr=c.get();sc_spawn([&,ptr](){ptr->flush();--remaining;done.notify(SC_ZERO_TIME);});}
        while(remaining)wait(done);
        for(auto& slot:hbm.slots)require(slot->used==0,"completion visibility still has outstanding writes");
    }
    // tnca_grid_dispatch.v: groups of `group` consecutive x along x, then y,
    // then z; each idle active core takes the next group with its own
    // argument block at arg_base + core * stride. A fault stops dispatch.
    Outcome dispatch(const Command& cmd){
        u64 groups_x=(u64(cmd.gx)+cmd.group-1)/cmd.group,total=groups_x*cmd.gy*cmd.gz;
        unsigned x0=0,y0=0,z0=0;u64 issued=0;Outcome first;bool faulted=false;u64 retired=0;
        unsigned live=0;sc_event done;
        for(auto& core:cores){
            if(!((cmd.core_mask>>core->id)&1))continue;
            ++live;auto* ptr=core.get();
            sc_spawn([&,ptr](){
                while(issued<total && !faulted){
                    Launch l=cmd.launch;l.px=x0;l.py=y0;l.pz=z0;l.nx=cmd.gx;l.ny=cmd.gy;l.nz=cmd.gz;l.arg=cmd.launch.arg+ptr->id*cmd.stride;
                    ++issued;
                    if(x0+cmd.group>=cmd.gx){x0=0;if(y0+1==cmd.gy){y0=0;++z0;}else ++y0;}else x0+=cmd.group;
                    if(!x.functional)wait(x.cycles(x.c.get("dispatch_cycles")+ptr->t_launch));
                    Outcome o=ptr->program(l);retired+=o.retired;
                    if(o.status!=STATUS_OK && !faulted){faulted=true;first=o;}
                }
                --live;done.notify(SC_ZERO_TIME);
            });
        }
        require(live>0 || total==0,"no active core for the grid");
        while(live)wait(done);
        first.retired=retired;return first;
    }
    // Server mode: host transfers since the previous launch and the fixed
    // per-launch host overhead elapse (device idle) before the next launch.
    bool serving=false;u64 host_debt=0;
    void host_charge(u64 bytes){
        if(!serving || x.functional)return;
        host_debt+=x.c.get("host_transfer_cycles")+(bytes*x.c.get("host_transfer_cycles_per_kib")+1023)/1024;
    }
    Outcome run_command(const Command& cmd){
        if(serving && x.functional)host_debt=0;
        else if(serving){u64 d=host_debt+x.c.get("host_launch_cycles");host_debt=0;if(d)wait(x.cycles(d));*c_host_cycles+=d;}
        for(auto& core:cores)core->deadline=cmd.deadline?sc_time_stamp()+x.cycles(cmd.deadline):SC_ZERO_TIME;
        start_cycle=x.now();Outcome out;
        if(cmd.grid){++*c_grid_launches;out=dispatch(cmd);}
        else{
            ++*c_single_launches;if(!x.functional)wait(x.cycles(cores[0]->t_launch));
            Launch l=cmd.launch;out=cores[0]->program(l);
        }
        end_cycle=x.now();++launches;
        // Post-hoc check as well: a launch that finished past its deadline is
        // what a device with the same deadline would have aborted.
        if(out.status==STATUS_OK && cmd.deadline && end_cycle-start_cycle>cmd.deadline){out.status=STATUS_TIMEOUT;out.why="device deadline expired";}
        return out;
    }
    // Host mode: the host loop (sc_main) sets `pending`, resumes simulation,
    // and gets control back through sc_pause once the command is complete.
    void run(){
        for(;;){
            while(!pending && !flush_request){sc_pause();wait(wake);}
            if(flush_request){flush();flush_request=false;command_done=true;continue;}
            auto cmd=std::move(pending);
            result=run_command(*cmd);
            command_done=true;
        }
    }
    void report(){for(auto& core:cores)core->report_classes();}
};

//==============================================================================
// Front ends: classifier, command script (validation) and host server.
//==============================================================================
static void classify(){
    std::string word;
    while(std::cin>>word){
        Inst in=decode(integer("0x"+word));
        bool ok=in.kind!=K::Refused && in.kind!=K::Illegal;
        std::cout<<word<<'\t'<<(ok?"implemented":in.kind==K::Illegal?"illegal":"refused")<<'\t'<<timing_class(in)<<'\t'<<kind_name(in.kind)<<' '<<in.why<<'\n';
    }
}
struct Model {
    Config cfg;Context x;HBM hbm;Network net;std::vector<std::unique_ptr<Slice>> caches;
    std::vector<std::unique_ptr<SocketMonitor>> probes;std::unique_ptr<Chip> chip;
    Model(const char* config,const std::string& out):cfg(config,std::getenv("TNCA_DRAM_MAP")!=nullptr),x(cfg,out),hbm("hbm",x),net("noc",x){
        for(unsigned i=0;i<cfg.cores();++i){
            caches.emplace_back(new Slice(("home"+std::to_string(i)).c_str(),x,i));
            auto home="home_"+std::to_string(i),mem="hbm_"+std::to_string(i);
            probes.emplace_back(new SocketMonitor(home.c_str(),x,home,500+i));
            net.homes[i].bind(probes.back()->target);probes.back()->downstream.bind(caches.back()->target);
            probes.emplace_back(new SocketMonitor(mem.c_str(),x,mem,600+i));
            caches.back()->memory.bind(probes.back()->target);probes.back()->downstream.bind(hbm.target);
        }
        chip.reset(new Chip("chip",x,hbm,net,caches));
        if(cfg.get("memory_model")==1){
            tmem.reset(new TimelineMemory(cfg,x,hbm.memory));chip->tm=tmem.get();
            for(auto& core:chip->cores){core->tm=tmem.get();core->tensor.tm=tmem.get();core->tensor.port=&core->port;core->tensor.rtl=cfg.get("core_pipeline")!=0;}
        }
        select_dram_path();
    }
    std::unique_ptr<TimelineMemory> tmem;
    // DRAM path (include/tnca_dram.hpp): fast, or slow = Ramulator2 by a
    // per-command fixed point of forked child passes.
    std::unique_ptr<SlowDram> slow;std::string dram_path_name="fast",dram_config,dram_config_sha256,dram_commit;
    std::unique_ptr<DramAddressMap> physical_map;std::string dram_map_path,dram_map_sha256;
    u64 dram_fixed=0,dram_ratio=10,dram_max_passes=3,dram_tol_ppm=5000,dram_commands=0;
    void select_dram_path(){
        u64 path=cfg.get("dram_path");
        if(const char* e=std::getenv("TNCA_DRAM_PATH")){
            std::string v=e;require(v=="fast" || v=="slow","TNCA_DRAM_PATH must be fast or slow");path=v=="slow";
        }
        require(path<=1,"dram_path is 0 (fast) or 1 (slow)");
        if(const char* map=std::getenv("TNCA_DRAM_MAP")){
            require(bool(tmem),"DRAM address map requires timeline memory");dram_map_path=map;
            physical_map.reset(new DramAddressMap(map,unsigned(cfg.get("pc_count")),cfg.get("pc_bytes")));
            // Replaces Config's logical PC check: the model window (local
            // region plus AXI addresses) must end at or below the mapped BO top.
            require(cfg.get("memory_base")+cfg.get("memory_bytes")<=physical_map->local_limit+physical_map->bo_size,
                "DRAM address map: model window exceeds the mapped AXI BO");
            tmem->physical_map=physical_map.get();
#ifdef TNCA_HAVE_RAMULATOR2
            dram_map_sha256=tnca_dram::sha256_file(map);
#endif
        }
        if(!path)return;
        require(cfg.get("memory_model")==1,"the slow DRAM path needs the timeline memory (memory_model=1)");
#ifndef TNCA_HAVE_RAMULATOR2
        throw std::runtime_error("MODEL_REFUSED slow DRAM path: this binary is built without Ramulator2 (-DRAMULATOR2_DIR)");
#else
        dram_path_name="slow";dram_commit=TNCA_RAMULATOR2_COMMIT;
        require(cfg.get("pc_bytes")==268435456,"slow DRAM path: HBM2 bridge requires 256 MiB physical pseudo channels");
        const char* cfgenv=std::getenv("TNCA_RAMULATOR2_CONFIG");dram_config=cfgenv?cfgenv:TNCA_RAMULATOR2_CONFIG;
        dram_config_sha256=tnca_dram::sha256_file(dram_config);require(!dram_config_sha256.empty(),"cannot read Ramulator2 config "+dram_config);
        auto t=tnca_dram::timing(dram_config);require(t.error.empty(),"Ramulator2 config: "+t.error);
        dram_fixed=cfg.get("dram_fixed_cycles");dram_ratio=cfg.get("dram_clock_ratio");
        dram_max_passes=cfg.get("dram_max_passes");dram_tol_ppm=cfg.get("dram_tolerance_ppm");
        require(dram_ratio>0 && dram_max_passes>=1,"slow DRAM path: dram_clock_ratio and dram_max_passes must be positive");
        require(dram_tol_ppm>0 && dram_tol_ppm<=1000000,"slow DRAM path: fixed-point tolerance must be in (0, 1000000] ppm");
        slow.reset(new SlowDram);slow->beats_extra=cfg.get("tl_outlet_beat_cycles");
        // Until a replay exists: the idle row-miss read (tRCD + RL) and write (tRCD + WL + BL).
        slow->fallback[0]=std::uint32_t(dram_fixed+(u64(t.nRCDRD+t.read_latency)+dram_ratio-1)/dram_ratio);
        slow->fallback[1]=std::uint32_t(dram_fixed+(u64(t.nRCDWR+t.nCWL+t.nBL)+dram_ratio-1)/dram_ratio);
        tmem->dram=slow.get();
#endif
    }
    // Run one command inside the simulation; false if the kernel starved.
    bool run_once(const Chip::Command* cmd,bool flush){
        chip->command_done=false;
        if(flush)chip->flush_request=true;else chip->pending.reset(new Chip::Command(*cmd));
        chip->wake.notify(SC_ZERO_TIME);
        sc_start();
        return chip->command_done;
    }
    bool execute(std::unique_ptr<Chip::Command> cmd,bool flush=false){
#ifdef TNCA_HAVE_RAMULATOR2
        if(slow && !x.functional)return execute_slow(cmd.get(),flush);
#endif
        return run_once(cmd.get(),flush);
    }
#ifdef TNCA_HAVE_RAMULATOR2
    // A child pass: the command runs on a copy-on-write snapshot of this
    // process with the current latency table, logging every DRAM request;
    // the trailer holds its makespan. It touches no descriptor of the parent
    // (host channel, trace, logs) and leaves with _exit.
    [[noreturn]] void child_pass(const Chip::Command* cmd,bool flush,const std::string& path){
        int code=6;
        try{
            ::close_range(3,~0U,0);
            int nul=::open("/dev/null",O_WRONLY);if(nul>=0){::dup2(nul,1);::dup2(nul,2);if(nul>2)::close(nul);}
            x.monitor_mode="off";
            FILE* f=std::fopen(path.c_str(),"wb");if(!f)::_exit(5);
            slow->log=f;slow->begin_command();
            u64 t0=x.now();bool ok=run_once(cmd,flush);
            u64 makespan=flush?x.now()-t0:chip->end_cycle-chip->start_cycle;
            SlowDram::Record trailer[2]={{UINT64_MAX,makespan},{UINT64_MAX-1,ok?1ULL:0ULL}};
            if(std::fwrite(trailer,sizeof trailer[0],2,f)!=2 || std::fflush(f)!=0)::_exit(7);
            code=ok?0:3;
        }catch(...){code=6;}
        ::_exit(code);
    }
    struct PassResult {bool ok=false,replayed=false;u64 makespan=0,requests=0;double child_s=0,replay_s=0;tnca_dram::Result replay;
        std::vector<u64> keys,arrival;std::vector<std::uint16_t> lat;std::uint32_t mean[2]={0,0};double mean_all=0;};
    PassResult pass(const Chip::Command* cmd,bool flush){
        PassResult out;
        std::string path=x.directory+"/dram_pass_"+std::to_string(::getpid())+".bin";
        auto t0=std::chrono::steady_clock::now();
        std::cout.flush();std::cerr.flush();
        const pid_t owner=::getpid();
        pid_t pid=::fork();require(pid>=0,"slow DRAM path: fork failed");
        if(pid==0){
            // fork clears the parent's death signal. Bind this pass to its
            // actual owner again, including death before prctl is armed.
            if(::prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || ::getppid()!=owner)::_exit(8);
            child_pass(cmd,flush,path);
        }
        int status=0;while(::waitpid(pid,&status,0)<0 && errno==EINTR){}
        auto t1=std::chrono::steady_clock::now();out.child_s=std::chrono::duration<double>(t1-t0).count();
        // The log, read into two arrays (16 bytes per request).
        u64 trailer[2][2]={{0,0},{0,0}};bool good=false;
        if(FILE* f=std::fopen(path.c_str(),"rb")){
            std::fseek(f,0,SEEK_END);long size=std::ftell(f);std::fseek(f,0,SEEK_SET);
            size_t total=size>0?size_t(size)/sizeof(SlowDram::Record):0;
            if(total>=2){
                size_t n=total-2;out.keys.resize(n);out.arrival.resize(n);
                std::vector<SlowDram::Record> chunk(1<<16);size_t at=0;good=true;
                while(at<n && good){
                    size_t k=std::min(chunk.size(),n-at);
                    if(std::fread(chunk.data(),sizeof(SlowDram::Record),k,f)!=k){good=false;break;}
                    for(size_t i=0;i<k;++i){out.keys[at+i]=chunk[i].key;out.arrival[at+i]=chunk[i].arrival;}
                    at+=k;
                }
                if(good && std::fread(trailer,sizeof(SlowDram::Record),2,f)!=2)good=false;
            }
            std::fclose(f);
        }
        std::remove(path.c_str());
        if(!good || !WIFEXITED(status) || trailer[0][0]!=UINT64_MAX || trailer[1][0]!=UINT64_MAX-1){
            std::vector<u64>().swap(out.keys);std::vector<u64>().swap(out.arrival);return out;}
        out.makespan=trailer[0][1];out.ok=WEXITSTATUS(status)==0 && trailer[1][1]==1;out.requests=out.keys.size();
        return out;
    }
    // Replay a pass's request log through Ramulator2 and build its table.
    struct DecodeContext {const u64* keys;u64 base,pcb;const std::vector<std::int32_t>* dense;const DramAddressMap* map;};
    static void decode_key(const void* c,size_t i,std::uint32_t& pc,std::uint64_t& offset,bool& write){
        auto& d=*static_cast<const DecodeContext*>(c);u64 k=d.keys[i],address=SlowDram::address_of(k);
        if(d.map){auto a=d.map->translate(address);pc=std::uint32_t((*d.dense)[a.pc]);offset=a.offset;}
        else{u64 a=address-d.base;pc=std::uint32_t((*d.dense)[size_t(a/d.pcb)]);offset=a%d.pcb;}
        write=SlowDram::write_of(k);
    }
    void replay(PassResult& out){
        auto t1=std::chrono::steady_clock::now();out.replayed=true;
        // Dense pseudo channels, DRAM-clock arrivals.
        u64 base=cfg.get("memory_base"),pcb=cfg.get("pc_bytes");
        std::vector<std::int32_t> dense;std::int32_t n_pc=0;
        for(u64 k:out.keys){u64 a=SlowDram::address_of(k);size_t pc=physical_map?physical_map->translate(a).pc:size_t((a-base)/pcb);
            if(pc>=dense.size())dense.resize(pc+1,-1);
            if(dense[pc]<0)dense[pc]=n_pc++;}
        for(auto& a:out.arrival)a*=dram_ratio;
        DecodeContext ctx{out.keys.data(),base,pcb,&dense,physical_map.get()};
        tnca_dram::Batch b;b.n=out.keys.size();b.arrival=out.arrival.data();b.context=&ctx;b.decode=decode_key;
        std::vector<std::uint32_t> latency;
        out.replay=tnca_dram::replay(dram_config,unsigned(n_pc),b,latency);
        require(out.replay.error.empty(),"Ramulator2 replay: "+out.replay.error);
        std::vector<u64>().swap(out.arrival);
        out.replay_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count();
        out.lat.resize(latency.size());
        double sum[2]={0,0};u64 n[2]={0,0};
        for(size_t i=0;i<latency.size();++i){
            u64 l=dram_fixed+(u64(latency[i])+dram_ratio-1)/dram_ratio;
            out.lat[i]=SlowDram::latency_value(l);
            unsigned w=SlowDram::write_of(out.keys[i])?1:0;sum[w]+=double(l);++n[w];
        }
        for(unsigned w=0;w<2;++w)out.mean[w]=n[w]?std::uint32_t(sum[w]/double(n[w])+0.5):slow->fallback[w];
        out.mean_all=(n[0]+n[1])?(sum[0]+sum[1])/double(n[0]+n[1]):0;
    }
    std::ofstream fixed_point_log;
    bool execute_slow(const Chip::Command* cmd,bool flush){
        ++dram_commands;
        std::vector<u64> makespans;double change=-1,child_total=0,replay_total=0;unsigned passes=0;
        bool damped=false,failed=false,converged=false;PassResult last;SlowDram::Stability stability;
        // Makespan agreement alone can hide changed streams or latencies.
        // Replay every traffic-bearing pass, including the last one, and
        // compare its request/latency vector with the table actually used.
        unsigned replays=0;std::vector<double> means{(double(slow->fallback[0])+double(slow->fallback[1]))/2}; // before replay 1: the fallback
        for(unsigned k=1;k<=dram_max_passes;++k){
            PassResult r=pass(cmd,flush);
            if(!r.ok){failed=true;++x.counts["dram_child_failures"];break;}
            ++passes;makespans.push_back(r.makespan);child_total+=r.child_s;
            if(passes>=2)change=double(r.makespan>makespans[passes-2]?r.makespan-makespans[passes-2]:makespans[passes-2]-r.makespan)/double(std::max<u64>(r.makespan,1));
            if(r.requests==0){change=0;converged=true;last=std::move(r);break;}
            replay(r);++replays;
            replay_total+=r.replay_s;
            stability=slow->stability(r.keys,r.lat,dram_tol_ppm);
            converged=passes>=2 && change*1e6<double(dram_tol_ppm) && stability.within_tolerance;
            // Retain the installed table, so the real run repeats this
            // observed child, whether validated or explicitly inconclusive.
            if(converged || k==dram_max_passes){last=std::move(r);break;}
            // Damping: when the mean latency reverses direction between
            // replays (an oscillating iteration), the new table is averaged
            // with the previous one, key by key.
            double mean=r.mean_all;
            if(replays>=2 && (mean-means.back())*(means.back()-means[means.size()-2])<0){
                damped=true;++x.counts["dram_slow_damped"];
                for(size_t i=0;i<r.keys.size();++i){
                    size_t pos;if(slow->lookup(r.keys[i],pos))r.lat[i]=std::uint16_t((u64(r.lat[i])+slow->lat[pos]+1)/2);
                }
            }
            means.push_back(mean);
            slow->install(std::move(r.keys),std::move(r.lat));slow->fallback[0]=r.mean[0];slow->fallback[1]=r.mean[1];
            last=std::move(r);
        }
        if(failed){
            if(!fixed_point_log.is_open())fixed_point_log.open(x.directory+"/dram_fixed_point.jsonl",std::ios::app);
            fixed_point_log<<"{\"command\":"<<dram_commands<<",\"child_failed\":true,\"converged\":false,\"prediction_status\":\"refused-child-pass\"}\n";
            fixed_point_log.flush();
            throw std::runtime_error("slow DRAM path: child pass failed; prediction unavailable");
        }
        slow->log=nullptr;slow->begin_command();
        auto w0=std::chrono::steady_clock::now();
        u64 t0=x.now();bool ok=run_once(cmd,flush);
        u64 makespan=flush?x.now()-t0:chip->end_cycle-chip->start_cycle;
        bool reproduced=ok && !makespans.empty() && makespan==makespans.back() && slow->requests==last.requests
            && (slow->requests==0 || slow->matched==slow->requests);
        converged=converged && reproduced;
        double real_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count();
        tnca_dram::PcStats tot;
        for(auto& p:last.replay.pcs){tot.reads+=p.reads;tot.writes+=p.writes;tot.row_hits+=p.row_hits;tot.row_misses+=p.row_misses;
            tot.accepted_bursts+=p.accepted_bursts;tot.completed_bursts+=p.completed_bursts;tot.queue_refusals+=p.queue_refusals;
            tot.peak_rw_queued_bursts=std::max(tot.peak_rw_queued_bursts,p.peak_rw_queued_bursts);
            tot.row_conflicts+=p.row_conflicts;tot.read_row_hits+=p.read_row_hits;tot.read_row_misses+=p.read_row_misses;
            tot.read_row_conflicts+=p.read_row_conflicts;tot.write_row_hits+=p.write_row_hits;tot.write_row_misses+=p.write_row_misses;
            tot.write_row_conflicts+=p.write_row_conflicts;tot.refreshes+=p.refreshes;}
        auto& c=x.counts;
        ++c["dram_slow_commands"];++c["dram_slow_fixed_point_checked"];
        c["dram_slow_child_passes"]+=passes;c["dram_slow_replays"]+=replays;if(!converged)++c["dram_slow_unconverged"];
        c["dram_slow_requests"]+=slow->requests;c["dram_slow_unmatched"]+=slow->requests-slow->matched;
        c["dram_replay_requests"]+=last.requests;c["dram_row_hits"]+=tot.row_hits;c["dram_row_misses"]+=tot.row_misses;
        c["dram_replay_accepted_bursts"]+=tot.accepted_bursts;c["dram_replay_completed_bursts"]+=tot.completed_bursts;
        c["dram_replay_queue_refusals"]+=tot.queue_refusals;
        c["dram_replay_peak_rw_queued_bursts_per_pc"]=std::max(c["dram_replay_peak_rw_queued_bursts_per_pc"],tot.peak_rw_queued_bursts);
        c["dram_row_conflicts"]+=tot.row_conflicts;c["dram_read_row_conflicts"]+=tot.read_row_conflicts;
        c["dram_write_row_conflicts"]+=tot.write_row_conflicts;c["dram_refreshes"]+=tot.refreshes;
        c["dram_replay_ticks"]+=last.replay.ticks;c["dram_replay_skipped"]+=last.replay.skipped;
        c["dram_pseudo_channels_max"]=std::max<u64>(c["dram_pseudo_channels_max"],last.replay.pcs.size());
        if(!fixed_point_log.is_open())fixed_point_log.open(x.directory+"/dram_fixed_point.jsonl",std::ios::app);
        fixed_point_log<<"{\"command\":"<<dram_commands<<",\"kind\":\""<<(flush?"flush":cmd->grid?"grid":"program")
            <<"\",\"passes\":"<<passes<<",\"replays\":"<<replays<<",\"child_failed\":"<<(failed?"true":"false")<<",\"converged\":"<<(converged?"true":"false")
            <<",\"prediction_status\":\""<<(converged?"uncalibrated-fixed-point-checked":"inconclusive-fixed-point")
            <<"\",\"same_requests\":"<<(stability.same_requests?"true":"false")<<",\"latencies_stable\":"<<(stability.within_tolerance?"true":"false")
            <<",\"latency_max_delta\":"<<stability.max_delta<<",\"unmatched_replay_keys\":"<<stability.missing<<",\"reproduced_last_pass\":"<<(reproduced?"true":"false")
            <<",\"damped\":"<<(damped?"true":"false")<<",\"last_rel_change\":"<<change<<",\"pass_makespans\":[";
        for(size_t i=0;i<makespans.size();++i)fixed_point_log<<(i?",":"")<<makespans[i];
        fixed_point_log<<"],\"makespan\":"<<makespan<<",\"requests\":"<<slow->requests<<",\"unmatched\":"<<slow->requests-slow->matched
            <<",\"pseudo_channels\":"<<last.replay.pcs.size()<<",\"row_hits\":"<<tot.row_hits<<",\"row_misses\":"<<tot.row_misses
            <<",\"row_conflicts\":"<<tot.row_conflicts<<",\"refreshes\":"<<tot.refreshes<<",\"mean_read_latency\":"<<slow->fallback[0]
            <<",\"mean_write_latency\":"<<slow->fallback[1]<<",\"replay_ticks\":"<<last.replay.ticks<<",\"replay_skipped\":"<<last.replay.skipped
            <<",\"replay_accepted_bursts\":"<<tot.accepted_bursts<<",\"replay_completed_bursts\":"<<tot.completed_bursts
            <<",\"replay_queue_refusals\":"<<tot.queue_refusals<<",\"replay_peak_rw_queued_bursts_per_pc\":"<<tot.peak_rw_queued_bursts
            <<",\"child_s\":"<<child_total<<",\"replay_s\":"<<replay_total<<",\"real_s\":"<<real_s<<"}\n";
        fixed_point_log.flush();
        return ok;
    }
#endif
    void counters_json(std::ostream& o){
        chip->report();
        o<<"{";bool first=true;for(auto& c:x.counts){if(!first)o<<',';first=false;o<<'"'<<c.first<<"\":"<<c.second;}o<<"}";
    }
    void summary(const std::string& path,const std::string& status,u64 checked){
        std::ofstream result(path);
        auto count=[&](const char* key){auto it=x.counts.find(key);return it==x.counts.end()?u64(0):it->second;};
        const char* prediction_status="uncalibrated";
        if(dram_path_name=="slow")prediction_status=(count("dram_slow_commands")>0 &&
            count("dram_slow_fixed_point_checked")==count("dram_slow_commands") &&
            !count("dram_slow_unconverged") && !count("dram_slow_unmatched") && !count("dram_child_failures"))?
            "uncalibrated-fixed-point-checked":"inconclusive-fixed-point";
        result<<"{\"schema\":\"tnca-systemc-result/1\",\"status\":\""<<status<<"\",\"dram_path\":\""<<dram_path_name<<"\",\"ramulator2_commit\":\""<<dram_commit
            <<"\",\"ramulator2_config\":\""<<dram_config<<"\",\"ramulator2_config_sha256\":\""<<dram_config_sha256<<"\",\"execution\":\"native-isa-event-driven\",\"coverage\":\"partial-ISA\",\"performance_gate\":\"not-measured\",\"calibration\":\"not-measured\",\"visible_cycles\":"<<x.now()<<",\"launches\":"<<chip->launches<<",\"checked_bytes\":"<<checked<<",\"counters\":";
        counters_json(result);result<<",\"prediction_status\":\""<<prediction_status
            <<"\",\"address_map_domain\":\""<<(physical_map?"axi-bo-physical-pc":"synthetic-logical")
            <<"\",\"address_map_sha256\":\""<<dram_map_sha256<<"\",\"bank_row_mapping\":\"unverified\"}\n";
    }
};
static Chip::Command grid_command(u64 pc,u64 arg,unsigned gx,unsigned gy,unsigned gz,u64 mask,u64 cores){
    require(gx && gy && gz && u64(gx)*gy<=16777216 && u64(gx)*gy*gz<=16777216,"invalid grid");
    Chip::Command c;c.grid=true;c.launch=Launch::from_mask(mask);c.launch.pc=pc;c.launch.arg=arg;c.gx=gx;c.gy=gy;c.gz=gz;c.core_mask=cores;
    return c;
}
static int script(const char* config,const char* commands,const std::string& out){
    Model m(config,out);std::ifstream input(commands);require(bool(input),"cannot open chip script");
    std::string line;bool host_owned=true;u64 checked=0;u64 all_cores=(1ULL<<m.cfg.cores())-1;
    auto launch=[&](Chip::Command c){
        require(m.execute(std::unique_ptr<Chip::Command>(new Chip::Command(c))),"native chip model deadlocked");
        auto& r=m.chip->result;
        if(r.status!=STATUS_OK)throw std::runtime_error("MODEL_FAULT status="+std::to_string(r.status)+" code="+std::to_string(r.code)+" pc="+std::to_string(r.pc)+": "+r.why);
    };
    while(std::getline(input,line)){
        if(line.empty() || line[0]=='#')continue;
        std::istringstream s(line);std::string kind;s>>kind;
        auto end=[&](){s>>std::ws;require(s.eof() || s.peek()=='#',"trailing chip command fields");};
        if(kind=="code"){
            std::string file;require(bool(s>>std::quoted(file)),"code requires a file");end();
            std::ifstream image(file);require(bool(image),"code image missing");
            std::vector<unsigned char> bytes;std::string word;
            while(image>>word){u64 v=integer("0x"+word);for(unsigned b=0;b<8;++b)bytes.push_back(static_cast<unsigned char>(v>>(8*b)));}
            require(!bytes.empty() && bytes.size()<=16384*8,"invalid code image size");
            m.chip->host_write(0,bytes.data(),bytes.size());
        }else if(kind=="data" || kind=="expect"){
            std::string addr,file;require(bool(s>>addr>>std::quoted(file)),"data/expect require address and file");end();
            u64 address=integer(addr);require(host_owned,"host data access requires drained ownership");
            std::ifstream in(file,std::ios::binary);require(bool(in),"data file missing");
            std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
            require(!bytes.empty(),"empty data file");
            if(kind=="data")m.chip->host_write(address,bytes.data(),bytes.size());
            else{
                std::vector<unsigned char> got(bytes.size());m.chip->host_read(address,got.data(),got.size());
                for(size_t i=0;i<got.size();++i)require(got[i]==bytes[i],"chip numerical mismatch at address "+std::to_string(address+i));
                checked+=got.size();
            }
        }else if(kind=="fence"){end();require(m.execute(nullptr,true),"maintenance deadlocked");host_owned=true;}
        else if(kind=="launch" || kind=="launch_roles" || kind=="launch_mask"){
            u64 id;unsigned gx,gy,gz;std::string pcstr,argstr;
            require(bool(s>>id>>pcstr>>argstr>>gx>>gy>>gz),"launch requires id pc arg gx gy gz");
            u64 mask=1;
            if(kind=="launch_roles"){
                std::string role_str,pair_str;require(bool(s>>role_str>>pair_str),"launch_roles requires mask and pair-alias flag");
                auto roles=integer(role_str),pair=integer(pair_str);require(roles>0 && roles<16 && pair<=1,"invalid role mapping");
                mask=roles|(pair<<28);
            }else if(kind=="launch_mask"){
                // The runtime's full 32-bit role mask: contexts, RPAIR (bit 28),
                // RSLICE (bit 31) and log2 slices (bits 30:29).
                std::string mask_str;require(bool(s>>mask_str),"launch_mask requires the role mask");mask=integer(mask_str);
            }
            end();
            auto start=sc_time_stamp();
            launch(grid_command(integer(pcstr),integer(argstr),gx,gy,gz,mask,all_cores));
            m.x.span("grid completion",300,start,id);host_owned=false;
        }else throw std::runtime_error("unknown chip script command: "+kind);
    }
    require(m.execute(nullptr,true),"maintenance deadlocked");
    m.summary(out+"/summary.json","completed",checked);m.x.close();
    return 0;
}
// Host server on two file descriptors (commands in, replies out); stdout and
// stderr stay free for SystemC and diagnostics. One line per command; W and
// R carry raw payload bytes after their line.
struct Channel {
    int in,out;std::vector<char> buffer;size_t head=0;
    bool fill(){if(head==buffer.size()){buffer.clear();head=0;}char chunk[65536];ssize_t n;
        do n=::read(in,chunk,sizeof chunk);while(n<0 && errno==EINTR);
        if(n<=0)return false;
        buffer.insert(buffer.end(),chunk,chunk+n);return true;}
    bool line(std::string& s){
        for(;;){for(size_t i=head;i<buffer.size();++i)if(buffer[i]=='\n'){s.assign(buffer.data()+head,i-head);head=i+1;return true;}
            if(!fill())return false;}
    }
    void bytes(unsigned char* p,u64 n){
        u64 got=0;while(got<n){if(head==buffer.size() && !fill())throw std::runtime_error("host channel closed mid-payload");
            u64 take=std::min<u64>(n-got,buffer.size()-head);std::memcpy(p+got,buffer.data()+head,take);head+=take;got+=take;}
    }
    void send(const void* p,size_t n){const char* c=static_cast<const char*>(p);while(n){ssize_t k=::write(out,c,n);if(k<0 && errno==EINTR)continue;
        require(k>0,"host reply channel closed");c+=k;n-=size_t(k);}}
    void send(const std::string& s){send(s.data(),s.size());}
};
static int serve(const char* config,const std::string& out,int fd_in,int fd_out){
    Model m(config,out);Channel ch{fd_in,fd_out,{},0};m.chip->serving=true;
    ch.send("READY "+std::to_string(m.cfg.cores())+" "+std::to_string(m.cfg.get("memory_base"))+" "+std::to_string(m.cfg.get("memory_bytes"))+"\n");
    std::string line;
    try{
        while(ch.line(line)){
            std::istringstream s(line);std::string op;s>>op;
            if(op=="W" || op=="R"){
                u64 address,n;require(bool(s>>address>>n),"W/R need address and length");
                std::vector<unsigned char> data(n);
                if(op=="W"){ch.bytes(data.data(),n);m.chip->host_write(address,data.data(),n);ch.send("OK\n");}
                else{m.chip->host_read(address,data.data(),n);ch.send("OK "+std::to_string(n)+"\n");ch.send(data.data(),n);}
                m.chip->host_charge(n);
            }else if(op=="L" || op=="G"){
                std::unique_ptr<Chip::Command> c(new Chip::Command);
                u64 pc,arg,mask,deadline;
                if(op=="L"){
                    unsigned px,py,pz,nx,ny,nz;require(bool(s>>pc>>arg>>mask>>px>>py>>pz>>nx>>ny>>nz>>deadline),"L fields");
                    c->launch=Launch::from_mask(mask);c->launch.pc=pc;c->launch.arg=arg;
                    c->launch.px=px;c->launch.py=py;c->launch.pz=pz;c->launch.nx=nx;c->launch.ny=ny;c->launch.nz=nz;
                }else{
                    unsigned gx,gy,gz,group;u64 cores,stride;require(bool(s>>pc>>arg>>mask>>gx>>gy>>gz>>cores>>stride>>group>>deadline),"G fields");
                    require(group==1 || group==2 || group==4,"physical group width");
                    require(cores && cores<(1ULL<<m.cfg.cores()),"active core mask outside the modeled cores");
                    *c=grid_command(pc,arg,gx,gy,gz,mask,cores);c->stride=stride;c->group=group;
                }
                c->deadline=deadline;
                if(!m.execute(std::move(c)))throw std::runtime_error("MODEL_DEADLOCK: no runnable process and the launch did not complete");
                auto& r=m.chip->result;
                std::ostringstream reply;reply<<"DONE "<<r.status<<' '<<r.code<<' '<<r.pc<<' '<<m.chip->start_cycle<<' '<<m.chip->end_cycle<<' '<<r.retired<<'\n';
                ch.send(reply.str());
                if(r.status!=STATUS_OK)std::cerr<<"launch status "<<r.status<<": "<<r.why<<'\n';
            }else if(op=="C"){std::ostringstream o;m.counters_json(o);auto t=o.str();ch.send("OK "+std::to_string(t.size())+"\n");ch.send(t);}
            else if(op=="F"){require(m.execute(nullptr,true),"maintenance deadlocked");ch.send("OK\n");}
            else if(op=="T"){unsigned on;require(bool(s>>on) && on<=1,"T needs 0 or 1");
                require(m.cfg.get("memory_model")==1 || on,"untimed execution needs the timeline memory (memory_model=1)");
                m.x.functional=!on;++m.x.counts[on?"host_timing_on":"host_timing_off"];ch.send("OK\n");}
            else if(op=="X"){m.summary(out+"/summary.json","completed",0);m.x.close();ch.send("BYE\n");return 0;}
            else throw std::runtime_error("unknown host command: "+op);
        }
        throw std::runtime_error("host channel closed without X");
    }catch(const std::exception& e){
        std::string why=e.what();for(auto& ch_:why)if(ch_=='\n')ch_=' ';
        try{ch.send("ERR "+why+"\n");}catch(...){}
        throw;
    }
}
int sc_main(int argc,char** argv){try{
    sc_set_time_resolution(1,SC_PS);
    if(argc==2 && std::string(argv[1])=="--classify"){classify();return 0;}
    if(argc==6 && std::string(argv[1])=="--serve")return serve(argv[2],argv[3],int(integer(argv[4])),int(integer(argv[5])));
    require(argc==4,"usage: tnca-chip CONFIG COMMANDS OUTPUT_DIRECTORY | --serve CONFIG OUTPUT_DIRECTORY IN_FD OUT_FD | --classify");
    return script(argv[1],argv[2],argv[3]);
}catch(const std::exception& e){std::cerr<<"MODEL_REFUSED: "<<e.what()<<'\n';return 2;}}
