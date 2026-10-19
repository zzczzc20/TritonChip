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
#include "tnca_isa_generated.hpp"
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unistd.h>

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
    // Program-scope fault and deadline state.
    bool aborting=false;Outcome fault;sc_time deadline=SC_ZERO_TIME;
    // Timing constants (configuration, uncalibrated).
    u64 t_issue,t_scalar,t_mul,t_vint,t_vfp,t_cvt,t_cmp,t_pred,t_shuf,t_red,t_fred,t_sfu_occ,t_sfu,t_load,t_bar,t_acc,
        t_rf,t_mac,t_mac_unbuf,t_pipe,t_done,t_zero,t_row,t_launch;bool abuf;
    // Counter slots bound once.
    u64 *c_instr,*c_sfu_queue,*c_sleep,*c_dep_wait,*c_mtx_wait,*c_mtx_busy,*c_mtx_cmds,*c_mma,*c_lsu_wait,*c_arrive_wait,
        *c_fence_wait,*c_programs_issued,*c_programs_completed;
    std::array<u64,unsigned(K::Count)> class_counts{};
    std::array<u64*,4> c_role_issued{},c_role_completed{};
    bool tracing=false;sc_time issue_time;std::string lsu_key;
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
        issue_time=x.cycles(t_issue);lsu_key=key("lsu");
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
    void ready(sc_time t){auto now=sc_time_stamp();if(t>now){*c_dep_wait+=cycles_of(t-now);wait(t-now);}}
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
        f.vr[index]=sc_time_stamp()+x.cycles(latency);
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
            case 14:return 5;case 15:return 16;case 16:return x.now();case 17:return f.retired;case 18:return 32;case 19:return 32;
            case 20:return 16;case 21:return 256;case 32:return f.arg;case 48:return 1;
            case 49:return 0x9efULL; // VECTOR_FP32|SFU|GATHER|ATOM_DEV|MATRIX_F16|MATRIX_BF16|TENSOR_DMA|MBARRIER|MATRIX_FP8
            case 64:case 65:return 0; // the faulting context stops; no surviving reader in this model
            default:return 0;}        // ISA sec 5.1: unimplemented ids read 0
    }
    void drain_own(Frame& f){for(auto& t:f.sr)ready(t);for(auto& t:f.vr)ready(t);for(auto& t:f.pr)ready(t);f.physical->commit(sc_time_stamp());}
    //--------------------------------------------------------------------------
    // Matrix engine: one command at a time, core-wide (tnca_matrix.v). Every
    // matrix-FU instruction, a.vread included, waits for it to be free.
    //--------------------------------------------------------------------------
    void engine_wait(){auto now=sc_time_stamp();if(mtx_free>now){*c_mtx_wait+=cycles_of(mtx_free-now);wait(mtx_free-now);}}
    void engine_start(const Frame& f,u64 busy){mtx_free=sc_time_stamp()+x.cycles(busy);mtx_role=f.role;*c_mtx_busy+=busy;++*c_mtx_cmds;}
    void mma(Frame& f,const Inst& in){
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
        tnca_matrix_ops::mma(acc[slot],A,B,K,dt);
        bool buffered=abuf && (bytes==1?K!=64:K==16);
        u64 sweep=(buffered?a_regs*t_rf:0)+K*(t_rf+16*(buffered?t_mac:t_mac_unbuf));
        u64 busy=sweep+t_pipe+1+t_done;
        engine_start(f,busy);++*c_mma;
        sc_time free_at=sc_time_stamp()+x.cycles(sweep),done_at=mtx_free;
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
    void execute(Frame& f){
        const std::string role_key=key("role_"+std::to_string(f.role));
        for(;;){
            if(deadline!=SC_ZERO_TIME && sc_time_stamp()>deadline)throw DeadlineExpired{};
            const Inst& in=code.at(f.pc);
            if(in.kind==K::Illegal)throw ArchFault(trapcode::illegal,in.why+" at pc "+std::to_string(f.pc));
            if(in.kind==K::Refused)throw std::runtime_error("MODEL_REFUSED "+in.why+" (word "+std::to_string(in.word)+" at pc "+std::to_string(f.pc)+")");
            const unsigned d=in.d,a=in.a,b=in.b,c=in.c,func=in.func,mod=in.mod;u64 next=f.pc+8;auto begin=sc_time_stamp();
            auto rs=[&](unsigned r)->u64{ready(f.sr[r]);return r?f.s[r]:0ULL;};
            auto ws=[&](unsigned r,u64 value,u64 latency){if(r){ready(f.sr[r]);f.s[r]=value;f.sr[r]=sc_time_stamp()+x.cycles(latency);}};
            auto rv=[&](unsigned r){ready(f.vr[r]);f.v[r]=f.physical->read(f.vbase+r,sc_time_stamp());};
            auto wp=[&](unsigned r,unsigned value,u64 latency){if(r){ready(f.pr[r]);f.p[r]=value&65535;f.pr[r]=sc_time_stamp()+x.cycles(latency);}};
            unsigned mask=65535;
            if(in.major>=16){ready(f.pr[in.pred]);mask=f.p[in.pred];if(in.invert)mask^=65535;}
            // No cycle polling: issue and dependencies reserve their next event.
            issue.use(x,issue_time,t_issue);
            ++class_counts[unsigned(in.kind)];
            switch(in.kind){
            case K::Exit:drain_own(f);++f.retired;++*c_instr;return;
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
                break;}
            case K::Lsu:{
                bool store=func==8 || func==24 || func==40,vector=func>=16,gather=func>=32;
                unsigned size=1U<<(mod&7);
                u64 base=rs(a);if(!gather)base+=in.simm;
                if(vector){rv(d);if(gather)rv(b);}else if(store)rs(d);
                // A scalar access is lane 0 of the mask (tnca_lsu.v).
                unsigned active=vector?mask:(mask&1);
                if(!active)break;
                auto wait_start=sc_time_stamp();lsu.acquire(x,lsu_key);*c_lsu_wait+=cycles_of(sc_time_stamp()-wait_start);
                struct Part {unsigned lane,offset,bytes,shift;};
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
            case K::Fence:{auto t=sc_time_stamp();drain_own(f);*c_fence_wait+=cycles_of(sc_time_stamp()-t);break;}
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
                if(func==24)wp(d,comparison,t_cmp);
                else publish(f,d,mask,t_vfp);
                break;}
            case K::Cvt:{
                rv(a);rv(d);
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=tnca_vector_ops::cvt(func,mod,f.v[a][lane]);
                publish(f,d,mask,t_cvt);break;}
            case K::Pred:{
                unsigned pa=0,pb=0;if(func!=19){ready(f.pr[a]);pa=f.p[a];}
                if(func==1 || func==2 || func==3 || func==5){ready(f.pr[b]);pb=f.p[b];}
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
            case K::Extract:rv(a);ws(d,f.v[a][mod&15],t_shuf);break;
            case K::Insert:{rv(d);auto value=std::uint32_t(rs(a));if(mask&(1<<(mod&15)))f.v[d][mod&15]=value;publish(f,d,mask&(1<<(mod&15)),t_shuf);break;}
            case K::Reduce:{
                rv(a);
                std::uint32_t reduced=tnca_vector_ops::reduce(func,f.v[a],mask);
                u64 latency=func>=8?t_fred:t_red;
                if(mod&1){rv(d);f.v[d].fill(reduced);publish(f,d,65535,latency);}
                else ws(d,reduced,latency);
                break;}
            case K::Sfu:{
                rv(a);rv(d);
                auto waiting=sc_time_stamp();ready(sfu_available);
                *c_sfu_queue+=cycles_of(sc_time_stamp()-waiting);
                sfu_available=sc_time_stamp()+x.cycles(t_sfu_occ);
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=tnca_sfu::evaluate(func,f.v[a][lane]);
                publish(f,d,mask,t_sfu);break;}
            case K::Desc:{
                unsigned slot=sliced(f,d,4);
                tensor.configure(func,slot,func==0?rs(a):0,in.imm);break;}
            case K::Xfer:{
                u64 token=token_of(f,rs(c));
                tensor.issue(sliced(f,a,4),sliced(f,d,4),token,1U<<(mod&7),func==8,mod&32,mask!=0,f.physical,f.vbase,f.role,
                    [&](unsigned rank){
                        arch(b+rank<=64,trapcode::descriptor,"DESCRIPTOR_FAULT: coordinates exceed scalar file");
                        std::array<u64,5> coordinates{};for(unsigned dim=0;dim<rank;++dim)coordinates[dim]=rs(b+dim);return coordinates;
                    },[&](unsigned base,unsigned length){
                        arch(base<64 && length<=64-base,trapcode::descriptor,"DESCRIPTOR_FAULT: transfer exceeds role mapping");
                        for(unsigned r=base;r<base+length;++r){ready(f.vr[r]);}
                        f.physical->commit(sc_time_stamp());
                    });
                break;}
            case K::Barrier:{
                if(func==0)barriers[sliced(f,d,5)].reset();
                else if(func==1){unsigned index=sliced(f,a,5);ws(d,barriers[index].arm(index,b,c),t_bar);}
                else if(func==2){
                    // ISA sec 9A.5 with RPAIR's precision (tnca_core.v stall_arrive):
                    // own register writes landed, own engine commands finished.
                    auto t=sc_time_stamp();drain_own(f);
                    while(tensor.outstanding[f.role])wait(tensor.role_done);
                    if(mtx_role==f.role)engine_wait();
                    *c_arrive_wait+=cycles_of(sc_time_stamp()-t);
                    barriers[sliced(f,a,5)].arrive();
                }else if(func==3 || func==4){
                    u64 token=token_of(f,rs(a));unsigned index=unsigned(token&65535);
                    if(func==3)ws(d,barriers[index].test(token),t_bar);
                    else {auto sleep=sc_time_stamp();barriers[index].sleep(token);*c_sleep+=cycles_of(sc_time_stamp()-sleep);}
                }else if(func==5)barriers[sliced(f,a,5)].invalidate();
                else throw std::runtime_error("reserved barrier function");
                break;}
            case K::MZero:case K::AZero:{
                // m.zero and a.zero are both the engine's zero sweep (cmd_op 0):
                // whole rows, no lane mask (tnca_core.v mtx_cmd_op).
                engine_wait();acc[sliced(f,d,4)]={};engine_start(f,t_zero+t_done);break;}
            case K::Mma:mma(f,in);break;
            case K::ARow:{
                // a.vwrite (direct), a.mul.row / a.fma.row (one rounding, ftz=0).
                u64 row=rs(a)&15;std::uint32_t scale=0;
                if(func==2)scale=std::uint32_t(rs(b));else{rv(b);if(func==3)scale=std::uint32_t(rs(c));}
                engine_wait();
                tnca_matrix_ops::row_op(acc[sliced(f,d,4)][row],func,scale,func==2?PhysicalRegisters::Vector{}:f.v[b],mask);
                engine_start(f,func==1?t_row+t_done:t_row+t_pipe+1+t_done);break;}
            case K::AVread:{
                u64 row=rs(b)&15;engine_wait();rv(d);
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
    u64 *c_grid_launches,*c_single_launches;
    Chip(sc_module_name n,Context& ctx,HBM& mem,Network& net,std::vector<std::unique_ptr<Slice>>& s):
        sc_module(n),x(ctx),hbm(mem),caches(s){
        code.read_word=[this](u64 pc){u64 v=0;unsigned char bytes[8];host_read(pc,bytes,8);for(unsigned b=0;b<8;++b)v|=u64(bytes[b])<<(8*b);return v;};
        for(unsigned i=0;i<x.c.cores();++i){
            cores.emplace_back(new Core(("core"+std::to_string(i)).c_str(),x,i,code));
            cores.back()->memory.bind(net.target);cores.back()->tensor.memory.bind(net.target);
        }
        c_grid_launches=&x.counts["chip_grid_launches"];c_single_launches=&x.counts["chip_single_launches"];
        SC_THREAD(run);
    }
    // Functionally coherent host access in zero modeled time: the device is
    // idle, so a line is either valid in its home slice (newest copy) or in
    // HBM. Writes update both copies. Nothing is flushed or perturbed.
    void host_access(u64 address,unsigned char* data,u64 n,bool write){
        if(!n)return;
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
    void flush(){unsigned remaining=unsigned(caches.size());sc_event done;
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
                    wait(x.cycles(x.c.get("dispatch_cycles")+ptr->t_launch));
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
    Outcome run_command(const Command& cmd){
        for(auto& core:cores)core->deadline=cmd.deadline?sc_time_stamp()+x.cycles(cmd.deadline):SC_ZERO_TIME;
        start_cycle=x.now();Outcome out;
        if(cmd.grid){++*c_grid_launches;out=dispatch(cmd);}
        else{
            ++*c_single_launches;wait(x.cycles(cores[0]->t_launch));
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
    Model(const char* config,const std::string& out):cfg(config),x(cfg,out),hbm("hbm",x),net("noc",x){
        for(unsigned i=0;i<cfg.cores();++i){
            caches.emplace_back(new Slice(("home"+std::to_string(i)).c_str(),x,i));
            auto home="home_"+std::to_string(i),mem="hbm_"+std::to_string(i);
            probes.emplace_back(new SocketMonitor(home.c_str(),x,home,500+i));
            net.homes[i].bind(probes.back()->target);probes.back()->downstream.bind(caches.back()->target);
            probes.emplace_back(new SocketMonitor(mem.c_str(),x,mem,600+i));
            caches.back()->memory.bind(probes.back()->target);probes.back()->downstream.bind(hbm.target);
        }
        chip.reset(new Chip("chip",x,hbm,net,caches));
    }
    // Run one command inside the simulation; false if the kernel starved.
    bool execute(std::unique_ptr<Chip::Command> cmd,bool flush=false){
        chip->command_done=false;
        if(flush)chip->flush_request=true;else chip->pending=std::move(cmd);
        chip->wake.notify(SC_ZERO_TIME);
        sc_start();
        return chip->command_done;
    }
    void counters_json(std::ostream& o){
        chip->report();
        o<<"{";bool first=true;for(auto& c:x.counts){if(!first)o<<',';first=false;o<<'"'<<c.first<<"\":"<<c.second;}o<<"}";
    }
    void summary(const std::string& path,const std::string& status,u64 checked){
        std::ofstream result(path);
        result<<"{\"schema\":\"tnca-systemc-result/1\",\"status\":\""<<status<<"\",\"execution\":\"native-isa-event-driven\",\"coverage\":\"partial-ISA\",\"performance_gate\":\"not-measured\",\"calibration\":\"not-measured\",\"visible_cycles\":"<<x.now()<<",\"launches\":"<<chip->launches<<",\"checked_bytes\":"<<checked<<",\"counters\":";
        counters_json(result);result<<"}\n";
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
    Model m(config,out);Channel ch{fd_in,fd_out,{},0};
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
