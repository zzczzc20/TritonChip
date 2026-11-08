//==============================================================================
// The cycle-level core (core_model=1): tools/performance_harness/align/CYCLE_CORE.md.
//
// Included by chip.cpp after `struct Core`. One `step()` per cycle: every wire
// from the pre-edge registers, then every register update at the edge
// (tools/performance_harness/align/CORE_TIMING.md is the rule book; its
// section numbers are cited as [2.7] etc.).
//==============================================================================

// The replay of the core's port (first slice) is the platform's CoreReplay
// (platform.inc): the device asks it what the RTL's memory accepted and when.

struct CycleCore {
    Core& c;Context& x;
    static constexpr unsigned FP_LAT=2,MUL_LAT=3,WB_DEPTH=8,NBEAT=8,NPORT=5;
    // the matrix engine's result pipeline (tnca_matrix.v PIPE: FMA_LAT + OPND_REG = 3; R41's MAC_DP lanes
    // OPND_REG + popcount(MAC_CUTS) = 5, core_mtx_pipe) and MTX_VRF_REG (core_mtx_vrf_reg: the banked file's
    // port-3 response registered, the matrix takes its data a cycle later) [4.1]
    unsigned mtx_pipe=3;bool mtx_vrf_reg=false;bool p3q_v=false;PhysicalRegisters::Vector p3q_data{};
    //-------------------------------------------------------------- state [1]
    struct Ctx {
        u64 pc=0;bool running=false,faulted=false,blocked=false,parked=false,launched=false;u64 instret=0;
        std::unique_ptr<Frame> f;bool slp_v=false;u64 slp_token=0;u64 pending_wake=0;bool wake_err=false;
    };
    std::array<Ctx,4> ctx;unsigned rr_ptr=0;
    struct Bundle {bool valid=false;unsigned rctx=0;u64 pc=0;const Inst* in=nullptr;};
    Bundle f1,fb,d,r,xb;                                  // [1.2]
    unsigned of_got=0,pf_tag=0;bool adv_q=false;unsigned vrf_rr=0;   // [1.3]
    struct BankRsp {bool valid=false;unsigned port=0;unsigned phys=0;};std::array<BankRsp,4> bank_rsp{};
    bool sb_s[4][64]={},sb_v[4][64]={},sb_p[4][16]={},sb_acc[16]={};unsigned wb_rsv=0;   // [1.4]
    struct Result {u64 port=0;unsigned cls=0,rctx=0,idx=0,mask=0,src=0;u64 sdata=0;PhysicalRegisters::Vector vdata{};}; // cls 1 S,2 V,3 P; src 0 single,1 fpnorm,2 mul,3 vector late (R41 F04)
    std::vector<Result> sched;                            // fp_v / mu_v tag pipelines
    struct WBStage {bool en=false;Result res;} w;         // w_*
    Result out_res;                                       // the hybrid's completion register (out_*) [3.6]
    bool red_pend=false;Result red;                       // red_*
    u64 fpu_start=0,fpu_busy_until=0,fpu_done_at=0;bool fpu_red=false; // u_vfpu sequencer: busy during (x+1 .. busy_until), done pulse at done_at
    //------------------------------------------------------------ LSU [3.2]
    struct Part {unsigned lane,offset,bytes,shift;};
    struct Beat {u64 addr=0;bool store=false;u64 data=0;unsigned be=0;std::vector<Part> parts;bool issued=false,done=false;u64 rsp=0,rdata=0;};
    struct Unit {
        unsigned st=0;                                    // 0 IDLE,1 SETUP,2 RUN,3 DRAIN,4 DONE
        u64 accept=0;std::vector<Beat> beats;unsigned next=0,outstanding=0;
        bool store=false,vector=false,amo=false,sign=false;unsigned rctx=0,dst=0,size=0,active=0;
        u64 last_issue=0;bool cpl=false;
        std::vector<unsigned> q;                          // REQ_REG: the two-entry queue of planned beats [3.6]
        bool line_ok=false;u64 line=0,lbe=0;unsigned loff=0;std::array<u64,8> lwdata{};   // the line path's form of the access
    };
    Unit units[2];unsigned old=0;unsigned nunits=2;
    //------------------------------------------------------------ R41 hybrid LSU [3.6]
    // core_lsu_hybrid=1 (HYBRID_LSU, tnca_lsu_hybrid.v): X lands an access in the core's command
    // register; the line path (tnca_lsu_line.v) takes a contiguous vector access of elements up to
    // four bytes that fits one 64-byte line as one line request, the word path (one tnca_lsu with
    // REQ_REG behind tnca_word_line_adapter.v) everything else; one path at a time (mode drain);
    // every completion passes the hybrid's completion register before the writeback port.
    bool hybrid=false,lsu_first=false;
    bool cmd_pending=false;Unit cmd;
    struct Line {bool valid=false,done=false,issued=false,store=false,sign=false;unsigned size=0,active=0,rctx=0,dst=0,off=0;u64 addr=0,be=0,rsp=0;std::array<u64,8> wdata{},rdata{};};
    std::vector<Line> lines;unsigned nlines=0;bool fq_valid=false;unsigned fq_tag=0,line_rr=0,held_pick=0;bool held=false;
    bool out_valid=false,out_fast=false,out_has=false;
    std::unordered_map<u64,std::array<u64,8>> wide_rsp;   // trace only: the line each response carries on the 512-bit port
    //------------------------------------------------------------ engines [4]
    unsigned mtx_rctx=0;
    bool mtx_busy_q=false,txf_busy_q=false;unsigned txf_rctx=0;
    // A beat's response target (declared before the engines, which name it).
    struct Target {unsigned kind=0,a=0,b=0;};   // 0 LSU beat (unit a, beat b); 1 tensor slot low (a); 2 slot high; 3 tensor tag (a); 4 descriptor word
#include "cycle_tensor.inc"
    CycleTensor txf{*this};
#include "cycle_matrix.inc"
    CycleMatrix mtx{*this};
    //------------------------------------------------------------ misc
    u64 t=0,launch_cycle=0,last_progress=0;bool launched_any=false;const Launch* cur=nullptr;
    // Debugging aid: TNCA_MODEL_TXF_DEBUG=core:launch:from:to:path dumps the transfer engine and the
    // port every cycle of that launch window (cycles relative to the launch) to `path`.
    u64 launch_n=0;long dbg_core=-1,dbg_launch=-1;u64 dbg_from=0,dbg_to=0;std::ofstream dbg;
    // TNCA_MODEL_VRF_DEBUG=core:launch:from:to:path dumps the register-file read arbiter per cycle (ports, banks, grants)
    long vdbg_core=-1,vdbg_launch=-1;u64 vdbg_from=0,vdbg_to=0;std::ofstream vdbg;
    // Diagnostic snapshot of the pre-edge state (the watchdog's message).
    std::string snapshot(const char* wires)const{
        std::ostringstream o;o<<"cycle "<<t<<" (launch+"<<(t-launch_cycle)<<") "<<wires<<"\n";
        for(unsigned i=0;i<4;++i){const Ctx& q=ctx[i];o<<" ctx"<<i<<": pc="<<std::hex<<q.pc<<std::dec<<" running="<<q.running<<" faulted="<<q.faulted<<" blocked="<<q.blocked<<" parked="<<q.parked<<" instret="<<q.instret<<" slp="<<q.slp_v<<"\n";}
        auto bd=[&](const char* n,const Bundle& b){o<<" "<<n<<": valid="<<b.valid<<" rctx="<<b.rctx<<" pc="<<std::hex<<b.pc<<std::dec<<(b.valid&&b.in?std::string(" kind=")+kind_name(b.in->kind):std::string())<<"\n";};
        bd("f1",f1);bd("fb",fb);bd("d",d);bd("r",r);bd("x",xb);
        o<<" w.en="<<w.en<<" red_pend="<<red_pend<<" wb_rsv="<<wb_rsv<<" of_got="<<of_got<<" pf_tag="<<pf_tag<<" sched="<<sched.size()<<" fpu_busy_until="<<fpu_busy_until<<" mtx_busy="<<mtx_busy()<<" txf_busy="<<txf_busy()<<"\n";
        for(unsigned i=0;i<nunits;++i){const Unit& u=units[i];o<<" lsu"<<i<<": st="<<u.st<<" beats="<<u.beats.size()<<" next="<<u.next<<" outstanding="<<u.outstanding<<" queued="<<u.q.size()<<" cpl="<<u.cpl<<"\n";}
        if(hybrid){o<<" hybrid: cmd_pending="<<cmd_pending<<" line_ok="<<cmd.line_ok<<" lines="<<nlines<<" fq_valid="<<fq_valid<<" held="<<held<<" out_valid="<<out_valid<<" out_fast="<<out_fast<<"\n";}
        for(unsigned rc=0;rc<4;++rc){o<<" sb"<<rc<<":";for(unsigned i=0;i<64;++i){if(sb_s[rc][i])o<<" s"<<i;}for(unsigned i=0;i<64;++i){if(sb_v[rc][i])o<<" v"<<i;}for(unsigned i=0;i<16;++i){if(sb_p[rc][i])o<<" p"<<i;}o<<"\n";}
        return o.str();}

    std::unique_ptr<CoreReplay> replay;u64 trace_tag=0;
    // Every beat on the port gets an id; its response is delivered at a cycle
    // (`rsp` fields set then), either scheduled here (timeline memory,
    // replay) or by the chip's cycle uncore (uncore_model=1).
    std::unordered_map<u64,Target> targets;
    struct Scheduled {u64 id,when,data;};std::vector<Scheduled> scheduled;u64 next_id=1;
    struct CycleUncore* uncore=nullptr;
    void schedule(u64 id,u64 when,u64 data){scheduled.push_back({id,when,data});}
    void deliver(u64 id,u64 when,u64 data){
        auto it=targets.find(id);require(it!=targets.end(),"response for an unknown beat");Target tg=it->second;targets.erase(it);
        switch(tg.kind){
            case 0:{Beat& b=units[tg.a].beats[tg.b];b.rsp=when;b.rdata=data;break;}
            case 1:{auto& sl=txf.sl[tg.a];sl.rsp_lo=when;sl.dlo=data;break;}
            case 2:{auto& sl=txf.sl[tg.a];sl.rsp_hi=when;sl.dhi=data;break;}
            case 3:{auto& g=txf.tg[tg.a];g.rsp=when;u64 keepm=0;for(unsigned j=0;j<8;++j)if((g.xkeep>>j)&1)keepm|=255ULL<<(j*8);g.data=data&keepm;break;}
            case 4:txf.dl_rsp=when;txf.dl_data=data;break;
            case 6:txf.ls[tg.a].rsp=when;break;          // R41 G4: an engine line slot (data captured at issue)
            case 7:txf.dl_lrsp[tg.a]=when;break;         // R41 G4: a TensorMap line
            case 5:lines[tg.a].rsp=when;break;          // a line request: its data was captured at issue
        }
        if(c.core_trace){
            if(hybrid){auto w=wide_rsp.find(id);std::array<u64,8> line{};if(w!=wide_rsp.end()){line=w->second;wide_rsp.erase(w);}c.core_trace.PW(when,id,line);}
            else c.core_trace.P(when,id,data);}
    }
    void deliveries(){  // at the start of a cycle: every response due now
        for(size_t i=0;i<scheduled.size();){if(scheduled[i].when<=t){Scheduled s=scheduled[i];scheduled[i]=scheduled.back();scheduled.pop_back();deliver(s.id,t,s.data);}else ++i;}
    }
    u64 *c_cycles,*c_instr,*c_stall,*c_replay_mismatch,*c_replay_unrecorded,*c_hazard,*c_lsu,*c_fpu,*c_mtx,*c_txf,*c_fence,*c_arrive,*c_rsv,*c_steal,*c_vrf,*c_park;
    Outcome fault;bool aborting=false;

    CycleCore(Core& core):c(core),x(core.x){
        auto key=[&](const char* k){return "core_"+std::to_string(c.id)+"_"+k;};
        c_cycles=&x.counts[key("cycles")];c_instr=&x.counts[key("instructions")];c_stall=&x.counts[key("stall_cycles")];
        c_replay_mismatch=&x.counts[key("replay_mismatches")];c_replay_unrecorded=&x.counts[key("replay_unrecorded")];
        c_hazard=&x.counts[key("stall_hazard")];c_lsu=&x.counts[key("stall_lsu")];c_fpu=&x.counts[key("stall_fpu")];c_mtx=&x.counts[key("stall_mtx")];
        c_txf=&x.counts[key("stall_txf")];c_fence=&x.counts[key("stall_fence")];c_arrive=&x.counts[key("stall_arrive")];c_rsv=&x.counts[key("stall_rsv")];
        c_steal=&x.counts[key("stall_wb_steal")];c_vrf=&x.counts[key("stall_vrf")];c_park=&x.counts[key("context_parks")];
        if(const char* p=std::getenv("TNCA_MODEL_TXF_DEBUG")){std::string s(p);std::vector<std::string> f;size_t b=0;
            for(size_t i=0;i<=s.size();++i)if(i==s.size()||s[i]==':'){f.push_back(s.substr(b,i-b));b=i+1;}
            if(f.size()==5){dbg_core=std::stol(f[0]);dbg_launch=std::stol(f[1]);dbg_from=std::stoull(f[2]);dbg_to=std::stoull(f[3]);if(dbg_core==long(c.id))dbg.open(f[4]);}}
        if(const char* p=std::getenv("TNCA_MODEL_VRF_DEBUG")){std::string s(p);std::vector<std::string> f;size_t b=0;
            for(;;){size_t e=s.find(':',b);if(e==std::string::npos){f.push_back(s.substr(b));break;}f.push_back(s.substr(b,e-b));b=e+1;}
            if(f.size()==5){vdbg_core=std::stol(f[0]);vdbg_launch=std::stol(f[1]);vdbg_from=std::stoull(f[2]);vdbg_to=std::stoull(f[3]);if(vdbg_core==long(c.id))vdbg.open(f[4]);}}
        replay=CoreReplay::open_for(c.id);
        nunits=unsigned(x.c.get("lsu_slots"));require(nunits==1 || nunits==2,"the cycle core builds one or two word units (lsu_slots)");
        hybrid=x.c.get_or("core_lsu_hybrid",0)!=0;lsu_first=x.c.get_or("core_tensor_line",0)!=0;
        require(!lsu_first || hybrid,"core_tensor_line=1 needs core_lsu_hybrid=1");
        if(hybrid){
            require(nunits==1,"core_lsu_hybrid=1 models R41's one word unit (lsu_slots=1, LSU_INFLIGHT 1)");
            u64 n=x.c.get_or("core_lsu_line_slots",64);require(n>=2 && n<=64 && (n&(n-1))==0,"core_lsu_line_slots is a power of two, 2..64");
            lines.resize(n);
        }
        valu_late=x.c.get_or("core_valu_late",0)!=0;switch_opcode=x.c.get_or("core_switch_opcode",0)!=0;
        mtx_pipe=unsigned(x.c.get_or("core_mtx_pipe",3));mtx_vrf_reg=x.c.get_or("core_mtx_vrf_reg",0)!=0;
        require(mtx_pipe>=1 && mtx_pipe<=8,"core_mtx_pipe is 1..8");
        txf.configure(lsu_first,unsigned(x.c.get_or("core_lsu_line_slots",64)),x.c.get_or("core_tensor_mul_pipe",0)!=0);
    }
    //------------------------------------------------------------ helpers
    static bool is_fpnorm(const Inst& in){return in.kind==K::Cvt || (in.kind==K::Vfp);} // VFP incl. vf.cmp, CVT [2.11]
    static bool is_fpmc(const Inst& in){return in.kind==K::Sfu || (in.kind==K::Reduce && in.func>=8);}
    static bool is_mul(const Inst& in){return in.kind==K::Salu && (in.func==3 || in.func==4 || in.func==5 || in.func==34);}
    static bool is_matrix_fu(const Inst& in){return in.kind==K::Mma || in.kind==K::MZero || in.kind==K::AZero || in.kind==K::ARow || in.kind==K::AVread;}
    static bool is_txfer_fu(const Inst& in){return in.kind==K::Desc || in.kind==K::Xfer || in.kind==K::Prefetch;}
    static bool is_mem(const Inst& in){return in.kind==K::Lsu || in.kind==K::Atomic;}
    static bool is_store(const Inst& in){return in.kind==K::Lsu && (in.func==8 || in.func==24 || in.func==40);}
    // Destination class: 0 none, 1 S, 2 V, 3 P, 4 A
    static unsigned dest_class(const Inst& in){
        switch(in.kind){
            case K::Special:case K::Extract:return 1;
            case K::Salu:return 1;
            case K::Branch:return in.func==8?1:0;
            case K::Pred:return (in.func>=16 && in.func!=19)?1:3;
            case K::Barrier:return (in.func==1 || in.func==3)?1:0;
            case K::Reduce:return (in.mod&1)?2:1;
            case K::Vint:return (in.func==24 || in.func==56)?3:2;
            case K::Vfp:return in.func==24?3:2;
            case K::Cvt:case K::Shuffle:case K::Insert:case K::Sfu:case K::AVread:return 2;
            case K::Lsu:if(is_store(in)){return 0;}return in.func>=16?2:1;
            case K::Mma:case K::MZero:case K::AZero:case K::ARow:return 4;
            default:return 0;
        }
    }
    // the X stage's single-cycle destination (not deferred to a pipeline/unit)
    // R41 F04 (core_valu_late=1, VALU_LATE): the vector multiply family and the integer reductions write back
    // one cycle after X, scheduled like the scalar multiply with writeback delay 1 [2.12]
    bool valu_late=false,switch_opcode=false;
    bool is_vlate(const Inst& in)const{return valu_late && ((in.kind==K::Vint && in.major==16 && (in.func==3 || in.func==4 || in.func==5 || in.func==34))
                                                          || (in.kind==K::Reduce && in.func<8));}
    bool x_defers(const Inst& in)const{return is_mul(in) || is_fpnorm(in) || is_fpmc(in) || is_mem(in) || is_vlate(in);}
    unsigned wb_delay(const Inst& in)const{return is_mul(in)?3:is_vlate(in)?1:is_fpnorm(in)?2:0;}
    static bool books_wb(const Inst& in){unsigned dc=dest_class(in);return (dc==1 || dc==2 || dc==3) && !(is_mem(in) || is_fpmc(in));}
    // Sources read through the scoreboards (decoder o_*_en) [2.7]
    // vector sources carry their read port: the decoder binds by field (a: va = port 0, b: vb = 1, c: vc = 2; a vector
    // store's data in d goes to vb, a scatter's to vc, v.insert's Vd to vb: tnca_decoder.v `TNCA_DEC_SRC_V*`, 524, 601, 614).
    // Under contention with the matrix (3) and tensor (4) ports the port decides who wins the bank [2.9]
    struct Srcs {unsigned ns=0,nv=0,np=0;unsigned s[8],v[3],vp[3],p[3];void S(unsigned i){if(ns<8)s[ns++]=i;}void V(unsigned i,unsigned port){if(nv<3){vp[nv]=port;v[nv++]=i;}}void P(unsigned i){if(np<3)p[np++]=i;}};
    static Srcs sources(const Inst& in){
        Srcs o;const unsigned f=in.func;
        if(in.major>=16)o.P(in.pred);
        switch(in.kind){
            case K::Salu:
                if(f>=64 && f<=66){if(f==65)o.S(in.d);}
                else if(f==25){o.S(in.c);o.S(in.a);o.S(in.b);}
                else{o.S(in.a);if(!(f>=32 && f<=63) && f!=0 && f!=20 && f!=21 && f!=22 && f!=23)o.S(in.b);}
                break;
            case K::Branch:if(f==1 || f==2 || f==4 || f==9){o.S(in.a);}break;
            case K::Lsu:{bool store=is_store(in),vector=f>=16,gather=f>=32;o.S(in.a);
                if(vector){if(gather){o.V(in.b,1);}if(store)o.V(in.d,gather?2:1);}else if(store){o.S(in.d);}break;}
            case K::Vint:{bool imm=f>=32 && f<=63;
                if(f==26)o.S(in.a);
                if(f!=64 && f!=27 && f!=26)o.V(in.a,0);
                if(!imm && f!=0 && f!=20 && f!=21 && f!=26 && f!=27 && f!=64)o.V(in.b,1);
                break;}
            case K::Vfp:o.V(in.a,0);
                if(f!=20 && f!=21)o.V(in.b,1);
                if(f==6 || f==7)o.V(in.c,2);
                break;
            case K::Cvt:case K::Sfu:case K::Reduce:case K::Extract:o.V(in.a,0);break;
            case K::Pred:if(f!=19){o.P(in.a);}else {o.S(in.a);}if(f==1 || f==2 || f==3 || f==5){o.P(in.b);}break;
            case K::Shuffle:o.V(in.a,0);if(f==3){o.V(in.b,1);}else {o.S(in.b);}break;
            case K::Insert:o.S(in.a);o.V(in.d,1);break;   // tnca_decoder.v: read-modify-write on Vd (o_vb_en)
            case K::Desc:if(f==0){o.S(in.a);}break;
            case K::Xfer:case K::Prefetch:o.S(in.c);o.S(in.b);o.S(in.b+1);o.S(in.b+2);o.S(in.b+3);o.S(in.b+4);break;
            case K::Barrier:if(f==3 || f==4){o.S(in.a);}break;
            case K::Mma:o.S(in.c);if(in.mod&16){o.S((in.c+1)&63);}break;
            case K::ARow:o.S(in.a);if(f==2)o.S(in.b);else{o.V(in.b,1);if(f==3)o.S(in.c);}break;
            case K::AVread:o.S(in.b);break;
            default:break;
        }
        return o;
    }
    Frame& F(unsigned rc){return *ctx[rc].f;}
    unsigned phys(unsigned rc,unsigned idx)const{return ctx[rc].f->vbase+idx;}
    bool wb_hit(unsigned cls,unsigned rc,unsigned idx)const{return w.en && w.res.cls==cls && w.res.rctx==rc && w.res.idx==idx;}
    // reads at X: write-first scalar/predicate files forward the WB stage [2.12]
    u64 rS(unsigned rc,unsigned i){if(!i){return 0;}if(wb_hit(1,rc,i)){return w.res.sdata;}return F(rc).s[i];}
    unsigned rP(unsigned rc,unsigned i){if(!i){return 65535;}if(wb_hit(3,rc,i)){return unsigned(w.res.sdata);}return F(rc).p[i];}
    PhysicalRegisters::Vector rV(unsigned rc,unsigned i){require(i<64,"vector index");return F(rc).physical->read(phys(rc,i),sc_time_stamp());}
    //------------------------------------------------------------ X effects (collected during the cycle, applied at the edge)
    struct XEffects {
        bool has_dest=false;Result single;              // nw_* from X
        std::vector<Result> deferred;                   // fp/mul pipeline entries
        bool red_start=false;Result redres;u64 red_done=0;bool red_is_sfu=false;
        bool mem_go=false;Unit req;
        bool mtx_cmd=false;u64 mtx_busy=0,mtx_srcfree=0;unsigned mtx_slot=0;u64 token=0,sf_token=0;bool bar_en=false;CycleMatrix::Cmd mcmd;
        bool txf_cmd=false;
        bool txf_cfg=false;unsigned cfg_func=0,cfg_slot=0,cfg_imm=0;u64 cfg_addr=0;   // applied at the edge (registered in tnca_tensor.v)
        bool txf_issue=false;CycleTensor::Cmd txf_cmd_v;
        bool redirect=false;u64 target=0;bool exit=false,sleep=false,trap=false;unsigned code=0;std::string why;
        bool bar_arm=false,bar_arrive=false,bar_init=false,bar_inval=false;unsigned bar=0;
    };
    //------------------------------------------------------------ the X stage [2.11]
    void execute(XEffects& e){
        const Inst& in=*xb.in;Frame& f=F(xb.rctx);const unsigned rc=xb.rctx;
        const unsigned dd=in.d,a=in.a,b=in.b,cc=in.c,func=in.func,mod=in.mod;
        auto single=[&](unsigned cls,unsigned idx,u64 s,unsigned mask,const PhysicalRegisters::Vector* v){
            e.has_dest=true;e.single=Result{t,cls,rc,idx,mask,0,s,v?*v:PhysicalRegisters::Vector{}};};
        auto wS=[&](unsigned idx,u64 v){if(idx)single(1,idx,v,0,nullptr);};
        auto wP=[&](unsigned idx,unsigned v){if(idx)single(3,idx,v&65535,0,nullptr);};
        auto wV=[&](unsigned idx,unsigned mask,const PhysicalRegisters::Vector& v){single(2,idx,0,mask,&v);};
        auto defer=[&](unsigned cls,unsigned idx,unsigned src,u64 s,unsigned mask,const PhysicalRegisters::Vector* v){
            if(cls!=2 && !idx)return;
            e.deferred.push_back(Result{t+(src==2?MUL_LAT:src==3?1:FP_LAT),cls,rc,idx,mask,src,s,v?*v:PhysicalRegisters::Vector{}});};
        unsigned mask=65535;if(in.major>=16){mask=rP(rc,in.pred);if(in.invert)mask^=65535;}
        u64 next=xb.pc+8;
        switch(in.kind){
        case K::Exit:e.exit=true;e.redirect=true;break;
        case K::Nop:case K::Yield:case K::Prefetch:break;
        case K::Trap:e.trap=true;e.redirect=true;e.code=trapcode::user;e.why="TRAP_USER: code "+std::to_string((a<<16)|(b<<8)|cc);break;
        case K::Special:{u64 v;switch(mod){case 16:v=t-launch_cycle;break;case 17:v=ctx[rc].instret;break;default:v=c.special(mod,f);}wS(dd,v);break;}
        case K::Salu:{
            u64 value=0;
            if(func>=64 && func<=66){u64 val=u64(in.imm)<<(16*(mod&3));value=func==64?val:func==66?~val:(rS(rc,dd)&~(0xffffULL<<(16*(mod&3))))|val;}
            else if(func==25)value=rS(rc,cc)?rS(rc,a):rS(rc,b);
            else{u64 left=rS(rc,a),right=0;unsigned af=func;
                if(func>=32 && func<=63){right=zero_extended_immediate(func)?in.imm:in.simm;af=immediate_alu(func);}
                else if(func!=0 && func!=20 && func!=21 && func!=22 && func!=23)right=rS(rc,b);
                value=alu(af,left,right,64,mod&15);}
            if(is_mul(in)){defer(1,dd,2,value,0,nullptr);}else {wS(dd,value);}break;}
        case K::Branch:{
            auto off24=std::int32_t((unsigned(a)<<24)|(unsigned(b)<<16)|(unsigned(cc)<<8))>>8;
            if(func==0)next=xb.pc+u64(std::int64_t(off24)*8);
            else if(func==1 || func==2){bool test=rS(rc,a)!=0;if(test==(func==1))next=xb.pc+in.simm*8;}
            else if(func==4 || func==9)next=rS(rc,a)&~7ULL;
            else if(func==8){wS(dd,xb.pc+8);next=xb.pc+u64(std::int64_t(off24)*8);}
            else throw std::runtime_error("uncovered control operation");
            if(next!=xb.pc+8){e.redirect=true;e.target=next;}
            break;}
        case K::Lsu:{
            bool store=is_store(in),vector=func>=16,gather=func>=32;unsigned size=1U<<(mod&7);
            u64 base=rS(rc,a);if(!gather)base+=in.simm;
            unsigned active=vector?mask:(mask&1);
            PhysicalRegisters::Vector vd{},vb{};if(vector){if(store){vd=rV(rc,dd);}if(gather)vb=rV(rc,b);}
            Unit& u=e.req;u.st=1;u.accept=t;u.store=store;u.vector=vector;u.amo=false;u.sign=(mod&8)!=0;u.rctx=rc;u.dst=dd;u.size=size;u.active=active;
            // the line path's eligibility and request [3.6] (tnca_lsu_line.v req_eligible: contiguous vector,
            // elements up to four bytes, inside one 64-byte line, below 8 GiB, not the TMU page)
            if(hybrid && vector && !gather && size<=4){const unsigned off=unsigned(base&63);
                if(off+16*size<=64 && (base>>33)==0 && (base>>12)!=(0x7fff0000ULL>>12)){
                    u.line_ok=true;u.line=base&~63ULL;u.loff=off;
                    for(unsigned lane=0;lane<16;++lane)if(active&(1U<<lane))for(unsigned b=0;b<size;++b){unsigned at=off+lane*size+b;u.lbe|=1ULL<<at;
                        if(store)u.lwdata[at>>3]|=((u64(vd[lane])>>(8*b))&255)<<((at&7)*8);}}}
            if(!active)break;             // nothing to do: the RTL still passes through the LSU? (tnca_lsu.v latches `active`; no beats, completes)
            std::map<u64,size_t> words;
            for(unsigned lane=0;lane<(vector?16U:1U);++lane){
                if(!(active&(1<<lane)))continue;
                u64 address=base+(vector?(gather?u64(std::int64_t(std::int32_t(vb[lane])))*size:lane*size):0);
                x.c.address(address,size);u64 value=vector?vd[lane]:rS(rc,dd);unsigned done=0;
                while(done<size){
                    u64 ap=address+done;unsigned off=unsigned(ap&7),n=std::min(size-done,8-off);u64 wa=ap&~7ULL;
                    auto it=words.find(wa);size_t bi;
                    if(it==words.end()){bi=u.beats.size();words[wa]=bi;Beat nb;nb.addr=wa;nb.store=store;u.beats.push_back(nb);}else bi=it->second;
                    Beat& beat=u.beats[bi];beat.parts.push_back({lane,off,n,done*8});
                    for(unsigned byte=0;byte<n;++byte){unsigned at=off+byte;beat.be|=1U<<at;beat.data=(beat.data&~(255ULL<<(at*8)))|(((value>>((done+byte)*8))&255)<<(at*8));}
                    done+=n;
                }
            }
            e.mem_go=true;break;}
        case K::Fence:break;
        case K::Vint:{
            bool cmp=func==24 || func==56,immediate=func>=32 && func<=63;
            PhysicalRegisters::Vector va{},vbv{},out{};
            if(!cmp)out=rV(rc,dd);
            if(func!=64 && func!=27 && func!=26)va=rV(rc,a);
            if(!immediate && func!=0 && func!=20 && func!=21 && func!=26 && func!=27 && func!=64)vbv=rV(rc,b);
            unsigned comparison=0;u64 scalar=func==26?rS(rc,a):0;
            for(unsigned lane=0;lane<16;++lane)if(func==25 || (mask&(1<<lane))){
                u64 value=0;
                if(func==27)value=lane;else if(func==26)value=scalar;else if(func==64)value=in.simm;
                else if(func==25)value=(mask&(1<<lane))?va[lane]:vbv[lane];
                else{u64 right=immediate?(zero_extended_immediate(func)?in.imm:in.simm):vbv[lane];value=alu(immediate?immediate_alu(func):func,va[lane],right,32,mod&15);}
                if(cmp){if(value)comparison|=1<<lane;}else out[lane]=std::uint32_t(value);
            }
            if(cmp){wP(dd,comparison);}else if(is_vlate(in)){defer(2,dd,3,0,func==25?65535:mask,&out);}else {wV(dd,func==25?65535:mask,out);}break;}
        case K::Vfp:{
            PhysicalRegisters::Vector va=rV(rc,a),vbv{},vc{},out{};if(func!=24)out=rV(rc,dd);
            if(func!=20 && func!=21)vbv=rV(rc,b);
            if(func==6 || func==7)vc=rV(rc,cc);
            unsigned comparison=0;
            for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                if(func==24){if(tnca_vector_ops::vfp_compare(mod,va[lane],vbv[lane]))comparison|=1<<lane;}
                else out[lane]=tnca_vector_ops::vfp(func,mod,va[lane],vbv[lane],(func==6 || func==7)?vc[lane]:0);
            }
            if(func==24){defer(3,dd,1,comparison,0,nullptr);}else {defer(2,dd,1,0,mask,&out);}break;}
        case K::Cvt:{PhysicalRegisters::Vector va=rV(rc,a),out=rV(rc,dd);
            for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))out[lane]=tnca_vector_ops::cvt(func,mod,va[lane]);
            defer(2,dd,1,0,mask,&out);break;}
        case K::Pred:{
            unsigned pa=0,pb=0;if(func!=19)pa=rP(rc,a);
            if(func==1 || func==2 || func==3 || func==5)pb=rP(rc,b);
            if(func>=16){
                if(func==16)wS(dd,pa!=0);else if(func==17)wS(dd,pa==65535);else if(func==18)wS(dd,pa);
                else if(func==19)wP(dd,unsigned(rS(rc,a)&65535));else if(func==20)wS(dd,unsigned(__builtin_popcount(pa)));
                else throw std::runtime_error("reserved predicate function");
            }else{require(func<=5,"reserved predicate function");
                unsigned value=func==0?pa:func==1?pa&pb:func==2?pa|pb:func==3?pa^pb:func==4?~pa:pa&~pb;wP(dd,value);}
            break;}
        case K::Shuffle:{
            PhysicalRegisters::Vector source=rV(rc,a),out=rV(rc,dd),indices{};unsigned shift=0;
            if(func==3){indices=rV(rc,b);}else {shift=unsigned(rS(rc,b))&15;}
            for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){unsigned from=lane;
                if(func==0)from=lane^shift;else if(func==1)from=lane>=shift?lane-shift:lane;
                else if(func==2)from=lane+shift<16?lane+shift:lane;else if(func==3){from=indices[lane]&15;}else {from=(lane+shift)&15;}
                out[lane]=source[from];}
            wV(dd,mask,out);break;}
        case K::Extract:{auto va=rV(rc,a);wS(dd,va[mod&15]);break;}
        case K::Insert:{auto out=rV(rc,dd);auto value=std::uint32_t(rS(rc,a));if(mask&(1<<(mod&15))){out[mod&15]=value;}wV(dd,mask&(1<<(mod&15)),out);break;}
        case K::Reduce:{
            auto va=rV(rc,a);std::uint32_t reduced=tnca_vector_ops::reduce(func,va,mask);
            if(func>=8){ // FP reduction: the sequencer [2.13]
                e.red_start=true;e.red_done=t+13;
                if(mod&1){PhysicalRegisters::Vector out;out.fill(reduced);e.redres=Result{0,2,rc,dd,65535,0,0,out};}
                else e.redres=Result{0,1,rc,dd,0,0,reduced,{}};
            }else{
                if(mod&1){PhysicalRegisters::Vector out;out.fill(reduced);if(is_vlate(in)){defer(2,dd,3,0,65535,&out);}else wV(dd,65535,out);}
                else if(is_vlate(in))defer(1,dd,3,reduced,0,nullptr);else wS(dd,reduced);
            }
            break;}
        case K::Sfu:{
            auto va=rV(rc,a),out=rV(rc,dd);
            for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))out[lane]=tnca_sfu::evaluate(func,va[lane]);
            e.red_start=true;e.red_is_sfu=true;e.red_done=t+24;e.redres=Result{0,2,rc,dd,mask,0,0,out};break;}
        case K::Desc:{e.txf_cfg=true;e.cfg_func=func;e.cfg_slot=Core::sliced(f,dd,4);e.cfg_addr=func==0?rS(rc,a):0;e.cfg_imm=in.imm;e.txf_cmd=true;break;}
        case K::Xfer:{
            u64 token=c.token_of(f,rS(rc,cc));c.verify_async_token(token);
            CycleTensor::Cmd cm;cm.store=func==8;cm.transpose=(mod&32)!=0;cm.active=mask!=0;cm.bar_en=true;cm.desc=Core::sliced(f,a,4);cm.win=Core::sliced(f,dd,4);
            cm.bytes=1U<<(mod&7);cm.rctx=rc;cm.role=f.role;cm.token=token;cm.pc=xb.pc;cm.sb=b;
            for(unsigned dim=0;dim<5;++dim)cm.coords[dim]=(b+dim<64)?rS(rc,b+dim):0;
            e.txf_issue=true;e.txf_cmd_v=cm;e.txf_cmd=true;break;}
        case K::Barrier:{
            if(func==0){e.bar_init=true;e.bar=Core::sliced(f,dd,5);c.barriers[e.bar].reset();}
            else if(func==1){unsigned index=Core::sliced(f,a,5);wS(dd,c.barriers[index].arm(index,b,cc));}
            else if(func==2){c.barriers[Core::sliced(f,a,5)].arrive();}
            else if(func==3 || func==4){
                u64 token=c.token_of(f,rS(rc,a));unsigned index=unsigned(token&65535);
                bool done=c.barriers[index].test(token);
                if(func==3)wS(dd,done);
                else if(!done){e.sleep=true;e.redirect=true;ctx[rc].slp_token=token;}
            }else if(func==5)c.barriers[Core::sliced(f,a,5)].invalidate();
            else throw std::runtime_error("reserved barrier function");
            break;}
        case K::MZero:case K::AZero:{unsigned slot=Core::sliced(f,dd,4);c.acc[slot]={};e.mtx_cmd=true;e.mtx_slot=slot;e.mtx_busy=17;e.mcmd.op=0;e.mcmd.slot=slot;e.mcmd.rctx=rc;break;}
        case K::Mma:{
            const bool dp=func==2;   // m.mma.dp.async (ISA 9C.4): B in its own format, mod[6:5]
            if(dp && !c.mdp)throw ArchFault(trapcode::illegal,"ILLEGAL_INSTRUCTION: m.mma.dp.async without CAPABILITY0.MATRIX_DP");
            unsigned slot=Core::sliced(f,dd,4),K=16U<<(mod&3),dt=(mod>>2)&3,bytes=tnca_matrix_ops::element_bytes(dt),dtb=dp?(mod>>5)&3:dt;
            bool source_free=mod&16;
            u64 token=c.token_of(f,rS(rc,cc)),sf_token=source_free?c.token_of(f,rS(rc,(cc+1)&63)):0;
            unsigned a_regs=tnca_matrix_ops::fragment_registers(16,K,dt),b_regs=tnca_matrix_ops::fragment_registers(K,16,dtb);
            c.verify_async_token(token);if(source_free)c.verify_async_token(sf_token);
            for(unsigned rr=0;rr<a_regs;++rr){unsigned p=f.vbase+a+rr;require(p<256,"MMA A fragment beyond the physical file");}
            for(unsigned rr=0;rr<b_regs;++rr){unsigned p=f.vbase+b+rr;require(p<256,"MMA B fragment beyond the physical file");}
            (void)bytes;
            e.mtx_cmd=true;e.mtx_slot=slot;e.token=token;e.bar_en=true;
            e.mcmd.op=1;e.mcmd.slot=slot;e.mcmd.va=a;e.mcmd.vb=b;e.mcmd.K=K;e.mcmd.dt=dt;e.mcmd.rctx=rc;e.mcmd.bar_en=true;e.mcmd.token=token;
            e.mcmd.sf_en=source_free;e.mcmd.sf_token=sf_token;e.mcmd.dp=dp;e.mcmd.bdt=dtb;
            break;}
        case K::ARow:{
            u64 row=rS(rc,a)&15;std::uint32_t scale=0;PhysicalRegisters::Vector vbv{};
            if(func==2)scale=std::uint32_t(rS(rc,b));else{vbv=rV(rc,b);if(func==3)scale=std::uint32_t(rS(rc,cc));}
            unsigned slot=Core::sliced(f,dd,4);
            tnca_matrix_ops::row_op(c.acc[slot][row],func,scale,func==2?PhysicalRegisters::Vector{}:vbv,mask);
            e.mtx_cmd=true;e.mtx_slot=slot;e.mtx_busy=func==1?2:7;e.mcmd.op=2;e.mcmd.slot=slot;e.mcmd.rctx=rc;e.mcmd.rowop=func==1?0:1;break;}
        case K::AVread:{
            u64 row=rS(rc,b)&15;auto out=rV(rc,dd);auto& rowv=c.acc[Core::sliced(f,a,4)][row];
            for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))out[lane]=rowv[lane];
            wV(dd,mask,out);break;}
        default:throw std::runtime_error("MODEL_REFUSED uncovered instruction kind "+std::string(kind_name(in.kind)));
        }
    }
    //------------------------------------------------------------ the memory port
    // One beat on the core port at cycle `at`; returns the response cycle.
    // One beat on the port at cycle `at`: the functional access now, the
    // response cycle by the memory timing in force. Returns the beat's id
    // (the trace's tag: bit 4 marks the transfer engine's beats).
    u64 issue_beat(u64 addr,bool store,u64 data,unsigned be,u64 at,bool replayed,u64 latency,bool engine,Target target){
        const u64 id=(next_id++<<5)|(engine?16:0);targets[id]=target;
        u64 got=0,when=0;
        if(uncore || replayed){got=c.tm->access(addr,store,data,store?be:255);}
        else when=c.tm->word(c.id,addr,store,data,store?be:255,at,got);
        if(replayed)when=at+latency;
        if(c.core_trace){
            if(hybrid){   // tnca_word_line_adapter.v: the word keeps its address, its bytes take their place in the line
                std::array<u64,8> wd{},line{};const unsigned w=unsigned((addr&63)>>3);if(store)wd[w]=data;
                c.core_trace.QW(at,store,addr,store?u64(be&255)<<(addr&63):0,id,wd);
                if(!store){const u64 base=addr&~63ULL;for(unsigned k=0;k<8;++k)line[k]=c.tm->access(base+8*k,false,0,0);}
                wide_rsp[id]=line;}
            else c.core_trace.Q(at,store,addr,store?be:0U,id,data);}
        // the uncore sees a hybrid word's bytes at their place in the line (tnca_word_line_adapter.v)
        if(uncore && !replayed)uncore_request(id,at,addr,store,hybrid?u64(be&255)<<(addr&63):u64(be),got);
        else schedule(id,when,got);
        return id;
    }
    // One line request of the hybrid's line path at cycle `at` [3.6]: the functional access of its
    // words now (a store writes its enabled bytes; a load reads the whole line), the response by the
    // memory timing in force.
    void issue_line(unsigned k,u64 at,bool replayed,u64 latency){
        Line& s=lines[k];const u64 id=next_id++<<5;targets[id]=Target{5,k,0};s.issued=true;s.rsp=0;
        u64 when=0;
        if(replayed || uncore){   // the cycle uncore times it (uncore_request below); the data are functional now
            for(unsigned w=0;w<8;++w){unsigned be=unsigned((s.be>>(8*w))&255);
                if(s.store){if(be)c.tm->access(s.addr+8*w,true,s.wdata[w],be);}else s.rdata[w]=c.tm->access(s.addr+8*w,false,0,255);}
            when=at+latency;
        }else{
            require(x.c.get_or("tl_core_beat_bytes",8)==64,"line requests on the timeline need tl_core_beat_bytes=64");
            std::vector<TimelineMemory::WideWord> words;
            for(unsigned w=0;w<8;++w){unsigned be=unsigned((s.be>>(8*w))&255);if(!s.store || be)words.push_back({s.addr+8*w,s.wdata[w],s.store?be:255U});}
            when=c.tm->wide(c.id,words,s.store,at);
            if(!s.store)for(auto& w:words)s.rdata[unsigned((w.address&63)>>3)]=w.result;
        }
        if(c.core_trace){c.core_trace.QW(at,s.store,s.addr,s.be,id,s.store?s.wdata:std::array<u64,8>{});wide_rsp[id]=s.store?std::array<u64,8>{}:s.rdata;}
        if(uncore && !replayed)uncore_request(id,at,s.addr,s.store,s.be,0);
        else schedule(id,when,0);
    }
    // One line request of the transfer engine (R41 G4 LINE_ENGINE) at cycle `at` [4.4]: the functional
    // access now (a store writes its enabled bytes, a load's line lands in its slot or the TensorMap buffer),
    // the response by the memory timing in force.
    u64 issue_engine_line(const CycleTensor::Req& r,u64 at,bool replayed,u64 latency,Target target){
        const u64 id=(next_id++<<5)|16;targets[id]=target;
        std::array<u64,8> rd{};u64 when=0;
        if(replayed || uncore){   // the cycle uncore times it (uncore_request below); the data are functional now
            for(unsigned w=0;w<8;++w){unsigned be=unsigned((r.lbe>>(8*w))&255);
                if(r.we){if(be)c.tm->access(r.addr+8*w,true,r.ldata[w],be);}else rd[w]=c.tm->access(r.addr+8*w,false,0,255);}
            when=at+latency;
        }else{
            require(x.c.get_or("tl_core_beat_bytes",8)==64,"line requests on the timeline need tl_core_beat_bytes=64");
            std::vector<TimelineMemory::WideWord> words;
            for(unsigned w=0;w<8;++w){unsigned be=unsigned((r.lbe>>(8*w))&255);if(!r.we || be)words.push_back({r.addr+8*w,r.ldata[w],r.we?be:255U});}
            when=c.tm->wide(c.id,words,r.we,at);
            if(!r.we)for(auto& w:words)rd[unsigned((w.address&63)>>3)]=w.result;
        }
        if(!r.we)txf.line_data(target,rd);
        if(c.core_trace){c.core_trace.QW(at,r.we,r.addr,r.lbe,id,r.we?r.ldata:std::array<u64,8>{});wide_rsp[id]=r.we?std::array<u64,8>{}:rd;}
        if(uncore && !replayed)uncore_request(id,at,r.addr,r.we,r.lbe,0);
        else schedule(id,when,0);
        return id;
    }
    // A completion's writeback: the word unit's (lanes assembled from its beats) or a line slot's.
    Result unit_result(const Unit& u,bool& has){
        has=!u.store;if(u.store)return Result{};   // a load always takes the port: with no active lane it writes nothing but releases its scoreboard bit [2.12]
        if(u.vector){PhysicalRegisters::Vector out=rV(u.rctx,u.dst);std::array<u64,16> values{};
            for(auto& beat:u.beats)for(auto& part:beat.parts){u64 pm=part.bytes==8?UINT64_MAX:((1ULL<<(part.bytes*8))-1);values[part.lane]|=((beat.rdata>>(part.offset*8))&pm)<<part.shift;}
            for(unsigned lane=0;lane<16;++lane)if(u.active&(1<<lane)){u64 v=values[lane];if(u.sign && u.size<8 && (v&(1ULL<<(u.size*8-1)))){v|=UINT64_MAX<<(u.size*8);}out[lane]=std::uint32_t(v);}
            return Result{t,2,u.rctx,u.dst,u.active,0,0,out};}
        u64 v=0;for(auto& beat:u.beats)for(auto& part:beat.parts){u64 pm=part.bytes==8?UINT64_MAX:((1ULL<<(part.bytes*8))-1);v|=((beat.rdata>>(part.offset*8))&pm)<<part.shift;}
        if(u.sign && u.size<8 && (v&(1ULL<<(u.size*8-1))))v|=UINT64_MAX<<(u.size*8);
        return Result{t,1,u.rctx,u.dst,0,0,v,{}};
    }
    Result line_result(const Line& s,bool& has){   // tnca_lsu_line.v cpl_vdata: the landing shifted by the offset, lanes extended
        has=!s.store;if(s.store)return Result{};
        PhysicalRegisters::Vector out=rV(s.rctx,s.dst);
        for(unsigned lane=0;lane<16;++lane)if(s.active&(1U<<lane)){u64 v=0;
            for(unsigned b=0;b<s.size;++b){unsigned at=s.off+lane*s.size+b;v|=((s.rdata[at>>3]>>((at&7)*8))&255)<<(8*b);}
            if(s.sign && s.size<4 && ((v>>(s.size*8-1))&1))v|=UINT64_MAX<<(s.size*8);
            out[lane]=std::uint32_t(v);}
        return Result{t,2,s.rctx,s.dst,s.active,0,0,out};
    }
    void uncore_request(u64 id,u64 at,u64 addr,bool store,u64 be,u64 got);   // defined after CycleUncore
    bool uncore_ready()const;
    void uncore_launch(u64 pc,unsigned roles);
    void port_issue(Beat& beat,unsigned unit,unsigned index,u64 at,bool replayed,u64 latency){
        beat.rsp=0;beat.issued=true;issue_beat(beat.addr,beat.store,beat.data,beat.be,at,replayed,latency,false,Target{0,unit,index});
    }
    // Replay: may this beat issue now? (the RTL's memory accepted it no earlier than recorded)
    bool replay_allows(u64& latency,bool& replayed){u64 earliest=0;latency=0;replayed=false;
        if(!replay){return true;}
        replayed=replay->peek(earliest,latency);
        return !(replayed && launch_cycle+earliest>t);}
    //------------------------------------------------------------ one cycle
    bool any_running()const{for(auto& k:ctx){if(k.running)return true;}return false;}
    bool fpu_busy()const{return fpu_busy_until>0 && t>fpu_start && t<=fpu_busy_until;}
    bool fpu_done()const{return fpu_done_at==t;}
    bool mtx_busy()const{return mtx.busy();}
    bool txf_busy()const{return txf.busy();}
    // hybrid [3.6]: lsu_cpl_valid is the completion register; lsu_ready_load is the command register
    // free (tnca_core.v g_lsu_hybrid: !pending); lsu_req_ready adds the hybrid idle (both paths quiet,
    // no completion held).
    bool hybrid_idle()const{return nlines==0 && !fq_valid && units[0].st==0 && !out_valid;}
    bool lsu_cpl_valid()const{return hybrid?out_valid:(units[0].cpl || units[1].cpl);}
    bool lsu_req_ready()const{return hybrid?(!cmd_pending && hybrid_idle()):(units[0].st==0 && units[1].st==0);}
    bool lsu_ready_load()const{
        if(hybrid)return !cmd_pending;
        if(lsu_req_ready())return true;
        if(nunits<2)return false;
        for(unsigned i=0;i<2;++i)if(units[i].st==0 && (units[1-i].st==3 || units[1-i].st==4) && !units[1-i].amo)return true;
        return false;}
    bool rctx_wr_busy(unsigned rc)const{for(unsigned i=1;i<64;++i){if(sb_s[rc][i])return true;}for(unsigned i=0;i<64;++i){if(sb_v[rc][i])return true;}for(unsigned i=1;i<16;++i){if(sb_p[rc][i])return true;}return false;}
    bool core_idle()const{
        return !any_running() && !f1.valid && !fb.valid && !d.valid && !r.valid && !xb.valid && lsu_req_ready() && !lsu_cpl_valid()
            && !fpu_busy() && !red_pend && !w.en && !mtx_busy() && !txf.busy_any() && wb_rsv==0;}   // busy_any: line commands in flight [4.4]

    // The hybrid LSU's edge [3.6]. Every decision reads the pre-edge state; `out_ack` is the writeback
    // port taking the completion register (ws_lsu), `fq_fired` / `sq_popped` the port taking the line
    // path's request register / the word queue's head this cycle, `x_cmd` an access in X.
    void hybrid_edge(bool out_ack,bool fq_fired,bool sq_popped,bool x_cmd,const Unit& xcmd){
        Unit& u=units[0];
        // pre-edge wires (tnca_lsu_hybrid.v)
        const bool slow_idle=u.st==0,fast_quiet=nlines==0 && !fq_valid;
        const bool fast_mode=slow_idle && !(out_valid && !out_fast),slow_mode=fast_quiet && !(out_valid && out_fast);
        int free_tag=-1;for(unsigned k=0;k<lines.size();++k)if(!lines[k].valid){free_tag=int(k);break;}   // lowest free slot
        const bool fast_ready=free_tag>=0 && (!fq_valid || fq_fired);
        const bool consume=cmd_pending && (cmd.line_ok?(fast_mode && fast_ready):(slow_mode && slow_idle));
        int pick=-1;   // tnca_lsu_line.v: a held pick, else round robin over the answered slots from rr
        if(held)pick=int(held_pick);
        else for(unsigned n=0;n<lines.size();++n){unsigned k=(line_rr+n)%unsigned(lines.size());if(lines[k].done){pick=int(k);break;}}
        const bool fv=pick>=0,sv=u.cpl,buffer_ready=!out_valid || out_ack;
        const bool have_slot=u.outstanding<NBEAT,q_full=u.q.size()>=2;
        // the word unit (tnca_lsu.v, REQ_REG): responses free beat slots at this edge
        bool rsp_now=false;
        for(auto& beat:u.beats)if(beat.issued && !beat.done && beat.rsp && beat.rsp<=t){beat.done=true;--u.outstanding;if(beat.rsp==t)rsp_now=true;}
        switch(u.st){
            case 1:u.st=2;break;                                   // SETUP -> RUN
            case 2:{const bool plan=u.next<u.beats.size(),issue_ok=plan && have_slot && !q_full;
                const bool q_empty=u.q.empty() || (u.q.size()==1 && sq_popped);
                if(sq_popped)u.q.erase(u.q.begin());
                if(issue_ok){u.q.push_back(u.next);++u.next;++u.outstanding;}   // the beat slot is reserved at the push
                if(!plan && !issue_ok && q_empty)u.st=3;          // RUN ends on the edge its queue empties
                break;}
            case 3:if(u.outstanding==0 && !rsp_now)u.st=4;break;
            case 4:if(!u.cpl)u.cpl=true;break;                     // cpl_valid is registered in S_DONE
            default:break;
        }
        // the line path's answered slots
        for(auto& s:lines)if(s.valid && s.issued && !s.done && s.rsp && s.rsp<=t)s.done=true;
        // the completion register: the writeback takes it; either path's completion lands when it is free
        if(out_ack)out_valid=false;
        if(buffer_ready && (fv || sv)){out_valid=true;out_fast=fv;out_res=fv?line_result(lines[unsigned(pick)],out_has):unit_result(u,out_has);}
        if(fv){if(buffer_ready){lines[unsigned(pick)]=Line{};--nlines;line_rr=(unsigned(pick)+1)%unsigned(lines.size());held=false;}
               else if(!held){held=true;held_pick=unsigned(pick);}}
        if(sv && buffer_ready && !fv){u.cpl=false;u.st=0;u.beats.clear();u.next=0;u.outstanding=0;u.q.clear();}
        // the line path's request register; the command register's access enters its path
        if(fq_valid && fq_fired)fq_valid=false;
        if(consume){
            if(cmd.line_ok){Line& s=lines[unsigned(free_tag)];s=Line{};++nlines;
                s.valid=true;s.done=cmd.active==0;s.store=cmd.store;s.sign=cmd.sign;s.size=cmd.size;s.active=cmd.active;s.rctx=cmd.rctx;s.dst=cmd.dst;
                s.off=cmd.loff;s.addr=cmd.line;s.be=cmd.lbe;s.wdata=cmd.lwdata;
                if(cmd.active){fq_valid=true;fq_tag=unsigned(free_tag);}}
            else{require(u.st==0,"hybrid word path accepted a request while busy");u=cmd;u.st=1;u.cpl=false;u.q.clear();u.next=0;u.outstanding=0;}
            cmd_pending=false;}
        if(x_cmd){require(!cmd_pending,"hybrid command landing register overflow");cmd_pending=true;cmd=xcmd;}
    }
    void step(){
        t=x.now();++*c_cycles;
        deliveries();
        //==================== wires from the pre-edge state
        // X decode flags
        const Inst* xin=xb.valid?xb.in:nullptr;
        bool x_has_dest=false,x_is_fpnorm=false,x_is_fpmc=false,x_is_fpred=false,x_mtx_cmd=false,x_txf_cmd=false,x_mem=false;
        if(xin){unsigned dc=dest_class(*xin);x_has_dest=(dc==1||dc==2||dc==3) && !x_defers(*xin);x_is_fpnorm=is_fpnorm(*xin);x_is_fpmc=is_fpmc(*xin);
            x_is_fpred=xin->kind==K::Reduce && xin->func>=8;x_mtx_cmd=is_matrix_fu(*xin) && xin->kind!=K::AVread;x_txf_cmd=xin->kind==K::Desc || xin->kind==K::Xfer;x_mem=is_mem(*xin);}
        // scheduled writebacks this cycle [2.12]
        int fp_wb=-1,mul_wb=-1;
        int vl_wb=-1;
        for(size_t i=0;i<sched.size();++i)if(sched[i].port==t){if(sched[i].src==1){fp_wb=int(i);}else if(sched[i].src==2){mul_wb=int(i);}else {vl_wb=int(i);}}
        bool fp_inflight=false;for(auto& s:sched)if(s.src==1)fp_inflight=true;
        bool sched_wb=x_has_dest || fp_wb>=0 || mul_wb>=0 || vl_wb>=0;
        bool wb_steal=(lsu_cpl_valid() || red_pend) && sched_wb;
        bool mtx_done=mtx.done_at==t;bool fpu_done_now=fpu_done();const bool mtx_busy_now=mtx_busy(),txf_busy_now=txf_busy();
        // ---- R hazards and stalls [2.7][2.8]
        bool hazard=false,stall_lsu=false,stall_fpu=false,stall_mtx=false,stall_txf=false,stall_fence=false,stall_arrive=false,stall_rsv=false,r_vsrc_haz=false;
        unsigned r_dc=0;Srcs rs{};
        if(r.valid){
            const Inst& in=*r.in;unsigned rc=r.rctx;rs=sources(in);r_dc=dest_class(in);
            auto x_prod=[&](unsigned cls,unsigned idx){return x_has_dest && dest_class(*xin)==cls && xb.rctx==rc && xin->d==idx;};
            for(unsigned i=0;i<rs.ns;++i){unsigned idx=rs.s[i];if(idx && idx<64 && sb_s[rc][idx] && !(wb_hit(1,rc,idx)||x_prod(1,idx)))hazard=true;}
            for(unsigned i=0;i<rs.nv;++i){unsigned idx=rs.v[i];if(idx<64 && sb_v[rc][idx] && !wb_hit(2,rc,idx)){hazard=true;r_vsrc_haz=true;}}
            for(unsigned i=0;i<rs.np;++i){unsigned idx=rs.p[i];if(idx && sb_p[rc][idx] && !(wb_hit(3,rc,idx)||x_prod(3,idx)))hazard=true;}
            if(r_dc==1 && in.d && sb_s[rc][in.d] && !wb_hit(1,rc,in.d))hazard=true;        // hz_d (WAW)
            if(r_dc==2 && sb_v[rc][in.d] && !wb_hit(2,rc,in.d))hazard=true;
            if(r_dc==3 && in.d && sb_p[rc][in.d] && !wb_hit(3,rc,in.d))hazard=true;
            if(r_dc==4){unsigned slot=Core::sliced(F(rc),in.d,4);if(sb_acc[slot] && !(mtx_done && mtx.done_slot==slot))hazard=true;}
            if(in.kind==K::AVread){unsigned slot=Core::sliced(F(rc),in.a,4);if(sb_acc[slot] && !(mtx_done && mtx.done_slot==slot))hazard=true;}
            if(is_mem(in))stall_lsu=!((in.kind==K::Atomic?lsu_req_ready():lsu_ready_load()) && !(x_mem && xb.valid));
            bool fold_next=(fpu_busy() && fpu_red) || x_is_fpred;
            bool red_busy_next=fpu_busy() || red_pend || fpu_done_now || x_is_fpmc;
            if(is_fpnorm(in) && fold_next)stall_fpu=true;
            if(in.kind==K::Reduce && in.func>=8 && (red_busy_next || fp_inflight || x_is_fpnorm))stall_fpu=true;
            if(in.kind==K::Sfu && red_busy_next)stall_fpu=true;
            if(is_matrix_fu(in) && (mtx_busy_now || x_mtx_cmd))stall_mtx=true;
            if(is_txfer_fu(in) && (txf_busy_now || x_txf_cmd))stall_txf=true;
            // hybrid: an access in X has not reached the command register yet, so it counts as issued [3.6]
            const bool x_claim=hybrid && x_mem && xb.valid;
            if(in.kind==K::Fence && (!lsu_req_ready() || lsu_cpl_valid() || rctx_wr_busy(rc) || x_claim))stall_fence=true;
            if(in.kind==K::Barrier && in.func==2){
                bool arr_mtx=(mtx_busy_now && mtx_rctx==rc) || (x_mtx_cmd && xb.rctx==rc);
                bool arr_txf=(txf_busy_now && txf_rctx==rc) || txf.inflight(rc) || (x_txf_cmd && xb.rctx==rc);   // inflight: R41 G4 line commands of this context
                if(rctx_wr_busy(rc) || !lsu_req_ready() || x_claim || arr_mtx || arr_txf)stall_arrive=true;}
            if(books_wb(in) && (wb_rsv>>(wb_delay(in)+1))&1)stall_rsv=true;
        }
        // ---- operand fetch [2.9]
        bool of_enable=r.valid && !r_vsrc_haz;unsigned of_need=0;unsigned need_idx[3]={0,0,0};
        if(of_enable)for(unsigned i=0;i<rs.nv;++i){of_need|=1U<<rs.vp[i];need_idx[rs.vp[i]]=rs.v[i];}
        unsigned of_arrive=0;for(auto& br:bank_rsp)if(br.valid && br.port<3)of_arrive|=1U<<br.port;
        of_arrive&=~(pf_tag&~(adv_q?7U:0U));
        unsigned of_have=of_got|of_arrive,of_want=of_need&~of_have;
        bool of_ready=(of_need&~of_have)==0,vrf_stall=of_enable && !of_ready;
        bool r_of_idle=of_got==0 && of_arrive==0 && of_want==0;
        // D prefetch [2.6]
        unsigned pf_need=0,pf_idx[3]={0,0,0};bool pf_haz=false;
        if(d.valid){Srcs ds=sources(*d.in);for(unsigned i=0;i<ds.nv;++i){pf_need|=1U<<ds.vp[i];pf_idx[ds.vp[i]]=ds.v[i];
                unsigned idx=ds.v[i];bool haz=(sb_v[d.rctx][idx] && !wb_hit(2,d.rctx,idx)) || (r.valid && r_dc==2 && r.rctx==d.rctx && r.in->d==idx);if(haz)pf_haz=true;}}
        bool pf_go=d.valid && pf_need && of_want==0 && !(r.valid && r_vsrc_haz) && !pf_haz;
        // ---- bank arbiter: ports 0..2 (3 and 4 not modelled) [2.9]
        unsigned req_mask=pf_go?pf_need:of_want;unsigned req_rctx=pf_go?d.rctx:r.rctx;const unsigned* req_idx=pf_go?pf_idx:need_idx;
        unsigned p4_idx=0;bool p4_req=txf.vr_req(p4_idx) && !txf.sread_pending;unsigned p4_phys=p4_req?phys(txf.cmd.rctx,p4_idx):0;   // rq_v[4] = vr_req && !vr_valid
        unsigned p3_idx=mtx.read_index();bool p3_req=p3_idx!=~0u;unsigned p3_phys=p3_req?phys(mtx.cmd.rctx,p3_idx):0;
        // last cycle's port-3 read answers now
        bool p3_raw=false;PhysicalRegisters::Vector p3_raw_data{};
        for(auto& br:bank_rsp)if(br.valid && br.port==3){p3_raw=true;p3_raw_data=F(mtx.cmd.rctx).physical->read(br.phys,sc_time_stamp());}
        const bool p3_data_now=mtx_vrf_reg?p3q_v:p3_raw;const PhysicalRegisters::Vector p3_data=mtx_vrf_reg?p3q_data:p3_raw_data;
        unsigned accepted=0;bool bank_taken[4]={};int first_served=-1;
        for(unsigned k=0;k<NPORT;++k){unsigned p=(vrf_rr+k)%NPORT;
            unsigned bk;
            if(p<3){if(!((req_mask>>p)&1)){continue;}bk=phys(req_rctx,req_idx[p])%4;}
            else if(p==3){if(!p3_req){continue;}bk=p3_phys%4;}
            else{if(!p4_req){continue;}bk=p4_phys%4;}
            if(bank_taken[bk]){continue;}
            bank_taken[bk]=true;accepted|=1U<<p;
            if(first_served<0){first_served=int(p);}}
        if(vdbg_core==long(c.id) && vdbg_launch+1==long(launch_n) && t-launch_cycle>=vdbg_from && t-launch_cycle<=vdbg_to){
            vdbg<<(t-launch_cycle)<<" rr="<<vrf_rr<<" req=";
            for(unsigned p=0;p<3;++p){if((req_mask>>p)&1){vdbg<<" p"<<p<<":"<<phys(req_rctx,req_idx[p])<<"/b"<<phys(req_rctx,req_idx[p])%4;}}
            if(p3_req){vdbg<<" p3:"<<p3_phys<<"/b"<<p3_phys%4;}if(p4_req){vdbg<<" p4:"<<p4_phys<<"/b"<<p4_phys%4;}
            vdbg<<" acc="<<accepted<<" first="<<first_served<<" pf_go="<<pf_go<<" mtx.st="<<mtx.st<<" kk="<<mtx.kk<<" mm="<<mtx.mm<<" la="<<mtx.la<<" pend="<<mtx.pending
                <<" r="<<r.valid<<"/"<<std::hex<<(r.valid?r.pc:0)<<" d="<<d.valid<<"/"<<(d.valid?d.pc:0)<<std::dec<<"\n";}
        // ---- stall summary [2.1]
        bool stall_other=hazard||stall_lsu||stall_fpu||stall_mtx||stall_txf||stall_fence||stall_arrive||stall_rsv||wb_steal;
        bool stall_raw=stall_other||vrf_stall;
        bool rp_go=r.valid && r_of_idle && (hazard||stall_mtx||stall_txf) && d.valid && d.rctx!=r.rctx;
        // ---- X executes [2.11]
        XEffects e;
        if(xb.valid){
            try{execute(e);}
            catch(const ArchFault& ex){e.trap=true;e.redirect=true;e.code=ex.code;e.why=ex.what();}
        }
        // x_redirect blocks rp_go [2.1]; R41 F01 (core_switch_opcode=1): X's opcode does -- any instruction that may
        // redirect (branch, exit, trap, atomic, mb.wait), whether or not it does
        if(rp_go && (switch_opcode?(xin && (xin->kind==K::Branch || xin->kind==K::Exit || xin->kind==K::Trap || xin->kind==K::Atomic
                                            || (xin->kind==K::Barrier && xin->func==4))):e.redirect))rp_go=false;
        bool stall_front=stall_raw && !rp_go;
        if(any_running()){if(stall_raw){++*c_stall;}if(hazard)++*c_hazard;else if(stall_lsu)++*c_lsu;else if(stall_fpu)++*c_fpu;else if(stall_mtx)++*c_mtx;else if(stall_txf)++*c_txf;
            else if(stall_fence)++*c_fence;else if(stall_arrive)++*c_arrive;else if(stall_rsv)++*c_rsv;else if(wb_steal)++*c_steal;else if(vrf_stall)++*c_vrf;}
        // ---- F selection [2.2]
        int f_sel=-1;
        for(unsigned k=0;k<4;++k){unsigned i=(rr_ptr+k)%4;const Ctx& q=ctx[i];if(q.running && !q.faulted && !q.blocked && !q.parked){f_sel=int(i);break;}}
        bool f_hit=f_sel>=0;
        bool f_issue=f_hit && !(stall_front && (fb.valid||f1.valid));
        unsigned xr=xb.rctx,rr=r.rctx;
        bool kill_f=(e.redirect && f_hit && unsigned(f_sel)==xr) || (rp_go && f_hit && unsigned(f_sel)==rr);
        bool kill_f1=(e.redirect && f1.valid && f1.rctx==xr) || (rp_go && f1.valid && f1.rctx==rr);
        bool kill_fb=(e.redirect && fb.valid && fb.rctx==xr) || (rp_go && fb.valid && fb.rctx==rr);
        bool kill_av=fb.valid?kill_fb:kill_f1;
        bool kill_d=e.redirect && d.valid && d.rctx==xr;
        bool kill_r=(e.redirect && r.valid && r.rctx==xr) || rp_go;
        bool issue=!stall_front && r.valid && !kill_r;
        // ---- writeback port arbitration [2.12]
        bool have_nw=false;Result nw;int cpl_unit=-1;bool red_grant=false,out_ack=false;
        if(sched_wb){have_nw=true;if(fp_wb>=0)nw=sched[fp_wb];else if(mul_wb>=0){nw=sched[mul_wb];}else if(vl_wb>=0){nw=sched[vl_wb];}else {nw=e.single;}}   // ws_fp, ws_mul, ws_vl, ws_x
        else if(hybrid && out_valid){out_ack=true;if(out_has){have_nw=true;nw=out_res;nw.port=t;}}   // ws_lsu: the completion register [3.6]
        else if(lsu_cpl_valid()){cpl_unit=units[old].cpl?int(old):int(1-old);bool has;Result res=unit_result(units[cpl_unit],has);
            if(has){have_nw=true;nw=res;}}
        else if(red_pend){have_nw=true;nw=red;red_grant=true;}
        // ---- LSU beats on the port this cycle [3.2][3.3]: one request, engine priority not modelled
        bool port_used=false;
        CycleTensor::Req er=txf.request();bool efired=false;u64 eid=0;
        const bool mem_ready=!uncore || uncore_ready();   // mem_req_ready from the NoC [UNCORE 1.3]
        bool fq_fired=false,sq_popped=false;
        auto engine_go=[&]{u64 latency;bool replayed;
            if(mem_ready && replay_allows(latency,replayed)){if(replay)replay->advance(er.we,er.addr);
                eid=er.line?issue_engine_line(er,t,replayed,latency,txf.request_target(er)):issue_beat(er.addr,er.we,er.data,er.be,t,replayed,latency,true,txf.request_target(er));
                efired=true;port_used=true;}};
        if(hybrid){
            // tnca_mem_mux.v: with TENSOR_LINE the LSU first (A_FIRST), else the LSU only when the engine has
            // no request; inside the hybrid the line path's request register before the word queue's head [3.6]
            Unit& u=units[0];const bool lsu_req=fq_valid || !u.q.empty();
            if(lsu_req && (lsu_first || !er.valid)){u64 latency;bool replayed;
                if(mem_ready && replay_allows(latency,replayed)){
                    if(fq_valid){if(replay)replay->advance(lines[fq_tag].store,lines[fq_tag].addr);issue_line(fq_tag,t,replayed,latency);fq_fired=true;}
                    else{const unsigned b=u.q.front();if(replay)replay->advance(u.beats[b].store,u.beats[b].addr);
                        port_issue(u.beats[b],0,b,t,replayed,latency);u.last_issue=t;sq_popped=true;}
                    port_used=true;}
            }else if(er.valid)engine_go();
        }else if(er.valid)engine_go();
        // tnca_mem_mux.v: the LSU is served only when the engine has no request at all [3.1]
        for(unsigned k=0;k<2 && !hybrid && !port_used && !er.valid && mem_ready;++k){unsigned i=(old+k)%2;Unit& u=units[i];
            if(u.st==2 && u.next<u.beats.size() && u.outstanding<NBEAT){
                u64 latency;bool replayed;
                if(!replay_allows(latency,replayed))continue;   // the RTL's memory accepted it later (back-pressure): wait
                if(replay)replay->advance(u.beats[u.next].store,u.beats[u.next].addr);
                port_issue(u.beats[u.next],i,u.next,t,replayed,latency);++u.outstanding;u.last_issue=t;++u.next;port_used=true;
            }}
        if(xb.valid || port_used)last_progress=t;
        if(t-last_progress>200000){std::ostringstream wr;wr<<"hazard="<<hazard<<" lsu="<<stall_lsu<<" fpu="<<stall_fpu<<" mtx="<<stall_mtx<<" txf="<<stall_txf<<" fence="<<stall_fence<<" arrive="<<stall_arrive<<" rsv="<<stall_rsv<<" steal="<<wb_steal<<" vrf="<<vrf_stall<<" rp_go="<<rp_go<<" f_sel="<<f_sel<<" of_need="<<of_need<<" of_have="<<of_have;
            throw std::runtime_error("MODEL_REFUSED cycle core made no progress for 200000 cycles\n"+snapshot(wr.str().c_str()));}
        //==================== the edge
        // launch is handled in launch(); here the pipeline rules.
        // responses that arrive this cycle: bt_valid cleared at this edge [3.2]
        if(hybrid)hybrid_edge(out_ack,fq_fired,sq_popped,xb.valid && xin->kind==K::Lsu && e.req.st==1,e.req);
        else{
        for(auto& u:units)for(auto& beat:u.beats)if(beat.issued && !beat.done && beat.rsp && beat.rsp<=t){beat.done=true;--u.outstanding;}
        // LSU unit state machines
        for(unsigned i=0;i<2;++i){Unit& u=units[i];
            switch(u.st){
                case 1:u.st=2;break;                                   // SETUP -> RUN at c+1 edge; beats from c+2
                case 2:if(u.next>=u.beats.size() && (u.beats.empty() || u.last_issue<t)){u.st=3;}break; // plan empty: DRAIN
                case 3:{bool rsp_now=false;for(auto& beat:u.beats)if(beat.issued && beat.rsp && beat.rsp==t)rsp_now=true;
                        if(u.outstanding==0 && !rsp_now)u.st=4;
                        break;}
                case 4:if(!u.cpl){u.cpl=true;}break;
                default:break;
            }
        }
        if(cpl_unit>=0 && !sched_wb){Unit& u=units[cpl_unit];u.cpl=false;u.st=0;u.beats.clear();u.next=0;u.outstanding=0;old=1-unsigned(cpl_unit);}
        // the memory request accepted in X [3.2]
        if(e.mem_go || (xb.valid && xin->kind==K::Lsu && !e.req.active && e.req.st==1)){
            unsigned i=units[0].st==0?0:1;require(units[i].st==0 && i<nunits,"LSU accepted a request with no idle unit");
            units[i]=e.req;units[i].st=1;if(units[0].st!=0 && units[1].st!=0){old=1-i;}else {old=i;}
        }
        }
        const bool w_pre_en=w.en;const unsigned w_pre_rctx=w.res.rctx;
        // the engine's register read answered this cycle (port 4), then its edge [4.2]
        for(auto& br:bank_rsp)if(br.valid && br.port==4){txf.vr_response(F(txf.cmd.rctx).physical->read(br.phys,sc_time_stamp(),txf.owner));}
        txf.edge(efired,eid);
        if(dbg_core==long(c.id) && dbg_launch+1==long(launch_n) && t-launch_cycle>=dbg_from && t-launch_cycle<=dbg_to){
            dbg<<(t-launch_cycle)<<" req="<<er.valid<<"/"<<std::hex<<er.addr<<std::dec<<" ready="<<mem_ready<<" fired="<<efired<<" w="<<w_pre_en<<"/"<<w.res.cls<<"/"<<w.res.idx<<" "<<txf.dump()<<"\n";}
        // the command of this cycle's X takes effect now [4.2]: cfg ops at the edge, a transfer or td.load enters its first state
        if(e.txf_cfg){try{txf.config(e.cfg_func,e.cfg_slot,e.cfg_addr,e.cfg_imm);}catch(const ArchFault& ex){Ctx& q=ctx[xb.rctx];q.running=false;q.faulted=true;record(STATUS_FAULT,ex.code,xb.pc,ex.what());}}
        if(e.txf_issue)txf.issue(e.txf_cmd_v);
        if(txf.done_at==t+1 && txf.done_bar){/* the done pulse is high during t+1; counted then */}
        // WB stage: commit w, release its scoreboard bit, then w <= nw
        if(w.en){const Result& R=w.res;Frame& f=F(R.rctx);
            if(R.cls==1){if(R.idx){f.s[R.idx]=R.sdata;}sb_s[R.rctx][R.idx]=false;}
            else if(R.cls==3){if(R.idx){f.p[R.idx]=unsigned(R.sdata)&65535;}sb_p[R.rctx][R.idx]=false;}
            else if(R.cls==2){f.physical->write(f.vbase+R.idx,R.mask,sc_time_stamp(),R.vdata);f.physical->commit(sc_time_stamp());sb_v[R.rctx][R.idx]=false;}
        }
        w.en=have_nw;if(have_nw)w.res=nw;
        if(red_grant)red_pend=false;
        // retire the scheduled pipeline entries that took the port
        if(fp_wb>=0 || mul_wb>=0 || vl_wb>=0){int k=fp_wb>=0?fp_wb:mul_wb>=0?mul_wb:vl_wb;sched.erase(sched.begin()+k);}
        for(auto& dres:e.deferred)sched.push_back(dres);
        // FP sequencer [2.13][2.15]
        if(fpu_done_now && !red_pend){red_pend=true;}       // red_pend from done+1
        if(e.red_start){fpu_red=!e.red_is_sfu;fpu_start=t;fpu_busy_until=e.red_is_sfu?t+23:t+12;fpu_done_at=e.red_done;red=e.redres;red.port=0;}
        // transfer engine completion pulse (high during this cycle) [5.2]
        if(txf.done_at==t){if(txf.done_bar)c.complete_async(txf.done_token,txf.done_err);
            if(txf.done_err){Ctx& q=ctx[txf.cmd.rctx];q.running=false;q.faulted=true;record(STATUS_FAULT,txf.fault_code,txf.cmd.pc,txf.fault_why);}}
        // matrix engine [4.1]
        if(mtx_done){sb_acc[mtx.done_slot]=false;if(mtx.done_bar)c.complete_async(mtx.done_token,false);}
        if(mtx.srcfree_at==t && mtx.srcfree_token){c.complete_async(mtx.srcfree_token,false);mtx.srcfree_token=0;}
        mtx.edge((accepted>>3)&1,p3_data_now,p3_data);
        if(mtx_vrf_reg){p3q_v=p3_raw;if(p3_raw)p3q_data=p3_raw_data;}
        if(e.mtx_cmd){mtx_rctx=xr;mtx.issue(e.mcmd);}
        if(e.txf_cmd)txf_rctx=xr;
        // wakes [2.16]: any WB write to the context, any mtx_done, any engine freed
        bool eng_freed=(mtx_busy_q && !mtx_busy_now) || (txf_busy_q && !txf_busy_now);
        bool wake[4]={};
        for(unsigned i=0;i<4;++i)wake[i]=(w_pre_en && w_pre_rctx==i) || mtx_done || eng_freed;
        mtx_busy_q=mtx_busy_now;txf_busy_q=txf_busy_now;
        // barrier sleepers [5.3]: examined every cycle; wake pulse next cycle -> blocked cleared the cycle after
        for(unsigned i=0;i<4;++i){Ctx& q=ctx[i];
            if(q.slp_v){unsigned index=unsigned(q.slp_token&65535);auto& bm=c.barriers[index];
                if(bm.error){q.slp_v=false;q.pending_wake=t+1;q.wake_err=true;}
                else if(bm.completed>=(q.slp_token>>16)){q.slp_v=false;q.pending_wake=t+1;}}
            if(q.pending_wake==t){q.blocked=false;q.pending_wake=0;if(q.wake_err){q.running=false;q.faulted=true;q.wake_err=false;record(STATUS_FAULT,trapcode::barrier,q.pc,"BARRIER_FAULT: invalidation woke waiter");}}
        }
        // X retire effects [2.11]
        if(xb.valid){Ctx& q=ctx[xr];++q.instret;++*c_instr;
            if(c.core_trace){c.core_trace.I(t,xr,xb.pc);const Inst& in=*xb.in;c.core_trace.K(xb.pc,kind_name(in.kind),in.major,in.func,in.mod,in.d,in.a,in.b,in.c);}
            if(e.exit)q.running=false;
            if(e.trap){q.running=false;q.faulted=true;record(STATUS_FAULT,e.code,xb.pc,e.why);}
            if(e.sleep){q.blocked=true;q.pc=xb.pc;q.slp_v=true;}
            else if(e.redirect && !e.exit && !e.trap)q.pc=e.target;
        }
        // R -> X, scoreboard claims, reservation [2.10]
        Bundle nx;if(!stall_front){nx=r;nx.valid=r.valid && !kill_r;}
        if(issue){const Inst& in=*r.in;unsigned rc=r.rctx;unsigned dc=dest_class(in);
            if(dc==1 && in.d)sb_s[rc][in.d]=true;else if(dc==2)sb_v[rc][in.d]=true;else if(dc==3 && in.d)sb_p[rc][in.d]=true;
            else if(dc==4)sb_acc[Core::sliced(F(rc),in.d,4)]=true;
            wb_rsv=(wb_rsv>>1)|(books_wb(in)?(1U<<wb_delay(in)):0);
        }else wb_rsv>>=1;
        if(rp_go){Ctx& q=ctx[rr];q.pc=r.pc;if(!wake[rr]){q.parked=true;}++*c_park;}
        for(unsigned i=0;i<4;++i)if(wake[i])ctx[i].parked=false;
        // D -> R [2.5]
        Bundle nr=r;if(!stall_front){nr=d;nr.valid=d.valid && !kill_d;}else if(kill_r)nr.valid=false;
        // F -> D [2.4]
        Bundle nd=d;Bundle avail=fb.valid?fb:f1;bool avail_v=fb.valid||f1.valid;
        if(!stall_front){nd=avail;nd.valid=avail_v && !kill_av;if(nd.valid && !nd.in)nd.in=&c.code.at(nd.pc);}
        else if(kill_d)nd.valid=false;
        // FB [2.3]
        Bundle nfb=fb;
        if(!stall_front)nfb.valid=false;
        else if(f1.valid && !fb.valid){nfb=f1;nfb.valid=!kill_f1;if(nfb.valid)nfb.in=&c.code.at(nfb.pc);}
        else if(fb.valid && kill_fb)nfb.valid=false;
        // F [2.2]
        Bundle nf1;nf1.valid=f_issue && !kill_f;
        if(f_issue){Ctx& q=ctx[f_sel];nf1.rctx=unsigned(f_sel);nf1.pc=q.pc;nf1.in=&c.code.at(q.pc);rr_ptr=(unsigned(f_sel)+1)%4;if(!kill_f)q.pc+=8;}
        // operand fetch state [2.9]
        if(!stall_front){of_got=0;}else {of_got=of_have;}
        pf_tag=pf_go?(accepted&pf_need):0;adv_q=!stall_front && d.valid && !kill_d;
        for(auto& br:bank_rsp)br.valid=false;
        for(unsigned p=0;p<3;++p)if((accepted>>p)&1){unsigned bk=phys(req_rctx,req_idx[p])%4;bank_rsp[bk].valid=true;bank_rsp[bk].port=p;}
        if((accepted>>4)&1){unsigned bk=p4_phys%4;bank_rsp[bk].valid=true;bank_rsp[bk].port=4;bank_rsp[bk].phys=p4_phys;txf.sread_pending=true;}
        if((accepted>>3)&1){unsigned bk=p3_phys%4;bank_rsp[bk].valid=true;bank_rsp[bk].port=3;bank_rsp[bk].phys=p3_phys;}
        if(first_served>=0)vrf_rr=(unsigned(first_served)+1)%NPORT;
        f1=nf1;fb=nfb;d=nd;r=nr;xb=nx;
    }
    void record(unsigned status,unsigned code,u64 pc,const std::string& why){
        if(!aborting){aborting=true;fault.status=status;fault.code=code;fault.pc=pc;fault.why=why;}
        for(auto& barrier:c.barriers)if(!barrier.error)barrier.invalidate();
    }
    //------------------------------------------------------------ launch [6.1]
    void launch(const Launch& l){
        t=x.now();launch_cycle=t;last_progress=t;cur=&l;aborting=false;fault=Outcome{};++launch_n;
        for(unsigned i=0;i<4;++i){Ctx& q=ctx[i];bool on=(l.roles>>i)&1;
            unsigned base=(l.pair?(i&~1U):i)*64;q.f.reset(new Frame(c.vrf,base));
            Frame& f=*q.f;f.pc=l.pc;f.arg=l.arg;f.px=l.px;f.py=l.py;f.pz=l.pz;f.nx=l.nx;f.ny=l.ny;f.nz=l.nz;
            f.role=i;f.role_count=unsigned(__builtin_popcount(l.roles));f.sliced=l.slice;f.slice_k=l.slice_k;f.slice_id=l.pair?i>>1:i;
            q.pc=l.pc;q.running=on;q.launched=on;q.faulted=q.blocked=q.parked=false;q.instret=0;q.slp_v=false;q.pending_wake=0;q.wake_err=false;
        }
        rr_ptr=0;f1=fb=d=r=xb=Bundle{};w.en=false;wb_rsv=0;sched.clear();
        std::memset(sb_s,0,sizeof sb_s);std::memset(sb_v,0,sizeof sb_v);std::memset(sb_p,0,sizeof sb_p);std::memset(sb_acc,0,sizeof sb_acc);
        of_got=pf_tag=0;adv_q=false;vrf_rr=0;for(auto& br:bank_rsp)br.valid=false;
        mtx.reset();mtx_busy_q=txf_busy_q=false;
        txf.reset();for(auto& barrier:c.barriers)barrier.reset();
        if(replay)replay->start_launch();
        scheduled.clear();targets.clear();wide_rsp.clear();
        require(!hybrid || (hybrid_idle() && !cmd_pending),"hybrid LSU busy at a launch");
        if(uncore)uncore_launch(l.pc,l.roles);
        if(c.core_trace)c.core_trace.L(t,l.pc,unsigned(l.roles));
    }
    Outcome program(const Launch& l){
        require(!x.functional,"core_model=1 has no untimed mode");
        launch(l);
        // the launch edge is E_L; the first pipeline cycle is L+1
        wait(c.cycle);
        for(;;){
            step();
            if(core_idle())break;
            if(c.deadline!=SC_ZERO_TIME && sc_time_stamp()>c.deadline){record(STATUS_TIMEOUT,0,0,"device deadline expired");break;}
            wait(c.cycle);
        }
        if(c.core_trace)c.core_trace.E(x.now()+1);
        if(replay){*c_replay_mismatch=replay->mismatches;*c_replay_unrecorded=replay->unrecorded;}
        Outcome out=aborting?fault:Outcome{};u64 retired=0;for(auto& q:ctx){retired+=q.instret;}out.retired=retired;
        return out;
    }
};
