//==============================================================================
// The cycle-level matrix engine (tnca_matrix.v) of the cycle core
// [CORE_TIMING.md 4.1]. Included inside `struct CycleCore`. One command at a
// time; fragment reads go through port 3 of the banked register file (request
// in one cycle, data the next; a refused request repeats).
//==============================================================================
struct CycleMatrix {
    CycleCore& k;
    enum St {IDLE,ZERO,LOADA,LOADB,MAC,ROW,DRAIN,DONE};
    St st=IDLE;
    struct Cmd {unsigned op=1;unsigned slot=0,va=0,vb=0,K=16,dt=0,rctx=0,rowop=0,row=0;bool bar_en=false,sf_en=false;u64 token=0,sf_token=0;
                bool elem8=false;unsigned a_regs=0;bool use_abuf=false;PhysicalRegisters::Vector vdata{};std::uint32_t scale=0;unsigned mask=0;
                // R41 G2 m.mma.dp (MAC_DP lanes): B's own format, and the k step (two products a MAC, four for E4M3 x E4M3)
                bool dp=false,elemB8=false;unsigned bdt=0,kstep=1;} cmd;
    unsigned kk=0,mm=0,la=0,drain=0;bool pending=false;   // pending: a port-3 read issued this cycle, data next cycle
    std::vector<PhysicalRegisters::Vector> A,B;std::vector<bool> a_have,b_have;
    u64 done_at=0,srcfree_at=0;unsigned done_slot=0;bool done_bar=false;u64 done_token=0,srcfree_token=0;
    explicit CycleMatrix(CycleCore& core):k(core){}
    bool busy()const{return st!=IDLE;}
    void reset(){st=IDLE;pending=false;done_at=srcfree_at=0;srcfree_token=0;}
    unsigned k_limit()const{return cmd.K;}
    // the register port 3 asks for this cycle (tnca_matrix.v 199-216), or ~0u
    unsigned read_index()const{
        if(pending)return ~0u;
        if(st==LOADA)return cmd.va+la;
        if(st==LOADB)return cmd.vb+(cmd.elemB8?kk/4:kk/2);
        if(st==MAC && !cmd.use_abuf){unsigned a_flat=mm*k_limit()+kk;return cmd.va+(cmd.elem8?a_flat/64:a_flat/32);}
        return ~0u;
    }
    bool vrf_req()const{return st==LOADA || st==LOADB || (st==MAC && !cmd.use_abuf);}
    void issue(const Cmd& c0){
        cmd=c0;kk=mm=la=drain=0;pending=false;
        if(cmd.op==0){st=ZERO;return;}
        if(cmd.op==2){st=ROW;return;}
        cmd.elem8=cmd.dt>=2;cmd.a_regs=cmd.elem8?cmd.K/4:cmd.K/2;
        if(!cmd.dp)cmd.bdt=cmd.dt;
        cmd.elemB8=cmd.bdt>=2;cmd.kstep=!cmd.dp?1:(cmd.dt==3 && cmd.bdt==3)?4:2;
        cmd.use_abuf=k.c.abuf && (cmd.elem8?cmd.K!=64:cmd.K==16);
        unsigned b_regs=tnca_matrix_ops::fragment_registers(cmd.K,16,cmd.bdt);
        A.assign(cmd.a_regs,PhysicalRegisters::Vector{});a_have.assign(cmd.a_regs,false);
        B.assign(b_regs,PhysicalRegisters::Vector{});b_have.assign(b_regs,false);
        st=cmd.use_abuf?LOADA:LOADB;
    }
    // `accepted`: the arbiter took this cycle's request; `data_now`: the response of last cycle's request
    void edge(bool accepted,bool data_now,const PhysicalRegisters::Vector& data){
        const u64 t=k.t;
        bool have=data_now;   // !vrf_wait during this cycle
        if(accepted)pending=true;
        if(data_now)pending=false;
        switch(st){
        case IDLE:break;
        case ZERO:if(mm==15){mm=0;st=DONE;}else ++mm;break;
        case LOADA:if(have){if(la<A.size()){A[la]=data;a_have[la]=true;}if(la+1==cmd.a_regs){st=LOADB;}else ++la;}break;
        case LOADB:if(have){unsigned bi=cmd.elemB8?kk/4:kk/2;if(bi<B.size()){B[bi]=data;b_have[bi]=true;}mm=0;st=MAC;}break;
        case MAC:if(cmd.use_abuf || have){
                if(!cmd.use_abuf){unsigned a_flat=mm*k_limit()+kk;unsigned ai=cmd.elem8?a_flat/64:a_flat/32;if(ai<A.size() && !a_have[ai]){A[ai]=data;a_have[ai]=true;}}
                if(mm==15){mm=0;
                    if(kk==k_limit()-cmd.kstep){if(cmd.sf_en){srcfree_at=t+1;srcfree_token=cmd.sf_token;}drain=k.mtx_pipe+1;finish_product();st=DRAIN;}
                    else{kk+=cmd.kstep;st=LOADB;}
                }else ++mm;
            }break;
        case ROW:if(cmd.rowop==0){st=DONE;}else{drain=k.mtx_pipe+1;st=DRAIN;}break;
        case DRAIN:if(drain==0){st=DONE;}else --drain;break;
        case DONE:done_at=t+1;done_slot=cmd.slot;done_bar=cmd.bar_en;done_token=cmd.token;st=IDLE;break;
        }
    }
    void finish_product(){
        // registers never read (a short fragment) stay as read at issue: fill from the file now
        Frame& f=k.F(cmd.rctx);
        for(unsigned i=0;i<A.size();++i)if(!a_have[i])A[i]=f.physical->read(f.vbase+cmd.va+i,sc_time_stamp());
        for(unsigned i=0;i<B.size();++i)if(!b_have[i])B[i]=f.physical->read(f.vbase+cmd.vb+i,sc_time_stamp());
        if(cmd.dp)tnca_matrix_ops::mma_dp(k.c.acc[cmd.slot],A,B,cmd.K,cmd.dt,cmd.bdt);
        else tnca_matrix_ops::mma(k.c.acc[cmd.slot],A,B,cmd.K,cmd.dt);
    }
};
