//==============================================================================
// The cycle-level uncore (uncore_model=1): the path from the cores' memory
// ports to the device's memory port -- core NI, mesh, home slices, memory
// arbiter, DRAM model -- stepped once per cycle, half a cycle after the cores
// (its edge E_t runs at SC time t+0.5: it sees what the cores fired during t,
// and what it makes valid for cycle t+1 reaches them at their step t+1).
// tools/performance_harness/align/UNCORE_TIMING.md is the rule book ([1.1],
// [2.4] ... cite its sections). Two configurations: the simulation bench
// (PORTS=1, tnca_dram_model, no link stages) and, with uncore_board=1, the
// board's (UNCORE_TIMING_BOARD.md, cited [B n]): home links, one port per
// home, an outlet link and a tnca_line_axi adapter per port, the bench's AXI
// line model behind it (deterministic here: every AXI ready high, the lowest
// due read first; the bench's $random readies and picks are not reproduced,
// the replay path takes the recorded accept and response cycles instead).
//==============================================================================
struct CycleUncore : sc_module {
    SC_HAS_PROCESS(CycleUncore);
    Context& x;std::vector<CycleCore*> cores;unsigned ncores,nx,ny,sets,ways,set_w,link_stages;
    static constexpr unsigned NV=2,VN_REQ=0,VN_RSP=1,NPORT=5,P_LOCAL=0,P_N=1,P_E=2,P_S=3,P_W=4,FIFO_DEPTH=4,RSQ_N=4,WBQ_N=4;
    unsigned rq_n=8,mshr_n=8,target_n=16;   // the slice's (R40) or, with HOME_MSHR_N 64, the home pipe's 16 / 64 / 128
    //------------------------------------------------------------- flits
    // be: the request's byte enables within its 64-byte line (a word's eight shifted to its place; a line request's 64)
    struct Flit {bool valid=false;unsigned dst=0,src=0,tag=0;u64 addr=0;bool we=false;u64 be=0;u64 data=0;u64 id=0;unsigned core=0;bool engine=false;u64 got=0;};
    // tnca_fifo: push/pop registered; a datum pushed at E_t is at the head during t+1; a full FIFO refuses a push even if popped this cycle [0]
    struct Fifo {std::deque<Flit> q;unsigned depth=FIFO_DEPTH;bool push_now=false;Flit push_v;bool pop_now=false;
        bool in_ready()const{return q.size()<depth;}bool out_valid()const{return !q.empty();}const Flit& head()const{return q.front();}
        void push(const Flit& f){push_now=true;push_v=f;}void pop(){pop_now=true;}
        void edge(){if(pop_now){q.pop_front();pop_now=false;}if(push_now){q.push_back(push_v);push_now=false;}}};
    //------------------------------------------------------------- routers [2.4]
    struct Router {std::array<std::array<Fifo,NV>,NPORT> in;std::array<std::array<unsigned,NV>,NPORT> rr{};unsigned xh=0,yh=0;};
    std::vector<Router> routers;
    unsigned route(const Router& r,const Flit& f)const{unsigned dx=f.dst%nx,dy=f.dst/nx;if(dx>r.xh){return P_E;}if(dx<r.xh){return P_W;}if(dy>r.yh){return P_N;}if(dy<r.yh){return P_S;}return P_LOCAL;}
    // elastic links (STAGES chained 2-deep FIFOs) between routers: [from][port][vn]
    struct Link {std::vector<Fifo> stages;};
    std::map<std::tuple<unsigned,unsigned,unsigned>,Link> links;
    unsigned neighbour(unsigned node,unsigned port)const{unsigned xx=node%nx,yy=node/nx;switch(port){case P_E:return node+1;case P_W:return node-1;case P_N:return node+nx;case P_S:return node-nx;default:return node;}(void)xx;(void)yy;}
    bool port_exists(unsigned node,unsigned port)const{unsigned xx=node%nx,yy=node/nx;switch(port){case P_E:return xx+1<nx;case P_W:return xx>0;case P_N:return yy+1<ny;case P_S:return yy>0;default:return true;}}
    unsigned opposite(unsigned port)const{switch(port){case P_E:return P_W;case P_W:return P_E;case P_N:return P_S;case P_S:return P_N;default:return P_LOCAL;}}
    //------------------------------------------------------------- the slice [3]
    struct Rq {bool v=false,blocked=false,busy=false;Flit f;u64 seq=0;};   // busy: in the home pipe's S1 (R41)
    // st: the slice's 0 FREE,1 WB,2 REQ,3 WAIT,4 FILL,5 WAKE; the home pipe's HP_FREE..HP_FILL
    struct Mshr {unsigned st=0;u64 line=0,victim_line=0;unsigned set=0,way=0;std::vector<unsigned> targets;bool byp=false;};
    struct Tgt {bool v=false;Flit f;};
    static constexpr unsigned HP_FREE=0,HP_REQ=1,HP_WAIT=2,HP_FILL=3;
    static constexpr unsigned K_HIT_RD=0,K_HIT_WR=1,K_INSTALL=3,K_ALLOC=4,K_MMIO_RD=5,K_MMIO_WR=6;
    struct Slice {
        unsigned id=0;std::vector<u64> tag;std::vector<bool> valid,dirty;std::vector<unsigned> victim_ptr;
        std::vector<Rq> rq;std::vector<Mshr> ms;std::vector<Tgt> tg;std::deque<Flit> rsq;u64 seq=0;
        unsigned est=0;/*0 SEL,1 LOOK,2 HIT,3 MISS,4 MREQ,5 WBRD,6 WBSND,7 FILL,8 WAKE*/unsigned cur_rq=0,cur_ms=0;bool cur_is_ms=false;
        // the memory request this cycle presents (from the pre-edge state) [3.5]
        struct MemReq {bool valid=false,write=false;u64 addr=0;unsigned ms=0;} mreq;
        bool mem_req_valid()const{return mreq.valid;}
        // R41 home pipe (tnca_home_pipe.v, UNCORE_TIMING 10.2): claimed ways, the writeback queue (line
        // numbers), S1 (the request whose set's tags are read), S3 (data array, response, writeback),
        // the fill unit and the locked memory request
        std::vector<bool> claimed;std::deque<u64> wbq;
        struct {bool v=false,stale=false;unsigned rq=0;u64 addr=0;} s1;
        struct {bool v=false,wb=false;unsigned kind=0,index=0;u64 wb_line=0;Flit f;} s3;
        struct {bool v=false,first=false,dirty=false;unsigned ms=0,pos=0;} fu;
        struct {bool v=false,wb=false;unsigned ms=0;} ml;
    };
    std::vector<Slice> slices;bool home_pipe=false,fill_chain=false,line_write_nofill=false;
    //------------------------------------------------------------- the board's memory side [B 1-4]
    bool board=false;unsigned home_stages=0,outlet_stages=0,dispatch_stages=0,active_outlets=1;
    // R41 (UNCORE_TIMING_BOARD 9.2-9.3): OUT_N slots, beats per line (512 / AXI_DATA_W), PIPELINED_ISSUE; the
    // grid dispatcher's admission stages before the first enqueue (R41 F06: 4)
    unsigned axi_slots=8,axi_beats=2,dispatch_admission=0;bool axi_pipelined=false;
    // tnca_memory_mode [B 2.3]: with every outlet active home h is port h (zero cycles); with one, the three
    // homes share port 0 through a tnca_mem_arbiter (round robin + hold, UNCORE_TIMING 4.2)
    unsigned port_of(unsigned home)const{return active_outlets==1?0:home%active_outlets;}
    struct DramEvent {bool fire=false;unsigned src=0,tag=0;};
    struct Port {
        std::vector<Fifo> hreq,oreq;                       // home link, outlet link: request chains of 2-deep stages [B 1.1]
        struct Slot {bool active=false,writing=false,sent=false;u64 addr=0;unsigned src=0,tag=0,rbeat=0;};std::vector<Slot> slot;   // tnca_line_axi [B 3.1]: OUT_N slots
        bool issue_valid=false;unsigned issue_slot=0,issue_age=0;unsigned admission=8;
        unsigned latency=25,write_delay=15;                // tnca_axi_line_model per port [B 4]
        struct Rd {bool v=false;u64 due=0;unsigned beat=0;};std::vector<Rd> rd;
        struct Wr {bool v=false;u64 due=0;};std::vector<Wr> wr;
        bool rvalid=false;unsigned rid=0;bool rlast=false; // the R beat presented during this cycle (picked at the previous edge)
        bool bvalid=false;unsigned bid=0;                  // the B presented during this cycle
        bool rsp_valid=false;unsigned rsp_src=0,rsp_tag=0; // the adapter's rsp_valid during this cycle [B 3.4]
        unsigned rr=0;bool held=false;unsigned held_pick=0; // the port's arbiter among its homes' link heads
    };
    std::vector<Port> ports;
    struct Due {u64 when;DramEvent ev;};std::vector<Due> slice_rsp;   // responses on their way to a slice (outlet + home link registers, never stalled)
    //------------------------------------------------------------- memory arbiter [4.2] + DRAM model [5]
    unsigned arb_rr=0;bool arb_held=false;unsigned arb_held_pick=0;
    struct Pool {bool v=false;unsigned cnt=0,tag=0,src=0;u64 id=0;unsigned core=0;u64 got=0;};
    std::array<Pool,32> pool;std::uint32_t lfsr=0xACE12345u;unsigned dram_latency=11;
    DramEvent rsp_pending;   // rsp_valid during the coming cycle (LFSR path)
    // replay of the device port (X accept cycles, Y response cycles): the platform's UncoreReplay (platform.inc)
    std::unique_ptr<UncoreReplay> rp;bool replaying=false;
    //------------------------------------------------------------- misc
    struct Req {unsigned core;u64 id,at,addr;bool store;u64 be;u64 got;bool engine;unsigned tag;};
    std::deque<Req> incoming;u64 t=0;UncoreTrace trace;u64 last_launch=0;std::ofstream port_debug;   // TNCA_MODEL_PORT_DEBUG=<path>: why an adapter refuses
    u64 *c_hits,*c_misses,*c_merges,*c_blocked,*c_wb,*c_dram_reads,*c_dram_writes,*c_rsq_full,*c_mreq_stall;
    u64 *c_installs,*c_replays,*c_s2_holds,*c_fill_chains,*c_wbq_full;   // the home pipe's paths (coverage)
    CycleUncore(sc_module_name n,Context& ctx,unsigned cores_):sc_module(n),x(ctx),ncores(cores_){
        nx=unsigned(x.c.get("nx"));ny=unsigned(x.c.get("ny"));sets=unsigned(x.c.get("sets"));ways=unsigned(x.c.get("ways"));
        set_w=0;while((1u<<set_w)<sets){++set_w;}link_stages=x.c.v.count("tl_noc_link_stages")?unsigned(x.c.get("tl_noc_link_stages")):0;
        routers.resize(nx*ny);for(unsigned i=0;i<routers.size();++i){routers[i].xh=i%nx;routers[i].yh=i/nx;}
        for(unsigned node=0;node<routers.size();++node)for(unsigned p=1;p<NPORT;++p)if(port_exists(node,p))for(unsigned vn=0;vn<NV;++vn){
            Link& l=links[std::make_tuple(node,p,vn)];l.stages.resize(link_stages);for(auto& s:l.stages)s.depth=2;}
        // tnca_shell.v: MSHR_N = HOME_MSHR_N, TARGET_N = 2*MSHR_N above 8 (else 16), RQ_N 16 above 8 (else 8);
        // tnca_tile.v: the pipelined home replaces the slice when HOME_LINE_PIPE && HYBRID_LSU
        mshr_n=unsigned(x.c.get_or("uncore_home_mshr_n",8));target_n=mshr_n>8?2*mshr_n:16;rq_n=mshr_n>8?16:8;
        {const unsigned hlp=unsigned(x.c.get_or("uncore_home_line_pipe",0));
         home_pipe=hlp>0 && x.c.get_or("core_lsu_hybrid",0)!=0;fill_chain=hlp>=2;
         line_write_nofill=x.c.get_or("uncore_home_line_write_nofill",0)!=0;}
        slices.resize(nx*ny);
        for(unsigned i=0;i<slices.size();++i){Slice& s=slices[i];s.id=i;s.tag.assign(sets*ways,0);s.valid.assign(sets*ways,false);s.dirty.assign(sets*ways,false);s.victim_ptr.assign(sets,0);
            s.claimed.assign(sets*ways,false);s.rq.resize(rq_n);s.ms.resize(mshr_n);s.tg.resize(target_n);}
        axi_slots=unsigned(x.c.get_or("uncore_axi_slots",8));axi_beats=512/unsigned(x.c.get_or("uncore_axi_data_w",256));
        axi_pipelined=x.c.get_or("uncore_axi_pipelined_issue",0)!=0;dispatch_admission=unsigned(x.c.get_or("uncore_dispatch_admission",0));
        host_dma=x.c.get_or("uncore_host_dma",0)!=0;
        require(axi_beats==1 || axi_beats==2,"uncore_axi_data_w is 256 or 512");
        require(!axi_pipelined || axi_beats==1,"uncore_axi_pipelined_issue models one-beat lines only (the pending slot is never used)");        dram_latency=x.c.v.count("tl_dram_latency")?unsigned(x.c.get("tl_dram_latency")):11;
        board=x.c.v.count("uncore_board") && x.c.get("uncore_board")==1;
        require(!home_pipe || board,"the home pipe is modelled on the board configuration only (uncore_board=1)");
        if(board){ // the U280 image's structure [B 0]: MEM_PORTS = N (home h -> port h), every link 2 stages, DISPATCH_STAGES 2
            home_stages=2;outlet_stages=2;dispatch_stages=2;
            if(!x.c.v.count("tl_noc_link_stages")){link_stages=2;for(auto& kv:links){kv.second.stages.resize(link_stages);for(auto& st:kv.second.stages)st.depth=2;}}
            ports.resize(slices.size());
            active_outlets=x.c.v.count("active_outlets")?unsigned(x.c.get("active_outlets")):unsigned(slices.size());
            require(active_outlets==1 || active_outlets==slices.size(),"uncore_board: active_outlets is 1 or the number of homes");
            unsigned lat=x.c.v.count("uncore_axi_latency")?unsigned(x.c.get("uncore_axi_latency")):25;
            unsigned step=x.c.v.count("uncore_axi_latency_step")?unsigned(x.c.get("uncore_axi_latency_step")):19;
            unsigned wdelay=x.c.v.count("uncore_axi_write_delay")?unsigned(x.c.get("uncore_axi_write_delay")):15;
            for(unsigned p=0;p<ports.size();++p){Port& q=ports[p];q.hreq.resize(home_stages);q.oreq.resize(outlet_stages);
                q.slot.resize(axi_slots);q.rd.resize(axi_slots);q.wr.resize(axi_slots);
                for(auto& st:q.hreq){st.depth=2;}
                for(auto& st:q.oreq){st.depth=2;}
                q.latency=lat+step*p;q.write_delay=wdelay;q.admission=x.c.v.count("outstanding")?unsigned(x.c.get("outstanding")):8;}
        }
        c_hits=&x.counts["uncore_hits"];c_misses=&x.counts["uncore_misses"];c_merges=&x.counts["uncore_merges"];c_blocked=&x.counts["uncore_blocked"];
        c_wb=&x.counts["uncore_writebacks"];c_dram_reads=&x.counts["uncore_dram_reads"];c_dram_writes=&x.counts["uncore_dram_writes"];c_rsq_full=&x.counts["uncore_rsq_full"];c_mreq_stall=&x.counts["uncore_mreq_stall"];
        if(home_pipe){c_installs=&x.counts["uncore_installs"];c_replays=&x.counts["uncore_replays"];c_s2_holds=&x.counts["uncore_s2_holds"];
            c_fill_chains=&x.counts["uncore_fill_chains"];c_wbq_full=&x.counts["uncore_wbq_full"];}
        trace.open(x,ncores);
        rp=UncoreReplay::open();replaying=bool(rp);if(rp)rp->ports(unsigned(ports.size()));
        if(const char* p=std::getenv("TNCA_MODEL_PORT_DEBUG")){if(*p)port_debug.open(p);}
        SC_THREAD(run);
    }
    //------------------------------------------------------------- home mapping [2.2]
    unsigned home(u64 addr)const{u64 line=addr>>6;unsigned h=0;for(unsigned b=0;b<8;++b)h^=unsigned((line>>(8*b))&255);
        if(nx==2 && ny==2){return unsigned((line&3)^((line>>2)&3));}return h%(nx*ny);}
    unsigned set_of(u64 addr)const{return unsigned((addr>>6)&((1u<<set_w)-1));}
    u64 tag_of(u64 addr)const{return addr>>(6+set_w);}
    //------------------------------------------------------------- the host warms the slices (one request per word, zero time) [7]
    void host_touch(u64 addr,bool write){
        Slice& s=slices[home(addr)];unsigned st=set_of(addr);u64 tg=tag_of(addr);
        for(unsigned w=0;w<ways;++w){unsigned i=st*ways+w;if(s.valid[i] && s.tag[i]==tg){if(write){s.dirty[i]=true;}return;}}
        int way=-1;for(unsigned w=0;w<ways;++w)if(!s.valid[st*ways+w]){way=int(w);break;}
        if(way<0){way=int(s.victim_ptr[st]%ways);}
        unsigned i=st*ways+unsigned(way);s.valid[i]=true;s.dirty[i]=write;s.tag[i]=tg;s.victim_ptr[st]=(unsigned(way)+1)%ways;
        // a host miss is a fill (and maybe a writeback) on the home's port, one at a time: the port arbiter's
        // pointer moves past this home exactly as it does for a launch's transaction (tnca_mem_arbiter.v rr)
        port_granted(s.id);
    }
    // The R41 runtime's DMA ownership handoff before a host transfer of [lo, hi) (multicore_mmio.py _handoff,
    // OP_CACHE_HANDOFF -> every home's maintenance scan): each line overlapping the range is written back if
    // dirty and invalidated; the replacement pointers stay. Zero time: the device is idle between launches, and
    // with every outlet active (home h on port h) no arbiter remembers the writebacks [10.3].
    bool host_dma=false;
    void host_handoff(u64 lo,u64 hi){
        for(u64 line=lo>>6;line<=(hi-1)>>6;++line){const u64 a=line<<6;Slice& s=slices[home(a)];const unsigned st=set_of(a);const u64 tg=tag_of(a);
            for(unsigned w=0;w<ways;++w){const unsigned i=st*ways+w;if(s.valid[i] && s.tag[i]==tg){s.valid[i]=false;s.dirty[i]=false;}}}
    }
    // the port arbiter of home h's port granted h: its round robin resumes after h, nothing is held
    void port_granted(unsigned h){
        if(!board){arb_held=false;arb_rr=(h+1)%slices.size();return;}
        Port& q=ports[port_of(h)];std::vector<unsigned> homes;for(unsigned k=0;k<slices.size();++k)if(port_of(k)==port_of(h))homes.push_back(k);
        unsigned idx=0;for(unsigned k=0;k<homes.size();++k)if(homes[k]==h)idx=k;
        q.held=false;q.rr=(idx+1)%homes.size();
    }
    //------------------------------------------------------------- requests from the cores
    bool req_ready(unsigned core)const{return routers[core].in[P_LOCAL][VN_REQ].in_ready();}   // [1.3]
    void request(const Req& r){incoming.push_back(r);}
    ~CycleUncore(){if(rp)x.counts["uncore_replay_mismatches"]=rp->mismatches;trace.close(t);}
    void launch_record(unsigned core,u64 cycle,u64 pc,unsigned roles){
        last_launch=cycle;if(rp){rp->start_launch();x.counts["uncore_replay_mismatches"]=rp->mismatches;}
        if(trace)trace.L(cycle,core,pc,roles);}
    void run(){
        wait(x.cycles(1)/2);
        for(;;){t=x.to_cycles(sc_time_stamp()-x.cycles(1)/2);step();wait(x.cycles(1));}
    }
    //------------------------------------------------------------- replay of the device port (platform.inc)
    const UncoreReplay::X* next_x(unsigned p){return rp->next_x(p,[this](unsigned s){return port_of(s);});}
    const UncoreReplay::Y* next_y(unsigned p){return rp->next_y(p,[this](unsigned s){return port_of(s);});}
    //------------------------------------------------------------- one cycle: decisions from the pre-edge state, then the edge
    void step(){
        //---- requests fired by the cores during t: pushed into the source router's local VN_REQ FIFO at E_t [1.1 #3-4]
        while(!incoming.empty()){Req r=incoming.front();incoming.pop_front();Flit f;f.valid=true;f.src=r.core;f.dst=home(r.addr);f.addr=r.addr;f.we=r.store;f.be=r.be;f.data=0;f.tag=r.tag;f.id=r.id;f.core=r.core;f.engine=r.engine;f.got=r.got;
            Fifo& q=routers[r.core].in[P_LOCAL][VN_REQ];require(q.in_ready(),"core fired into a full NoC FIFO");q.push(f);}
        //---- the memory request each slice presents this cycle (pre-edge engine state) [3.5]
        for(auto& s:slices){s.mreq=Slice::MemReq{};if(home_pipe){hp_mreq(s);continue;}if(s.est==4||s.est==6){const Mshr& m=s.ms[s.cur_ms];s.mreq.valid=true;s.mreq.write=s.est==6;s.mreq.addr=(s.est==6?m.victim_line:m.line)<<6;s.mreq.ms=s.cur_ms;}}
        //---- DRAM (sim bench): this cycle's ready and the response it presents [5]; the board: the home links' ready and the responses due [B 1-4]
        bool dram_ready=false;DramEvent dev;std::vector<DramEvent> board_rsp;
        int grant=-1;std::vector<bool> hgrant(slices.size(),false);
        if(board){
            for(auto& s:slices)if(s.mem_req_valid() && ports[s.id].hreq[0].in_ready())hgrant[s.id]=true;   // mem_req_ready = home link stage 0 [B 1.2]
            for(size_t i=0;i<slice_rsp.size();){if(slice_rsp[i].when<=t){board_rsp.push_back(slice_rsp[i].ev);slice_rsp[i]=slice_rsp.back();slice_rsp.pop_back();}else ++i;}
        }else{
            dev=dram_cycle(dram_ready);
            //---- slice engines decide (pre-edge state); their memory requests want the arbiter [3][4.2]
            std::vector<int> want;for(auto& s:slices)if(s.mem_req_valid())want.push_back(int(s.id));
            if(!want.empty()){unsigned pick=arb_held?arb_held_pick:0;bool ok=false;
                if(arb_held){ok=slices[arb_held_pick].mem_req_valid();pick=arb_held_pick;}
                if(!ok){for(unsigned ai=0;ai<slices.size();++ai){unsigned g=(arb_rr+ai)%slices.size();if(slices[g].mem_req_valid()){pick=g;ok=true;break;}}}
                if(ok){if(dram_ready)grant=int(pick);else{arb_held=true;arb_held_pick=pick;}}}
            if(replaying && grant>=0 && !replay_allows_x(slices[grant])){grant=-1;}
        }
        //---- routers: output arbitration from the pre-edge FIFO heads [2.4]
        struct Xfer {unsigned node,in_port,vn,out_port;Flit f;};std::vector<Xfer> xfers;
        std::vector<Flit> slice_in(slices.size());std::vector<Flit> core_out(ncores);
        for(unsigned node=0;node<routers.size();++node){Router& r=routers[node];
            for(unsigned op=0;op<NPORT;++op)for(unsigned vn=0;vn<NV;++vn){
                if(op!=P_LOCAL && !port_exists(node,op))continue;
                // candidates (rr+k)%5, k=0 first
                int sel=-1;for(unsigned k=0;k<NPORT;++k){unsigned ip=(r.rr[op][vn]+k)%NPORT;const Fifo& q=r.in[ip][vn];if(!q.out_valid()){continue;}if(route(r,q.head())!=op){continue;}sel=int(ip);break;}
                if(sel<0)continue;
                // out_ready
                bool ready=false;
                if(op==P_LOCAL){if(vn==VN_REQ){Slice& s=slices[node];unsigned freec=0;for(auto& e:s.rq){if(!e.v)++freec;}ready=freec>0;}else ready=true;}
                else{Link& l=links[std::make_tuple(node,op,vn)];if(l.stages.empty()){ready=routers[neighbour(node,op)].in[opposite(op)][vn].in_ready();}else ready=l.stages[0].in_ready();}
                if(!ready)continue;
                xfers.push_back({node,unsigned(sel),vn,op,r.in[sel][vn].head()});
            }}
        //---- the slices' responses that leave for the home router this cycle (pre-edge rsq head) [1.1 #13]
        std::vector<bool> rsq_pop(slices.size(),false);
        for(auto& s:slices){Fifo& q=routers[s.id].in[P_LOCAL][VN_RSP];if(!s.rsq.empty() && q.in_ready())rsq_pop[s.id]=true;}
        //---- slice engines: one engine cycle each [3.3]
        for(auto& s:slices){const bool g=board?hgrant[s.id]:grant==int(s.id);if(home_pipe)home_cycle(s,g);else slice_cycle(s,g,slice_in,dram_ready);}
        //==== the edge
        // transfers: pop sources, push destinations (into FIFOs or the slice / core)
        for(auto& xf:xfers){Router& r=routers[xf.node];r.in[xf.in_port][xf.vn].pop();r.rr[xf.out_port][xf.vn]=xf.in_port==4?0:xf.in_port+1;
            if(xf.out_port==P_LOCAL){
                if(xf.vn==VN_REQ)slice_accept(slices[xf.node],xf.f);
                else{ // core NI register: rsp_valid during t+1 [1.1 #15]
                    cores[xf.f.core]->schedule(xf.f.id,t+1,xf.f.got);}
            }else{Link& l=links[std::make_tuple(xf.node,xf.out_port,xf.vn)];
                if(l.stages.empty()){routers[neighbour(xf.node,xf.out_port)].in[opposite(xf.out_port)][xf.vn].push(xf.f);}else {l.stages[0].push(xf.f);}}
        }
        // links advance: stage i head -> stage i+1 or the next router's FIFO
        for(auto& kv:links){Link& l=kv.second;if(l.stages.empty()){continue;}unsigned node=std::get<0>(kv.first),op=std::get<1>(kv.first),vn=std::get<2>(kv.first);
            for(int i=int(l.stages.size())-1;i>=0;--i){Fifo& st=l.stages[i];if(!st.out_valid())continue;
                if(i+1<int(l.stages.size())){if(l.stages[i+1].in_ready()){l.stages[i+1].push(st.head());st.pop();}}
                else{Fifo& nq=routers[neighbour(node,op)].in[opposite(op)][vn];if(nq.in_ready()){nq.push(st.head());st.pop();}}}}
        // slices' responses into the home router's local VN_RSP FIFO [1.1 #13]
        for(auto& s:slices)if(rsq_pop[s.id]){Fifo& q=routers[s.id].in[P_LOCAL][VN_RSP];Flit f=s.rsq.front();s.rsq.pop_front();q.push(f);
            if(trace)trace.R(t,s.id,f.src,f.tag);}
        if(board){
            // the responses valid at the slices during t are captured now [B 1.2]
            for(auto& ev:board_rsp)dram_deliver(ev);
            // slices fire into their home links [B 1.2]
            for(auto& s:slices)if(hgrant[s.id]){Flit f;f.valid=true;f.we=s.mreq.write;f.addr=s.mreq.addr;f.tag=s.mreq.ms;f.src=s.id;ports[s.id].hreq[0].push(f);
                if(trace)trace.M(t,s.id,f.we,f.addr,f.tag);
                if(f.we)++*c_dram_writes;else ++*c_dram_reads;}
            for(unsigned p=0;p<active_outlets;++p)port_edge(p);
            for(auto& q:ports)for(int i=int(q.hreq.size())-2;i>=0;--i){Fifo& st=q.hreq[i];if(st.out_valid() && q.hreq[i+1].in_ready()){q.hreq[i+1].push(st.head());st.pop();}}
        }else{
            // DRAM: the response valid during t is captured by its slice now; the pool counts down; the accept; next cycle's response
            if(dev.fire)dram_deliver(dev);
            pool_tick();
            if(grant>=0){Slice& s=slices[grant];dram_accept(s,dev);arb_held=false;arb_rr=(unsigned(grant)+1)%slices.size();}
            dram_pick_next();
        }
        // FIFO edges
        for(auto& r:routers)for(auto& pq:r.in)for(auto& q:pq)q.edge();
        for(auto& kv:links)for(auto& st:kv.second.stages)st.edge();
        for(auto& q:ports){for(auto& st:q.hreq)st.edge();for(auto& st:q.oreq)st.edge();}
        lfsr=(lfsr<<1)|(((lfsr>>31)^(lfsr>>21)^(lfsr>>1)^lfsr)&1u);
    }
    //------------------------------------------------------------- the board's port p at the edge of cycle t [B 1-4]
    // Pre-edge facts: the request at the outlet link's last stage, the adapter's ready, the R beat and the B
    // presented during t. Then the edge: the accept, the issue register, the links, the slave's picks for t+1.
    void port_edge(unsigned p){
        Port& q=ports[p];
        // ---- the adapter's request this cycle [B 3.2]
        Fifo& last=q.oreq.back();const unsigned nslot=unsigned(q.slot.size());
        // ---- the issue register: AR during a+1; AW+W0 during a+1, W1 during a+2 (every AXI ready high) [B 3.3];
        // with one beat per line a write's AW and W both go during a+1 [B 9.2]. With PIPELINED_ISSUE the
        // request accepted at this edge takes the register the finishing issue leaves (issue_done), so the
        // register is advanced first; without it no request is accepted while it is valid.
        const bool issue_was_valid=q.issue_valid;
        if(issue_was_valid){++q.issue_age;Port::Slot& sl=q.slot[q.issue_slot];
            if(!sl.writing && q.issue_age==1){q.issue_valid=false;sl.sent=true;
                q.rd[q.issue_slot]=Port::Rd{true,t+q.latency+(q.issue_slot*11)%31,0};}          // due: eligible at E_{x+L+skew}, first beat during +1 [B 4.2]
            else if(sl.writing && q.issue_age==axi_beats){q.issue_valid=false;sl.sent=true;
                q.wr[q.issue_slot]=Port::Wr{true,t+q.latency+q.write_delay+(q.issue_slot*7)%23};} // B during due+1 [B 4.3]
        }
        if(last.out_valid()){const Flit& f=last.head();
            int free_slot=-1;for(unsigned k=0;k<nslot;++k)if(!q.slot[k].active){free_slot=int(k);break;}
            bool hazard=false;unsigned active_count=0;
            for(unsigned k=0;k<nslot;++k){const Port::Slot& sl=q.slot[k];if(!sl.active)continue;++active_count;if(sl.addr==f.addr && (sl.writing || f.we))hazard=true;}
            // the issue register as it was during t: refused while valid unless pipelined (the pending slot is
            // never occupied with every AXI ready high: an issue finishes in the cycle it is valid)
            bool ready=(axi_pipelined || !issue_was_valid) && free_slot>=0 && !hazard && active_count<q.admission;
            if(ready && replaying){const UncoreReplay::X* e=next_x(p);if(e && last_launch+e->cycle>t)ready=false;}   // no earlier than the recorded accept
            if(port_debug && !ready){const UncoreReplay::X* e=replaying?next_x(p):nullptr;
                port_debug<<t<<" port "<<p<<" held: addr="<<std::hex<<f.addr<<std::dec<<" issue="<<q.issue_valid<<"/"<<q.issue_age<<" free="<<free_slot<<" hazard="<<hazard<<" active="<<active_count
                          <<" next_x="<<(e?long(last_launch+e->cycle):-1L)<<" last_launch="<<last_launch<<"\n";}
            if(ready){
                Port::Slot& sl=q.slot[free_slot];sl.active=true;sl.writing=f.we;sl.sent=false;sl.addr=f.addr;sl.src=f.src;sl.tag=f.tag;sl.rbeat=0;
                q.issue_valid=true;q.issue_slot=unsigned(free_slot);q.issue_age=0;
                if(trace)trace.X(t,f.we,f.addr,f.src,f.tag);
                if(replaying){const UncoreReplay::X* e=next_x(p);if(!e || e->we!=f.we || e->addr!=f.addr || e->src!=f.src){++rp->mismatches;
                    if(port_debug)port_debug<<t<<" port "<<p<<" X mismatch: model we="<<f.we<<" addr="<<std::hex<<f.addr<<std::dec<<" src="<<f.src
                        <<" record="<<(e?long(e->cycle):-1L)<<"/"<<(e?e->we:0)<<"/"<<std::hex<<(e?e->addr:0)<<std::dec<<"/"<<(e?e->src:0)<<"\n";}
                    if(e)rp->take_x(p);}
                last.pop();}
        }
        // ---- the R beat presented during t is taken (rready = 1) [B 3.4]; the B unless a final R beat coincides [B 3.5]
        bool read_final=false;
        if(!replaying && q.rvalid){Port::Slot& sl=q.slot[q.rid];Port::Rd& r=q.rd[q.rid];++sl.rbeat;++r.beat;
            if(q.rlast){read_final=true;r.v=false;sl.active=false;
                q.rsp_valid=true;q.rsp_src=sl.src;q.rsp_tag=sl.tag;}   // rsp_valid during t+1
            q.rvalid=false;}
        if(q.bvalid && !read_final){q.slot[q.bid].active=false;q.wr[q.bid].v=false;q.bvalid=false;}
        // ---- replay: the adapter's rsp_valid is the recorded Y of this port [B 7]; the slot is released at the
        // edge before it (the final R beat's), which the admission of a request in the Y cycle already sees [B 3.4]
        if(replaying){const UncoreReplay::Y* y=next_y(p);
            if(y && last_launch+y->cycle<=t+1){const u64 yc=last_launch+y->cycle;bool found=false;
                for(unsigned k=0;k<nslot;++k){Port::Slot& sl=q.slot[k];if(sl.active && !sl.writing && sl.src==y->src && sl.tag==y->tag){sl.active=false;q.rd[k].v=false;found=true;break;}}
                if(found){slice_rsp.push_back({yc+outlet_stages+home_stages,DramEvent{true,y->src,y->tag}});
                    if(trace)trace.Y(yc,y->src,y->tag);}
                else{++rp->mismatches;if(port_debug)port_debug<<t<<" port "<<p<<" Y unmatched: src="<<y->src<<" tag="<<y->tag<<" cycle="<<yc<<"\n";}
                rp->take_y(p);}
        }else if(q.rsp_valid){ // set at the previous edge: valid during t [B 3.4], at the slice +outlet +home [B 1.2]
            slice_rsp.push_back({t+outlet_stages+home_stages,DramEvent{true,q.rsp_src,q.rsp_tag}});
            if(trace)trace.Y(t,q.rsp_src,q.rsp_tag);
            q.rsp_valid=false;}
        // ---- the slave's picks for t+1: the lowest due read's next beat; the lowest due write's B [B 4.2-4.3]
        if(!replaying){int pick=-1;for(unsigned k=0;k<nslot;++k)if(q.rd[k].v && q.rd[k].due<=t){pick=int(k);break;}
            if(pick>=0){q.rvalid=true;q.rid=unsigned(pick);q.rlast=q.rd[pick].beat+1==axi_beats;}}
        if(!q.bvalid){int pick=-1;for(unsigned k=0;k<nslot;++k)if(q.wr[k].v && q.wr[k].due<=t){pick=int(k);break;}
            if(pick>=0){q.bvalid=true;q.bid=unsigned(pick);}}
        // ---- the outlet link advances [B 1.2]
        for(int i=int(q.oreq.size())-2;i>=0;--i){Fifo& st=q.oreq[i];if(st.out_valid() && q.oreq[i+1].in_ready()){q.oreq[i+1].push(st.head());st.pop();}}
        // ---- the port's arbiter: its homes' link heads -> (tnca_mem_ports, tnca_memory_mode) -> outlet link stage 0 [B 2.2-2.3]
        // one home per port: wires; every home on port 0: round robin from rr, the pick held while the link is not ready
        {std::vector<unsigned> homes;for(unsigned h=0;h<slices.size();++h)if(port_of(h)==p)homes.push_back(h);
         int pick=-1;
         if(q.held && ports[q.held_pick].hreq.back().out_valid())pick=int(q.held_pick);
         if(pick<0)for(unsigned k=0;k<homes.size();++k){unsigned h=homes[(q.rr+k)%homes.size()];if(ports[h].hreq.back().out_valid()){pick=int(h);break;}}
         if(pick>=0){Fifo& src=ports[pick].hreq.back();Fifo& nq=q.oreq[0];
            if(nq.in_ready()){nq.push(src.head());src.pop();q.held=false;unsigned idx=0;for(unsigned k=0;k<homes.size();++k)if(homes[k]==unsigned(pick))idx=k;q.rr=(idx+1)%homes.size();}
            else{q.held=true;q.held_pick=unsigned(pick);}}}
    }
    //------------------------------------------------------------- the slice's request buffer [3.2]
    void slice_accept(Slice& s,const Flit& f){
        for(unsigned i=0;i<rq_n;++i)if(!s.rq[i].v){s.rq[i].v=true;s.rq[i].blocked=false;s.rq[i].busy=false;s.rq[i].f=f;s.rq[i].seq=s.seq++;
            if(trace)trace.S(t,s.id,f.src,f.tag,f.we,f.addr);
            return;}
        require(false,"slice request buffer overflow");
    }
    int select_request(const Slice& s)const{
        int best=-1;
        for(unsigned i=0;i<rq_n;++i){const Rq& e=s.rq[i];if(!e.v || e.blocked || e.busy)continue;
            bool shadowed=false;for(unsigned j=0;j<rq_n;++j){const Rq& o=s.rq[j];if(j!=i && o.v && o.seq<e.seq && (o.f.addr>>6)==(e.f.addr>>6))shadowed=true;}
            if(shadowed){continue;}if(best<0 || e.seq<s.rq[best].seq)best=int(i);}
        return best;
    }
    int ms_pick(const Slice& s)const{   // WAKE > FILL > WB > REQ, lowest index
        for(unsigned st:{5u,4u,1u,2u})for(unsigned i=0;i<mshr_n;++i)if(s.ms[i].st==st)return int(i);
        return -1;
    }
    int ms_match(const Slice& s,u64 addr)const{for(unsigned i=0;i<mshr_n;++i){if(s.ms[i].st!=0 && s.ms[i].line==(addr>>6))return int(i);}return -1;}
    int hit_way(const Slice& s,u64 addr)const{unsigned st=set_of(addr);u64 tg=tag_of(addr);for(unsigned w=0;w<ways;++w){unsigned i=st*ways+w;if(s.valid[i] && s.tag[i]==tg)return int(w);}return -1;}
    bool way_claimed(const Slice& s,unsigned set,unsigned way)const{for(auto& m:s.ms){if(m.st!=0 && !m.byp && m.set==set && m.way==way)return true;}return false;}
    //------------------------------------------------------------- the engine [3.3][3.4][3.5]
    void slice_cycle(Slice& s,bool granted,std::vector<Flit>& slice_in,bool dram_ready){
        (void)slice_in;(void)dram_ready;
        switch(s.est){
        case 0:{ // E_SEL: MSHR work > new request (only when the response queue is not full)
            int mi=ms_pick(s);
            if(mi>=0){s.cur_ms=unsigned(mi);s.cur_is_ms=true;unsigned mst=s.ms[mi].st;s.est=mst==5?8:mst==4?7:mst==1?5:4;break;}
            if(s.rsq.size()>=RSQ_N){++*c_rsq_full;break;}
            int ri=select_request(s);if(ri>=0){s.cur_rq=unsigned(ri);s.cur_is_ms=false;s.est=1;}
            break;}
        case 1:{ // E_LOOK
            const Flit& f=s.rq[s.cur_rq].f;
            if(hit_way(s,f.addr)>=0 && ms_match(s,f.addr)<0){s.est=2;}else {s.est=3;}break;}
        case 2:{ // E_HIT: response pushed, entry freed
            Rq& e=s.rq[s.cur_rq];const Flit& f=e.f;unsigned w=unsigned(hit_way(s,f.addr));unsigned i=set_of(f.addr)*ways+w;if(f.we)s.dirty[i]=true;
            Flit r=f;r.dst=f.src;s.rsq.push_back(r);e.v=false;++*c_hits;s.est=0;break;}
        case 3:{ // E_MISS (sim): merge / allocate / block
            Rq& e=s.rq[s.cur_rq];const Flit& f=e.f;int mi=ms_match(s,f.addr);
            if(mi>=0){Mshr& m=s.ms[mi];if(m.byp && f.we){e.blocked=true;++*c_blocked;}
                else{int ti=-1;for(unsigned k=0;k<target_n;++k)if(!s.tg[k].v){ti=int(k);break;}
                    if(ti>=0){s.tg[ti].v=true;s.tg[ti].f=f;m.targets.push_back(unsigned(ti));e.v=false;++*c_merges;}else{e.blocked=true;++*c_blocked;}}
            }else{
                int mfree=-1;for(unsigned k=0;k<mshr_n;++k)if(s.ms[k].st==0){mfree=int(k);break;}
                int tfree=-1;for(unsigned k=0;k<target_n;++k)if(!s.tg[k].v){tfree=int(k);break;}
                unsigned set=set_of(f.addr);int way=-1;
                for(unsigned w=0;w<ways;++w)if(!s.valid[set*ways+w] && !way_claimed(s,set,w)){way=int(w);break;}
                if(way<0)for(unsigned k=0;k<ways;++k){unsigned w=(s.victim_ptr[set]+k)%ways;if(!way_claimed(s,set,w)){way=int(w);break;}}
                if(mfree>=0 && tfree>=0 && way>=0){
                    Mshr& m=s.ms[mfree];unsigned i=set*ways+unsigned(way);bool wb=s.valid[i] && s.dirty[i];
                    m.line=f.addr>>6;m.victim_line=(s.tag[i]<<set_w)|set;m.set=set;m.way=unsigned(way);m.byp=false;m.targets.clear();m.targets.push_back(unsigned(tfree));
                    s.tg[tfree].v=true;s.tg[tfree].f=f;s.valid[i]=false;s.victim_ptr[set]=(unsigned(way)+1)%ways;m.st=wb?1:2;e.v=false;++*c_misses;if(wb)++*c_wb;
                }else{e.blocked=true;++*c_blocked;}
            }
            s.est=0;break;}
        case 4:{ // E_MREQ: holds until the memory accepts
            if(granted){s.ms[s.cur_ms].st=3;s.est=0;}else ++*c_mreq_stall;break;}
        case 5:s.est=6;break;   // E_WBRD
        case 6:{ // E_WBSND: the writeback, then MS_REQ
            if(granted){s.ms[s.cur_ms].st=2;s.est=0;}else ++*c_mreq_stall;break;}
        case 7:{ // E_FILL: the line is valid in its way
            Mshr& m=s.ms[s.cur_ms];unsigned i=m.set*ways+m.way;s.valid[i]=true;s.tag[i]=m.line>>set_w;s.dirty[i]=false;m.st=5;s.est=0;break;}
        case 8:{ // E_WAKE: one target per SEL+WAKE; waits while the response queue is full
            Mshr& m=s.ms[s.cur_ms];
            if(s.rsq.size()>=RSQ_N){++*c_rsq_full;break;}
            require(!m.targets.empty(),"MSHR wake with no target");
            unsigned ti=m.targets.front();m.targets.erase(m.targets.begin());Flit f=s.tg[ti].f;s.tg[ti].v=false;
            if(f.we){unsigned i=m.set*ways+m.way;s.dirty[i]=true;}
            Flit r=f;r.dst=f.src;s.rsq.push_back(r);
            if(m.targets.empty()){m.st=0;for(auto& e:s.rq)e.blocked=false;}
            s.est=0;break;}
        }
    }
    //------------------------------------------------------------- the R41 home pipe [10.2] (tnca_home_pipe.v)
    // The memory request presented during t (pre-edge): the locked choice of a request memory refused, else
    // the writeback queue's head, else the lowest MS_REQ MSHR that no queued writeback (nor S3's) of the
    // same line precedes. A writeback's tag is 0, a read's its MSHR [10.2.5].
    int hp_rd_pick(const Slice& s)const{
        for(unsigned k=0;k<mshr_n;++k){const Mshr& m=s.ms[k];if(m.st!=HP_REQ)continue;
            bool blocked=s.s3.v && s.s3.wb && s.s3.wb_line==m.line;for(u64 l:s.wbq)if(l==m.line)blocked=true;
            if(!blocked)return int(k);}
        return -1;
    }
    void hp_mreq(Slice& s){
        bool wb=false;unsigned ms=0;
        if(s.ml.v){wb=s.ml.wb;ms=s.ml.ms;}
        else if(!s.wbq.empty())wb=true;
        else{const int k=hp_rd_pick(s);if(k<0)return;ms=unsigned(k);}
        s.mreq.valid=true;s.mreq.write=wb;s.mreq.addr=(wb?s.wbq.front():s.ms[ms].line)<<6;s.mreq.ms=wb?0:ms;
    }
    // One cycle of the home pipe: every decision from the pre-edge registers, then the edge. A request taken
    // from the buffer at E_a reads its set's tags into S1; S2 decides during a+1 (combinational over S1);
    // S3 pushes the response at E_{a+2} (valid during a+3). S2 checks the live tags: S2 writes a tag only
    // when it claims a way, and a request whose set it claims in the cycle the tags are read is replayed
    // (s1.stale), so the live array equals S1's registered read whenever S2 decides [10.2.2].
    void home_cycle(Slice& s,bool granted){
        auto& s1=s.s1;auto& s3=s.s3;auto& fu=s.fu;
        // ---- pre-edge facts
        const bool rsq_full=s.rsq.size()>=RSQ_N,wbq_full=s.wbq.size()>=WBQ_N;
        const bool s3_push=s3.v && s3.kind!=K_ALLOC;
        const bool s3_dw=s3.v && (s3.kind==K_HIT_WR || s3.kind==K_INSTALL);
        const bool s3_hold=s3.v && ((s3_push && rsq_full) || (s3.wb && wbq_full));   // a TMU store is taken at once
        const bool s3_go=s3.v && !s3_hold;
        int fill_pick=-1,fill_next=-1;   // the lowest MS_FILL; FILL_CHAIN: the lowest other than the fill unit's
        for(unsigned k=0;k<mshr_n;++k)if(s.ms[k].st==HP_FILL){if(fill_pick<0)fill_pick=int(k);if(fill_next<0 && k!=fu.ms)fill_next=int(k);}
        const bool fu_last=fu.v && fu.pos+1==s.ms[fu.ms].targets.size();
        const unsigned fu_index=s.ms[fu.ms].set*ways+s.ms[fu.ms].way;
        const bool fu_step=fu.v && !rsq_full && !s3_push && !(fu_last && s3_dw) && !(fu_last && s3.v && s3.wb && s3.index==fu_index);
        int tg_free=-1;for(unsigned k=0;k<target_n;++k)if(!s.tg[k].v){tg_free=int(k);break;}
        int ms_free=-1;for(unsigned k=0;k<mshr_n;++k)if(s.ms[k].st==HP_FREE){ms_free=int(k);break;}
        // ---- S2 [10.2.2]
        enum {S2_NONE,S2_REPLAY,S2_HOLD,S2_BLOCK,S2_ACT} s2=S2_NONE;
        unsigned kind=0,set=0,vic_way=0,hit_way=0;bool merge=false;int s2_ms=-1;
        if(s1.v && s1.stale)s2=S2_REPLAY;
        else if(s1.v){
            const Flit& f=s.rq[s1.rq].f;set=set_of(f.addr);const u64 tg=tag_of(f.addr);
            for(unsigned k=0;k<mshr_n;++k)if(s.ms[k].st!=HP_FREE && s.ms[k].line==(f.addr>>6))s2_ms=int(k);
            int hw=-1;for(unsigned w=0;w<ways;++w){const unsigned i=set*ways+w;if(s.valid[i] && s.tag[i]==tg)hw=int(w);}
            // the victim: the lowest invalid unclaimed way, else the first unclaimed way from the set's pointer
            bool vic_ok=false;
            for(unsigned w=ways;w-->0;)if(!s.valid[set*ways+w] && !s.claimed[set*ways+w]){vic_ok=true;vic_way=w;}
            if(!vic_ok)for(unsigned k=ways;k-->0;){const unsigned w=(s.victim_ptr[set]+k)%ways;if(!s.claimed[set*ways+w]){vic_ok=true;vic_way=w;}}
            const bool mmio=(f.addr>>12)==(0x7FFF0000ULL>>12);
            const bool full_store=line_write_nofill && f.we && f.be==~0ULL;
            if(s3_hold)s2=S2_HOLD;
            else if(mmio){s2=S2_ACT;kind=f.we?K_MMIO_WR:K_MMIO_RD;}
            else if(s2_ms>=0){merge=true;   // the fill unit closing this MSHR's list: wait, then hit
                s2=(fu.v && fu.ms==unsigned(s2_ms))?S2_HOLD:tg_free>=0?S2_ACT:S2_BLOCK;}
            else if(hw>=0){s2=S2_ACT;kind=f.we?K_HIT_WR:K_HIT_RD;hit_way=unsigned(hw);}
            else if(!vic_ok)s2=S2_BLOCK;
            else if(full_store){s2=S2_ACT;kind=K_INSTALL;}
            else{kind=K_ALLOC;s2=(ms_free<0 || tg_free<0)?S2_BLOCK:S2_ACT;}
        }
        const bool s2_hold=s2==S2_HOLD;
        if(s2==S2_REPLAY){++*c_replays;}
        if(s2_hold){++*c_s2_holds;}
        if(s3_push && rsq_full){++*c_rsq_full;}
        if(s3.v && s3.wb && wbq_full){++*c_wbq_full;}
        if(!granted && s.mreq.valid){++*c_mreq_stall;}
        // the arbiter's pick (pre-edge: S1's request is busy, S2's still valid and shadows its line) [10.2.1]
        const int sel=s2_hold?-1:select_request(s);
        //==== the edge
        // ---- the memory request [10.2.5]
        if(granted){s.ml.v=false;if(s.mreq.write){s.wbq.pop_front();++*c_wb;}else s.ms[s.mreq.ms].st=HP_WAIT;}
        else if(s.mreq.valid){s.ml.v=true;s.ml.wb=s.mreq.write;s.ml.ms=s.mreq.ms;}
        // ---- the fill unit: start on the lowest filled MSHR, else one target per step [10.2.4]
        bool free_resources=false;
        if(!fu.v && fill_pick>=0){fu.v=true;fu.ms=unsigned(fill_pick);fu.pos=0;fu.first=true;fu.dirty=false;}
        else if(fu_step){Mshr& m=s.ms[fu.ms];const unsigned ti=m.targets[fu.pos];const Flit f=s.tg[ti].f;s.tg[ti].v=false;
            Flit r=f;r.dst=f.src;s.rsq.push_back(r);fu.first=false;
            if(fu_last){const unsigned i=m.set*ways+m.way;s.valid[i]=true;s.claimed[i]=false;s.dirty[i]=fu.dirty || f.we;
                m.st=HP_FREE;m.targets.clear();free_resources=true;
                if(fill_chain && fill_next>=0){fu.ms=unsigned(fill_next);fu.pos=0;fu.first=true;fu.dirty=false;++*c_fill_chains;}else fu.v=false;}
            else{fu.dirty=fu.dirty || f.we;++fu.pos;}}
        // ---- S3: the response, the victim's writeback [10.2.3]
        if(s3_go){if(s3_push){Flit r=s3.f;r.dst=s3.f.src;s.rsq.push_back(r);}if(s3.wb)s.wbq.push_back(s3.wb_line);}
        if(!s3_hold)s3.v=false;   // S2's request (below) when it acts and does not merge
        // ---- S2's action
        if(s2==S2_REPLAY)s.rq[s1.rq].busy=false;
        else if(s2==S2_BLOCK){Rq& e=s.rq[s1.rq];e.busy=false;e.blocked=true;++*c_blocked;}
        else if(s2==S2_ACT){Rq& e=s.rq[s1.rq];const Flit f=e.f;e.v=false;e.busy=false;
            if(merge){s.tg[tg_free].v=true;s.tg[tg_free].f=f;s.ms[s2_ms].targets.push_back(unsigned(tg_free));++*c_merges;}
            else{s3.v=true;s3.f=f;s3.wb=false;s3.kind=kind;
                if(kind==K_HIT_RD || kind==K_HIT_WR){s3.index=set*ways+hit_way;if(f.we)s.dirty[s3.index]=true;++*c_hits;}
                else if(kind==K_INSTALL || kind==K_ALLOC){const unsigned i=set*ways+vic_way;s3.index=i;
                    s3.wb=s.valid[i] && s.dirty[i];s3.wb_line=(s.tag[i]<<set_w)|set;
                    s.victim_ptr[set]=(vic_way+1)%ways;s.tag[i]=tag_of(f.addr);++*c_misses;
                    if(kind==K_INSTALL){s.valid[i]=true;s.dirty[i]=true;++*c_installs;}
                    else{s.valid[i]=false;s.dirty[i]=false;s.claimed[i]=true;
                        Mshr& m=s.ms[ms_free];m.st=HP_REQ;m.line=f.addr>>6;m.set=set;m.way=vic_way;m.targets.assign(1,unsigned(tg_free));
                        s.tg[tg_free].v=true;s.tg[tg_free].f=f;}}}
        }
        // ---- S1: load, hold or empty; a tag S2 writes now in the set S1 reads is not in the read [10.2.2]
        if(!s2_hold){
            if(sel>=0){Rq& e=s.rq[sel];s1.v=true;s1.rq=unsigned(sel);s1.addr=e.f.addr;e.busy=true;
                s1.stale=s2==S2_ACT && (kind==K_INSTALL || kind==K_ALLOC) && set_of(e.f.addr)==set;}
            else{s1.v=false;s1.stale=false;}}
        // ---- resources freed (an MSHR's last fill): blocked requests try again, over S2's block of this edge
        if(free_resources)for(auto& e:s.rq)e.blocked=false;
    }
    //------------------------------------------------------------- the DRAM model [5] and its replay
    bool replay_allows_x(const Slice& s){
        const UncoreReplay::X* e=rp->next_x();(void)s;
        return !e || last_launch+e->cycle<=t;
    }
    DramEvent dram_cycle(bool& ready){
        DramEvent ev;
        if(replaying){ // ready when the next recorded accept is due; response when the next recorded Y is due
            ready=false;
            if(const UncoreReplay::X* e=rp->next_x())if(last_launch+e->cycle<=t)ready=true;
            if(const UncoreReplay::Y* y=rp->next_y())if(last_launch+y->cycle<=t){
                for(unsigned i=0;i<pool.size();++i)if(pool[i].v && pool[i].src==y->src && pool[i].tag==y->tag){ev.fire=true;ev.src=y->src;ev.tag=y->tag;rp->take_y();break;}
                if(!ev.fire)++rp->mismatches;}
            return ev;}
        ready=(lfsr>>5)&1u;
        // the response presented during this cycle was chosen at the previous edge
        ev=rsp_pending;rsp_pending=DramEvent{};
        return ev;
    }
    // at the edge: pick next cycle's response among the eligible entries (valid, cnt==0, pre-edge), lfsr[9:0] % nelig in ascending pool order [5]
    void dram_pick_next(){
        if(replaying)return;
        std::vector<unsigned> elig;for(unsigned i=0;i<pool.size();++i)if(pool[i].v && pool[i].cnt==0)elig.push_back(i);
        if(!elig.empty()){unsigned pick=unsigned(lfsr&1023u)%unsigned(elig.size());unsigned i=elig[pick];rsp_pending.fire=true;rsp_pending.src=pool[i].src;rsp_pending.tag=pool[i].tag;pool[i].v=false;}
    }
    void dram_accept(Slice& s,DramEvent& ev){
        (void)ev;const bool write=s.mreq.write;const u64 addr=s.mreq.addr;const unsigned tag=s.mreq.ms;
        if(trace){trace.M(t,s.id,write,addr,tag);trace.X(t,write,addr,s.id,tag);}
        if(replaying){const UncoreReplay::X* e=rp->next_x();if(!e || e->we!=write || e->addr!=addr || e->src!=s.id){++rp->mismatches;}if(e)rp->take_x();}
        if(write){++*c_dram_writes;return;}
        ++*c_dram_reads;
        for(auto& p:pool)if(!p.v){p.v=true;p.cnt=dram_latency-1;p.tag=tag;p.src=s.id;return;}
        require(false,"DRAM pool overflow");
    }
    void dram_deliver(const DramEvent& ev){
        if(replaying)for(auto& p:pool)if(p.v && p.src==ev.src && p.tag==ev.tag){p.v=false;break;}
        Slice& s=slices[ev.src];Mshr& m=s.ms[ev.tag];
        if(home_pipe){require(m.st==HP_WAIT,"memory response to a home MSHR not waiting");m.st=HP_FILL;}   // MS_FILL at E_r [10.2.4]
        else{require(m.st==3,"DRAM response to an MSHR not waiting");m.st=4;}
        if(trace){if(!board)trace.Y(t,ev.src,ev.tag);trace.D(t,ev.src,ev.tag);}
    }
    void pool_tick(){for(auto& p:pool)if(p.v && p.cnt)--p.cnt;}
};
inline void CycleCore::uncore_request(u64 id,u64 at,u64 addr,bool store,u64 be,u64 got){
    uncore->request(CycleUncore::Req{c.id,id,at,addr,store,be,got,(id&16)!=0,unsigned(id&31)});
}
inline bool CycleCore::uncore_ready()const{return uncore->req_ready(c.id);}
inline void CycleCore::uncore_launch(u64 pc,unsigned roles){uncore->launch_record(c.id,t,pc,roles);}
