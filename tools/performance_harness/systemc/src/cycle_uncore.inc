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
    static constexpr unsigned NV=2,VN_REQ=0,VN_RSP=1,NPORT=5,P_LOCAL=0,P_N=1,P_E=2,P_S=3,P_W=4,FIFO_DEPTH=4,RQ_N=8,MSHR_N=8,TARGET_N=16,RSQ_N=4;
    //------------------------------------------------------------- flits
    struct Flit {bool valid=false;unsigned dst=0,src=0,tag=0;u64 addr=0;bool we=false;unsigned be=0;u64 data=0;u64 id=0;unsigned core=0;bool engine=false;u64 got=0;};
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
    struct Rq {bool v=false,blocked=false;Flit f;u64 seq=0;};
    struct Mshr {unsigned st=0;/*0 FREE,1 WB,2 REQ,3 WAIT,4 FILL,5 WAKE*/u64 line=0,victim_line=0;unsigned set=0,way=0;std::vector<unsigned> targets;bool byp=false;};
    struct Tgt {bool v=false;Flit f;};
    struct Slice {
        unsigned id=0;std::vector<u64> tag;std::vector<bool> valid,dirty;std::vector<unsigned> victim_ptr;
        std::array<Rq,RQ_N> rq;std::array<Mshr,MSHR_N> ms;std::array<Tgt,TARGET_N> tg;std::deque<Flit> rsq;u64 seq=0;
        unsigned est=0;/*0 SEL,1 LOOK,2 HIT,3 MISS,4 MREQ,5 WBRD,6 WBSND,7 FILL,8 WAKE*/unsigned cur_rq=0,cur_ms=0;bool cur_is_ms=false;
        // the memory request this cycle presents (from the pre-edge state) [3.5]
        struct MemReq {bool valid=false,write=false;u64 addr=0;unsigned ms=0;} mreq;
        bool mem_req_valid()const{return mreq.valid;}
    };
    std::vector<Slice> slices;
    //------------------------------------------------------------- the board's memory side [B 1-4]
    bool board=false;unsigned home_stages=0,outlet_stages=0,dispatch_stages=0,active_outlets=1;
    // tnca_memory_mode [B 2.3]: with every outlet active home h is port h (zero cycles); with one, the three
    // homes share port 0 through a tnca_mem_arbiter (round robin + hold, UNCORE_TIMING 4.2)
    unsigned port_of(unsigned home)const{return active_outlets==1?0:home%active_outlets;}
    struct DramEvent {bool fire=false;unsigned src=0,tag=0;};
    struct Port {
        std::vector<Fifo> hreq,oreq;                       // home link, outlet link: request chains of 2-deep stages [B 1.1]
        struct Slot {bool active=false,writing=false,sent=false;u64 addr=0;unsigned src=0,tag=0,rbeat=0;};std::array<Slot,8> slot;   // tnca_line_axi [B 3.1]
        bool issue_valid=false;unsigned issue_slot=0,issue_age=0;unsigned admission=8;
        unsigned latency=25,write_delay=15;                // tnca_axi_line_model per port [B 4]
        struct Rd {bool v=false;u64 due=0;unsigned beat=0;};std::array<Rd,8> rd;
        struct Wr {bool v=false;u64 due=0;};std::array<Wr,8> wr;
        bool rvalid=false;unsigned rid=0;bool rlast=false; // the R beat presented during this cycle (picked at the previous edge)
        bool bvalid=false;unsigned bid=0;                  // the B presented during this cycle
        bool rsp_valid=false;unsigned rsp_src=0,rsp_tag=0; // the adapter's rsp_valid during this cycle [B 3.4]
        unsigned r_xi=0,r_yi=0;                            // replay cursors over this port's X/Y records
        unsigned rr=0;bool held=false;unsigned held_pick=0; // the port's arbiter among its homes' link heads
    };
    std::vector<Port> ports;
    struct Due {u64 when;DramEvent ev;};std::vector<Due> slice_rsp;   // responses on their way to a slice (outlet + home link registers, never stalled)
    //------------------------------------------------------------- memory arbiter [4.2] + DRAM model [5]
    unsigned arb_rr=0;bool arb_held=false;unsigned arb_held_pick=0;
    struct Pool {bool v=false;unsigned cnt=0,tag=0,src=0;u64 id=0;unsigned core=0;u64 got=0;};
    std::array<Pool,32> pool;std::uint32_t lfsr=0xACE12345u;unsigned dram_latency=11;
    DramEvent rsp_pending;   // rsp_valid during the coming cycle (LFSR path)
    // replay of the device port (X accept cycles, Y response cycles)
    struct ReplayX {u64 cycle;bool we;u64 addr;unsigned src,tag;};struct ReplayY {u64 cycle;unsigned src,tag;};
    std::vector<std::vector<ReplayX>> rx;std::vector<std::vector<ReplayY>> ry;unsigned r_launch=0,r_xi=0,r_yi=0;bool r_open=false;u64 r_start=0;u64 r_mismatch=0;
    bool replaying=false;
    //------------------------------------------------------------- misc
    struct Req {unsigned core;u64 id,at,addr;bool store;unsigned be;u64 got;bool engine;unsigned tag;};
    std::deque<Req> incoming;u64 t=0;std::ofstream trace;u64 last_launch=0;std::ofstream port_debug;   // TNCA_MODEL_PORT_DEBUG=<path>: why an adapter refuses
    u64 *c_hits,*c_misses,*c_merges,*c_blocked,*c_wb,*c_dram_reads,*c_dram_writes,*c_rsq_full,*c_mreq_stall;
    CycleUncore(sc_module_name n,Context& ctx,unsigned cores_):sc_module(n),x(ctx),ncores(cores_){
        nx=unsigned(x.c.get("nx"));ny=unsigned(x.c.get("ny"));sets=unsigned(x.c.get("sets"));ways=unsigned(x.c.get("ways"));
        set_w=0;while((1u<<set_w)<sets){++set_w;}link_stages=x.c.v.count("tl_noc_link_stages")?unsigned(x.c.get("tl_noc_link_stages")):0;
        routers.resize(nx*ny);for(unsigned i=0;i<routers.size();++i){routers[i].xh=i%nx;routers[i].yh=i/nx;}
        for(unsigned node=0;node<routers.size();++node)for(unsigned p=1;p<NPORT;++p)if(port_exists(node,p))for(unsigned vn=0;vn<NV;++vn){
            Link& l=links[std::make_tuple(node,p,vn)];l.stages.resize(link_stages);for(auto& s:l.stages)s.depth=2;}
        slices.resize(nx*ny);
        for(unsigned i=0;i<slices.size();++i){Slice& s=slices[i];s.id=i;s.tag.assign(sets*ways,0);s.valid.assign(sets*ways,false);s.dirty.assign(sets*ways,false);s.victim_ptr.assign(sets,0);}
        dram_latency=x.c.v.count("tl_dram_latency")?unsigned(x.c.get("tl_dram_latency")):11;
        board=x.c.v.count("uncore_board") && x.c.get("uncore_board")==1;
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
                for(auto& st:q.hreq){st.depth=2;}
                for(auto& st:q.oreq){st.depth=2;}
                q.latency=lat+step*p;q.write_delay=wdelay;q.admission=x.c.v.count("outstanding")?unsigned(x.c.get("outstanding")):8;}
        }
        c_hits=&x.counts["uncore_hits"];c_misses=&x.counts["uncore_misses"];c_merges=&x.counts["uncore_merges"];c_blocked=&x.counts["uncore_blocked"];
        c_wb=&x.counts["uncore_writebacks"];c_dram_reads=&x.counts["uncore_dram_reads"];c_dram_writes=&x.counts["uncore_dram_writes"];c_rsq_full=&x.counts["uncore_rsq_full"];c_mreq_stall=&x.counts["uncore_mreq_stall"];
        if(const char* p=std::getenv("TNCA_MODEL_CORE_TRACE")){if(*p){std::string f=std::string(p)=="1"?x.directory+"/uncore_trace.txt":std::string(p)+".uncore";
            trace.open(f);if(trace)trace<<"TNCA_UNCORE_TRACE 1 cores="<<ncores<<"\n";}}
        if(const char* p=std::getenv("TNCA_MODEL_UNCORE_REPLAY")){if(*p){load_replay(p);replaying=true;}}
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
    }
    //------------------------------------------------------------- requests from the cores
    bool req_ready(unsigned core)const{return routers[core].in[P_LOCAL][VN_REQ].in_ready();}   // [1.3]
    void request(const Req& r){incoming.push_back(r);}
    ~CycleUncore(){if(trace){trace<<"END "<<t<<"\n";trace.close();}}
    void launch_record(unsigned core,u64 cycle,u64 pc,unsigned roles){
        last_launch=cycle;if(replaying)replay_launch();
        if(trace){char line[96];std::snprintf(line,sizeof line,"L %llu %u %016llx %u\n",(unsigned long long)cycle,core,(unsigned long long)pc,roles);trace<<line;}}
    void run(){
        wait(x.cycles(1)/2);
        for(;;){t=x.to_cycles(sc_time_stamp()-x.cycles(1)/2);step();wait(x.cycles(1));}
    }
    //------------------------------------------------------------- replay of the device port
    void load_replay(const std::string& path){
        std::ifstream f(path);require(bool(f),"cannot open the uncore replay trace");std::string line;bool in=false;
        while(std::getline(f,line)){std::istringstream s(line);std::string k;s>>k;
            if(k=="L"){u64 c;s>>c;rx.emplace_back();ry.emplace_back();in=true;r_start=c;}
            else if(in && k=="X"){u64 c,we,addr,src,tag;s>>std::dec>>c>>we>>std::hex>>addr>>std::dec>>src>>tag;rx.back().push_back({c-r_start,we!=0,addr,unsigned(src),unsigned(tag)});}
            else if(in && k=="Y"){u64 c,src,tag;s>>std::dec>>c>>src>>tag;ry.back().push_back({c-r_start,unsigned(src),unsigned(tag)});}}
        require(!rx.empty(),"uncore replay trace has no launches");
    }
    void replay_launch(){if(r_open){++r_launch;}r_open=true;r_xi=0;r_yi=0;for(auto& q:ports){q.r_xi=0;q.r_yi=0;}}
    // the next recorded X / Y of port p (its src is the home = the port) [B 7]
    const ReplayX* next_x(unsigned p){if(r_launch>=rx.size())return nullptr;auto& v=rx[r_launch];Port& q=ports[p];while(q.r_xi<v.size() && port_of(v[q.r_xi].src)!=p)++q.r_xi;return q.r_xi<v.size()?&v[q.r_xi]:nullptr;}
    const ReplayY* next_y(unsigned p){if(r_launch>=ry.size())return nullptr;auto& v=ry[r_launch];Port& q=ports[p];while(q.r_yi<v.size() && port_of(v[q.r_yi].src)!=p)++q.r_yi;return q.r_yi<v.size()?&v[q.r_yi]:nullptr;}
    //------------------------------------------------------------- one cycle: decisions from the pre-edge state, then the edge
    void step(){
        //---- requests fired by the cores during t: pushed into the source router's local VN_REQ FIFO at E_t [1.1 #3-4]
        while(!incoming.empty()){Req r=incoming.front();incoming.pop_front();Flit f;f.valid=true;f.src=r.core;f.dst=home(r.addr);f.addr=r.addr;f.we=r.store;f.be=r.be;f.data=0;f.tag=r.tag;f.id=r.id;f.core=r.core;f.engine=r.engine;f.got=r.got;
            Fifo& q=routers[r.core].in[P_LOCAL][VN_REQ];require(q.in_ready(),"core fired into a full NoC FIFO");q.push(f);}
        //---- the memory request each slice presents this cycle (pre-edge engine state) [3.5]
        for(auto& s:slices){s.mreq=Slice::MemReq{};if(s.est==4||s.est==6){const Mshr& m=s.ms[s.cur_ms];s.mreq.valid=true;s.mreq.write=s.est==6;s.mreq.addr=(s.est==6?m.victim_line:m.line)<<6;s.mreq.ms=s.cur_ms;}}
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
        for(auto& s:slices)slice_cycle(s,board?hgrant[s.id]:grant==int(s.id),slice_in,dram_ready);
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
            if(trace){char line[64];std::snprintf(line,sizeof line,"R %llu %u %u %u\n",(unsigned long long)t,s.id,f.src,f.tag);trace<<line;}}
        if(board){
            // the responses valid at the slices during t are captured now [B 1.2]
            for(auto& ev:board_rsp)dram_deliver(ev);
            // slices fire into their home links [B 1.2]
            for(auto& s:slices)if(hgrant[s.id]){Flit f;f.valid=true;f.we=s.mreq.write;f.addr=s.mreq.addr;f.tag=s.mreq.ms;f.src=s.id;ports[s.id].hreq[0].push(f);
                if(trace){char line[96];std::snprintf(line,sizeof line,"M %llu %u %u %016llx %u\n",(unsigned long long)t,s.id,f.we?1U:0U,(unsigned long long)f.addr,f.tag);trace<<line;}
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
        Fifo& last=q.oreq.back();bool accept=false;
        if(last.out_valid()){const Flit& f=last.head();
            int free_slot=-1;for(unsigned k=0;k<8;++k)if(!q.slot[k].active){free_slot=int(k);break;}
            bool hazard=false;unsigned active_count=0;
            for(unsigned k=0;k<8;++k){const Port::Slot& sl=q.slot[k];if(!sl.active)continue;++active_count;if(sl.addr==f.addr && (sl.writing || f.we))hazard=true;}
            bool ready=!q.issue_valid && free_slot>=0 && !hazard && active_count<q.admission;
            if(ready && replaying){const ReplayX* e=next_x(p);if(e && last_launch+e->cycle>t)ready=false;}   // no earlier than the recorded accept
            if(port_debug && !ready){const ReplayX* e=replaying?next_x(p):nullptr;
                port_debug<<t<<" port "<<p<<" held: addr="<<std::hex<<f.addr<<std::dec<<" issue="<<q.issue_valid<<"/"<<q.issue_age<<" free="<<free_slot<<" hazard="<<hazard<<" active="<<active_count
                          <<" next_x="<<(e?long(last_launch+e->cycle):-1L)<<" last_launch="<<last_launch<<"\n";}
            if(ready){accept=true;
                Port::Slot& sl=q.slot[free_slot];sl.active=true;sl.writing=f.we;sl.sent=false;sl.addr=f.addr;sl.src=f.src;sl.tag=f.tag;sl.rbeat=0;
                q.issue_valid=true;q.issue_slot=unsigned(free_slot);q.issue_age=0;
                if(trace){char line[96];std::snprintf(line,sizeof line,"X %llu %u %016llx %u %u\n",(unsigned long long)t,f.we?1U:0U,(unsigned long long)f.addr,f.src,f.tag);trace<<line;}
                if(replaying){const ReplayX* e=next_x(p);if(!e || e->we!=f.we || e->addr!=f.addr || e->src!=f.src){++r_mismatch;}if(e)++q.r_xi;}
                last.pop();}
        }
        // ---- the issue register: AR during a+1; AW+W0 during a+1, W1 during a+2 (every AXI ready high) [B 3.3]
        if(q.issue_valid && !accept){++q.issue_age;Port::Slot& sl=q.slot[q.issue_slot];
            if(!sl.writing && q.issue_age==1){q.issue_valid=false;sl.sent=true;
                q.rd[q.issue_slot]=Port::Rd{true,t+q.latency+(q.issue_slot*11)%31,0};}          // due: eligible at E_{x+L+skew}, first beat during +1 [B 4.2]
            else if(sl.writing && q.issue_age==2){q.issue_valid=false;sl.sent=true;
                q.wr[q.issue_slot]=Port::Wr{true,t+q.latency+q.write_delay+(q.issue_slot*7)%23};} // B during due+1 [B 4.3]
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
        if(replaying){const ReplayY* y=next_y(p);
            if(y && last_launch+y->cycle<=t+1){const u64 yc=last_launch+y->cycle;bool found=false;
                for(unsigned k=0;k<8;++k){Port::Slot& sl=q.slot[k];if(sl.active && !sl.writing && sl.src==y->src && sl.tag==y->tag){sl.active=false;q.rd[k].v=false;found=true;break;}}
                if(found){slice_rsp.push_back({yc+outlet_stages+home_stages,DramEvent{true,y->src,y->tag}});
                    if(trace){char line[64];std::snprintf(line,sizeof line,"Y %llu %u %u\n",(unsigned long long)yc,y->src,y->tag);trace<<line;}}
                else ++r_mismatch;
                ++q.r_yi;}
        }else if(q.rsp_valid){ // set at the previous edge: valid during t [B 3.4], at the slice +outlet +home [B 1.2]
            slice_rsp.push_back({t+outlet_stages+home_stages,DramEvent{true,q.rsp_src,q.rsp_tag}});
            if(trace){char line[64];std::snprintf(line,sizeof line,"Y %llu %u %u\n",(unsigned long long)t,q.rsp_src,q.rsp_tag);trace<<line;}
            q.rsp_valid=false;}
        // ---- the slave's picks for t+1: the lowest due read's next beat; the lowest due write's B [B 4.2-4.3]
        if(!replaying){int pick=-1;for(unsigned k=0;k<8;++k)if(q.rd[k].v && q.rd[k].due<=t){pick=int(k);break;}
            if(pick>=0){q.rvalid=true;q.rid=unsigned(pick);q.rlast=q.rd[pick].beat==1;}}
        if(!q.bvalid){int pick=-1;for(unsigned k=0;k<8;++k)if(q.wr[k].v && q.wr[k].due<=t){pick=int(k);break;}
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
        for(unsigned i=0;i<RQ_N;++i)if(!s.rq[i].v){s.rq[i].v=true;s.rq[i].blocked=false;s.rq[i].f=f;s.rq[i].seq=s.seq++;
            if(trace){char line[96];std::snprintf(line,sizeof line,"S %llu %u %u %u %u %016llx\n",(unsigned long long)t,s.id,f.src,f.tag,f.we?1U:0U,(unsigned long long)f.addr);trace<<line;}
            return;}
        require(false,"slice request buffer overflow");
    }
    int select_request(const Slice& s)const{
        int best=-1;
        for(unsigned i=0;i<RQ_N;++i){const Rq& e=s.rq[i];if(!e.v || e.blocked)continue;
            bool shadowed=false;for(unsigned j=0;j<RQ_N;++j){const Rq& o=s.rq[j];if(j!=i && o.v && o.seq<e.seq && (o.f.addr>>6)==(e.f.addr>>6))shadowed=true;}
            if(shadowed){continue;}if(best<0 || e.seq<s.rq[best].seq)best=int(i);}
        return best;
    }
    int ms_pick(const Slice& s)const{   // WAKE > FILL > WB > REQ, lowest index
        for(unsigned st:{5u,4u,1u,2u})for(unsigned i=0;i<MSHR_N;++i)if(s.ms[i].st==st)return int(i);
        return -1;
    }
    int ms_match(const Slice& s,u64 addr)const{for(unsigned i=0;i<MSHR_N;++i){if(s.ms[i].st!=0 && s.ms[i].line==(addr>>6))return int(i);}return -1;}
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
                else{int ti=-1;for(unsigned k=0;k<TARGET_N;++k)if(!s.tg[k].v){ti=int(k);break;}
                    if(ti>=0){s.tg[ti].v=true;s.tg[ti].f=f;m.targets.push_back(unsigned(ti));e.v=false;++*c_merges;}else{e.blocked=true;++*c_blocked;}}
            }else{
                int mfree=-1;for(unsigned k=0;k<MSHR_N;++k)if(s.ms[k].st==0){mfree=int(k);break;}
                int tfree=-1;for(unsigned k=0;k<TARGET_N;++k)if(!s.tg[k].v){tfree=int(k);break;}
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
    //------------------------------------------------------------- the DRAM model [5] and its replay
    bool replay_allows_x(const Slice& s){
        if(r_launch>=rx.size() || r_xi>=rx[r_launch].size())return true;
        const ReplayX& e=rx[r_launch][r_xi];return last_launch+e.cycle<=t;(void)s;
    }
    DramEvent dram_cycle(bool& ready){
        DramEvent ev;
        if(replaying){ // ready when the next recorded accept is due; response when the next recorded Y is due
            ready=false;
            if(r_launch<rx.size() && r_xi<rx[r_launch].size() && last_launch+rx[r_launch][r_xi].cycle<=t)ready=true;
            if(r_launch<ry.size() && r_yi<ry[r_launch].size() && last_launch+ry[r_launch][r_yi].cycle<=t){
                const ReplayY& y=ry[r_launch][r_yi];
                for(unsigned i=0;i<pool.size();++i)if(pool[i].v && pool[i].src==y.src && pool[i].tag==y.tag){ev.fire=true;ev.src=y.src;ev.tag=y.tag;++r_yi;break;}
                if(!ev.fire)++r_mismatch;}
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
        if(trace){char line[96];std::snprintf(line,sizeof line,"M %llu %u %u %016llx %u\nX %llu %u %016llx %u %u\n",(unsigned long long)t,s.id,write?1U:0U,(unsigned long long)addr,tag,
            (unsigned long long)t,write?1U:0U,(unsigned long long)addr,s.id,tag);trace<<line;}
        if(replaying){const ReplayX& e=rx[r_launch][r_xi];if(e.we!=write || e.addr!=addr || e.src!=s.id){++r_mismatch;}++r_xi;}
        if(write){++*c_dram_writes;return;}
        ++*c_dram_reads;
        for(auto& p:pool)if(!p.v){p.v=true;p.cnt=dram_latency-1;p.tag=tag;p.src=s.id;return;}
        require(false,"DRAM pool overflow");
    }
    void dram_deliver(const DramEvent& ev){
        if(replaying)for(auto& p:pool)if(p.v && p.src==ev.src && p.tag==ev.tag){p.v=false;break;}
        Slice& s=slices[ev.src];Mshr& m=s.ms[ev.tag];require(m.st==3,"DRAM response to an MSHR not waiting");m.st=4;
        if(trace){char line[64];
            if(board)std::snprintf(line,sizeof line,"D %llu %u %u\n",(unsigned long long)t,ev.src,ev.tag);
            else std::snprintf(line,sizeof line,"Y %llu %u %u\nD %llu %u %u\n",(unsigned long long)t,ev.src,ev.tag,(unsigned long long)t,ev.src,ev.tag);
            trace<<line;}
    }
    void pool_tick(){for(auto& p:pool)if(p.v && p.cnt)--p.cnt;}
};
inline void CycleCore::uncore_request(u64 id,u64 at,u64 addr,bool store,unsigned be,u64 got){
    uncore->request(CycleUncore::Req{c.id,id,at,addr,store,be,got,(id&16)!=0,unsigned(id&31)});
}
inline bool CycleCore::uncore_ready()const{return uncore->req_ready(c.id);}
inline void CycleCore::uncore_launch(u64 pc,unsigned roles){uncore->launch_record(c.id,t,pc,roles);}
