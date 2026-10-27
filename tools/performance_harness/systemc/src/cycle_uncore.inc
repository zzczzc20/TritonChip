//==============================================================================
// The cycle-level uncore (uncore_model=1): the path from the cores' memory
// ports to the device's memory port -- core NI, mesh, home slices, memory
// arbiter, DRAM model -- stepped once per cycle, half a cycle after the cores
// (its edge E_t runs at SC time t+0.5: it sees what the cores fired during t,
// and what it makes valid for cycle t+1 reaches them at their step t+1).
// tools/performance_harness/align/UNCORE_TIMING.md is the rule book ([1.1],
// [2.4] ... cite its sections). This version is the simulation bench's
// configuration (PORTS=1, tnca_dram_model, no link stages); the board's links
// and outlet are the next step.
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
    //------------------------------------------------------------- memory arbiter [4.2] + DRAM model [5]
    unsigned arb_rr=0;bool arb_held=false;unsigned arb_held_pick=0;
    struct Pool {bool v=false;unsigned cnt=0,tag=0,src=0;u64 id=0;unsigned core=0;u64 got=0;};
    std::array<Pool,32> pool;std::uint32_t lfsr=0xACE12345u;unsigned dram_latency=11;
    struct DramEvent {bool fire=false;unsigned src=0,tag=0;};DramEvent rsp_pending;   // rsp_valid during the coming cycle (LFSR path)
    // replay of the device port (X accept cycles, Y response cycles)
    struct ReplayX {u64 cycle;bool we;u64 addr;unsigned src,tag;};struct ReplayY {u64 cycle;unsigned src,tag;};
    std::vector<std::vector<ReplayX>> rx;std::vector<std::vector<ReplayY>> ry;unsigned r_launch=0,r_xi=0,r_yi=0;bool r_open=false;u64 r_start=0;u64 r_mismatch=0;
    bool replaying=false;
    //------------------------------------------------------------- misc
    struct Req {unsigned core;u64 id,at,addr;bool store;unsigned be;u64 got;bool engine;unsigned tag;};
    std::deque<Req> incoming;u64 t=0;std::ofstream trace;u64 last_launch=0;
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
        c_hits=&x.counts["uncore_hits"];c_misses=&x.counts["uncore_misses"];c_merges=&x.counts["uncore_merges"];c_blocked=&x.counts["uncore_blocked"];
        c_wb=&x.counts["uncore_writebacks"];c_dram_reads=&x.counts["uncore_dram_reads"];c_dram_writes=&x.counts["uncore_dram_writes"];c_rsq_full=&x.counts["uncore_rsq_full"];c_mreq_stall=&x.counts["uncore_mreq_stall"];
        if(const char* p=std::getenv("TNCA_MODEL_CORE_TRACE")){if(*p){std::string f=std::string(p)=="1"?x.directory+"/uncore_trace.txt":std::string(p)+".uncore";
            trace.open(f);if(trace)trace<<"TNCA_UNCORE_TRACE 1 cores="<<ncores<<"\n";}}
        if(const char* p=std::getenv("TNCA_MODEL_UNCORE_REPLAY")){if(*p){load_replay(p);replaying=true;}}
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
    void replay_launch(){if(r_open){++r_launch;}r_open=true;r_xi=0;r_yi=0;}
    //------------------------------------------------------------- one cycle: decisions from the pre-edge state, then the edge
    void step(){
        //---- requests fired by the cores during t: pushed into the source router's local VN_REQ FIFO at E_t [1.1 #3-4]
        while(!incoming.empty()){Req r=incoming.front();incoming.pop_front();Flit f;f.valid=true;f.src=r.core;f.dst=home(r.addr);f.addr=r.addr;f.we=r.store;f.be=r.be;f.data=0;f.tag=r.tag;f.id=r.id;f.core=r.core;f.engine=r.engine;f.got=r.got;
            Fifo& q=routers[r.core].in[P_LOCAL][VN_REQ];require(q.in_ready(),"core fired into a full NoC FIFO");q.push(f);}
        //---- DRAM: this cycle's ready and the response it presents [5]
        bool dram_ready;DramEvent dev=dram_cycle(dram_ready);
        //---- the memory request each slice presents this cycle (pre-edge engine state) [3.5]
        for(auto& s:slices){s.mreq=Slice::MemReq{};if(s.est==4||s.est==6){const Mshr& m=s.ms[s.cur_ms];s.mreq.valid=true;s.mreq.write=s.est==6;s.mreq.addr=(s.est==6?m.victim_line:m.line)<<6;s.mreq.ms=s.cur_ms;}}
        //---- slice engines decide (pre-edge state); their memory requests want the arbiter [3][4.2]
        std::vector<int> want;for(auto& s:slices)if(s.mem_req_valid())want.push_back(int(s.id));
        int grant=-1;
        if(!want.empty()){unsigned pick=arb_held?arb_held_pick:0;bool ok=false;
            if(arb_held){ok=slices[arb_held_pick].mem_req_valid();pick=arb_held_pick;}
            if(!ok){for(unsigned ai=0;ai<slices.size();++ai){unsigned g=(arb_rr+ai)%slices.size();if(slices[g].mem_req_valid()){pick=g;ok=true;break;}}}
            if(ok){if(dram_ready)grant=int(pick);else{arb_held=true;arb_held_pick=pick;}}}
        if(replaying && grant>=0 && !replay_allows_x(slices[grant])){grant=-1;}
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
        for(auto& s:slices)slice_cycle(s,grant==int(s.id),slice_in,dram_ready);
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
        // DRAM: the response valid during t is captured by its slice now; the pool counts down; the accept; next cycle's response
        if(dev.fire)dram_deliver(dev);
        pool_tick();
        if(grant>=0){Slice& s=slices[grant];dram_accept(s,dev);arb_held=false;arb_rr=(unsigned(grant)+1)%slices.size();}
        dram_pick_next();
        // FIFO edges
        for(auto& r:routers)for(auto& pq:r.in)for(auto& q:pq)q.edge();
        for(auto& kv:links)for(auto& st:kv.second.stages)st.edge();
        lfsr=(lfsr<<1)|(((lfsr>>31)^(lfsr>>21)^(lfsr>>1)^lfsr)&1u);
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
        if(trace){char line[64];std::snprintf(line,sizeof line,"Y %llu %u %u\nD %llu %u %u\n",(unsigned long long)t,ev.src,ev.tag,(unsigned long long)t,ev.src,ev.tag);trace<<line;}
    }
    void pool_tick(){for(auto& p:pool)if(p.v && p.cnt)--p.cnt;}
};
inline void CycleCore::uncore_request(u64 id,u64 at,u64 addr,bool store,unsigned be,u64 got){
    uncore->request(CycleUncore::Req{c.id,id,at,addr,store,be,got,(id&16)!=0,unsigned(id&31)});
}
inline bool CycleCore::uncore_ready()const{return uncore->req_ready(c.id);}
inline void CycleCore::uncore_launch(u64 pc,unsigned roles){uncore->launch_record(c.id,t,pc,roles);}
