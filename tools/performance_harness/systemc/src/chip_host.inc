//==============================================================================
// The platform's host side: the chip that holds the cores and the uncore, the
// command loop the host drives (launches, host words, flushes), the cycle count
// per command, --serve and --classify, and main. Included at the end of
// chip.cpp; evaluator-owned in the agent harness (align/README.md section 6).
//==============================================================================
// The chip: cores, NoC, NUCA homes, HBM; grid dispatch and host access.
struct Chip : sc_module {
    SC_HAS_PROCESS(Chip);
    Context& x;HBM& hbm;std::vector<std::unique_ptr<Slice>>& caches;CodeCache code;
    std::vector<std::unique_ptr<Core>> cores;std::unique_ptr<CycleUncore> uncore;   // uncore_model=1
    // Host-driven command interface: one pending command, one completion.
    struct Command {bool grid=false;Launch launch;u64 stride=0,core_mask=1,deadline=0;unsigned gx=1,gy=1,gz=1,group=1;};
    std::unique_ptr<Command> pending;bool command_done=false,flush_request=false,shutdown=false;
    Outcome result;u64 start_cycle=0,end_cycle=0,launches=0;sc_event wake;
    u64 *c_grid_launches,*c_single_launches,*c_host_cycles,*c_drain_cycles;
    Chip(sc_module_name n,Context& ctx,HBM& mem,Network& net,std::vector<std::unique_ptr<Slice>>& s):
        sc_module(n),x(ctx),hbm(mem),caches(s){
        code.read_word=[this](u64 pc){u64 v=0;unsigned char bytes[8];host_read(pc,bytes,8);for(unsigned b=0;b<8;++b)v|=u64(bytes[b])<<(8*b);return v;};
        for(unsigned i=0;i<x.c.cores();++i){
            cores.emplace_back(new Core(("core"+std::to_string(i)).c_str(),x,i,code));
            cores.back()->memory.bind(net.target);cores.back()->tensor.memory.bind(net.target);
        }
        c_grid_launches=&x.counts["chip_grid_launches"];c_single_launches=&x.counts["chip_single_launches"];
        // Bound only when the R41 launch drain is on, so default counters are unchanged.
        c_drain_cycles=x.c.get_or("tl_launch_drain",0)?&x.counts["chip_launch_drain_cycles"]:nullptr;
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
        // The code store (tnca_multicore_shell.v: CODE_WORDS words at 0) is not memory: host words below it never reach a slice.
        // With the R41 runtime's HBM DMA (uncore_host_dma=1) every word from LOCAL_LIMIT (= MEM_BASE, the code
        // store's end) goes by DMA after the ownership handoff, which writes back and invalidates the range in
        // the homes; without it every word goes through a home (UNCORE_TIMING 10.3, 7).
        if(uncore && address+n>16384ULL*8){
            if(uncore->host_dma)uncore->host_handoff(std::max<u64>(address,16384ULL*8),address+n);
            else for(u64 a=std::max<u64>(address&~7ULL,16384ULL*8);a<address+n;a+=8)uncore->host_touch(a,write);}
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
        auto next_block=[&](Core* ptr){
            Launch l=cmd.launch;l.px=x0;l.py=y0;l.pz=z0;l.nx=cmd.gx;l.ny=cmd.gy;l.nz=cmd.gz;l.arg=cmd.launch.arg+ptr->id*cmd.stride;
            ++issued;
            if(x0+cmd.group>=cmd.gx){x0=0;if(y0+1==cmd.gy){y0=0;++z0;}else ++y0;}else x0+=cmd.group;
            return l;};
        if(uncore){
            // tnca_grid_dispatch.v [UNCORE 6.2-6.3, BOARD 5]: one enqueue per cycle, the first
            // candidate from `rr` (the core after the last choice) that is active and neither
            // busy nor pending; the block is bound to the core at the enqueue (cycle P). The
            // command link launches the core at P+1+stages; a core that went idle at i is a
            // candidate again at i+1+stages (its program() returned at i-1); done_valid is
            // 1+stages cycles after the last idle. Sim bench: stages 0 (L+2, i+2, i+1);
            // board: 2 (L+4, i+6, i+3).
            const unsigned n=cores.size();const unsigned stages=uncore->dispatch_stages;
            std::vector<u64> ready_at(n,0);std::vector<char> busy(n,0),active(n,0);std::vector<Launch> slot(n);
            std::vector<sc_event> go(n);bool stop=false;
            for(auto& core:cores){
                if(!((cmd.core_mask>>core->id)&1))continue;
                active[core->id]=1;++live;auto* ptr=core.get();
                sc_spawn([&,ptr](){
                    for(;;){
                        wait(go[ptr->id]);if(stop)break;
                        wait(x.cycles(1+stages));   // the command link
                        Outcome o=ptr->program(slot[ptr->id]);retired+=o.retired;
                        if(o.status!=STATUS_OK && !faulted){faulted=true;first=o;}
                        ready_at[ptr->id]=x.now()+2+stages;busy[ptr->id]=0;
                    }
                    --live;done.notify(SC_ZERO_TIME);
                });
            }
            require(live>0 || total==0,"no active core for the grid");
            unsigned rr=0;
            if(total){
                // the first enqueue at L+1, after R41's admission stages at L+1+A (F06: A = 4, first launch L+8) [BOARD 9.3]
                u64 first_at=start_cycle+1+uncore->dispatch_admission;if(x.now()<first_at)wait(x.cycles(first_at-x.now()));
                while(issued<total && !faulted){
                    int choice=-1;
                    for(unsigned j=0;j<n && choice<0;++j){unsigned c=(rr+j)%n;if(active[c] && !busy[c] && ready_at[c]<=x.now())choice=c;}
                    if(choice>=0){
                        auto* ptr=cores[choice].get();slot[choice]=next_block(ptr);busy[choice]=1;rr=(choice+1)%n;
                        go[choice].notify();   // the launch happens in this cycle
                    }
                    wait(x.cycles(1));
                }
                for(;;){bool any=false;for(unsigned c=0;c<n;++c)any|=bool(busy[c]);if(!any)break;wait(x.cycles(1));}
            }
            stop=true;for(unsigned c=0;c<n;++c)if(active[c])go[c].notify();
            while(live)wait(done);
            wait(x.cycles(2+stages));   // done_valid: sim the cycle after the last idle cycle, board i+3 [UNCORE 6.3, BOARD 5.2]
            first.retired=retired;return first;
        }
        for(auto& core:cores){
            if(!((cmd.core_mask>>core->id)&1))continue;
            ++live;auto* ptr=core.get();
            sc_spawn([&,ptr](){
                while(issued<total && !faulted){
                    Launch l=next_block(ptr);
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
        if(cmd.grid){++*c_grid_launches;
            // R41 tnca_grid_dispatch.v: registered admission cycles before the first program (the event
            // model's tl_grid_admission_cycles; the cycle uncore counts them in dispatch(), uncore_dispatch_admission).
            if(!x.functional){u64 adm=x.c.get_or("tl_grid_admission_cycles",0);if(adm)wait(x.cycles(adm));}
            out=dispatch(cmd);}
        else{
            ++*c_single_launches;if(!x.functional)wait(x.cycles(cores[0]->t_launch));
            Launch l=cmd.launch;out=cores[0]->program(l);
        }
        // R41: done waits for memory_drained, every outlet transaction complete
        // (write responses of the launch's stores and evictions included).
        if(tm && !x.functional && x.c.get_or("tl_launch_drain",0)){
            u64 d=tm->outlets_idle(),n=x.now();if(d>n){*c_drain_cycles+=d-n;wait(x.cycles(d-n));}
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
        if(cfg.v.count("uncore_model") && cfg.get("uncore_model")==1){
            chip->uncore.reset(new CycleUncore("uncore",x,cfg.cores()));
            for(auto& core:chip->cores){require(bool(core->cyc),"uncore_model=1 requires core_model=1");chip->uncore->cores.push_back(core->cyc.get());core->cyc->uncore=chip->uncore.get();}
        }
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
