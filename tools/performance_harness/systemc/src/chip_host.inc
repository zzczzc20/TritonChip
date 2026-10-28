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
        // The code store (tnca_multicore_shell.v: CODE_WORDS words at 0) is not memory: host words below it never reach a slice.
        if(uncore)for(u64 a=std::max<u64>(address&~7ULL,16384ULL*8);a<address+n;a+=8)uncore->host_touch(a,write);
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
                u64 first_at=start_cycle+1;if(x.now()<first_at)wait(x.cycles(first_at-x.now()));   // the first enqueue at L+1
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
        if(cfg.v.count("uncore_model") && cfg.get("uncore_model")==1){
            chip->uncore.reset(new CycleUncore("uncore",x,cfg.cores()));
            for(auto& core:chip->cores){require(bool(core->cyc),"uncore_model=1 requires core_model=1");chip->uncore->cores.push_back(core->cyc.get());core->cyc->uncore=chip->uncore.get();}
        }
        if(cfg.get("memory_model")==1){
            tmem.reset(new TimelineMemory(cfg,x,hbm.memory));chip->tm=tmem.get();
            for(auto& core:chip->cores){core->tm=tmem.get();core->tensor.tm=tmem.get();core->tensor.port=&core->port;core->tensor.rtl=cfg.get("core_pipeline")!=0;}
        }
    }
    std::unique_ptr<TimelineMemory> tmem;
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
