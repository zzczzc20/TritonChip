//==============================================================================
// The platform: the model's instruments (tools/performance_harness/align/README.md
// section 6). Every trace record the model writes and every recorded boundary
// it replays goes through the structs here and nowhere else: device code (Core,
// CycleCore, CycleUncore and the engines) records through these functions and
// takes a recorded boundary through these functions; it never opens a trace or
// a recording itself. A replay input is a boundary file (`TNCA_REPLAY 1`, the
// one stream the boundary needs, cycles relative to each launch), never a
// trace: align/replay_extract.py makes one from a recording. In the agent
// harness this file is evaluator-owned, like the testbench and the relay.
//==============================================================================

// The core trace (align/README.md section 2): TNCA_MODEL_CORE_TRACE names the
// file ("1": core_trace.txt in the run directory), TNCA_MODEL_CORE_TRACE_CORE
// the core (default 0; "all": every core, core c>0 to <file>.core<c>, as the
// RTL monitor names them). Off when the variable is unset.
struct CoreTrace {
    std::ofstream out,decode;u64 launches=0;std::string decode_path;std::unordered_map<u64,bool> decoded;
    // the core TNCA_MODEL_CORE_TRACE_CORE selects (the replay follows the same choice)
    static bool selected(unsigned id){
        const char* which=std::getenv("TNCA_MODEL_CORE_TRACE_CORE");
        if(which && std::string(which)=="all")return true;
        return unsigned(which && *which?std::strtoul(which,nullptr,10):0)==id;
    }
    void open(const Context& x,unsigned id){
        const char* path=std::getenv("TNCA_MODEL_CORE_TRACE");if(!path || !*path || !selected(id))return;
        const bool own=std::string(path)=="1";   // the run's own directory: one trace per point under tlm_backend
        std::string file=own?x.directory+"/core_trace.txt":std::string(path);
        decode_path=own?x.directory+"/core_decode.txt":std::string(path)+".decode";
        if(id)file+=".core"+std::to_string(id);
        out.open(file);require(bool(out),"cannot create the core trace");
        out<<"TNCA_CORE_TRACE 1 core="<<id<<"\n";
    }
    explicit operator bool()const{return out.is_open();}
    void L(u64 cycle,u64 pc,unsigned roles){++launches;char line[64];
        std::snprintf(line,sizeof line,"L %llu %016llx %u\n",(unsigned long long)cycle,(unsigned long long)pc,roles);out<<line;}
    void I(u64 cycle,unsigned rctx,u64 pc){char line[64];
        std::snprintf(line,sizeof line,"I %llu %u %016llx\n",(unsigned long long)cycle,rctx,(unsigned long long)pc);out<<line;}
    void Q(u64 cycle,bool we,u64 addr,unsigned be,u64 tag,u64 wdata){char line[96];
        std::snprintf(line,sizeof line,"Q %llu %u %016llx %02x %llu 0 %016llx\n",(unsigned long long)cycle,we?1U:0U,
            (unsigned long long)addr,be,(unsigned long long)tag,(unsigned long long)wdata);out<<line;}
    void P(u64 cycle,u64 tag,u64 rdata){char line[64];
        std::snprintf(line,sizeof line,"P %llu %llu %016llx\n",(unsigned long long)cycle,(unsigned long long)tag,(unsigned long long)rdata);out<<line;}
    // R41's line-wide core port (HYBRID_LSU, tnca_core_trace.v at MEM_DW 512): 64 byte enables and
    // 512 data bits, most significant word first, as the RTL monitor prints them.
    static void hex512(char* p,const std::array<u64,8>& d){for(int w=7;w>=0;--w){std::snprintf(p,17,"%016llx",(unsigned long long)d[unsigned(w)]);p+=16;}}
    void QW(u64 cycle,bool we,u64 addr,u64 be,u64 tag,const std::array<u64,8>& wdata){char h[129];hex512(h,wdata);char line[240];
        std::snprintf(line,sizeof line,"Q %llu %u %016llx %016llx %llu 0 %s\n",(unsigned long long)cycle,we?1U:0U,
            (unsigned long long)addr,(unsigned long long)be,(unsigned long long)tag,h);out<<line;}
    void PW(u64 cycle,u64 tag,const std::array<u64,8>& rdata){char h[129];hex512(h,rdata);char line[180];
        std::snprintf(line,sizeof line,"P %llu %llu %s\n",(unsigned long long)cycle,(unsigned long long)tag,h);out<<line;}
    void E(u64 cycle){out<<"E "<<cycle<<"\n";out.flush();}
    // Beside the trace: one line per distinct pc, `K pc kind major func mod d a b c` (what the model decoded there)
    void K(u64 pc,const char* kind,unsigned major,unsigned func,unsigned mod,unsigned d,unsigned a,unsigned b,unsigned c){
        if(decoded.count(pc)){return;}
        decoded[pc]=true;
        if(!decode.is_open()){decode.open(decode_path);}
        decode<<"K "<<std::hex<<pc<<std::dec<<' '<<kind<<' '<<major<<' '<<func<<' '<<mod<<' '<<d<<' '<<a<<' '<<b<<' '<<c<<"\n";}
    void close(u64 cycle){if(out.is_open()){out<<"END "<<launches<<' '<<cycle<<"\n";out.close();}}
};

// The uncore trace (align/README.md section 2b): with TNCA_MODEL_CORE_TRACE set
// it is uncore_trace.txt in the run directory ("1") or <file>.uncore.
struct UncoreTrace {
    std::ofstream out;
    void open(const Context& x,unsigned ncores){
        const char* p=std::getenv("TNCA_MODEL_CORE_TRACE");if(!p || !*p)return;
        std::string f=std::string(p)=="1"?x.directory+"/uncore_trace.txt":std::string(p)+".uncore";
        out.open(f);if(out)out<<"TNCA_UNCORE_TRACE 1 cores="<<ncores<<"\n";
    }
    explicit operator bool()const{return out.is_open();}
    void L(u64 cycle,unsigned core,u64 pc,unsigned roles){char line[96];
        std::snprintf(line,sizeof line,"L %llu %u %016llx %u\n",(unsigned long long)cycle,core,(unsigned long long)pc,roles);out<<line;}
    void S(u64 cycle,unsigned slice,unsigned src,unsigned tag,bool we,u64 addr){char line[96];
        std::snprintf(line,sizeof line,"S %llu %u %u %u %u %016llx\n",(unsigned long long)cycle,slice,src,tag,we?1U:0U,(unsigned long long)addr);out<<line;}
    void R(u64 cycle,unsigned slice,unsigned src,unsigned tag){char line[64];
        std::snprintf(line,sizeof line,"R %llu %u %u %u\n",(unsigned long long)cycle,slice,src,tag);out<<line;}
    void M(u64 cycle,unsigned slice,bool we,u64 addr,unsigned tag){char line[96];
        std::snprintf(line,sizeof line,"M %llu %u %u %016llx %u\n",(unsigned long long)cycle,slice,we?1U:0U,(unsigned long long)addr,tag);out<<line;}
    void X(u64 cycle,bool we,u64 addr,unsigned src,unsigned tag){char line[96];
        std::snprintf(line,sizeof line,"X %llu %u %016llx %u %u\n",(unsigned long long)cycle,we?1U:0U,(unsigned long long)addr,src,tag);out<<line;}
    void Y(u64 cycle,unsigned src,unsigned tag){char line[64];
        std::snprintf(line,sizeof line,"Y %llu %u %u\n",(unsigned long long)cycle,src,tag);out<<line;}
    void D(u64 cycle,unsigned src,unsigned tag){char line[64];
        std::snprintf(line,sizeof line,"D %llu %u %u\n",(unsigned long long)cycle,src,tag);out<<line;}
    void close(u64 cycle){if(out.is_open()){out<<"END "<<cycle<<"\n";out.close();}}
};

// A replay input's header: `TNCA_REPLAY 1 level=<level>`. Anything else (a
// trace, for one) is refused: the model takes a boundary, not a recording.
static void replay_header(std::istream& f,const char* level){
    std::string line;std::getline(f,line);
    require(line.rfind("TNCA_REPLAY 1 ",0)==0 && line.find(std::string("level=")+level)!=std::string::npos,
        std::string("the replay input is not a TNCA_REPLAY 1 level=")+level+" file (align/replay_extract.py makes one from a recording)");
}

// The recorded beats of one core's memory port (first slice): per launch, in
// port order, each with the latency the RTL's memory answered after. Read from
// TNCA_MODEL_CORE_REPLAY (core c>0: <file>.core<c>), for the cores
// TNCA_MODEL_CORE_TRACE_CORE selects.
struct CoreReplay {
    struct Beat {u64 cycle=0;bool we=false;u64 addr=0;unsigned be=0;u64 latency=0;bool answered=false;};
    std::vector<std::vector<Beat>> launches;
    unsigned launch=0,next=0;bool open=false;
    u64 consumed=0,mismatches=0,unrecorded=0;
    static std::unique_ptr<CoreReplay> open_for(unsigned id){
        const char* p=std::getenv("TNCA_MODEL_CORE_REPLAY");if(!p || !*p || !CoreTrace::selected(id))return nullptr;
        std::string f=p;if(id)f+=".core"+std::to_string(id);
        std::unique_ptr<CoreReplay> r(new CoreReplay);r->load(f);return r;
    }
    void load(const std::string& path){
        std::ifstream f(path);require(bool(f),"cannot open the core replay input");replay_header(f,"core");
        std::string line;
        while(std::getline(f,line)){
            std::istringstream s(line);std::string k;s>>k;
            if(k=="L"){launches.emplace_back();}
            else if(k=="B" && !launches.empty()){
                u64 c,we,addr,be;std::string lat;s>>std::dec>>c>>we>>std::hex>>addr>>be>>std::dec>>lat;
                Beat b;b.cycle=c;b.we=we!=0;b.addr=addr;b.be=unsigned(be);
                if(lat!="-"){b.latency=std::stoull(lat);b.answered=true;}
                launches.back().push_back(b);
            }
        }
        // an empty input (a core the RTL never launched) replays nothing: every beat counts as unrecorded
    }
    void start_launch(){if(open){++launch;}open=true;next=0;}
    // The next recorded beat: earliest issue cycle (relative to the launch) and
    // the latency; false when the record is exhausted (the caller uses its own
    // memory then).
    bool peek(u64& earliest,u64& latency)const{
        if(launch>=launches.size() || next>=launches[launch].size())return false;
        const Beat& b=launches[launch][next];earliest=b.cycle;latency=b.answered?b.latency:0;return b.answered;
    }
    void advance(bool we,u64 addr){
        if(launch>=launches.size() || next>=launches[launch].size()){++unrecorded;return;}
        const Beat& b=launches[launch][next++];++consumed;if(b.we!=we || b.addr!=addr)++mismatches;
    }
};

// The recorded device memory port (second slice): per launch, the accepted
// requests `X` and the responses `Y` with their cycles. Read from
// TNCA_MODEL_UNCORE_REPLAY. Cursors: one pair over the whole launch (the
// simulation bench's single port) and one pair per board port.
struct UncoreReplay {
    struct X {u64 cycle;bool we;u64 addr;unsigned src,tag;};
    struct Y {u64 cycle;unsigned src,tag;};
    std::vector<std::vector<X>> rx;std::vector<std::vector<Y>> ry;
    unsigned launch=0,xi=0,yi=0;bool open_=false;u64 mismatches=0;
    std::vector<unsigned> pxi,pyi;
    static std::unique_ptr<UncoreReplay> open(){
        const char* p=std::getenv("TNCA_MODEL_UNCORE_REPLAY");if(!p || !*p)return nullptr;
        std::unique_ptr<UncoreReplay> r(new UncoreReplay);r->load(p);return r;
    }
    void load(const std::string& path){
        std::ifstream f(path);require(bool(f),"cannot open the uncore replay input");replay_header(f,"uncore");
        std::string line;
        while(std::getline(f,line)){std::istringstream s(line);std::string k;s>>k;
            if(k=="L"){rx.emplace_back();ry.emplace_back();}
            else if(k=="X" && !rx.empty()){u64 c,we,addr,src,tag;s>>std::dec>>c>>we>>std::hex>>addr>>std::dec>>src>>tag;rx.back().push_back({c,we!=0,addr,unsigned(src),unsigned(tag)});}
            else if(k=="Y" && !ry.empty()){u64 c,src,tag;s>>std::dec>>c>>src>>tag;ry.back().push_back({c,unsigned(src),unsigned(tag)});}}
        require(!rx.empty(),"the uncore replay input has no launches");
    }
    void ports(unsigned n){pxi.assign(n,0);pyi.assign(n,0);}
    void start_launch(){if(open_){++launch;}open_=true;xi=yi=0;for(auto& c:pxi)c=0;for(auto& c:pyi)c=0;}
    // the single port's next recorded X / Y; none before the first launch (the records are launch-relative,
    // and the cycles before it would otherwise be compared with the first launch's records)
    const X* next_x()const{if(!open_ || launch>=rx.size() || xi>=rx[launch].size())return nullptr;return &rx[launch][xi];}
    const Y* next_y()const{if(!open_ || launch>=ry.size() || yi>=ry[launch].size())return nullptr;return &ry[launch][yi];}
    void take_x(){++xi;}void take_y(){++yi;}
    // board port p's next recorded X / Y: the records whose home maps to p (port_of)
    template<class F> const X* next_x(unsigned p,F port_of){
        if(!open_ || launch>=rx.size()){return nullptr;}
        auto& v=rx[launch];unsigned& c=pxi[p];
        while(c<v.size() && port_of(v[c].src)!=p){++c;}
        return c<v.size()?&v[c]:nullptr;}
    template<class F> const Y* next_y(unsigned p,F port_of){
        if(!open_ || launch>=ry.size()){return nullptr;}
        auto& v=ry[launch];unsigned& c=pyi[p];
        while(c<v.size() && port_of(v[c].src)!=p){++c;}
        return c<v.size()?&v[c]:nullptr;}
    void take_x(unsigned p){++pxi[p];}void take_y(unsigned p){++pyi[p];}
};
