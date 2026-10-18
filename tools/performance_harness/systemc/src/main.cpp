#include "../include/tnca_model.hpp"
struct Row {u64 id,cycle,address,value,compare,expected;unsigned core,op,be;bool check;char kind;};
static bool readrow(std::istream& in,Row& r) {
    std::string s;while(std::getline(in,s)){
        if(s.empty() || s[0]=='#')continue;
        std::istringstream a(s);std::vector<std::string> fields;std::string f;while(a>>f)fields.push_back(f);
        require(fields.size()==11,"trace requires 11 fields: id cycle core op address value be compare check expected kind");
        r.id=integer(fields[0]);r.cycle=integer(fields[1]);auto core=integer(fields[2]),op=integer(fields[3]),be=integer(fields[6]);
        require(core<=255 && op<=13 && be<=255,"invalid source/op/byte mask");r.core=unsigned(core);r.op=unsigned(op);r.be=unsigned(be);
        r.address=integer(fields[4]);r.value=integer(fields[5]);r.compare=integer(fields[7]);require(integer(fields[8])<=1,"invalid check flag");r.check=integer(fields[8]);r.expected=integer(fields[9]);
        require(fields[10].size()==1 && std::string("TIFH").find(fields[10][0])!=std::string::npos,"invalid row kind");r.kind=fields[10][0];return true;
    }return false;
}
struct Driver : sc_module {
    SC_HAS_PROCESS(Driver);
    tlm_utils::simple_initiator_socket<Driver> socket;
    Context& x;HBM& hbm;std::vector<std::unique_ptr<Slice>>& caches;
    std::ifstream input;std::ofstream responses,summary;
    std::vector<unsigned> active;unsigned pending=0;u64 issued=0,checked=0,failures=0,last_id=0,last_cycle=0;
    sc_event changed;u64 graph_end=0,visible_end=0;bool terminal=false;
    Driver(sc_module_name n,Context& ctx,HBM& mem,std::vector<std::unique_ptr<Slice>>& s,const char* trace,const std::string& out):
        sc_module(n),socket("socket"),x(ctx),hbm(mem),caches(s),input(trace),responses(out+"/responses.tsv"),summary(out+"/summary.json"),active(x.c.cores(),0){
        require(bool(input) && bool(responses) && bool(summary),"cannot open model I/O");SC_THREAD(run);
    }
    void drain(){while(pending)wait(changed);}
    void flush(){
        drain();unsigned left=caches.size();sc_event completed;
        for(auto& cache:caches){auto* c=cache.get();sc_spawn([&,c](){c->flush();--left;completed.notify(SC_ZERO_TIME);});}
        while(left)wait(completed);
        for(auto& slot:hbm.slots)require(slot->used==0,"write response outstanding after maintenance");
    }
    void execute(Row r) {
        Request ext;ext.id=r.id;ext.core=r.core;ext.op=r.op;ext.be=r.op==0?255:r.be;ext.value=r.value;ext.compare=r.compare;
        std::array<unsigned char,8> data{},enables{};tlm::tlm_generic_payload tx;
        for(unsigned b=0;b<8;++b){data[b]=unsigned(r.value>>(b*8))&255;enables[b]=(ext.be&(1<<b))?255:0;}
        tx.set_extension(&ext);tx.set_command(r.op==0?tlm::TLM_READ_COMMAND:tlm::TLM_WRITE_COMMAND);
        tx.set_address(r.address);tx.set_data_ptr(data.data());tx.set_data_length(8);tx.set_streaming_width(8);
        tx.set_byte_enable_ptr(enables.data());tx.set_byte_enable_length(8);
        auto start=sc_time_stamp();sc_time delay=SC_ZERO_TIME;socket->b_transport(tx,delay);tx.clear_extension<Request>();
        require(tx.is_response_ok() && delay==SC_ZERO_TIME,"invalid word response");
        u64 result=0;for(unsigned b=0;b<8;++b)result|=u64(data[b])<<(b*8);
        if(r.check){++checked;if(result!=r.expected){++failures;std::cerr<<"MISMATCH id="<<r.id<<" got="<<std::hex<<result<<" expected="<<r.expected<<std::dec<<'\n';}}
        responses<<r.id<<'\t'<<x.now()<<'\t'<<"0x"<<std::hex<<result<<std::dec<<'\n';
        ++x.counts["core_"+std::to_string(r.core)+"_completed"];
        x.span("core transaction",r.core,start,r.id);--active[r.core];--pending;changed.notify(SC_ZERO_TIME);
    }
    void run(){
        Row r{};bool started=false;
        while(readrow(input,r)){
            require(r.id>last_id,"trace IDs must strictly increase");last_id=r.id;
            require(r.cycle>=last_cycle,"trace release times must not decrease");last_cycle=r.cycle;
            require(r.core<x.c.cores() && r.address%8==0,"invalid trace core/address");
            x.c.address(r.address,8);
            if(r.kind=='I'){require(!started && r.cycle==0,"initialization must precede requests");hbm.initialize(r.address,r.value,r.be);continue;}
            started=true;
            sc_time release=x.cycles(r.cycle);if(sc_time_stamp()<release)wait(release-sc_time_stamp());
            if(r.kind=='F'){flush();continue;}
            if(r.kind=='H'){
                drain();u64 got=Slice::load(hbm.memory[r.address&~63ULL],unsigned(r.address&63));
                require(r.check,"host visibility probe needs expectation");++checked;if(got!=r.expected)++failures;continue;
            }
            while(active[r.core]>=x.c.get("core_window"))wait(changed);
            x.counts["release_slip_cycles"]+=u64((sc_time_stamp()-release)/x.cycles(1)+.5);
            ++active[r.core];++pending;++issued;
            ++x.counts["core_"+std::to_string(r.core)+"_issued"];sc_spawn([this,r](){execute(r);});
            // Deterministic arrival order, including equal-timestamp requests.
            wait(SC_ZERO_TIME);
        }
        drain();graph_end=x.now();flush();visible_end=x.now();
        summary<<"{\"schema\":\"tnca-systemc-result/1\",\"status\":\""<<(failures?"functional-mismatch":"completed")
            <<"\",\"performance_gate\":\"not-measured\",\"calibration\":\"not-measured\",\"graph_cycles\":"<<graph_end
            <<",\"visible_cycles\":"<<visible_end<<",\"issued\":"<<issued<<",\"checked\":"<<checked<<",\"mismatches\":"<<failures<<",\"counters\":{";
        bool first=true;for(auto& kv:x.counts){if(!first)summary<<',';first=false;summary<<'"'<<kv.first<<"\":"<<kv.second;}summary<<"}}\n";
        summary.close();responses.close();terminal=true;x.close();sc_stop();
    }
};
int sc_main(int argc,char** argv){
    try {
        require(argc==4,"usage: tnca-tlm CONFIG TRACE OUTPUT_DIRECTORY (fresh directory required by runner)");
        sc_set_time_resolution(1,SC_PS);Config cfg(argv[1]);Context x(cfg,argv[3]);HBM hbm("hbm",x);Network net("noc",x);
        std::vector<std::unique_ptr<Slice>> caches;
        for(unsigned i=0;i<cfg.cores();++i){caches.emplace_back(new Slice(("home"+std::to_string(i)).c_str(),x,i));net.homes[i].bind(caches.back()->target);caches.back()->memory.bind(hbm.target);}
        Driver driver("driver",x,hbm,caches,argv[2],argv[3]);driver.socket.bind(net.target);sc_start();
        require(driver.terminal,"model deadlocked: no runnable process and no completed report");
        return driver.failures?1:0;
    }catch(const std::exception& e){std::cerr<<"MODEL_REFUSED: "<<e.what()<<'\n';return 2;}
}
