// Independent boundary tests for fast timing and slow request identities.
#include "../include/tnca_dram.hpp"
#include <limits>

template<class F> static void refuses(F f,const char* fragment){
    try{f();}catch(const std::runtime_error& error){
        require(std::string(error.what()).find(fragment)!=std::string::npos,"unexpected refusal");
        return;
    }
    throw std::runtime_error("expected explicit domain refusal");
}

int sc_main(int argc,char** argv){
    try{
        require(argc==2,"usage: tnca-dram-path-test MODEL.cfg");
        Config config(argv[1]);FastDram fast(config);
        for(u64 arrival:{0ULL,1234ULL,1000000ULL}){
            require(fast.complete({0,0x400000,false,arrival})==arrival+config.get("hbm_latency_cycles")+config.get("tl_outlet_beat_cycles"),"fast read boundary");
            require(fast.complete({2,0x400000,true,arrival})==arrival+config.get("tl_hbm_write_latency_cycles"),"fast write boundary");
        }
        SlowDram slow;slow.fallback[0]=7;slow.fallback[1]=11;slow.beats_extra=3;
        require(slow.complete({0,0x400000,false,100})==110,"read fallback boundary");
        require(slow.complete({0,0x400000,true,100})==111,"write response boundary");
        std::vector<u64> keys;
        for(unsigned port=0;port<4;++port)for(bool write:{false,true})for(unsigned occ=0;occ<2;++occ){
            u64 key=(SlowDram::base_key(port,0xfffffffc0ULL,write)<<SlowDram::OCC_BITS)|occ;
            require(SlowDram::address_of(key)==0xfffffffc0ULL && SlowDram::port_of(key)==port && SlowDram::write_of(key)==write,"request key round trip");
            keys.push_back(key);
        }
        auto installed=keys;
        std::vector<std::uint16_t> lat(keys.size());
        for(size_t i=0;i<lat.size();++i)lat[i]=std::uint16_t(100+i);
        slow.install(std::move(keys),std::move(lat));slow.begin_command();
        for(size_t i=installed.size();i-->0;){
            std::uint32_t value=0;
            require(slow.find(installed[i],value) && value==100+i,"reordered request lookup");
        }
        require(slow.complete({3,0xfffffffc0ULL,true,1000})==1114,"matched first occurrence");
        require(slow.complete({3,0xfffffffc0ULL,true,1000})==1115,"matched repeated occurrence");
        require(slow.requests==2 && slow.matched==2,"request accounting");
        std::vector<std::uint16_t> replay_lat(installed.size());
        for(size_t i=0;i<replay_lat.size();++i)replay_lat[i]=std::uint16_t(100+i);
        require(slow.stability(installed,replay_lat,0).within_tolerance,"exact fixed point");
        std::reverse(installed.begin(),installed.end());std::reverse(replay_lat.begin(),replay_lat.end());
        require(slow.stability(installed,replay_lat,0).within_tolerance,"reordered fixed point");
        ++replay_lat[0];
        require(!slow.stability(installed,replay_lat,5000).within_tolerance,"unstable latency at equal request count");
        require(slow.stability(installed,replay_lat,10000).within_tolerance,"explicit latency tolerance");
        installed[0]^=(1ULL<<(SlowDram::OCC_BITS+20));
        require(!slow.stability(installed,replay_lat,1000000).same_requests,"changed request at equal count");
        installed.pop_back();replay_lat.pop_back();
        require(!slow.stability(installed,replay_lat,1000000).same_requests,"missing request");
        refuses([&]{slow.complete({4,0,false,0});},"outlet");
        refuses([&]{slow.complete({0,1ULL<<36,false,0});},"address");
        auto base=SlowDram::base_key(0,0x400000,false);
        slow.occurrence[base]=(1U<<SlowDram::OCC_BITS)-1;
        slow.complete({0,0x400000,false,0}); // final unique representable occurrence
        refuses([&]{slow.complete({0,0x400000,false,0});},"occurrence");
        refuses([&]{slow.install({1},{2,3});},"inconsistent");
        refuses([&]{slow.install({1,1},{2,3});},"duplicate");
        require(SlowDram::latency_value(65535)==65535,"maximum supported latency");
        refuses([]{SlowDram::latency_value(65536);},"latency outside");
        std::cout<<"{\"status\":\"pass\",\"checks\":[\"fast-boundaries\",\"slow-fallback\",\"key-roundtrip\",\"reordered-repeats\",\"domain-refusals\"]}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
