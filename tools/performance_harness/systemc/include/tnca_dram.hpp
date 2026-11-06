#pragma once
// The DRAM behind an outlet, as one interface with two paths (C++17).
//
//   fast (dram_path=0): the M2 latency model -- a read completes
//        hbm_latency_cycles + tl_outlet_beat_cycles after it reaches the HBM
//        side, a write tl_hbm_write_latency_cycles after.
//   slow (dram_path=1): Ramulator2 HBM2 per pseudo channel, by a per-command
//        fixed point (Model::execute, src/chip.cpp): a forked child runs the
//        command reading its latencies from a table and logs every DRAM
//        request; the parent replays the log through Ramulator2
//        (ramulator/tnca_dram_bridge.hpp) and rebuilds the table, until the
//        command's makespan settles; then the command runs for real.
//
// The environment variable TNCA_DRAM_PATH=fast|slow overrides dram_path.
#include "tnca_model.hpp"
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <vector>

struct DramRequest {unsigned port; u64 address; bool write; u64 arrival;}; // arrival: core cycle at the HBM side

struct DramPath {
    virtual ~DramPath()=default;
    virtual u64 complete(const DramRequest&)=0;
};

struct FastDram : DramPath {
    u64 latency,write_latency,beats_extra;
    explicit FastDram(const Config& c):latency(c.get("hbm_latency_cycles")),write_latency(c.get("tl_hbm_write_latency_cycles")),
        beats_extra(c.get("tl_outlet_beat_cycles")){}
    u64 complete(const DramRequest& r) override{return r.write?r.arrival+write_latency:r.arrival+latency+beats_extra;}
};

// Latency table of the slow path. A request is keyed by (line, port, rw)
// and its occurrence index within the command, so a request matches the
// same request of the previous pass even when timing reorders the stream.
struct SlowDram : DramPath {
    static constexpr unsigned OCC_BITS=24;
    static std::uint16_t latency_value(u64 cycles){
        require(cycles<=UINT16_MAX,"slow DRAM path: replay latency outside the supported 16-bit core-cycle domain");
        return std::uint16_t(cycles);
    }
    static u64 base_key(unsigned port,u64 address,bool write){return ((address>>6)<<3)|(u64(port&3)<<1)|(write?1:0);}
    static u64 address_of(u64 key){return ((key>>OCC_BITS)>>3)<<6;}
    static unsigned port_of(u64 key){return unsigned((key>>OCC_BITS)>>1)&3;}
    static bool write_of(u64 key){return ((key>>OCC_BITS)&1)!=0;}
    struct Record {u64 key,arrival;};
    // The table, in the previous pass's request order (a pass repeats that
    // order almost exactly, so a cursor finds most keys in O(1)), with an
    // open-addressing index for the rest: 18 bytes per request.
    std::vector<u64> keys;std::vector<std::uint16_t> lat;  // latency in core cycles (complete - arrival)
    std::vector<std::uint32_t> slots;u64 mask=0;size_t cursor=0;
    std::uint32_t fallback[2]={0,0};                       // read, write: the last replay's mean
    std::unordered_map<u64,std::uint32_t> occurrence;
    FILE* log=nullptr;u64 beats_extra=0;
    u64 requests=0,matched=0;
    static u64 mix(u64 k){k^=k>>33;k*=0xff51afd7ed558ccdULL;k^=k>>33;k*=0xc4ceb9fe1a85ec53ULL;k^=k>>33;return k;}
    void begin_command(){occurrence.clear();requests=matched=0;cursor=0;}
    void clear(){std::vector<u64>().swap(keys);std::vector<std::uint16_t>().swap(lat);std::vector<std::uint32_t>().swap(slots);mask=0;}
    void install(std::vector<u64>&& k,std::vector<std::uint16_t>&& l){
        require(k.size()==l.size(),"slow DRAM path: inconsistent latency table");
        clear();keys=std::move(k);lat=std::move(l);
        require(keys.size()<0xffffffffULL,"slow DRAM path: more than 2^32 requests in one command");
        u64 cap=16;while(cap<2*keys.size())cap<<=1;
        slots.assign(cap,0);mask=cap-1;
        for(size_t i=0;i<keys.size();++i){
            u64 h=mix(keys[i])&mask;
            while(slots[h]){
                require(keys[slots[h]-1]!=keys[i],"slow DRAM path: duplicate request key");
                h=(h+1)&mask;
            }
            slots[h]=std::uint32_t(i+1);
        }
    }
    bool lookup(u64 key,size_t& pos)const{
        if(slots.empty())return false;
        for(u64 h=mix(key)&mask;slots[h];h=(h+1)&mask)if(keys[slots[h]-1]==key){pos=slots[h]-1;return true;}
        return false;
    }
    bool find(u64 key,std::uint32_t& l){
        for(size_t i=cursor,e=std::min(keys.size(),cursor+8);i<e;++i)if(keys[i]==key){l=lat[i];cursor=i+1;return true;}
        size_t pos;if(!lookup(key,pos))return false;
        l=lat[pos];cursor=pos+1;return true;
    }
    struct Stability {bool same_requests=false,within_tolerance=false;u64 missing=0,max_delta=0;};
    Stability stability(const std::vector<u64>& k,const std::vector<std::uint16_t>& l,u64 ppm)const{
        require(k.size()==l.size(),"slow DRAM path: inconsistent replay comparison");
        require(ppm<=1000000,"slow DRAM path: invalid fixed-point tolerance");
        Stability result;bool close=true;
        for(size_t i=0;i<k.size();++i){
            size_t pos;
            if(!lookup(k[i],pos)){++result.missing;continue;}
            u64 a=l[i],b=lat[pos],delta=a>b?a-b:b-a;
            result.max_delta=std::max(result.max_delta,delta);
            if(delta*1000000>ppm*std::max<u64>({a,b,1}))close=false;
        }
        result.same_requests=k.size()==keys.size() && result.missing==0;
        result.within_tolerance=result.same_requests && close;
        return result;
    }
    u64 complete(const DramRequest& r) override{
        require(r.address<(u64(1)<<36),"slow DRAM path: address beyond the key range");
        require(r.port<4,"slow DRAM path: outlet beyond the key range");
        u64 b=base_key(r.port,r.address,r.write);
        std::uint32_t& n=occurrence[b];
        require(n<(u64(1)<<OCC_BITS),"slow DRAM path: request occurrence beyond the key range");
        u64 occ=n++;
        u64 key=(b<<OCC_BITS)|occ;
        std::uint32_t l=fallback[r.write?1:0];
        if(find(key,l))++matched;
        ++requests;
        if(log){Record rec{key,r.arrival};require(std::fwrite(&rec,sizeof rec,1,log)==1,"slow DRAM path: request log write failed");}
        return r.arrival+l+(r.write?0:beats_extra);
    }
};
