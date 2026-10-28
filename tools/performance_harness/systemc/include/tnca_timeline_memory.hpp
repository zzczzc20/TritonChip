#pragma once
// Analytic ("timeline") timing of the memory hierarchy, structured after the
// RTL of the 3-core U280 image (rtl/noc, rtl/nuca, rtl/fpga; see the M2 fit
// report for line references): one-flit requests and responses on a request
// and a response virtual network with a fixed latency per hop; a home slice
// whose single engine is occupied SEL/LOOK/HIT cycles per hit and
// SEL/LOOK/MISS/ALLOC, MREQ, FILL and per-target WAKE cycles per miss; finite
// MSHRs; write-back/write-allocate lines with a dirty-victim write before the
// fill; per-outlet admission (outstanding lines held until the last read beat
// or the write response), a line issue stage of 2 (read) or 3 (write) cycles
// and the HBM/interconnect latency. Each word request computes its completion
// through busy-interval timelines when it is issued; the requester waits
// once. Data are functional in the backing store at issue time, which race-
// free programs cannot distinguish. Every constant is a configuration key.
#include "tnca_model.hpp"
#include "tnca_dram.hpp"
#include "tnca_dram_map.hpp"
#include <memory>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Timeline {
    std::map<u64,u64> busy; // start -> end (cycles), disjoint, sorted
    u64 *service=nullptr,*queue=nullptr;
    // Earliest start >= t with [start, start+d) free; reserve it.
    u64 reserve(u64 t,u64 d,u64 now){
        while(!busy.empty() && busy.begin()->second<=now)busy.erase(busy.begin());
        u64 start=t;
        auto it=busy.upper_bound(start);
        if(it!=busy.begin()){auto p=std::prev(it);if(p->second>start)start=p->second;}
        while(it!=busy.end() && it->first<start+d){start=std::max(start,it->second);++it;}
        if(d){
            u64 s=start,e=start+d;
            auto n=busy.find(e);if(n!=busy.end()){e=n->second;busy.erase(n);}
            auto p=busy.lower_bound(s);
            if(p!=busy.begin()){auto q=std::prev(p);if(q->second==s){s=q->first;busy.erase(q);}}
            busy[s]=e;
        }
        if(service){*service+=d;*queue+=start-t;}
        return start;
    }
};

struct ParallelTimeline {
    std::vector<Timeline> lanes;
    explicit ParallelTimeline(unsigned n=1):lanes(n){}
    void counters(u64 *service,u64 *queue){for(auto& lane:lanes){lane.service=service;lane.queue=queue;}}
    u64 reserve(u64 t,u64 d,u64 now){
        unsigned best=0;u64 end=UINT64_MAX;
        for(unsigned i=0;i<lanes.size();++i){u64 candidate=lanes[i].busy.empty()?0:lanes[i].busy.rbegin()->second;
            if(candidate<end){end=candidate;best=i;}}
        return lanes[best].reserve(t,d,now);
    }
};

struct TimelineMemory {
    const Config& c;Context& x;Store& store;
    struct Tag {u64 address=0,ready=0;bool valid=false,dirty=false;};
    struct Home {
        std::vector<Tag> tags;std::vector<unsigned> rr;std::vector<u64> fills;ParallelTimeline engine;
        std::array<u64*,11> slots{};
    };
    std::vector<Home> homes;
    struct Outlet {Timeline issue;std::vector<u64> inflight;u64 *reads=nullptr,*writes=nullptr,*peak=nullptr,*waited=nullptr;unsigned peak_n=0;};
    std::vector<Outlet> outlets;std::vector<u64*> pc_bytes;
    std::map<std::tuple<unsigned,unsigned,unsigned>,ParallelTimeline> links;std::map<std::tuple<unsigned,unsigned,unsigned>,u64*> link_bytes;
    unsigned nx,ways;u64 hop,inject,eject,hit_occ,hit_lat,miss_occ,mreq_occ,fill_occ,wake_occ,wb_occ,to_outlet,from_outlet,
        read_issue,write_issue,beats_extra,latency,write_latency,mshrs,active_outlets,noutlets,outstanding,base,pc_first,pc_size;
    FastDram fast_dram{c};DramPath* dram=&fast_dram; // Model points it at the slow path when selected
    const DramAddressMap* physical_map=nullptr;
    TimelineMemory(const Config& cfg,Context& ctx,Store& backing):c(cfg),x(ctx),store(backing){
        auto g=[&](const char* k){return c.get(k);};
        nx=unsigned(g("nx"));ways=unsigned(g("ways"));
        hop=g("noc_stages");inject=g("tl_noc_inject_cycles");eject=g("tl_noc_eject_cycles");
        hit_occ=g("tl_slice_hit_occupancy");hit_lat=g("tl_slice_hit_latency");miss_occ=g("tl_slice_miss_occupancy");
        mreq_occ=g("tl_slice_mreq_occupancy");fill_occ=g("tl_slice_fill_occupancy");wake_occ=g("tl_slice_wake_occupancy");
        wb_occ=g("tl_slice_writeback_occupancy");to_outlet=g("tl_slice_to_outlet_cycles");from_outlet=g("tl_outlet_to_slice_cycles");
        read_issue=g("tl_outlet_read_issue_cycles");write_issue=g("tl_outlet_write_issue_cycles");beats_extra=g("tl_outlet_beat_cycles");
        latency=g("hbm_latency_cycles");write_latency=g("tl_hbm_write_latency_cycles");
        mshrs=g("mshrs");active_outlets=g("active_outlets");noutlets=g("outlets");outstanding=g("outstanding");
        base=g("memory_base");pc_first=g("pc_first");pc_size=g("pc_bytes");
        homes.resize(c.cores());
        for(unsigned h=0;h<homes.size();++h){
            homes[h].tags.resize(size_t(g("sets"))*ways);homes[h].rr.assign(g("sets"),0);
            homes[h].engine=ParallelTimeline(c.get_or("tl_memory_lanes",1));
            homes[h].engine.counters(&x.counts["home_"+std::to_string(h)+"_engine_service_cycles"],
                                     &x.counts["home_"+std::to_string(h)+"_engine_queue_cycles"]);
        }
        outlets.resize(noutlets);
        for(unsigned p=0;p<noutlets;++p){auto k="outlet_"+std::to_string(p);
            outlets[p].issue.service=&x.counts[k+"_service_cycles"];outlets[p].issue.queue=&x.counts[k+"_queue_cycles"];
            outlets[p].reads=&x.counts[k+"_reads"];outlets[p].writes=&x.counts[k+"_writes"];
            outlets[p].peak=&x.counts[k+"_peak_outstanding"];outlets[p].waited=&x.counts[k+"_wait_cycles"];}
        pc_bytes.assign(g("pc_count"),nullptr);
    }
    u64& hc(unsigned h,unsigned i){
        static const char* names[11]={"writebacks","target_full_events","merges","hits","mshr_full_events","set_busy_events",
            "misses","peak_mshrs","evictions","resolution_cycles","maintenance_completed"};
        auto& s=homes[h].slots[i];if(!s)s=&x.counts["home_"+std::to_string(h)+"_"+names[i]];return *s;
    }
    // One flit over XY hops on virtual network `vn` (0 request, 1 response):
    // one cycle of link occupancy and `hop` cycles of latency per hop.
    u64 route(unsigned from,unsigned to,unsigned vn,u64 bytes,u64 t,u64 now){
        while(from!=to){unsigned next=from;
            if(from%nx!=to%nx)next=from%nx<to%nx?from+1:from-1;
            else next=from/nx<to/nx?from+nx:from-nx;
            auto key=std::make_tuple(from,next,vn);
            auto it=links.find(key);
            if(it==links.end()){
                auto k="link_"+std::to_string(from)+"_"+std::to_string(next);
                it=links.emplace(key,ParallelTimeline(c.get_or("tl_memory_lanes",1))).first;
                it->second.counters(&x.counts[k+"_service_cycles"],&x.counts[k+"_queue_cycles"]);
                link_bytes[key]=&x.counts[k+"_bytes"];
            }
            t=it->second.reserve(t,1,now)+hop;*link_bytes[key]+=bytes;from=next;
        }
        return t;
    }
    // A line at an outlet (tnca_line_axi.v): admission while fewer than
    // `outstanding` lines are active, a 2/3-cycle issue stage, then the
    // read data (latency + second beat) or the write response.
    u64 line(unsigned home,u64 address,bool write,u64 t,u64 now){
        u64 stripe=c.get_or("tl_pc_stripe_bytes",0);
        unsigned pc=physical_map?physical_map->translate(address).pc:(stripe?pc_first+unsigned(((address-base)/stripe)%c.get("pc_count")):unsigned(pc_first+(address-base)/pc_size));
        unsigned port=active_outlets==1?0:(noutlets>c.cores()?pc%noutlets:unsigned(home%noutlets));auto& o=outlets[port];
        auto& f=o.inflight;f.erase(std::remove_if(f.begin(),f.end(),[t](u64 d){return d<=t;}),f.end());
        u64 arrival=t;
        while(f.size()>=outstanding){auto m=std::min_element(f.begin(),f.end());t=std::max(t,*m);f.erase(m);}
        *o.waited+=t-arrival;
        u64 issue=write?write_issue:read_issue;
        u64 b=o.issue.reserve(t,issue,now);
        // The HBM side receives the line after the issue stage (dram_path).
        u64 done=dram->complete(DramRequest{port,address,write,b+issue});
        if(pc<pc_bytes.size()){if(!pc_bytes[pc])pc_bytes[pc]=&x.counts["pc_"+std::to_string(pc)+"_bytes"];*pc_bytes[pc]+=64;}
        f.push_back(done);o.peak_n=std::max<unsigned>(o.peak_n,unsigned(f.size()));*o.peak=o.peak_n;
        ++*(write?o.writes:o.reads);
        return done;
    }
    // Home slice (tnca_nuca_slice.v); returns the cycle its response leaves.
    u64 slice(unsigned h,u64 a,bool write,u64 t,u64 now){
        auto& H=homes[h];u64 lineaddr=a&~63ULL;
        unsigned s=c.set(a),b=s*ways;Tag* hit=nullptr;
        for(unsigned w=0;w<ways;++w){auto& g=H.tags[b+w];if(g.valid && g.address==lineaddr){hit=&g;break;}}
        u64 arrival=t;
        if(hit && hit->ready<=t){ // SEL, LOOK, HIT
            ++hc(h,3);u64 st=H.engine.reserve(t,hit_occ,now);
            if(write)hit->dirty=true;
            return st+hit_lat;
        }
        if(hit){ // merge into the fill in flight: SEL/LOOK/MISS(target), then WAKE after the fill
            ++hc(h,2);u64 st=H.engine.reserve(t,miss_occ,now);(void)st;
            u64 w=H.engine.reserve(hit->ready,wake_occ,now)+wake_occ;
            if(write)hit->dirty=true;
            hc(h,9)+=w-arrival;return w;
        }
        // miss: an MSHR (retry when one retires), then SEL/LOOK/MISS/ALLOC
        auto& f=H.fills;f.erase(std::remove_if(f.begin(),f.end(),[t](u64 d){return d<=t;}),f.end());
        while(f.size()>=mshrs){++hc(h,4);auto m=std::min_element(f.begin(),f.end());t=std::max(t,*m);f.erase(m);}
        Tag* victim=nullptr;
        for(;;){
            for(unsigned w=0;w<ways && !victim;++w){auto& g=H.tags[b+w];if(!g.valid){victim=&g;H.rr[s]=(w+1)%ways;}}
            for(unsigned i=0;i<ways && !victim;++i){unsigned w=(H.rr[s]+i)%ways;auto& g=H.tags[b+w];if(g.ready<=t){victim=&g;H.rr[s]=(w+1)%ways;}}
            if(victim)break;
            ++hc(h,5);u64 m=UINT64_MAX;for(unsigned w=0;w<ways;++w)m=std::min(m,H.tags[b+w].ready);t=std::max(t,m);
        }
        ++hc(h,6);
        u64 st=H.engine.reserve(t,miss_occ,now)+miss_occ;
        if(victim->valid){
            ++hc(h,8);
            if(victim->dirty){ // E_WBRD/E_WBSND, the dirty line leaves for the outlet
                ++hc(h,0);st=H.engine.reserve(st,wb_occ,now)+wb_occ;
                line(h,victim->address,true,st+to_outlet,now);
            }
        }
        u64 mreq=H.engine.reserve(st,mreq_occ,now)+mreq_occ;
        u64 data=line(h,lineaddr,false,mreq+to_outlet,now)+from_outlet;
        u64 fill=H.engine.reserve(data,fill_occ,now)+fill_occ;
        u64 wake=H.engine.reserve(fill,wake_occ,now)+wake_occ;
        victim->address=lineaddr;victim->valid=true;victim->dirty=write;victim->ready=wake;
        f.push_back(wake);hc(h,7)=std::max<u64>(hc(h,7),f.size());
        hc(h,9)+=wake-arrival;
        return wake;
    }
    // A word request from a core issued at cycle t: functional effect now,
    // returns the cycle its response is back at the core.
    u64 word(unsigned core,u64 a,bool write,u64 data,unsigned be,u64 t,u64& result){
        x.c.address(a,8);u64 now=x.now();
        unsigned h=x.c.home(a);
        u64 r=route(core,h,0,32,t+inject,now);
        r=slice(h,a,write,r,now);
        r=route(h,core,1,16,r+eject,now);
        auto& l=store[a&~63ULL];unsigned off=unsigned(a&63);u64 old=0;
        for(unsigned bb=0;bb<8;++bb)old|=u64(l[off+bb])<<(8*bb);
        if(write)for(unsigned bb=0;bb<8;++bb)if(be&(1U<<bb))l[off+bb]=static_cast<unsigned char>(data>>(8*bb));
        {static u64 lo=0,hi=0;static bool init=false;if(!init){init=true;if(const char* w=std::getenv("TLM_DEBUG_WATCH")){lo=std::strtoull(w,nullptr,0);const char* c=std::strchr(w,',');hi=lo+(c?std::strtoull(c+1,nullptr,0):0);}}
         if(a+8>lo && a<hi)std::fprintf(stderr,"WATCH core %u t %llu now %llu %s a %llx data %016llx be %x old %016llx\n",core,(unsigned long long)t,(unsigned long long)now,write?"W":"R",(unsigned long long)a,(unsigned long long)data,be,(unsigned long long)old);}
        result=old;
        return r;
    }
    // Exploratory wide coalesced request: one aligned region, one cache service,
    // and explicit serialization when the receiving NoC is narrower. Functional
    // words remain exact 64-bit values; this does not widen arithmetic or atomics.
    struct WideWord {u64 address,data;unsigned be;u64 result=0;};
    u64 wide(unsigned core,std::vector<WideWord>& words,bool write,u64 t){
        require(!words.empty(),"empty wide request");
        u64 width=c.get_or("tl_core_beat_bytes",8),payload=c.get_or("tl_noc_payload_bytes",8);
        u64 region=words.front().address&~(width-1),now=x.now();
        for(auto& w:words){require((w.address&7)==0 && (w.address&~(width-1))==region,"wide request crosses region");x.c.address(w.address,8);}
        unsigned h=x.c.home(region);
        // The historical model treats fixed headers as one flit. Extra useful
        // data occupies additional link cycles; retain that exact baseline.
        u64 req_data=write?width:8,rsp_data=write?8:width;
        auto transfer=[&](unsigned from,unsigned to,unsigned vn,u64 data,u64 at){
            u64 n=(data+payload-1)/payload;
            u64 last=at;
            for(u64 i=0;i<n;++i)last=std::max(last,route(from,to,vn,std::min(payload,data-i*payload),at+i,now));
            return last;
        };
        u64 r=transfer(core,h,0,req_data,t+inject);
        r=slice(h,region,write,r,now);
        r=transfer(h,core,1,rsp_data,r+eject);
        for(auto& w:words){auto& l=store[w.address&~63ULL];unsigned off=unsigned(w.address&63);w.result=0;
            for(unsigned b=0;b<8;++b)w.result|=u64(l[off+b])<<(8*b);
            if(write)for(unsigned b=0;b<8;++b)if(w.be&(1U<<b))l[off+b]=static_cast<unsigned char>(w.data>>(8*b));}
        x.counts["wide_requests"]++;x.counts["wide_word_components"]+=words.size();
        return r;
    }
    // The functional effect alone (untimed mode): no timing state changes,
    // except that the line is placed in its home slice as a timed access
    // would leave it, so a following timed call starts with that content.
    u64 access(u64 a,bool write,u64 data,unsigned be){
        x.c.address(a,8);
        {auto& H=homes[x.c.home(a)];u64 lineaddr=a&~63ULL;unsigned st=c.set(a),b0=st*ways;Tag* hit=nullptr;
         for(unsigned w=0;w<ways;++w){auto& g=H.tags[b0+w];if(g.valid && g.address==lineaddr){hit=&g;break;}}
         if(!hit){
             for(unsigned w=0;w<ways && !hit;++w){auto& g=H.tags[b0+w];if(!g.valid){hit=&g;H.rr[st]=(w+1)%ways;}}
             if(!hit){unsigned w=H.rr[st];hit=&H.tags[b0+w];H.rr[st]=(w+1)%ways;}
             hit->address=lineaddr;hit->valid=true;hit->dirty=false;hit->ready=0;
         }
         if(write)hit->dirty=true;}
        auto& l=store[a&~63ULL];unsigned off=unsigned(a&63);u64 old=0;
        for(unsigned bb=0;bb<8;++bb)old|=u64(l[off+bb])<<(8*bb);
        if(write)for(unsigned bb=0;bb<8;++bb)if(be&(1U<<bb))l[off+bb]=static_cast<unsigned char>(data>>(8*bb));
        return old;
    }
    // Write back every dirty line and invalidate (maintenance); returns completion.
    u64 flush(u64 t){
        u64 now=x.now(),done=t;
        for(unsigned h=0;h<homes.size();++h){
            for(auto& g:homes[h].tags){
                if(g.valid && g.dirty){u64 st=homes[h].engine.reserve(std::max(t,g.ready),wb_occ,now)+wb_occ;
                    done=std::max(done,line(h,g.address,true,st+to_outlet,now));++hc(h,0);}
                g=Tag{};
            }
            ++hc(h,10);homes[h].fills.clear();
        }
        // An evicted dirty line can already have left the cache while its
        // accepted outlet write still awaits completion. Maintenance must
        // drain those older transactions as well as the writes above.
        for(const auto& outlet:outlets)for(u64 completion:outlet.inflight)done=std::max(done,completion);
        return done;
    }
};
