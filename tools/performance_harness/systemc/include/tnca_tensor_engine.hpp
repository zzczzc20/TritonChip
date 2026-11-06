#pragma once
#include "tnca_tensor_model.hpp"
#include "tnca_barrier_model.hpp"
#include "tnca_timeline_memory.hpp"
#include <functional>
#include <cstdlib>
#include <cstdio>

// One transfer controller, a bounded admission queue, bounded memory tags and
// a 64-bit RF port. A queued command owns immutable descriptor/coordinate/window
// snapshots. No asynchronous callback retains a role's stack frame.
struct TensorEngine : sc_module {
    tlm_utils::simple_initiator_socket<TensorEngine> memory;
    struct Descriptor {std::shared_ptr<const TensorMapModel> snapshot;unsigned references=0;u64 source_address=0;};
    struct Window {bool valid=false;unsigned base=0,length=0,references=0;};
    std::array<Descriptor,16> descriptors;
    std::array<Window,32> windows;
    Context& x;unsigned core;Limit admission,word_slots;sc_mutex controller;Resource rf_port;
    Pool unit_workers,beat_workers,command_worker;
    std::function<void(u64)> verify_token;
    std::function<void(u64,bool)> completion;
    u64 sequence=0;
    // Commands issued and not yet completed, per issuing context: what an
    // mb.arrive waits for under RPAIR's precise release (tnca_core.v arr_txf).
    std::array<unsigned,4> outstanding{};sc_event role_done;
    TensorEngine(sc_module_name name,Context& context,unsigned id,
                 std::function<void(u64)> verify,std::function<void(u64,bool)> done):
        sc_module(name),memory("memory"),x(context),core(id),
        admission(unsigned(x.c.get("dma_queue"))),word_slots(unsigned(x.c.get("dma_outstanding"))),
        unit_workers(unsigned(x.c.get("dma_unit_slots"))),beat_workers(unsigned(x.c.get("dma_unit_slots"))),command_worker(1),
        verify_token(std::move(verify)),completion(std::move(done)){}
    std::string key(const std::string& suffix)const{return "core_"+std::to_string(core)+"_dma_"+suffix;}
    std::string units_key=key("units"),tags_key=key("tags"),rf_key=key("rf"),read_key=key("read_words"),write_key=key("write_words"),peak_key=key("peak_tags");
    u64 *reads=nullptr,*writes=nullptr,*peak=nullptr;u64 rf_cycles=x.c.get("dma_rf_cycles");
    // Timeline path (memory_model=1): the core's memory port and the analytic
    // memory system; the engine's RF port and tag window are timelines too.
    TimelineMemory* tm=nullptr;Timeline* port=nullptr;Timeline rf_tl;
    // core_pipeline=1 (tnca_tensor.v): setup 2+rank, a load box unit holds
    // one of NTAG/2 slots until written to the window, a store reads each
    // window register (S_SREAD) before its words, then the engine drains.
    bool rtl=false;
    // One 8-byte unit: returns completion cycle; `result` gets the assembled bytes.
    u64 fast_unit(const TensorMapModel& d,const std::array<u64,5>& coords,unsigned bytes,unsigned flat_byte,bool store,
                  u64 data,u64& result,u64& t,std::vector<u64>& window,u64 now){
        struct Beat {u64 value=0;unsigned mask=0;};
        std::array<TensorMapModel::Byte,8> plan{};std::map<u64,Beat> beats;
        for(unsigned b=0;b<8;++b){
            plan[b]=d.at(flat_byte+b,bytes,coords);
            if(plan[b].valid){auto& beat=beats[plan[b].address&~7ULL];unsigned offset=unsigned(plan[b].address&7);
                beat.mask|=1<<offset;beat.value|=((data>>(b*8))&255)<<(offset*8);}
        }
        require(beats.size()<=2,"DMA unit expanded beyond two aligned memory words");
        u64 done=t;
        if(x.functional)for(auto& beat:beats)beat.second.value=tm->access(beat.first,store,beat.second.value,store?beat.second.mask:255);
        else for(auto& beat:beats){
            u64 cap=x.c.get("dma_outstanding");
            while(window.size()>=cap){auto m=std::min_element(window.begin(),window.end());t=std::max(t,*m);window.erase(m);}
            u64 at=port->reserve(t,1,now);t=at+1;u64 value=0;
            u64 when=tm->word(core,beat.first,store,beat.second.value,store?beat.second.mask:255,at,value);
            window.push_back(when);done=std::max(done,when);beat.second.value=value;
            if(store){if(!writes)writes=&x.counts[write_key];++*writes;}else{if(!reads)reads=&x.counts[read_key];++*reads;}
        }
        result=0;
        if(!store)for(unsigned b=0;b<8;++b){
            unsigned char value=plan[b].padding;
            if(plan[b].valid)value=static_cast<unsigned char>(beats.at(plan[b].address&~7ULL).value>>((plan[b].address&7)*8));
            result|=u64(value)<<(b*8);
        }
        return done;
    }
    static bool xlog(){static const bool on=std::getenv("TLM_DEBUG_XPOSE_LOG")!=nullptr;return on;}
    void fast_rf_write(const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,unsigned byte,u64 value,u64 owner,u64 due,unsigned size=8){
        if(x.functional)due=x.now();
        unsigned index=base+byte/64,lane=(byte%64)/4;PhysicalRegisters::Vector data{};
        data[lane]=std::uint32_t(value);if(size>4)data[lane+1]=std::uint32_t(value>>32);
        bank->write(index,(size>4?3U:1U)<<lane,x.cycles(due),data,owner);
    }
    // The transposed write's corner turn (tnca_tensor.v S_XWRITE), for 16-bit
    // and (R41 X8) 1-byte elements. A column block is 8/bytes source columns;
    // destination row `column+col` takes element `col` of every buffered
    // source unit, in source-row order, and is written as pieces of at most
    // 8 bytes (a 1-byte row of 4 elements is one 4-byte lane).
    template<class F> static void corner_turn(const std::vector<u64>& source,unsigned column,unsigned col,unsigned rows,unsigned bytes,F&& emit){
        unsigned bits=8*bytes,group=8/bytes;u64 mask=(bytes==8)?~0ULL:((1ULL<<bits)-1);
        for(unsigned row=0;row<rows;row+=group){
            unsigned n=std::min(group,rows-row);u64 value=0;
            for(unsigned i=0;i<n;++i)value|=((source[row+i]>>(col*bits))&mask)<<(i*bits);
            emit(((column+col)*rows+row)*bytes,value,n*bytes);
        }
    }
    // The whole command analytically; one wait for its completion.
    void fast_transfer(const TensorMapModel& d,const std::array<u64,5>& coords,unsigned bytes,unsigned total,bool store,
                       bool transpose,const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,u64 owner){
        u64 now=x.now(),t=now,end=now;std::vector<u64> window;
        if(!combine.empty())t=std::max(t,flush_held(t,now));
        if(x.functional){ // untimed: the data movement alone, no timing state touched
            if(transpose){
                unsigned rows=d.block[0],columns=d.block[1],per=8/bytes;
                for(unsigned column=0;column<columns;column+=per){
                    std::vector<u64> source(rows);
                    for(unsigned row=0;row<rows;++row){u64 r=0;fast_unit(d,coords,bytes,(row*columns+column)*bytes,false,0,r,t,window,now);source[row]=r;}
                    for(unsigned col=0;col<per;++col)corner_turn(source,column,col,rows,bytes,[&](unsigned byte,u64 value,unsigned size){
                        fast_rf_write(bank,base,byte,value,owner,now,size);});
                }
            }else for(unsigned byte=0;byte<total;byte+=8){
                u64 value=0,result=0;
                if(store){unsigned index=base+byte/64,lane=(byte%64)/4;auto v=bank->read(index,sc_time_stamp(),owner);value=u64(v[lane])|(u64(v[lane+1])<<32);}
                fast_unit(d,coords,bytes,byte,store,value,result,t,window,now);
                if(!store)fast_rf_write(bank,base,byte,result,owner,now);
            }
            bank->commit(sc_time_stamp());
            return;
        }
        if(transpose && rtl){
            // tnca_tensor.v S_XRUN/S_XDRAIN/S_XWRITE: one column block at a
            // time -- one 8-byte unit per source row, one request a cycle
            // while a tag is free, drain every response, then write the
            // 8/esz destination rows one a cycle -- and only then the next
            // block (no overlap between blocks).
            // A 1-byte block is eight destination rows, so eight write cycles.
            unsigned rows=d.block[0],columns=d.block[1],per=8/bytes;u64 cap=x.c.get("dma_outstanding");
            for(unsigned column=0;column<columns;column+=per){
                std::vector<u64> source(rows);u64 drained=t;window.clear();
                for(unsigned row=0;row<rows;++row){
                    while(window.size()>=cap){auto m=std::min_element(window.begin(),window.end());t=std::max(t,*m);window.erase(m);}
                    u64 r=0;drained=std::max(drained,fast_unit(d,coords,bytes,(row*columns+column)*bytes,false,0,r,t,window,now));source[row]=r;
                }
                u64 w=std::max(t,drained)+1;
                for(unsigned col=0;col<per;++col)corner_turn(source,column,col,rows,bytes,[&](unsigned byte,u64 value,unsigned size){
                    fast_rf_write(bank,base,byte,value,owner,w+col,size);end=std::max(end,w+col+1);});
                t=w+per;
            }
            end=std::max(end,t)+x.c.get("dma_drain_cycles");
        }else if(transpose){
            unsigned rows=d.block[0],columns=d.block[1],per=8/bytes;
            for(unsigned column=0;column<columns;column+=per){
                std::vector<u64> source(rows);u64 ready=t;
                for(unsigned row=0;row<rows;++row){u64 r=0;ready=std::max(ready,fast_unit(d,coords,bytes,(row*columns+column)*bytes,false,0,r,t,window,now));source[row]=r;}
                for(unsigned col=0;col<per;++col)corner_turn(source,column,col,rows,bytes,[&](unsigned byte,u64 value,unsigned size){
                    u64 w=rf_tl.reserve(ready,rf_cycles,now)+rf_cycles;
                    if(xlog())std::fprintf(stderr,"XPOSE fast core %u base %u byte %u value %llx now %llu due %llu\n",core,base,byte,(unsigned long long)value,(unsigned long long)now,(unsigned long long)w);
                    fast_rf_write(bank,base,byte,value,owner,w,size);end=std::max(end,w);});
            }
        }else if(rtl){
            u64 cap=x.c.get("dma_unit_slots"),reg_cycles=x.c.get("dma_store_register_cycles");
            std::vector<u64> slots;unsigned current=~0U;std::vector<std::uint32_t> words;
            for(unsigned byte=0;byte<total;byte+=8){
                u64 value=0,result=0;
                if(store){
                    unsigned index=base+byte/64,lane=(byte%64)/4;
                    if(index!=current){current=index;t+=reg_cycles;auto v=bank->read(index,sc_time_stamp(),owner);words.assign(v.begin(),v.end());}
                    value=u64(words[lane])|(u64(words[lane+1])<<32);
                    end=std::max(end,fast_unit(d,coords,bytes,byte,true,value,result,t,window,now));
                }else{
                    if(slots.size()>=cap){auto m=std::min_element(slots.begin(),slots.end());t=std::max(t,*m);slots.erase(m);}
                    u64 done=fast_unit(d,coords,bytes,byte,false,0,result,t,window,now);
                    u64 w=rf_tl.reserve(done,rf_cycles,now)+rf_cycles;fast_rf_write(bank,base,byte,result,owner,w);
                    slots.push_back(w);end=std::max(end,w);
                }
            }
            end=std::max(end,t)+x.c.get("dma_drain_cycles");
        }else{
            for(unsigned byte=0;byte<total;byte+=8){
                u64 value=0;
                if(store){
                    u64 r=rf_tl.reserve(t,rf_cycles,now);t=std::max(t,r+rf_cycles);
                    unsigned index=base+byte/64,lane=(byte%64)/4;auto v=bank->read(index,sc_time_stamp(),owner);
                    value=u64(v[lane])|(u64(v[lane+1])<<32);
                }
                u64 result=0,done=fast_unit(d,coords,bytes,byte,store,value,result,t,window,now);
                if(!store){u64 w=rf_tl.reserve(done,rf_cycles,now)+rf_cycles;fast_rf_write(bank,base,byte,result,owner,w);done=w;}
                end=std::max(end,done);
            }
        }
        if(!peak)peak=&x.counts[peak_key];
        *peak=std::max<u64>(*peak,x.c.get("dma_outstanding"));
        if(end>now && !x.functional)wait(x.cycles(end-now));
        bank->commit(sc_time_stamp());
    }
    // ---- R41 G4 what-if: the line-wide, multi-command engine (dma_line_engine=1) ----
    // Memory requests are whole 64-byte lines on the core's wide port, one a
    // cycle; each line in flight holds one of `dma_line_buffers` engine line
    // buffers (not a vector register) from its request to its register write
    // (load) or from its register read to its request (store). The register
    // file port moves one 64-byte register per `dma_line_rf_cycles`. Commands
    // overlap: the controller is held only for a command's issue phase, and a
    // store's read lease ends when its registers have been read into line
    // buffers. With dma_line_combine=1, a store's partial lines wait in their
    // buffers for a following queued store that completes them, so a line
    // written in pieces reaches its home as one full-line write.
    struct Held {u64 line;std::map<u64,std::pair<u64,unsigned>> words;bool flushed=false;u64 done=0;sc_event event;};
    std::vector<u64> line_release;              // release cycles of occupied line buffers
    unsigned queued=0;                           // line commands submitted and not yet started
    u64 last_issue_end=0;                        // when the previous line command finished issuing
    std::map<u64,std::shared_ptr<Held>> combine; // partial store lines held for completion
    Timeline line_rf,line_rf_read; // the window port's write side (loads) and read side (stores)
    struct LineResult {u64 issue_end=0,rf_end=0,end=0;std::vector<std::shared_ptr<Held>> held;};
    bool line_mode()const{return tm && !x.functional && x.c.get_or("dma_line_engine",0)!=0;}
    u64 take_buffer(u64 t,u64 now){
        unsigned cap=unsigned(x.c.get_or("dma_line_buffers",64));
        line_release.erase(std::remove_if(line_release.begin(),line_release.end(),[now](u64 d){return d<=now;}),line_release.end());
        while(line_release.size()>=cap){auto m=std::min_element(line_release.begin(),line_release.end());t=std::max(t,*m);line_release.erase(m);}
        auto& peak=x.counts[key("line_buffers_peak")];peak=std::max<u64>(peak,line_release.size()+1);
        return t;
    }
    u64 issue_line(std::vector<TimelineMemory::WideWord>& words,bool store,u64 t,u64 now,u64& at){
        at=port->reserve(t,1,now);
        u64 done=tm->wide(core,words,store,at);
        ++x.counts[key(store?"line_writes":"line_reads")];
        if(store){bool full=words.size()==8;for(auto& w:words)full=full && w.be==255;if(!full)++x.counts[key("partial_line_writes")];}
        return done;
    }
    u64 flush_held(u64 t,u64 now){ // issue every held partial line as it is
        u64 done=t;
        for(auto& [line,h]:combine){
            std::vector<TimelineMemory::WideWord> words;
            for(auto& [a,v]:h->words)words.push_back({a,v.first,v.second,0});
            u64 at=0;h->done=issue_line(words,true,t,now,at);t=at+1;h->flushed=true;h->event.notify(SC_ZERO_TIME);
            auto rel=std::find(line_release.begin(),line_release.end(),UINT64_MAX/2);if(rel!=line_release.end())*rel=at+1;
            done=std::max(done,h->done);
        }
        combine.clear();return done;
    }
    LineResult line_transfer(const TensorMapModel& d,const std::array<u64,5>& coords,unsigned bytes,unsigned total,bool store,
                             const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,u64 owner){
        LineResult r;u64 now=x.now(),t=now;u64 rf=x.c.get_or("dma_line_rf_cycles",1);
        bool merge=x.c.get_or("dma_line_combine",0)!=0;
        if(!store && !combine.empty())t=std::max(t,flush_held(t,now)); // a load sees every earlier store
        // Each 8-byte unit: its byte plan and the aligned memory words it touches.
        unsigned units=total/8;std::vector<std::array<TensorMapModel::Byte,8>> plan(units);
        std::map<u64,std::map<u64,std::pair<u64,unsigned>>> lines;std::vector<u64> order;std::map<u64,unsigned> last_unit;
        for(unsigned u=0;u<units;++u)for(unsigned b=0;b<8;++b){
            plan[u][b]=d.at(u*8+b,bytes,coords);
            if(!plan[u][b].valid)continue;
            u64 a=plan[u][b].address,w=a&~7ULL,l=a&~63ULL;
            if(!lines.count(l))order.push_back(l);
            auto& word=lines[l][w];word.second|=1U<<(a&7);last_unit[l]=u;
        }
        if(!store){
            // requests in first-use order; a register is written when its lines are back
            std::map<u64,u64> back,issued;std::map<u64,u64> value;
            for(u64 l:order){
                std::vector<TimelineMemory::WideWord> words;
                for(auto& [a,v]:lines[l])words.push_back({a,0,255,0});
                t=take_buffer(t,now);u64 at=0;u64 done=issue_line(words,false,t,now,at);t=at+1;
                for(auto& w:words)value[w.address]=w.result;
                back[l]=done;issued[l]=at;r.issue_end=t;
                line_release.push_back(done+rf); // provisional: corrected below when the register is written
            }
            std::map<u64,u64> release;
            for(unsigned reg=0;reg*64<total;++reg){
                u64 ready=now;std::vector<u64> used;
                for(unsigned u=reg*8;u<std::min(units,reg*8+8);++u)for(unsigned b=0;b<8;++b)
                    if(plan[u][b].valid){u64 l=plan[u][b].address&~63ULL;ready=std::max(ready,back[l]);used.push_back(l);}
                u64 w=line_rf.reserve(ready,rf,now)+rf;
                for(u64 l:used)release[l]=std::max(release[l],w);
                for(unsigned u=reg*8;u<std::min(units,reg*8+8);++u){
                    u64 result=0;
                    for(unsigned b=0;b<8;++b){unsigned char v=plan[u][b].padding;
                        if(plan[u][b].valid){u64 a=plan[u][b].address;v=static_cast<unsigned char>(value[a&~7ULL]>>((a&7)*8));}
                        result|=u64(v)<<(b*8);}
                    fast_rf_write(bank,base,u*8,result,owner,w);
                }
                r.rf_end=std::max(r.rf_end,w);r.end=std::max(r.end,w);
            }
            // replace the provisional releases of this command's buffers by the register-write cycles
            for(u64 l:order){auto it=std::find(line_release.begin(),line_release.end(),back[l]+rf);
                if(it!=line_release.end())*it=std::max(release[l],back[l]);}
            r.end=std::max(r.end,r.issue_end)+x.c.get("dma_drain_cycles");
            if(std::getenv("TLM_DEBUG_LINE"))std::fprintf(stderr,"LINE core %u now %llu store 0 lines %zu issue_end %llu rf_end %llu end %llu\n",core,(unsigned long long)now,order.size(),(unsigned long long)r.issue_end,(unsigned long long)r.rf_end,(unsigned long long)r.end);
            return r;
        }
        // store: read registers in order into line buffers; a line leaves when its last register is read
        std::map<u64,std::map<u64,std::pair<u64,unsigned>>> data; // line -> word -> (value, byte mask)
        std::map<unsigned,std::vector<u64>> closes;for(auto& [l,u]:last_unit)closes[u].push_back(l);
        u64 rt=now;r.end=now;
        for(unsigned reg=0;reg*64<total;++reg){
            rt=line_rf_read.reserve(rt,rf,now)+rf;
            auto v=bank->read(base+reg,sc_time_stamp(),owner);
            for(unsigned u=reg*8;u<std::min(units,reg*8+8);++u){
                unsigned lane=(u%8)*2;u64 unit=u64(v[lane])|(u64(v[lane+1])<<32);
                for(unsigned b=0;b<8;++b)if(plan[u][b].valid){u64 a=plan[u][b].address;auto& word=data[a&~63ULL][a&~7ULL];
                    word.first|=((unit>>(b*8))&255)<<((a&7)*8);word.second|=1U<<(a&7);}
                for(u64 l:closes[u]){
                    auto& words=data[l];
                    if(merge){
                        auto it=combine.find(l);
                        if(it!=combine.end()){ // complete a held line
                            for(auto& [a,wv]:it->second->words){auto& w=words[a];
                                w.first=(w.first&~mask64(wv.second & ~w.second))|(wv.first & mask64(wv.second & ~w.second));w.second|=wv.second;}
                            ++x.counts[key("line_merges")];
                        }
                        bool full=words.size()==8;for(auto& [a,w]:words)full=full && w.second==255;
                        if(!full){
                            auto h=it!=combine.end()?it->second:std::make_shared<Held>();h->line=l;h->words=words;
                            if(it==combine.end()){t=take_buffer(std::max(t,rt),now);line_release.push_back(UINT64_MAX/2);combine[l]=h;}
                            r.held.push_back(h);++x.counts[key("line_holds")];continue;
                        }
                        if(it!=combine.end()){
                            auto h=it->second;combine.erase(it);
                            auto rel=std::find(line_release.begin(),line_release.end(),UINT64_MAX/2);if(rel!=line_release.end())line_release.erase(rel);
                            std::vector<TimelineMemory::WideWord> ws;for(auto& [a,w]:words)ws.push_back({a,w.first,w.second,0});
                            u64 at=0;u64 done=issue_line(ws,true,std::max(t,rt),now,at);t=at+1;
                            h->done=done;h->flushed=true;h->event.notify(SC_ZERO_TIME);r.end=std::max(r.end,done);
                            continue;
                        }
                    }
                    t=take_buffer(std::max(t,rt),now);
                    std::vector<TimelineMemory::WideWord> ws;for(auto& [a,w]:words)ws.push_back({a,w.first,w.second,0});
                    u64 at=0;u64 done=issue_line(ws,true,t,now,at);t=at+1;line_release.push_back(at+1);
                    r.end=std::max(r.end,done);
                }
            }
        }
        r.rf_end=rt;r.issue_end=std::max(t,rt);
        // a held line waits only while another store is queued behind this one
        if(merge && !combine.empty() && queued==0)r.end=std::max(r.end,flush_held(r.issue_end,now));
        r.end=std::max(r.end,r.issue_end)+x.c.get("dma_drain_cycles");
        if(std::getenv("TLM_DEBUG_LINE"))std::fprintf(stderr,"LINE core %u now %llu store 1 lines %zu issue_end %llu rf_end %llu end %llu\n",core,(unsigned long long)now,data.size(),(unsigned long long)r.issue_end,(unsigned long long)r.rf_end,(unsigned long long)r.end);
        return r;
    }
    static u64 mask64(unsigned bytes){u64 m=0;for(unsigned b=0;b<8;++b)if(bytes&(1U<<b))m|=255ULL<<(8*b);return m;}
    void drain(){while(admission.used)wait(admission.changed);}
    void reset(){
        require(admission.used==0 && word_slots.used==0,"DMA reset before command drain");
        for(auto& d:descriptors)d=Descriptor{};
        for(auto& w:windows)w=Window{};
    }
    u64 word(u64 address,bool store,u64 value,unsigned mask){
        if(tm && x.functional){u64 r=tm->access(address,store,value,store?mask:255);
            if(store){if(!writes)writes=&x.counts[write_key];++*writes;}else{if(!reads)reads=&x.counts[read_key];++*reads;}
            return r;}
        if(tm){ // descriptor snapshots on the timeline path
            u64 now=x.now(),at=port->reserve(now,1,now),result=0;
            u64 done=tm->word(core,address,store,value,store?mask:255,at,result);
            if(store){if(!writes)writes=&x.counts[write_key];++*writes;}else{if(!reads)reads=&x.counts[read_key];++*reads;}
            if(done>now)wait(x.cycles(done-now));
            return result;
        }
        x.c.address(address,8);word_slots.acquire(x,tags_key);
        std::array<unsigned char,8> data{},enables{};
        for(unsigned b=0;b<8;++b){data[b]=static_cast<unsigned char>(value>>(b*8));enables[b]=(mask&(1<<b))?255:0;}
        Request request;request.id=++sequence;request.core=core;request.op=store?1:0;request.be=mask;
        tlm::tlm_generic_payload payload;payload.set_command(store?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
        payload.set_address(address);payload.set_data_ptr(data.data());payload.set_data_length(8);payload.set_streaming_width(8);
        payload.set_byte_enable_ptr(enables.data());payload.set_byte_enable_length(8);payload.set_extension(&request);
        sc_time delay=SC_ZERO_TIME;memory->b_transport(payload,delay);payload.clear_extension<Request>();
        require(payload.is_response_ok() && delay==SC_ZERO_TIME,"DMA memory response failure");
        word_slots.release();
        if(store){if(!writes)writes=&x.counts[write_key];++*writes;}else{if(!reads)reads=&x.counts[read_key];++*reads;}
        if(!peak)peak=&x.counts[peak_key];
        *peak=word_slots.peak;
        u64 result=0;for(unsigned b=0;b<8;++b)result|=u64(data[b])<<(b*8);return result;
    }
    void configure(unsigned function,unsigned slot,u64 address,unsigned immediate){
        arch((function<16 && slot<16) || (function>=16 && slot<32),4,"DESCRIPTOR_FAULT: configuration slot");
        if(function==2)arch(descriptors[slot].references==0,6,"RESOURCE_EXHAUSTED: descriptor still referenced");
        if(function==16 || function==17)arch(windows[slot].references==0,6,"RESOURCE_EXHAUSTED: window still referenced");
        admission.acquire(x,key("admission"));controller.lock();
        try{
            if(function==0){
                require(address%8==0,"unaligned TensorMap snapshot address is not implemented");
                std::array<unsigned char,256> data{};
                if(line_mode()){
                    // the line engine reads the 256-byte map as whole lines, all in flight at once
                    u64 now=x.now(),t=now,done=now;std::map<u64,std::vector<TimelineMemory::WideWord>> lines;
                    for(unsigned i=0;i<32;++i)lines[(address+8*i)&~63ULL].push_back({address+8*i,0,255,0});
                    std::map<u64,u64> value;
                    for(auto& [l,words]:lines){u64 at=0;done=std::max(done,issue_line(words,false,t,now,at));t=at+1;for(auto& w:words)value[w.address]=w.result;}
                    for(unsigned i=0;i<32;++i){u64 v=value[address+8*i];for(unsigned b=0;b<8;++b)data[i*8+b]=static_cast<unsigned char>(v>>(b*8));}
                    ++x.counts[key("line_snapshots")];
                    if(done>now)wait(x.cycles(done-now));
                }else
                for(unsigned i=0;i<32;++i){u64 value=word(address+8*i,false,0,255);for(unsigned b=0;b<8;++b)data[i*8+b]=static_cast<unsigned char>(value>>(b*8));}
                descriptors[slot].snapshot=std::make_shared<TensorMapModel>(data);descriptors[slot].source_address=address;++x.counts[key("descriptor_snapshots")];
            }else if(function==2)descriptors[slot].snapshot.reset();
            else if(function==16){
                unsigned base=(immediate>>8)&255,length=immediate&255;
                arch(length && base<64 && length<=64-base,4,"DESCRIPTOR_FAULT: window exceeds role register mapping");
                windows[slot]={true,base,length,0};
            }else if(function==17)windows[slot]=Window{};
            else throw std::runtime_error("reserved descriptor/window operation");
        }catch(...){controller.unlock();admission.release();throw;}
        controller.unlock();admission.release();
    }
    u64 rf_read(const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,unsigned byte,u64 owner){
        rf_port.use(x,rf_cycles,rf_key);
        auto data=bank->read(base+byte/64,sc_time_stamp(),owner);unsigned lane=(byte%64)/4;
        return u64(data[lane])|(u64(data[lane+1])<<32);
    }
    void rf_write(const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,unsigned byte,u64 value,u64 owner,unsigned size=8){
        rf_port.use(x,rf_cycles,rf_key);
        unsigned index=base+byte/64,lane=(byte%64)/4;auto data=bank->read(index,sc_time_stamp(),owner);
        data[lane]=std::uint32_t(value);if(size>4)data[lane+1]=std::uint32_t(value>>32);
        bank->write(index,(size>4?3U:1U)<<lane,sc_time_stamp(),data,owner);bank->commit(sc_time_stamp());
    }
    u64 unit(const TensorMapModel& descriptor,const std::array<u64,5>& coordinates,unsigned bytes,
             unsigned flat_byte,bool store,u64 data){
        struct Beat {u64 value=0;unsigned mask=0;};
        std::array<TensorMapModel::Byte,8> plan{};std::map<u64,Beat> beats;
        for(unsigned b=0;b<8;++b){
            plan[b]=descriptor.at(flat_byte+b,bytes,coordinates);
            if(plan[b].valid){auto& beat=beats[plan[b].address&~7ULL];unsigned offset=unsigned(plan[b].address&7);
                beat.mask|=1<<offset;beat.value|=((data>>(b*8))&255)<<(offset*8);}
        }
        require(beats.size()<=2,"DMA unit expanded beyond two aligned memory words");
        // At most two beats: the second is issued in parallel by a pooled
        // worker, the first by this unit's own process.
        unsigned pending=0;sc_event done;
        auto first=beats.begin();
        if(beats.size()==2){auto* beat=&*std::next(first);++pending;
            beat_workers.submit([&,beat](){beat->second.value=word(beat->first,store,beat->second.value,store?beat->second.mask:255);if(--pending==0)done.notify(SC_ZERO_TIME);});}
        if(first!=beats.end())first->second.value=word(first->first,store,first->second.value,store?first->second.mask:255);
        while(pending)wait(done);
        u64 result=0;
        if(!store)for(unsigned b=0;b<8;++b){
            unsigned char value=plan[b].padding;
            if(plan[b].valid)value=static_cast<unsigned char>(beats.at(plan[b].address&~7ULL).value>>((plan[b].address&7)*8));
            result|=u64(value)<<(b*8);
        }
        return result;
    }
    void transfer(const TensorMapModel& descriptor,const std::array<u64,5>& coordinates,unsigned bytes,
                  unsigned total_bytes,bool store,bool transpose,const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,u64 owner){
        if(xlog())std::fprintf(stderr,"XCMD core %u t %llu store %d xpose %d base %u dbase %llx c0 %lld c1 %lld total %u ext0 %llu ext1 %llu\n",core,(unsigned long long)x.now(),int(store),int(transpose),base,
            (unsigned long long)descriptor.base,(long long)coordinates[0],(long long)coordinates[1],total_bytes,(unsigned long long)descriptor.extent[0],(unsigned long long)descriptor.extent[1]);
        static const bool slow=std::getenv("TLM_DEBUG_SLOW_DMA")!=nullptr; // differential debugging only
        static const bool slow_x=std::getenv("TLM_DEBUG_SLOW_XPOSE")!=nullptr,slow_s=std::getenv("TLM_DEBUG_SLOW_STORE")!=nullptr;
        if(tm && !slow && !(slow_x && transpose) && !(slow_s && store)){fast_transfer(descriptor,coordinates,bytes,total_bytes,store,transpose,bank,base,owner);return;}
        if(transpose){
            // The hardware corner-turn buffer collects one four-column group
            // across all source rows before writing the destination rows.
            unsigned rows=descriptor.block[0],columns=descriptor.block[1],per=8/bytes;
            for(unsigned column=0;column<columns;column+=per){
                std::vector<u64> source(rows);unsigned pending=0;sc_event done;
                for(unsigned row=0;row<rows;++row){++pending;unit_workers.submit([&,row](){source[row]=unit(descriptor,coordinates,bytes,(row*columns+column)*bytes,false,0);if(--pending==0)done.notify(SC_ZERO_TIME);});}
                while(pending)wait(done);
                for(unsigned col=0;col<per;++col)corner_turn(source,column,col,rows,bytes,[&](unsigned byte,u64 value,unsigned size){
                    if(xlog())std::fprintf(stderr,"XPOSE slow core %u base %u byte %u value %llx now %llu\n",core,base,byte,(unsigned long long)value,(unsigned long long)x.now());
                    rf_write(bank,base,byte,value,owner,size);});
            }
            return;
        }
        Limit units(unsigned(x.c.get("dma_unit_slots")));unsigned pending=0;sc_event done;
        for(unsigned byte=0;byte<total_bytes;byte+=8){
            units.acquire(x,units_key);++pending;
            unit_workers.submit([&,byte](){
                u64 value=store?rf_read(bank,base,byte,owner):0;
                value=unit(descriptor,coordinates,bytes,byte,store,value);
                if(!store)rf_write(bank,base,byte,value,owner);
                units.release();if(--pending==0)done.notify(SC_ZERO_TIME);
            });
        }
        while(pending)wait(done);
        x.counts[key("peak_units")]=std::max<u64>(x.counts[key("peak_units")],units.peak);
    }
    void issue(unsigned descriptor_slot,unsigned window_slot,u64 token,
               unsigned bytes,bool store,bool transpose,bool active,std::shared_ptr<PhysicalRegisters> bank,unsigned vbase,unsigned role,
               const std::function<std::array<u64,5>(unsigned)>& coordinate_reader,
               const std::function<void(unsigned,unsigned)>& prepare){
        admission.acquire(x,key("admission"));
        std::shared_ptr<const TensorMapModel> descriptor;Window window;unsigned total=0,used_registers=0,base=0;u64 owner=0;
        std::array<u64,5> coordinates{};
        try{
            verify_token(token);
            if(active){
                arch(descriptor_slot<16 && descriptors[descriptor_slot].snapshot,4,"DESCRIPTOR_FAULT: unloaded descriptor");
                arch(window_slot<32 && windows[window_slot].valid,4,"DESCRIPTOR_FAULT: undefined window");
                descriptor=descriptors[descriptor_slot].snapshot;window=windows[window_slot];
                total=descriptor->elements(bytes,window.length)*bytes;
                coordinates=coordinate_reader(descriptor->rank);
                if(descriptor->padding==2)arch(descriptor->inside(coordinates),4,"DESCRIPTOR_FAULT: out-of-bounds fault padding");
                if(transpose)descriptor->check_transpose(bytes,store,coordinates);
                used_registers=(total+63)/64;base=vbase+window.base;prepare(window.base,used_registers);
                owner=bank->new_owner();bank->acquire(base,used_registers,owner,!store,sc_time_stamp());
                ++descriptors[descriptor_slot].references;++windows[window_slot].references;
            }
        }catch(...){admission.release();throw;}
        ++x.counts[key("commands_issued")];require(role<4,"transfer issuer context");++outstanding[role];
        // Admitted commands run in order on one persistent controller process.
        if(active && !transpose && line_mode()){
            ++queued;
            command_worker.submit([=](){
                controller.lock();--queued;auto start=sc_time_stamp();LineResult lr;
                try{
                    // command decode overlaps the previous command's line issue (pipelined front end)
                    u64 setup=x.c.get("dma_setup_cycles")+(rtl?descriptor->rank:0),now0=x.now();
                    if(now0<last_issue_end+setup){u64 ready=last_issue_end;if(ready>now0)wait(x.cycles(ready-now0));}
                    else wait(x.cycles(setup));
                    lr=line_transfer(*descriptor,coordinates,bytes,total,store,bank,base,owner);
                    u64 now=x.now();if(lr.issue_end>now)wait(x.cycles(lr.issue_end-now));
                    last_issue_end=x.now();
                }catch(...){completion(token,true);controller.unlock();admission.release();--outstanding[role];role_done.notify(SC_ZERO_TIME);throw;}
                controller.unlock();++x.counts[key("line_commands")];
                // one waiter per command (a dynamic process): completions never queue behind each other
                sc_spawn([=](){
                    LineResult l=lr;u64 now=x.now();
                    if(store){if(l.rf_end>now)wait(x.cycles(l.rf_end-now));bank->release(base,used_registers,owner,false,sc_time_stamp());}
                    for(auto& h:l.held){while(!h->flushed)wait(h->event);l.end=std::max(l.end,h->done);}
                    now=x.now();if(l.end>now)wait(x.cycles(l.end-now));
                    bank->commit(sc_time_stamp());
                    if(!store)bank->release(base,used_registers,owner,true,sc_time_stamp());
                    --descriptors[descriptor_slot].references;--windows[window_slot].references;
                    completion(token,false);++x.counts[key("commands_completed")];
                    x.span(store?"DMA store complete":"DMA load complete",400+core,start,token);
                    admission.release();--outstanding[role];role_done.notify(SC_ZERO_TIME);
                });
            });
            return;
        }
        command_worker.submit([=](){
            controller.lock();auto start=sc_time_stamp();
            try{
                if(!x.functional)wait(x.cycles(active?x.c.get("dma_setup_cycles")+(rtl?descriptor->rank:0):1));
                if(active){
                    transfer(*descriptor,coordinates,bytes,total,store,transpose,bank,base,owner);
                    bank->release(base,used_registers,owner,!store,sc_time_stamp());
                    --descriptors[descriptor_slot].references;--windows[window_slot].references;
                }
                completion(token,false);++x.counts[key("commands_completed")];
                x.span(store?"DMA store complete":"DMA load complete",400+core,start,token);
            }catch(...){completion(token,true);controller.unlock();admission.release();--outstanding[role];role_done.notify(SC_ZERO_TIME);throw;}
            controller.unlock();admission.release();--outstanding[role];role_done.notify(SC_ZERO_TIME);
        });
    }
};
