#pragma once
#include "tnca_tensor_model.hpp"
#include "tnca_barrier_model.hpp"
#include <functional>

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
    std::function<void(u64)> verify_token;
    std::function<void(u64,bool)> completion;
    u64 sequence=0;
    TensorEngine(sc_module_name name,Context& context,unsigned id,
                 std::function<void(u64)> verify,std::function<void(u64,bool)> done):
        sc_module(name),memory("memory"),x(context),core(id),
        admission(unsigned(x.c.get("dma_queue"))),word_slots(unsigned(x.c.get("dma_outstanding"))),
        verify_token(std::move(verify)),completion(std::move(done)){}
    std::string key(const std::string& suffix)const{return "core_"+std::to_string(core)+"_dma_"+suffix;}
    void drain(){while(admission.used)wait(admission.changed);}
    void reset(){
        require(admission.used==0 && word_slots.used==0,"DMA reset before command drain");
        for(auto& d:descriptors)d=Descriptor{};
        for(auto& w:windows)w=Window{};
    }
    u64 word(u64 address,bool store,u64 value,unsigned mask){
        x.c.address(address,8);word_slots.acquire(x,key("tags"));
        std::array<unsigned char,8> data{},enables{};
        for(unsigned b=0;b<8;++b){data[b]=static_cast<unsigned char>(value>>(b*8));enables[b]=(mask&(1<<b))?255:0;}
        Request request;request.id=++sequence;request.core=core;request.op=store?1:0;request.be=mask;
        tlm::tlm_generic_payload payload;payload.set_command(store?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
        payload.set_address(address);payload.set_data_ptr(data.data());payload.set_data_length(8);payload.set_streaming_width(8);
        payload.set_byte_enable_ptr(enables.data());payload.set_byte_enable_length(8);payload.set_extension(&request);
        sc_time delay=SC_ZERO_TIME;memory->b_transport(payload,delay);payload.clear_extension<Request>();
        require(payload.is_response_ok() && delay==SC_ZERO_TIME,"DMA memory response failure");
        word_slots.release();++x.counts[key(store?"write_words":"read_words")];
        x.counts[key("peak_tags")]=word_slots.peak;
        u64 result=0;for(unsigned b=0;b<8;++b)result|=u64(data[b])<<(b*8);return result;
    }
    void configure(unsigned function,unsigned slot,u64 address,unsigned immediate){
        require((function<16 && slot<16) || (function>=16 && slot<32),"DESCRIPTOR_FAULT: configuration slot");
        if(function==2)require(descriptors[slot].references==0,"RESOURCE_EXHAUSTED: descriptor still referenced");
        if(function==16 || function==17)require(windows[slot].references==0,"RESOURCE_EXHAUSTED: window still referenced");
        admission.acquire(x,key("admission"));controller.lock();
        try{
            if(function==0){
                require(address%8==0,"unaligned TensorMap snapshot address is not implemented");
                std::array<unsigned char,256> data{};
                for(unsigned i=0;i<32;++i){u64 value=word(address+8*i,false,0,255);for(unsigned b=0;b<8;++b)data[i*8+b]=static_cast<unsigned char>(value>>(b*8));}
                descriptors[slot].snapshot=std::make_shared<TensorMapModel>(data);descriptors[slot].source_address=address;++x.counts[key("descriptor_snapshots")];
            }else if(function==2)descriptors[slot].snapshot.reset();
            else if(function==16){
                unsigned base=(immediate>>8)&255,length=immediate&255;
                require(length && base<64 && length<=64-base,"DESCRIPTOR_FAULT: window exceeds role register mapping");
                windows[slot]={true,base,length,0};
            }else if(function==17)windows[slot]=Window{};
            else throw std::runtime_error("reserved descriptor/window operation");
        }catch(...){controller.unlock();admission.release();throw;}
        controller.unlock();admission.release();
    }
    u64 rf_read(const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,unsigned byte,u64 owner){
        rf_port.use(x,x.c.get("dma_rf_cycles"),key("rf"));
        auto data=bank->read(base+byte/64,sc_time_stamp(),owner);unsigned lane=(byte%64)/4;
        return u64(data[lane])|(u64(data[lane+1])<<32);
    }
    void rf_write(const std::shared_ptr<PhysicalRegisters>& bank,unsigned base,unsigned byte,u64 value,u64 owner){
        rf_port.use(x,x.c.get("dma_rf_cycles"),key("rf"));
        unsigned index=base+byte/64,lane=(byte%64)/4;auto data=bank->read(index,sc_time_stamp(),owner);
        data[lane]=std::uint32_t(value);data[lane+1]=std::uint32_t(value>>32);
        bank->write(index,3U<<lane,sc_time_stamp(),data,owner);bank->commit(sc_time_stamp());
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
        unsigned pending=0;sc_event done;
        for(auto& item:beats){auto* beat=&item;++pending;sc_spawn([&,beat](){beat->second.value=word(beat->first,store,beat->second.value,store?beat->second.mask:255);--pending;done.notify(SC_ZERO_TIME);});}
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
        if(transpose){
            // The hardware corner-turn buffer collects one four-column group
            // across all source rows before writing the destination rows.
            unsigned rows=descriptor.block[0],columns=descriptor.block[1];
            for(unsigned column=0;column<columns;column+=4){
                std::vector<u64> source(rows);unsigned pending=0;sc_event done;
                for(unsigned row=0;row<rows;++row){++pending;sc_spawn([&,row](){source[row]=unit(descriptor,coordinates,bytes,row*columns*2+column*2,false,0);--pending;done.notify(SC_ZERO_TIME);});}
                while(pending)wait(done);
                for(unsigned col=0;col<4;++col)for(unsigned row=0;row<rows;row+=4){
                    u64 value=0;for(unsigned i=0;i<4;++i)value|=((source[row+i]>>(col*16))&65535)<<(i*16);
                    rf_write(bank,base,((column+col)*rows+row)*2,value,owner);
                }
            }
            return;
        }
        Limit units(unsigned(x.c.get("dma_unit_slots")));unsigned pending=0;sc_event done;
        for(unsigned byte=0;byte<total_bytes;byte+=8){
            units.acquire(x,key("units"));++pending;
            sc_spawn([&,byte](){
                u64 value=store?rf_read(bank,base,byte,owner):0;
                value=unit(descriptor,coordinates,bytes,byte,store,value);
                if(!store)rf_write(bank,base,byte,value,owner);
                units.release();--pending;done.notify(SC_ZERO_TIME);
            });
        }
        while(pending)wait(done);
        x.counts[key("peak_units")]=std::max<u64>(x.counts[key("peak_units")],units.peak);
    }
    void issue(unsigned descriptor_slot,unsigned window_slot,u64 token,
               unsigned bytes,bool store,bool transpose,bool active,std::shared_ptr<PhysicalRegisters> bank,unsigned vbase,
               const std::function<std::array<u64,5>(unsigned)>& coordinate_reader,
               const std::function<void(unsigned,unsigned)>& prepare){
        admission.acquire(x,key("admission"));
        std::shared_ptr<const TensorMapModel> descriptor;Window window;unsigned total=0,used_registers=0,base=0;u64 owner=0;
        std::array<u64,5> coordinates{};
        try{
            verify_token(token);
            if(active){
                require(descriptor_slot<16 && descriptors[descriptor_slot].snapshot,"DESCRIPTOR_FAULT: unloaded descriptor");
                require(window_slot<32 && windows[window_slot].valid,"DESCRIPTOR_FAULT: undefined window");
                descriptor=descriptors[descriptor_slot].snapshot;window=windows[window_slot];
                total=descriptor->elements(bytes,window.length)*bytes;
                coordinates=coordinate_reader(descriptor->rank);
                if(descriptor->padding==2)require(descriptor->inside(coordinates),"DESCRIPTOR_FAULT: out-of-bounds fault padding");
                if(transpose)descriptor->check_transpose(bytes,store,coordinates);
                used_registers=(total+63)/64;base=vbase+window.base;prepare(window.base,used_registers);
                owner=bank->new_owner();bank->acquire(base,used_registers,owner,!store,sc_time_stamp());
                ++descriptors[descriptor_slot].references;++windows[window_slot].references;
            }
        }catch(...){admission.release();throw;}
        ++x.counts[key("commands_issued")];
        sc_spawn([=](){
            controller.lock();auto start=sc_time_stamp();
            try{
                wait(x.cycles(active?x.c.get("dma_setup_cycles"):1));
                if(active){
                    transfer(*descriptor,coordinates,bytes,total,store,transpose,bank,base,owner);
                    bank->release(base,used_registers,owner,!store,sc_time_stamp());
                    --descriptors[descriptor_slot].references;--windows[window_slot].references;
                }
                completion(token,false);++x.counts[key("commands_completed")];
                x.span(store?"DMA store complete":"DMA load complete",400+core,start,token);
            }catch(...){completion(token,true);controller.unlock();admission.release();throw;}
            controller.unlock();admission.release();
        });
    }
};
