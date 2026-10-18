#pragma once
#include "tnca_model.hpp"

// ISA 9A generation state, independent arrival/operation accounting. Wakeups
// are events rather than polling cycles. Faults refuse the experiment; they
// must never be published as a successful zero-time completion.
struct BarrierModel {
    u64 epoch=0,completed=0;
    unsigned arrivals=0,operations=0,waiters=0;
    bool armed=false,error=false;
    sc_event changed;
    void reset(){
        require(waiters==0,"BARRIER_FAULT: init under a sleeping waiter");
        changed.cancel();epoch=completed=0;arrivals=operations=0;armed=error=false;
    }
    void settle(){
        if(armed && arrivals==0 && operations==0){completed=epoch;armed=false;changed.notify(SC_ZERO_TIME);}
    }
    u64 arm(unsigned id,unsigned expected_arrivals,unsigned expected_operations){
        require(!error,"BARRIER_FAULT: arm invalid barrier");
        require(!armed,"BARRIER_FAULT: rearm with outstanding obligations");
        require(epoch<((1ULL<<48)-1),"BARRIER_FAULT: generation wrap");
        require(expected_arrivals<256 && expected_operations<256,"BARRIER_FAULT: count overflow");
        ++epoch;arrivals=expected_arrivals;operations=expected_operations;armed=true;settle();
        return (epoch<<16)|id;
    }
    void arrive(){
        require(!error && armed && arrivals>0,"BARRIER_FAULT: arrival underflow");
        --arrivals;settle();
    }
    void complete(bool failed=false){
        if(failed){invalidate();return;}
        require(!error && armed && operations>0,"BARRIER_FAULT: operation completion underflow");
        --operations;settle();
    }
    bool test(u64 token)const{
        require(!error,"BARRIER_FAULT: invalidated generation");return completed>=(token>>16);
    }
    void sleep(u64 token){
        if(test(token))return;
        ++waiters;
        while(!error && completed<(token>>16))wait(changed);
        --waiters;require(!error,"BARRIER_FAULT: invalidation woke waiter");
    }
    void invalidate(){error=true;completed=epoch;armed=false;arrivals=operations=0;changed.notify(SC_ZERO_TIME);}
};

struct PhysicalRegisters {
    using Vector=std::array<std::uint32_t,16>;
    struct Write {unsigned index,mask;sc_time due;u64 sequence;Vector data;};
    std::array<Vector,256> values{};
    std::vector<Write> pending;
    u64 sequence=0;
    void commit(sc_time now){
        std::stable_sort(pending.begin(),pending.end(),[](const Write& a,const Write& b){return a.due<b.due || (a.due==b.due && a.sequence<b.sequence);});
        auto end=pending.begin();
        while(end!=pending.end() && end->due<=now){
            for(unsigned lane=0;lane<16;++lane)if(end->mask&(1<<lane))values[end->index][lane]=end->data[lane];
            ++end;
        }
        pending.erase(pending.begin(),end);
    }
    void write(unsigned index,unsigned mask,sc_time due,const Vector& data){
        require(index<256,"physical vector register overflow");pending.push_back({index,mask,due,sequence++,data});
    }
    Vector read(unsigned index,sc_time now){require(index<256,"physical vector register overflow");commit(now);return values[index];}
};
