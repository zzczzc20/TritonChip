// Native event-driven execution of compiled TNCA machine words. No RTL runtime.
// This first execution slice deliberately refuses uncovered instruction groups;
// it must not be called a complete-chip qualification until those are delivered.
#include "../include/tnca_model.hpp"
#include "../include/tnca_numeric.hpp"
#include "../include/tnca_barrier_model.hpp"
#include "tnca_isa_generated.hpp"
#include <cstring>
#include <cmath>

static bool compare(u64 a,u64 b,unsigned cc,unsigned bits=64){
    auto sa=bits==64?std::int64_t(a):std::int64_t(std::int32_t(a));
    auto sb=bits==64?std::int64_t(b):std::int64_t(std::int32_t(b));
    switch(cc){case 0:return a==b;case 1:return a!=b;case 2:return sa<sb;case 3:return sa<=sb;
        case 4:return sa>sb;case 5:return sa>=sb;case 6:return a<b;case 7:return a<=b;
        case 8:return a>b;case 9:return a>=b;default:throw std::runtime_error("uncovered comparison condition");}
}
static u64 alu(unsigned f,u64 a,u64 b,unsigned bits,unsigned cc){
    const u64 mask=bits==64?UINT64_MAX:0xffffffffULL;a&=mask;b&=mask;
    auto sa=bits==64?std::int64_t(a):std::int64_t(std::int32_t(a));
    auto sb=bits==64?std::int64_t(b):std::int64_t(std::int32_t(b));
    switch(f){case 0:return a;case 1:return (a+b)&mask;case 2:return (a-b)&mask;case 3:return (a*b)&mask;
        case 4:return bits==64?tnca_numeric::multiply_high(a,b,true):u64(std::int64_t(std::int32_t(a))*std::int64_t(std::int32_t(b)))>>32;
        case 5:return bits==64?tnca_numeric::multiply_high(a,b,false):(a*b)>>32;
        case 8:return a&b;case 9:return a|b;case 10:return a^b;case 11:return a&~b;
        case 12:return (a<<(b&(bits-1)))&mask;case 13:return a>>(b&(bits-1));
        case 14:return u64(sa>>(b&(bits-1)))&mask;
        case 16:return sa<sb?a:b;case 17:return sa>sb?a:b;case 18:return std::min(a,b);case 19:return std::max(a,b);
        case 20:return bits==64?(a?unsigned(__builtin_clzll(a)):64):(a?unsigned(__builtin_clz(unsigned(a))):32);
        case 21:return unsigned(__builtin_popcountll(a));case 22:return u64(std::int64_t(std::int32_t(a)));
        case 23:return a&0xffffffffULL;case 24:return compare(a,b,cc,bits);
        default:throw std::runtime_error("uncovered ALU operation "+std::to_string(f));}
}
static float as_float(std::uint32_t value){float result;std::memcpy(&result,&value,4);return result;}
static std::uint32_t as_bits(float value){std::uint32_t result;std::memcpy(&result,&value,4);return result;}
struct Frame {
    std::shared_ptr<PhysicalRegisters> physical;
    unsigned vbase=0;
    std::array<u64,64> s{};std::array<PhysicalRegisters::Vector,64> v{};
    std::array<unsigned,16> p{};
    std::array<sc_time,64> sr{};std::array<sc_time,64> vr{};std::array<sc_time,16> pr{};
    u64 pid=0,pc=0,arg=0,retired=0;unsigned gx=1,gy=1,gz=1,role=0,role_count=1;
    Frame():Frame(std::make_shared<PhysicalRegisters>(),0){}
    Frame(std::shared_ptr<PhysicalRegisters> bank,unsigned base):physical(std::move(bank)),vbase(base){require(base<=192,"vector mapping overflow");p[0]=65535;}
};
struct Core : sc_module {
    tlm_utils::simple_initiator_socket<Core> memory;
    Context& x;unsigned id;Resource issue;Limit lsu;sc_time sfu_available=SC_ZERO_TIME;
    std::array<BarrierModel,32> barriers;
    explicit Core(sc_module_name n,Context& ctx,unsigned i):sc_module(n),memory("memory"),x(ctx),id(i),lsu(2){}
    void ready(sc_time t){if(t>sc_time_stamp())wait(t-sc_time_stamp());}
    void publish(Frame& frame,unsigned index,unsigned mask,u64 latency){
        require(index<64,"logical vector register exceeds role allocation");
        frame.vr[index]=sc_time_stamp()+x.cycles(latency);
        frame.physical->write(frame.vbase+index,mask,frame.vr[index],frame.v[index]);
    }
    u64 word(u64 address,bool store,u64 data,unsigned be,u64 instruction){
        if(!store)be=255; // NUCA returns the complete aligned word; LSU extracts bytes.
        Request ext;ext.id=instruction;ext.core=id;ext.op=store?1:0;ext.be=be;
        std::array<unsigned char,8> bytes{},enables{};
        for(unsigned b=0;b<8;++b){bytes[b]=unsigned(data>>(b*8))&255;enables[b]=be&(1<<b)?255:0;}
        tlm::tlm_generic_payload tx;tx.set_address(address);tx.set_command(store?tlm::TLM_WRITE_COMMAND:tlm::TLM_READ_COMMAND);
        tx.set_data_ptr(bytes.data());tx.set_data_length(8);tx.set_streaming_width(8);
        tx.set_byte_enable_ptr(enables.data());tx.set_byte_enable_length(8);tx.set_extension(&ext);
        sc_time delay=SC_ZERO_TIME;memory->b_transport(tx,delay);tx.clear_extension<Request>();
        require(tx.is_response_ok() && delay==SC_ZERO_TIME,"core memory transport failed");
        u64 result=0;for(unsigned b=0;b<8;++b)result|=u64(bytes[b])<<(b*8);return result;
    }
    u64 access(u64 address,unsigned size,bool store,u64 data,bool sign,u64 instruction){
        require(size==1 || size==2 || size==4 || size==8,"reserved LSU size");
        x.c.address(address,size);u64 result=0;unsigned done=0;
        while(done<size){
            u64 a=address+done;unsigned off=unsigned(a&7),n=std::min(size-done,8-off);
            u64 v=word(a&~7ULL,store,data<<(off*8),unsigned(((1<<n)-1)<<off),instruction);
            u64 mask=n==8?UINT64_MAX:((1ULL<<(n*8))-1);
            result|=((v>>(off*8))&mask)<<(done*8);data>>=n==8?0:n*8;done+=n;
        }
        if(sign && size<8 && (result&(1ULL<<(size*8-1))))result|=UINT64_MAX<<(size*8);
        return result;
    }
    u64 special(unsigned spec,const Frame& f){
        switch(spec){case 0:return f.pid%f.gx;case 1:return (f.pid/f.gx)%f.gy;case 2:return f.pid/u64(f.gx)/f.gy;
            case 4:return f.gx;case 5:return f.gy;case 6:return f.gz;case 8:return f.role;case 9:return f.role_count;
            case 10:return id;case 11:return x.c.cores();case 12:return 16;case 13:return 64;
            case 16:return x.now();case 17:return f.retired;case 32:return f.arg;
            default:throw std::runtime_error("uncovered special register "+std::to_string(spec));}
    }
    void execute(const std::vector<u64>& code,Frame& f,u64 max_instructions){
        while(true){
            require(f.retired<max_instructions,"instruction limit exceeded (loop or unsupported termination)");
            require(f.pc%8==0 && f.pc/8<code.size(),"instruction fetch outside uploaded text");
            u64 ins=code[f.pc/8];unsigned major=unsigned(ins>>56),func=unsigned(ins>>48)&255;
            unsigned d=unsigned(ins>>40)&255,a=unsigned(ins>>32)&255,b=unsigned(ins>>24)&255,c=unsigned(ins>>16)&255;
            unsigned pred=unsigned(ins>>12)&15,mod=unsigned(ins)&1023;bool invert=(ins>>10)&1;
            unsigned imm=(b<<8)|c;u64 simm=u64(std::int64_t(std::int16_t(imm))),next=f.pc+8;
            unsigned opcode=(major<<8)|func;auto begin=sc_time_stamp();
            require(isa::defined(opcode),"undefined ISA opcode");
            require(isa::maskable(opcode) || (pred==0 && !invert),"predicate on unmaskable instruction");
            auto rs=[&](unsigned r){require(r<64,"scalar register index");ready(f.sr[r]);return r?f.s[r]:0ULL;};
            auto ws=[&](unsigned r,u64 value,u64 latency){require(r<64,"scalar destination");if(r){ready(f.sr[r]);f.s[r]=value;f.sr[r]=sc_time_stamp()+x.cycles(latency);}};
            auto rv=[&](unsigned r){require(r<64,"logical vector register exceeds role allocation");ready(f.vr[r]);f.v[r]=f.physical->read(f.vbase+r,sc_time_stamp());};
            unsigned mask=65535;
            if(major>=16){ready(f.pr[pred]);mask=f.p[pred];if(invert)mask^=65535;}
            // No cycle polling: issue and dependencies reserve their next event.
            issue.use(x,1,"core_"+std::to_string(id)+"_issue");
            if(opcode==isa::exit){for(auto& t:f.sr)ready(t);for(auto& t:f.vr)ready(t);f.physical->commit(sc_time_stamp());++f.retired;++x.counts["core_"+std::to_string(id)+"_instructions"];break;}
            if(opcode==isa::nop || opcode==isa::yield){}
            else if(opcode==isa::s_rdspecial)ws(d,special(mod,f),2);
            else if(major==1){
                u64 value=0;
                if(func>=64 && func<=66){u64 val=u64(imm)<<(16*(mod&3));value=func==64?val:func==66?~val:(rs(d)&~(0xffffULL<<(16*(mod&3))))|val;}
                else if(func==25)value=rs(c)?rs(a):rs(b);
                else {
                    u64 left=rs(a),right=0;unsigned af=func;
                    if(func>=32 && func<=63){
                        right=(func==40 || func==41 || func==42 || func==44 || func==45 || func==46)?imm:simm;
                        af=func==32?1:func==33?2:func==34?3:func-32;
                    }else if(func!=0 && func!=20 && func!=21 && func!=22 && func!=23)right=rs(b);
                    value=alu(af,left,right,64,mod&15);
                }
                ws(d,value,(func==3 || func==34)?5:2);
            }else if(major==2){
                auto off24=std::int32_t((unsigned(a)<<24)|(unsigned(b)<<16)|(unsigned(c)<<8))>>8;
                if(opcode==isa::br)next=f.pc+u64(std::int64_t(off24)*8);
                else if(opcode==isa::br_nz || opcode==isa::br_z){bool test=rs(a)!=0;if(test==(opcode==isa::br_nz))next=f.pc+simm*8;}
                else if(opcode==isa::br_r || opcode==isa::ret)next=rs(a)&~7ULL;
                else if(opcode==isa::call){ws(d,f.pc+8,1);next=f.pc+u64(std::int64_t(off24)*8);}
                else throw std::runtime_error("uncovered control operation or user trap");
            }else if(major==32){
                require((mod&7)<=3 && (mod>>6)==0 && ((mod>>4)&3)!=3,"reserved memory modifiers");
                bool store=func==8 || func==24 || func==40,vector=func>=16,gather=func>=32;
                require(func==0 || func==8 || func==16 || func==24 || func==32 || func==40,"reserved LSU function");
                unsigned size=1U<<(mod&7);require(!vector || size<=4,"reserved vector access width");
                u64 base=rs(a);if(!gather)base+=simm;
                if(vector){rv(d);if(gather)rv(b);}else if(store)rs(d);
                lsu.acquire(x,"core_"+std::to_string(id)+"_lsu");
                struct Part {unsigned lane,offset,bytes,shift;};
                struct Beat {u64 data=0,value=0;unsigned enables=0;std::vector<Part> parts;};
                std::map<u64,Beat> beats;
                for(unsigned lane=0;lane<(vector?16U:1U);++lane){
                    if(vector && !(mask&(1<<lane)))continue;
                    u64 address=base+(vector?(gather?u64(std::int64_t(std::int32_t(f.v[b][lane])))*size:lane*size):0);
                    x.c.address(address,size);u64 value=vector?f.v[d][lane]:f.s[d];unsigned done=0;
                    while(done<size){
                        u64 address_part=address+done;unsigned off=unsigned(address_part&7),n=std::min(size-done,8-off);
                        auto& beat=beats[address_part&~7ULL];beat.parts.push_back({lane,off,n,done*8});
                        for(unsigned byte=0;byte<n;++byte){
                            unsigned at=off+byte;beat.enables|=1<<at;
                            beat.data=(beat.data&~(255ULL<<(at*8)))|(((value>>((done+byte)*8))&255)<<(at*8));
                        }done+=n;
                    }
                }
                Limit beat_slots(8);unsigned pending=0;sc_event beat_done;
                for(auto& beat:beats){
                    beat_slots.acquire(x,"core_"+std::to_string(id)+"_lsu_beats");++pending;
                    auto* entry=&beat;sc_spawn([&,entry](){entry->second.value=word(entry->first,store,entry->second.data,entry->second.enables,f.retired);beat_slots.release();--pending;beat_done.notify(SC_ZERO_TIME);});
                }
                while(pending)wait(beat_done);
                if(!store){
                    std::array<u64,16> values{};
                    for(auto& beat:beats)for(auto& part:beat.second.parts){
                        u64 part_mask=part.bytes==8?UINT64_MAX:((1ULL<<(part.bytes*8))-1);
                        values[part.lane]|=((beat.second.value>>(part.offset*8))&part_mask)<<part.shift;
                    }
                    for(unsigned lane=0;lane<(vector?16U:1U);++lane)if(!vector || (mask&(1<<lane))){
                        u64 value=values[lane];if((mod&8) && size<8 && (value&(1ULL<<(size*8-1))))value|=UINT64_MAX<<(size*8);
                        if(vector)f.v[d][lane]=std::uint32_t(value);else ws(d,value,1);
                    }
                }
                if(vector && !store)publish(f,d,mask,1);
                lsu.release();
            }else if(major==16){
                rv(d);if(func!=64 && func!=27 && func!=26)rv(a);
                bool immediate=func>=32 && func<=63;
                if(!immediate && func!=0 && func!=20 && func!=21 && func!=26 && func!=27 && func!=64)rv(b);
                unsigned comparison=0;
                for(unsigned lane=0;lane<16;++lane)if(func==25 || (mask&(1<<lane))){
                    u64 value=0;
                    if(func==27)value=lane;else if(func==26)value=rs(a);else if(func==64)value=simm;
                    else if(func==25)value=(mask&(1<<lane))?f.v[a][lane]:f.v[b][lane];
                    else {u64 right=immediate?((func==40 || func==41 || func==42 || func==44 || func==45 || func==46)?imm:simm):f.v[b][lane];
                        unsigned af=immediate?(func==32?1:func==33?2:func==34?3:func-32):func;
                        value=alu(af,f.v[a][lane],right,32,mod&15);}
                    if(func==24 || func==56){if(value)comparison|=1<<lane;}else f.v[d][lane]=std::uint32_t(value);
                }
                if(func==24 || func==56){require(d<16,"predicate destination");if(d){f.p[d]=comparison;f.pr[d]=sc_time_stamp()+x.cycles(1);}}
                else publish(f,d,func==25?65535:mask,3);
            }else if(major==80){
                require(mod==0,"reserved barrier modifier");
                if(func==0){require(d<32,"barrier index");barriers[d].reset();}
                else if(func==1){require(a<32,"barrier index");ws(d,barriers[a].arm(a,b,c),2);}
                else if(func==2){
                    require(a<32,"barrier index");
                    for(auto& t:f.sr)ready(t);
                    for(auto& t:f.vr)ready(t);
                    f.physical->commit(sc_time_stamp());
                    barriers[a].arrive();
                }else if(func==3 || func==4){
                    u64 token=rs(a);unsigned index=unsigned(token&65535);require(index<32,"BARRIER_FAULT: token index");
                    if(func==3)ws(d,barriers[index].test(token),2);
                    else {auto sleep=sc_time_stamp();barriers[index].sleep(token);x.counts["core_"+std::to_string(id)+"_barrier_sleep_cycles"]+=u64((sc_time_stamp()-sleep)/x.cycles(1)+.5);}
                }else if(func==5){require(a<32,"barrier index");barriers[a].invalidate();}
                else throw std::runtime_error("reserved barrier function");
            }else if(major==19){
                require((a<16 || func==19) && (func>=16 || (d<16 && b<16)),"predicate index");
                unsigned pa=0,pb=0;if(func!=19){ready(f.pr[a]);pa=f.p[a];}
                if(func==1 || func==2 || func==3 || func==5){ready(f.pr[b]);pb=f.p[b];}
                if(func>=16){
                    if(func==16)ws(d,pa!=0,2);else if(func==17)ws(d,pa==65535,2);
                    else if(func==18)ws(d,pa,2);else if(func==19){require(d<16,"predicate destination");if(d){f.p[d]=rs(a)&65535;f.pr[d]=sc_time_stamp()+x.cycles(2);}}
                    else if(func==20)ws(d,unsigned(__builtin_popcount(pa)),2);else throw std::runtime_error("reserved predicate function");
                }else if(d){
                    unsigned value=func==0?pa:func==1?pa&pb:func==2?pa|pb:func==3?pa^pb:func==4?~pa:func==5?pa&~pb:0;
                    require(func<=5,"reserved predicate function");f.p[d]=value&65535;f.pr[d]=sc_time_stamp()+x.cycles(2);
                }
            }else if(major==17){
                require((mod&~15U)==0,"uncovered FP modifier (including FTZ)");rv(a);rv(d);
                if(func!=20 && func!=21)rv(b);
                if(func==6 || func==7)rv(c);
                unsigned comparison=0;
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                    float left=as_float(f.v[a][lane]),right=as_float(f.v[b][lane]),value=0;
                    if(func==1)value=left+right;else if(func==2)value=left-right;else if(func==3)value=left*right;
                    else if(func==6 || func==7)value=std::fma(left,right,func==6?as_float(f.v[c][lane]):-as_float(f.v[c][lane]));
                    else if(func==16 || func==17){f.v[d][lane]=tnca_numeric::minmax(f.v[a][lane],f.v[b][lane],func==17,mod&3);continue;}
                    else if(func==20){f.v[d][lane]=f.v[a][lane]^0x80000000U;continue;}
                    else if(func==21){f.v[d][lane]=f.v[a][lane]&0x7fffffffU;continue;}
                    else if(func==24){bool condition=false;unsigned cc=mod&15;
                        switch(cc){case 0:condition=left==right;break;case 1:condition=left!=right;break;
                            case 2:condition=left<right;break;case 3:condition=left<=right;break;case 4:condition=left>right;break;case 5:condition=left>=right;break;
                            case 10:condition=!std::isnan(left) && !std::isnan(right);break;case 11:condition=std::isnan(left) || std::isnan(right);break;
                            default:throw std::runtime_error("reserved floating comparison");}
                        if(condition){comparison|=1<<lane;}
                        continue;
                    }else throw std::runtime_error("uncovered floating function");
                    f.v[d][lane]=std::isnan(value)?0x7fc00000U:as_bits(value);
                }
                if(func==24){require(d<16,"predicate destination");if(d){f.p[d]=comparison;f.pr[d]=sc_time_stamp()+x.cycles(1);}}
                else publish(f,d,mask,5);
            }else if(major==18){
                rv(a);rv(d);
                require(func<=4,"reserved conversion function");
                if(func==4)require((mod>>6)==0 && (mod&7)<5 && ((mod>>3)&7)<5,"reserved conversion format/modifier");
                else require((mod>>3)==0 && (func<2?mod==0:mod<=3),"reserved conversion modifier");
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                    auto input=f.v[a][lane];std::uint32_t value=0;
                    if(func==0)value=as_bits(float(std::int32_t(input)));
                    else if(func==1)value=as_bits(float(input));
                    else if(func==2 || func==3)value=tnca_numeric::float_to_integer(input,func==2,mod&7);
                    else value=tnca_numeric::convert(mod&7,(mod>>3)&7,input);
                    f.v[d][lane]=value;
                }publish(f,d,mask,5);
            }else if(major==22){
                require(func<=3 && mod==0,"reserved SFU function/modifier");rv(a);rv(d);
                auto waiting=sc_time_stamp();ready(sfu_available);
                x.counts["core_"+std::to_string(id)+"_sfu_queue_cycles"]+=u64((sc_time_stamp()-waiting)/x.cycles(1)+.5);
                sfu_available=sc_time_stamp()+x.cycles(24);
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane))f.v[d][lane]=tnca_numeric::sfu(func,f.v[a][lane]);
                publish(f,d,mask,28);
            }else if(major==21){
                require(func<=10 && (mod&~1U)==0,"reserved reduction function/modifier");rv(a);
                std::array<std::uint32_t,16> values{};
                std::uint32_t identity=func==1?0x80000000U:func==2?0x7fffffffU:func==3 || func==7?0xffffffffU:func==9?0xff800000U:func==10?0x7f800000U:0;
                for(unsigned lane=0;lane<16;++lane)values[lane]=(mask&(1<<lane))?f.v[a][lane]:identity;
                for(unsigned width=16;width>1;width/=2)for(unsigned i=0;i<width/2;++i){
                    auto left=values[i*2],right=values[i*2+1];std::uint32_t value=0;
                    if(func==0)value=left+right;
                    else if(func==1 || func==2)value=std::uint32_t(alu(func==1?17:16,left,right,32,0));
                    else if(func==3)value=left&right;else if(func==4)value=left|right;else if(func==5)value=left^right;
                    else if(func==6)value=std::max(left,right);else if(func==7)value=std::min(left,right);
                    else if(func==8){value=as_bits(as_float(left)+as_float(right));if(tnca_numeric::nan(0,value))value=0x7fc00000U;}
                    else value=tnca_numeric::minmax(left,right,func==9);
                    values[i]=value;
                }
                if(mod&1){rv(d);f.v[d].fill(values[0]);publish(f,d,65535,3);}
                else ws(d,values[0],3);
            }else if(opcode==isa::v_extract){rv(a);ws(d,f.v[a][mod&15],2);}
            else if(opcode==isa::v_insert){rv(d);auto value=std::uint32_t(rs(a));if(mask&(1<<(mod&15)))f.v[d][mod&15]=value;publish(f,d,mask&(1<<(mod&15)),3);}
            else if(major==20 && func<=4){
                rv(a);rv(d);auto source=f.v[a];auto indices=f.v[b];unsigned shift=0;
                if(func==3){rv(b);indices=f.v[b];}else shift=unsigned(rs(b))&15;
                for(unsigned lane=0;lane<16;++lane)if(mask&(1<<lane)){
                    unsigned from=lane;
                    if(func==0)from=lane^shift;else if(func==1)from=lane>=shift?lane-shift:lane;
                    else if(func==2)from=lane+shift<16?lane+shift:lane;else if(func==3)from=indices[lane]&15;
                    else from=(lane+shift)&15;
                    f.v[d][lane]=source[from];
                }publish(f,d,mask,3);
            }
            else throw std::runtime_error("uncovered instruction group major="+std::to_string(major)+" func="+std::to_string(func));
            ++f.retired;++x.counts["core_"+std::to_string(id)+"_instructions"];
            x.span("native instruction",id,begin,f.retired);f.pc=next;
        }
    }
    void program(const std::vector<u64>& code,u64 pid,u64 pc,u64 arg,unsigned gx,unsigned gy,unsigned gz,unsigned roles,bool pair_alias){
        require(roles>0 && roles<16,"invalid role mask");
        for(auto& barrier:barriers)barrier.reset();
        auto values=std::make_shared<PhysicalRegisters>();
        unsigned active=unsigned(__builtin_popcount(roles)),remaining=active;sc_event done;
        for(unsigned role=0;role<4;++role)if(roles&(1<<role)){
            sc_spawn([&,role](){
                unsigned base=(pair_alias?(role&~1U):role)*64;
                Frame f(values,base);f.pid=pid;f.pc=pc;f.arg=arg;f.gx=gx;f.gy=gy;f.gz=gz;f.role=role;f.role_count=active;
                ++x.counts["core_"+std::to_string(id)+"_role_"+std::to_string(role)+"_issued"];
                execute(code,f,10000000);
                ++x.counts["core_"+std::to_string(id)+"_role_"+std::to_string(role)+"_completed"];
                --remaining;done.notify(SC_ZERO_TIME);
            });
        }
        while(remaining)wait(done);
    }
};
struct Chip : sc_module {
    SC_HAS_PROCESS(Chip);
    Context& x;HBM& hbm;std::vector<std::unique_ptr<Slice>>& caches;
    std::vector<std::unique_ptr<Core>> cores;std::vector<u64> code;std::ifstream commands;
    std::ofstream result;bool terminal=false;u64 launches=0,checked=0;
    Chip(sc_module_name n,Context& ctx,HBM& mem,Network& net,std::vector<std::unique_ptr<Slice>>& s,const char* script,const std::string& out):
        sc_module(n),x(ctx),hbm(mem),caches(s),commands(script),result(out+"/summary.json"){
        require(bool(commands) && bool(result),"cannot open chip script/output");
        for(unsigned i=0;i<x.c.cores();++i){cores.emplace_back(new Core(("core"+std::to_string(i)).c_str(),x,i));cores.back()->memory.bind(net.target);}
        SC_THREAD(run);
    }
    void flush(){unsigned remaining=caches.size();sc_event done;
        for(auto& c:caches){auto* ptr=c.get();sc_spawn([&,ptr](){ptr->flush();--remaining;done.notify(SC_ZERO_TIME);});}
        while(remaining)wait(done);
        for(auto& slot:hbm.slots)require(slot->used==0,"completion visibility still has outstanding writes");
    }
    void run(){
        std::string line;bool host_owned=true;
        while(std::getline(commands,line)){
            if(line.empty() || line[0]=='#')continue;
            std::istringstream s(line);std::string kind;s>>kind;
            auto end=[&](){s>>std::ws;require(s.eof() || s.peek()=='#',"trailing chip command fields");};
            if(kind=="code"){
                std::string file;require(bool(s>>std::quoted(file)),"code requires a file");end();
                std::ifstream input(file);require(bool(input),"code image missing");
                code.clear();std::string word;while(input>>word)code.push_back(integer("0x"+word));
                require(!code.empty() && code.size()<=16384,"invalid code image size");
            }else if(kind=="data" || kind=="expect"){
                std::string addr,file;require(bool(s>>addr>>std::quoted(file)),"data/expect require address and file");end();
                u64 address=integer(addr);require(host_owned,"host data access requires drained ownership");
                std::ifstream input(file,std::ios::binary);require(bool(input),"data file missing");char c;u64 count=0;
                while(input.get(c)){x.c.address(address+count,1);auto& l=hbm.memory[(address+count)&~63ULL];unsigned off=unsigned((address+count)&63);
                    if(kind=="data")l[off]=static_cast<unsigned char>(c);
                    else {require(l[off]==static_cast<unsigned char>(c),"chip numerical mismatch at address "+std::to_string(address+count));++checked;}++count;}
                require(count>0,"empty data file");
            }else if(kind=="fence"){end();flush();host_owned=true;}
            else if(kind=="launch" || kind=="launch_roles"){
                u64 id,pc,arg,total;unsigned gx,gy,gz;std::string pcstr,argstr;
                require(bool(s>>id>>pcstr>>argstr>>gx>>gy>>gz),"launch requires id pc arg gx gy gz");pc=integer(pcstr);arg=integer(argstr);
                unsigned roles=1;bool pair_alias=false;
                if(kind=="launch_roles"){
                    std::string role_str,pair_str;require(bool(s>>role_str>>pair_str),"launch_roles requires mask and pair-alias flag");
                    auto mask=integer(role_str),pair=integer(pair_str);require(mask>0 && mask<16 && pair<=1,"invalid role mapping");roles=unsigned(mask);pair_alias=pair;
                }
                end();
                require(gx && gy && gz && u64(gx)*gy<=16777216 && u64(gx)*gy*gz<=16777216,"invalid grid");
                total=u64(gx)*gy*gz;u64 next=0;unsigned live=cores.size();sc_event done;auto start=sc_time_stamp();
                for(auto& core:cores){auto* ptr=core.get();sc_spawn([&,ptr](){
                    while(next<total){u64 pid=next++;
                        ++x.counts["core_"+std::to_string(ptr->id)+"_programs_issued"];
                        ptr->program(code,pid,pc,arg,gx,gy,gz,roles,pair_alias);++x.counts["core_"+std::to_string(ptr->id)+"_programs_completed"];}
                    --live;done.notify(SC_ZERO_TIME);
                });}
                while(live)wait(done);
                x.span("grid completion",300,start,id);++launches;host_owned=false;
            }else throw std::runtime_error("unknown chip script command: "+kind);
        }
        flush();result<<"{\"schema\":\"tnca-systemc-result/1\",\"status\":\"completed\",\"execution\":\"native-isa-event-driven\",\"coverage\":\"partial-ISA\",\"performance_gate\":\"not-measured\",\"calibration\":\"not-measured\",\"visible_cycles\":"<<x.now()<<",\"launches\":"<<launches<<",\"checked_bytes\":"<<checked<<",\"counters\":{";
        bool first=true;for(auto& c:x.counts){if(!first)result<<',';first=false;result<<'"'<<c.first<<"\":"<<c.second;}result<<"}}\n";result.close();x.close();terminal=true;sc_stop();
    }
};
int sc_main(int argc,char** argv){try{
    require(argc==4,"usage: tnca-chip CONFIG COMMANDS OUTPUT_DIRECTORY");sc_set_time_resolution(1,SC_PS);
    Config cfg(argv[1]);Context x(cfg,argv[3]);HBM hbm("hbm",x);Network net("noc",x);std::vector<std::unique_ptr<Slice>> caches;
    for(unsigned i=0;i<cfg.cores();++i){caches.emplace_back(new Slice(("home"+std::to_string(i)).c_str(),x,i));net.homes[i].bind(caches.back()->target);caches.back()->memory.bind(hbm.target);}
    Chip chip("chip",x,hbm,net,caches,argv[2],argv[3]);sc_start();require(chip.terminal,"native chip model deadlocked");return 0;
}catch(const std::exception& e){std::cerr<<"MODEL_REFUSED: "<<e.what()<<'\n';return 2;}}
