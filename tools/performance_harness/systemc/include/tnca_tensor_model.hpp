#pragma once
#include "tnca_model.hpp"

// Immutable command semantics, separate from the descriptor slot's lifetime.
// Coordinates and strides carry their device's 64-bit two's-complement bits;
// address arithmetic is unsigned modulo 2^64, never host signed-overflow UB.
struct TensorMapModel {
    std::array<unsigned char,256> raw{};
    u64 base=0,uid=0,version=0;
    std::array<u64,5> extent{},stride{};
    std::array<unsigned,5> block{};
    unsigned rank=0,dtype=0,padding=0,flags=0;
    static u64 little(const std::array<unsigned char,256>& data,unsigned at,unsigned n){
        u64 v=0;for(unsigned b=0;b<n;++b)v|=u64(data[at+b])<<(b*8);return v;
    }
    explicit TensorMapModel(const std::array<unsigned char,256>& data):raw(data){
        base=little(raw,0,8);uid=little(raw,112,8);version=little(raw,120,8);
        for(unsigned d=0;d<5;++d){extent[d]=little(raw,8+8*d,8);stride[d]=little(raw,48+8*d,8);block[d]=unsigned(little(raw,88+4*d,4));}
        rank=raw[108];dtype=raw[109];padding=raw[110];flags=raw[111];
    }
    unsigned elements(unsigned bytes,unsigned window_registers)const{
        arch(rank>=1 && rank<=5,4,"DESCRIPTOR_FAULT: tensor rank");
        require(flags==0,"unimplemented TensorMap semantic flags");
        arch(dtype<=9,4,"DESCRIPTOR_FAULT: reserved dtype");
        arch(padding==0 || padding==2 || padding==3,4,"DESCRIPTOR_FAULT: reserved padding mode");
        arch(bytes==1 || bytes==2 || bytes==4 || bytes==8,1,"ILLEGAL_INSTRUCTION: transfer element size");
        u64 n=1;
        for(unsigned d=0;d<rank;++d){
            arch(block[d]>=1 && block[d]<=256,4,"DESCRIPTOR_FAULT: block dimension");
            arch(n<=u64(window_registers)*64/bytes/block[d],4,"DESCRIPTOR_FAULT: box exceeds window");n*=block[d];
        }
        arch((u64(block[rank-1])*bytes)%8==0 && stride[rank-1]==bytes,4,"DESCRIPTOR_FAULT: noncontiguous or partial-unit row");
        if(padding==3){fill(bytes);}
        return unsigned(n);
    }
    u64 fill(unsigned bytes)const{
        if(padding!=3)return 0;
        switch(dtype){case 0:arch(bytes==2,4,"DESCRIPTOR_FAULT: padding dtype/size");return 0x7e00;
            case 1:arch(bytes==2,4,"DESCRIPTOR_FAULT: padding dtype/size");return 0x7fc0;
            case 2:arch(bytes==4,4,"DESCRIPTOR_FAULT: padding dtype/size");return 0x7fc00000;
            case 7:case 8:arch(bytes==1,4,"DESCRIPTOR_FAULT: padding dtype/size");return 0x7f;
            case 9:arch(bytes==8,4,"DESCRIPTOR_FAULT: padding dtype/size");return 0x7ff8000000000000ULL;
            default:throw ArchFault(4,"DESCRIPTOR_FAULT: NaN padding for integer dtype");}
    }
    bool inside(const std::array<u64,5>& origin)const{
        for(unsigned d=0;d<rank;++d)if((origin[d]>>63) || origin[d]+block[d]>extent[d])return false;
        return true;
    }
    struct Byte {bool valid;u64 address;unsigned char padding;};
    Byte at(unsigned flat,unsigned bytes,const std::array<u64,5>& origin)const{
        unsigned element=flat/bytes,byte=flat%bytes;u64 address=base;bool valid=true;
        for(unsigned reverse=0;reverse<rank;++reverse){unsigned d=rank-1-reverse,index=element%block[d];element/=block[d];
            u64 coordinate=origin[d]+index;
            if((coordinate>>63) || coordinate>=extent[d])valid=false;
            address+=coordinate*stride[d];
        }
        return {valid,address+byte,static_cast<unsigned char>(fill(bytes)>>(byte*8))};
    }
    void check_transpose(unsigned bytes,bool store,const std::array<u64,5>& origin)const{
        arch(!store && rank==2 && bytes==2,4,"DESCRIPTOR_FAULT: transposed transfer kind/rank/size");
        require((block[0]==4 || block[0]==8 || block[0]==16) && base%8==0 && stride[0]%8==0 &&
                !(origin[1]>>63) && (origin[1]*bytes)%8==0,"DESCRIPTOR_FAULT: transposed transfer geometry");
    }
};
