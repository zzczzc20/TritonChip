#pragma once
// Per-lane vector FP, conversion and reduction semantics shared by the native
// executor (src/chip.cpp) and the oracle checker (src/numeric_probe.cpp), so
// the code validated against RTL-recorded vectors is the code that executes.
#include "tnca_numeric.hpp"
#include "tnca_sfu.hpp"
#include <algorithm>
#include <array>

namespace tnca_vector_ops {
using U32=std::uint32_t;
constexpr U32 ONE=0x3f800000U;
// major 0x11 lane result for every function except vf.cmp.
inline U32 vfp(unsigned func,unsigned mod,U32 a,U32 b,U32 c){
    bool ftz=mod&0x100;
    switch(func){
        case 1:return tnca_numeric::fma32(a,ONE,b,false,false,ftz);
        case 2:return tnca_numeric::fma32(a,ONE,b,true,false,ftz);
        case 3:return tnca_numeric::fma32(a,b,0,false,true,ftz);
        case 6:case 7:return tnca_numeric::fma32(a,b,c,func==7,false,ftz);
        case 16:case 17:return tnca_numeric::minmax_rtl(a,b,func==17,mod&3);
        case 20:return a^0x80000000U;
        case 21:return a&0x7fffffffU;
        default:throw std::runtime_error("uncovered floating function");
    }
}
inline bool vfp_compare(unsigned mod,U32 a,U32 b){return tnca_numeric::fcmp_rtl(a,b,mod&15);}
// major 0x12 lane result.
inline U32 cvt(unsigned func,unsigned mod,U32 a){
    if(func==0)return tnca_numeric::bits(float(std::int32_t(a)));
    if(func==1)return tnca_numeric::bits(float(a));
    if(func==2 || func==3)return tnca_numeric::float_to_integer(a,func==2,mod&7);
    if(func==4)return tnca_numeric::convert(mod&7,(mod>>3)&7,a);
    throw std::runtime_error("reserved conversion function");
}
// major 0x15: identity for inactive lanes, then the balanced tree over lane
// index (fixed for vf.reduce.add by ISA sec 8.7; the RTL fold for max/min).
inline U32 reduce(unsigned func,const std::array<U32,16>& lanes,unsigned mask){
    std::array<U32,16> v{};
    U32 identity=func==1?0x80000000U:func==2?0x7fffffffU:(func==3 || func==7)?0xffffffffU:func==9?0xff800000U:func==10?0x7f800000U:0;
    for(unsigned lane=0;lane<16;++lane)v[lane]=(mask&(1U<<lane))?lanes[lane]:identity;
    for(unsigned width=16;width>1;width/=2)for(unsigned i=0;i<width/2;++i){
        U32 l=v[i*2],r=v[i*2+1],out=0;
        switch(func){
            case 0:out=l+r;break;
            case 1:out=std::int32_t(l)>std::int32_t(r)?l:r;break;
            case 2:out=std::int32_t(l)<std::int32_t(r)?l:r;break;
            case 3:out=l&r;break;case 4:out=l|r;break;case 5:out=l^r;break;
            case 6:out=std::max(l,r);break;case 7:out=std::min(l,r);break;
            case 8:out=tnca_numeric::fma32(l,ONE,r);break;
            case 9:case 10:out=tnca_numeric::fold_minmax(l,r,func==9);break;
            default:throw std::runtime_error("reserved reduction function");
        }
        v[i]=out;
    }
    return v[0];
}
}
