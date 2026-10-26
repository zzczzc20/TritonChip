#pragma once
// Bit-exact port of xls/tnca_sfu.x, the DSLX source of the RTL SFU pipelines
// (rtl/gen/tnca_sfu_*_xls.v). Fixed-width DSLX arithmetic wraps; every
// operation below reproduces that with unsigned intermediates, so huge or
// tiny arguments take exactly the RTL's (not the mathematical) result.
#include "tnca_sfu_tables_generated.hpp"
#include <cstdint>
#include <stdexcept>

namespace tnca_sfu {
using U32=std::uint32_t;using U64=std::uint64_t;using S32=std::int32_t;using S64=std::int64_t;
namespace T=tnca_sfu_tables;
constexpr unsigned FRAC=30;
constexpr S64 ONE=S64(1)<<FRAC;
constexpr U32 QNAN=0x7fc00000U;
inline S32 s32(S64 v){return S32(U32(U64(v)));}
inline S32 add32(S32 a,S32 b){return S32(U32(a)+U32(b));}
inline S64 sra(S64 v,unsigned n){return v>>n;}                 // arithmetic (DSLX signed >>)
inline S64 shl(S64 v,unsigned n){return S64(U64(v)<<n);}       // wrapping
inline S32 poly2(S32 c0,S32 c1,S32 c2,S32 u){
    S32 t=add32(c1,s32(sra(S64(c2)*S64(u),FRAC)));
    return add32(c0,s32(sra(S64(t)*S64(u),FRAC)));
}
inline U32 pack(U32 sign,S32 unb,S64 mant){
    S64 round_add=S64(U64(mant)+(U64(1)<<6));
    bool carried=round_add>=(S64(1)<<31);
    S64 m=carried?sra(round_add,1):round_add;
    S32 e=carried?add32(unb,1):unb;
    U32 frac=U32(U64(sra(m,7))&0x7fffffU);
    S32 bexp=add32(e,127);
    if(bexp>=255)return (sign<<31)|0x7f800000U;
    if(bexp<=0)return sign<<31;
    return (sign<<31)|(U32(bexp&255)<<23)|frac;
}
inline U32 fixed_to_f32(S64 v){
    if(v==0)return 0;
    bool neg=v<0;U64 mag=neg?U64(0)-U64(v):U64(v);
    U64 msb=63-U64(__builtin_clzll(mag));
    S32 unb=S32(msb)-S32(FRAC);
    U64 mant=msb<30?(mag<<(30-msb)):(mag>>(msb-30));
    return pack(neg,unb,S64(mant));
}
struct Parts{U32 sign,bexp,frac;};
inline Parts parts(U32 x){return {x>>31,(x>>23)&255,x&0x7fffffU};}
inline bool is_nan(Parts p){return p.bexp==255 && p.frac;}
inline bool is_inf(Parts p){return p.bexp==255 && !p.frac;}
inline bool zeroish(Parts p){return p.bexp==0;}
inline void seg(U32 frac,U32& idx,S32& u){
    idx=frac>>(23-T::LOG2_SEGMENTS);
    S32 f30=S32(frac<<(FRAC-23));
    u=add32(f30,-S32(idx<<(FRAC-T::LOG2_SEGMENTS)));
}
inline U32 exp2(U32 x){
    Parts p=parts(x);S32 unb=S32(p.bexp)-127;S64 sig=(S64(1)<<23)|S64(p.frac);
    S32 sh=unb-23+S32(FRAC);
    S64 mag=sh>=0?(sh>40?(S64(1)<<62):shl(sig,unsigned(sh))):(sh<-40?0:sra(sig,unsigned(-sh)));
    S64 v=p.sign?S64(U64(0)-U64(mag)):mag;
    S32 i=s32(sra(v,FRAC));
    S64 r=S64(U64(v)-U64(shl(S64(i),FRAC)));
    U32 idx=U32(U64(sra(r,FRAC-T::LOG2_SEGMENTS)));
    S32 u=s32(S64(U64(r)-U64(shl(S64(idx),FRAC-T::LOG2_SEGMENTS))));
    if(is_nan(p))return QNAN;
    if(is_inf(p))return p.sign?0U:0x7f800000U;
    if(zeroish(p))return 0x3f800000U;
    if(i>=128)return 0x7f800000U;
    if(i<-150)return 0;
    if(idx>=T::SEGMENTS)idx=T::SEGMENTS-1; // XLS array_index clamps an out-of-bounds index
    S64 m=poly2(T::EXP2_C0[idx],T::EXP2_C1[idx],T::EXP2_C2[idx],u);
    return pack(0,i,m);
}
inline U32 log2(U32 x){
    Parts p=parts(x);S32 unb=S32(p.bexp)-127;U32 idx;S32 u;seg(p.frac,idx,u);
    S64 lg=poly2(T::LOG2_C0[idx],T::LOG2_C1[idx],T::LOG2_C2[idx],u);
    S64 total=S64(U64(shl(S64(unb),FRAC))+U64(lg));
    if(is_nan(p))return QNAN;
    if(p.sign && !zeroish(p))return QNAN;
    if(zeroish(p))return 0xff800000U;
    if(is_inf(p))return p.sign?QNAN:0x7f800000U;
    return fixed_to_f32(total);
}
inline U32 rcp(U32 x){
    Parts p=parts(x);S32 unb=S32(p.bexp)-127;U32 idx;S32 u;seg(p.frac,idx,u);
    S64 v=poly2(T::RCP_C0[idx],T::RCP_C1[idx],T::RCP_C2[idx],u);
    bool low=v<ONE;S64 mant=low?shl(v,1):v;S32 e=low?-unb-1:-unb;
    if(is_nan(p))return QNAN;
    if(zeroish(p))return (p.sign<<31)|0x7f800000U;
    if(is_inf(p))return p.sign<<31;
    return pack(p.sign,e,mant);
}
inline U32 rsqrt(U32 x){
    Parts p=parts(x);S32 unb=S32(p.bexp)-127;U32 idx;S32 u;seg(p.frac,idx,u);
    S32 h=unb>>1;bool odd=(unb&1)!=0;
    S64 v=odd?poly2(T::RSQRT2_C0[idx],T::RSQRT2_C1[idx],T::RSQRT2_C2[idx],u)
             :poly2(T::RSQRT_C0[idx],T::RSQRT_C1[idx],T::RSQRT_C2[idx],u);
    bool low=v<ONE;S64 mant=low?shl(v,1):v;S32 e=low?-h-1:-h;
    if(is_nan(p))return QNAN;
    if(zeroish(p))return (p.sign<<31)|0x7f800000U;
    if(p.sign)return QNAN;
    if(is_inf(p))return 0;
    return pack(0,e,mant);
}
// ISA sec 8.8 function codes: 0 exp2, 1 rcp, 2 rsqrt, 3 log2.
inline U32 evaluate(unsigned function,U32 x){
    switch(function){case 0:return exp2(x);case 1:return rcp(x);case 2:return rsqrt(x);case 3:return log2(x);
        default:throw std::runtime_error("reserved SFU function");}
}
}
