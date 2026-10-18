#pragma once
// Host-independent integer rounding for TNCA interchange formats. No RTL or
// host half-precision extension is used. NaN payload is not architectural.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tnca_numeric {
#ifdef __FAST_MATH__
#error "TNCA numeric semantics require IEEE arithmetic without fast-math"
#endif
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
              "binary32 host floating-point required");
using U32=std::uint32_t;
using U64=std::uint64_t;
inline float floating(U32 x){float v;std::memcpy(&v,&x,4);return v;}
inline U32 bits(float x){U32 v;std::memcpy(&v,&x,4);return v;}
struct Format {unsigned exponent,fraction;bool infinity;};
inline Format format(unsigned code){
    switch(code){case 0:return {8,23,true};case 1:return {8,7,true};
        case 2:return {5,10,true};case 3:return {5,2,true};case 4:return {4,3,false};
        default:throw std::runtime_error("reserved float format");}
}
inline bool nan(unsigned code,U32 x){
    auto f=format(code);U32 e=(1U<<f.exponent)-1,m=(1U<<f.fraction)-1;
    return ((x>>f.fraction)&e)==e && (f.infinity?(x&m)!=0:(x&m)==m);
}
inline U32 widen(unsigned code,U32 x){
    auto f=format(code);
    if(code==0)return x;
    if(code==1)return (x&65535U)<<16;
    U32 sign=(x>>(f.exponent+f.fraction))&1,frac=x&((1U<<f.fraction)-1);
    U32 exp=(x>>f.fraction)&((1U<<f.exponent)-1),top=(1U<<f.exponent)-1;
    if(nan(code,x))return 0x7fc00000U;
    if(f.infinity && exp==top)return (sign<<31)|0x7f800000U;
    if(exp==0 && frac==0)return sign<<31;
    int bias=(1<<(f.exponent-1))-1;
    float value=std::ldexp(float(exp?frac+(1U<<f.fraction):frac),
                         int(exp?exp:1)-bias-int(f.fraction));
    return bits(sign?-value:value);
}
inline U64 round_shift(U64 value,int shift){
    if(shift<=0)return value<<unsigned(-shift);
    if(shift>=64)return 0;
    U64 q=value>>shift,remainder=value&((1ULL<<shift)-1),half=1ULL<<(shift-1);
    return q+unsigned(remainder>half || (remainder==half && (q&1)));
}
inline U32 narrow(unsigned code,U32 value){
    auto f=format(code);if(code==0)return value;
    unsigned top=(1U<<f.exponent)-1,frac_mask=(1U<<f.fraction)-1;
    U32 sign=(value>>31)<<(f.exponent+f.fraction),source_exp=(value>>23)&255,mant=value&0x7fffffU;
    U32 largest=((top-unsigned(f.infinity))<<f.fraction)|(frac_mask-unsigned(!f.infinity));
    if(source_exp==255){
        if(mant)return (top<<f.fraction)|frac_mask;
        return sign|(f.infinity?top<<f.fraction:largest);
    }
    if(source_exp==0 && mant==0)return sign;
    int source_power=(source_exp?int(source_exp)-127:-126)-23;
    if(source_exp)mant|=1U<<23;
    int leading=31-__builtin_clz(mant),e=source_power+leading;
    int bias=(1<<(f.exponent-1))-1,emin=1-bias,emax=int(top)-int(f.infinity)-bias;
    int quantization=std::max(e,emin)-int(f.fraction);
    U64 q=round_shift(mant,quantization-source_power);
    if(e<emin)return sign|U32(q); // Carry naturally reaches the first normal.
    if(q>=(1ULL<<(f.fraction+1))){q>>=1;++e;}
    if(e>emax || (e==emax && !f.infinity && q>((1U<<f.fraction)+frac_mask-1)))
        return sign|(f.infinity?top<<f.fraction:largest);
    return sign|(U32(e+bias)<<f.fraction)|(U32(q)&frac_mask);
}
inline U32 convert(unsigned src,unsigned dst,U32 x){return narrow(dst,widen(src,x));}
inline U64 multiply_high(U64 a,U64 b,bool signed_product){
    U64 a0=U32(a),a1=a>>32,b0=U32(b),b1=b>>32;
    U64 t=a0*b0,t1=a1*b0+(t>>32),t2=a0*b1+U32(t1);
    U64 high=a1*b1+(t1>>32)+(t2>>32);
    if(signed_product){if(a>>63)high-=b;if(b>>63)high-=a;}
    return high;
}
inline U32 float_to_integer(U32 raw,bool is_signed,unsigned rounding){
    if(rounding>3)throw std::runtime_error("reserved integer rounding mode");
    double value=floating(raw);if(std::isnan(value))return 0;
    double rounded=0;
    if(rounding==0)rounded=std::trunc(value);
    else if(rounding==2)rounded=std::floor(value);
    else if(rounding==3)rounded=std::ceil(value);
    else {
        double floor=std::floor(value),fraction=value-floor;
        rounded=floor+(fraction>.5 || (fraction==.5 && std::fmod(std::fabs(floor),2.0)==1));
        if(std::isinf(value))rounded=value;
    }
    if(is_signed){if(rounded<=-2147483648.0)return 0x80000000U;if(rounded>=2147483647.0)return 0x7fffffffU;return U32(std::int32_t(rounded));}
    if(rounded<=0)return 0;
    if(rounded>=4294967295.0)return 0xffffffffU;
    return U32(rounded);
}
inline U32 minmax(U32 a,U32 b,bool maximum,unsigned mode=0){
    if(mode>2)throw std::runtime_error("reserved FP min/max mode");
    bool na=nan(0,a),nb=nan(0,b);
    if((na && nb) || (mode==1 && (na || nb)))return 0x7fc00000U;
    if(na)return b;
    if(nb)return a;
    if((a&0x7fffffffU)==0 && (b&0x7fffffffU)==0)return maximum?(a&b):(a|b);
    return maximum?(floating(a)>floating(b)?a:b):(floating(a)<floating(b)?a:b);
}
inline U32 sfu(unsigned function,U32 raw){
    if(function>3)throw std::runtime_error("reserved SFU function");
    if((raw&0x7f800000U)==0)raw&=0x80000000U; // Input FTZ, preserving sign.
    float input=floating(raw),value=0;
    if(function==0)value=std::exp2(input);
    else if(function==1)value=1.0f/input;
    else if(function==2)value=1.0f/std::sqrt(input);
    else value=std::log2(input);
    U32 result=bits(value);
    if(nan(0,result))return 0x7fc00000U;
    if((result&0x7f800000U)==0)result&=0x80000000U; // Result FTZ.
    return result;
}
}
