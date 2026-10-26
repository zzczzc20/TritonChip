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
// binary32 -> 32-bit integer exactly as rtl/core/tnca_vector_fpu.v computes
// it, including its fixed-width intermediate arithmetic: `csh` is an 8-bit
// register, so for exponents 151..157 (|x| in [2^24, 2^31)) the shift
// amount wraps, and `64'd1 << (csh - 1)` is evaluated at 32 bits. Those
// corners differ from ISA sec 8.4 (saturating RNE/RTZ conversion); the RTL
// is the authority for what a TNCA device computes, so the model follows it
// and validate_numeric records each divergence from the ISA text.
inline U32 float_to_integer(U32 raw,bool is_signed,unsigned rounding){
    if(rounding>3)throw std::runtime_error("reserved integer rounding mode");
    unsigned cexp=(raw>>23)&255;bool cneg=raw>>31;U64 cman=(1U<<23)|(raw&0x7fffffU);
    if(cexp==255 && (raw&0x7fffffU))return 0;
    if(cexp<127){
        switch(rounding){
            case 1:return (cexp==126 && (raw&0x7fffffU))?(cneg?0xffffffffU:1U):0U;
            case 2:return (cneg && (raw&0x7fffffffU))?0xffffffffU:0U;
            case 3:return (!cneg && (raw&0x7fffffffU))?1U:0U;
            default:return 0;
        }
    }
    if(cexp>=158)return is_signed?(cneg?0x80000000U:0x7fffffffU):(cneg?0U:0xffffffffU);
    unsigned csh=(150U-cexp)&255U;
    U64 cmag=csh>=64?0:(cman>>csh);
    U64 cfrac_mask=(csh>=64?0:(1ULL<<csh))-1;
    U64 cfrac=cman&cfrac_mask;
    std::uint32_t half_shift=std::uint32_t(csh)-1U;
    U64 chalf=half_shift>=64?0:(1ULL<<half_shift);
    bool inc=false;
    switch(rounding){
        case 1:inc=cfrac>chalf || (cfrac==chalf && (cmag&1));break;
        case 2:inc=cneg && cfrac!=0;break;
        case 3:inc=!cneg && cfrac!=0;break;
        default:break;
    }
    cmag+=inc;
    std::int64_t fixv=cneg?-std::int64_t(cmag):std::int64_t(cmag);
    if(is_signed)return fixv>2147483647LL?0x7fffffffU:fixv<-2147483648LL?0x80000000U:U32(fixv);
    return cneg?0U:(cmag>0xffffffffULL?0xffffffffU:U32(cmag));
}
// tnca_fp32_fma: one rounding (RNE), subnormals kept unless ftz, every NaN
// result the canonical quiet NaN. vf.add/sub are fma(a,1,+-b), vf.mul is
// mul_only (tnca_vector_fpu.v header).
inline U32 flush(U32 x){return (x&0x7f800000U)==0?(x&0x80000000U):x;}
inline U32 fma32(U32 a,U32 b,U32 c,bool sub_c=false,bool mul_only=false,bool ftz=false){
    if(sub_c)c^=0x80000000U;
    if(ftz){a=flush(a);b=flush(b);c=flush(c);}
    float r=mul_only?floating(a)*floating(b):std::fma(floating(a),floating(b),floating(c));
    U32 y=bits(r);
    if(nan(0,y))return 0x7fc00000U;
    return ftz?flush(y):y;
}
// The RTL's ordered comparison on the monotone integer image (+0 == -0).
inline bool lt32(U32 x,U32 y){
    U32 mx=(x>>31)?(0x80000000U-(x&0x7fffffffU)):(0x80000000U+(x&0x7fffffffU));
    U32 my=(y>>31)?(0x80000000U-(y&0x7fffffffU)):(0x80000000U+(y&0x7fffffffU));
    return mx<my;
}
inline bool is_zero32(U32 x){return (x&0x7fffffffU)==0;}
// vf.min/vf.max (ISA sec 8.3) as tnca_vector_fpu.v selects them.
inline U32 minmax_rtl(U32 a,U32 b,bool is_max,unsigned mode){
    if(mode>2)throw std::runtime_error("reserved FP min/max mode");
    bool xn=nan(0,a),yn=nan(0,b),ltv=lt32(a,b),pick_a=is_max?!ltv:ltv;
    if(mode==1 && (xn || yn))return 0x7fc00000U;
    if(xn && yn)return 0x7fc00000U;
    if(xn)return b;
    if(yn)return a;
    if(is_zero32(a) && is_zero32(b) && mode!=2)return is_max?((a>>31)==0?a:b):((a>>31)==1?a:b);
    return pick_a?a:b;
}
// vf.cmp condition codes (ISA sec 4) as the RTL evaluates them.
inline bool fcmp_rtl(U32 a,U32 b,unsigned cc){
    bool un=nan(0,a) || nan(0,b),eq=!un && (a==b || (is_zero32(a) && is_zero32(b))),lt=lt32(a,b);
    switch(cc){case 0:return eq;case 1:return un || !eq;case 2:return !un && lt && !eq;case 3:return !un && (lt || eq);
        case 4:return !un && !lt && !eq;case 5:return !un && (!lt || eq);case 10:return !un;case 11:return un;
        default:throw std::runtime_error("reserved floating comparison");}
}
// One level of the vf.reduce.max/min fold (tnca_vector_fpu.v red_step): a
// NaN operand yields the other operand unchanged, ties keep the left one.
inline U32 fold_minmax(U32 left,U32 right,bool is_max){
    if(nan(0,left))return right;
    if(nan(0,right))return left;
    bool l=lt32(left,right);
    return is_max?(l?right:left):(l?left:right);
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
