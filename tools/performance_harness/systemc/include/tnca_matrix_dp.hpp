#pragma once
// m.mma.dp.async arithmetic (ISA sec 9C.4): the K dimension is taken in groups
// of G consecutive k, in increasing order, and for each group
//
//     C <- RNE_f32( C + RO_48( sum over the group of exact(A) * exact(B) ) )
//
// G is 4 for E4M3 x E4M3 and 2 for every other pair. RO_48 rounds the exact
// group sum to 48 significant bits, round to odd, unbounded exponent. The sum
// is formed here in a fixed-width sign-magnitude integer (1024 bits cover the
// widest BF16 x BF16 spread plus the binary32 addend), so no host floating
// point is involved in the result. NaN payloads are canonical (0x7fc00000).
// The independent oracle is tools/model/fdp.py (Fractions); this file is not
// derived from it line by line and the two are compared by validate_matrix_dp.
#include "tnca_numeric.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tnca_matrix_dp {
using U32=std::uint32_t;
using U64=std::uint64_t;

// Unsigned magnitude, little-endian 32-bit limbs.
struct Mag {
    static constexpr int N=32;
    std::array<U32,N> w{};
    bool zero()const{for(U32 x:w)if(x)return false;return true;}
    int bitlen()const{for(int i=N-1;i>=0;--i)if(w[i])return i*32+32-__builtin_clz(w[i]);return 0;}
    bool bit(int i)const{return i>=0 && i<N*32 && ((w[i/32]>>(i%32))&1);}
    void set_bit(int i){w[i/32]|=1U<<(i%32);}
    bool any_below(int n)const{ // any set bit with index < n
        for(int i=0;i<n && i<N*32;){if(i%32==0 && i+32<=n){if(w[i/32])return true;i+=32;}else{if(bit(i))return true;++i;}}
        return false;
    }
    void shl(int s){
        if(s<=0)return;
        int limbs=s/32,bits=s%32;
        for(int i=N-1;i>=0;--i){
            U64 v=0;int src=i-limbs;
            if(src>=0){v=U64(w[src])<<bits;if(bits&&src>0)v|=U64(w[src-1])>>(32-bits);}
            w[i]=U32(v);
        }
    }
    void shr(int s){
        if(s<=0)return;
        int limbs=s/32,bits=s%32;
        for(int i=0;i<N;++i){
            U64 v=0;int src=i+limbs;
            if(src<N){v=U64(w[src])>>bits;if(bits&&src+1<N)v|=U64(w[src+1])<<(32-bits);}
            w[i]=U32(v);
        }
    }
    static int cmp(const Mag& a,const Mag& b){for(int i=N-1;i>=0;--i){if(a.w[i]!=b.w[i])return a.w[i]<b.w[i]?-1:1;}return 0;}
    void add(const Mag& o){U64 c=0;for(int i=0;i<N;++i){U64 s=U64(w[i])+o.w[i]+c;w[i]=U32(s);c=s>>32;}}
    void sub(const Mag& o){U64 b=0;for(int i=0;i<N;++i){U64 d=U64(w[i])-o.w[i]-b;w[i]=U32(d);b=(d>>63)&1;}}
};
// sign * mag * 2^exp
struct Num {bool neg=false;Mag m;int exp=0;};
inline void add_signed(Num& acc,bool neg,const Mag& m){ // acc.exp already <= every operand's exponent
    if(acc.m.zero()){acc.neg=neg;acc.m=m;return;}
    if(acc.neg==neg){acc.m.add(m);return;}
    int c=Mag::cmp(acc.m,m);
    if(c==0){acc.m=Mag{};acc.neg=false;}
    else if(c>0)acc.m.sub(m);
    else{Mag t=m;t.sub(acc.m);acc.m=t;acc.neg=neg;}
}

enum Kind {Fin,Zero,Inf,NaN};
struct Elem {Kind kind=Zero;bool neg=false;U32 mant=0;int exp=0;}; // value = mant * 2^exp
// format code as tnca_numeric: 1 bf16, 2 f16, 3 e5m2, 4 e4m3.
inline Elem decode(unsigned code,U32 x){
    auto f=tnca_numeric::format(code);Elem e;
    unsigned width=1+f.exponent+f.fraction;x&=(width==32)?0xffffffffU:((1U<<width)-1);
    e.neg=(x>>(width-1))&1;
    U32 frac=x&((1U<<f.fraction)-1),ex=(x>>f.fraction)&((1U<<f.exponent)-1),top=(1U<<f.exponent)-1;
    if(tnca_numeric::nan(code,x)){e.kind=NaN;return e;}
    if(f.infinity && ex==top){e.kind=Inf;return e;}
    int bias=(1<<(f.exponent-1))-1;
    if(ex==0){e.mant=frac;e.exp=1-bias-int(f.fraction);}
    else{e.mant=frac|(1U<<f.fraction);e.exp=int(ex)-bias-int(f.fraction);}
    e.kind=e.mant?Fin:Zero;
    return e;
}
inline unsigned group_size(unsigned fmt_a,unsigned fmt_b){return (fmt_a==4 && fmt_b==4)?4:2;}

// binary32 bits -> sign, magnitude, exponent
inline U32 qnan(){return 0x7fc00000U;}
inline U32 inf32(bool neg){return (U32(neg)<<31)|0x7f800000U;}

// One accumulation step: C <- RNE_f32(C + RO_48(sum_k a[k]*b[k])).
inline U32 group_step(unsigned fa,unsigned fb,const U32* a,const U32* b,unsigned g,U32 c){
    bool nan=false,have_inf=false,inf_neg=false,all_neg_zero=true;
    Elem ea[4],eb[4];int emin=0x7fffffff;
    for(unsigned i=0;i<g;++i){
        ea[i]=decode(fa,a[i]);eb[i]=decode(fb,b[i]);
        bool s=ea[i].neg^eb[i].neg;
        if(ea[i].kind==NaN||eb[i].kind==NaN){nan=true;continue;}
        if((ea[i].kind==Inf&&eb[i].kind==Zero)||(eb[i].kind==Inf&&ea[i].kind==Zero)){nan=true;continue;}
        if(ea[i].kind==Inf||eb[i].kind==Inf){if(have_inf&&inf_neg!=s)nan=true;have_inf=true;inf_neg=s;continue;}
        if(ea[i].kind==Fin&&eb[i].kind==Fin){all_neg_zero=false;emin=std::min(emin,ea[i].exp+eb[i].exp);}
        else if(!s)all_neg_zero=false; // a zero product of positive sign
    }
    bool c_nan=tnca_numeric::nan(0,c),c_inf=((c>>23)&255)==255 && !(c&0x7fffffU),c_neg=c>>31;
    if(nan||c_nan)return qnan();
    if(have_inf)return (c_inf&&c_neg!=inf_neg)?qnan():inf32(inf_neg);
    if(c_inf)return c;
    // exact finite group sum
    Num sum;
    if(emin==0x7fffffff)emin=0; // no nonzero product
    sum.exp=emin;
    for(unsigned i=0;i<g;++i){
        if(ea[i].kind!=Fin||eb[i].kind!=Fin)continue;
        Mag p;U64 prod=U64(ea[i].mant)*eb[i].mant;p.w[0]=U32(prod);p.w[1]=U32(prod>>32);
        p.shl(ea[i].exp+eb[i].exp-emin);
        add_signed(sum,ea[i].neg^eb[i].neg,p);
    }
    // RO_48
    int L=sum.m.bitlen();
    if(L>48){
        int drop=L-48;bool sticky=sum.m.any_below(drop);
        sum.m.shr(drop);if(sticky)sum.m.set_bit(0);sum.exp+=drop;
    }
    bool r_zero=sum.m.zero();
    // C as a signed integer at its own exponent
    Num total;
    unsigned cexp=(c>>23)&255;U32 cman=(c&0x7fffffU)|(cexp?0x800000U:0);int cx=(cexp?int(cexp):1)-150;
    bool c_zero=cman==0;
    if(r_zero&&c_zero)return U32(all_neg_zero&&c_neg)<<31; // IEEE sum of zeros
    int e0=r_zero?cx:(c_zero?sum.exp:std::min(cx,sum.exp));
    total.exp=e0;
    if(!r_zero){Mag t=sum.m;t.shl(sum.exp-e0);add_signed(total,sum.neg,t);}
    if(!c_zero){Mag t;t.w[0]=cman;t.shl(cx-e0);add_signed(total,c_neg,t);}
    if(total.m.zero())return 0; // exact cancellation: +0 under RNE
    // round to binary32: RNE, subnormals kept, overflow to infinity
    int L2=total.m.bitlen(),E=e0+L2-1;
    int lsb=std::max(E-23,-149),drop=lsb-e0;
    Mag q=total.m;
    if(drop>0){
        bool half=q.bit(drop-1),below=q.any_below(drop-1);
        q.shr(drop);
        if(half&&(below||q.bit(0))){Mag one;one.w[0]=1;q.add(one);}
    }else q.shl(-drop);
    U64 mant=q.w[0]|(U64(q.w[1])<<32);
    int exp=lsb;
    if(mant>>24){mant>>=1;++exp;}              // carry out of the 24-bit field (mant was 2^24 exactly)
    int top=exp+23;                            // exponent of the leading bit if mant has 24 bits
    if(mant==0)return U32(total.neg)<<31;
    if(mant>>23){ // normal
        if(top>127)return inf32(total.neg);
        return (U32(total.neg)<<31)|(U32(top+127)<<23)|U32(mant&0x7fffffU);
    }
    return (U32(total.neg)<<31)|U32(mant);     // subnormal (exp == -149)
}
}
