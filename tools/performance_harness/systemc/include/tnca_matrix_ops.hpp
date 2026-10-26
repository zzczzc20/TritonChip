#pragma once
// Matrix-engine arithmetic shared by the native executor and the oracle
// checker: ISA sec 9C.1 (one rounding per k, increasing k, exact widened
// products) over the sec 9C.2 flat row-major fragment layout, and the sec 9B
// accumulator row operations, all through tnca_fpu_multi's widen + fma.
#include "tnca_numeric.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace tnca_matrix_ops {
using U32=std::uint32_t;
using Vector=std::array<U32,16>;
using Tile=std::array<std::array<U32,16>,16>;
// mod[3:2] dtype -> tnca_formats code: f16, bf16, e5m2, e4m3.
inline unsigned format_of(unsigned dtype){static const unsigned f[4]={2,1,3,4};return f[dtype&3];}
inline unsigned element_bytes(unsigned dtype){return (dtype&3)>=2?1:2;}
inline unsigned fragment_registers(unsigned rows,unsigned cols,unsigned dtype){return (rows*cols*element_bytes(dtype)+63)/64;}
inline U32 element(const std::vector<Vector>& regs,unsigned flat,unsigned bytes){
    unsigned offset=flat*bytes,reg=offset/64,byte=offset%64;
    if(reg>=regs.size())throw std::runtime_error("MMA fragment element outside its registers");
    return (regs[reg][byte/4]>>((byte%4)*8))&(bytes==2?0xffffU:0xffU);
}
// C[16x16] <- C + A[16xK] . B[Kx16]
inline void mma(Tile& C,const std::vector<Vector>& A,const std::vector<Vector>& B,unsigned K,unsigned dtype){
    unsigned fmt=format_of(dtype),bytes=element_bytes(dtype);
    std::vector<U32> a(16*K),b(K*16);
    for(unsigned e=0;e<16*K;++e)a[e]=tnca_numeric::widen(fmt,element(A,e,bytes));
    for(unsigned e=0;e<K*16;++e)b[e]=tnca_numeric::widen(fmt,element(B,e,bytes));
    for(unsigned m=0;m<16;++m)for(unsigned n=0;n<16;++n){
        U32 v=C[m][n];
        for(unsigned k=0;k<K;++k)v=tnca_numeric::fma32(a[m*K+k],b[k*16+n],v);
        C[m][n]=v;
    }
}
// a.vwrite (func 1), a.mul.row (2), a.fma.row (3) on one accumulator row.
inline void row_op(std::array<U32,16>& row,unsigned func,U32 scale,const Vector& v,unsigned mask){
    for(unsigned lane=0;lane<16;++lane)if(mask&(1U<<lane)){
        if(func==1)row[lane]=v[lane];
        else if(func==2)row[lane]=tnca_numeric::fma32(row[lane],scale,0,false,true,false);
        else if(func==3)row[lane]=tnca_numeric::fma32(row[lane],scale,v[lane]);
        else throw std::runtime_error("reserved accumulator row operation");
    }
}
}
