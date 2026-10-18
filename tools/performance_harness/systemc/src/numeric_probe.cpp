// Standalone oracle-vector checker for the native semantic helpers. It is not
// a simulator backend and never contributes workload timing predictions.
#include "../include/tnca_numeric.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: tnca-numeric-probe formats|mul|ints|sfu VECTORS");
        std::ifstream input(argv[2]);if(!input)throw std::runtime_error("vector file missing");
        std::string mode=argv[1],line;std::uint64_t checked=0,failed=0;
        while(std::getline(input,line)){
            if(line.empty() || line[0]=='#')continue;
            std::istringstream row(line);bool equal=false;
            if(mode=="formats"){
                unsigned src,dst;std::uint32_t value,want;
                if(!(row>>src>>dst>>std::hex>>value>>want))throw std::runtime_error("bad format vector");
                auto got=tnca_numeric::convert(src,dst,value);
                equal=got==want || (tnca_numeric::nan(dst,got) && tnca_numeric::nan(dst,want));
            }else if(mode=="mul"){
                std::uint64_t a,b,unsigned_want,signed_want;
                if(!(row>>std::hex>>a>>b>>unsigned_want>>signed_want))throw std::runtime_error("bad multiply vector");
                equal=tnca_numeric::multiply_high(a,b,false)==unsigned_want && tnca_numeric::multiply_high(a,b,true)==signed_want;
            }else if(mode=="ints"){
                unsigned sign,rounding;std::uint32_t value,want;
                if(!(row>>sign>>rounding>>std::hex>>value>>want))throw std::runtime_error("bad integer conversion vector");
                equal=tnca_numeric::float_to_integer(value,sign,rounding)==want;
            }else if(mode=="sfu"){
                unsigned op,is_nan;std::uint32_t value,low,high;
                if(!(row>>op>>std::hex>>value>>low>>high>>std::dec>>is_nan) || op>3 || is_nan>1)throw std::runtime_error("bad SFU vector");
                const unsigned functions[4]={0,3,1,2};auto got=tnca_numeric::sfu(functions[op],value);
                equal=is_nan?tnca_numeric::nan(0,got):low==high?got==low:
                    !tnca_numeric::nan(0,got) && tnca_numeric::floating(got)>=tnca_numeric::floating(low) && tnca_numeric::floating(got)<=tnca_numeric::floating(high);
            }else throw std::runtime_error("unknown probe mode");
            ++checked;if(!equal){if(failed++<12)std::cerr<<"MISMATCH "<<line<<'\n';}
        }
        if(!checked)throw std::runtime_error("empty oracle vector file");
        std::cout<<"{\"checked\":"<<checked<<",\"mismatches\":"<<failed<<"}\n";return failed?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
