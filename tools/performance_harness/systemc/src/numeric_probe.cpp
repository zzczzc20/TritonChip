// Standalone oracle-vector checker for the native semantic helpers. It is not
// a simulator backend and never contributes workload timing predictions.
#include "../include/tnca_numeric.hpp"
#include "../include/tnca_sfu.hpp"
#include "../include/tnca_vector_ops.hpp"
#include "../include/tnca_matrix_ops.hpp"
#include "../include/tnca_matrix_dp.hpp"
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: tnca-numeric-probe formats|mul|ints|sfu|vfpu VECTORS | matrix ORACLE_DIR");
        if(std::string(argv[1])=="matrix"){
            // Replay rtl_oracle/matrix_oracle.py's command stream through the
            // executor's matrix arithmetic and compare every accumulator word.
            std::string dir=argv[2];std::ifstream stream(dir+"/stream.txt"),results(dir+"/results.txt");
            if(!stream || !results)throw std::runtime_error("matrix oracle files missing");
            std::vector<tnca_matrix_ops::Vector> vrf(256);std::array<tnca_matrix_ops::Tile,16> acc{};
            std::uint64_t words=0,bad=0,commands=0;std::string line,rline;std::map<std::string,std::vector<unsigned>> rtl_cycles;
            while(std::getline(stream,line)){
                std::istringstream row(line);std::string kind;row>>kind;
                if(kind=="V"){unsigned idx;row>>std::hex>>idx;for(auto& w:vrf.at(idx))row>>w;continue;}
                if(!std::getline(results,rline))throw std::runtime_error("matrix results shorter than stream");
                std::istringstream res(rline);std::string rkind;res>>rkind;
                if(kind=="C"){
                    unsigned op,slot,va,vb,k,dt,rowop,r,scale,mask,unused;tnca_matrix_ops::Vector v{};
                    row>>std::hex>>op>>slot>>va>>vb>>k>>dt>>rowop>>r>>scale>>mask>>unused;for(auto& w:v)row>>w;
                    unsigned cycles;if(rkind!="C" || !(res>>cycles))throw std::runtime_error("matrix result/command mismatch");
                    ++commands;
                    if(op==0){acc.at(slot)={};rtl_cycles["zero"].push_back(cycles);}
                    else if(op==1){
                        unsigned K=16U<<k;
                        std::vector<tnca_matrix_ops::Vector> A(tnca_matrix_ops::fragment_registers(16,K,dt)),B(tnca_matrix_ops::fragment_registers(K,16,dt));
                        for(unsigned i=0;i<A.size();++i)A[i]=vrf.at(va+i);
                        for(unsigned i=0;i<B.size();++i)B[i]=vrf.at(vb+i);
                        tnca_matrix_ops::mma(acc.at(slot),A,B,K,dt);
                        rtl_cycles["mma.k"+std::to_string(K)+".dt"+std::to_string(dt)].push_back(cycles);
                    }else{tnca_matrix_ops::row_op(acc.at(slot)[r],rowop+1,scale,v,mask);rtl_cycles["row"+std::to_string(rowop)].push_back(cycles);}
                }else if(kind=="R"){
                    unsigned slot,got_slot;row>>std::hex>>slot;res>>std::dec>>got_slot;
                    if(rkind!="R" || got_slot!=slot)throw std::runtime_error("matrix dump mismatch");
                    for(unsigned m=0;m<16;++m)for(unsigned n=0;n<16;++n){
                        std::uint32_t want;res>>std::hex>>want;++words;
                        if(want!=acc[slot][m][n]){if(bad++<20)std::cerr<<"ACC slot "<<slot<<" ["<<m<<"]["<<n<<"] model "<<std::hex<<acc[slot][m][n]<<" rtl "<<want<<std::dec<<'\n';}
                    }
                }else throw std::runtime_error("unknown matrix stream line");
            }
            std::cout<<"{\"checked\":"<<words<<",\"mismatches\":"<<bad<<",\"commands\":"<<commands<<",\"rtl_busy_cycles\":{";
            bool first=true;for(auto& kv:rtl_cycles){if(!first)std::cout<<',';first=false;std::cout<<'"'<<kv.first<<"\":[";
                bool f2=true;for(auto c:kv.second){if(!f2)std::cout<<',';f2=false;std::cout<<c;}std::cout<<']';}
            std::cout<<"}}\n";return bad?1:0;
        }
        std::ifstream input(argv[2]);if(!input)throw std::runtime_error("vector file missing");
        std::string mode=argv[1],line;std::uint64_t checked=0,failed=0;
        std::map<std::string,std::array<std::uint64_t,2>> per_op;
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
                const unsigned functions[4]={0,3,1,2};auto got=tnca_sfu::evaluate(functions[op],value);
                equal=is_nan?tnca_numeric::nan(0,got):low==high?got==low:
                    !tnca_numeric::nan(0,got) && tnca_numeric::floating(got)>=tnca_numeric::floating(low) && tnca_numeric::floating(got)<=tnca_numeric::floating(high);
            }else if(mode=="matrix_dp"){
                // fa fb g | a[g] | b[g] | c want (m.mma.dp.async, ISA sec 9C.4, one group step)
                unsigned fa,fb,g;std::uint32_t a[4]{},b[4]{},c,want;
                if(!(row>>std::dec>>fa>>fb>>g) || g>4 || g<2)throw std::runtime_error("bad matrix_dp vector");
                row>>std::hex;for(unsigned i=0;i<g;++i)row>>a[i];for(unsigned i=0;i<g;++i)row>>b[i];
                if(!(row>>c>>want))throw std::runtime_error("bad matrix_dp vector");
                if(tnca_matrix_dp::group_size(fa,fb)!=g)throw std::runtime_error("matrix_dp group size disagrees with sec 9C.4");
                equal=tnca_matrix_dp::group_step(fa,fb,a,b,g,c)==want;
                ++per_op[std::to_string(fa)+"x"+std::to_string(fb)][equal];
            }else if(mode=="vfpu"){
                // RTL-recorded tnca_vector_fpu outputs (rtl_oracle/rtl_oracle.py):
                // stimulus | 16 result lanes, predicate bits, scalar result.
                unsigned major,func,mod,mask;std::array<std::uint32_t,16> a{},b{},c{},want{};std::string bar;unsigned want_p;std::uint32_t want_s;
                if(!(row>>std::hex>>major>>func>>mod>>mask))throw std::runtime_error("bad vfpu vector");
                for(auto* v:{&a,&b,&c})for(auto& x:*v)row>>x;
                row>>bar;if(bar!="|")throw std::runtime_error("bad vfpu separator");
                for(auto& x:want)row>>x;
                if(!(row>>want_p>>want_s))throw std::runtime_error("bad vfpu result");
                equal=true;
                if(major==0x11 && func==0x18){
                    unsigned got=0;for(unsigned l=0;l<16;++l)if((mask>>l)&1 && tnca_vector_ops::vfp_compare(mod,a[l],b[l]))got|=1U<<l;
                    equal=got==want_p;
                }else if(major==0x15){
                    equal=tnca_vector_ops::reduce(func,a,mask)==want_s;
                }else for(unsigned l=0;l<16;++l){
                    std::uint32_t got=major==0x11?tnca_vector_ops::vfp(func,mod,a[l],b[l],c[l]):
                                      major==0x12?tnca_vector_ops::cvt(func,mod,a[l]):tnca_sfu::evaluate(func,a[l]);
                    if(got!=want[l]){equal=false;if(failed<40)std::cerr<<"LANE "<<std::hex<<major<<'.'<<func<<'.'<<mod<<" in "<<a[l]<<' '<<b[l]<<' '<<c[l]<<" model "<<got<<" rtl "<<want[l]<<std::dec<<'\n';break;}
                }
                ++per_op[std::to_string(major)+"."+std::to_string(func)+"."+std::to_string(mod)][equal];
            }else throw std::runtime_error("unknown probe mode");
            ++checked;if(!equal){if(failed++<12)std::cerr<<"MISMATCH "<<line<<'\n';}
        }
        if(!checked)throw std::runtime_error("empty oracle vector file");
        std::cout<<"{\"checked\":"<<checked<<",\"mismatches\":"<<failed;
        if(!per_op.empty()){std::cout<<",\"by_operation\":{";bool first=true;
            for(auto& kv:per_op){if(!first)std::cout<<',';first=false;std::cout<<'"'<<kv.first<<"\":{\"pass\":"<<kv.second[1]<<",\"fail\":"<<kv.second[0]<<'}';}
            std::cout<<'}';}
        std::cout<<"}\n";return failed?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
