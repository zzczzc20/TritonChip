#pragma once
// Explicit AXI BO translation. Physical PC IDs are topology tags, not a
// quotient of logical addresses. Bank/row decoding remains a separate model.
#include "tnca_model.hpp"
#include <set>

struct DramAddressMap {
    struct Bank {u64 pc,base,size,allocated_base,allocated_size;};
    struct Address {unsigned pc;u64 offset,physical;};
    u64 local_limit=0,bo_base=0,bo_size=0;
    std::vector<Bank> banks;
    DramAddressMap(const char* path,unsigned pc_count,u64 pc_bytes){
        std::ifstream input(path);require(bool(input),"DRAM address map: cannot open mapping");
        load(input,pc_count,pc_bytes);
    }
    DramAddressMap(std::istream& input,unsigned pc_count,u64 pc_bytes){load(input,pc_count,pc_bytes);}
    void load(std::istream& input,unsigned pc_count,u64 pc_bytes){
        std::string line;require(bool(std::getline(input,line)) && line=="TNCA_DRAM_MAP_V1","DRAM address map: wrong format");
        bool window=false;std::set<u64> ids;
        while(std::getline(input,line)){
            line=line.substr(0,line.find('#'));std::istringstream row(line);std::string kind;
            if(!(row>>kind))continue;
            auto value=[&](){std::string text;require(bool(row>>text),"DRAM address map: missing field");return integer(text);};
            if(kind=="window"){
                require(!window && banks.empty(),"DRAM address map: duplicate or late window");
                local_limit=value();bo_base=value();bo_size=value();window=true;
                require(bo_size && bo_size<=UINT64_MAX-bo_base && bo_size<=UINT64_MAX-local_limit &&
                    !((local_limit|bo_base|bo_size)&63),"DRAM address map: invalid AXI window");
            }else if(kind=="pc"){
                require(window,"DRAM address map: missing AXI window");
                Bank bank{value(),value(),value(),value(),value()};
                require(bank.pc<pc_count && ids.insert(bank.pc).second,"DRAM address map: invalid or repeated PC");
                require(bank.size==pc_bytes && bank.size<=UINT64_MAX-bank.base && bank.allocated_size &&
                    bank.allocated_size<=bank.size && bank.allocated_base>=bank.base &&
                    bank.allocated_base-bank.base<=bank.size-bank.allocated_size &&
                    !((bank.base|bank.size|bank.allocated_base|bank.allocated_size)&63),"DRAM address map: invalid PC geometry");
                banks.push_back(bank);
            }else throw std::runtime_error("DRAM address map: unknown record");
            row>>std::ws;require(row.eof(),"DRAM address map: trailing fields");
        }
        require(window && !banks.empty(),"DRAM address map: empty mapping");
        std::sort(banks.begin(),banks.end(),[](const Bank& a,const Bank& b){return a.allocated_base<b.allocated_base;});
        u64 cursor=bo_base,previous_end=0;
        for(const auto& bank:banks){
            require(bank.allocated_base==cursor && bank.allocated_size<=bo_base+bo_size-cursor &&
                bank.base>=previous_end,"DRAM address map: gaps, overlap or excess coverage");
            cursor+=bank.allocated_size;previous_end=bank.base+bank.size;
        }
        require(cursor==bo_base+bo_size,"DRAM address map: incomplete BO coverage");
    }
    Address translate(u64 logical,u64 bytes=64)const{
        require(bytes && logical>=local_limit && bytes<=bo_size && logical-local_limit<=bo_size-bytes,
            "DRAM address map: SRAM or address outside AXI BO");
        u64 physical=bo_base+(logical-local_limit);
        for(const auto& bank:banks){
            if(physical>=bank.allocated_base && bytes<=bank.allocated_size &&
                physical-bank.allocated_base<=bank.allocated_size-bytes)
                return {unsigned(bank.pc),physical-bank.base,physical};
        }
        throw std::runtime_error("DRAM address map: line crosses or misses a PC");
    }
};
