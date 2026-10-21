// A real accepted outlet write must finish before maintenance reports done,
// including when no dirty cache line remains to trigger another writeback.
#include "../include/tnca_timeline_memory.hpp"
#include <filesystem>

int sc_main(int argc,char** argv){
    try{
        require(argc==3,"usage: tnca-timeline-drain-test MODEL.cfg FRESH_OUTPUT");
        require(!std::filesystem::exists(argv[2]),"preserve existing drain evidence");
        std::filesystem::create_directories(argv[2]);
        Config config(argv[1]);Context context(config,argv[2]);Store store;
        store.configure(config.get("memory_base"),config.get("memory_bytes"));
        TimelineMemory memory(config,context,store);
        u64 accepted_done=memory.line(config.home(0x40000),0x40000,true,0,0);
        require(accepted_done>0,"fixture did not admit a future write");
        u64 maintenance_done=memory.flush(0);
        std::cout<<"{\"accepted_write_complete\":"<<accepted_done<<",\"maintenance_complete\":"<<maintenance_done<<"}\n";
        require(maintenance_done>=accepted_done,"maintenance reported completion with accepted writes outstanding");
        u64 read_done=memory.line(config.home(0x50000),0x50000,false,maintenance_done+1,0);
        u64 read_drained=memory.flush(maintenance_done+1);
        require(read_drained>=read_done,"maintenance reported completion with accepted reads outstanding");
        std::cout<<"{\"accepted_read_complete\":"<<read_done<<",\"maintenance_complete\":"<<read_drained<<"}\n";
        context.close();
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
