// Functional shared-BO model for runtime outlet switching. The historical
// fixed-home model has independent home shards; that is insufficient once
// the same physical address can use a different outlet after a drained switch.
// Mirror committed AXI writes into every view, matching the actual board's
// three aliases of one BO. Ownership excludes simultaneous host DMA writes.
// This test-only replication is not a memory-bandwidth or timing model.
// The B ID is AXI_ID_W bits per outlet (the including bench defines it); a
// fixed 3-bit slice mirrored the wrong slot's line once AXI_OUT_N exceeded 8.
generate if(MEMORY_CONTROLS)begin:g_shared_hbm
 for(genvar writer=0;writer<PORTS;writer=writer+1)begin:g_writer
  for(genvar reader=0;reader<PORTS;reader=reader+1)begin:g_reader
   if(writer!=reader)begin:g_mirror
    always @(posedge clk)if(rst_n_async && m_bvalid[writer] && m_bready[writer])begin
     g_ram[reader].ram.memory[g_ram[writer].ram.location[m_bid[writer*AXI_ID_W+:AXI_ID_W]]]
       <=g_ram[writer].ram.write_data[m_bid[writer*AXI_ID_W+:AXI_ID_W]];
    end
   end
  end
 end
end endgenerate
