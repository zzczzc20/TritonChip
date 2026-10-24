// Test-only BO byte access. Device requests have one canonical home; each
// outlet model stores its home shard. Host writes initialize every view and
// host reads select the canonical home, as a shared physical BO would.
// This bypasses cache and MMIO word accesses; it is not an XRT timing model.
reg [63:0] dma_offset=0;
reg [7:0] dma_byte=0;
reg dma_write=0,dma_wide=0;
reg [63:0] dma_word=0;
wire [PORTS*8-1:0] dma_read;
wire [PORTS*64-1:0] dma_read_word;
wire [63:0] dma_virtual_line=(64'h20000+dma_offset)>>6;
wire [7:0] dma_hash=dma_virtual_line[7:0]^dma_virtual_line[15:8]^
 dma_virtual_line[23:16]^dma_virtual_line[31:24]^dma_virtual_line[39:32]^
 dma_virtual_line[47:40]^dma_virtual_line[55:48]^dma_virtual_line[63:56];
wire [7:0] dma_home=(NX==2 && N/NX==2)?
 {6'd0,(dma_virtual_line[1:0]^dma_virtual_line[3:2])}:dma_hash%N;
wire [7:0] dma_port=dma_home%PORTS;
// The model line holding BO byte dma_offset: dense, or through the same
// sparse map every outlet model loaded (-1: unbacked, refused by the caller).
`include "tb/common/tnca_sparse_map.vh"
initial if(SPARSE)sparse_load(SPARSE_MAP);
// Explicit sensitivity: a function's globals are not in @*, and the map
// loads at time zero, after which the same offset must be re-located.
integer dma_line=0;
always @(dma_offset or sparse_count) dma_line=SPARSE ? sparse_index(dma_offset>>6) : dma_offset>>6;
generate for(genvar dma_p=0;dma_p<PORTS;dma_p=dma_p+1)begin:g_dma
 assign dma_read[dma_p*8+:8]=g_ram[dma_p].ram.memory[dma_line][dma_offset[5:0]*8+:8];
 assign dma_read_word[dma_p*64+:64]=g_ram[dma_p].ram.memory[dma_line][dma_offset[5:0]*8+:64];
 always @(posedge dma_write)if(rst_n_async)begin
  if(dma_wide)g_ram[dma_p].ram.memory[dma_line][dma_offset[5:0]*8+:64]<=dma_word;
  else g_ram[dma_p].ram.memory[dma_line][dma_offset[5:0]*8+:8]<=dma_byte;
 end
end endgenerate
