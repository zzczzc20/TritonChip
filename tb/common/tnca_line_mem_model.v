// Functional wide-cache-word fixture with the original 64-bit instruction
// image format. This models data transactions, not HBM/NoC performance.
`timescale 1ns/1ps
module tnca_line_mem_model #(
 parameter WORDS=65536,MEM_DW=512,MEM_BYTES=64,TAG_W=5,LATENCY=1,RANDOM_STALL=0,REORDER=0
)(
 input wire clk,rst_n,
 input wire [63:0] imem_addr,
 output reg [63:0] imem_data,
 input wire req_valid,
 output wire req_ready,
 input wire [63:0] req_addr,
 input wire req_we,
 input wire [63:0] req_be,
 input wire [511:0] req_wdata,
 input wire [TAG_W-1:0] req_tag,
 input wire req_amo,
 input wire [3:0] req_aop,
 input wire [511:0] req_cmp,
 output reg rsp_valid,
 output reg [TAG_W-1:0] rsp_tag,
 output reg [511:0] rsp_rdata
);
 `include "tnca_amo.vh"
 reg [63:0] mem[0:WORDS-1];
 reg [31:0] rng=32'h12345678;
 reg [31:0] cycle=0;
 reg [31:0] n_stall_cycles=0,n_reorder=0,n_multi_eligible=0;
 reg [63:0] mem_trace=0;
 localparam POOL=32;
 reg [POOL-1:0] valid;
 reg [TAG_W-1:0] tags[0:POOL-1];
 reg [511:0] data[0:POOL-1];
 integer due[0:POOL-1];
 reg free_found;
 integer free_slot;
 always @*begin
  free_found=0;free_slot=0;
  for(integer i=POOL-1;i>=0;i=i-1)if(!valid[i])begin free_found=1;free_slot=i;end
 end
 assign req_ready=free_found && (!RANDOM_STALL || rng[0]);
 initial for(integer i=0;i<WORDS;i=i+1)mem[i]=0;
 always @(posedge clk)imem_data<=imem_addr/8<WORDS?mem[imem_addr/8]:64'd0;
 integer pick,choices,base;
 reg [511:0] snapshot,merged;
 reg [63:0] word_value;
 always @(posedge clk)begin
  if(!rst_n)begin valid<=0;rsp_valid<=0;cycle<=0;n_stall_cycles<=0;n_reorder<=0;n_multi_eligible<=0;mem_trace<=0;end
  else begin
   cycle<=cycle+1;rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};rsp_valid<=0;
   if(req_valid && !req_ready)n_stall_cycles<=n_stall_cycles+1;
   pick=-1;choices=0;
   for(integer i=0;i<POOL;i=i+1)if(valid[i] && due[i]<=cycle)begin
    choices=choices+1;
    if(pick<0 || (REORDER && ((rng>>(i%16))&1)))pick=i;
   end
   if(choices>1)n_multi_eligible<=n_multi_eligible+1;
   if(pick>=0)begin
    rsp_valid<=1;rsp_tag<=tags[pick];rsp_rdata<=data[pick];valid[pick]<=0;
    if(REORDER && choices>1)n_reorder<=n_reorder+1;
   end
   if(req_valid && req_ready)begin
    base=req_addr & ~64'd63;
    if(req_addr[2:0]!=0 || base<0 || base+63>=WORDS*8)$fatal(1,"wide fixture address out of range %h",req_addr);
    snapshot=0;
    for(integer w=0;w<8;w=w+1)snapshot[w*64+:64]=mem[base/8+w];
    merged=snapshot;
    if(req_we || req_amo)begin
     for(integer w=0;w<8;w=w+1)begin
      word_value=req_amo?amo_new(req_aop,req_be[w*8+:8],snapshot[w*64+:64],req_wdata[w*64+:64],req_cmp[63:0]):req_wdata[w*64+:64];
      for(integer b=0;b<8;b=b+1)if(req_be[w*8+b])merged[w*64+b*8+:8]=word_value[b*8+:8];
      mem[base/8+w]<=merged[w*64+:64];
     end
    end
    valid[free_slot]<=1;tags[free_slot]<=req_tag;data[free_slot]<=snapshot;
    due[free_slot]<=cycle+LATENCY+(REORDER?rng[5:2]:0);
    mem_trace<={mem_trace[56:0],mem_trace[63:57]}^req_addr^{59'd0,req_tag};
   end
  end
 end
 task load_hex;input [1023:0] path;begin $readmemh(path,mem);end endtask
 task dump_hex;input [1023:0] path;input integer first,count;integer fd;begin
  fd=$fopen(path,"w");for(integer i=first;i<first+count;i=i+1)$fdisplay(fd,"%016h",mem[i]);$fclose(fd);
 end endtask
endmodule
