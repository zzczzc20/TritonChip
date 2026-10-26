`timescale 1ns/1ps
module tb_nuca_maintenance;
 parameter SYNC_TAG_BANK=1;
 parameter LATENCY=8,STALL=1,REORDER=1,FULL_TAG=1;
 reg clk=0,rst_n=0; always #5 clk=~clk;
 reg req_valid=0,req_we=0,rsp_ready=1;
 reg [63:0] req_addr=0,req_data=0;
 reg [7:0] req_be=8'hff;
 wire req_ready,rsp_valid;wire [63:0] rsp_data;
 reg maint_valid=0;reg [63:0] maint_base=0,maint_size=0;
 wire maint_ready,maint_done,maint_busy;
 wire mem_valid,mem_ready,model_ready,mem_we,mem_tag,mem_rsp,mem_rsp_tag;
 wire [63:0] mem_addr;wire [511:0] mem_data,mem_rsp_data;
 reg stall_memory=0;
 integer writes=0,reads=0,cycles=0,i,j,read_before,write_before;
 reg [63:0] last_write_addr,last_write_word;
 reg waiting_memory=0;reg [577:0] held;
 wire [577:0] offered={mem_we,mem_tag,mem_addr,mem_data};
 assign mem_ready=model_ready && !stall_memory;
 tnca_nuca_slice #(.MAINTENANCE(1),.SYNC_TAG_BANK(SYNC_TAG_BANK),.FULL_LINE_TAG(FULL_TAG),
  .SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1),.MSHR_N(2),.MSHR_W(1),
  .TARGET_N(4),.TARGET_W(2),.RQ_N(4),.RQ_W(2),.TAG_W(5),.DCO_T(1),.DCO_DEAD(1)) dut(
  .clk(clk),.rst_n(rst_n),.maint_valid(maint_valid),.maint_ready(maint_ready),
  .maint_base(maint_base),.maint_size(maint_size),.maint_done(maint_done),.maint_busy(maint_busy),
  .req_valid(req_valid),.req_ready(req_ready),.req_addr(req_addr),.req_we(req_we),
  .req_be(req_be),.req_wdata(req_data),.req_src(4'd0),.req_tag(5'd0),
  .req_amo(1'b0),.req_aop(4'd0),.req_cmp(64'd0),
  .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_rdata(rsp_data),
  .mem_req_valid(mem_valid),.mem_req_ready(mem_ready),.mem_req_addr(mem_addr),
  .mem_req_we(mem_we),.mem_req_wdata(mem_data),.mem_req_tag(mem_tag),
  .mem_rsp_valid(mem_rsp),.mem_rsp_tag(mem_rsp_tag),.mem_rsp_rdata(mem_rsp_data),
  .dco_cfg(76'd0),.dco_t_valid(1'b0),.dco_t_base(64'd0),.dco_t_size(64'd0),
  .dco_t_bypass(1'b0),.dco_t_lsb(6'd0),.dco_dead_valid(1'b0),.dco_dead_id(16'd0),
  .dco_slow(16'd0),.tmu_wr_ready(1'b1),.tmu_rd_data(64'd0),.tmu_tl_ready(1'b1));
 // Validate selected-set alignment, including requests draining while a
 // maintenance request is pending. Checking data alone can miss a stale tag
 // read when the request happens to miss in both implementations.
 genvar tag_way;
 generate if(SYNC_TAG_BANK) begin : g_check_tags
  for(tag_way=0;tag_way<2;tag_way=tag_way+1)begin : g_way
   always @(negedge clk)
    if(rst_n && dut.est==1 && dut.v_bits[dut.cur_set*2+tag_way] &&
       dut.lookup_tag[tag_way] !== dut.g_sync_tags.g_way[tag_way].bank[dut.cur_set])
     $fatal(1,"synchronous tag read did not follow selected request set");
  end
 end endgenerate
 tnca_dram_model #(.LINES(128),.TAG_W(1),.LATENCY(LATENCY),
  .RANDOM_STALL(STALL),.REORDER(REORDER)) memory(
  .clk(clk),.rst_n(rst_n),.req_valid(mem_valid && !stall_memory),.req_ready(model_ready),
  .req_addr(mem_addr),.req_we(mem_we),.req_wdata(mem_data),.req_tag(mem_tag),.req_src(4'd0),
  .rsp_valid(mem_rsp),.rsp_tag(mem_rsp_tag),.rsp_rdata(mem_rsp_data));
 function [63:0] address;
  input integer line,word;
  begin
   if(FULL_TAG)address=line*64+word*8;
   else address=(line/4)*1024+(line%4)*256+(line%4)*64+word*8;
  end
 endfunction
 always @(posedge clk)begin
  cycles=cycles+1;
  if(cycles>20000)$fatal(1,"maintenance test timed out");
  if(rst_n)begin
   if(maint_busy && req_ready)$fatal(1,"core request admitted during maintenance");
   if(waiting_memory && (!mem_valid || offered!==held))$fatal(1,"memory request changed under backpressure");
   waiting_memory=mem_valid && !mem_ready;held=offered;
   if(mem_valid && mem_ready)begin
    if(mem_we)begin writes=writes+1;last_write_addr=mem_addr;last_write_word=mem_data[127:64];end
    else reads=reads+1;
   end
  end else waiting_memory=0;
 end
 task access;
  input [63:0] addr,data,expected;input write;input [7:0] mask;
  begin
   @(negedge clk);req_valid=1;req_addr=addr;req_we=write;req_data=data;req_be=mask;
   @(posedge clk);while(!req_ready)@(posedge clk);
   @(negedge clk);req_valid=0;
   @(posedge clk);while(!(rsp_valid && rsp_ready))@(posedge clk);
   if(!write && rsp_data!==expected)$fatal(1,"read %h expected %h got %h",addr,expected,rsp_data);
   @(negedge clk);
  end
 endtask
 task maintain;
  input [63:0] base,size;
  begin
   @(negedge clk);maint_valid=1;maint_base=base;maint_size=size;
   @(posedge clk);while(!maint_ready)@(posedge clk);
   @(negedge clk);maint_valid=0;
   // Inputs change after admission: the accepted range must be immutable.
   maint_base=64'hffffffffffffffff;maint_size=64'hffffffffffffffff;
   while(!maint_done)@(negedge clk);
   @(negedge clk);if(maint_done)$fatal(1,"completion was not a pulse");
  end
 endtask
 initial begin
  repeat(2)@(negedge clk);
  for(i=0;i<128;i=i+1)for(j=0;j<8;j=j+1)memory.mem[i][j*64+:64]=64'h100+i*8+j;
  rst_n=1;
  access(address(0,1),64'h1122334455667788,0,1,8'hff);
  access(address(0,2),64'haabb,0,1,8'h03);
  access(address(1,1),64'h7777,0,1,8'hff);
  maintain(address(0,1)+1,1);
  if(writes!=1 || last_write_addr!=address(0,0))$fatal(1,"wrong range written back");
  if(memory.mem[0][127:64]!==64'h1122334455667788 || memory.mem[0][191:128]!==64'haabb)
   $fatal(1,"dirty neighbours lost during partial-line handoff");
  read_before=reads;access(address(1,1),0,64'h7777,0,8'hff);
  if(reads!=read_before)$fatal(1,"unrelated line invalidated");
  // Host DMA changes one byte in backing memory after ownership transfer.
  memory.mem[0][79:72]=8'haa;
  access(address(0,1),0,64'h112233445566aa88,0,8'hff);
  access(address(0,2),0,64'haabb,0,8'hff);
  read_before=reads;write_before=writes;maintain(address(0,1),0);
  access(address(0,1),0,64'h112233445566aa88,0,8'hff);
  if(reads!=read_before || writes!=write_before)$fatal(1,"zero range changed cache");
  maintain(address(0,1)+1,1);memory.mem[0][79:72]=8'hcc;
  access(address(0,1),0,64'h112233445566cc88,0,8'hff);
  if(writes!=write_before)$fatal(1,"clean line wrote back");
  // Populate unequal tags in another set, then return the read port to set 0.
  access(address(5,0),0,64'h100+address(5,0)/8,0,8'hff);
  access(address(0,1),0,64'h112233445566cc88,0,8'hff);
  // Admission waits for queued requests and a stalled response to drain.
  rsp_ready=0;
  fork
   access(address(9,0),0,64'h100+address(9,0)/8,0,8'hff);
   begin wait(req_valid && req_ready);maintain(address(9,0),1);end
   begin wait(req_valid && req_ready);repeat(20)@(negedge clk);
    if(maint_ready || maint_busy || maint_done)$fatal(1,"maintenance overtook response");
    rsp_ready=1;
   end
  join
  access(64'h100000008,64'hbada55,0,1,8'hff);
  maintain(64'h100000009,1);
  if(last_write_addr!==64'h100000000 || last_write_word!==64'hbada55)$fatal(1,"high address truncated");
  access(64'hfffffffffffffff8,64'h1234,0,1,8'hff);
  maintain(64'hfffffffffffffff8,16);
  if(last_write_addr!==64'hffffffffffffffc0)$fatal(1,"range end wrapped");
  // Reset cancels an unaccepted dirty writeback, never returns stale done.
  access(address(8,1),64'hdead,0,1,8'hff);write_before=writes;stall_memory=1;
  @(negedge clk);maint_valid=1;maint_base=address(8,0);maint_size=64;
  @(posedge clk);while(!maint_ready)@(posedge clk);
  @(negedge clk);maint_valid=0;
  wait(mem_valid && mem_we);repeat(5)@(negedge clk);
  if(maint_done)$fatal(1,"completed before write acceptance");
  rst_n=0;repeat(2)@(negedge clk);rst_n=1;stall_memory=0;
  repeat(5)@(negedge clk);
  if(maint_busy || maint_done || writes!=write_before)$fatal(1,"stale maintenance after reset");
  $display("NUCA MAINTENANCE PASS full_tag=%0d latency=%0d stall=%0d reorder=%0d writes=%0d",FULL_TAG,LATENCY,STALL,REORDER,writes);
  $finish;
 end
endmodule
