`timescale 1ns/1ps
// Check actual TMU commits, including interleaved staging and unsigned overflow.
module tb_tmu_registered_ranges;
parameter PORTS=3;
localparam HIGH_PORT=PORTS-1;
reg clk=0; always #5 clk=~clk;
reg rst_n=0;
reg [PORTS-1:0] valid=0;
wire [PORTS-1:0] ready;
reg [PORTS*12-1:0] off=0;
reg [PORTS*64-1:0] data=0;
reg [PORTS*5-1:0] src=0;
wire [7:0] tv;
wire [511:0] bases,sizes;
wire [519:0] endpoints;
reg [PORTS-1:0] notices=0;
reg [PORTS-1:0] drops=0;
reg [PORTS*64-1:0] notice_addr=0;
wire [PORTS-1:0] notice_ready;
wire [63:0] dead_ids;
wire [3:0] dead_valid;
reg [PORTS-1:0] notice_accepted=0;
integer notice_step=0,notice_hits=0,dead_checks=0;
integer checks=0,commits=0;
tnca_tmu #(.REGISTERED_RANGES(1),.N_SLICE(PORTS),.N_SRC(16),.SRC_W(5),.TENSORS(8),.T_W(3),
 .LIVE(4),.LIVE_W(2),.DEAD(4),.DEAD_W(2)) dut(
 .clk(clk),.rst_n(rst_n),.mw_valid(valid),.mw_ready(ready),.mw_off(off),.mw_data(data),.mw_src(src),
 .mr_off({PORTS*12{1'b0}}),.mr_data(),.tl_valid(notices),.tl_ready(notice_ready),.tl_addr(notice_addr),.tl_drop(drops),
 .policy(),.b_bits(),.d_lsb(),.gear_init(),.gear_window(),.gear_ub(),.gear_lb(),
 .t_valid(tv),.t_base(bases),.t_size(sizes),.t_end(endpoints),.t_bypass(),.t_lsb(),.dead_valid(dead_valid),.dead_id(dead_ids));
wire [PORTS-1:0] reference_ready,reference_notice_ready;
wire [3:0] reference_dead_valid;
wire [63:0] reference_dead_ids;
tnca_tmu #(.REGISTERED_RANGES(0),.N_SLICE(PORTS),.N_SRC(16),.SRC_W(5),.TENSORS(8),.T_W(3),
 .LIVE(4),.LIVE_W(2),.DEAD(4),.DEAD_W(2)) reference(
 .clk(clk),.rst_n(rst_n),.mw_valid(valid),.mw_ready(reference_ready),.mw_off(off),.mw_data(data),.mw_src(src),
 .mr_off({PORTS*12{1'b0}}),.mr_data(),.tl_valid(notices),.tl_ready(reference_notice_ready),.tl_addr(notice_addr),.tl_drop(drops),
 .policy(),.b_bits(),.d_lsb(),.gear_init(),.gear_window(),.gear_ub(),.gear_lb(),
 .t_valid(),.t_base(),.t_size(),.t_end(),.t_bypass(),.t_lsb(),.dead_valid(reference_dead_valid),.dead_id(reference_dead_ids));
always @(posedge clk)begin
 notice_accepted<=notices & notice_ready;
 if(rst_n && dut.tl_go && dut.nt_hit)notice_hits=notice_hits+1;
end
integer peer,slot;
always @(negedge clk)begin
 if(rst_n)begin
  if(ready!==reference_ready || notice_ready!==reference_notice_ready)$fatal(1,"TMU handshake changed");
  if(dut.nt_hit!==reference.nt_hit || (dut.nt_hit && dut.nt_idx!==reference.nt_idx))$fatal(1,"TMU owner selection changed");
  if(dead_valid!==reference_dead_valid)$fatal(1,"TMU dead validity changed");
  for(peer=0;peer<4;peer=peer+1)if(dead_valid[peer])begin
   if(dead_ids[peer*16+:16]!==reference_dead_ids[peer*16+:16])$fatal(1,"TMU dead tile changed");
   dead_checks=dead_checks+1;
  end
  if(dut.stat_tlw!==reference.stat_tlw || dut.stat_dead!==reference.stat_dead)$fatal(1,"TMU counting changed");
  notice_step=notice_step+1;
  for(peer=0;peer<PORTS;peer=peer+1)if(!notices[peer] || notice_accepted[peer])begin
   slot=(notice_step+peer)%8;
   notices[peer]=tv[slot];
   notice_addr[peer*64+:64]=bases[slot*64+:64]+(peer==2 ? 64'hffffffffffffffff : 64'd0);
  end
 end else notices=0;
end
integer k;
reg [64:0] expected;
always @(negedge clk) if(rst_n) begin
 for(k=0;k<8;k=k+1) if(tv[k])begin
  expected={1'b0,bases[k*64+:64]}+{1'b0,sizes[k*64+:64]};
  if(sizes[k*64+:64]==0)expected={1'b1,64'd0};
  if(endpoints[k*65+:65]!==expected)$fatal(1,"endpoint epoch mismatch slot=%0d",k);
  checks=checks+1;
 end
end

task automatic put;
 input integer port,requester;
 input [11:0] address;
 input [63:0] value;
 begin
  @(negedge clk); off[port*12+:12]=address;data[port*64+:64]=value;
  src[port*5+:5]=requester;valid[port]=1;
  @(posedge clk);while(!ready[port])@(posedge clk);
  @(negedge clk);valid[port]=0;
 end
endtask

task automatic check_entry;
 input [63:0] base,size;
 integer i,found;
 begin
  #1;found=0;
  for(i=0;i<8;i=i+1)if(tv[i] && bases[i*64+:64]==base)begin
   if(sizes[i*64+:64]!==size)$fatal(1,"wrong committed size");
   found=found+1;
  end
  if(found!=1)$fatal(1,"registration missing or duplicated base=%h size=%h valid=%h bases=%h",base,size,tv,bases);
 end
endtask

task automatic register_tensor;
 input integer port,requester;
 input [63:0] base,size;
 begin
  put(port,requester,12'h100,base);
  put(port,requester,12'h108,size);
  put(port,requester,12'h110,64'h0000_0900_0000_0003);
  check_entry(base,size);commits=commits+1;
 end
endtask

task automatic stream;
 input integer port,requester;
 input [63:0] origin;
 integer step;
 begin
  for(step=0;step<12;step=step+1)
   register_tensor(port,requester,origin+64'd32*step,
                   step%3==0 ? 64'd0 : (64'd1 << (step+40)));
 end
endtask
reg [31:0] before_drop;
reg [7:0] before_valid;
reg [511:0] before_base,before_size;
reg [519:0] before_end;
initial begin
 repeat(3)@(negedge clk);rst_n=1;
 before_drop=dut.stat_drop;drops={PORTS{1'b1}};
 repeat(3)@(negedge clk);#1;
 if(dut.stat_drop-before_drop!=PORTS*3)$fatal(1,"drop count overflowed for concurrent notices");
 drops=0;
 register_tensor(0,15,64'hfffffffffffffff0,64'd32);
 register_tensor(1,5,64'h8000000000000000,64'd0);
 register_tensor(HIGH_PORT,0,64'd0,64'hffffffffffffffff);
 // BASE/SIZE staging must not publish any part of a new metadata epoch.
 before_valid=tv;before_base=bases;before_size=sizes;before_end=endpoints;
 put(0,15,12'h100,64'hfffffffffffffff0);
 put(0,15,12'h108,64'd4);
 #1;
 if(tv!==before_valid || bases!==before_base || sizes!==before_size || endpoints!==before_end)
  $fatal(1,"staged metadata visible before DESC commit");
 // Out-of-range IDs must not alias legal staging or clear global metadata.
 put(1,16,12'h100,64'h2222);
 put(1,16,12'h108,64'd64);
 put(1,16,12'h110,64'd3);
 put(1,31,12'h018,64'd1);
 #1;
 if(tv!==before_valid || bases!==before_base || sizes!==before_size || endpoints!==before_end)
  $fatal(1,"invalid requester changed metadata");
 put(0,15,12'h110,64'd7);
 check_entry(64'hfffffffffffffff0,64'd4);
 fork
  stream(0,0,64'h1000);
  stream(1,4,64'h8000000000000000);
  stream(HIGH_PORT,12,64'hffffffffffff0000);
 join
 put(0,0,12'h018,64'd1);#1;
 if(tv!==0)$fatal(1,"CLEAR did not invalidate endpoints");
 register_tensor(0,3,64'hffffffffffffffff,64'd1);
 register_tensor(1,14,64'hffffffffffffffff,64'd0);
 @(negedge clk);rst_n=0;@(negedge clk);#1;
 if(tv!==0)$fatal(1,"reset did not invalidate endpoints");
 rst_n=1;register_tensor(HIGH_PORT,7,64'h8000000000000000,64'h8000000000000000);
 repeat(16)@(negedge clk);
 if(notice_hits==0 || dead_checks==0)$fatal(1,"no tile notices/deaths exercised");
 if(checks<100 || commits<40)$fatal(1,"insufficient exercised commits");
 $display("TMU REGISTERED RANGES PASS ports=%0d requester_width=5 commits=%0d invariant_checks=%0d",PORTS,commits,checks);
 $finish;
end
initial begin #200000;$fatal(1,"TMU range bench timeout");end
endmodule
