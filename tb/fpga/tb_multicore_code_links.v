`timescale 1ns/1ps
module tb_multicore_code_links;
 parameter N=3,STAGES=2;
 localparam NX=N>4?4:N,SRC_W=N>4?$clog2(N):2;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg host_valid=0,host_we=0;reg [63:0] host_addr=0,host_wdata=0;reg [7:0] host_be=0;
 wire host_ready,host_rsp_valid,cmd_ready,all_idle;
 wire [63:0] host_rsp_data;wire [3:0] host_rsp_status;
 tnca_multicore_shell #(.NX(NX),.NY(N/NX),.CODE_WORDS(16),.CODE_LINK_STAGES(STAGES),
  .SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut(
  .clk(clk),.rst_n(rst_n),.cmd_valid(1'b0),.cmd_ready(cmd_ready),
  .cmd_id(64'd0),.cmd_pc(64'd0),.cmd_arg_base(64'd0),.cmd_arg_stride(64'd0),.cmd_deadline(64'd0),
  .cmd_gx(32'd0),.cmd_gy(32'd0),.cmd_gz(32'd0),.cmd_group_log2(2'd0),
  .cmd_cores({N{1'b0}}),.cmd_roles(4'd0),.cmd_slice(4'd0),.cmd_vbase(32'd0),.done_ready(1'b1),
  .host_valid(host_valid),.host_ready(host_ready),.host_we(host_we),.host_addr(host_addr),
  .host_wdata(host_wdata),.host_be(host_be),.host_rsp_valid(host_rsp_valid),
  .host_rsp_data(host_rsp_data),.host_rsp_status(host_rsp_status),.all_idle(all_idle),
  .platform_fault(1'b0),.ports_req_ready(1'b1),.ports_rsp_valid(1'b0),.ports_rsp_tag(3'd0),
  .ports_rsp_src({SRC_W{1'b0}}),.ports_rsp_rdata(512'd0),
  .mem_req_ready(1'b1),.mem_rsp_valid(1'b0),.mem_rsp_tag(3'd0),
  .mem_rsp_src({SRC_W{1'b0}}),.mem_rsp_rdata(512'd0));
 reg check_words=0;reg [63:0] expected=0;
 generate for(genvar c=0;c<N;c=c+1)begin:g_check
  always @(posedge clk)if(check_words && dut.g_code[c].code[15]!==expected)
   $fatal(1,"private RAM %0d differs: %h vs %h",c,dut.g_code[c].code[15],expected);
 end endgenerate
 integer cycles,responses=0;
 always @(posedge clk)if(rst_n && host_rsp_valid)responses=responses+1;
 task transaction;
  input write;input [63:0] data;input [7:0] be;input [63:0] read_expected;
  begin
   @(negedge clk);host_valid=1;host_we=write;host_addr=120;host_wdata=data;host_be=be;
   cycles=0;
   begin:admission
    forever begin @(posedge clk);if(host_ready)disable admission;cycles=cycles+1;if(cycles>100)$fatal(1,"admission timeout");end
   end
   @(negedge clk);host_valid=0;host_addr=0;host_wdata=~data;host_be=0;
   cycles=0;
   while(!host_rsp_valid)begin
    if(cmd_ready || all_idle)$fatal(1,"upload reported idle before all acknowledgements");
    @(negedge clk);cycles=cycles+1;if(cycles>100)$fatal(1,"reply timeout");
   end
   if(host_rsp_status!=0 || (!write && host_rsp_data!==read_expected))$fatal(1,"wrong readback/status");
   if(STAGES>0 && cycles<2*STAGES)$fatal(1,"reply crossed the pipeline too early");
   @(posedge clk);@(negedge clk);
   repeat(3)@(negedge clk);
   if(!all_idle || !cmd_ready || host_rsp_valid)$fatal(1,"code link failed to drain");
  end
 endtask
 initial begin
  repeat(5)@(negedge clk);rst_n=1;repeat(5)@(negedge clk);
  transaction(1,64'h0123456789abcdef,8'hff,0);
  expected=64'h0123456789abcdef;check_words=1;@(negedge clk);check_words=0;
  transaction(1,64'hfedcba9876543210,8'h0f,0);
  expected=64'h0123456776543210;check_words=1;@(negedge clk);check_words=0;
  transaction(0,0,0,expected);
  transaction(1,64'hfedcba9876543210,8'hf0,0);
  expected=64'hfedcba9876543210;check_words=1;@(negedge clk);check_words=0;
  transaction(0,0,0,expected);
  if(responses!=5)$fatal(1,"duplicate/missing response %0d",responses);
  $display("MULTICORE CODE LINKS PASS cores=%0d stages=%0d snapshot/byte-masks/private-RAM/drain",N,STAGES);$finish;
 end
 initial begin #100000;$fatal(1,"watchdog");end
endmodule
