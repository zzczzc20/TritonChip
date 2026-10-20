`timescale 1ns/1ps
module tb_mmio_memory_controls;
 parameter ENABLE=1;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg [8:0] address=0;reg [63:0] data=0;reg we=0,idle=1,ready=0,done=0,host_ready=0,host_rsp=0;
 wire [63:0] value;wire [7:0] ports,limit;
 tnca_multicore_mmio #(.PORTS(3),.REG_ADDR_W(9),.CACHE_MAINTENANCE(1),.MEMORY_CONTROLS(ENABLE),.AXI_OUT_N(8)) dut(
  .clk(clk),.rst_n(rst_n),.reg_addr(address),.reg_wdata(data),.reg_we(we),.reg_re(1'b1),.reg_rdata(value),
  .cmd_ready(1'b0),.done_valid(1'b0),.done_status(4'd0),.done_id(64'd0),.done_cycles(64'd0),
  .issued(96'd0),.completed(96'd0),.peak_active(32'd0),.active_cycles(192'd0),
  .host_ready(host_ready),.host_rsp_valid(host_rsp),.host_rsp_data(64'd0),.host_rsp_status(4'd0),
  .all_idle(idle),.axi_fault(3'd0),.memory_reads(192'd0),.memory_writes(192'd0),
  .cache_maint_ready(ready),.cache_maint_done(done),.memory_active_ports(ports),.memory_outstanding(limit));
 task write;
  input [8:0] a;input [63:0] v;
  begin @(negedge clk);address=a;data=v;we=1;@(negedge clk);we=0;end
 endtask
 task check;
  input [8:0] a;input [63:0] v;
  begin address=a;#1;if(value!==v)$fatal(1,"register %h expected %h got %h",a,v,value);end
 endtask
 task set_config;
  input [63:0] word,expected;
  begin write(9'h110,word);check(9'h110,expected);end
 endtask
 initial begin
  repeat(3)@(negedge clk);rst_n=1;
  check(9'hb8,(64'h20<<32)|64'd18|(ENABLE?64'd32:0));
  check(9'h118,ENABLE?64'h5234304d00010308:0);
  check(9'h110,ENABLE?64'h308:0);
  if(!ENABLE)begin
   set_config(64'h101,0);if(ports!=3 || limit!=8)$fatal(1,"disabled control changed");
   $display("MMIO MEMORY CONTROLS PASS disabled");$finish;
  end
  set_config(64'h101,64'h101);check(9'h10,0);check(9'h18,0);
  set_config(64'h308,64'h308);set_config(64'h108,64'h108);set_config(64'h301,64'h301);
  set_config(0,64'h301);set_config(64'h201,64'h301);set_config(64'h401,64'h301);
  set_config(64'h300,64'h301);set_config(64'h309,64'h301);set_config(64'h1000308,64'h301);
  idle=0;set_config(64'h108,64'h301);idle=1;
  // An accepted ownership command blocks changes even if the idle pin is high.
  write(9'hb0,4096);write(0,1);write(8,18);write(16,64'h20000);write(24,64'd64);write(72,1);
  set_config(64'h108,64'h301);ready=1;@(negedge clk);ready=0;
  set_config(64'h108,64'h301);done=1;@(negedge clk);done=0;
  repeat(4)@(negedge clk);set_config(64'h108,64'h108);
  // Pending coherent host traffic also blocks a routing/admission change.
  write(9'h70,0);write(9'h88,2);set_config(64'h301,64'h108);
  host_ready=1;repeat(2)@(negedge clk);host_ready=0;host_rsp=1;
  @(negedge clk);host_rsp=0;repeat(2)@(negedge clk);set_config(64'h301,64'h301);
  // SETTLE: an accepted routing/aperture change holds admission until the
  // registered copies in the memory mode and AXI adapters have it. Busy
  // reads 1 meanwhile, and a window access in that interval is refused.
  write(9'h110,64'h308);check(9'h48,1);
  write(9'h88,2);check(9'h90,3'b100);check(9'h48,0);
  write(9'h88,2);check(9'h90,3'b001);
  host_ready=1;@(negedge clk);host_ready=0;host_rsp=1;@(negedge clk);host_rsp=0;@(negedge clk);
  write(9'hb0,64'd8192);check(9'h48,1);repeat(2)@(negedge clk);check(9'h48,0);
  rst_n=0;repeat(2)@(negedge clk);rst_n=1;check(9'h110,64'h308);
  $display("MMIO MEMORY CONTROLS PASS quiescence invalid counts ownership host reset aliases");$finish;
 end
 initial begin #100000;$fatal(1,"memory control bench timeout");end
endmodule
