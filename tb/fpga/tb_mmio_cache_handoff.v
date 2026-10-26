`timescale 1ns/1ps
module tb_mmio_cache_handoff;
 parameter ENABLE=1;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg [7:0] address=0;reg [63:0] data=0;reg we=0;
 wire [63:0] read_data;
 reg ready=0,done=0,idle=1,fault=0;
 wire valid;wire [63:0] base,size;
 tnca_multicore_mmio #(.N(3),.PORTS(1),.CACHE_MAINTENANCE(ENABLE)) dut(
  .clk(clk),.rst_n(rst_n),.reg_addr(address),.reg_wdata(data),.reg_we(we),.reg_re(1'b1),.reg_rdata(read_data),
  .cmd_ready(1'b1),.done_valid(1'b0),.done_status(4'd0),.done_id(64'd0),.done_cycles(64'd0),
  .issued(96'd0),.completed(96'd0),.peak_active(32'd0),.active_cycles(192'd0),
  .host_ready(1'b0),.host_rsp_valid(1'b0),.host_rsp_data(64'd0),.host_rsp_status(4'd0),
  .all_idle(idle),.axi_fault(fault),.memory_reads(64'd0),.memory_writes(64'd0),
  .cache_maint_valid(valid),.cache_maint_ready(ready),.cache_maint_done(done),
  .cache_maint_base(base),.cache_maint_size(size));
 task write;
  input [7:0] a;input [63:0] v;
  begin @(negedge clk);address=a;data=v;we=1;@(negedge clk);we=0;end
 endtask
 task check_reg;
  input [7:0] a;input [63:0] v;
  begin address=a;#1;if(read_data!==v)$fatal(1,"register %h expected %h got %h",a,v,read_data);end
 endtask
 task request;
  input [63:0] seq,a0,a1;
  begin write(0,seq);write(8,18);write(16,a0);write(24,a1);write(72,1);end
 endtask
 initial begin
  repeat(2)@(negedge clk);rst_n=1;
  write(8'hb0,4096);
  check_reg(8'hb8,(64'h20<<32)|2|(ENABLE?16:0));
  request(1,64'h20009,1);
  if(!ENABLE)begin
   if(valid)$fatal(1,"disabled image accepted ownership");
   check_reg(80,1);check_reg(88,3);
   $display("MMIO CACHE HANDOFF PASS disabled");$finish;
  end
  if(!valid || base!==64'h20009 || size!==1)$fatal(1,"range not offered");
  check_reg(80,0);
  // Descriptor mutation cannot modify the accepted command.
  write(16,64'hffffffffffffffff);write(24,64'hffffffffffffffff);
  write(0,2);write(8,11);write(72,1);
  if(base!==64'h20009 || size!==1)$fatal(1,"range mutated after admission");
  check_reg(8'ha0,1);
  ready=1;@(negedge clk);ready=0;idle=0;
  if(valid)$fatal(1,"offered twice after acceptance");
  done=1;@(negedge clk);done=0;
  repeat(4)@(negedge clk);check_reg(80,0);
  idle=1;@(negedge clk);check_reg(80,1);check_reg(88,0);
  // Invalid ranges do not reach the ownership controller.
  request(3,0,1);check_reg(80,3);check_reg(88,3);if(valid)$fatal(1,"bad range admitted");
  request(4,64'h21000,1);check_reg(88,3);if(valid)$fatal(1,"past-end range admitted");
  request(5,64'h20000,64'hffffffffffffffff);check_reg(88,3);
  // Physical faults block admission and are reported at completion as well.
  fault=1;request(6,64'h20000,8);check_reg(80,6);check_reg(88,1);if(valid)$fatal(1,"faulted admission");
  fault=0;request(7,64'h20000,8);ready=1;@(negedge clk);ready=0;idle=0;fault=1;
  done=1;@(negedge clk);done=0;idle=1;@(negedge clk);check_reg(80,7);check_reg(88,1);
  // Reset cancels an unfinished ownership epoch and never publishes its seq.
  fault=0;request(8,64'h20000,8);
  rst_n=0;@(negedge clk);rst_n=1;
  repeat(3)@(negedge clk);check_reg(80,0);if(valid)$fatal(1,"stale ownership after reset");
  $display("MMIO CACHE HANDOFF PASS snapshot drain range faults reset");$finish;
 end
 initial begin #100000;$fatal(1,"test timed out");end
endmodule
