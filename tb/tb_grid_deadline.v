`timescale 1ns/1ps
module tb_grid_deadline;
reg clk=0;always #5 clk=~clk;
reg rst=0,start=0,active=0;
reg [63:0] deadline=0;
wire abort,timed_out;wire [63:0] elapsed;
tnca_grid_deadline dut(.clk(clk),.rst_n(rst),.start(start),.active(active),.deadline(deadline),
.abort_valid(abort),.timed_out(timed_out),.elapsed(elapsed));
initial begin
 repeat(3)@(negedge clk);rst=1;start=1;active=1;deadline=3;
 @(negedge clk);start=0;deadline=99;
 repeat(2)begin @(negedge clk);if(abort || timed_out)$fatal(1,"early timeout");end
 @(negedge clk);if(!abort || timed_out || elapsed!=3)$fatal(1,"deadline not latched");
 @(negedge clk);if(abort || !timed_out || !active)$fatal(1,"timeout not one-shot or active changed");
 repeat(5)@(negedge clk);if(!timed_out)$fatal(1,"timeout forgotten during drain");
 active=0;@(negedge clk);if(!timed_out)$fatal(1,"timeout forgotten after drain");
 start=1;deadline=0;active=1;
 @(negedge clk);start=0;
 repeat(20)begin @(negedge clk);if(abort || timed_out)$fatal(1,"zero deadline timed out");end
 rst=0;@(negedge clk);if(timed_out || elapsed!=0)$fatal(1,"reset failed");
 $display("GRID DEADLINE PASS");$finish;
end
endmodule
