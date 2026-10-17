`timescale 1ns/1ps
module tb_idle_host_mux;
reg clk=0;always #5 clk=~clk;
reg rst=0,allow=0,cv=0,hv=0,nr=0,rv=0;
reg [4:0] rt=0;
wire cr,hr,busy,nv,nwe,na,hrv,crv;
wire [63:0] addr,wd,cmp,hd,cd;
wire [7:0] be;wire [3:0] op;wire [4:0] tag,ct;
tnca_idle_host_mux dut(.clk(clk),.rst_n(rst),.host_allow(allow),
.c_valid(cv),.c_ready(cr),.c_addr(64'h100),.c_wdata(64'h111),.c_cmp(64'h222),
.c_we(1'b1),.c_amo(1'b1),.c_be(8'h81),.c_aop(4'd3),.c_tag(5'd7),
.c_rsp_valid(crv),.c_rsp_tag(ct),.c_rsp_data(cd),.h_valid(hv),.h_ready(hr),
.h_we(1'b0),.h_addr(64'h200),.h_wdata(64'h333),.h_be(8'hff),
.h_rsp_valid(hrv),.h_rsp_data(hd),.host_busy(busy),.n_valid(nv),.n_ready(nr),
.n_addr(addr),.n_wdata(wd),.n_cmp(cmp),.n_we(nwe),.n_amo(na),.n_be(be),.n_aop(op),.n_tag(tag),
.n_rsp_valid(rv),.n_rsp_tag(rt),.n_rsp_data(64'habcdef));
initial begin
 repeat(3)@(negedge clk);rst=1;hv=1;nr=1;
 #1;if(hr || nv)$fatal(1,"host admitted while forbidden");
 cv=1;#1;if(!cr || !nv || addr!=256 || tag!=7 || !na || op!=3 || cmp!=64'h222 || be!=8'h81)$fatal(1,"core fields corrupted");
 @(negedge clk);cv=0;nr=0;allow=1;
 @(negedge clk);allow=0;cv=1;
 #1;if(!busy || !nv || addr!=512 || tag!=0 || na || nwe || op!=0 || cmp!=0 || cr || hr)$fatal(1,"stalled host ownership lost");
 repeat(3)@(negedge clk);
 nr=1;#1;if(!hr || cr)$fatal(1,"reserved request cannot retire");
 @(negedge clk);hv=0;
 #1;if(nv || !busy || cr)$fatal(1,"duplicate host request or early release");
 rv=1;rt=7;#1;if(hrv || crv)$fatal(1,"stray tag consumed");
 @(negedge clk);rt=0;#1;if(!hrv || crv || hd!=64'habcdef)$fatal(1,"host response wrong");
 @(negedge clk);rv=0;#1;if(busy || !cr || !nv || tag!=7)$fatal(1,"core not released");
 rv=1;rt=7;#1;if(!crv || ct!=7 || cd!=64'habcdef || hrv)$fatal(1,"core response wrong");
 @(negedge clk);cv=0;rv=0;allow=1;hv=1;nr=0;
 @(negedge clk);rst=0;
 @(negedge clk);hv=0;allow=0;rst=1;#1;if(busy)$fatal(1,"reset ownership retained");
 $display("IDLE HOST MUX PASS");$finish;
end
endmodule
