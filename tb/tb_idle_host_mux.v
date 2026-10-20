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
// Randomized core/host interleaving after the directed checks. The bench
// respects the enclosing contract: permission is offered only while no core
// request is outstanding. Checked every cycle: one side owns the port, the
// core ready is n_ready && !host_busy, a grant follows a cycle with
// permission, no core request and a host request, at most one host request
// per grant, and no core response is delivered while the host owns the port.
reg probe,random_phase=0;
integer grants=0,host_done=0,core_done=0,core_out=0,host_out=0,host_issued_n=0,rsp_wait=0,rng=7;
reg prev_allow=0,prev_cv=0,prev_hv=0,prev_busy=0,core_acc=0,host_acc=0;
function integer rnd;input integer m;begin rng=rng*1103515245+12345;rnd=((rng>>>16)&32767)%m;end endfunction
always @(posedge clk) if(rst && random_phase) begin
 if(cr && hr)$fatal(1,"core and host both ready");
 if(cr!==(nr && !busy))$fatal(1,"core ready is not n_ready && !host_busy");
 if(busy && !prev_busy && !(prev_allow && !prev_cv && prev_hv))$fatal(1,"grant without permission/idle core/host request");
 if(busy && nv && tag!==0)$fatal(1,"core request issued while host owns the port");
 if(crv && busy)$fatal(1,"core response delivered to a host-owned port");
 host_acc=hv && hr;core_acc=cv && cr;
 if(host_acc)begin host_issued_n=host_issued_n+1;host_out=host_out+1;end
 if(core_acc)core_out=core_out+1;
 if(busy && !prev_busy)grants=grants+1;
 if(host_issued_n>grants)$fatal(1,"more host requests than grants");
 if(hrv)begin host_done=host_done+1;host_out=host_out-1;end
 if(crv)begin core_done=core_done+1;core_out=core_out-1;end
 prev_allow=allow;prev_cv=cv;prev_hv=hv;prev_busy=busy;
end
always @(negedge clk) if(rst && random_phase) begin
 if(core_acc)cv=0; else if(!cv && core_out==0 && !busy && rnd(4)==0)cv=1;
 if(host_acc)hv=0; else if(!hv && host_out==0 && !busy && rnd(6)==0)hv=1;
 core_acc=0;host_acc=0;
 allow=(core_out==0 && !cv && rnd(3)!=0);
 nr=(rnd(4)!=0);
 rv=0;
 if(rsp_wait>0)rsp_wait=rsp_wait-1;
 else if(host_out>0 && busy)begin rv=1;rt=0;rsp_wait=rnd(5);end
 else if(core_out>0 && !busy)begin rv=1;rt=7;rsp_wait=rnd(5);end
end
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
 // The grant is registered: the core-side ready never depends on host_allow.
 // (With no core request and a host request waiting, the old combinational
 // offer made c_ready follow host_allow within the same cycle.)
 cv=0;hv=1;nr=1;allow=0;#1;probe=cr;allow=1;#1;if(cr!==probe)$fatal(1,"c_ready depends on host_allow");
 allow=0;hv=0;#1;if(cr!==probe)$fatal(1,"c_ready depends on host_allow");
 @(negedge clk);
 random_phase=1;repeat(20000)@(negedge clk);random_phase=0;
 cv=0;hv=0;allow=0;repeat(40)@(negedge clk);
 if(host_done==0 || core_done==0)$fatal(1,"random phase made no progress host=%0d core=%0d",host_done,core_done);
 $display("IDLE HOST MUX PASS directed + random grants=%0d host=%0d core=%0d",grants,host_done,core_done);$finish;
end
endmodule
