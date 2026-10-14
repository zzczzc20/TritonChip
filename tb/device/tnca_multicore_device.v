`timescale 1ns/1ps
module tnca_multicore_device;
parameter N=3;
parameter TOTAL_WORDS=1048576;
parameter VRF_BYPASS=1,FE_NOBUBBLE=1,VRF_PREFETCH=1,RCTX_SWITCH=2;
parameter LSU_INFLIGHT=2,FPU_OVERLAP=1,MTX_ABUF=1;
localparam NX=N>4?4:N,SW=N>4?$clog2(N):2;
reg clk=0;always #5 clk=~clk;
reg rst=0,cv=0,ack=0,hv=0,hw=0;
reg [63:0] pc=0,id=11,ha=0,hd=0,deadline=0;
wire timed_out;
wire cr,dv,hr,hrv,busy,idle;
wire [3:0] ds,hs;
wire [63:0] did,dc,hrd;
wire [N*32-1:0] issued,completed;
wire [31:0] peak_active;
wire [N*64-1:0] active_cycles;
wire mr,mready,mwe,mrv;
wire [63:0] ma;wire [511:0] mw,md;
wire [2:0] mt,mrt;wire [SW-1:0] ms,mrs;
reg [31:0] gx=1,gy=1,gz=1,role_flags=1;
reg [N-1:0] cores={N{1'b1}};
reg [63:0] arg_base=0,arg_stride=0;
reg [1023:0] req_path,rsp_path,up_path,down_path;
reg signed [63:0] seq,served=0,a0,a1,a2,a3,a4,a5;
reg [63:0] r0,r1,word_data,read_data;
integer op,status,fd,n,j,k,pollcycles=20;
reg poisoned=0;
integer cycles=0;
tnca_multicore_shell #(.NX(NX),.NY(N/NX),.CODE_WORDS(16384),.VRF_BYPASS(VRF_BYPASS),.FE_NOBUBBLE(FE_NOBUBBLE),
.VRF_PREFETCH(VRF_PREFETCH),.RCTX_SWITCH(RCTX_SWITCH),.LSU_INFLIGHT(LSU_INFLIGHT),
.FPU_OVERLAP(FPU_OVERLAP),.MTX_ABUF(MTX_ABUF),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut(
.clk(clk),.rst_n(rst),.cmd_valid(cv),.cmd_ready(cr),.cmd_id(id),.cmd_pc(pc),
.cmd_group_log2(role_flags[9:8]),.cmd_deadline(deadline),.timed_out(timed_out),.cmd_arg_base(arg_base),.cmd_arg_stride(arg_stride),.cmd_gx(gx),.cmd_gy(gy),.cmd_gz(gz),
.cmd_cores(cores),.cmd_roles(role_flags[3:0]),.cmd_slice({role_flags[28],role_flags[31:29]}),.cmd_vbase(32'd0),
.done_valid(dv),.done_ready(ack),.done_status(ds),.done_id(did),.done_cycles(dc),
.issued(issued),.completed(completed),.host_valid(hv),.host_ready(hr),.host_we(hw),
.peak_active(peak_active),.active_cycles(active_cycles),
.host_addr(ha),.host_wdata(hd),.host_be(8'hff),.host_rsp_valid(hrv),.host_rsp_data(hrd),.host_rsp_status(hs),
.busy(busy),.all_idle(idle),.mem_req_valid(mr),.mem_req_ready(mready),.mem_req_addr(ma),
.mem_req_we(mwe),.mem_req_wdata(mw),.mem_req_tag(mt),.mem_req_src(ms),
.mem_rsp_valid(mrv),.mem_rsp_tag(mrt),.mem_rsp_src(mrs),.mem_rsp_rdata(md));
tnca_dram_model #(.SRC_W(SW),.TAG_W(3),.LINES(TOTAL_WORDS/8),.LATENCY(11),.RANDOM_STALL(1),.REORDER(1)) ram(
.clk(clk),.rst_n(rst),.req_valid(mr),.req_ready(mready),.req_addr(ma),.req_we(mwe),
.req_wdata(mw),.req_tag(mt),.req_src(ms),.rsp_valid(mrv),.rsp_tag(mrt),.rsp_src(mrs),.rsp_rdata(md));
// Only clocks, external memory and file-to-port translation live here.
task response;
 begin
  fd=$fopen(rsp_path,"w");
  $fwrite(fd,"%0d %0d %0d %0d\n",seq,status,r0,r1);$fclose(fd);
 end
endtask
task host_word;
 input write_enable;input [63:0] address,value;
 begin
  @(negedge clk);hv=1;hw=write_enable;ha=address;hd=value;
  @(posedge clk);while(hr!==1'b1)@(posedge clk);
  @(negedge clk);hv=0;
  while(hrv!==1'b1)@(negedge clk);
  if(hs!=0)status=hs;
  read_data=hrd;
  @(negedge clk);
 end
endtask
task launch_grid;
 begin
  @(negedge clk);pc=a0;arg_base=a1;role_flags=a2[31:0];
  gx=a3[31:0];gy=a3[63:32];gz=a4[31:0];cores=a4[63:32];arg_stride=a5;id=seq;cv=1;
  @(posedge clk);while(cr!==1'b1)@(posedge clk);
  @(negedge clk);cv=0;
  while(dv!==1'b1 && !timed_out)@(negedge clk);
  // A deadline may leave an infinite program draining. Report its diagnostic,
  // poison the epoch, and keep host writes/launches refused until process reset.
  if(!dv && timed_out)begin status=2;poisoned=1;r0=0;r1=0;end
  else begin
   if(did!=seq)begin status=1;poisoned=1;end
   else begin status=ds;r0=dc[31:0];r1=dc[63:32];end
   ack=1;@(negedge clk);ack=0;
  end
 end
endtask
initial begin
 if(!$value$plusargs("req=%s",req_path) || !$value$plusargs("rsp=%s",rsp_path))begin $display("FATAL missing mailbox paths");$finish;end
 if($value$plusargs("up=%s",up_path));
 if($value$plusargs("down=%s",down_path));
 if($value$plusargs("pollcycles=%d",pollcycles));
 if($value$plusargs("deadline=%d",deadline));
 repeat(5)@(negedge clk);rst=1;
 seq=0;status=0;r0=TOTAL_WORDS;r1=N;response();
 $display("DEVICE READY words=%0d physical_cores=%0d",TOTAL_WORDS,N);$fflush();
 forever begin
  repeat(pollcycles)@(negedge clk);
  fd=$fopen(req_path,"r");
  if(fd!=0)begin
   n=$fscanf(fd,"%d %d %d %d %d %d %d %d",seq,op,a0,a1,a2,a3,a4,a5);$fclose(fd);
   if(n==8 && seq>served)begin
    served=seq;status=0;r0=0;r1=0;
    if(poisoned && op!=5 && op!=6 && op!=10)status=1;
    else case(op)
     1:begin
      if(a0<0 || a0%8!=0 || a1<=0 || a1>TOTAL_WORDS || a0/8>TOTAL_WORDS-a1)status=3;
      else begin
       // Validate the complete payload before modifying any device word.
       fd=$fopen(up_path,"r");
       if(fd==0)status=3;
       else begin
        for(j=0;j<a1;j=j+1)begin k=$fscanf(fd,"%h",word_data);if(k!=1)status=3;end
        $fclose(fd);
       end
       if(status==0)begin
        fd=$fopen(up_path,"r");
        for(j=0;j<a1;j=j+1)begin k=$fscanf(fd,"%h",word_data);host_word(1,a0+j*8,word_data);end
        $fclose(fd);
       end
      end
     end
     2:begin
      if(a0<0 || a0%8!=0 || a1<=0 || a1>TOTAL_WORDS || a0/8>TOTAL_WORDS-a1)status=3;
      else begin
       fd=$fopen(down_path,"w");
       if(fd==0)status=3;
       else begin
        for(j=0;j<a1;j=j+1)begin host_word(0,a0+j*8,0);$fwrite(fd,"%016x\n",read_data);end
        $fclose(fd);
       end
      end
     end
     4:begin if(busy || !idle)status=1;end
     5:begin response();$display("DEVICE SHUTDOWN after %0d requests",served);$finish;end
     6:begin r0=served;end
     8:begin r0=32;r1=16777216;end
     10:begin r0=N;r1=16777216;end // physical grid capability
     12:begin // synthesized per-core program counts, last accepted launch
      if(a0<0 || a0>=N)status=3;
      else begin r0=issued[a0*32+:32];r1=completed[a0*32+:32];end
     end
     13:begin // synthesized concurrency peak and per-core occupied cycles
      if(a0<0 || a0>=N)status=3;
      else begin r0=peak_active;r1=active_cycles[a0*64+:64];end
     end
     14:begin r0=1;r1=4;end // physical-x-chunk-v1, maximum four lanes
     11:begin
      if(a2[63:32]!=0 || (a2 & 64'h0ffffcf0)!=0 || a2[9:8]==3 || a4[63:32]==0 || (a4[63:32]>>N)!=0)status=3;
      else launch_grid();
     end
     default:status=3;
    endcase
    response();
   end
  end
 end
end
endmodule
