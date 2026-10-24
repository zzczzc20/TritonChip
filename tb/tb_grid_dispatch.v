`timescale 1ns/1ps
module tb_grid_dispatch;
parameter N=3;
localparam CW=N>1?$clog2(N):1;
reg clk=0; always #5 clk=~clk;
reg rst_n=0,lv=0,dr=0,ga=0;
reg [31:0] gx=1,gy=1,gz=1;
reg [1:0] group_log2=0;
integer lane_count[0:N-1],lane,lanes;
reg [N-1:0] mask={N{1'b1}},ready=0,cd=0,ab=0;
reg [N*4-1:0] cs=0;
wire lr,dv; wire [N-1:0] cv,cancel,busy;
wire [N*32-1:0] px,py,pz,issued,completed;
wire [N*64-1:0] args;
wire [63:0] id,pc,did,cycles;wire [31:0] nx,ny,nz;
wire [3:0] status;wire [CW-1:0] fc;
wire [31:0] peak;
wire [N*64-1:0] occupied;
tnca_grid_dispatch #(.CORE_N(N)) dut(.clk(clk),.rst_n(rst_n),.launch_valid(lv),.launch_ready(lr),
.abort_valid(ga),.abort_status(4'd2),.launch_id(64'd71),.launch_pc(64'h100),.launch_arg_base(64'h1000),.launch_arg_stride(64'h100),
.launch_group_log2(group_log2),.launch_gx(gx),.launch_gy(gy),.launch_gz(gz),.launch_cores(mask),.cmd_valid(cv),.cmd_ready(ready),
.cmd_id(id),.cmd_pc(pc),.cmd_arg_base(args),.cmd_x(px),.cmd_y(py),.cmd_z(pz),
.cmd_nx(nx),.cmd_ny(ny),.cmd_nz(nz),.core_done(cd),.core_abort(ab),.core_status(cs),.cmd_cancel(cancel),
.done_valid(dv),.done_ready(dr),.done_status(status),.done_id(did),.done_cycles(cycles),.fault_core(fc),
.core_busy(busy),.issued(issued),.completed(completed),.peak_active(peak),.active_cycles(occupied));
// seen[] covers the largest grid run to completion here (>65,535 programs).
localparam SEEN=70000;
integer timer[0:N-1],seen[0:SEEN];
integer oracle_peak=0,oracle_active=0;
reg [63:0] oracle_cycles[0:N-1];
always @(posedge clk) begin
 if(!rst_n || (lv && lr)) begin
  oracle_peak=0;
  for(integer m=0;m<N;m=m+1)oracle_cycles[m]=0;
 end else if(!dv) begin
  oracle_active=0;
  for(integer m=0;m<N;m=m+1)if(busy[m])begin
   oracle_active=oracle_active+1;oracle_cycles[m]=oracle_cycles[m]+1;
  end
  if(oracle_active>oracle_peak)oracle_peak=oracle_active;
 end
end
integer tick=0,i,key,count=0,finish=0,inject=-1,fault_seen=0;
// Admission-window events (F06): adm_delay counts negedges after the
// accepting posedge; at 0 the event fires in that cycle (0 = stage A).
// adm_event 1 pulses the global abort for one cycle (a deadline pulse),
// 2 raises a stray completion from an idle core.
integer adm_delay=-1,adm_event=0;
reg pulse_ga=0;
reg [N-1:0] held=0;
reg [N*32-1:0] oldx,oldy,oldz;
always @(negedge clk) if(rst_n) begin
 tick=tick+1;cd=0;cs=0;ab=0;
 if(pulse_ga)begin ga=0;pulse_ga=0;end
 if(adm_delay==0 && adm_event==1)begin ga=1;pulse_ga=1;fault_seen=1;end
 if(adm_delay==0 && adm_event==2)begin cd[0]=1;fault_seen=1;end
 if(adm_delay>=0)adm_delay=adm_delay-1;
 if(inject==-2 && count>=N)begin ga=1;fault_seen=1;end
 for(i=0;i<N;i=i+1) begin
  ready[i]=((tick+i*3)%7<3);
  if(timer[i]>0) begin
   timer[i]=timer[i]-1;
   if(timer[i]==3 && i==inject && !fault_seen) begin ab[i]=1;cs[i*4+:4]=5;fault_seen=1;end
   if(timer[i]==0) cd[i]=1;
  end
 end
end
always @(posedge clk) if(rst_n) begin
 for(i=0;i<N;i=i+1) begin
  if(held[i] && !cancel[i] && (!cv[i] || px[i*32+:32]!=oldx[i*32+:32] || py[i*32+:32]!=oldy[i*32+:32] || pz[i*32+:32]!=oldz[i*32+:32])) $fatal(1,"unstable stalled command");
  if(cv[i]&&ready[i]) begin
   if(timer[i]!=0 || !mask[i] || fault_seen) $fatal(1,"invalid issue");
   key=px[i*32+:32]+gx*(py[i*32+:32]+gy*pz[i*32+:32]);
   if(key>=gx*gy*gz || seen[key]!=0) $fatal(1,"duplicate or outside grid PID %d",key);
   lanes=(1<<group_log2);
   if(px[i*32+:32]+lanes>gx)lanes=gx-px[i*32+:32];
   if(px[i*32+:32]%(1<<group_log2)!=0)$fatal(1,"misaligned program chunk");
   for(lane=0;lane<lanes;lane=lane+1)begin
    if(seen[key+lane]!=0)$fatal(1,"duplicate chunk lane");
    seen[key+lane]=1;
   end
   lane_count[i]=lanes;count=count+lanes;
   if(args[i*64+:64]!=64'h1000+i*256 || id!=71 || pc!=256 || nx!=gx || ny!=gy || nz!=gz) $fatal(1,"metadata mismatch");
   timer[i]=7+(key%11);
  end
  if(cd[i]) finish=finish+lane_count[i];
 end
 held=cv & ~ready;oldx=px;oldy=py;oldz=pz;
 if(dv && (busy!=0 || finish!=count)) $fatal(1,"early completion");
end
// Launch and hold launch_valid through the whole admission window: a held
// request must not be accepted twice, launch_ready must stay low, nothing
// may issue, and the decision lands exactly ADM cycles after acceptance.
localparam ADM=4;
task admit;
 input integer ax,ay,az;
 input integer event_kind,event_stage;
 begin
 @(negedge clk);gx=ax;gy=ay;gz=az;lv=1;
 @(posedge clk);if(!lr)$fatal(1,"launch not ready");
 if(event_kind!=0)begin adm_event=event_kind;adm_delay=event_stage;end
 for(integer k=1;k<=ADM;k=k+1)begin
  @(negedge clk);
  if(lr || cv!=0 || busy!=0 || dv || dut.running)$fatal(1,"admission window not closed at stage %0d",k);
  if(dut.adm!==(4'b0001<<(k-1)))$fatal(1,"admission stage %0d not one-hot",k);
 end
 lv=0;
 @(negedge clk);
 if(dut.adm!=0 || !(dut.running || dv))$fatal(1,"no decision after %0d admission cycles",ADM);
 adm_event=0;adm_delay=-1;
 end
endtask
task finish_grid;
 input integer expect_status;
 input integer max_cycles;
 integer w;
 begin
 w=0;while(!dv && w<max_cycles)begin @(negedge clk);w=w+1;end
 if(!dv || status!=expect_status || busy!=0)$fatal(1,"grid ended with status %0d, expected %0d",status,expect_status);
 if(finish!=count)$fatal(1,"completion before drain");
 repeat(2)begin @(negedge clk);if(!dv || lr)$fatal(1,"completion not retained");end
 dr=1;@(negedge clk);dr=0;
 end
endtask
task invalid_grid;
 input integer ax,ay,az;
 begin
 count=0;finish=0;fault_seen=1;
 admit(ax,ay,az,0,0);
 if(!dv || status!=3 || cv!=0 || busy!=0 || fc!=0 || cycles!=0)$fatal(1,"invalid grid accepted");
 if(peak!=0 || occupied!=0)$fatal(1,"invalid grid telemetry not reset");
 dr=1;@(negedge clk);dr=0;fault_seen=0;
 $display("PASS N=%0d invalid grid %0dx%0dx%0d refused",N,ax,ay,az);
 end
endtask
task zero_grid;
 input integer ax,ay,az;
 begin
 count=0;finish=0;fault_seen=1;
 admit(ax,ay,az,0,0);
 if(!dv || status!=0 || cv!=0 || busy!=0 || cycles!=0)$fatal(1,"empty grid not completed");
 dr=1;@(negedge clk);dr=0;fault_seen=0;
 $display("PASS N=%0d empty grid %0dx%0dx%0d completed",N,ax,ay,az);
 end
endtask
// The largest admissible grids: admitted with the exact chunk count, then
// stopped by an abort after the first commands drain.
task admit_large;
 input integer ax,ay,az,expect_remaining;
 integer k;
 begin
 for(k=0;k<N;k=k+1)begin timer[k]=0;lane_count[k]=0;end
 for(k=0;k<=SEEN;k=k+1)seen[k]=0;
 count=0;finish=0;fault_seen=0;inject=-1;held=0;
 admit(ax,ay,az,0,0);
 if(!dut.running || dut.remaining!=expect_remaining)$fatal(1,"large grid %0dx%0dx%0d remaining %0d, expected %0d",ax,ay,az,dut.remaining,expect_remaining);
 repeat(40)@(negedge clk);
 ga=1;fault_seen=1;@(negedge clk);
 finish_grid(2,1000);ga=0;fault_seen=0;
 if(count==0)$fatal(1,"large grid issued nothing before the abort");
 $display("PASS N=%0d large grid %0dx%0dx%0d admitted remaining=%0d, %0d programs before abort",N,ax,ay,az,expect_remaining,count);
 end
endtask
// An abort pulse or a stray completion inside the admission window is
// recorded as in the first running cycle: nothing issues, and a refused or
// empty grid still reports its acceptance-edge decision.
task admission_event;
 input integer ax,ay,az,kind,stage,expect_status;
 integer k;
 begin
 for(k=0;k<N;k=k+1)begin timer[k]=0;lane_count[k]=0;end
 count=0;finish=0;fault_seen=0;inject=-1;held=0;
 admit(ax,ay,az,kind,stage);
 finish_grid(expect_status,50);
 if(count!=0 || issued!=0)$fatal(1,"command issued after an admission-window event");
 if(fc!=0)$fatal(1,"admission-window fault core");
 $display("PASS N=%0d event %0d at admission stage %0d, grid %0dx%0dx%0d status=%0d",N,kind,stage,ax,ay,az,expect_status);
 end
endtask
task run;
 input integer ax,ay,az,badcore;
 integer k,deadline,expected,sum_i,sum_c;
 begin
 @(negedge clk);gx=ax;gy=ay;gz=az;inject=badcore;fault_seen=0;ga=0;count=0;finish=0;held=0;
 for(k=0;k<N;k=k+1) timer[k]=0;
 for(k=0;k<=SEEN;k=k+1) seen[k]=0;
 for(k=0;k<N;k=k+1) lane_count[k]=0;
 lv=1;@(posedge clk);if(!lr)$fatal(1,"launch not ready");@(negedge clk);lv=0;
 deadline=0;while(!dv && deadline<3000000) begin @(negedge clk);deadline=deadline+1;end
 if(!dv)$fatal(1,"timeout");
 expected=ax*ay*az;
 if(badcore==-1 && (status!=0 || count!=expected)) $fatal(1,"incomplete grid %d/%d status %d",count,expected,status);
 if(badcore==-2 && (status!=2 || count==0 || count>=expected))$fatal(1,"global abort not drained");
 if(badcore>=0 && (status!=5 || fc!=badcore || count>=expected))$fatal(1,"fault not drained");
 sum_i=0;sum_c=0;for(k=0;k<N;k=k+1) begin sum_i=sum_i+issued[k*32+:32];sum_c=sum_c+completed[k*32+:32];end
 if(sum_i!=count || sum_c!=finish)$fatal(1,"counter mismatch");
 if(peak!=oracle_peak)$fatal(1,"peak activity mismatch");
 for(k=0;k<N;k=k+1)if(occupied[k*64+:64]!=oracle_cycles[k])$fatal(1,"occupied cycle mismatch");
 repeat(4)begin @(negedge clk);if(!dv || lr || did!=71)$fatal(1,"completion not retained");end
 dr=1;@(negedge clk);dr=0;
 $display("PASS N=%0d grid=%0dx%0dx%0d issued=%0d fault=%0d",N,ax,ay,az,count,badcore);
 end
endtask
initial begin
 for(i=0;i<N;i=i+1)timer[i]=0;
 repeat(3)@(negedge clk);rst_n=1;
 invalid_grid(16777217,1,1);invalid_grid(-1,-1,-1);
 mask=0;invalid_grid(1,1,1);zero_grid(0,5,5);mask={N{1'b1}};
 // F06 dimension boundaries: 2^24 programs is the largest admissible grid;
 // one more program, or any product that wraps 32 or 64 bits, is refused.
 invalid_grid(4097,4096,1);invalid_grid(4096,4097,1);invalid_grid(1,4096,4097);
 invalid_grid(65537,256,1);invalid_grid(16777216,2,1);invalid_grid(1,1,16777217);
 invalid_grid(65536,65536,1);invalid_grid(2,-2147483648,1);invalid_grid(65536,65536,65536);
 invalid_grid(-1,1,1);invalid_grid(1,-1,1);invalid_grid(1,1,-1);
 zero_grid(0,-1,-1);zero_grid(-1,0,-1);zero_grid(-1,-1,0);zero_grid(0,0,0);
 group_log2=3;invalid_grid(1,1,1);invalid_grid(0,1,1);group_log2=0;
 admit_large(16777216,1,1,16777216);admit_large(4096,4096,1,16777216);
 admit_large(256,256,256,16777216);admit_large(1,1,16777216,16777216);
 admit_large(65536,256,1,16777216);admit_large(70001,239,1,16730239);
 group_log2=2;admit_large(16777216,1,1,4194304);admit_large(4095,4096,1,4194304);
 admit_large(4097,1,4095,4197375);invalid_grid(16777217,1,1);group_log2=0;
 // Deadline/abort and stray completions at every admission stage.
 for(integer st=0;st<ADM;st=st+1)begin
  admission_event(13,7,5,1,st,2);admission_event(16777217,1,1,1,st,3);
  admission_event(0,7,5,1,st,0);admission_event(13,7,5,2,st,1);
 end
 run(1,1,1,-1);run(19,1,1,-1);run(13,7,5,-1);run(6145,1,1,-1);run(0,3,2,-1);
 for(integer f=0;f<N;f=f+1)run(101,3,2,f);
 mask=1;run(47,2,1,-1);mask={N{1'b1}};
 run(101,3,2,-2);run(23,2,1,-1);
 group_log2=1;run(5,2,3,-1);run(6145,1,1,-1);
 group_log2=2;run(17,3,2,-1);run(1,1,1,-1);
 // Grids above 65,535 programs run to completion through bounded admission.
 group_log2=0;run(65537,1,1,-1);run(257,256,1,-1);
 group_log2=2;run(65541,1,1,-1);
 group_log2=0;$display("GRID DISPATCH PASS");$finish;
end
endmodule
