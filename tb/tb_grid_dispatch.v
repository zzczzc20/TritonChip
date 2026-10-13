`timescale 1ns/1ps
module tb_grid_dispatch;
parameter N=3;
localparam CW=N>1?$clog2(N):1;
reg clk=0; always #5 clk=~clk;
reg rst_n=0,lv=0,dr=0,ga=0;
reg [31:0] gx=1,gy=1,gz=1;
reg [N-1:0] mask={N{1'b1}},ready=0,cd=0,ab=0;
reg [N*4-1:0] cs=0;
wire lr,dv; wire [N-1:0] cv,cancel,busy;
wire [N*32-1:0] px,py,pz,issued,completed;
wire [N*64-1:0] args;
wire [63:0] id,pc,did,cycles;wire [31:0] nx,ny,nz;
wire [3:0] status;wire [CW-1:0] fc;
tnca_grid_dispatch #(.CORE_N(N)) dut(.clk(clk),.rst_n(rst_n),.launch_valid(lv),.launch_ready(lr),
.abort_valid(ga),.abort_status(4'd2),.launch_id(64'd71),.launch_pc(64'h100),.launch_arg_base(64'h1000),.launch_arg_stride(64'h100),
.launch_gx(gx),.launch_gy(gy),.launch_gz(gz),.launch_cores(mask),.cmd_valid(cv),.cmd_ready(ready),
.cmd_id(id),.cmd_pc(pc),.cmd_arg_base(args),.cmd_x(px),.cmd_y(py),.cmd_z(pz),
.cmd_nx(nx),.cmd_ny(ny),.cmd_nz(nz),.core_done(cd),.core_abort(ab),.core_status(cs),.cmd_cancel(cancel),
.done_valid(dv),.done_ready(dr),.done_status(status),.done_id(did),.done_cycles(cycles),.fault_core(fc),
.core_busy(busy),.issued(issued),.completed(completed));
integer timer[0:N-1],seen[0:7000];
integer tick=0,i,key,count=0,finish=0,inject=-1,fault_seen=0;
reg [N-1:0] held=0;
reg [N*32-1:0] oldx,oldy,oldz;
always @(negedge clk) if(rst_n) begin
 tick=tick+1;cd=0;cs=0;ab=0;
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
   seen[key]=1;count=count+1;
   if(args[i*64+:64]!=64'h1000+i*256 || id!=71 || pc!=256 || nx!=gx || ny!=gy || nz!=gz) $fatal(1,"metadata mismatch");
   timer[i]=7+(key%11);
  end
  if(cd[i]) finish=finish+1;
 end
 held=cv & ~ready;oldx=px;oldy=py;oldz=pz;
 if(dv && (busy!=0 || finish!=count)) $fatal(1,"early completion");
end
task invalid_grid;
 input integer ax,ay,az;
 begin
 @(negedge clk);gx=ax;gy=ay;gz=az;lv=1;
 @(negedge clk);lv=0;
 if(!dv || status!=3 || cv!=0 || busy!=0)$fatal(1,"invalid grid accepted");
 dr=1;@(negedge clk);dr=0;
 $display("PASS N=%0d invalid grid refused",N);
 end
endtask
task run;
 input integer ax,ay,az,badcore;
 integer k,deadline,expected,sum_i,sum_c;
 begin
 @(negedge clk);gx=ax;gy=ay;gz=az;inject=badcore;fault_seen=0;ga=0;count=0;finish=0;held=0;
 for(k=0;k<N;k=k+1) timer[k]=0;
 for(k=0;k<7001;k=k+1) seen[k]=0;
 lv=1;@(posedge clk);if(!lr)$fatal(1,"launch not ready");@(negedge clk);lv=0;
 deadline=0;while(!dv && deadline<100000) begin @(negedge clk);deadline=deadline+1;end
 if(!dv)$fatal(1,"timeout");
 expected=ax*ay*az;
 if(badcore==-1 && (status!=0 || count!=expected)) $fatal(1,"incomplete grid %d/%d status %d",count,expected,status);
 if(badcore==-2 && (status!=2 || count==0 || count>=expected))$fatal(1,"global abort not drained");
 if(badcore>=0 && (status!=5 || fc!=badcore || count>=expected))$fatal(1,"fault not drained");
 sum_i=0;sum_c=0;for(k=0;k<N;k=k+1) begin sum_i=sum_i+issued[k*32+:32];sum_c=sum_c+completed[k*32+:32];end
 if(sum_i!=count || sum_c!=finish)$fatal(1,"counter mismatch");
 repeat(4)begin @(negedge clk);if(!dv || lr || did!=71)$fatal(1,"completion not retained");end
 dr=1;@(negedge clk);dr=0;
 $display("PASS N=%0d grid=%0dx%0dx%0d issued=%0d fault=%0d",N,ax,ay,az,count,badcore);
 end
endtask
initial begin
 for(i=0;i<N;i=i+1)timer[i]=0;
 repeat(3)@(negedge clk);rst_n=1;
 invalid_grid(16777217,1,1);invalid_grid(-1,-1,-1);
 mask=0;invalid_grid(1,1,1);mask={N{1'b1}};
 run(1,1,1,-1);run(19,1,1,-1);run(13,7,5,-1);run(6145,1,1,-1);run(0,3,2,-1);
 for(integer f=0;f<N;f=f+1)run(101,3,2,f);
 mask=1;run(47,2,1,-1);mask={N{1'b1}};
 run(101,3,2,-2);run(23,2,1,-1);$display("GRID DISPATCH PASS");$finish;
end
endmodule
