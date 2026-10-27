`timescale 1ns/1ps
module tb_stream_axi;
 parameter STALL=1,ERROR=0;
 parameter [63:0] PC_BASE=64'h100000000;
 localparam LINES=4096,Q=16;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg cv=0,dr=0;wire cr,busy,fault,dv;wire [3:0] status;wire [31:0] sum;
 reg [1:0] mode=0;reg [63:0] base=PC_BASE;
 reg [27:0] src=0,dst=0;reg [22:0] n=0;reg [31:0] pattern=0;
 wire [63:0] cycles,reads,writes;
 wire av,wav,wv,wl,rr,br,aid,waid;
 wire [63:0] aa,waa;wire [7:0] al,wal;wire [2:0] az,waz;wire [1:0] ab,wab;
 wire [511:0] wd;wire [63:0] ws;
 reg ar=0,war=0,wr=0,rv=0,bv=0,rl=0,rid=0,bid=0;
 reg [511:0] rd=0;reg [1:0] re=0,be=0;
 tnca_stream_axi dut(.clk(clk),.rst_n(rst_n),.cmd_valid(cv),.cmd_ready(cr),
  .cmd_mode(mode),.cmd_pc_base(base),.cmd_src(src),.cmd_dst(dst),.cmd_lines(n),.cmd_pattern(pattern),
  .busy(busy),.fault(fault),.done_valid(dv),.done_ready(dr),.done_status(status),.result_sum(sum),
  .cycles(cycles),.read_lines(reads),.write_lines(writes),
  .m_arvalid(av),.m_arready(ar),.m_araddr(aa),.m_arid(aid),.m_arlen(al),.m_arsize(az),.m_arburst(ab),
  .m_rvalid(rv),.m_rready(rr),.m_rdata(rd),.m_rid(rid),.m_rresp(re),.m_rlast(rl),
  .m_awvalid(wav),.m_awready(war),.m_awaddr(waa),.m_awid(waid),.m_awlen(wal),.m_awsize(waz),.m_awburst(wab),
  .m_wvalid(wv),.m_wready(wr),.m_wdata(wd),.m_wstrb(ws),.m_wlast(wl),
  .m_bvalid(bv),.m_bready(br),.m_bid(bid),.m_bresp(be));
 reg [511:0] memory[0:LINES-1],oracle[0:LINES-1];
 integer raddr[0:Q-1],rlen[0:Q-1],rdue[0:Q-1];
 integer waddr[0:Q-1],wlen[0:Q-1],bdue[0:Q-1];
 integer rh=0,rt=0,rc=0,rpos=0,wh=0,wt=0,wc=0,wpos=0,bh=0,bt=0,bc=0;
 integer tick=0,seed=4141,ar_count=0,aw_count=0,tests=0,peak_credits=0,peak_buffered=0;
 reg inject_error=ERROR!=0;
 reg ar_stall=0,aw_stall=0,w_stall=0;
 reg [63:0] old_aa,old_waa;reg [7:0] old_al,old_wal;reg [511:0] old_wd;reg old_wl;
 always @(posedge clk)begin
  if(!rst_n)begin
   rh=0;rt=0;rc=0;rpos=0;wh=0;wt=0;wc=0;wpos=0;bh=0;bt=0;bc=0;
   ar<=0;war<=0;wr<=0;rv<=0;bv<=0;ar_stall=0;aw_stall=0;w_stall=0;
  end else begin
   tick=tick+1;
   if(dut.reserved>128 || dut.write_bursts>8)$fatal(1,"stream credit limit");
   if(dut.reserved>peak_credits)peak_credits=dut.reserved;
   if(dut.copy_data.count>peak_buffered)peak_buffered=dut.copy_data.count;
   if(ar_stall && (!av || aa!==old_aa || al!==old_al))$fatal(1,"AR changed under backpressure");
   if(aw_stall && (!wav || waa!==old_waa || wal!==old_wal))$fatal(1,"AW changed under backpressure");
   if(w_stall && (!wv || wd!==old_wd || wl!==old_wl))$fatal(1,"W changed under backpressure");
   ar_stall=av&&!ar;old_aa=aa;old_al=al;
   aw_stall=wav&&!war;old_waa=waa;old_wal=wal;
   w_stall=wv&&!wr;old_wd=wd;old_wl=wl;
   if(av && ar)begin
    if(aid!=0 || az!=6 || ab!=1 || aa[5:0]!=0 || aa<PC_BASE ||
       aa-PC_BASE+(al+1)*64>LINES*64 || aa[11:0]+(al+1)*64>4096 || rc>=Q)
     $fatal(1,"AR geometry/range/queue");
    raddr[rt]=(aa-PC_BASE)/64;rlen[rt]=al+1;rdue[rt]=tick+25;
    rt=(rt+1)%Q;rc=rc+1;ar_count=ar_count+1;
   end
   if(wav && war)begin
    if(waid!=0 || waz!=6 || wab!=1 || waa[5:0]!=0 || waa<PC_BASE ||
       waa-PC_BASE+(wal+1)*64>LINES*64 || waa[11:0]+(wal+1)*64>4096 || wc>=Q)
     $fatal(1,"AW geometry/range/queue");
    waddr[wt]=(waa-PC_BASE)/64;wlen[wt]=wal+1;wt=(wt+1)%Q;wc=wc+1;aw_count=aw_count+1;
   end
   if(wv && wr)begin
    if(wc==0 || ws!==64'hffffffffffffffff || wl!=(wpos==wlen[wh]-1))$fatal(1,"W geometry/order");
    memory[waddr[wh]+wpos]=wd;
    if(wl)begin
     wpos=0;wh=(wh+1)%Q;wc=wc-1;
     if(bc>=Q)$fatal(1,"B queue overflow");
     bdue[bt]=tick+19;bt=(bt+1)%Q;bc=bc+1;
    end else wpos=wpos+1;
   end
   if(rv && rr)begin
    // The slave's own burst length, independent of injected RLAST.
    if(rpos==rlen[rh]-1)begin rpos=0;rh=(rh+1)%Q;rc=rc-1;end
    else rpos=rpos+1;
   end
   if(bv && br)begin bh=(bh+1)%Q;bc=bc-1;end
   if(!rv || rr)begin
    rv<=rc!=0 && rdue[rh]<=tick && (!STALL || (($random(seed)&32'h7fffffff)%100)>20);
    if(rc!=0)begin
     rd<=memory[raddr[rh]+rpos];rl<=(inject_error && ERROR==3) ? 1'b1 : rpos==rlen[rh]-1;
     re<=inject_error && ERROR==1 ? 2'b10 : 2'b00;rid<=inject_error && ERROR==4;
    end
   end
   if(!bv || br)begin
    bv<=bc!=0 && bdue[bh]<=tick;be<=inject_error && ERROR==2 ? 2'b10 : 2'b00;bid<=inject_error && ERROR==5;
   end
   ar<=rc<Q-2 && (!STALL || (($random(seed)&32'h7fffffff)%100)>35);
   war<=wc<Q-2 && (!STALL || (($random(seed)&32'h7fffffff)%100)>40);
   wr<=!STALL || (($random(seed)&32'h7fffffff)%100)>45;
   if(tick>100000)$fatal(1,"stream engine timeout");
  end
 end
 integer i,j;
 reg [31:0] expected_sum,word;
 reg [63:0] held_cycles;
 task command;
  input integer op,source,destination,count,expected_status;
  input [31:0] fill;
  integer before_ar,before_aw;
  begin
   expected_sum=0;
   if(expected_status==0)begin
    if(op==0)for(integer q=0;q<count;q=q+1)oracle[q]=memory[source+q];
    if(op==1)for(integer q=0;q<count;q=q+1)oracle[q]={16{fill}};
    if(op==2)for(integer q=0;q<count;q=q+1)for(integer w=0;w<16;w=w+1)
     expected_sum=expected_sum+memory[source+q][w*32+:32];
   end
   before_ar=ar_count;before_aw=aw_count;
   @(negedge clk);cv=1;mode=op;src=source*64;dst=destination*64;n=count;pattern=fill;
   @(posedge clk);while(!cr)@(posedge clk);
   @(negedge clk);cv=0;mode=3;src=1;dst=1;n=0;pattern=32'hdeadbeef;
   wait(dv || fault);@(negedge clk);
   if(fault || status!=expected_status)$fatal(1,"command completion op=%0d status=%0d fault=%0d",op,status,fault);
   held_cycles=cycles;
   repeat(3)@(negedge clk);
   if(!dv || cycles!==held_cycles || busy)$fatal(1,"completion did not hold until acknowledged");
   if(expected_status==0)begin
    if(reads!=(op==1 ? 0 : count) || writes!=(op==2 ? 0 : count))$fatal(1,"line counters");
    if(op==0 || op==1)for(integer q=0;q<count;q=q+1)
     if(memory[destination+q]!==oracle[q])$fatal(1,"copy/fill byte mismatch at line %0d",q);
    if(op==2 && sum!==expected_sum)$fatal(1,"uint32 reduction expected=%h got=%h",expected_sum,sum);
   end else if(ar_count!=before_ar || aw_count!=before_aw || reads || writes)$fatal(1,"invalid command had side effects");
   tests=tests+1;
   $display("STREAM COMMAND PASS mode=%0d lines=%0d status=%0d cycles=%0d read_lines=%0d write_lines=%0d sum=%h",op,count,status,cycles,reads,writes,sum);
   dr=1;@(negedge clk);dr=0;
  end
 endtask
 initial begin
  for(i=0;i<LINES;i=i+1)for(j=0;j<16;j=j+1)begin
   word=32'hf1234567^(i*32'h1020305+j*32'h31415927);memory[i][j*32+:32]=word;
  end
  repeat(4)@(negedge clk);rst_n=1;
  if(ERROR!=0)begin
   @(negedge clk);cv=1;mode=(ERROR==2 || ERROR==5) ? 1 : 2;src=0;dst=2048*64;n=128;
   @(posedge clk);while(!cr)@(posedge clk);
   @(negedge clk);cv=0;
   if(ERROR==6)begin force bv=1;@(negedge clk);release bv;end
   if(ERROR==7)begin force rv=1;@(negedge clk);release rv;end
   wait(fault);repeat(30)@(negedge clk);
   if(dv || cr || !busy)$fatal(1,"fault fabricated completion or readmission");
   rst_n=0;inject_error=0;repeat(4)@(negedge clk);rst_n=1;
   command(0,0,2048,130,0,0);
  end else begin
   command(2,0,0,0,0,0);
   command(0,1,2048,1,0,0);
   command(0,62,2110,130,0,0);
   command(0,0,2048,1024,0,0);
   command(1,0,2051,511,0,32'ha5a51234);
   command(2,2048,0,1024,0,0);
   command(2,0,0,1025,0,0);
   command(1,0,0,0,0,32'hdeadbeef);
   command(0,0,1,2,3,0);
   command(0,0,2048,4194305,3,0);
   command(2,4194303,0,2,3,0);
   command(3,0,0,1,3,0);
   base=PC_BASE+1;command(1,0,0,1,3,0);base=PC_BASE;
   // A byte-misaligned source is refused without traffic.
   @(negedge clk);mode=2;src=1;dst=0;n=1;cv=1;
   @(posedge clk);while(!cr)@(posedge clk);
   @(negedge clk);cv=0;wait(dv);@(negedge clk);
   if(status!=3 || reads || writes || fault)$fatal(1,"misaligned source not refused");
   dr=1;@(negedge clk);dr=0;tests=tests+1;
  end
  if(ERROR==0 && (peak_credits<=16 || (STALL && peak_buffered==0)))$fatal(1,"test did not exercise deep buffering");
  $display("STREAM AXI PASS stall=%0d error=%0d pc_base=%h tests=%0d peak_credits=%0d peak_buffered=%0d",STALL,ERROR,PC_BASE,tests,peak_credits,peak_buffered);$finish;
 end
endmodule
