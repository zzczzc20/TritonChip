`timescale 1ns/1ps
module tb_stream_bank;
 parameter LANES=32;
 reg clk=0,rst_n=0;always #5 clk=~clk;
 reg [11:0] address=0;reg [63:0] value=0;reg we=0;
 wire [63:0] got;wire busy,fault;
 wire [LANES-1:0] av,ar,rv,rr,rid,rl,wav,war,wv,wr,wl,bv,br,bid,aid,waid;
 wire [LANES*64-1:0] aa,waa,ws;wire [LANES*8-1:0] al,wal;
 wire [LANES*3-1:0] az,waz;wire [LANES*2-1:0] ab,wab;
 wire [LANES*512-1:0] rd,wd;
 tnca_stream_bank #(.LANES(LANES)) dut(.clk(clk),.rst_n(rst_n),.reg_addr(address),.reg_wdata(value),
  .reg_we(we),.reg_rdata(got),.busy(busy),.fault(fault),
  .m_arvalid(av),.m_arready(ar),.m_araddr(aa),.m_arid(aid),.m_arlen(al),.m_arsize(az),.m_arburst(ab),
  .m_rvalid(rv),.m_rready(rr),.m_rdata(rd),.m_rid(rid),.m_rresp({LANES*2{1'b0}}),.m_rlast(rl),
  .m_awvalid(wav),.m_awready(war),.m_awaddr(waa),.m_awid(waid),.m_awlen(wal),.m_awsize(waz),.m_awburst(wab),
  .m_wvalid(wv),.m_wready(wr),.m_wdata(wd),.m_wstrb(ws),.m_wlast(wl),
  .m_bvalid(bv),.m_bready(br),.m_bid(bid),.m_bresp({LANES*2{1'b0}}));
 integer tick=0;always @(posedge clk)tick<=tick+1;
 reg new_window=0;
 integer phase_mode=0;reg [31:0] phase_pattern=0;
 integer read_count[0:LANES-1],write_count[0:LANES-1],read_issued[0:LANES-1],write_issued[0:LANES-1];
 generate for(genvar p=0;p<LANES;p=p+1)begin:slave
  reg rhave=0,whave=0,bpending=0;
  integer rfirst=0,rpos=0,rlen=0,rdue=0,wlen=0,wpos=0,bdue=0;
  wire [31:0] read_word=32'h12340000+p*32'h10101+rfirst+rpos;
  wire [31:0] copy_word=32'h12340000+p*32'h10101+(p*64+62)+write_count[p];
  assign ar[p]=!rhave && (tick+p)%3!=0;
  assign rv[p]=rhave && tick>=rdue;assign rl[p]=rpos==rlen-1;assign rid[p]=0;
  assign rd[p*512+:512]={16{read_word}};
  assign war[p]=!whave && !bpending && (tick+p)%4!=0;
  assign wr[p]=whave && (tick+p)%3!=0;
  assign bv[p]=bpending && tick>=bdue;assign bid[p]=0;
  always @(posedge clk)begin
   if(!rst_n)begin
    rhave<=0;whave<=0;bpending<=0;rpos<=0;wpos<=0;
    read_count[p]<=0;write_count[p]<=0;read_issued[p]<=0;write_issued[p]<=0;
   end else begin
    if(new_window)begin
     if(rhave || whave || bpending)$fatal(1,"new window before physical drain");
     read_count[p]<=0;write_count[p]<=0;read_issued[p]<=0;write_issued[p]<=0;
    end
    if(av[p] && ar[p])begin
     if(aa[p*64+:64]!==((64'(p)<<28)+64*(p*64+62+read_issued[p])) || aid[p]!=0 ||
        az[p*3+:3]!=6 || ab[p*2+:2]!=1 || aa[p*64+:12]+(al[p*8+:8]+1)*64>4096)
      $fatal(1,"bank AR mapping/geometry lane=%0d",p);
     rhave<=1;rpos<=0;rlen<=al[p*8+:8]+1;rfirst<=aa[p*64+6+:22];rdue<=tick+7+p;
     read_issued[p]<=read_issued[p]+al[p*8+:8]+1;
    end
    if(rv[p] && rr[p])begin
     read_count[p]<=read_count[p]+1;
     if(rl[p])rhave<=0;else rpos<=rpos+1;
    end
    if(wav[p] && war[p])begin
     if(waa[p*64+:64]!==((64'(p)<<28)+64*(4096+p*64+61+write_issued[p])) || waid[p]!=0 ||
        waz[p*3+:3]!=6 || wab[p*2+:2]!=1 || waa[p*64+:12]+(wal[p*8+:8]+1)*64>4096)
      $fatal(1,"bank AW mapping/geometry lane=%0d",p);
     whave<=1;wpos<=0;wlen<=wal[p*8+:8]+1;write_issued[p]<=write_issued[p]+wal[p*8+:8]+1;
    end
    if(wv[p] && wr[p])begin
     if(ws[p*64+:64]!==64'hffffffffffffffff || wl[p]!=(wpos==wlen-1))$fatal(1,"bank W geometry");
     if(wd[p*512+:512]!==(phase_mode==1 ? {16{phase_pattern}} : {16{copy_word}}))
      $fatal(1,"bank write payload lane=%0d beat=%0d",p,write_count[p]);
     write_count[p]<=write_count[p]+1;
     if(wl[p])begin whave<=0;bpending<=1;bdue<=tick+11+p;end else wpos<=wpos+1;
    end
    if(bv[p] && br[p])bpending<=0;
   end
  end
 end endgenerate
 task put;
  input [11:0] a;input [63:0] v;
  begin @(negedge clk);address=a;value=v;we=1;@(negedge clk);we=0;end
 endtask
 task check;
  input [11:0] a;input [63:0] v;
  begin address=a;#1;if(got!==v)$fatal(1,"bank register %h expected %h got %h",a,v,got);end
 endtask
 integer request_number=0,tests=0,i,j;reg [31:0] expected_sum,mask;
 reg [63:0] expected_reads,expected_writes,held_cycles;
 task run;
  input integer op;input [31:0] active_mask;input integer bad;
  begin
   request_number=request_number+1;phase_mode=op;phase_pattern=32'hdeadcafe;
   expected_sum=0;expected_reads=0;expected_writes=0;
   put(12'h040,(64'(op)<<32)|phase_pattern);put(12'h048,active_mask);
   for(i=0;i<LANES;i=i+1)begin
    put(12'h050,i);put(12'h058,(i*64+62)*64);put(12'h060,(4096+i*64+61)*64);put(12'h068,129+i);
    if(active_mask[i] && !bad)begin
     if(op!=1)expected_reads=expected_reads+129+i;
     if(op!=2)expected_writes=expected_writes+129+i;
     if(op==2)for(j=0;j<129+i;j=j+1)
      expected_sum=expected_sum+16*(32'h12340000+i*32'h10101+(i*64+62)+j);
    end
   end
   // The last enabled lane is invalid: the bank must not start earlier lanes.
   if(bad)begin put(12'h050,LANES-1);put(12'h068,23'd4194305);end
   put(12'h010,request_number);
   @(negedge clk);new_window=1;@(negedge clk);new_window=0;
   put(12'h018,1);
   if(!bad && active_mask!=0)begin
    if(!busy)$fatal(1,"bank did not start");
    put(12'h050,0);put(12'h058,1);put(12'h040,64'h300000001);put(12'h010,999);
    check(12'h050,LANES-1);check(12'h058,((LANES-1)*64+62)*64);
   end
   wait(!busy);@(negedge clk);
   if(fault)$fatal(1,"unexpected bank fault");
   check(12'h028,request_number);check(12'h030,bad?3:0);check(12'h070,expected_sum);
   check(12'h078,expected_reads);check(12'h080,expected_writes);check(12'h010,request_number);
   address=12'h038;#1;held_cycles=got;repeat(4)@(negedge clk);check(12'h038,held_cycles);
   for(i=0;i<LANES;i=i+1)begin
    if(read_count[i]!==((!bad && active_mask[i] && op!=1)?129+i:0))$fatal(1,"lane read accounting");
    if(write_count[i]!==((!bad && active_mask[i] && op!=2)?129+i:0))$fatal(1,"lane write accounting");
   end
   // A stale doorbell cannot replay a completed command.
   put(12'h018,1);if(busy)$fatal(1,"stale stream request_number replayed");
   tests=tests+1;
   $display("STREAM BANK COMMAND PASS lanes=%0d mode=%0d mask=%h bad=%0d cycles=%0d",LANES,op,active_mask,bad,held_cycles);
  end
 endtask
 initial begin
  repeat(4)@(negedge clk);rst_n=1;
  check(0,64'h52343153424e4b01);
  put(12'h088,64'h200000000);put(12'h090,0);
  mask=32'hffffffff>>(32-LANES);
  run(1,mask,0);run(0,mask & 32'haaaaaaaa,0);run(2,mask,0);run(1,0,0);run(0,mask,1);
  // A stray response faults the bank and blocks every later admission.
  @(negedge clk);force rv[LANES-1]=1;
  @(negedge clk);release rv[LANES-1];
  if(!fault || busy)$fatal(1,"bank stray response did not latch fault");
  put(12'h010,999);put(12'h018,1);check(12'h010,request_number);
  if(busy || |av || |wav)$fatal(1,"faulted bank admitted work");
  rst_n=0;repeat(4)@(negedge clk);rst_n=1;
  put(12'h088,64'h200000000);put(12'h090,0);
  run(1,mask,0);
  $display("STREAM BANK AXI PASS lanes=%0d tests=%0d",LANES,tests);$finish;
 end
 initial begin #5000000;$fatal(1,"stream bank timeout");end
endmodule
