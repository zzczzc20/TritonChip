`timescale 1ns/1ps
module tb_lsu_line;
 parameter SLOTS=8,TAG_W=3,STALL=1,STREAM=0,SEED=1,NACC=512;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0;
 integer cycle=0,pc=0,nreq=0,ncpl=0,seed=SEED,ms=SEED+31;
 integer first=-1,last=-1;
 reg [1:0] sizes[0:NACC-1];
 reg stores[0:NACC-1],signs[0:NACC-1];
 reg [63:0] addresses[0:NACC-1];
 reg [511:0] vectors[0:NACC-1];
 reg [15:0] masks[0:NACC-1];
 reg [511:0] expected[0:NACC-1];
 reg [NACC-1:0] completed=0,expect_valid=0;
 reg [7:0] actual_mem[0:8191],oracle[0:8191];
 reg [SLOTS-1:0] pending=0;
 reg [511:0] response[0:SLOTS-1];
 integer due[0:SLOTS-1];
 wire eligible,ready,quiet,cv,mv,mwe;
 reg mready=0,ack=0,rv=0;
 reg [TAG_W-1:0] rt=0;
 reg [511:0] rd=0;
 wire [TAG_W-1:0] mt;
 wire [63:0] ma,be;
 wire [511:0] mw,cd;
 wire [7:0] ci;
 wire [1:0] cc;
 wire [2:0] dc;
 wire [15:0] cm;
 wire [31:0] accepted,retired,rstall,sstall,peak;
 reg [3:0] invalid=0;
 wire [31:0] p=pc<NACC?pc:NACC-1;
 wire reqvalid=pc<NACC || invalid!=0;
 wire [63:0] addr=invalid==5?64'h7fff0000:invalid==6?64'h200000000:invalid==4?64'h103f:addresses[p];
 wire [2:0] size=invalid==4?3'd2:invalid==3?3'd3:{1'b0,sizes[p]};
 tnca_lsu_line #(.SLOTS(SLOTS),.TAG_W(TAG_W)) dut(
  .clk(clk),.rst_n(rst_n),.req_valid(reqvalid),.req_eligible(eligible),.req_ready(ready),.quiescent(quiet),
  .req_store(stores[p]),.req_vector(invalid!=1),.req_indexed(invalid==2),.req_amo(invalid==7),
  .req_size(size),.req_signed(signs[p]),.req_hint(2'd2),.req_base(addr),.req_offset(64'd0),
  .req_vdata(vectors[p]),.req_mask(masks[p]),.req_rctx(p[1:0]),.req_dst_class(3'd2),.req_dst_idx(p[7:0]),
  .cpl_valid(cv),.cpl_ack(ack),.cpl_rctx(cc),.cpl_dst_class(dc),.cpl_dst_idx(ci),.cpl_vdata(cd),.cpl_lane_en(cm),
  .mem_req_valid(mv),.mem_req_ready(mready),.mem_req_addr(ma),.mem_req_we(mwe),.mem_req_be(be),
  .mem_req_wdata(mw),.mem_req_tag(mt),.mem_rsp_valid(rv),.mem_rsp_tag(rt),.mem_rsp_rdata(rd),
  .perf_accepted(accepted),.perf_completed(retired),.perf_request_stall(rstall),.perf_slot_stall(sstall),.perf_peak(peak));
 integer i,j,l,b,k,id,nbytes,byteaddr,chosen;
 reg [63:0] wanted_mask;
 reg [511:0] wanted_data,snapshot;
 reg [31:0] value;
 reg [SLOTS-1:0] responded=0;
 initial begin
  for(i=0;i<8192;i=i+1)begin actual_mem[i]=(i*31+13)%256;oracle[i]=actual_mem[i];end
  for(i=0;i<NACC;i=i+1)begin
   sizes[i]=STREAM?2:$unsigned($random(seed))%3;
   stores[i]=STREAM?0:$random(seed);signs[i]=$random(seed);
   addresses[i]=4096+($unsigned($random(seed))%32)*64;
   addresses[i]=addresses[i]+(STREAM?0:$unsigned($random(seed))%(65-(16<<sizes[i])));
   masks[i]=STREAM?16'hffff:$random(seed);
   if(!STREAM && i%19==0)masks[i]=0;
   for(l=0;l<16;l=l+1)vectors[i][l*32+:32]=$random(seed);
  end
 end
 // Drive independent memory and RF backpressure and one possibly reordered reply.
 always @(negedge clk) if(rst_n)begin
  mready=!STALL || $unsigned($random(ms))%5!=0;
  ack=!STALL || $unsigned($random(ms))%3!=0;
  rv=0;chosen=-1;
  for(k=0;k<SLOTS;k=k+1) if(pending[k] && due[k]<=cycle &&
       (chosen<0 || (STALL && $unsigned($random(ms))%2==0)))chosen=k;
  if(chosen>=0)begin rv=1;rt=chosen;rd=response[chosen];pending[chosen]=0;end
 end
 always @(posedge clk) if(rst_n)begin
  cycle<=cycle+1;
  if(reqvalid && eligible && ready)begin
   if(invalid!=0)$fatal(1,"unsupported command admitted");
   if(first<0)first=cycle;last=cycle;
   if(masks[pc]==0)begin expected[pc]=0;expect_valid[pc]=1;end
   pc<=pc+1;
  end
  if(mv && mready)begin
   while(nreq<NACC && masks[nreq]==0)nreq=nreq+1;
   if(nreq>=NACC || pending[mt])$fatal(1,"extra request or live response tag reused");
   id=nreq;nreq=nreq+1;nbytes=1<<sizes[id];wanted_data=0;wanted_mask=0;expected[id]=0;
   if(ma!==(addresses[id]&~64'd63) || mwe!==stores[id])$fatal(1,"line request metadata");
   for(l=0;l<16;l=l+1)if(masks[id][l])begin
    byteaddr=addresses[id]+l*nbytes;value=0;
    for(b=0;b<nbytes;b=b+1)begin
     value=value | ({24'd0,oracle[byteaddr+b]}<<(8*b));
     wanted_mask[byteaddr+b-ma]=1;
     wanted_data[(byteaddr+b-ma)*8+:8]=vectors[id][l*32+b*8+:8];
     if(stores[id])oracle[byteaddr+b]=vectors[id][l*32+b*8+:8];
    end
    if(signs[id] && nbytes<4 && (value & (32'd1<<(8*nbytes-1))))value=value | (~32'd0<<(8*nbytes));
    if(!stores[id])expected[id][l*32+:32]=value;
   end
   expect_valid[id]=1;
   if(be!==wanted_mask)$fatal(1,"line mask mismatch");
   snapshot=0;
   for(b=0;b<64;b=b+1)begin
    snapshot[b*8+:8]=actual_mem[ma+b];
    if(mwe && be[b])begin
     if(mw[b*8+:8]!==wanted_data[b*8+:8])$fatal(1,"store packing mismatch");
     actual_mem[ma+b]=mw[b*8+:8];
    end
   end
   pending[mt]=1;response[mt]=snapshot;due[mt]=cycle+(STALL?1+$unsigned($random(ms))%32:1);
  end
  if(cv && ack)begin
   id=-1;
   for(j=0;j<pc;j=j+1) if(!completed[j] && expect_valid[j] && j%256==ci)begin
    if(id>=0)$fatal(1,"ambiguous completion identity");id=j;
   end
   if(id<0 || cd!==expected[id] || cc!==id%4 || dc!==(stores[id]?3'd0:3'd2) || cm!==(stores[id]?16'd0:masks[id]))
    $fatal(1,"completion mismatch id=%0d pc=%0d tag=%0d got=%h expected=%h",id,pc,ci,cd,expected[id]);
   completed[id]=1;ncpl=ncpl+1;
  end
  if(cycle>100000)$fatal(1,"line LSU timeout pc=%0d complete=%0d",pc,ncpl);
 end
 reg stalled=0,held=0;
 reg [640+TAG_W:0] request_hold;
 reg [540:0] complete_hold;
 always @(posedge clk)if(rst_n)begin
  if(stalled && (!mv || {ma,mwe,be,mw,mt}!==request_hold))$fatal(1,"request changed under backpressure");
  stalled=mv && !mready;request_hold={ma,mwe,be,mw,mt};
  if(held && (!cv || {cc,dc,ci,cd,cm}!==complete_hold))$fatal(1,"completion changed under backpressure");
  held=cv && !ack;complete_hold={cc,dc,ci,cd,cm};
 end
 initial begin
  repeat(4)@(negedge clk);rst_n=1;
  while(ncpl<NACC)@(negedge clk);
  repeat(3)@(negedge clk);
  if(!quiet || accepted!=NACC || retired!=NACC || peak>SLOTS || pending!=0)$fatal(1,"counter/drain mismatch");
  for(i=0;i<8192;i=i+1)if(actual_mem[i]!==oracle[i])$fatal(1,"final memory mismatch");
  if(STREAM && !STALL && SLOTS==8 && last-first!=NACC-1)$fatal(1,"did not sustain one command/cycle span=%0d",last-first);
  for(i=1;i<=7;i=i+1)begin
   invalid=i;repeat(3)@(negedge clk);
   if(eligible || mv || !quiet || accepted!=NACC)$fatal(1,"unsupported command changed state kind=%0d",i);
  end
  invalid=0;
  $display("LINE LSU PASS slots=%0d stall=%0d stream=%0d seed=%0d commands=%0d peak=%0d admission_span=%0d request_stall=%0d slot_stall=%0d",SLOTS,STALL,STREAM,SEED,accepted,peak,last-first+1,rstall,sstall);
  $finish;
 end
endmodule
