// Multi-unit candidate versus the original two-unit LSU, using the
// randomized memory/semantic oracle from tb_lsu_reqreg. Four-bit tags remain
// fixed; candidate units share exactly sixteen total beat identities.
`timescale 1ns/1ps
`include "tnca_defs.vh"
`default_nettype none
module tb_lsu_bank;
parameter UNITS=4;
parameter FORCE_LOADS=0;
parameter SEED = 1;
parameter ACCESSES = 1500;
parameter STALL_PCT = 30;       // % of cycles the port is not ready
parameter HOLD_PCT = 5;         // % of cycles that start a long port stall
parameter MAXLAT = 24;          // response delay 1..MAXLAT cycles
parameter REORDER = 1;
parameter DIRECTED = 0;
localparam VLEN=16, ELEN=32, RCTX_W=2, NB=8, BW=4;
localparam MW = 512;            // memory words (4 KiB window)

reg clk=0; always #5 clk=~clk;
reg rst_n=0;
integer cycle=0; always @(posedge clk) cycle<=cycle+1;

// ---- the access program -------------------------------------------------------
localparam NACC = DIRECTED ? 8 : ACCESSES;
reg        a_store[0:NACC-1], a_vector[0:NACC-1], a_indexed[0:NACC-1], a_signed[0:NACC-1], a_amo[0:NACC-1];
reg [2:0]  a_size[0:NACC-1], a_dstc[0:NACC-1];
reg [1:0]  a_hint[0:NACC-1];
reg [63:0] a_base[0:NACC-1], a_offset[0:NACC-1], a_sdata[0:NACC-1], a_cmp[0:NACC-1];
reg [VLEN*ELEN-1:0] a_index[0:NACC-1], a_vdata[0:NACC-1];
reg [3:0]  a_aop[0:NACC-1];
reg [VLEN-1:0] a_mask[0:NACC-1];
reg [RCTX_W-1:0] a_rctx[0:NACC-1];
integer seed, n, l, kind;
initial begin
 seed=SEED;
 for(n=0;n<NACC;n=n+1)begin
  kind=$unsigned($random(seed))%8;
  a_store[n]=0;a_vector[n]=0;a_indexed[n]=0;a_amo[n]=0;
  a_signed[n]=$random(seed);a_hint[n]=$random(seed);a_rctx[n]=$random(seed);
  a_dstc[n]=1+($unsigned($random(seed))%6);a_aop[n]=$unsigned($random(seed))%12;
  a_base[n]=1024+($unsigned($random(seed))%1024);a_offset[n]=$unsigned($random(seed))%64;
  a_sdata[n]={$random(seed),$random(seed)};a_cmp[n]={$random(seed),$random(seed)};
  for(l=0;l<VLEN;l=l+1)begin
   a_index[n][l*ELEN+:ELEN]=($unsigned($random(seed))%64)-32;
   a_vdata[n][l*ELEN+:ELEN]=$random(seed);
  end
  a_mask[n]=$random(seed);
  if($unsigned($random(seed))%4==0)a_mask[n]={VLEN{1'b1}};
  case(kind)
  0,1,2: begin a_size[n]=$unsigned($random(seed))%4;a_mask[n][0]=1'b1; end
  3,4:   begin a_store[n]=1;a_size[n]=$unsigned($random(seed))%4;a_mask[n][0]=1'b1; end
  5:     begin a_vector[n]=1;a_indexed[n]=$random(seed);a_size[n]=$unsigned($random(seed))%3; end
  6:     begin a_vector[n]=1;a_store[n]=1;a_indexed[n]=$random(seed);a_size[n]=$unsigned($random(seed))%3; end
  default: begin
   a_amo[n]=1;a_size[n]=$random(seed) ? `TNCA_SZ_D : `TNCA_SZ_W;a_mask[n][0]=1'b1;
   a_offset[n]=0;a_base[n]=a_base[n] & ~64'd7;
  end
  endcase
 end
 if(FORCE_LOADS) for(n=0;n<NACC;n=n+1) begin
  a_amo[n]=0; a_store[n]=0; a_vector[n]=0; a_indexed[n]=0;
  a_mask[n]=16'h1; a_size[n]=`TNCA_SZ_D; a_base[n]=1024+(n%128)*8; a_offset[n]=0;
 end

 if(DIRECTED)begin
  // 0 scalar load; 1 aligned 16x32 vector load (8 beats); 2 the same store;
  // 3 an unaligned 16x32 vector load (straddlers: 9 beats); 4 a 16-lane
  // scattered gather; 5 a scalar store; 6 an atomic; 7 a scalar load.
  for(n=0;n<NACC;n=n+1)begin
   a_store[n]=0;a_vector[n]=0;a_indexed[n]=0;a_amo[n]=0;a_mask[n]={VLEN{1'b1}};
   a_base[n]=2048;a_offset[n]=0;a_size[n]=`TNCA_SZ_W;
  end
  a_size[0]=`TNCA_SZ_D;
  a_vector[1]=1;a_vector[2]=1;a_store[2]=1;a_vector[3]=1;a_offset[3]=2;
  a_vector[4]=1;a_indexed[4]=1;
  for(l=0;l<VLEN;l=l+1)a_index[4][l*ELEN+:ELEN]=l*3;
  a_store[5]=1;a_size[5]=`TNCA_SZ_D;a_amo[6]=1;a_size[6]=`TNCA_SZ_D;a_size[7]=`TNCA_SZ_D;
 end
end

// ---- two pairs ---------------------------------------------------------------
reg  [1:0] rv;
reg  [31:0] pc[0:1];
wire [1:0] rdy, rdy_load, cplv, mv, mwe, mamo;
reg  [1:0] mready, rspv, ack;
reg  [BW-1:0] rspt[0:1];
reg  [63:0] rspd[0:1];
wire [63:0] maddr[0:1], mwd[0:1], mcmp[0:1];
wire [7:0] mbe[0:1];
wire [BW-1:0] mtag[0:1];
wire [1:0] mhint[0:1];
wire [3:0] maop[0:1];
wire [RCTX_W-1:0] crctx[0:1];
wire [2:0] cdstc[0:1];
wire [7:0] cdsti[0:1];
wire [63:0] csd[0:1];
wire [VLEN*ELEN-1:0] cvd[0:1];
wire [VLEN-1:0] clane[0:1];
genvar g;
generate for(g=0;g<2;g=g+1)begin:g_dut
 wire [31:0] p=pc[g];
 if(g==0) begin:g_reference
 tnca_lsu_pair #(.VLEN(VLEN),.ELEN(ELEN),.RCTX_W(RCTX_W),.MEM_DW(64),.MEM_BYTES(8),.MEM_SHIFT(3),
                 .NUM_BEATS(NB),.BEAT_W(BW),.REQ_REG(g)) dut(
  .clk(clk),.rst_n(rst_n),.req_ready(rdy[g]),.req_ready_load(rdy_load[g]),.req_valid(rv[g]),
  .req_store(a_store[p]),.req_vector(a_vector[p]),.req_indexed(a_indexed[p]),.req_size(a_size[p]),
  .req_signed(a_signed[p]),.req_hint(a_hint[p]),.req_base(a_base[p]),.req_offset(a_offset[p]),
  .req_index(a_index[p]),.req_sdata(a_sdata[p]),.req_amo(a_amo[p]),.req_aop(a_aop[p]),.req_cmp(a_cmp[p]),
  .req_vdata(a_vdata[p]),.req_mask(a_mask[p]),.req_rctx(a_rctx[p]),.req_dst_class(a_dstc[p]),.req_dst_idx(p[7:0]),
  .cpl_valid(cplv[g]),.cpl_ack(ack[g]),.cpl_rctx(crctx[g]),.cpl_dst_class(cdstc[g]),.cpl_dst_idx(cdsti[g]),
  .cpl_sdata(csd[g]),.cpl_vdata(cvd[g]),.cpl_lane_en(clane[g]),
  .mem_req_valid(mv[g]),.mem_req_ready(mready[g]),.mem_req_addr(maddr[g]),.mem_req_we(mwe[g]),
  .mem_req_be(mbe[g]),.mem_req_wdata(mwd[g]),.mem_req_tag(mtag[g]),.mem_req_hint(mhint[g]),
  .mem_req_amo(mamo[g]),.mem_req_aop(maop[g]),.mem_req_cmp(mcmp[g]),
  .mem_rsp_valid(rspv[g]),.mem_rsp_tag(rspt[g]),.mem_rsp_rdata(rspd[g]));
 end else begin:g_candidate
 tnca_lsu_bank #(.UNITS(UNITS),.VLEN(VLEN),.ELEN(ELEN),.RCTX_W(RCTX_W),.MEM_DW(64),.MEM_BYTES(8),.MEM_SHIFT(3),
                 .NUM_BEATS(16/UNITS),.BEAT_W(BW),.REQ_REG(g)) dut(
  .clk(clk),.rst_n(rst_n),.req_ready(rdy[g]),.req_ready_load(rdy_load[g]),.req_valid(rv[g]),
  .req_store(a_store[p]),.req_vector(a_vector[p]),.req_indexed(a_indexed[p]),.req_size(a_size[p]),
  .req_signed(a_signed[p]),.req_hint(a_hint[p]),.req_base(a_base[p]),.req_offset(a_offset[p]),
  .req_index(a_index[p]),.req_sdata(a_sdata[p]),.req_amo(a_amo[p]),.req_aop(a_aop[p]),.req_cmp(a_cmp[p]),
  .req_vdata(a_vdata[p]),.req_mask(a_mask[p]),.req_rctx(a_rctx[p]),.req_dst_class(a_dstc[p]),.req_dst_idx(p[7:0]),
  .cpl_valid(cplv[g]),.cpl_ack(ack[g]),.cpl_rctx(crctx[g]),.cpl_dst_class(cdstc[g]),.cpl_dst_idx(cdsti[g]),
  .cpl_sdata(csd[g]),.cpl_vdata(cvd[g]),.cpl_lane_en(clane[g]),
  .mem_req_valid(mv[g]),.mem_req_ready(mready[g]),.mem_req_addr(maddr[g]),.mem_req_we(mwe[g]),
  .mem_req_be(mbe[g]),.mem_req_wdata(mwd[g]),.mem_req_tag(mtag[g]),.mem_req_hint(mhint[g]),
  .mem_req_amo(mamo[g]),.mem_req_aop(maop[g]),.mem_req_cmp(mcmp[g]),
  .mem_rsp_valid(rspv[g]),.mem_rsp_tag(rspt[g]),.mem_rsp_rdata(rspd[g]));
 end
end endgenerate

// ---- memories, logs ------------------------------------------------------------
reg [63:0] mem0[0:MW-1], mem1[0:MW-1];
localparam LOGN = 40000;
reg [208:0] rlog0[0:LOGN-1], rlog1[0:LOGN-1];
reg [671:0] clog0[0:NACC-1], clog1[0:NACC-1];
integer rn[0:1], cn[0:1], cbase[0:1];
reg [NACC-1:0] cdone[0:1];
// responses in flight: tag, data, due cycle
reg [BW-1:0] pt[0:1][0:15]; reg [63:0] pd[0:1][0:15]; integer pdue[0:1][0:15]; reg pv[0:1][0:15];
reg [15:0] out_tags[0:1];
integer hold[0:1];
integer ms0, ms1;
// per-instance stimulus streams, so one side's draws do not shift the other's
function integer rk;
 input integer kk;
 begin if(kk) rk=$random(ms1); else rk=$random(ms0); end
endfunction
integer k, j, b, cand, ncand;
reg [63:0] old, nw, w;
initial begin
 for(k=0;k<MW;k=k+1)begin mem0[k]={$random,$random};mem1[k]=mem0[k];end
 for(k=0;k<2;k=k+1)begin
  rn[k]=0;cn[k]=0;cbase[k]=0;cdone[k]=0;out_tags[k]=0;hold[k]=0;pc[k]=0;
  for(j=0;j<16;j=j+1)pv[k][j]=0;
 end
 ms0=SEED*7+1;ms1=SEED*13+5;
 rv=0;mready=0;rspv=0;ack=0;
end

function [63:0] amo_new;
 input [63:0] o, d, c; input [3:0] op;
 amo_new = (op==`TNCA_AOP_CAS) ? ((o==c) ? d : o) : (o + d + op);
endfunction

// requests, responses and completions, at the edge where they transfer
always @(posedge clk) if(rst_n) begin
 for(k=0;k<2;k=k+1)begin
  if(mv[k] && mready[k])begin
   w = k ? mem1[maddr[k][11:3]] : mem0[maddr[k][11:3]];
   old = w;
   if(mwe[k] || mamo[k])begin
    nw = mamo[k] ? amo_new(old,mwd[k],mcmp[k],maop[k]) : mwd[k];
    for(b=0;b<8;b=b+1)if(mbe[k][b])w[b*8+:8]=nw[b*8+:8];
    if(k)mem1[maddr[k][11:3]]=w;else mem0[maddr[k][11:3]]=w;
   end
   if(out_tags[k][mtag[k]])$fatal(1,"dut%0d: tag %0d reused while outstanding",k,mtag[k]);
   out_tags[k][mtag[k]]=1'b1;
   if(k)rlog1[rn[k]]={maddr[k],mwe[k],mbe[k],mwd[k],mamo[k],maop[k],mcmp[k]};
   else rlog0[rn[k]]={maddr[k],mwe[k],mbe[k],mwd[k],mamo[k],maop[k],mcmp[k]};
   rn[k]=rn[k]+1;
   cand=-1;for(j=15;j>=0;j=j-1)if(!pv[k][j])cand=j;
   if(cand<0)$fatal(1,"dut%0d: more than 16 beats in flight",k);
   pv[k][cand]=1;pt[k][cand]=mtag[k];pd[k][cand]=old;
   pdue[k][cand]=cycle+1+(DIRECTED ? 3 : $unsigned(rk(k))%MAXLAT);
  end
  if(rspv[k])out_tags[k][rspt[k]]=1'b0;
  if(cplv[k] && ack[k])begin
   // The pair may complete the younger of its two accesses first; file
   // each completion under its access number (dst_idx is that number mod 256).
   if(cn[k]>=NACC)$fatal(1,"dut%0d: extra completion",k);
   cand=-1;
   for(j=pc[k];j>=cbase[k];j=j-1)if(j<NACC && !cdone[k][j] && j%256==cdsti[k])cand=j;
   if(cand<0)$fatal(1,"dut%0d: completion for no outstanding access",k);
   cdone[k][cand]=1'b1;
   while(cbase[k]<NACC && cdone[k][cbase[k]])cbase[k]=cbase[k]+1;
   if(k)clog1[cand]={crctx[k],cdstc[k],cdsti[k],csd[k],cvd[k],clane[k]};
   else clog0[cand]={crctx[k],cdstc[k],cdsti[k],csd[k],cvd[k],clane[k]};
   cn[k]=cn[k]+1;
  end
 end
end

// drive the request, port readiness, one response and the completion ack
always @(negedge clk) if(rst_n) begin
 for(k=0;k<2;k=k+1)begin
  if(rv[k])pc[k]=pc[k]+1;          // taken at the edge: only offered when ready
  rv[k]=0;
  if(pc[k]<NACC && (DIRECTED ? (rdy[k] && !cplv[k]) : ($unsigned(rk(k))%4!=0)) &&
     (a_amo[pc[k]] ? rdy[k] : rdy_load[k]))
   rv[k]=1;
  if(DIRECTED)mready[k]=1;
  else begin
   if(hold[k]>0)hold[k]=hold[k]-1;
   else if($unsigned(rk(k))%100<HOLD_PCT)hold[k]=$unsigned(rk(k))%40;
   mready[k]=(hold[k]==0) && ($unsigned(rk(k))%100>=STALL_PCT);
  end
  rspv[k]=0;ncand=0;cand=-1;
  for(j=0;j<16;j=j+1)if(pv[k][j] && pdue[k][j]<=cycle)begin
   ncand=ncand+1;
   if(cand<0 || (REORDER && $unsigned(rk(k))%ncand==0) || (!REORDER && pdue[k][j]<pdue[k][cand]))cand=j;
  end
  if(cand>=0)begin rspv[k]=1;rspt[k]=pt[k][cand];rspd[k]=pd[k][cand];pv[k][cand]=0;end
  ack[k]=DIRECTED ? 1'b1 : ($unsigned(rk(k))%3!=0);
 end
end

// ---- checks on the registered boundary (instance 1) and on both ports ----------
integer peak=0;
reg completion_stalled=0;
reg [671:0] held_completion;
reg stalled=0;
reg [208+BW:0] held_req;
always @(posedge clk) if(rst_n) begin
 if(completion_stalled && (!cplv[1] || {crctx[1],cdstc[1],cdsti[1],csd[1],cvd[1],clane[1]} !== held_completion))
  $fatal(1,"completion changed under RF backpressure");
 completion_stalled=cplv[1] && !ack[1];
 held_completion={crctx[1],cdstc[1],cdsti[1],csd[1],cvd[1],clane[1]};
 if(g_dut[1].g_candidate.dut.count>peak) peak=g_dut[1].g_candidate.dut.count;
 if(stalled && (!mv[1] || {maddr[1],mwe[1],mbe[1],mwd[1],mamo[1],maop[1],mcmp[1],mtag[1]}!==held_req))
  $fatal(1,"registered request changed while stalled");
 stalled = mv[1] && !mready[1];
 held_req = {maddr[1],mwe[1],mbe[1],mwd[1],mamo[1],maop[1],mcmp[1],mtag[1]};

end

// ---- directed timing -------------------------------------------------------------
integer t_acc[0:1][0:7], t_first[0:1][0:7], t_last[0:1][0:7], t_cpl[0:1][0:7], cur[0:1], nreq[0:1][0:7];
initial for(k=0;k<2;k=k+1)for(j=0;j<8;j=j+1)begin t_first[k][j]=-1;nreq[k][j]=0;end
always @(posedge clk) if(rst_n && DIRECTED) for(k=0;k<2;k=k+1)begin
 if(rv[k])begin t_acc[k][pc[k]]=cycle;cur[k]=pc[k];end
 if(mv[k] && mready[k])begin
  if(t_first[k][cur[k]]<0)t_first[k][cur[k]]=cycle;
  t_last[k][cur[k]]=cycle;nreq[k][cur[k]]=nreq[k][cur[k]]+1;
 end
 if(cplv[k] && ack[k])t_cpl[k][cdsti[k]]=cycle;
end

// ---- run and compare -----------------------------------------------------------
integer done_cycle[0:1];
initial begin
 done_cycle[0]=-1;done_cycle[1]=-1;
 repeat(4)@(negedge clk);rst_n=1;
 while(done_cycle[0]<0 || done_cycle[1]<0)begin
  @(negedge clk);
  for(k=0;k<2;k=k+1)
   if(done_cycle[k]<0 && pc[k]==NACC && !rv[k] && rdy[k] && !cplv[k] && cn[k]==NACC)done_cycle[k]=cycle;
  if(cycle>100*NACC+10000)$fatal(1,"timeout pc=%0d/%0d cpl=%0d/%0d",pc[0],pc[1],cn[0],cn[1]);
 end
 if(rn[0]!=rn[1])$fatal(1,"request count %0d vs %0d",rn[0],rn[1]);
 for(k=0;k<rn[0];k=k+1)if(rlog0[k]!==rlog1[k])$fatal(1,"request %0d differs: %h vs %h",k,rlog0[k],rlog1[k]);
 for(k=0;k<NACC;k=k+1)if(clog0[k]!==clog1[k])$fatal(1,"completion %0d differs:\n%h\n%h",k,clog0[k],clog1[k]);
 for(k=0;k<MW;k=k+1)if(mem0[k]!==mem1[k])$fatal(1,"memory word %0d differs",k);
 if(DIRECTED)begin
  for(j=0;j<NACC;j=j+1)begin
   $display("ACCESS %0d beats=%0d | REQ_REG=0: first+%0d span %0d done+%0d | REQ_REG=1: first+%0d span %0d done+%0d",
            j,nreq[0][j],t_first[0][j]-t_acc[0][j],t_last[0][j]-t_first[0][j]+1,t_cpl[0][j]-t_acc[0][j],
            t_first[1][j]-t_acc[1][j],t_last[1][j]-t_first[1][j]+1,t_cpl[1][j]-t_acc[1][j]);
   if(nreq[0][j]!=nreq[1][j] || t_last[0][j]-t_first[0][j]!=t_last[1][j]-t_first[1][j])
    $fatal(1,"beat count or issue rate differs");
   if(nreq[0][j]!=t_last[0][j]-t_first[0][j]+1)$fatal(1,"access %0d not one beat per cycle",j);
   if(t_first[1][j]-t_acc[1][j]!=t_first[0][j]-t_acc[0][j]+1)$fatal(1,"request latency is not +1");
  end
 end
 if(FORCE_LOADS && peak!=UNITS) $fatal(1,"did not occupy all units peak=%0d",peak);
 $display("LSU BANK CAPACITY units=%0d peak=%0d",UNITS,peak);
 $display("LSU BANK MITER PASS seed=%0d accesses=%0d requests=%0d stall=%0d%% cycles REQ_REG=0 %0d REQ_REG=1 %0d",
          SEED,NACC,rn[0],STALL_PCT,done_cycle[0],done_cycle[1]);
 $finish;
end
endmodule
`default_nettype wire
