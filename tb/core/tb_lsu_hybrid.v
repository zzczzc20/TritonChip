// Multi-unit candidate versus the original two-unit LSU, using the
// randomized memory/semantic oracle from tb_lsu_reqreg. BEAT_W scales the
// port tags and response bookkeeping; its default preserves the original geometry.
`timescale 1ns/1ps
`include "tnca_defs.vh"
`default_nettype none
module tb_lsu_hybrid;
parameter BURST=0;
parameter BEAT_W=4;
parameter CAND_BEATS=4;
parameter BURST_PERIOD=32;
parameter BURST_DEPTH=24;
parameter BURST_BASE=2048;
parameter BURST_LINES=16;
parameter MEM_WORDS=512;       // memory words (4 KiB window by default)
parameter CHAIN_PERIOD=64;      // accesses between the known store/CAS chains (>= 37)
parameter FORCE_LOADS=0;
parameter FORCE_INDEXED=0;
parameter SEED = 1;
parameter ACCESSES = 1500;
parameter STALL_PCT = 30;       // % of cycles the port is not ready
parameter HOLD_PCT = 5;         // % of cycles that start a long port stall
parameter MAXLAT = 24;          // response delay 1..MAXLAT cycles
parameter REORDER = 1;
parameter DIRECTED = 0;
localparam VLEN=16, ELEN=32, RCTX_W=2, NB=8, BW=BEAT_W, TAGS=1<<BEAT_W;
localparam MW = MEM_WORDS;

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
 if(BURST)for(n=0;n<NACC;n=n+1)if(n%BURST_PERIOD<BURST_DEPTH)begin
  a_vector[n]=1;a_indexed[n]=0;a_amo[n]=0;a_size[n]=`TNCA_SZ_W;
  a_base[n]=BURST_BASE+(n%BURST_LINES)*64;a_offset[n]=0;a_mask[n]=16'hffff;a_store[n]=n%32>=16;
 end

 for(n=0;n<NACC;n=n+1)if(a_amo[n] && a_size[n]==`TNCA_SZ_W && n%2) a_base[n]=a_base[n]+4;
 // Known store/CAS-success/read/CAS-failure/read chains, across fast/slow transitions.
 for(n=0;n<NACC;n=n+1)if(n%CHAIN_PERIOD>=32 && n%CHAIN_PERIOD<=36)begin
  a_vector[n]=0;a_indexed[n]=0;a_amo[n]=0;a_store[n]=0;a_size[n]=`TNCA_SZ_D;
  a_base[n]=2048;a_offset[n]=0;a_mask[n]=1;a_signed[n]=0;
  if(n%CHAIN_PERIOD==32)begin a_store[n]=1;a_sdata[n]=64'h123456789abcdef0;end
  if(n%CHAIN_PERIOD==33 || n%CHAIN_PERIOD==35)begin
   a_amo[n]=1;a_aop[n]=`TNCA_AOP_CAS;a_sdata[n]=64'h0102030405060708;
   a_cmp[n]=n%CHAIN_PERIOD==33?64'h123456789abcdef0:64'hffffffffffffffff;
  end
 end

 if(FORCE_LOADS) for(n=0;n<NACC;n=n+1) begin
  a_amo[n]=0; a_store[n]=0; a_vector[n]=0; a_indexed[n]=0;
  a_mask[n]=16'h1; a_size[n]=`TNCA_SZ_D; a_base[n]=1024+(n%128)*8; a_offset[n]=0;
 end
 if(FORCE_INDEXED) for(n=0;n<NACC;n=n+1) begin
  a_amo[n]=0; a_store[n]=0; a_vector[n]=1; a_indexed[n]=1;
  a_mask[n]=16'hffff; a_size[n]=`TNCA_SZ_W; a_base[n]=2048; a_offset[n]=0;
  for(l=0;l<VLEN;l=l+1) a_index[n][l*ELEN+:ELEN]=l*8;
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
reg [511:0] rspd[0:1];
wire [63:0] maddr[0:1],mcmp[0:1];
wire [511:0] mwd[0:1];
wire [63:0] mbe[0:1];
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
 wire [63:0] wd;
 wire [7:0] be;
 assign mwd[g]={448'd0,wd};assign mbe[g]={56'd0,be};
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
  .mem_req_be(be),.mem_req_wdata(wd),.mem_req_tag(mtag[g]),.mem_req_hint(mhint[g]),
  .mem_req_amo(mamo[g]),.mem_req_aop(maop[g]),.mem_req_cmp(mcmp[g]),
  .mem_rsp_valid(rspv[g]),.mem_rsp_tag(rspt[g]),.mem_rsp_rdata(rspd[g][63:0]));
 end else begin:g_candidate
 tnca_lsu_hybrid #(.VLEN(VLEN),.ELEN(ELEN),.RCTX_W(RCTX_W),.MEM_DW(512),.MEM_BYTES(64),.MEM_SHIFT(6),
                 .NUM_BEATS(CAND_BEATS),.BEAT_W(BW),.REQ_REG(g)) dut(
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
reg [BW-1:0] pt[0:1][0:TAGS-1]; reg [511:0] pd[0:1][0:TAGS-1]; integer pdue[0:1][0:TAGS-1]; reg pv[0:1][0:TAGS-1];
reg [TAGS-1:0] out_tags[0:1];
integer hold[0:1];
integer ms0, ms1;
// per-instance stimulus streams, so one side's draws do not shift the other's
function integer rk;
 input integer kk;
 begin if(kk) rk=$random(ms1); else rk=$random(ms0); end
endfunction
integer k, j, b, cand, ncand;
reg [511:0] old,w;
reg [63:0] nw;
integer groupno,baseaddr;
integer fast_commands=0,slow_requests=0;
initial begin
 for(k=0;k<MW;k=k+1)begin mem0[k]={$random,$random};mem1[k]=mem0[k];end
 for(k=0;k<2;k=k+1)begin
  rn[k]=0;cn[k]=0;cbase[k]=0;cdone[k]=0;out_tags[k]=0;hold[k]=0;pc[k]=0;
  for(j=0;j<TAGS;j=j+1)pv[k][j]=0;
 end
 ms0=SEED*7+1;ms1=SEED*13+5;
 rv=0;mready=0;rspv=0;ack=0;
end

`include "tnca_amo.vh"

// requests, responses and completions, at the edge where they transfer
always @(posedge clk) if(rst_n) begin
 for(k=0;k<2;k=k+1)begin
  if(mv[k] && mready[k])begin
   baseaddr=maddr[k] & (k?~64'd63:~64'd7);
   w=0;
   for(b=0;b<(k?64:8);b=b+1) w[b*8+:8]=k?mem1[(baseaddr+b)/8][((baseaddr+b)%8)*8+:8]:mem0[(baseaddr+b)/8][((baseaddr+b)%8)*8+:8];
   old=w;
   if(k)begin if(mtag[k][BW-1])fast_commands=fast_commands+1;else slow_requests=slow_requests+1;end
   if(mwe[k] || mamo[k])begin
    for(groupno=0;groupno<(k?8:1);groupno=groupno+1)begin
     nw=mamo[k]?amo_new(maop[k],mbe[k][groupno*8+:8],w[groupno*64+:64],mwd[k][groupno*64+:64],mcmp[k]):mwd[k][groupno*64+:64];
     for(b=0;b<8;b=b+1)if(mbe[k][groupno*8+b])w[groupno*64+b*8+:8]=nw[b*8+:8];
    end
    for(b=0;b<(k?64:8);b=b+1)begin
     if(k)mem1[(baseaddr+b)/8][((baseaddr+b)%8)*8+:8]=w[b*8+:8];
     else mem0[(baseaddr+b)/8][((baseaddr+b)%8)*8+:8]=w[b*8+:8];
    end
   end
   if(out_tags[k][mtag[k]])$fatal(1,"dut%0d: tag %0d reused while outstanding",k,mtag[k]);
   out_tags[k][mtag[k]]=1'b1;
   if(k)rlog1[rn[k]]={maddr[k],mwe[k],mbe[k],mwd[k],mamo[k],maop[k],mcmp[k]};
   else rlog0[rn[k]]={maddr[k],mwe[k],mbe[k],mwd[k],mamo[k],maop[k],mcmp[k]};
   rn[k]=rn[k]+1;
   cand=-1;for(j=TAGS-1;j>=0;j=j-1)if(!pv[k][j])cand=j;
   if(cand<0)$fatal(1,"dut%0d: more than %0d beats in flight",k,TAGS);
   pv[k][cand]=1;pt[k][cand]=mtag[k];pd[k][cand]=old;
   pdue[k][cand]=cycle+1+(FORCE_INDEXED ? MAXLAT : DIRECTED ? 3 : $unsigned(rk(k))%MAXLAT);
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
  if(rv[k])pc[k]=pc[k]+1;
  rv[k]=0;
  if(DIRECTED)mready[k]=1;
  else begin
   if(hold[k]>0)hold[k]=hold[k]-1;
   else if($unsigned(rk(k))%100<HOLD_PCT)hold[k]=$unsigned(rk(k))%40;
   mready[k]=(hold[k]==0) && ($unsigned(rk(k))%100>=STALL_PCT);
  end
  rspv[k]=0;ncand=0;cand=-1;
  for(j=0;j<TAGS;j=j+1)if(pv[k][j] && pdue[k][j]<=cycle)begin
   ncand=ncand+1;
   if(cand<0 || (REORDER && $unsigned(rk(k))%ncand==0) || (!REORDER && pdue[k][j]<pdue[k][cand]))cand=j;
  end
  if(cand>=0)begin rspv[k]=1;rspt[k]=pt[k][cand];rspd[k]=pd[k][cand];pv[k][cand]=0;end
  ack[k]=DIRECTED ? 1'b1 : ($unsigned(rk(k))%3!=0);
 end
 // Eligibility and resource readiness depend on the new instruction and
 // current fabric ready. Let those combinational paths settle before offer.
 #1;
 for(k=0;k<2;k=k+1)
  if(pc[k]<NACC && ($unsigned(rk(k))%4!=0) && (a_amo[pc[k]]?rdy[k]:rdy_load[k]))rv[k]=1;
end

// ---- checks on the registered boundary (instance 1) and on both ports ----------
integer peak=0;
reg completion_stalled=0;
reg [671:0] held_completion;
reg stalled=0;
reg [709+BW:0] held_req;
always @(posedge clk) if(rst_n) begin
 if(completion_stalled && (!cplv[1] || {crctx[1],cdstc[1],cdsti[1],csd[1],cvd[1],clane[1]} !== held_completion))
  $fatal(1,"completion changed under RF backpressure");
 completion_stalled=cplv[1] && !ack[1];
 held_completion={crctx[1],cdstc[1],cdsti[1],csd[1],cvd[1],clane[1]};
 if(g_dut[1].g_candidate.dut.fast.count>peak) peak=g_dut[1].g_candidate.dut.fast.count;
 if(stalled && (!mv[1] || {maddr[1],mwe[1],mbe[1],mwd[1],mamo[1],maop[1],mcmp[1],mtag[1]}!==held_req))
  $fatal(1,"registered request changed while stalled");
 stalled = mv[1] && !mready[1];
 held_req = {maddr[1],mwe[1],mbe[1],mwd[1],mamo[1],maop[1],mcmp[1],mtag[1]};

end

// ---- run and compare -----------------------------------------------------------
integer done_cycle[0:1];
integer narrow_peak=0, narrow_count, q;
always @(negedge clk) if(rst_n) begin
 narrow_count=0;
 for(q=0;q<(1<<(BW-1));q=q+1) if(out_tags[1][q]) narrow_count=narrow_count+1;
 if(narrow_count>narrow_peak) narrow_peak=narrow_count;
end
initial begin
 done_cycle[0]=-1;done_cycle[1]=-1;
 repeat(4)@(negedge clk);rst_n=1;
 while(done_cycle[0]<0 || done_cycle[1]<0)begin
  @(negedge clk);
  for(k=0;k<2;k=k+1)
   if(done_cycle[k]<0 && pc[k]==NACC && !rv[k] && rdy[k] && !cplv[k] && cn[k]==NACC)done_cycle[k]=cycle;
  // Hang detector. The original bound holds for MAXLAT <= 64; the widened
  // configurations (BEAT_W > 5, MAXLAT 1024) scale it with the latency.
  if(cycle>(MAXLAT>64 ? 4*MAXLAT : 100)*NACC+10000)$fatal(1,"timeout pc=%0d/%0d cpl=%0d/%0d",pc[0],pc[1],cn[0],cn[1]);
 end
 for(k=0;k<NACC;k=k+1)if(clog0[k]!==clog1[k])$fatal(1,"completion %0d differs:\n%h\n%h",k,clog0[k],clog1[k]);
 for(k=0;k<MW;k=k+1)if(mem0[k]!==mem1[k])$fatal(1,"memory word %0d differs",k);
 if(g_dut[1].g_candidate.dut.perf_fast_accepted+g_dut[1].g_candidate.dut.perf_slow_accepted!=NACC ||
    g_dut[1].g_candidate.dut.perf_retired!=NACC ||
    g_dut[1].g_candidate.dut.perf_fast_retired!=g_dut[1].g_candidate.dut.perf_fast_accepted)
  $fatal(1,"hybrid counter mismatch");
 if(out_tags[0]!=0 || out_tags[1]!=0)$fatal(1,"undrained memory transactions");
 if(!FORCE_LOADS && !FORCE_INDEXED && (fast_commands==0 || slow_requests==0))$fatal(1,"both paths not covered");
 if(FORCE_INDEXED && narrow_peak<2*CAND_BEATS)$fatal(1,"narrow slots not filled peak=%0d expected=%0d",narrow_peak,2*CAND_BEATS);
 if(FORCE_INDEXED)$display("HYBRID NARROW PEAK %0d",narrow_peak);
 $display("HYBRID COVERAGE fast=%0d slow_beats=%0d",fast_commands,slow_requests);
 if(BURST && MAXLAT>=32 && peak!=(1<<(BW-1)))$fatal(1,"fast slots not fully exercised peak=%0d expected=%0d",peak,1<<(BW-1));
 $display("HYBRID FAST PEAK %0d",peak);
 $display("HYBRID LSU MITER PASS beat_w=%0d seed=%0d accesses=%0d requests=%0d stall=%0d%% cycles REQ_REG=0 %0d REQ_REG=1 %0d",
          BW,SEED,NACC,rn[0],STALL_PCT,done_cycle[0],done_cycle[1]);
 $finish;
end
endmodule
`default_nettype wire
