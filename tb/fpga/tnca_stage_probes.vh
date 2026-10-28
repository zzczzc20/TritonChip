// Observation-only stage counters for the multicore register benches.
//
// Enabled by the +stageprobe plusarg; without it nothing is counted or printed,
// and with it nothing is driven: every counter is a testbench register fed by
// hierarchical reads of existing RTL signals, so the design under test and the
// register protocol are unchanged. Used by tools/r41/link_budget_rtl.py
// (evidence/r41/hbm_bw/LINK_BUDGET.md).
//
// The including bench defines, before the include:
//   `TNCA_PROBE_TOP    hierarchical path of the tnca_top instance
//   `TNCA_PROBE_START  expression true in the cycle a grid command is accepted
//   `TNCA_PROBE_DONE   expression true in the cycle a grid completion is taken
// and has clk, rst_n_async and the m_* AXI outlet vectors in scope.
//
// At every grid start and completion it prints cumulative counts since reset:
//   STAGE CYCLE   at=BEGIN|END cycle=<n>
//   STAGE TILE    one line per tile: core port, NI, home slice, home memory side,
//                 home engine state cycles, core pipeline counters
//   STAGE PORT    one line per AXI outlet: handshakes and valid-not-ready cycles
// The analysis takes differences between consecutive prints.
reg stage_probe_on=0;
initial stage_probe_on=$test$plusargs("stageprobe");
reg [63:0] stage_cycle=0;
always @(posedge clk) if(stage_probe_on) stage_cycle<=stage_cycle+1;
wire stage_begin=stage_probe_on && rst_n_async && (`TNCA_PROBE_START);
wire stage_end=stage_probe_on && rst_n_async && (`TNCA_PROBE_DONE);
always @(posedge clk) if(stage_begin || stage_end)
 $display("STAGE CYCLE at=%0s cycle=%0d",stage_end?"END":"BEGIN",stage_cycle);
generate for(genvar sp_c=0;sp_c<N;sp_c=sp_c+1)begin:g_stage_tile
 `define SPT `TNCA_PROBE_TOP.g_tile[sp_c].u_tile
 reg [63:0] creq=0,creq_we=0,creq_stall=0,crsp=0,inj=0,inj_stall=0,ej=0,ej_stall=0;
 reg [63:0] hreq=0,hreq_stall=0,hrsp=0,hrsp_stall=0,mreq_rd=0,mreq_wr=0,mreq_stall=0,mrsp=0;
 // Little's law: the per-cycle sum of requests accepted and not yet answered,
 // at the core port and at the home slice, gives their mean round-trip latency.
 reg [63:0] c_occ=0,c_occ_sum=0,h_occ=0,h_occ_sum=0;
 reg [63:0] e_sel=0,e_sel_wait=0,e_look=0,e_hit=0,e_miss=0,e_wbrd=0,e_wbsnd=0,e_mreq=0,e_fill=0,e_wake=0,e_other=0;
 always @(posedge clk) if(stage_probe_on && rst_n_async) begin
  if(`SPT.c_req_valid && `SPT.c_req_ready) begin creq<=creq+1; if(`SPT.c_req_we) creq_we<=creq_we+1; end
  if(`SPT.c_req_valid && !`SPT.c_req_ready) creq_stall<=creq_stall+1;
  if(`SPT.c_rsp_valid) crsp<=crsp+1;
  if(`SPT.ni_in_v_req && `SPT.ni_in_r_req) inj<=inj+1;
  if(`SPT.ni_in_v_req && !`SPT.ni_in_r_req) inj_stall<=inj_stall+1;
  if(`SPT.ni_out_v_rsp && `SPT.ni_out_r_rsp) ej<=ej+1;
  if(`SPT.ni_out_v_rsp && !`SPT.ni_out_r_rsp) ej_stall<=ej_stall+1;
  if(`SPT.s_req_valid && `SPT.s_req_ready) hreq<=hreq+1;
  if(`SPT.s_req_valid && !`SPT.s_req_ready) hreq_stall<=hreq_stall+1;
  if(`SPT.s_rsp_valid && `SPT.s_rsp_ready) hrsp<=hrsp+1;
  if(`SPT.s_rsp_valid && !`SPT.s_rsp_ready) hrsp_stall<=hrsp_stall+1;
  if(`SPT.mem_req_valid && `SPT.mem_req_ready) begin
   if(`SPT.mem_req_we) mreq_wr<=mreq_wr+1; else mreq_rd<=mreq_rd+1;
  end
  if(`SPT.mem_req_valid && !`SPT.mem_req_ready) mreq_stall<=mreq_stall+1;
  if(`SPT.mem_rsp_valid) mrsp<=mrsp+1;
  c_occ<=c_occ+(`SPT.c_req_valid && `SPT.c_req_ready)-`SPT.c_rsp_valid;
  c_occ_sum<=c_occ_sum+c_occ;
  h_occ<=h_occ+(`SPT.s_req_valid && `SPT.s_req_ready)-(`SPT.s_rsp_valid && `SPT.s_rsp_ready);
  h_occ_sum<=h_occ_sum+h_occ;
  case(`SPT.u_slice.est)
   4'd0: begin e_sel<=e_sel+1; if(`SPT.s_req_valid) e_sel_wait<=e_sel_wait+1; end
   4'd1: e_look<=e_look+1;
   4'd2: e_hit<=e_hit+1;
   4'd3: e_miss<=e_miss+1;
   4'd4: e_wbrd<=e_wbrd+1;
   4'd5: e_wbsnd<=e_wbsnd+1;
   4'd6: e_mreq<=e_mreq+1;
   4'd7: e_fill<=e_fill+1;
   4'd8: e_wake<=e_wake+1;
   default: e_other<=e_other+1;
  endcase
 end
 always @(posedge clk) if(stage_begin || stage_end)
  $display("STAGE TILE at=%0s c=%0d t=%0d creq=%0d creq_we=%0d creq_stall=%0d crsp=%0d inj=%0d inj_stall=%0d ej=%0d ej_stall=%0d hreq=%0d hreq_stall=%0d hrsp=%0d hrsp_stall=%0d mreq_rd=%0d mreq_wr=%0d mreq_stall=%0d mrsp=%0d c_occ_sum=%0d h_occ_sum=%0d e_sel=%0d e_sel_wait=%0d e_look=%0d e_hit=%0d e_miss=%0d e_wbrd=%0d e_wbsnd=%0d e_mreq=%0d e_fill=%0d e_wake=%0d e_other=%0d hit=%0d miss=%0d merge=%0d mshr_full=%0d wb=%0d cycles=%0d issued=%0d stall=%0d x_hazard=%0d x_lsu=%0d x_fpu=%0d x_txf=%0d x_vrf=%0d x_vrfwb=%0d x_wb=%0d x_frontend=%0d x_fence=%0d x_rsv=%0d",
   stage_end?"END":"BEGIN",stage_cycle,sp_c,creq,creq_we,creq_stall,crsp,inj,inj_stall,ej,ej_stall,hreq,hreq_stall,hrsp,hrsp_stall,mreq_rd,mreq_wr,mreq_stall,mrsp,c_occ_sum,h_occ_sum,
   e_sel,e_sel_wait,e_look,e_hit,e_miss,e_wbrd,e_wbsnd,e_mreq,e_fill,e_wake,e_other,
   `SPT.u_slice.perf_hit,`SPT.u_slice.perf_miss,`SPT.u_slice.perf_merge,`SPT.u_slice.perf_mshr_full,`SPT.u_slice.perf_wb,
   `SPT.u_core.perf_cycles,`SPT.u_core.perf_issued,`SPT.u_core.perf_stall_total,`SPT.u_core.perf_x_hazard,`SPT.u_core.perf_x_lsu,
   `SPT.u_core.perf_x_fpu,`SPT.u_core.perf_x_txf,`SPT.u_core.perf_x_vrf,`SPT.u_core.perf_x_vrfwb,`SPT.u_core.perf_x_wb,
   `SPT.u_core.perf_x_frontend,`SPT.u_core.perf_x_fence,`SPT.u_core.perf_x_rsv);
 `undef SPT
end endgenerate
generate for(genvar sp_p=0;sp_p<PORTS;sp_p=sp_p+1)begin:g_stage_port
 reg [63:0] ar=0,ar_stall=0,r=0,r_stall=0,aw=0,aw_stall=0,w=0,w_stall=0,b=0,busy=0;
 reg [63:0] rd_occ=0,rd_occ_sum=0,wr_occ=0,wr_occ_sum=0;
 always @(posedge clk) if(stage_probe_on && rst_n_async) begin
  if(m_arvalid[sp_p] && m_arready[sp_p]) ar<=ar+1;
  if(m_arvalid[sp_p] && !m_arready[sp_p]) ar_stall<=ar_stall+1;
  if(m_rvalid[sp_p] && m_rready[sp_p]) r<=r+1;
  if(m_rvalid[sp_p] && !m_rready[sp_p]) r_stall<=r_stall+1;
  if(m_awvalid[sp_p] && m_awready[sp_p]) aw<=aw+1;
  if(m_awvalid[sp_p] && !m_awready[sp_p]) aw_stall<=aw_stall+1;
  if(m_wvalid[sp_p] && m_wready[sp_p]) w<=w+1;
  if(m_wvalid[sp_p] && !m_wready[sp_p]) w_stall<=w_stall+1;
  if(m_bvalid[sp_p] && m_bready[sp_p]) b<=b+1;
  if(m_arvalid[sp_p] || m_rvalid[sp_p] || m_awvalid[sp_p] || m_wvalid[sp_p]) busy<=busy+1;
  rd_occ<=rd_occ+(m_arvalid[sp_p] && m_arready[sp_p])-(m_rvalid[sp_p] && m_rready[sp_p] && m_rlast[sp_p]);
  rd_occ_sum<=rd_occ_sum+rd_occ;
  wr_occ<=wr_occ+(m_awvalid[sp_p] && m_awready[sp_p])-(m_bvalid[sp_p] && m_bready[sp_p]);
  wr_occ_sum<=wr_occ_sum+wr_occ;
 end
 always @(posedge clk) if(stage_begin || stage_end)
  $display("STAGE PORT at=%0s c=%0d p=%0d ar=%0d ar_stall=%0d r=%0d r_stall=%0d aw=%0d aw_stall=%0d w=%0d w_stall=%0d b=%0d busy=%0d rd_occ_sum=%0d wr_occ_sum=%0d",
   stage_end?"END":"BEGIN",stage_cycle,sp_p,ar,ar_stall,r,r_stall,aw,aw_stall,w,w_stall,b,busy,rd_occ_sum,wr_occ_sum);
end endgenerate
