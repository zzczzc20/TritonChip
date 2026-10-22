// Out-of-context timing harness for R41 F10, physical screening only: the
// real LSU pair, the core's memory mux, tile 0's host mux, the core NI and a
// real router (whose local request FIFO is the census endpoint), plus the
// real cache-ownership handoff fed by the LSU's idle state, as tnca_core ->
// tnca_tile -> tnca_top connect them at the B40 image parameters. Every
// other input is a register and every output lands in one, so the routed
// census families (LSU pending -> mesh input FIFO; LSU FSM -> handoff
// home_base/home_size enable) are register-to-register paths here.
// It cannot qualify placement, SLR crossings or routing.
`include "tnca_noc_pkg.vh"
`default_nettype none
module tnca_lsu_mesh_ooc #(
    parameter VLEN=16, ELEN=32, RCTX_W=2, NUM_BEATS=8, CTAG_W=4, TAG_W=5,
    parameter NX=3, NY=1, XW=2, YW=2, N=3
)(
    input wire clk, rst_n,
    // core X-stage request, registered here as in the core
    input wire req_valid, req_store, req_vector, req_indexed, req_signed, req_amo, cpl_ack,
    input wire [2:0] req_size, req_dst_class,
    input wire [1:0] req_hint,
    input wire [63:0] req_base, req_offset, req_sdata, req_cmp,
    input wire [VLEN*ELEN-1:0] req_index, req_vdata,
    input wire [3:0] req_aop,
    input wire [VLEN-1:0] req_mask,
    input wire [RCTX_W-1:0] req_rctx,
    input wire [7:0] req_dst_idx,
    // transfer engine (memory mux client B)
    input wire tm_valid, tm_we,
    input wire [63:0] tm_addr, tm_wdata,
    input wire [7:0] tm_be,
    input wire [CTAG_W-1:0] tm_tag,
    // host port of tile 0
    input wire host_allow, h_valid, h_we,
    input wire [63:0] h_addr, h_wdata,
    input wire [7:0] h_be,
    // the router's other ports and its local outputs' readiness
    input wire [5*2-1:0] r_in_valid_ext, r_out_ready_ext,
    input wire [5*2*(XW+YW+`TNCA_NOC_PAYLOAD_W)-1:0] r_in_flit_ext,
    // the rest of the machine's quiescence, and the ownership request
    input wire other_idle, maint_valid, memory_drained,
    input wire [63:0] maint_base, maint_size,
    input wire [N-1:0] home_ready, home_done,
    output reg [1023:0] observed
);
    localparam FW = XW + YW + `TNCA_NOC_PAYLOAD_W;
    localparam NC = 10;
    // ---- input registers -----------------------------------------------------
    reg rv, rs, rvec, rix, rsg, ramo, ack; reg [2:0] rsz, rdc; reg [1:0] rh;
    reg [63:0] rb, ro, rsd, rcmp; reg [VLEN*ELEN-1:0] rix_d, rvd; reg [3:0] raop;
    reg [VLEN-1:0] rm; reg [RCTX_W-1:0] rrc; reg [7:0] rdi;
    reg tv, twe; reg [63:0] ta, twd; reg [7:0] tbe; reg [CTAG_W-1:0] ttag;
    reg ha, hv, hw; reg [63:0] hadr, hwd; reg [7:0] hbe;
    reg [NC-1:0] riv, ror; reg [NC*FW-1:0] rif;
    reg oi, mv, md; reg [63:0] mb, ms; reg [N-1:0] hr, hd;
    always @(posedge clk) begin
        rv<=req_valid; rs<=req_store; rvec<=req_vector; rix<=req_indexed; rsg<=req_signed; ramo<=req_amo; ack<=cpl_ack;
        rsz<=req_size; rdc<=req_dst_class; rh<=req_hint; rb<=req_base; ro<=req_offset; rsd<=req_sdata; rcmp<=req_cmp;
        rix_d<=req_index; rvd<=req_vdata; raop<=req_aop; rm<=req_mask; rrc<=req_rctx; rdi<=req_dst_idx;
        tv<=tm_valid; twe<=tm_we; ta<=tm_addr; twd<=tm_wdata; tbe<=tm_be; ttag<=tm_tag;
        ha<=host_allow; hv<=h_valid; hw<=h_we; hadr<=h_addr; hwd<=h_wdata; hbe<=h_be;
        riv<=r_in_valid_ext; ror<=r_out_ready_ext; rif<=r_in_flit_ext;
        oi<=other_idle; mv<=maint_valid; md<=memory_drained; mb<=maint_base; ms<=maint_size; hr<=home_ready; hd<=home_done;
    end
    // ---- LSU pair -> memory mux -> host mux -> core NI -> router ------------
    wire lsu_ready, lsu_ready_load, cpl_valid;
    wire [RCTX_W-1:0] cpl_rctx; wire [2:0] cpl_dstc; wire [7:0] cpl_dsti;
    wire [63:0] cpl_sdata; wire [VLEN*ELEN-1:0] cpl_vdata; wire [VLEN-1:0] cpl_lane;
    wire lm_v, lm_r, lm_we, lm_amo; wire [63:0] lm_a, lm_wd, lm_cmp; wire [7:0] lm_be;
    wire [CTAG_W-1:0] lm_tag; wire [1:0] lm_hint; wire [3:0] lm_aop;
    wire lm_rv; wire [CTAG_W-1:0] lm_rt; wire [63:0] lm_rd;
    tnca_lsu_pair #(.VLEN(VLEN),.ELEN(ELEN),.RCTX_W(RCTX_W),.MEM_DW(64),.MEM_BYTES(8),.MEM_SHIFT(3),
                    .NUM_BEATS(NUM_BEATS),.BEAT_W(CTAG_W)) u_lsu (
        .clk(clk),.rst_n(rst_n),.req_ready(lsu_ready),.req_ready_load(lsu_ready_load),.req_valid(rv),
        .req_store(rs),.req_vector(rvec),.req_indexed(rix),.req_size(rsz),.req_signed(rsg),.req_hint(rh),
        .req_base(rb),.req_offset(ro),.req_index(rix_d),.req_sdata(rsd),.req_amo(ramo),.req_aop(raop),
        .req_cmp(rcmp),.req_vdata(rvd),.req_mask(rm),.req_rctx(rrc),.req_dst_class(rdc),.req_dst_idx(rdi),
        .cpl_valid(cpl_valid),.cpl_ack(ack),.cpl_rctx(cpl_rctx),.cpl_dst_class(cpl_dstc),.cpl_dst_idx(cpl_dsti),
        .cpl_sdata(cpl_sdata),.cpl_vdata(cpl_vdata),.cpl_lane_en(cpl_lane),
        .mem_req_valid(lm_v),.mem_req_ready(lm_r),.mem_req_addr(lm_a),.mem_req_we(lm_we),.mem_req_be(lm_be),
        .mem_req_wdata(lm_wd),.mem_req_tag(lm_tag),.mem_req_hint(lm_hint),.mem_req_amo(lm_amo),.mem_req_aop(lm_aop),
        .mem_req_cmp(lm_cmp),.mem_rsp_valid(lm_rv),.mem_rsp_tag(lm_rt),.mem_rsp_rdata(lm_rd));
    wire tm_r, tm_rv; wire [CTAG_W-1:0] tm_rt; wire [63:0] tm_rd;
    wire c_v, c_r, c_we, c_amo; wire [63:0] c_a, c_wd, c_cmp; wire [7:0] c_be; wire [TAG_W-1:0] c_tag;
    wire [1:0] c_hint; wire [3:0] c_aop; wire c_rv; wire [TAG_W-1:0] c_rt; wire [63:0] c_rd;
    tnca_mem_mux #(.MEM_DW(64),.MEM_BYTES(8),.CTAG_W(CTAG_W),.TAG_W(TAG_W)) u_mmux (
        .a_req_valid(lm_v),.a_req_ready(lm_r),.a_req_addr(lm_a),.a_req_we(lm_we),.a_req_be(lm_be),
        .a_req_wdata(lm_wd),.a_req_tag(lm_tag),.a_req_hint(lm_hint),.a_req_amo(lm_amo),.a_req_aop(lm_aop),
        .a_req_cmp(lm_cmp),.a_rsp_valid(lm_rv),.a_rsp_tag(lm_rt),.a_rsp_rdata(lm_rd),
        .b_req_valid(tv),.b_req_ready(tm_r),.b_req_addr(ta),.b_req_we(twe),.b_req_be(tbe),.b_req_wdata(twd),
        .b_req_tag(ttag),.b_rsp_valid(tm_rv),.b_rsp_tag(tm_rt),.b_rsp_rdata(tm_rd),
        .m_req_valid(c_v),.m_req_ready(c_r),.m_req_addr(c_a),.m_req_we(c_we),.m_req_be(c_be),.m_req_wdata(c_wd),
        .m_req_tag(c_tag),.m_req_hint(c_hint),.m_req_amo(c_amo),.m_req_aop(c_aop),.m_req_cmp(c_cmp),
        .m_rsp_valid(c_rv),.m_rsp_tag(c_rt),.m_rsp_rdata(c_rd));
    reg host_allow_q; always @(posedge clk) host_allow_q <= rst_n && ha;
    wire h_ready, h_rsp_valid, host_busy; wire [63:0] h_rsp_data;
    wire n_v, n_r, n_we, n_amo; wire [63:0] n_a, n_wd, n_cmp; wire [7:0] n_be; wire [3:0] n_aop;
    wire [TAG_W-1:0] n_tag; wire n_rv; wire [TAG_W-1:0] n_rt; wire [63:0] n_rd;
    tnca_idle_host_mux #(.TAG_W(TAG_W)) u_host (
        .clk(clk),.rst_n(rst_n),.host_allow(host_allow_q),
        .c_valid(c_v),.c_ready(c_r),.c_addr(c_a),.c_wdata(c_wd),.c_cmp(c_cmp),.c_we(c_we),.c_amo(c_amo),
        .c_be(c_be),.c_aop(c_aop),.c_tag(c_tag),.c_rsp_valid(c_rv),.c_rsp_tag(c_rt),.c_rsp_data(c_rd),
        .h_valid(hv),.h_ready(h_ready),.h_we(hw),.h_addr(hadr),.h_wdata(hwd),.h_be(hbe),
        .h_rsp_valid(h_rsp_valid),.h_rsp_data(h_rsp_data),.host_busy(host_busy),
        .n_valid(n_v),.n_ready(n_r),.n_addr(n_a),.n_wdata(n_wd),.n_cmp(n_cmp),.n_we(n_we),.n_amo(n_amo),
        .n_be(n_be),.n_aop(n_aop),.n_tag(n_tag),.n_rsp_valid(n_rv),.n_rsp_tag(n_rt),.n_rsp_data(n_rd));
    wire ni_v, ni_r, ni_ov, ni_or; wire [FW-1:0] ni_f, ni_of;
    tnca_core_ni #(.NX(NX),.NY(NY),.XW(XW),.YW(YW),.TAG_W(TAG_W),.SRC_W(4),.MY_ID(0),.LINE_SH(6)) u_cni (
        .clk(clk),.rst_n(rst_n),.req_valid(n_v),.req_ready(n_r),.req_addr(n_a),.req_we(n_we),.req_be(n_be),
        .req_wdata(n_wd),.req_tag(n_tag),.req_amo(n_amo),.req_aop(n_aop),.req_cmp(n_cmp),
        .rsp_valid(n_rv),.rsp_tag(n_rt),.rsp_rdata(n_rd),
        .net_in_valid_req(ni_v),.net_in_ready_req(ni_r),.net_in_flit_req(ni_f),
        .net_in_valid_rsp(),.net_in_ready_rsp(1'b1),.net_in_flit_rsp(),
        .net_out_valid_req(1'b0),.net_out_ready_req(),.net_out_flit_req({FW{1'b0}}),
        .net_out_valid_rsp(ni_ov),.net_out_ready_rsp(ni_or),.net_out_flit_rsp(ni_of));
    // Router: local VN_REQ input is the core; the rest come from registers.
    wire [NC-1:0] in_valid = {riv[NC-1:1], ni_v};
    wire [NC*FW-1:0] in_flit = {rif[NC*FW-1:FW], ni_f};
    wire [NC-1:0] in_ready, out_valid;
    wire [NC*FW-1:0] out_flit;
    // local VN_RSP output returns to the core NI; the others leave by registers
    wire [NC-1:0] out_ready = {ror[NC-1:2], ni_or, ror[0]};
    assign ni_r = in_ready[0];
    assign ni_ov = out_valid[1];
    assign ni_of = out_flit[FW +: FW];
    tnca_router #(.XW(XW),.YW(YW),.PAYLOAD_W(`TNCA_NOC_PAYLOAD_W),.NV(2),.DEPTH(4),.AW(2),.X_HERE(0),.Y_HERE(0)) u_r (
        .clk(clk),.rst_n(rst_n),.in_valid(in_valid),.in_ready(in_ready),.in_flit(in_flit),
        .out_valid(out_valid),.out_ready(out_ready),.out_flit(out_flit));
    // ---- cache ownership handoff, quiesced by this LSU's idle state ---------
    wire maint_ready, maint_done, maint_busy; wire [N-1:0] home_valid; wire [63:0] home_base, home_size;
    tnca_cache_handoff #(.N(N)) ownership (
        .clk(clk),.rst_n(rst_n),.valid(mv),.ready(maint_ready),
        .quiescent(lsu_ready && !cpl_valid && oi && !host_busy),
        .base(mb),.size(ms),.home_valid(home_valid),.home_ready(hr),.home_done(hd),
        .home_base(home_base),.home_size(home_size),.memory_drained(md),.done(maint_done),.busy(maint_busy));
    // ---- fold every output into registers ------------------------------------
    wire [8191:0] all = {lsu_ready_load, cpl_valid, cpl_rctx, cpl_dstc, cpl_dsti, cpl_sdata, cpl_vdata, cpl_lane,
                         tm_r, tm_rv, tm_rt, tm_rd, h_ready, h_rsp_valid, h_rsp_data, c_hint,
                         in_ready, out_valid, out_flit,
                         maint_ready, maint_done, maint_busy, home_valid, home_base, home_size};
    integer k;
    always @(posedge clk)
        for (k=0; k<1024; k=k+1)
            observed[k] <= ^{all[k], all[k+1024], all[k+2048], all[k+3072],
                             all[k+4096], all[k+5120], all[k+6144], all[k+7168]};
endmodule
`default_nettype wire
