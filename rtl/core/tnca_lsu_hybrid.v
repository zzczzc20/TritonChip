// Compact vector-line path plus the original narrow pair. Mode transitions
// drain the other path, preserving program order without alias speculation.
`include "tnca_defs.vh"
`default_nettype none
module tnca_lsu_hybrid #(
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter MEM_DW    = 512,
    parameter MEM_BYTES = 64,
    parameter MEM_SHIFT = 6,
    parameter NUM_BEATS = 4,          // outstanding beats per unit
    parameter BEAT_W    = 4,          // the port's tag width: unit bit + beat tag
    parameter REQ_REG   = 1           // registered request boundary per unit
) (
    input  wire                     clk,
    input  wire                     rst_n,

    output wire                     req_ready,       // both idle: anything
    output wire                     req_ready_load,  // a plain load now
    input  wire                     req_valid,
    input  wire                     req_store,
    input  wire                     req_vector,
    input  wire                     req_indexed,
    input  wire [2:0]               req_size,
    input  wire                     req_signed,
    input  wire [1:0]               req_hint,
    input  wire [63:0]              req_base,
    input  wire [63:0]              req_offset,
    input  wire [VLEN*ELEN-1:0]     req_index,
    input  wire [63:0]              req_sdata,
    input  wire                     req_amo,
    input  wire [`TNCA_AOP_W-1:0]   req_aop,
    input  wire [63:0]              req_cmp,
    input  wire [VLEN*ELEN-1:0]     req_vdata,
    input  wire [VLEN-1:0]          req_mask,
    input  wire [RCTX_W-1:0]        req_rctx,
    input  wire [2:0]               req_dst_class,
    input  wire [7:0]               req_dst_idx,

    output wire                     cpl_valid,
    input  wire                     cpl_ack,
    output wire [RCTX_W-1:0]        cpl_rctx,
    output wire [2:0]               cpl_dst_class,
    output wire [7:0]               cpl_dst_idx,
    output wire [63:0]              cpl_sdata,
    output wire [VLEN*ELEN-1:0]     cpl_vdata,
    output wire [VLEN-1:0]          cpl_lane_en,

    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [MEM_BYTES-1:0]     mem_req_be,
    output wire [MEM_DW-1:0]        mem_req_wdata,
    output wire [BEAT_W-1:0]        mem_req_tag,
    output wire [1:0]               mem_req_hint,
    output wire                     mem_req_amo,
    output wire [`TNCA_AOP_W-1:0]   mem_req_aop,
    output wire [63:0]              mem_req_cmp,

    input  wire                     mem_rsp_valid,
    input  wire [BEAT_W-1:0]        mem_rsp_tag,
    input  wire [MEM_DW-1:0]        mem_rsp_rdata,
    output wire [31:0] perf_fast_accepted,perf_fast_buffered,perf_fast_request_stall,perf_fast_slot_stall,perf_fast_peak,
    output reg [31:0] perf_slow_accepted,perf_retired,perf_fast_retired,perf_switch_stall,perf_writeback_stall
);

    localparam NT=BEAT_W-1;
    wire fast_eligible,fast_ready,fast_quiet,slow_idle,slow_ready;
    reg out_valid,out_fast,atomic_pending;
    wire fast_mode=slow_idle && !(out_valid && !out_fast);
    wire slow_mode=fast_quiet && !(out_valid && out_fast);
    assign req_ready=fast_quiet && slow_idle && !out_valid && !atomic_pending;
    assign req_ready_load=!atomic_pending && (fast_eligible ? (fast_mode && fast_ready) : (slow_mode && slow_ready));
    wire fast_present=req_valid && fast_eligible && fast_mode && !atomic_pending;
    wire fast_admit=fast_present && fast_ready;
    wire slow_admit=req_valid && !fast_eligible && (req_amo ? req_ready : req_ready_load);
    wire fv,sv,fwe,swe,samo,fmv,smv;
    wire [RCTX_W-1:0] fc,sc;
    wire [2:0] fd,sd;
    wire [7:0] fi,si;
    wire [511:0] fdata,sdata,fwdata,swdata;
    wire [63:0] scalar,fscalar,fa,sa,fb,sb,scmp;
    wire [15:0] fl,sl;
    wire [NT-1:0] ft,st;
    wire [1:0] fh,sh;
    wire [3:0] sop;
    // Reserve a real completion landing register before acknowledging either
    // inner engine. It holds every output stable until the core RF accepts it.
    reg [RCTX_W-1:0] out_rctx;
    reg [2:0] out_dst;
    reg [7:0] out_idx;
    reg [63:0] out_scalar;
    reg [511:0] out_vector;
    reg [15:0] out_mask;
    wire buffer_ready=!out_valid || cpl_ack;
    assign cpl_valid=out_valid;
    assign cpl_rctx=out_rctx;
    assign cpl_dst_class=out_dst;
    assign cpl_dst_idx=out_idx;
    assign cpl_sdata=out_scalar;
    assign cpl_vdata=out_vector;
    assign cpl_lane_en=out_mask;
    always @(posedge clk) begin
        if(!rst_n) begin out_valid<=0;out_fast<=0;atomic_pending<=0;end
        else begin
            if(out_valid && cpl_ack) begin out_valid<=0;if(atomic_pending)atomic_pending<=0;end
            if(slow_admit && req_amo) atomic_pending<=1;
            if(buffer_ready && (fv || sv)) begin
                out_valid<=1;out_fast<=fv;
                out_rctx<=fv?fc:sc;out_dst<=fv?fd:sd;out_idx<=fv?fi:si;
                out_scalar<=fv?fscalar:scalar;out_vector<=fv?fdata:sdata;out_mask<=fv?fl:sl;
            end
        end
    end
    assign mem_req_valid=fmv || smv;
    assign mem_req_addr=fmv ? fa : sa;
    assign mem_req_we=fmv ? fwe : swe;
    assign mem_req_be=fmv ? fb : sb;
    assign mem_req_wdata=fmv ? fwdata : swdata;
    assign mem_req_tag=fmv ? {1'b1,ft} : {1'b0,st};
    assign mem_req_hint=fmv ? fh : sh;
    assign mem_req_amo=!fmv && samo;
    assign mem_req_aop=fmv ? 4'd0 : sop;
    assign mem_req_cmp=fmv ? 64'd0 : scmp;
    tnca_lsu_line #(.SLOTS(1<<NT),.TAG_W(NT),.RCTX_W(RCTX_W)) fast(
        .clk(clk),.rst_n(rst_n),.req_valid(fast_present),.req_eligible(fast_eligible),.req_ready(fast_ready),.quiescent(fast_quiet),
        .req_store(req_store),.req_vector(req_vector),.req_indexed(req_indexed),.req_amo(req_amo),
        .req_size(req_size),.req_signed(req_signed),.req_hint(req_hint),.req_base(req_base),.req_offset(req_offset),
        .req_vdata(req_vdata),.req_mask(req_mask),.req_rctx(req_rctx),.req_dst_class(req_dst_class),.req_dst_idx(req_dst_idx),
        .cpl_valid(fv),.cpl_ack(buffer_ready && fv),.cpl_rctx(fc),.cpl_dst_class(fd),.cpl_dst_idx(fi),.cpl_vdata(fdata),.cpl_sdata(fscalar),.cpl_lane_en(fl),
        .mem_req_valid(fmv),.mem_req_ready(mem_req_ready),.mem_req_addr(fa),.mem_req_we(fwe),.mem_req_be(fb),.mem_req_wdata(fwdata),
        .mem_req_tag(ft),.mem_req_hint(fh),.mem_rsp_valid(mem_rsp_valid && mem_rsp_tag[BEAT_W-1]),
        .mem_rsp_tag(mem_rsp_tag[NT-1:0]),.mem_rsp_rdata(mem_rsp_rdata),
        .perf_accepted(perf_fast_accepted),.perf_completed(perf_fast_buffered),
        .perf_request_stall(perf_fast_request_stall),.perf_slot_stall(perf_fast_slot_stall),.perf_peak(perf_fast_peak));
    wire nv,nready,nwe,namo,nrv;
    wire [63:0] na,nwd,ncmp,nrd;
    wire [7:0] nb;
    wire [NT-1:0] nt,nrt;
    wire [1:0] nh;
    wire [3:0] nop;
    tnca_lsu_pair #(.VLEN(VLEN),.ELEN(ELEN),.RCTX_W(RCTX_W),.NUM_BEATS(NUM_BEATS),.BEAT_W(NT),.REQ_REG(REQ_REG)) slow(
        .clk(clk),.rst_n(rst_n),.req_ready(slow_idle),.req_ready_load(slow_ready),.req_valid(slow_admit),
        .req_store(req_store),.req_vector(req_vector),.req_indexed(req_indexed),.req_size(req_size),.req_signed(req_signed),
        .req_hint(req_hint),.req_base(req_base),.req_offset(req_offset),.req_index(req_index),.req_sdata(req_sdata),
        .req_amo(req_amo),.req_aop(req_aop),.req_cmp(req_cmp),.req_vdata(req_vdata),.req_mask(req_mask),
        .req_rctx(req_rctx),.req_dst_class(req_dst_class),.req_dst_idx(req_dst_idx),
        .cpl_valid(sv),.cpl_ack(buffer_ready && sv && !fv),.cpl_rctx(sc),.cpl_dst_class(sd),.cpl_dst_idx(si),
        .cpl_sdata(scalar),.cpl_vdata(sdata),.cpl_lane_en(sl),
        .mem_req_valid(nv),.mem_req_ready(nready),.mem_req_addr(na),.mem_req_we(nwe),.mem_req_be(nb),.mem_req_wdata(nwd),
        .mem_req_tag(nt),.mem_req_hint(nh),.mem_req_amo(namo),.mem_req_aop(nop),.mem_req_cmp(ncmp),
        .mem_rsp_valid(nrv),.mem_rsp_tag(nrt),.mem_rsp_rdata(nrd));
    tnca_word_line_adapter #(.TAG_W(NT)) adapter(
        .clk(clk),.rst_n(rst_n),.n_req_valid(nv),.n_req_ready(nready),.n_req_addr(na),.n_req_wdata(nwd),
        .n_req_we(nwe),.n_req_amo(namo),.n_req_be(nb),.n_req_tag(nt),.n_req_hint(nh),.n_req_aop(nop),.n_req_cmp(ncmp),
        .n_rsp_valid(nrv),.n_rsp_tag(nrt),.n_rsp_rdata(nrd),
        .w_req_valid(smv),.w_req_ready(mem_req_ready && !fmv),.w_req_addr(sa),.w_req_be(sb),.w_req_wdata(swdata),
        .w_req_we(swe),.w_req_amo(samo),.w_req_tag(st),.w_req_hint(sh),.w_req_aop(sop),.w_req_cmp(scmp),
        .w_rsp_valid(mem_rsp_valid && !mem_rsp_tag[BEAT_W-1]),.w_rsp_tag(mem_rsp_tag[NT-1:0]),.w_rsp_rdata(mem_rsp_rdata));
    always @(posedge clk) begin
        if(!rst_n) begin
            perf_slow_accepted<=0;perf_retired<=0;perf_fast_retired<=0;perf_switch_stall<=0;perf_writeback_stall<=0;
        end else begin
            if(slow_admit) perf_slow_accepted<=perf_slow_accepted+1;
            if(out_valid && cpl_ack) begin perf_retired<=perf_retired+1;if(out_fast)perf_fast_retired<=perf_fast_retired+1;end
            if(req_valid && (atomic_pending || (fast_eligible ? !fast_mode : !slow_mode)))perf_switch_stall<=perf_switch_stall+1;
            if(out_valid && !cpl_ack)perf_writeback_stall<=perf_writeback_stall+1;
        end
    end
`ifndef SYNTHESIS
    initial if(VLEN!=16 || ELEN!=32 || MEM_DW!=512 || MEM_BYTES!=64 || MEM_SHIFT!=6 || BEAT_W<4 || BEAT_W>7 || NUM_BEATS<1 || NUM_BEATS>(1<<(BEAT_W-2)))
        $fatal(1,"invalid hybrid LSU geometry");
    always @(posedge clk) if(rst_n) begin
        if(fmv && smv) $fatal(1,"hybrid LSU paths issue concurrently");
        if(fv && sv) $fatal(1,"hybrid LSU modes overlap completion");
        if(fast_admit && !slow_idle) $fatal(1,"hybrid LSU crossed a slow command");
        if(slow_admit && !fast_quiet) $fatal(1,"hybrid LSU crossed a fast command");
    end
`endif
endmodule
`default_nettype wire
