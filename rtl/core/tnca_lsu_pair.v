//==============================================================================
// tnca_lsu_pair -- two load/store units behind one request port, so a load can
// start while the access ahead of it waits for memory (R39-08, LSU_INFLIGHT=2)
//
// tnca_lsu keeps one instruction in flight: it plans and issues the beats of an
// access, then waits for every response, then completes, and only then takes
// the next. On the board the wait is most of a scalar load and half of a
// vector one. Here a second unit takes the next access as soon as the first
// has issued all its beats, so the second's beats go out while the first's
// responses come back.
//
// ORDERING
// --------
// Only a plain load overtakes, and only plain loads are overtaken: a load
// accepted while the other unit drains a load reads memory nothing in flight
// writes. A store or an atomic waits for both units to be idle, as it always
// has, and so does anything behind one -- so no access is reordered with a
// write, and `fence` (which waits for req_ready) still means every access is
// complete. req_ready keeps that meaning: both units idle. req_ready_load is
// the new, weaker one the core uses for a load.
//
// Only one unit issues beats at a time (the second is accepted only once the
// first has issued its last), so the memory port is the issuing unit's. The
// unit's beat tag gets the unit number as its top bit, and a response goes
// back by it. Two completions waiting together leave oldest first.
//==============================================================================
`include "tnca_defs.vh"
`default_nettype none

module tnca_lsu_pair #(
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter MEM_SHIFT = 3,
    parameter NUM_BEATS = 8,          // outstanding beats per unit
    parameter BEAT_W    = 4           // the port's tag width: unit bit + beat tag
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
    input  wire [MEM_DW-1:0]        mem_rsp_rdata
);
    localparam UB = BEAT_W - 1;       // a unit's own tag width

    wire [1:0] u_ready, u_drain, u_load, u_cplv, u_mvalid, u_take, u_ack;
    wire [RCTX_W-1:0]    u_cpl_rctx  [0:1];
    wire [2:0]           u_cpl_dstc  [0:1];
    wire [7:0]           u_cpl_dsti  [0:1];
    wire [63:0]          u_cpl_sdata [0:1];
    wire [VLEN*ELEN-1:0] u_cpl_vdata [0:1];
    wire [VLEN-1:0]      u_cpl_lane  [0:1];
    wire [63:0]          u_addr  [0:1];
    wire [1:0]           u_we, u_amo;
    wire [MEM_BYTES-1:0] u_be    [0:1];
    wire [MEM_DW-1:0]    u_wdata [0:1];
    wire [UB-1:0]        u_tag   [0:1];
    wire [1:0]           u_hint  [0:1];
    wire [`TNCA_AOP_W-1:0] u_aop [0:1];
    wire [63:0]          u_cmp   [0:1];

    // An idle unit with the other one draining a load takes a load.
    assign req_ready      = &u_ready;
    assign req_ready_load = (&u_ready)
                          | (u_ready[0] & u_drain[1] & u_load[1])
                          | (u_ready[1] & u_drain[0] & u_load[0]);
    assign u_take[0] = req_valid && u_ready[0];
    assign u_take[1] = req_valid && !u_ready[0] && u_ready[1];

    // Which unit holds the older access, for the completion order.
    reg old;
    always @(posedge clk) begin
        if (!rst_n)
            old <= 1'b0;
        else if (u_take[0] && u_ready[1])
            old <= 1'b0;                       // the only one in flight
        else if (u_take[1] && u_ready[0])
            old <= 1'b1;
        else if (u_ack[old])
            old <= ~old;                       // the older completed
    end

    wire pick = u_cplv[old] ? old : ~old;
    assign cpl_valid     = |u_cplv;
    assign u_ack[0]      = cpl_ack && (pick == 1'b0) && u_cplv[0];
    assign u_ack[1]      = cpl_ack && (pick == 1'b1) && u_cplv[1];
    assign cpl_rctx      = u_cpl_rctx[pick];
    assign cpl_dst_class = u_cpl_dstc[pick];
    assign cpl_dst_idx   = u_cpl_dsti[pick];
    assign cpl_sdata     = u_cpl_sdata[pick];
    assign cpl_vdata     = u_cpl_vdata[pick];
    assign cpl_lane_en   = u_cpl_lane[pick];

    // One unit issues at a time; the port is its.
    wire mu = u_mvalid[1];
    assign mem_req_valid = |u_mvalid;
    assign mem_req_addr  = u_addr[mu];
    assign mem_req_we    = u_we[mu];
    assign mem_req_be    = u_be[mu];
    assign mem_req_wdata = u_wdata[mu];
    assign mem_req_tag   = {mu, u_tag[mu]};
    assign mem_req_hint  = u_hint[mu];
    assign mem_req_amo   = u_amo[mu];
    assign mem_req_aop   = u_aop[mu];
    assign mem_req_cmp   = u_cmp[mu];

    genvar g;
    generate
        for (g = 0; g < 2; g = g + 1) begin : g_unit
            tnca_lsu #(
                .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
                .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES), .MEM_SHIFT(MEM_SHIFT),
                .NUM_BEATS(NUM_BEATS), .BEAT_W(UB)
            ) u (
                .clk(clk), .rst_n(rst_n),
                .req_ready(u_ready[g]), .req_valid(u_take[g]),
                .req_store(req_store), .req_vector(req_vector),
                .req_indexed(req_indexed), .req_size(req_size),
                .req_signed(req_signed), .req_hint(req_hint),
                .req_base(req_base), .req_offset(req_offset),
                .req_index(req_index), .req_sdata(req_sdata),
                .req_amo(req_amo), .req_aop(req_aop), .req_cmp(req_cmp),
                .req_vdata(req_vdata), .req_mask(req_mask),
                .req_rctx(req_rctx), .req_dst_class(req_dst_class),
                .req_dst_idx(req_dst_idx),
                .cpl_valid(u_cplv[g]), .cpl_ack(u_ack[g]),
                .cpl_rctx(u_cpl_rctx[g]), .cpl_dst_class(u_cpl_dstc[g]),
                .cpl_dst_idx(u_cpl_dsti[g]), .cpl_sdata(u_cpl_sdata[g]),
                .cpl_vdata(u_cpl_vdata[g]), .cpl_lane_en(u_cpl_lane[g]),
                .o_draining(u_drain[g]), .o_load(u_load[g]),
                .mem_req_valid(u_mvalid[g]),
                .mem_req_ready(mem_req_ready && (mu == g)),
                .mem_req_addr(u_addr[g]), .mem_req_we(u_we[g]),
                .mem_req_be(u_be[g]), .mem_req_wdata(u_wdata[g]),
                .mem_req_tag(u_tag[g]), .mem_req_hint(u_hint[g]),
                .mem_req_amo(u_amo[g]), .mem_req_aop(u_aop[g]),
                .mem_req_cmp(u_cmp[g]),
                .mem_rsp_valid(mem_rsp_valid && (mem_rsp_tag[UB] == g)),
                .mem_rsp_tag(mem_rsp_tag[UB-1:0]),
                .mem_rsp_rdata(mem_rsp_rdata)
            );
        end
    endgenerate

`ifndef SYNTHESIS
    // synthesis translate_off
    always @(posedge clk)
        if (rst_n && (&u_mvalid))
            $display("FAIL tnca_lsu_pair assert @%0t: both units issuing at once", $time);
    // synthesis translate_on
`endif

endmodule

`default_nettype wire
