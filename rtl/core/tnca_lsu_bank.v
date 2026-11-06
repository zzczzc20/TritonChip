// Finite multi-command LSU for the wide-path candidate.
// Ordinary accesses retain issue order; atomics and fence use full quiescence.
`include "tnca_defs.vh"
`default_nettype none
module tnca_lsu_bank #(
    parameter UNITS     = 4,
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter MEM_SHIFT = 3,
    parameter NUM_BEATS = 16/UNITS,          // outstanding beats per unit
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
    input  wire [MEM_DW-1:0]        mem_rsp_rdata
);
    localparam UW = $clog2(UNITS);
    localparam UB = BEAT_W - UW;       // a unit's own tag width

    wire [UNITS-1:0] u_ready, u_drain, u_atomic, u_cplv, u_mvalid, u_take, u_ack;
    wire [RCTX_W-1:0]    u_cpl_rctx  [0:UNITS-1];
    wire [2:0]           u_cpl_dstc  [0:UNITS-1];
    wire [7:0]           u_cpl_dsti  [0:UNITS-1];
    wire [63:0]          u_cpl_sdata [0:UNITS-1];
    wire [VLEN*ELEN-1:0] u_cpl_vdata [0:UNITS-1];
    wire [VLEN-1:0]      u_cpl_lane  [0:UNITS-1];
    wire [63:0]          u_addr  [0:UNITS-1];
    wire [UNITS-1:0]     u_we, u_amo;
    wire [MEM_BYTES-1:0] u_be    [0:UNITS-1];
    wire [MEM_DW-1:0]    u_wdata [0:UNITS-1];
    wire [UB-1:0]        u_tag   [0:UNITS-1];
    wire [1:0]           u_hint  [0:UNITS-1];
    wire [`TNCA_AOP_W-1:0] u_aop [0:UNITS-1];
    wire [63:0]          u_cmp   [0:UNITS-1];

    // Admission preserves complete-command request order. Only one unit may
    // still issue; older units can wait for replies. Atomics are barriers.
    reg free_found, drained;
    reg [UW-1:0] free_unit, mu;
    always @* begin
        free_found=0; free_unit=0; drained=1; mu=0;
        for (integer k=UNITS-1;k>=0;k=k-1) begin
            if (u_ready[k]) begin free_found=1; free_unit=k; end
            if (!u_ready[k] && (!u_drain[k] || u_atomic[k])) drained=0;
            if (u_mvalid[k]) mu=k;
        end
    end
    assign req_ready = &u_ready;
    assign req_ready_load = free_found && drained;
    wire admit = req_valid && (req_amo ? req_ready : req_ready_load);

    // Oldest ready completion wins the single RF writeback port. A delayed
    // reply cannot block unrelated completed destinations. Core scoreboards
    // retain the same tagged out-of-order completion contract as the pair.
    // Compact the finite age queue on acknowledgement; no wrapping age keys.
    reg [UW-1:0] order [0:UNITS-1];
    reg [UW:0] count;
    reg [UW-1:0] pick, position;
    reg found;
    reg held;
    reg [UW-1:0] held_pick, held_position;
    always @* begin
        pick=0; position=0; found=0;
        for (integer k=0;k<UNITS;k=k+1)
            if (k<count && !found)
                if (u_cplv[order[k]]) begin pick=order[k]; position=k; found=1; end
        if (held) begin pick=held_pick; position=held_position; found=1; end
    end
    assign cpl_valid = found;
    wire retire = cpl_valid && cpl_ack;
    always @(posedge clk) begin
        if (!rst_n) begin held<=0; held_pick<=0; held_position<=0; end
        else if (retire) held<=0;
        else if (found && !held) begin held<=1; held_pick<=pick; held_position<=position; end
    end
    always @(posedge clk) begin
        if (!rst_n) count<=0;
        else begin
            if (retire)
                for (integer k=0;k<UNITS-1;k=k+1)
                    if (k>=position) order[k]<=order[k+1];
            if (admit) order[count-retire]<=free_unit;
            case ({admit,retire})
                2'b10: count<=count+1'b1;
                2'b01: count<=count-1'b1;
                default: count<=count;
            endcase
        end
    end
    assign cpl_rctx = u_cpl_rctx[pick];
    assign cpl_dst_class = u_cpl_dstc[pick];
    assign cpl_dst_idx = u_cpl_dsti[pick];
    assign cpl_sdata = u_cpl_sdata[pick];
    assign cpl_vdata = u_cpl_vdata[pick];
    assign cpl_lane_en = u_cpl_lane[pick];
    assign mem_req_valid = |u_mvalid;
    assign mem_req_addr = u_addr[mu];
    assign mem_req_we = u_we[mu];
    assign mem_req_be = u_be[mu];
    assign mem_req_wdata = u_wdata[mu];
    assign mem_req_tag = {mu,u_tag[mu]};
    assign mem_req_hint = u_hint[mu];
    assign mem_req_amo = u_amo[mu];
    assign mem_req_aop = u_aop[mu];
    assign mem_req_cmp = u_cmp[mu];

    genvar g;
    generate
        for (g = 0; g < UNITS; g = g + 1) begin : g_unit
            assign u_take[g] = admit && free_unit == g;
            assign u_ack[g] = retire && pick == g;
            tnca_lsu #(
                .VLEN(VLEN), .ELEN(ELEN), .RCTX_W(RCTX_W),
                .MEM_DW(MEM_DW), .MEM_BYTES(MEM_BYTES), .MEM_SHIFT(MEM_SHIFT),
                .NUM_BEATS(NUM_BEATS), .BEAT_W(UB), .REQ_REG(REQ_REG)
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
                .o_draining(u_drain[g]), .o_amo(u_atomic[g]),
                .mem_req_valid(u_mvalid[g]),
                .mem_req_ready(mem_req_ready && (mu == g)),
                .mem_req_addr(u_addr[g]), .mem_req_we(u_we[g]),
                .mem_req_be(u_be[g]), .mem_req_wdata(u_wdata[g]),
                .mem_req_tag(u_tag[g]), .mem_req_hint(u_hint[g]),
                .mem_req_amo(u_amo[g]), .mem_req_aop(u_aop[g]),
                .mem_req_cmp(u_cmp[g]),
                .mem_rsp_valid(mem_rsp_valid && (mem_rsp_tag[BEAT_W-1:UB] == g)),
                .mem_rsp_tag(mem_rsp_tag[UB-1:0]),
                .mem_rsp_rdata(mem_rsp_rdata)
            );
        end
    endgenerate

`ifndef SYNTHESIS
    initial begin
        if (!(UNITS==2 || UNITS==4 || UNITS==8) || UB<1 || NUM_BEATS<1 || NUM_BEATS>(1<<UB))
            $fatal(1,"LSU bank units/beat slots exceed tag capacity");
    end
    always @(posedge clk) if (rst_n) begin
        if ((u_mvalid & (u_mvalid-1'b1)) != 0) $fatal(1,"LSU bank concurrent issuers");
        if (admit && count==UNITS) $fatal(1,"LSU bank completion order overflow");
        if (retire && count==0) $fatal(1,"LSU bank completion underflow");
    end
`endif
endmodule
`default_nettype wire
