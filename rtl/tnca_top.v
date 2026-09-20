//==============================================================================
// tnca_top -- NX x NY tiles on a mesh, with one shared memory port
//
// This is the first configuration in which the architecture's central claim is
// actually testable: there is no private data cache anywhere, every physical
// line has exactly one home slice, and a core reaching a line it does not host
// crosses the mesh to get it (arch overview sec 1.1, sec 10.1).
//
// What the memory hierarchy looks like from a lane, end to end:
//
//   vector register  <- LSU beat  <- core NI  <- mesh  <- home NUCA slice
//                                                          <- DRAM
//
// and nothing in that chain is software-addressable except the registers at one
// end and DRAM at the other.
//
// The instruction port and the DRAM port are left to the surrounding level, so
// this file stays free of behavioural memory.
//==============================================================================
`include "tnca_noc_pkg.vh"

module tnca_top #(
    parameter NX          = 2,
    parameter NY          = 2,
    parameter XW          = 2,
    parameter YW          = 2,
    parameter NUM_RCTX    = 2,
    parameter RCTX_W      = 1,
    parameter VLEN        = 16,
    parameter ELEN        = 32,
    parameter LW          = 4,
    parameter VREGS_PER_RCTX = 64,
    parameter PHYS_VREGS  = 128,
    parameter HAS_SFU     = 0,
    parameter HAS_ATOMICS = 1,
    parameter PHYS_W      = 7,
    parameter LSU_BEATS   = 8,
    parameter LSU_TAG_W   = 5,
    parameter LINE_BYTES  = 64,
    parameter LINE_BITS   = 512,
    parameter SETS        = 64,
    parameter SET_W       = 6,
    parameter WAYS        = 8,
    parameter WAY_W       = 3,
    parameter MSHR_N      = 8,
    parameter MSHR_W      = 3,
    parameter TARGET_N    = 16,
    parameter TARGET_W    = 4,
    parameter RQ_N        = 8,
    parameter RQ_W        = 3,
    parameter NOC_DEPTH   = 4,
    parameter NOC_AW      = 2
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     launch_valid,
    input  wire [63:0]              launch_pc,
    input  wire [NUM_RCTX-1:0]      launch_role_mask,
    input  wire [63:0]              launch_arg_base,
    input  wire [NUM_RCTX*PHYS_W-1:0] launch_vbase,
    output wire                     all_idle,

    output wire [NX*NY-1:0]         fault_valid,
    output wire [NX*NY*8-1:0]       fault_code,
    output wire [NX*NY*64-1:0]      fault_pc,
    output wire [NX*NY*24-1:0]      fault_detail,

    // ---- per-tile instruction ports -----------------------------------------
    output wire [NX*NY*64-1:0]      imem_addr,
    input  wire [NX*NY*64-1:0]      imem_data,

    // ---- one shared DRAM port -----------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [LINE_BITS-1:0]     mem_req_wdata,
    output wire [MSHR_W-1:0]        mem_req_tag,
    output wire [1:0]               mem_req_src,
    input  wire                     mem_rsp_valid,
    input  wire [MSHR_W-1:0]        mem_rsp_tag,
    input  wire [1:0]               mem_rsp_src,
    input  wire [LINE_BITS-1:0]     mem_rsp_rdata,

    // ---- counters -----------------------------------------------------------
    output wire [NX*NY*64-1:0]      perf_cycles,
    output wire [NX*NY*64-1:0]      perf_issued,
    output wire [NX*NY*32-1:0]      perf_hit,
    output wire [NX*NY*32-1:0]      perf_miss,
    output wire [NX*NY*32-1:0]      perf_merge,
    output wire [NX*NY*32-1:0]      perf_wb,
    output wire [NX*NY*32-1:0]      perf_mshr_full,
    output wire [NX*NY*32-1:0]      perf_hit_past_block
);

    localparam N  = NX * NY;
    localparam NV = `TNCA_NOC_NV;
    localparam FW = XW + YW + `TNCA_NOC_PAYLOAD_W;

    wire [N*NV-1:0]     li_valid, li_ready, lo_valid, lo_ready;
    wire [N*NV*FW-1:0]  li_flit,  lo_flit;

    wire [N-1:0]            m_req_valid, m_req_ready, m_req_we;
    wire [N*64-1:0]         m_req_addr;
    wire [N*LINE_BITS-1:0]  m_req_wdata;
    wire [N*MSHR_W-1:0]     m_req_tag;
    wire [N-1:0]            m_rsp_valid;
    wire [N*MSHR_W-1:0]     m_rsp_tag;
    wire [N*LINE_BITS-1:0]  m_rsp_rdata;

    wire [N-1:0] tile_idle;
    assign all_idle = &tile_idle;

    genvar t;
    generate
        for (t = 0; t < N; t = t + 1) begin : g_tile
            tnca_tile #(
                .TILE_ID(t), .NUM_TILES(N), .XW(XW), .YW(YW),
                .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN),
                .LW(LW), .VREGS_PER_RCTX(VREGS_PER_RCTX),
                .PHYS_VREGS(PHYS_VREGS), .PHYS_W(PHYS_W),
                .HAS_SFU(HAS_SFU), .HAS_ATOMICS(HAS_ATOMICS),
                .LSU_BEATS(LSU_BEATS), .LSU_TAG_W(LSU_TAG_W),
                .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
                .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
                .MSHR_N(MSHR_N), .MSHR_W(MSHR_W),
                .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
                .RQ_N(RQ_N), .RQ_W(RQ_W)
            ) u_tile (
                .clk(clk), .rst_n(rst_n),
                .launch_valid(launch_valid), .launch_pc(launch_pc),
                .launch_role_mask(launch_role_mask),
                .launch_arg_base(launch_arg_base),
                .launch_vbase(launch_vbase),
                .core_idle(tile_idle[t]),
                .fault_valid(fault_valid[t]),
                .fault_code(fault_code[t*8 +: 8]),
                .fault_pc(fault_pc[t*64 +: 64]),
                .fault_detail(fault_detail[t*24 +: 24]),
                .imem_addr(imem_addr[t*64 +: 64]),
                .imem_data(imem_data[t*64 +: 64]),
                .l_in_valid(li_valid[t*NV +: NV]),
                .l_in_ready(li_ready[t*NV +: NV]),
                .l_in_flit(li_flit[t*NV*FW +: NV*FW]),
                .l_out_valid(lo_valid[t*NV +: NV]),
                .l_out_ready(lo_ready[t*NV +: NV]),
                .l_out_flit(lo_flit[t*NV*FW +: NV*FW]),
                .mem_req_valid(m_req_valid[t]), .mem_req_ready(m_req_ready[t]),
                .mem_req_addr(m_req_addr[t*64 +: 64]),
                .mem_req_we(m_req_we[t]),
                .mem_req_wdata(m_req_wdata[t*LINE_BITS +: LINE_BITS]),
                .mem_req_tag(m_req_tag[t*MSHR_W +: MSHR_W]),
                .mem_rsp_valid(m_rsp_valid[t]),
                .mem_rsp_tag(m_rsp_tag[t*MSHR_W +: MSHR_W]),
                .mem_rsp_rdata(m_rsp_rdata[t*LINE_BITS +: LINE_BITS]),
                .perf_cycles(perf_cycles[t*64 +: 64]),
                .perf_issued(perf_issued[t*64 +: 64]),
                .perf_hit(perf_hit[t*32 +: 32]),
                .perf_miss(perf_miss[t*32 +: 32]),
                .perf_merge(perf_merge[t*32 +: 32]),
                .perf_wb(perf_wb[t*32 +: 32]),
                .perf_mshr_full(perf_mshr_full[t*32 +: 32]),
                .perf_hit_past_block(perf_hit_past_block[t*32 +: 32])
            );
        end
    endgenerate

    tnca_mesh #(
        .NX(NX), .NY(NY), .XW(XW), .YW(YW),
        .PAYLOAD_W(`TNCA_NOC_PAYLOAD_W), .NV(NV),
        .DEPTH(NOC_DEPTH), .AW(NOC_AW)
    ) u_mesh (
        .clk(clk), .rst_n(rst_n),
        .l_in_valid(li_valid), .l_in_ready(li_ready), .l_in_flit(li_flit),
        .l_out_valid(lo_valid), .l_out_ready(lo_ready), .l_out_flit(lo_flit)
    );

    tnca_mem_arbiter #(
        .N(N), .N_W(2), .ADDR_W(64), .LINE_BITS(LINE_BITS), .TAG_W(MSHR_W)
    ) u_marb (
        .clk(clk), .rst_n(rst_n),
        .s_req_valid(m_req_valid), .s_req_ready(m_req_ready),
        .s_req_addr(m_req_addr), .s_req_we(m_req_we),
        .s_req_wdata(m_req_wdata), .s_req_tag(m_req_tag),
        .s_rsp_valid(m_rsp_valid), .s_rsp_tag(m_rsp_tag),
        .s_rsp_rdata(m_rsp_rdata),
        .m_req_valid(mem_req_valid), .m_req_ready(mem_req_ready),
        .m_req_addr(mem_req_addr), .m_req_we(mem_req_we),
        .m_req_wdata(mem_req_wdata), .m_req_tag(mem_req_tag),
        .m_req_src(mem_req_src),
        .m_rsp_valid(mem_rsp_valid), .m_rsp_tag(mem_rsp_tag),
        .m_rsp_src(mem_rsp_src), .m_rsp_rdata(mem_rsp_rdata)
    );

endmodule
