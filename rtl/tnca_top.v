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
    parameter GLOBAL_GRID = 0,
    parameter HOST_ACCESS = 0,
    parameter NX          = 2,
    parameter NY          = 2,
    parameter MEM_SRC_W   = (NX*NY>4 ? $clog2(NX*NY) : 2),
    parameter XW          = 2,
    parameter YW          = 2,
    parameter NUM_RCTX    = 2,
    parameter RCTX_W      = 1,
    parameter VLEN        = 16,
    parameter ELEN        = 32,
    parameter LW          = 4,
    parameter VREGS_PER_RCTX = 64,
    parameter PHYS_VREGS  = 128,
    // Optional R39/v5 core mechanisms; legacy system defaults stay unchanged.
    parameter USE_BANKED_VRF = 0,
    parameter VRF_BYPASS = 0,
    parameter FE_NOBUBBLE = 0,
    parameter VRF_PREFETCH = 0,
    parameter RCTX_SWITCH = 0,
    parameter MTX_ABUF = 0,
    parameter LSU_INFLIGHT = 1,
    parameter FPU_OVERLAP = 0,
    parameter RSLICE = 0,
    parameter RPAIR = 0,
    parameter HAS_SYS_ATOMICS = 0,
    parameter HAS_TENSOR_XPOSE = 0,
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
    parameter NOC_AW      = 2,
    // DCO: tensors the TMU tracks, dead-tile FIFO depth, and where its
    // registers live (rtl/nuca/tnca_tmu.v)
    parameter DCO_T       = 8,
    parameter DCO_T_W     = 3,
    parameter DCO_DEAD    = 16,
    parameter [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     launch_valid,
    input  wire [63:0]              launch_pc,
    input  wire [NUM_RCTX-1:0]      launch_role_mask,
    input  wire [63:0]              launch_arg_base,
    input  wire [NUM_RCTX*PHYS_W-1:0] launch_vbase,
    input wire grid_valid, grid_abort,
    input wire [3:0] grid_abort_status,
    input wire [3:0] grid_slice,
    output wire grid_ready,
    input wire [63:0] grid_id, grid_arg_stride,
    input wire [31:0] grid_x, grid_y, grid_z,
    input wire [NX*NY-1:0] grid_cores,
    output wire grid_done_valid,
    input wire grid_done_ready,
    output wire [3:0] grid_done_status,
    output wire [63:0] grid_done_id, grid_done_cycles,
    output wire [((NX*NY>1)?$clog2(NX*NY):1)-1:0] grid_fault_core,
    output wire [NX*NY*32-1:0] grid_issued, grid_completed,
    output wire [31:0] grid_peak_active,
    output wire [NX*NY*64-1:0] grid_active_cycles,
    input wire host_valid, host_we,
    input wire [63:0] host_addr, host_wdata,
    input wire [7:0] host_be,
    output wire host_ready, host_rsp_valid, host_busy,
    output wire [63:0] host_rsp_data,
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
    output wire [MEM_SRC_W-1:0]     mem_req_src,
    input  wire                     mem_rsp_valid,
    input  wire [MSHR_W-1:0]        mem_rsp_tag,
    input  wire [MEM_SRC_W-1:0]     mem_rsp_src,
    input  wire [LINE_BITS-1:0]     mem_rsp_rdata,

    // ---- counters -----------------------------------------------------------
    output wire [NX*NY*64-1:0]      perf_cycles,
    output wire [NX*NY*64-1:0]      perf_issued,
    output wire [NX*NY*32-1:0]      perf_hit,
    output wire [NX*NY*32-1:0]      perf_miss,
    output wire [NX*NY*32-1:0]      perf_merge,
    output wire [NX*NY*32-1:0]      perf_wb,
    output wire [NX*NY*32-1:0]      perf_mshr_full,
    output wire [NX*NY*32-1:0]      perf_hit_past_block,
    output wire [NX*NY*32-1:0]      perf_bypass,
    output wire [NX*NY*32-1:0]      perf_evict,
    output wire [NX*NY*32-1:0]      perf_dead_evict
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

    wire [N-1:0] tile_launch, dispatch_busy;
    wire [N-1:0] host_readies, host_responses, host_busy_tiles;
    wire [N*64-1:0] host_response_data;
    wire dispatch_ready;
    wire host_allow=GLOBAL_GRID && HOST_ACCESS && dispatch_ready && !grid_valid && all_idle;
    assign host_ready=host_readies[0];assign host_rsp_valid=host_responses[0];
    assign host_busy=host_busy_tiles[0];assign host_rsp_data=host_response_data[63:0];
    wire [63:0] dispatch_pc;
    wire [N*64-1:0] dispatch_args;
    wire [N*32-1:0] dispatch_x, dispatch_y, dispatch_z;
    wire [31:0] dispatch_nx, dispatch_ny, dispatch_nz;
    wire [NUM_RCTX-1:0] tile_roles;
    wire [3:0] tile_slice;
    wire [NUM_RCTX*PHYS_W-1:0] tile_vbase;
    generate if (GLOBAL_GRID) begin: g_dispatch
        reg [NUM_RCTX-1:0] roles_q;
        reg [3:0] slice_q;
        reg [NUM_RCTX*PHYS_W-1:0] vbase_q;
        wire [N-1:0] command_valid;
        wire [N*4-1:0] status;
        for(genvar d=0;d<N;d=d+1) begin: g_status
            assign status[d*4+:4]=fault_valid[d] ? 4'd1 : 4'd0;
        end
        always @(posedge clk) begin
            if(!rst_n) begin roles_q<=0;vbase_q<=0;slice_q<=0;end
            else if(grid_valid && grid_ready) begin roles_q<=launch_role_mask;vbase_q<=launch_vbase;slice_q<=grid_slice;end
        end
        assign grid_ready=dispatch_ready && !host_busy;
        assign tile_slice=slice_q;
        assign tile_roles=roles_q;assign tile_vbase=vbase_q;
        assign tile_launch=command_valid & tile_idle;
        tnca_grid_dispatch #(.CORE_N(N)) u_dispatch (
            .clk(clk),.rst_n(rst_n),.abort_valid(grid_abort),.abort_status(grid_abort_status),.launch_valid(grid_valid && !host_busy),.launch_ready(dispatch_ready),
            .launch_id(grid_id),.launch_pc(launch_pc),.launch_arg_base(launch_arg_base),
            .launch_arg_stride(grid_arg_stride),.launch_gx(grid_x),.launch_gy(grid_y),.launch_gz(grid_z),
            .launch_cores((|launch_role_mask) ? grid_cores : {N{1'b0}}),
            .cmd_valid(command_valid),.cmd_ready(tile_idle),.cmd_id(),.cmd_pc(dispatch_pc),
            .cmd_arg_base(dispatch_args),.cmd_x(dispatch_x),.cmd_y(dispatch_y),.cmd_z(dispatch_z),
            .cmd_nx(dispatch_nx),.cmd_ny(dispatch_ny),.cmd_nz(dispatch_nz),.cmd_cancel(),
            .core_done(dispatch_busy & tile_idle),.core_abort(fault_valid),.core_status(status),
            .done_valid(grid_done_valid),.done_ready(grid_done_ready),.done_status(grid_done_status),
            .done_id(grid_done_id),.done_cycles(grid_done_cycles),.fault_core(grid_fault_core),
            .core_busy(dispatch_busy),.issued(grid_issued),.completed(grid_completed),
            .peak_active(grid_peak_active),.active_cycles(grid_active_cycles));
    end else begin: g_legacy_launch
        assign tile_launch={N{launch_valid}};
        assign tile_slice=0;
        assign tile_roles=launch_role_mask;assign tile_vbase=launch_vbase;
        assign dispatch_pc=launch_pc;assign dispatch_args={N{launch_arg_base}};
        assign dispatch_x=0;assign dispatch_y=0;assign dispatch_z=0;
        assign dispatch_nx=0;assign dispatch_ny=0;assign dispatch_nz=0;
        assign dispatch_busy=0;assign dispatch_ready=0;assign grid_ready=0;assign grid_done_valid=0;
        assign grid_done_status=0;assign grid_done_id=0;assign grid_done_cycles=0;
        assign grid_fault_core=0;assign grid_issued=0;assign grid_completed=0;
        assign grid_peak_active=0;assign grid_active_cycles=0;
    end endgenerate

    //==========================================================================
    // DCO: one TMU for the machine (doc/DCO.pdf; rtl/nuca/tnca_tmu.v)
    //==========================================================================
    localparam DCO_ID_W = 16;
    localparam SRC_W    = 4;          // the slices' requester id: a core
    wire [N-1:0]            tm_wv, tm_wr, tm_tv, tm_tr, tm_drop;
    wire [N*12-1:0]         tm_woff, tm_roff;
    wire [N*64-1:0]         tm_wdata, tm_rdata, tm_taddr;
    wire [N*SRC_W-1:0]      tm_wsrc;
    wire [7:0]              d_policy;
    wire [3:0]              d_bbits, d_ginit;
    wire [5:0]              d_lsb;
    wire [15:0]             d_win, d_ub, d_lb;
    wire [DCO_T-1:0]        d_tv, d_tbyp;
    wire [DCO_T*64-1:0]     d_tbase, d_tsize;
    wire [DCO_T*6-1:0]      d_tlsb;
    wire [DCO_DEAD-1:0]     d_dv;
    wire [DCO_DEAD*DCO_ID_W-1:0] d_did;
    wire [75:0] dco_cfg = {d_lb, d_ub, d_win, d_ginit, 6'd0, d_lsb, d_bbits, d_policy};

    tnca_tmu #(
        .N_SLICE(N), .SRC_W(SRC_W), .N_SRC(N),
        .TENSORS(DCO_T), .T_W(DCO_T_W), .DEAD(DCO_DEAD), .DEAD_W(4),
        .ID_W(DCO_ID_W)
    ) u_tmu (
        .clk(clk), .rst_n(rst_n),
        .mw_valid(tm_wv), .mw_ready(tm_wr), .mw_off(tm_woff), .mw_data(tm_wdata),
        .mw_src(tm_wsrc), .mr_off(tm_roff), .mr_data(tm_rdata),
        .tl_valid(tm_tv), .tl_ready(tm_tr), .tl_addr(tm_taddr), .tl_drop(tm_drop),
        .policy(d_policy), .b_bits(d_bbits), .d_lsb(d_lsb),
        .gear_init(d_ginit), .gear_window(d_win), .gear_ub(d_ub), .gear_lb(d_lb),
        .t_valid(d_tv), .t_base(d_tbase), .t_size(d_tsize), .t_bypass(d_tbyp),
        .t_lsb(d_tlsb),
        .dead_valid(d_dv), .dead_id(d_did)
    );

    // gqa_bypass's "slower core of a pair" (sec IV-E): the one of tiles
    // {2k, 2k+1} that has committed fewer instructions so far.
    reg [15:0] dco_slow;
    integer sp;
    always @(*) begin
        dco_slow = 16'd0;
        for (sp = 0; sp < N; sp = sp + 1)
            if ((sp ^ 1) < N)
                dco_slow[sp] = perf_issued[sp*64 +: 64] < perf_issued[(sp ^ 1)*64 +: 64];
    end

    genvar t;
    generate
        for (t = 0; t < N; t = t + 1) begin : g_tile
            tnca_tile #(
                .TILE_ID(t), .NUM_TILES(N), .GLOBAL_GRID(GLOBAL_GRID), .HOST_ACCESS(HOST_ACCESS && GLOBAL_GRID && t==0), .NX(NX), .NY(NY), .XW(XW), .YW(YW),
                .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN),
                .LW(LW), .VREGS_PER_RCTX(VREGS_PER_RCTX),
                .PHYS_VREGS(PHYS_VREGS), .PHYS_W(PHYS_W),
                .USE_BANKED_VRF(USE_BANKED_VRF),
                .VRF_BYPASS(VRF_BYPASS),
                .FE_NOBUBBLE(FE_NOBUBBLE),
                .VRF_PREFETCH(VRF_PREFETCH),
                .RCTX_SWITCH(RCTX_SWITCH),
                .MTX_ABUF(MTX_ABUF),
                .LSU_INFLIGHT(LSU_INFLIGHT),
                .FPU_OVERLAP(FPU_OVERLAP),
                .RSLICE(RSLICE),
                .RPAIR(RPAIR),
                .HAS_SYS_ATOMICS(HAS_SYS_ATOMICS),
                .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),
                .HAS_SFU(HAS_SFU), .HAS_ATOMICS(HAS_ATOMICS),
                .LSU_BEATS(LSU_BEATS), .LSU_TAG_W(LSU_TAG_W),
                .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
                .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
                .MSHR_N(MSHR_N), .MSHR_W(MSHR_W),
                .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
                .RQ_N(RQ_N), .RQ_W(RQ_W),
                .DCO_T(DCO_T), .DCO_DEAD(DCO_DEAD), .DCO_ID_W(DCO_ID_W),
                .TMU_BASE(TMU_BASE)
            ) u_tile (
                .clk(clk), .rst_n(rst_n),
                .launch_valid(tile_launch[t]), .launch_pc(dispatch_pc),
                .launch_role_mask(tile_roles),
                .launch_arg_base(dispatch_args[t*64+:64]),
                .launch_vbase(tile_vbase),
                .grid_slice(tile_slice), .grid_pid_x(dispatch_x[t*32+:32]), .grid_pid_y(dispatch_y[t*32+:32]),
                .grid_pid_z(dispatch_z[t*32+:32]), .grid_npid_x(dispatch_nx),
                .grid_npid_y(dispatch_ny), .grid_npid_z(dispatch_nz),
                .host_allow(host_allow), .host_valid(host_valid && t==0), .host_we(host_we),
                .host_addr(host_addr), .host_wdata(host_wdata), .host_be(host_be),
                .host_ready(host_readies[t]), .host_rsp_valid(host_responses[t]),
                .host_rsp_data(host_response_data[t*64+:64]), .host_busy(host_busy_tiles[t]),
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
                .perf_hit_past_block(perf_hit_past_block[t*32 +: 32]),
                .dco_cfg(dco_cfg), .dco_t_valid(d_tv), .dco_t_base(d_tbase),
                .dco_t_size(d_tsize), .dco_t_bypass(d_tbyp), .dco_t_lsb(d_tlsb),
                .dco_dead_valid(d_dv), .dco_dead_id(d_did), .dco_slow(dco_slow),
                .tmu_wr_valid(tm_wv[t]), .tmu_wr_ready(tm_wr[t]),
                .tmu_wr_off(tm_woff[t*12 +: 12]), .tmu_wr_data(tm_wdata[t*64 +: 64]),
                .tmu_wr_src(tm_wsrc[t*SRC_W +: SRC_W]),
                .tmu_rd_off(tm_roff[t*12 +: 12]), .tmu_rd_data(tm_rdata[t*64 +: 64]),
                .tmu_tl_valid(tm_tv[t]), .tmu_tl_ready(tm_tr[t]),
                .tmu_tl_addr(tm_taddr[t*64 +: 64]), .tmu_tl_drop(tm_drop[t]),
                .perf_bypass(perf_bypass[t*32 +: 32]),
                .perf_evict(perf_evict[t*32 +: 32]),
                .perf_dead_evict(perf_dead_evict[t*32 +: 32]),
                .perf_gear()
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
        .N(N), .N_W(MEM_SRC_W), .ADDR_W(64), .LINE_BITS(LINE_BITS), .TAG_W(MSHR_W)
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
