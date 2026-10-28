//==============================================================================
// tnca_tile -- one core, one NUCA slice, and their network interfaces
//
// The local mesh port carries four independent channels and each has exactly
// one user, so the core and the slice never contend for it:
//
//   in  VN_REQ   <- the core's requests, heading for some home slice
//   in  VN_RSP   <- this slice's responses, heading back to some core
//   out VN_REQ   -> requests that homed here, into this slice
//   out VN_RSP   -> responses to this core's earlier requests
//
// A core's request usually leaves the tile: the home slice is a function of the
// address (arch overview sec 10.1), not of who asked, so roughly 3 of every 4
// accesses in a 2x2 system cross the mesh. That is the design, not a defect --
// the whole point of a single shared NUCA level is that a line has one home and
// everyone reaches it, rather than each core keeping its own copy.
//
// Instruction fetch is local. Sec 1.1 allows a small instruction buffer
// explicitly and says the restriction is on the DATA storage hierarchy; if
// instruction fetch went through the NUCA its traffic would have to be counted
// in the mesh and slice bandwidth budgets.
//==============================================================================
`include "tnca_noc_pkg.vh"

module tnca_tile #(
    parameter TILE_ID     = 0,
    parameter NUM_TILES   = 4,
    parameter GLOBAL_GRID = 0,
    parameter HOST_ACCESS = 0,
    parameter NX = 2, NY = 2,
    parameter XW          = 2,
    parameter YW          = 2,
    // core
    parameter NUM_RCTX    = 2,
    parameter RCTX_W      = 1,
    parameter VLEN        = 16,
    parameter ELEN        = 32,
    parameter LW          = 4,
    parameter VREGS_PER_RCTX = 64,
    // R41 G2 resource sizing (defaults: the delivered machine): accumulator slots, mbarriers, the barrier epoch width
    parameter ACC_N = 16, ACC_W = 4, BAR_N = 32, BAR_W = 5, BAR_EPOCH_W = 48,
    // and the home slice's physical address width (64: unchanged; below 64 an
    // address at or above 2^(PHYS_ADDR_W-1) is folded so it stays out of range)
    parameter PHYS_ADDR_W = 64,
    parameter PHYS_VREGS  = 128,
    parameter PHYS_W      = 7,
    // Capability parameters, so the PROFILE decides and not this file. The
    // system profile keeps the SFU off -- that is the correct configuration,
    // not a defect -- but hard-coding it here is what let the target
    // description claim otherwise (audit F03). spec/r1_profiles.json is the
    // one place the value is chosen.
    // Optional R39/v5 core mechanisms; legacy system defaults stay unchanged.
    parameter USE_BANKED_VRF = 0,
    parameter VRF_BYPASS = 0,
    parameter FE_NOBUBBLE = 0,
    parameter VRF_PREFETCH = 0,
    parameter RCTX_SWITCH = 0,
    parameter MTX_ABUF = 0,
    parameter HYBRID_LSU = 0,
    parameter HOME_BACKGROUND_READS = 0,
    parameter SFU_LANES = 1,
    parameter NOC_PAYLOAD_W = HYBRID_LSU ? 720 : 216,
    parameter LSU_INFLIGHT = 1,
    parameter FPU_OVERLAP = 0,
    parameter RSLICE = 0,
    parameter RPAIR = 0,
    parameter HAS_SYS_ATOMICS = 0,
    parameter CACHE_MAINTENANCE = 0,
    parameter HAS_TENSOR_XPOSE = 0,
    parameter HAS_SFU     = 0,
    parameter HAS_ATOMICS = 1,
    parameter LSU_BEATS   = 8,
    parameter LSU_TAG_W   = 5,
    // slice
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
    parameter SLICE_SRC_W = 4,
    parameter SLICE_TAG_W = 5,
    // DCO (rtl/nuca/tnca_tmu.v)
    parameter DCO_T       = 8,
    parameter DCO_REGISTERED_RANGES = 0,
    parameter NUCA_REGISTER_VICTIM_SELECT = 0,
    parameter NUCA_CACHE_DATA_PIPELINE = 0,
    parameter NUCA_SHARED_DATA_READ = 0,   // R41 G2: one LUTRAM read port per slice line array
    parameter NUCA_REGISTER_VICTIM_RANK = 0,
    parameter DCO_DEAD    = 16,
    parameter DCO_ID_W    = 16,
    parameter [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- launch -------------------------------------------------------------
    input  wire                     launch_valid,
    input  wire [63:0]              launch_pc,
    input  wire [NUM_RCTX-1:0]      launch_role_mask,
    input  wire [63:0]              launch_arg_base,
    input  wire [NUM_RCTX*PHYS_W-1:0] launch_vbase,
    input wire [3:0] grid_slice,
    input wire [31:0] grid_pid_x, grid_pid_y, grid_pid_z,
    input wire [31:0] grid_npid_x, grid_npid_y, grid_npid_z,
    input wire host_allow, host_valid, host_we,
    input wire cache_maint_valid,
    input wire [63:0] cache_maint_base,cache_maint_size,
    output wire cache_maint_ready,cache_maint_done,
    input wire [63:0] host_addr, host_wdata,
    input wire [7:0] host_be,
    output wire host_ready, host_rsp_valid, host_busy,
    output wire [63:0] host_rsp_data,
    output wire                     core_idle,

    output wire                     fault_valid,
    output wire [7:0]               fault_code,
    output wire [63:0]              fault_pc,
    output wire [23:0]              fault_detail,

    // ---- instruction port (local; see the header) ---------------------------
    output wire [63:0]              imem_addr,
    input  wire [63:0]              imem_data,

    // ---- mesh local port ----------------------------------------------------
    output wire [`TNCA_NOC_NV-1:0]                          l_in_valid,
    input  wire [`TNCA_NOC_NV-1:0]                          l_in_ready,
    output wire [`TNCA_NOC_NV*(XW+YW+NOC_PAYLOAD_W)-1:0] l_in_flit,
    input  wire [`TNCA_NOC_NV-1:0]                          l_out_valid,
    output wire [`TNCA_NOC_NV-1:0]                          l_out_ready,
    input  wire [`TNCA_NOC_NV*(XW+YW+NOC_PAYLOAD_W)-1:0] l_out_flit,

    // ---- this slice's memory port -------------------------------------------
    output wire                     mem_req_valid,
    input  wire                     mem_req_ready,
    output wire [63:0]              mem_req_addr,
    output wire                     mem_req_we,
    output wire [LINE_BITS-1:0]     mem_req_wdata,
    output wire [MSHR_W-1:0]        mem_req_tag,
    input  wire                     mem_rsp_valid,
    input  wire [MSHR_W-1:0]        mem_rsp_tag,
    input  wire [LINE_BITS-1:0]     mem_rsp_rdata,

    // ---- DCO: this slice's side of the TMU (straight through) -----------------
    input  wire [75:0]              dco_cfg,
    input  wire                       dco_owner_epoch,
    input  wire [DCO_T-1:0]         dco_t_valid,
    input  wire [DCO_T*64-1:0]      dco_t_base,
    input  wire [DCO_T*64-1:0]      dco_t_size,
    input  wire [DCO_T*65-1:0]      dco_t_end,
    input  wire [DCO_T-1:0]         dco_t_bypass,
    input  wire [DCO_T*6-1:0]       dco_t_lsb,
    input  wire [DCO_DEAD-1:0]      dco_dead_valid,
    input  wire [DCO_DEAD*DCO_ID_W-1:0] dco_dead_id,
    input  wire [15:0]              dco_slow,
    output wire                     tmu_wr_valid,
    input  wire                     tmu_wr_ready,
    output wire [11:0]              tmu_wr_off,
    output wire [63:0]              tmu_wr_data,
    output wire [SLICE_SRC_W-1:0]   tmu_wr_src,
    output wire [11:0]              tmu_rd_off,
    input  wire [63:0]              tmu_rd_data,
    output wire                     tmu_tl_valid,
    input  wire                     tmu_tl_ready,
    output wire [63:0]              tmu_tl_addr,
    output wire                     tmu_tl_drop,
    output wire [31:0]              perf_bypass,
    output wire [31:0]              perf_evict,
    output wire [31:0]              perf_dead_evict,
    output wire [3:0]               perf_gear,

    // ---- counters -----------------------------------------------------------
    output wire [63:0]              perf_cycles,
    output wire [63:0]              perf_issued,
    output wire [31:0]              perf_hit,
    output wire [31:0]              perf_miss,
    output wire [31:0]              perf_merge,
    output wire [31:0]              perf_wb,
    output wire [31:0]              perf_mshr_full,
    output wire [31:0]              perf_hit_past_block
);

    localparam CORE_DW=HYBRID_LSU ? 512 : 64;
    localparam CORE_BYTES=CORE_DW/8;
    localparam FW = XW + YW + NOC_PAYLOAD_W;
    localparam [31:0] TILE_ID_32   = TILE_ID;
    localparam [31:0] NUM_TILES_32 = NUM_TILES;
    localparam VR = `TNCA_VN_REQ;
    localparam VS = `TNCA_VN_RSP;

    //==========================================================================
    // Core
    //==========================================================================
    wire                 c_req_valid, c_req_ready, c_req_we;
    wire [63:0] c_req_addr;
    wire [CORE_DW-1:0] c_req_wdata;
    wire [CORE_BYTES-1:0] c_req_be;
    wire [LSU_TAG_W-1:0] c_req_tag;
    wire [1:0]           c_req_hint;
    wire                 c_req_amo;
    wire [3:0]           c_req_aop;
    wire [CORE_DW-1:0] c_req_cmp;
    wire                 c_rsp_valid;
    wire [LSU_TAG_W-1:0] c_rsp_tag;
    wire [CORE_DW-1:0] c_rsp_rdata;

    tnca_core #(
        .HYBRID_LSU(HYBRID_LSU),.SFU_LANES(SFU_LANES),
        .MEM_DW(CORE_DW),.MEM_BYTES(CORE_BYTES),.MEM_SHIFT(HYBRID_LSU ? 6 : 3),
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN), .LW(LW),
        .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .PHYS_VREGS(PHYS_VREGS), .PHYS_W(PHYS_W),
        .CORE_ID(TILE_ID), .NUM_CORES(NUM_TILES),
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
        .LSU_BEATS(HYBRID_LSU ? 4 : LSU_BEATS), .LSU_TAG_W(LSU_TAG_W)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .launch_valid(launch_valid), .launch_pc(launch_pc),
        .launch_role_mask(launch_role_mask), .launch_slice(GLOBAL_GRID ? grid_slice : 4'd0),
        .launch_pid_x(GLOBAL_GRID ? grid_pid_x : TILE_ID_32),
        .launch_pid_y(GLOBAL_GRID ? grid_pid_y : 32'd0), .launch_pid_z(GLOBAL_GRID ? grid_pid_z : 32'd0),
        .launch_npid_x(GLOBAL_GRID ? grid_npid_x : NUM_TILES_32),
        .launch_npid_y(GLOBAL_GRID ? grid_npid_y : 32'd1), .launch_npid_z(GLOBAL_GRID ? grid_npid_z : 32'd1),
        .launch_arg_base(launch_arg_base), .launch_vbase(launch_vbase),
        .core_idle(core_idle),
        .fault_valid(fault_valid), .fault_code(fault_code),
        .fault_pc(fault_pc), .fault_rctx(), .fault_detail(fault_detail),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(c_req_valid), .mem_req_ready(c_req_ready),
        .mem_req_addr(c_req_addr), .mem_req_we(c_req_we),
        .mem_req_be(c_req_be), .mem_req_wdata(c_req_wdata),
        .mem_req_tag(c_req_tag), .mem_req_hint(c_req_hint),
        .mem_req_amo(c_req_amo), .mem_req_aop(c_req_aop),
        .mem_req_cmp(c_req_cmp),
        .mem_rsp_valid(c_rsp_valid), .mem_rsp_tag(c_rsp_tag),
        .mem_rsp_rdata(c_rsp_rdata),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(), .perf_stall_lsu(), .perf_stall_wb(), .perf_squash()
    );

    //==========================================================================
    // Network interfaces
    //==========================================================================
    wire            ni_in_v_req, ni_in_r_req;
    wire [FW-1:0]   ni_in_f_req;
    wire            ni_out_v_rsp, ni_out_r_rsp;
    wire [FW-1:0]   ni_out_f_rsp;

    wire n_req_valid,n_req_ready,n_req_we,n_req_amo,n_rsp_valid;
    wire [63:0] n_req_addr,n_req_cmp;
    wire [CORE_DW-1:0] n_req_wdata,n_rsp_rdata;
    wire [CORE_BYTES-1:0] n_req_be;wire [3:0] n_req_aop;
    wire [LSU_TAG_W-1:0] n_req_tag,n_rsp_tag;
    generate if(HOST_ACCESS) begin: g_host_mux
        // Second copy of the registered quiescence permission, inside this
        // tile (and so inside its SLR pblock). tnca_top proves the global
        // permission is constant while a host access is pending; the grant
        // flop is the only consumer.
        reg host_allow_q;
        always @(posedge clk) host_allow_q <= rst_n && host_allow;
        tnca_idle_host_mux #(.TAG_W(LSU_TAG_W),.DATA_W(CORE_DW)) u_host (
            .clk(clk),.rst_n(rst_n),.host_allow(host_allow_q),
            .c_valid(c_req_valid),.c_ready(c_req_ready),.c_addr(c_req_addr),.c_wdata(c_req_wdata),
            .c_cmp(c_req_cmp[63:0]),.c_we(c_req_we),.c_amo(c_req_amo),.c_be(c_req_be),.c_aop(c_req_aop),.c_tag(c_req_tag),
            .c_rsp_valid(c_rsp_valid),.c_rsp_tag(c_rsp_tag),.c_rsp_data(c_rsp_rdata),
            .h_valid(host_valid),.h_ready(host_ready),.h_we(host_we),.h_addr(host_addr),
            .h_wdata(host_wdata),.h_be(host_be),.h_rsp_valid(host_rsp_valid),.h_rsp_data(host_rsp_data),.host_busy(host_busy),
            .n_valid(n_req_valid),.n_ready(n_req_ready),.n_addr(n_req_addr),.n_wdata(n_req_wdata),
            .n_cmp(n_req_cmp),.n_we(n_req_we),.n_amo(n_req_amo),.n_be(n_req_be),.n_aop(n_req_aop),.n_tag(n_req_tag),
            .n_rsp_valid(n_rsp_valid),.n_rsp_tag(n_rsp_tag),.n_rsp_data(n_rsp_rdata));
    end else begin: g_core_only
        assign n_req_valid=c_req_valid;assign c_req_ready=n_req_ready;
        assign n_req_addr=c_req_addr;assign n_req_wdata=c_req_wdata;assign n_req_cmp=c_req_cmp[63:0];
        assign n_req_we=c_req_we;assign n_req_amo=c_req_amo;assign n_req_be=c_req_be;
        assign n_req_aop=c_req_aop;assign n_req_tag=c_req_tag;
        assign c_rsp_valid=n_rsp_valid;assign c_rsp_tag=n_rsp_tag;assign c_rsp_rdata=n_rsp_rdata;
        assign host_ready=0;assign host_rsp_valid=0;assign host_rsp_data=0;assign host_busy=0;
    end endgenerate

    tnca_core_ni #(
        .DATA_W(CORE_DW),.PAYLOAD_W(NOC_PAYLOAD_W),
        .NX(NX), .NY(NY), .XW(XW), .YW(YW), .TAG_W(LSU_TAG_W), .SRC_W(SLICE_SRC_W),
        .MY_ID(TILE_ID), .LINE_SH(6)
    ) u_cni (
        .clk(clk), .rst_n(rst_n),
        .req_valid(n_req_valid), .req_ready(n_req_ready),
        .req_addr(n_req_addr), .req_we(n_req_we), .req_be(n_req_be),
        .req_wdata(n_req_wdata), .req_tag(n_req_tag),
        .req_amo(n_req_amo), .req_aop(n_req_aop), .req_cmp(n_req_cmp),
        .rsp_valid(n_rsp_valid), .rsp_tag(n_rsp_tag), .rsp_rdata(n_rsp_rdata),
        .net_in_valid_req(ni_in_v_req), .net_in_ready_req(ni_in_r_req),
        .net_in_flit_req(ni_in_f_req),
        .net_in_valid_rsp(), .net_in_ready_rsp(1'b1), .net_in_flit_rsp(),
        .net_out_valid_req(1'b0), .net_out_ready_req(), .net_out_flit_req({FW{1'b0}}),
        .net_out_valid_rsp(ni_out_v_rsp), .net_out_ready_rsp(ni_out_r_rsp),
        .net_out_flit_rsp(ni_out_f_rsp)
    );

    wire                    s_req_valid, s_req_ready, s_req_we;
    wire [63:0] s_req_addr;
    wire [CORE_DW-1:0] s_req_wdata;
    wire [CORE_BYTES-1:0] s_req_be;
    wire                    s_req_amo;
    wire [3:0]              s_req_aop;
    wire [63:0]             s_req_cmp;
    wire [SLICE_SRC_W-1:0]  s_req_src;
    wire [SLICE_TAG_W-1:0]  s_req_tag;
    wire                    s_rsp_valid, s_rsp_ready;
    wire [SLICE_SRC_W-1:0]  s_rsp_src;
    wire [SLICE_TAG_W-1:0]  s_rsp_tag;
    wire [CORE_DW-1:0] s_rsp_rdata;

    wire            ni_in_v_rsp, ni_in_r_rsp;
    wire [FW-1:0]   ni_in_f_rsp;
    wire            ni_out_v_req, ni_out_r_req;
    wire [FW-1:0]   ni_out_f_req;

    tnca_slice_ni #(
        .DATA_W(CORE_DW),.PAYLOAD_W(NOC_PAYLOAD_W),
        .NX(NX), .NY(NY), .XW(XW), .YW(YW), .TAG_W(SLICE_TAG_W), .SRC_W(SLICE_SRC_W)
    ) u_sni (
        .clk(clk), .rst_n(rst_n),
        .req_valid(s_req_valid), .req_ready(s_req_ready),
        .req_addr(s_req_addr), .req_we(s_req_we), .req_be(s_req_be),
        .req_wdata(s_req_wdata), .req_src(s_req_src), .req_tag(s_req_tag),
        .req_amo(s_req_amo), .req_aop(s_req_aop), .req_cmp(s_req_cmp),
        .rsp_valid(s_rsp_valid), .rsp_ready(s_rsp_ready),
        .rsp_src(s_rsp_src), .rsp_tag(s_rsp_tag), .rsp_rdata(s_rsp_rdata),
        .net_in_valid_req(), .net_in_ready_req(1'b1), .net_in_flit_req(),
        .net_in_valid_rsp(ni_in_v_rsp), .net_in_ready_rsp(ni_in_r_rsp),
        .net_in_flit_rsp(ni_in_f_rsp),
        .net_out_valid_req(ni_out_v_req), .net_out_ready_req(ni_out_r_req),
        .net_out_flit_req(ni_out_f_req),
        .net_out_valid_rsp(1'b0), .net_out_ready_rsp(), .net_out_flit_rsp({FW{1'b0}})
    );

    //==========================================================================
    // R41 G2: the home slice's physical address width. With PHYS_ADDR_W < 64
    // the slice keeps PHYS_ADDR_W bits: the low PHYS_ADDR_W-1 bits of an
    // address and, as the top bit, whether any higher bit was set. An address
    // below 2^(PHYS_ADDR_W-1) is unchanged; one at or above it reaches memory
    // as an address at or above 2^(PHYS_ADDR_W-1), which no outlet aperture of
    // the image contains, so it still faults instead of aliasing a real line.
    // A maintenance size that does not fit saturates (it covers every line).
    //==========================================================================
    localparam SA_W = PHYS_ADDR_W;
    function [SA_W-1:0] g2_fold;
        input [63:0] a;
        begin
            if (SA_W >= 64) g2_fold = a[SA_W-1:0];
            else g2_fold = {|(a >> (SA_W-1)), a[SA_W-2:0]};
        end
    endfunction
    function [SA_W-1:0] g2_sat;
        input [63:0] a;
        begin
            if (SA_W >= 64) g2_sat = a[SA_W-1:0];
            else g2_sat = (|(a >> SA_W)) ? {SA_W{1'b1}} : a[SA_W-1:0];
        end
    endfunction
    wire [SA_W-1:0] sl_req_addr = g2_fold(s_req_addr);
    wire [SA_W-1:0] sl_maint_base = g2_fold(cache_maint_base);
    wire [SA_W-1:0] sl_maint_size = g2_sat(cache_maint_size);
    wire [SA_W-1:0] sl_mem_req_addr, sl_tmu_tl_addr;
    generate if (SA_W < 64) begin : g_narrow_home
        assign mem_req_addr = {{(64-SA_W){1'b0}}, sl_mem_req_addr};
        assign tmu_tl_addr  = {{(64-SA_W){1'b0}}, sl_tmu_tl_addr};
    end else begin : g_full_home
        assign mem_req_addr = sl_mem_req_addr;
        assign tmu_tl_addr  = sl_tmu_tl_addr;
    end endgenerate

    //==========================================================================
    // NUCA slice
    //==========================================================================
    // Keep private cache lookup/replacement logic with its state, so the
    // tile's SLR constraint also owns the complete combinational cone.
    (* KEEP_HIERARCHY = "yes" *) tnca_nuca_slice #(
        .REGISTER_VICTIM_OWNER(1), .REGISTER_VICTIM_SELECT(NUCA_REGISTER_VICTIM_SELECT), .CACHE_DATA_PIPELINE(NUCA_CACHE_DATA_PIPELINE), .SHARED_DATA_READ(NUCA_SHARED_DATA_READ), .REGISTER_VICTIM_RANK(NUCA_REGISTER_VICTIM_RANK), .MAINTENANCE(CACHE_MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES),
        .FULL_LINE_TAG(!(NX==2 && NY==2)),
        .ADDR_W(SA_W), .LINE_BYTES(LINE_BYTES), .LINE_BITS(LINE_BITS),
        .WORD_BITS(CORE_DW), .WORD_BYTES(CORE_BYTES), .HIT_CHAIN(HYBRID_LSU), .FAST_FILL_READ(HYBRID_LSU), .BACKGROUND_READ_REQUESTS(HOME_BACKGROUND_READS),
        .SETS(SETS), .SET_W(SET_W), .WAYS(WAYS), .WAY_W(WAY_W),
        .MSHR_N(MSHR_N), .MSHR_W(MSHR_W),
        .TARGET_N(TARGET_N), .TARGET_W(TARGET_W),
        .RQ_N(RQ_N), .RQ_W(RQ_W),
        .SRC_W(SLICE_SRC_W), .TAG_W(SLICE_TAG_W),
        .SLICE_ID(TILE_ID),
        .DCO_T(DCO_T), .DCO_DEAD(DCO_DEAD), .DCO_ID_W(DCO_ID_W),
        .TMU_BASE(TMU_BASE)
    ) u_slice (
        .clk(clk), .rst_n(rst_n),
        .maint_valid(cache_maint_valid),.maint_base(sl_maint_base),.maint_size(sl_maint_size),
        .maint_ready(cache_maint_ready),.maint_done(cache_maint_done),.maint_busy(),
        .req_ready(s_req_ready), .req_valid(s_req_valid),
        .req_addr(sl_req_addr), .req_we(s_req_we), .req_be(s_req_be),
        .req_wdata(s_req_wdata), .req_src(s_req_src), .req_tag(s_req_tag),
        .req_amo(s_req_amo), .req_aop(s_req_aop), .req_cmp({{(CORE_DW-64){1'b0}},s_req_cmp}),
        .rsp_valid(s_rsp_valid), .rsp_ready(s_rsp_ready),
        .rsp_src(s_rsp_src), .rsp_tag(s_rsp_tag), .rsp_rdata(s_rsp_rdata),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(sl_mem_req_addr), .mem_req_we(mem_req_we),
        .mem_req_wdata(mem_req_wdata), .mem_req_tag(mem_req_tag),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_rdata(mem_rsp_rdata),
        .perf_hit(perf_hit), .perf_miss(perf_miss), .perf_merge(perf_merge),
        .perf_mshr_full(perf_mshr_full), .perf_target_full(),
        .perf_wb(perf_wb), .perf_hit_past_block(perf_hit_past_block),
        .dco_owner_epoch(dco_owner_epoch), .dco_cfg(dco_cfg), .dco_t_valid(dco_t_valid), .dco_t_base(dco_t_base),
        .dco_t_end(dco_t_end), .dco_t_size(dco_t_size), .dco_t_bypass(dco_t_bypass), .dco_t_lsb(dco_t_lsb),
        .dco_dead_valid(dco_dead_valid), .dco_dead_id(dco_dead_id),
        .dco_slow(dco_slow),
        .tmu_wr_valid(tmu_wr_valid), .tmu_wr_ready(tmu_wr_ready),
        .tmu_wr_off(tmu_wr_off), .tmu_wr_data(tmu_wr_data), .tmu_wr_src(tmu_wr_src),
        .tmu_rd_off(tmu_rd_off), .tmu_rd_data(tmu_rd_data),
        .tmu_tl_valid(tmu_tl_valid), .tmu_tl_ready(tmu_tl_ready),
        .tmu_tl_addr(sl_tmu_tl_addr), .tmu_tl_drop(tmu_tl_drop),
        .perf_bypass(perf_bypass), .perf_evict(perf_evict),
        .perf_dead_evict(perf_dead_evict), .perf_gear(perf_gear)
    );

    //==========================================================================
    // Local mesh port
    //==========================================================================
    assign l_in_valid[VR]            = ni_in_v_req;
    assign l_in_flit[VR*FW +: FW]    = ni_in_f_req;
    assign ni_in_r_req               = l_in_ready[VR];

    assign l_in_valid[VS]            = ni_in_v_rsp;
    assign l_in_flit[VS*FW +: FW]    = ni_in_f_rsp;
    assign ni_in_r_rsp               = l_in_ready[VS];

    assign ni_out_v_req              = l_out_valid[VR];
    assign ni_out_f_req              = l_out_flit[VR*FW +: FW];
    assign l_out_ready[VR]           = ni_out_r_req;

    assign ni_out_v_rsp              = l_out_valid[VS];
    assign ni_out_f_rsp              = l_out_flit[VS*FW +: FW];
    assign l_out_ready[VS]           = ni_out_r_rsp;

    wire _unused = &{1'b0, c_req_hint, 1'b0};

endmodule
