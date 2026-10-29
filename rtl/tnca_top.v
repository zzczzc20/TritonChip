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
    parameter CACHE_MAINTENANCE = 0,
    parameter DCO_REGISTERED_RANGES = 0,
    parameter NUCA_REGISTER_VICTIM_SELECT = 0,
    parameter NUCA_CACHE_DATA_PIPELINE = 0,
    parameter NUCA_SHARED_DATA_READ = 0,   // R41 G2: one LUTRAM read port per slice line array
    parameter NUCA_REGISTER_VICTIM_RANK = 0,
    parameter DCO_TLW_PIPE_STAGES = 0,
    parameter DCO_REGISTER_WRITES = 0,
    parameter DCO_LOCAL_REPLICAS = 0,
    parameter NX          = 2,
    parameter NY          = 2,
    parameter MEM_PORTS   = 1,
    parameter NOC_LINK_STAGES=0,DISPATCH_STAGES=0,MEM_HOME_STAGES=0,
    parameter MEM_SRC_W   = (NX*NY>4 ? $clog2(NX*NY) : 2),
    parameter XW          = 2,
    parameter YW          = 2,
    parameter NUM_RCTX    = 2,
    parameter RCTX_W      = 1,
    parameter VLEN        = 16,
    parameter ELEN        = 32,
    parameter LW          = 4,
    parameter VREGS_PER_RCTX = 64,
    // R41 G2 resource sizing (defaults: the delivered machine): accumulator slots, mbarriers, the barrier epoch width
    parameter ACC_N = 16, ACC_W = 4, BAR_N = 32, BAR_W = 5, BAR_EPOCH_W = 48, FMA_IMPL = 0, MAC_DP = 0, MAC_CUTS = 30,
    // and the home slice's physical address width (64: unchanged; below 64 an
    // address at or above 2^(PHYS_ADDR_W-1) is folded so it stays out of range)
    parameter PHYS_ADDR_W = 64,
    parameter PHYS_VREGS  = 128,
    // Optional R39/v5 core mechanisms; legacy system defaults stay unchanged.
    parameter USE_BANKED_VRF = 0,
    parameter VRF_BYPASS = 0,
    parameter FE_NOBUBBLE = 0,
    parameter VRF_PREFETCH = 0,
    parameter RCTX_SWITCH = 0,
    parameter MTX_ABUF = 0,
    parameter MTX_VRF_REG = 0,
    parameter HYBRID_LSU = 0,
    parameter HOME_BACKGROUND_READS = 0,
    parameter HOME_LINE_WRITE_NOFILL = 0,   // R41 G4: full-line stores allocate without a fill
    parameter HOME_LINE_PIPE = 0,                // R41 G4: pipelined home (tnca_home_pipe) when HYBRID_LSU
    parameter TENSOR_LINE = 0,                   // R41 G4: the transfer engine's line path when HYBRID_LSU
    parameter SFU_LANES = 1,
    parameter NOC_PAYLOAD_W = HYBRID_LSU ? 720 : 216,
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
    input wire [1:0] grid_group_log2,
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
    input wire cache_maint_valid,
    input wire [63:0] cache_maint_base,cache_maint_size,
    input wire cache_maint_memory_drained,
    output wire cache_maint_ready,cache_maint_done,

    output wire [NX*NY-1:0]         fault_valid,
    output wire [NX*NY*8-1:0]       fault_code,
    output wire [NX*NY*64-1:0]      fault_pc,
    output wire [NX*NY*24-1:0]      fault_detail,

    // ---- per-tile instruction ports -----------------------------------------
    output wire [NX*NY*64-1:0]      imem_addr,
    input  wire [NX*NY*64-1:0]      imem_data,

    // Multi-controller interface, used only when MEM_PORTS > 1. Legacy scalar
    // memory ports remain the MEM_PORTS=1 contract.
    output wire [MEM_PORTS-1:0] ports_req_valid,
    input wire [MEM_PORTS-1:0] ports_req_ready,
    output wire [MEM_PORTS*64-1:0] ports_req_addr,
    output wire [MEM_PORTS-1:0] ports_req_we,
    output wire [MEM_PORTS*LINE_BITS-1:0] ports_req_wdata,
    output wire [MEM_PORTS*MSHR_W-1:0] ports_req_tag,
    output wire [MEM_PORTS*MEM_SRC_W-1:0] ports_req_src,
    input wire [MEM_PORTS-1:0] ports_rsp_valid,
    input wire [MEM_PORTS*MSHR_W-1:0] ports_rsp_tag,
    input wire [MEM_PORTS*MEM_SRC_W-1:0] ports_rsp_src,
    input wire [MEM_PORTS*LINE_BITS-1:0] ports_rsp_rdata,
    output wire [MEM_PORTS-1:0] memory_protocol_fault,
    output reg [MEM_PORTS*64-1:0] memory_reads,memory_writes,

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
    localparam FW = XW + YW + NOC_PAYLOAD_W;

    wire [N*NV-1:0]     li_valid, li_ready, lo_valid, lo_ready;
    wire [N*NV*FW-1:0]  li_flit,  lo_flit;

    wire [N-1:0]            m_req_valid, m_req_ready, m_req_we;
    wire [N*64-1:0]         m_req_addr;
    wire [N*LINE_BITS-1:0]  m_req_wdata;
    wire [N*MSHR_W-1:0]     m_req_tag;
    wire [N-1:0]            m_rsp_valid;
    wire [N*MSHR_W-1:0]     m_rsp_tag;
    wire [N*LINE_BITS-1:0]  m_rsp_rdata;

    wire [N-1:0] home_link_idle;
    wire [N-1:0] outlet_req_valid,outlet_req_ready,outlet_req_we,outlet_rsp_valid;
    wire [N*64-1:0] outlet_req_addr;
    wire [N*LINE_BITS-1:0] outlet_req_wdata,outlet_rsp_rdata;
    wire [N*MSHR_W-1:0] outlet_req_tag,outlet_rsp_tag;
    generate if(MEM_HOME_STAGES<0 || (MEM_HOME_STAGES>0 && LINE_BITS!=512))begin:g_bad_home_pipeline
        tnca_invalid_home_pipeline_geometry invalid_configuration();
    end endgenerate
    generate for(genvar h=0;h<N;h=h+1)begin:g_home_link
        localparam [MEM_SRC_W-1:0] HOME_SRC=h;
        if(MEM_HOME_STAGES==0)begin:g_direct
            assign home_link_idle[h]=1'b1;
            assign outlet_req_valid[h]=m_req_valid[h];assign m_req_ready[h]=outlet_req_ready[h];
            assign outlet_req_we[h]=m_req_we[h];assign outlet_req_addr[h*64+:64]=m_req_addr[h*64+:64];
            assign outlet_req_wdata[h*LINE_BITS+:LINE_BITS]=m_req_wdata[h*LINE_BITS+:LINE_BITS];
            assign outlet_req_tag[h*MSHR_W+:MSHR_W]=m_req_tag[h*MSHR_W+:MSHR_W];
            assign m_rsp_valid[h]=outlet_rsp_valid[h];assign m_rsp_tag[h*MSHR_W+:MSHR_W]=outlet_rsp_tag[h*MSHR_W+:MSHR_W];
            assign m_rsp_rdata[h*LINE_BITS+:LINE_BITS]=outlet_rsp_rdata[h*LINE_BITS+:LINE_BITS];
        end else begin:g_pipeline
            tnca_line_link #(.SRC_W(MEM_SRC_W),.TAG_W(MSHR_W),.STAGES(MEM_HOME_STAGES)) link(
                .clk(clk),.rst_n(rst_n),.idle(home_link_idle[h]),
                .s_req_valid(m_req_valid[h]),.s_req_ready(m_req_ready[h]),.s_req_addr(m_req_addr[h*64+:64]),
                .s_req_we(m_req_we[h]),.s_req_wdata(m_req_wdata[h*LINE_BITS+:LINE_BITS]),.s_req_src(HOME_SRC),.s_req_tag(m_req_tag[h*MSHR_W+:MSHR_W]),
                .s_rsp_valid(m_rsp_valid[h]),.s_rsp_rdata(m_rsp_rdata[h*LINE_BITS+:LINE_BITS]),.s_rsp_src(),.s_rsp_tag(m_rsp_tag[h*MSHR_W+:MSHR_W]),
                .m_req_valid(outlet_req_valid[h]),.m_req_ready(outlet_req_ready[h]),.m_req_addr(outlet_req_addr[h*64+:64]),
                .m_req_we(outlet_req_we[h]),.m_req_wdata(outlet_req_wdata[h*LINE_BITS+:LINE_BITS]),.m_req_src(),.m_req_tag(outlet_req_tag[h*MSHR_W+:MSHR_W]),
                .m_rsp_valid(outlet_rsp_valid[h]),.m_rsp_rdata(outlet_rsp_rdata[h*LINE_BITS+:LINE_BITS]),.m_rsp_src(HOME_SRC),.m_rsp_tag(outlet_rsp_tag[h*MSHR_W+:MSHR_W]));
        end
    end endgenerate

    wire [N-1:0] tile_idle;
    wire cache_quiescent = (&tile_idle) && !(|dispatch_busy) && (&home_link_idle);
    wire maintenance_requested=(CACHE_MAINTENANCE!=0) && cache_maint_valid;
    wire maintenance_busy;
    wire [N-1:0] maintenance_valid,maintenance_ready,maintenance_done;
    wire [N-1:0] home_maintenance_valid,home_maintenance_ready,home_maintenance_done;
    wire [N*128-1:0] home_maintenance_range;
    wire [63:0] maintenance_base,maintenance_size;
    assign all_idle = cache_quiescent && !maintenance_busy;
    generate if(CACHE_MAINTENANCE!=0)begin:g_cache_handoff
        tnca_cache_handoff #(.N(N)) ownership(
            .clk(clk),.rst_n(rst_n),.valid(maintenance_requested),.ready(cache_maint_ready),
            .quiescent(cache_quiescent && !host_busy && (GLOBAL_GRID || !launch_valid)),
            .base(cache_maint_base),.size(cache_maint_size),
            .home_valid(maintenance_valid),.home_ready(maintenance_ready),.home_done(maintenance_done),
            .home_base(maintenance_base),.home_size(maintenance_size),
            .memory_drained(cache_maint_memory_drained && (&home_link_idle)),
            .done(cache_maint_done),.busy(maintenance_busy));
        for(genvar h=0;h<N;h=h+1)begin:g_home_link
            tnca_elastic_link #(.W(128),.STAGES(MEM_HOME_STAGES)) command(
                .clk(clk),.rst_n(rst_n),.in_valid(maintenance_valid[h]),.in_ready(maintenance_ready[h]),
                .in_data({maintenance_base,maintenance_size}),
                .out_valid(home_maintenance_valid[h]),.out_ready(home_maintenance_ready[h]),
                .out_data(home_maintenance_range[h*128+:128]),.empty());
            // One outstanding handoff per home; its response path is empty
            // before the next request, and the receiver always consumes it.
            tnca_elastic_link #(.W(1),.STAGES(MEM_HOME_STAGES)) response(
                .clk(clk),.rst_n(rst_n),.in_valid(home_maintenance_done[h]),.in_ready(),.in_data(1'b1),
                .out_valid(maintenance_done[h]),.out_ready(1'b1),.out_data(),.empty());
        end
    end else begin:g_no_cache_handoff
        assign cache_maint_ready=0;assign cache_maint_done=0;assign maintenance_busy=0;
        assign home_maintenance_valid=0;assign home_maintenance_range=0;
    end endgenerate

    wire [N-1:0] tile_launch, dispatch_busy;
    wire [N-1:0] host_readies, host_responses, host_busy_tiles;
    wire [N*64-1:0] host_response_data;
    wire dispatch_ready;
    wire host_allow_live=GLOBAL_GRID && HOST_ACCESS && dispatch_ready && !grid_valid && all_idle && !maintenance_requested;
    // Registered at the source; tile 0 registers it again in its own SLR.
    // The tile's grant uses a two-cycle-old permission. That is exact for the
    // grant: a host access is admitted only with the whole system idle and no
    // command or maintenance pending (tnca_multicore_mmio serializes them and
    // tnca_multicore_shell masks host_valid while a command is offered), so
    // every input of host_allow_live is constant from admission until the
    // access completes. A stale 1 after that has no h_valid to grant; a stale
    // 0 only delays the grant.
    reg host_allow;
    always @(posedge clk) host_allow <= rst_n && host_allow_live;
    assign host_ready=host_readies[0];assign host_rsp_valid=host_responses[0];
    assign host_busy=host_busy_tiles[0];assign host_rsp_data=host_response_data[63:0];
`ifndef SYNTHESIS
    // A grant (host_busy rising) must never happen where the unregistered
    // permission was false: the registered permission may only delay grants.
    reg host_allow_live_d=1'b0,host_busy_d=1'b0;
    always @(posedge clk) begin
        if(rst_n && host_busy && !host_busy_d && !host_allow_live_d)
            $fatal(1,"tnca_top: host grant without live quiescence permission");
        host_allow_live_d<=host_allow_live;host_busy_d<=rst_n && host_busy;
    end
`endif
    wire [63:0] dispatch_pc;
    wire [N*64-1:0] dispatch_args;
    wire [N*32-1:0] dispatch_x, dispatch_y, dispatch_z;
    wire [31:0] dispatch_nx, dispatch_ny, dispatch_nz;
    wire [NUM_RCTX-1:0] tile_roles;
    wire [3:0] tile_slice;
    wire [NUM_RCTX*PHYS_W-1:0] tile_vbase;
    wire [N*64-1:0] launch_pcs,launch_args;
    wire [N*192-1:0] launch_grids;
    wire [N*NUM_RCTX-1:0] launch_roles;
    wire [N*4-1:0] launch_slices;
    wire [N*NUM_RCTX*PHYS_W-1:0] launch_vbases;
    generate if (GLOBAL_GRID) begin: g_dispatch
        reg [NUM_RCTX-1:0] roles_q;
        reg [3:0] slice_q;
        reg [NUM_RCTX*PHYS_W-1:0] vbase_q;
        wire [N-1:0] command_valid,command_ready,completion_valid,completion_abort;
        wire [N*4-1:0] completion_status;
        wire [N*4-1:0] status;
        for(genvar d=0;d<N;d=d+1) begin: g_status
            assign status[d*4+:4]=fault_valid[d] ? 4'd1 : 4'd0;
        end
        always @(posedge clk) begin
            if(!rst_n) begin roles_q<=0;vbase_q<=0;slice_q<=0;end
            else if(grid_valid && grid_ready) begin roles_q<=launch_role_mask;vbase_q<=launch_vbase;slice_q<=grid_slice;end
        end
        assign grid_ready=dispatch_ready && !host_busy && !maintenance_requested && !maintenance_busy;
        assign tile_slice=slice_q;
        assign tile_roles=roles_q;assign tile_vbase=vbase_q;
        localparam COMMAND_W=128+192+NUM_RCTX+4+NUM_RCTX*PHYS_W;
        for(genvar c=0;c<N;c=c+1)begin:g_command_link
            wire cv,cr;
            wire [COMMAND_W-1:0] payload;
            tnca_elastic_link #(.W(COMMAND_W),.STAGES(DISPATCH_STAGES)) commands(
                .clk(clk),.rst_n(rst_n),.in_valid(command_valid[c]),.in_ready(command_ready[c]),
                .in_data({dispatch_pc,dispatch_args[c*64+:64],dispatch_x[c*32+:32],dispatch_y[c*32+:32],dispatch_z[c*32+:32],dispatch_nx,dispatch_ny,dispatch_nz,tile_roles,tile_slice,tile_vbase}),
                .out_valid(cv),.out_ready(cr),.out_data(payload),.empty());
            assign {launch_pcs[c*64+:64],launch_args[c*64+:64],launch_grids[c*192+:192],launch_roles[c*NUM_RCTX+:NUM_RCTX],launch_slices[c*4+:4],launch_vbases[c*NUM_RCTX*PHYS_W+:NUM_RCTX*PHYS_W]}=payload;
            if(DISPATCH_STAGES==0)begin:g_direct
                assign cr=tile_idle[c];assign tile_launch[c]=cv && cr;
                assign completion_valid[c]=dispatch_busy[c] && tile_idle[c];
                assign completion_abort[c]=fault_valid[c];assign completion_status[c*4+:4]=status[c*4+:4];
            end else begin:g_pipeline
                reg started;
                wire ready;
                assign cr=tile_idle[c] && !started;
                assign tile_launch[c]=cv && cr;
                always @(posedge clk)begin
                    if(!rst_n)started<=0;
                    else begin
                        if(tile_launch[c])started<=1;
                        if(started && tile_idle[c] && ready)started<=0;
                    end
                end
                tnca_elastic_link #(.W(4),.STAGES(DISPATCH_STAGES)) completions(
                    .clk(clk),.rst_n(rst_n),.in_valid(started && tile_idle[c]),.in_ready(ready),.in_data(status[c*4+:4]),
                    .out_valid(completion_valid[c]),.out_ready(1'b1),.out_data(completion_status[c*4+:4]),.empty());
                assign completion_abort[c]=completion_valid[c] && completion_status[c*4+:4]!=0;
            end
        end
        tnca_grid_dispatch #(.CORE_N(N)) u_dispatch (
            .clk(clk),.rst_n(rst_n),.abort_valid(grid_abort || (|memory_protocol_fault)),.abort_status((|memory_protocol_fault) ? 4'd1 : grid_abort_status),.launch_valid(grid_valid && !host_busy && !maintenance_requested && !maintenance_busy),.launch_ready(dispatch_ready),
            .launch_group_log2(grid_group_log2),.launch_id(grid_id),.launch_pc(launch_pc),.launch_arg_base(launch_arg_base),
            .launch_arg_stride(grid_arg_stride),.launch_gx(grid_x),.launch_gy(grid_y),.launch_gz(grid_z),
            .launch_cores((|launch_role_mask) ? grid_cores : {N{1'b0}}),
            .cmd_valid(command_valid),.cmd_ready(command_ready),.cmd_id(),.cmd_pc(dispatch_pc),
            .cmd_arg_base(dispatch_args),.cmd_x(dispatch_x),.cmd_y(dispatch_y),.cmd_z(dispatch_z),
            .cmd_nx(dispatch_nx),.cmd_ny(dispatch_ny),.cmd_nz(dispatch_nz),.cmd_cancel(),
            .core_done(completion_valid),.core_abort(completion_abort),.core_status(completion_status),
            .done_valid(grid_done_valid),.done_ready(grid_done_ready),.done_status(grid_done_status),
            .done_id(grid_done_id),.done_cycles(grid_done_cycles),.fault_core(grid_fault_core),
            .core_busy(dispatch_busy),.issued(grid_issued),.completed(grid_completed),
            .peak_active(grid_peak_active),.active_cycles(grid_active_cycles));
    end else begin: g_legacy_launch
        assign tile_launch={N{launch_valid && !maintenance_requested && !maintenance_busy}};
        assign launch_pcs={N{launch_pc}};assign launch_args={N{launch_arg_base}};
        assign launch_grids=0;assign launch_roles={N{launch_role_mask}};
        assign launch_slices=0;assign launch_vbases={N{launch_vbase}};
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
    wire [N*(8)-1:0] d_policy;
    wire [N*(4)-1:0] d_bbits;
    wire [N*(4)-1:0] d_ginit;
    wire [N*(6)-1:0] d_lsb;
    wire [N*(16)-1:0] d_win;
    wire [N*(16)-1:0] d_ub;
    wire [N*(16)-1:0] d_lb;
    wire [N*(1)-1:0] d_owner_epoch;
    wire [N*(DCO_T)-1:0] d_tv;
    wire [N*(DCO_T)-1:0] d_tbyp;
    wire [N*(DCO_T*64)-1:0] d_tbase;
    wire [N*(DCO_T*64)-1:0] d_tsize;
    wire [N*(DCO_T*65)-1:0] d_tend;
    wire [N*(DCO_T*6)-1:0] d_tlsb;
    wire [N*(DCO_DEAD)-1:0] d_dv;
    wire [N*(DCO_DEAD*DCO_ID_W)-1:0] d_did;
    wire [N*76-1:0] dco_cfg;
    genvar dco_copy;
    generate for(dco_copy=0;dco_copy<N;dco_copy=dco_copy+1)begin:g_dco_cfg
        assign dco_cfg[dco_copy*76+:76]={d_lb[dco_copy*16+:16],d_ub[dco_copy*16+:16],d_win[dco_copy*16+:16],
            d_ginit[dco_copy*4+:4],6'd0,d_lsb[dco_copy*6+:6],d_bbits[dco_copy*4+:4],d_policy[dco_copy*8+:8]};
    end endgenerate
    generate if(DCO_LOCAL_REPLICAS)begin:g_local_tmu
    tnca_replicated_tmu #(
        .TLW_PIPE_STAGES(DCO_TLW_PIPE_STAGES), .REGISTER_WRITES(DCO_REGISTER_WRITES),
        .N_SLICE(N), .SRC_W(SRC_W), .N_SRC(N), .REGISTERED_RANGES(DCO_REGISTERED_RANGES),
        .TENSORS(DCO_T), .T_W(DCO_T_W), .DEAD(DCO_DEAD), .DEAD_W(4),
        .ID_W(DCO_ID_W)
    ) u_tmu (
        .clk(clk), .rst_n(rst_n),
        .mw_valid(tm_wv), .mw_ready(tm_wr), .mw_off(tm_woff), .mw_data(tm_wdata),
        .mw_src(tm_wsrc), .mr_off(tm_roff), .mr_data(tm_rdata),
        .tl_valid(tm_tv), .tl_ready(tm_tr), .tl_addr(tm_taddr), .tl_drop(tm_drop),
        .policy(d_policy), .b_bits(d_bbits), .d_lsb(d_lsb),
        .gear_init(d_ginit), .gear_window(d_win), .gear_ub(d_ub), .gear_lb(d_lb),
        .owner_epoch(d_owner_epoch), .t_valid(d_tv), .t_base(d_tbase), .t_size(d_tsize), .t_end(d_tend), .t_bypass(d_tbyp),
        .t_lsb(d_tlsb),
        .dead_valid(d_dv), .dead_id(d_did)
    );

    end else begin:g_shared_tmu
    tnca_tmu #(
        .TLW_PIPE_STAGES(DCO_TLW_PIPE_STAGES), .REGISTER_WRITES(DCO_REGISTER_WRITES),
        .N_SLICE(N), .SRC_W(SRC_W), .N_SRC(N), .REGISTERED_RANGES(DCO_REGISTERED_RANGES),
        .TENSORS(DCO_T), .T_W(DCO_T_W), .DEAD(DCO_DEAD), .DEAD_W(4),
        .ID_W(DCO_ID_W)
    ) u_tmu (
        .clk(clk), .rst_n(rst_n),
        .mw_valid(tm_wv), .mw_ready(tm_wr), .mw_off(tm_woff), .mw_data(tm_wdata),
        .mw_src(tm_wsrc), .mr_off(tm_roff), .mr_data(tm_rdata),
        .tl_valid(tm_tv), .tl_ready(tm_tr), .tl_addr(tm_taddr), .tl_drop(tm_drop),
        .policy(d_policy[0+:(8)]), .b_bits(d_bbits[0+:(4)]), .d_lsb(d_lsb[0+:(6)]),
        .gear_init(d_ginit[0+:(4)]), .gear_window(d_win[0+:(16)]), .gear_ub(d_ub[0+:(16)]), .gear_lb(d_lb[0+:(16)]),
        .owner_epoch(d_owner_epoch[0+:(1)]), .t_valid(d_tv[0+:(DCO_T)]), .t_base(d_tbase[0+:(DCO_T*64)]), .t_size(d_tsize[0+:(DCO_T*64)]), .t_end(d_tend[0+:(DCO_T*65)]), .t_bypass(d_tbyp[0+:(DCO_T)]),
        .t_lsb(d_tlsb[0+:(DCO_T*6)]),
        .dead_valid(d_dv[0+:(DCO_DEAD)]), .dead_id(d_did[0+:(DCO_DEAD*DCO_ID_W)])
    );

        for(dco_copy=1;dco_copy<N;dco_copy=dco_copy+1)begin:g_broadcast
            assign d_policy[dco_copy*(8)+:(8)]=d_policy[0+:(8)];
            assign d_bbits[dco_copy*(4)+:(4)]=d_bbits[0+:(4)];
            assign d_ginit[dco_copy*(4)+:(4)]=d_ginit[0+:(4)];
            assign d_lsb[dco_copy*(6)+:(6)]=d_lsb[0+:(6)];
            assign d_win[dco_copy*(16)+:(16)]=d_win[0+:(16)];
            assign d_ub[dco_copy*(16)+:(16)]=d_ub[0+:(16)];
            assign d_lb[dco_copy*(16)+:(16)]=d_lb[0+:(16)];
            assign d_owner_epoch[dco_copy*(1)+:(1)]=d_owner_epoch[0+:(1)];
            assign d_tv[dco_copy*(DCO_T)+:(DCO_T)]=d_tv[0+:(DCO_T)];
            assign d_tbyp[dco_copy*(DCO_T)+:(DCO_T)]=d_tbyp[0+:(DCO_T)];
            assign d_tbase[dco_copy*(DCO_T*64)+:(DCO_T*64)]=d_tbase[0+:(DCO_T*64)];
            assign d_tsize[dco_copy*(DCO_T*64)+:(DCO_T*64)]=d_tsize[0+:(DCO_T*64)];
            assign d_tend[dco_copy*(DCO_T*65)+:(DCO_T*65)]=d_tend[0+:(DCO_T*65)];
            assign d_tlsb[dco_copy*(DCO_T*6)+:(DCO_T*6)]=d_tlsb[0+:(DCO_T*6)];
            assign d_dv[dco_copy*(DCO_DEAD)+:(DCO_DEAD)]=d_dv[0+:(DCO_DEAD)];
            assign d_did[dco_copy*(DCO_DEAD*DCO_ID_W)+:(DCO_DEAD*DCO_ID_W)]=d_did[0+:(DCO_DEAD*DCO_ID_W)];
        end
    end endgenerate

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
                .HYBRID_LSU(HYBRID_LSU),.HOME_BACKGROUND_READS(HOME_BACKGROUND_READS),.HOME_LINE_WRITE_NOFILL(HOME_LINE_WRITE_NOFILL),.HOME_LINE_PIPE(HOME_LINE_PIPE),.TENSOR_LINE(TENSOR_LINE),.SFU_LANES(SFU_LANES),.NOC_PAYLOAD_W(NOC_PAYLOAD_W),
                .CACHE_MAINTENANCE(CACHE_MAINTENANCE), .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES), .NUCA_REGISTER_VICTIM_SELECT(NUCA_REGISTER_VICTIM_SELECT), .NUCA_CACHE_DATA_PIPELINE(NUCA_CACHE_DATA_PIPELINE), .NUCA_SHARED_DATA_READ(NUCA_SHARED_DATA_READ), .NUCA_REGISTER_VICTIM_RANK(NUCA_REGISTER_VICTIM_RANK),
                .TILE_ID(t), .NUM_TILES(N), .GLOBAL_GRID(GLOBAL_GRID), .HOST_ACCESS(HOST_ACCESS && GLOBAL_GRID && t==0), .NX(NX), .NY(NY), .XW(XW), .YW(YW),
                .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN),
                .LW(LW), .VREGS_PER_RCTX(VREGS_PER_RCTX), .ACC_N(ACC_N), .ACC_W(ACC_W), .BAR_N(BAR_N), .BAR_W(BAR_W), .BAR_EPOCH_W(BAR_EPOCH_W), .FMA_IMPL(FMA_IMPL), .MAC_DP(MAC_DP), .MAC_CUTS(MAC_CUTS), .PHYS_ADDR_W(PHYS_ADDR_W),
                .PHYS_VREGS(PHYS_VREGS), .PHYS_W(PHYS_W),
                .USE_BANKED_VRF(USE_BANKED_VRF),
                .VRF_BYPASS(VRF_BYPASS),
                .FE_NOBUBBLE(FE_NOBUBBLE),
                .VRF_PREFETCH(VRF_PREFETCH),
                .RCTX_SWITCH(RCTX_SWITCH),
                .MTX_ABUF(MTX_ABUF),
                .MTX_VRF_REG(MTX_VRF_REG),
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
                .launch_valid(tile_launch[t]), .launch_pc(launch_pcs[t*64+:64]),
                .cache_maint_valid(home_maintenance_valid[t]),.cache_maint_ready(home_maintenance_ready[t]),.cache_maint_done(home_maintenance_done[t]),
                .cache_maint_base(home_maintenance_range[t*128+64+:64]),.cache_maint_size(home_maintenance_range[t*128+:64]),
                .launch_role_mask(launch_roles[t*NUM_RCTX+:NUM_RCTX]),
                .launch_arg_base(launch_args[t*64+:64]),
                .launch_vbase(launch_vbases[t*NUM_RCTX*PHYS_W+:NUM_RCTX*PHYS_W]),
                .grid_slice(launch_slices[t*4+:4]), .grid_pid_x(launch_grids[t*192+160+:32]), .grid_pid_y(launch_grids[t*192+128+:32]),
                .grid_pid_z(launch_grids[t*192+96+:32]), .grid_npid_x(launch_grids[t*192+64+:32]),
                .grid_npid_y(launch_grids[t*192+32+:32]), .grid_npid_z(launch_grids[t*192+:32]),
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
                .dco_owner_epoch(d_owner_epoch[t*(1)+:(1)]), .dco_cfg(dco_cfg[t*76+:76]), .dco_t_valid(d_tv[t*(DCO_T)+:(DCO_T)]), .dco_t_base(d_tbase[t*(DCO_T*64)+:(DCO_T*64)]),
                .dco_t_end(d_tend[t*(DCO_T*65)+:(DCO_T*65)]), .dco_t_size(d_tsize[t*(DCO_T*64)+:(DCO_T*64)]), .dco_t_bypass(d_tbyp[t*(DCO_T)+:(DCO_T)]), .dco_t_lsb(d_tlsb[t*(DCO_T*6)+:(DCO_T*6)]),
                .dco_dead_valid(d_dv[t*(DCO_DEAD)+:(DCO_DEAD)]), .dco_dead_id(d_did[t*(DCO_DEAD*DCO_ID_W)+:(DCO_DEAD*DCO_ID_W)]), .dco_slow(dco_slow),
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
        .PAYLOAD_W(NOC_PAYLOAD_W), .NV(NV),
        .DEPTH(NOC_DEPTH), .AW(NOC_AW),.LINK_STAGES(NOC_LINK_STAGES)
    ) u_mesh (
        .clk(clk), .rst_n(rst_n),
        .l_in_valid(li_valid), .l_in_ready(li_ready), .l_in_flit(li_flit),
        .l_out_valid(lo_valid), .l_out_ready(lo_ready), .l_out_flit(lo_flit)
    );

    always @(posedge clk)begin
        if(!rst_n)begin memory_reads<=0;memory_writes<=0;end
        else if(MEM_PORTS==1)begin
            if(mem_req_valid && mem_req_ready)begin
                if(mem_req_we)memory_writes<=memory_writes+1'b1;
                else memory_reads<=memory_reads+1'b1;
            end
        end else begin
            for(integer outlet=0;outlet<MEM_PORTS;outlet=outlet+1)
                if(ports_req_valid[outlet] && ports_req_ready[outlet])begin
                    if(ports_req_we[outlet])memory_writes[outlet*64+:64]<=memory_writes[outlet*64+:64]+1'b1;
                    else memory_reads[outlet*64+:64]<=memory_reads[outlet*64+:64]+1'b1;
                end
        end
    end
    generate if(MEM_PORTS==1) begin: g_single_memory
    tnca_mem_arbiter #(
        .N(N), .N_W(MEM_SRC_W), .ADDR_W(64), .LINE_BITS(LINE_BITS), .TAG_W(MSHR_W)
    ) u_marb (
        .clk(clk), .rst_n(rst_n),
        .s_req_valid(outlet_req_valid), .s_req_ready(outlet_req_ready),
        .s_req_addr(outlet_req_addr), .s_req_we(outlet_req_we),
        .s_req_wdata(outlet_req_wdata), .s_req_tag(outlet_req_tag),
        .s_rsp_valid(outlet_rsp_valid), .s_rsp_tag(outlet_rsp_tag),
        .s_rsp_rdata(outlet_rsp_rdata),
        .m_req_valid(mem_req_valid), .m_req_ready(mem_req_ready),
        .m_req_addr(mem_req_addr), .m_req_we(mem_req_we),
        .m_req_wdata(mem_req_wdata), .m_req_tag(mem_req_tag),
        .m_req_src(mem_req_src),
        .m_rsp_valid(mem_rsp_valid), .m_rsp_tag(mem_rsp_tag),
        .m_rsp_src(mem_rsp_src), .m_rsp_rdata(mem_rsp_rdata)
    );

        assign ports_req_valid=0;assign ports_req_addr=0;assign ports_req_we=0;
        assign ports_req_wdata=0;assign ports_req_tag=0;assign ports_req_src=0;
        reg single_protocol_fault;
        always @(posedge clk)begin
            if(!rst_n)single_protocol_fault<=0;
            else if(mem_rsp_valid && mem_rsp_src>=N)single_protocol_fault<=1;
        end
        assign memory_protocol_fault=single_protocol_fault;
    end else begin: g_multiple_memory
        assign mem_req_valid=0;assign mem_req_addr=0;assign mem_req_we=0;
        assign mem_req_wdata=0;assign mem_req_tag=0;assign mem_req_src=0;
        tnca_mem_ports #(.N(N),.PORTS(MEM_PORTS),.N_W(MEM_SRC_W),
          .LINE_BITS(LINE_BITS),.TAG_W(MSHR_W)) outlets(
          .clk(clk),.rst_n(rst_n),.s_req_valid(outlet_req_valid),.s_req_ready(outlet_req_ready),
          .s_req_addr(outlet_req_addr),.s_req_we(outlet_req_we),.s_req_wdata(outlet_req_wdata),.s_req_tag(outlet_req_tag),
          .s_rsp_valid(outlet_rsp_valid),.s_rsp_tag(outlet_rsp_tag),.s_rsp_rdata(outlet_rsp_rdata),
          .m_req_valid(ports_req_valid),.m_req_ready(ports_req_ready),.m_req_addr(ports_req_addr),
          .m_req_we(ports_req_we),.m_req_wdata(ports_req_wdata),.m_req_tag(ports_req_tag),
          .m_req_src(ports_req_src),.m_rsp_valid(ports_rsp_valid),.m_rsp_tag(ports_rsp_tag),
          .m_rsp_src(ports_rsp_src),.m_rsp_rdata(ports_rsp_rdata),.protocol_fault(memory_protocol_fault));
    end endgenerate

endmodule
