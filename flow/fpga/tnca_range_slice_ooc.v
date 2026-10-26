// Boundary-aware OOC wrapper: actual three-requester TMU and actual slice.
// Metadata is clocked by the producer; range arithmetic cannot hide at an unconstrained input.
module tnca_range_slice_ooc #(

    parameter ADDR_W     = 64,
    parameter LINE_BYTES = 64,
    parameter LINE_BITS  = 512,
    parameter WORD_BITS  = 64,
    parameter WORD_BYTES = 8,
    parameter SETS       = 64,
    parameter SET_W      = 6,        
    parameter WAYS       = 8,
    parameter WAY_W      = 3,        
    parameter MSHR_N     = 8,
    parameter MSHR_W     = 3,
    parameter TARGET_N   = 16,
    parameter TARGET_W   = 4,
    parameter RQ_N       = 8,        
    parameter RQ_W       = 3,
    parameter SRC_W      = 4,
    parameter TAG_W      = 4,
    
    
    
    parameter SLICE_ID   = 0,
    parameter HOME_W     = 2,
    
    parameter FULL_LINE_TAG = 0,
    
    
    parameter DCO_T      = 8,
    parameter DCO_REGISTERED_RANGES = 0,
    parameter SYNC_TAG_BANK = 1,
    parameter REGISTER_VICTIM_OWNER = 0,
    parameter BALANCED_VICTIM = 1,
    parameter DCO_DEAD   = 16,
    parameter MAINTENANCE = 0,
    parameter DCO_ID_W   = 16,
    parameter [63:0] TMU_BASE = 64'h0000_0000_7FFF_0000

)(
input  wire                     clk,
input  wire                     rst_n,
input  wire                     maint_valid,
output wire                     maint_ready,
input  wire [ADDR_W-1:0]         maint_base, maint_size,
output wire                      maint_done,
output wire                     maint_busy,
output wire                     req_ready,
input  wire                     req_valid,
input  wire [ADDR_W-1:0]        req_addr,
input  wire                     req_we,
input  wire [WORD_BYTES-1:0]    req_be,
input  wire [WORD_BITS-1:0]     req_wdata,
input  wire [SRC_W-1:0]         req_src,
input  wire [TAG_W-1:0]         req_tag,
input  wire                     req_amo,
input  wire [3:0]               req_aop,
input  wire [WORD_BITS-1:0]     req_cmp,
output wire                     rsp_valid,
input  wire                     rsp_ready,
output wire [SRC_W-1:0]         rsp_src,
output wire [TAG_W-1:0]         rsp_tag,
output wire [WORD_BITS-1:0]     rsp_rdata,
output wire                     mem_req_valid,
input  wire                     mem_req_ready,
output wire [ADDR_W-1:0]        mem_req_addr,
output wire                     mem_req_we,
output wire [LINE_BITS-1:0]     mem_req_wdata,
output wire [MSHR_W-1:0]        mem_req_tag,
input  wire                     mem_rsp_valid,
input  wire [MSHR_W-1:0]        mem_rsp_tag,
input  wire [LINE_BITS-1:0]     mem_rsp_rdata,
output wire  [31:0]              perf_bypass,
output wire  [31:0]              perf_evict,
output wire  [31:0]              perf_dead_evict,
output wire  [3:0]               perf_gear,
output wire  [31:0]              perf_hit,
output wire  [31:0]              perf_miss,
output wire  [31:0]              perf_merge,
output wire  [31:0]              perf_mshr_full,
output wire  [31:0]              perf_target_full,
output wire  [31:0]              perf_wb,
output wire  [31:0]              perf_hit_past_block,
input wire [15:0] dco_slow,
input wire [1:0] peer_wr_valid,
output wire [1:0] peer_wr_ready,
input wire [23:0] peer_wr_off,
input wire [127:0] peer_wr_data,
input wire [7:0] peer_wr_src,
input wire [1:0] peer_tl_valid,
output wire [1:0] peer_tl_ready,
input wire [127:0] peer_tl_addr,
input wire [1:0] peer_tl_drop
);
wire [75:0]              dco_cfg;
wire [DCO_T-1:0]         dco_t_valid;
wire [DCO_T*64-1:0]      dco_t_base;
wire [DCO_T*64-1:0]      dco_t_size;
wire [DCO_T*65-1:0]      dco_t_end;
wire [DCO_T-1:0]         dco_t_bypass;
wire [DCO_T*6-1:0]       dco_t_lsb;
wire [DCO_DEAD-1:0]      dco_dead_valid;
wire [DCO_DEAD*DCO_ID_W-1:0] dco_dead_id;
wire tmu_wr_valid;
wire tmu_wr_ready;
wire [11:0]              tmu_wr_off;
wire [63:0]              tmu_wr_data;
wire [SRC_W-1:0]         tmu_wr_src;
wire [11:0]              tmu_rd_off;
wire [63:0]              tmu_rd_data;
wire tmu_tl_valid;
wire tmu_tl_ready;
wire [63:0]              tmu_tl_addr;
wire tmu_tl_drop;
wire [7:0] policy; wire [3:0] bbits,ginit; wire [5:0] granule;
wire [15:0] win,ub,lb; wire [2:0] wr_ready,tl_ready; wire [191:0] read_data;
assign dco_cfg={lb,ub,win,ginit,6'd0,granule,bbits,policy};
assign tmu_wr_ready=wr_ready[0];assign peer_wr_ready=wr_ready[2:1];
assign tmu_tl_ready=tl_ready[0];assign peer_tl_ready=tl_ready[2:1];
assign tmu_rd_data=read_data[63:0];
wire owner_epoch;
tnca_tmu #(.REGISTERED_RANGES(DCO_REGISTERED_RANGES),.N_SLICE(3),.N_SRC(3),.SRC_W(4),.TENSORS(DCO_T),.T_W(DCO_T>1?$clog2(DCO_T):1),.DEAD(DCO_DEAD),.ID_W(DCO_ID_W)) tmu(
.clk(clk),.rst_n(rst_n),.mw_valid({peer_wr_valid,tmu_wr_valid}),.mw_ready(wr_ready),
.mw_off({peer_wr_off,tmu_wr_off}),.mw_data({peer_wr_data,tmu_wr_data}),.mw_src({peer_wr_src,tmu_wr_src}),
.mr_off({24'd0,tmu_rd_off}),.mr_data(read_data),.tl_valid({peer_tl_valid,tmu_tl_valid}),.tl_ready(tl_ready),
.tl_addr({peer_tl_addr,tmu_tl_addr}),.tl_drop({peer_tl_drop,tmu_tl_drop}),
.policy(policy),.b_bits(bbits),.d_lsb(granule),.gear_init(ginit),.gear_window(win),.gear_ub(ub),.gear_lb(lb),
.owner_epoch(owner_epoch),.t_valid(dco_t_valid),.t_base(dco_t_base),.t_size(dco_t_size),.t_end(dco_t_end),
.t_bypass(dco_t_bypass),.t_lsb(dco_t_lsb),.dead_valid(dco_dead_valid),.dead_id(dco_dead_id));
tnca_nuca_slice #(
.ADDR_W(ADDR_W),
.LINE_BYTES(LINE_BYTES),
.LINE_BITS(LINE_BITS),
.WORD_BITS(WORD_BITS),
.WORD_BYTES(WORD_BYTES),
.SETS(SETS),
.SET_W(SET_W),
.WAYS(WAYS),
.WAY_W(WAY_W),
.MSHR_N(MSHR_N),
.MSHR_W(MSHR_W),
.TARGET_N(TARGET_N),
.TARGET_W(TARGET_W),
.RQ_N(RQ_N),
.RQ_W(RQ_W),
.SRC_W(SRC_W),
.TAG_W(TAG_W),
.SLICE_ID(SLICE_ID),
.HOME_W(HOME_W),
.FULL_LINE_TAG(FULL_LINE_TAG),
.DCO_T(DCO_T),
.DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES),
.SYNC_TAG_BANK(SYNC_TAG_BANK),
.REGISTER_VICTIM_OWNER(REGISTER_VICTIM_OWNER),
.BALANCED_VICTIM(BALANCED_VICTIM),
.DCO_DEAD(DCO_DEAD),
.MAINTENANCE(MAINTENANCE),
.DCO_ID_W(DCO_ID_W),
.TMU_BASE(TMU_BASE)
) slice(
.clk(clk),
.rst_n(rst_n),
.maint_valid(maint_valid),
.maint_ready(maint_ready),
.maint_base(maint_base),
.maint_size(maint_size),
.maint_done(maint_done),
.maint_busy(maint_busy),
.req_ready(req_ready),
.req_valid(req_valid),
.req_addr(req_addr),
.req_we(req_we),
.req_be(req_be),
.req_wdata(req_wdata),
.req_src(req_src),
.req_tag(req_tag),
.req_amo(req_amo),
.req_aop(req_aop),
.req_cmp(req_cmp),
.rsp_valid(rsp_valid),
.rsp_ready(rsp_ready),
.rsp_src(rsp_src),
.rsp_tag(rsp_tag),
.rsp_rdata(rsp_rdata),
.mem_req_valid(mem_req_valid),
.mem_req_ready(mem_req_ready),
.mem_req_addr(mem_req_addr),
.mem_req_we(mem_req_we),
.mem_req_wdata(mem_req_wdata),
.mem_req_tag(mem_req_tag),
.mem_rsp_valid(mem_rsp_valid),
.mem_rsp_tag(mem_rsp_tag),
.mem_rsp_rdata(mem_rsp_rdata),
.dco_owner_epoch(owner_epoch),.dco_cfg(dco_cfg),
.dco_t_valid(dco_t_valid),
.dco_t_base(dco_t_base),
.dco_t_size(dco_t_size),
.dco_t_end(dco_t_end),
.dco_t_bypass(dco_t_bypass),
.dco_t_lsb(dco_t_lsb),
.dco_dead_valid(dco_dead_valid),
.dco_dead_id(dco_dead_id),
.dco_slow(dco_slow),
.tmu_wr_valid(tmu_wr_valid),
.tmu_wr_ready(tmu_wr_ready),
.tmu_wr_off(tmu_wr_off),
.tmu_wr_data(tmu_wr_data),
.tmu_wr_src(tmu_wr_src),
.tmu_rd_off(tmu_rd_off),
.tmu_rd_data(tmu_rd_data),
.tmu_tl_valid(tmu_tl_valid),
.tmu_tl_ready(tmu_tl_ready),
.tmu_tl_addr(tmu_tl_addr),
.tmu_tl_drop(tmu_tl_drop),
.perf_bypass(perf_bypass),
.perf_evict(perf_evict),
.perf_dead_evict(perf_dead_evict),
.perf_gear(perf_gear),
.perf_hit(perf_hit),
.perf_miss(perf_miss),
.perf_merge(perf_merge),
.perf_mshr_full(perf_mshr_full),
.perf_target_full(perf_target_full),
.perf_wb(perf_wb),
.perf_hit_past_block(perf_hit_past_block)
);
endmodule
