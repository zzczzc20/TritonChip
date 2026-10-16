// Three real slices sharing one real TMU, for physical screening only.
// No cores, NoC or HBM shell: this cannot qualify full-system packing, timing,
// image or workload gates. Every slice input is connected; no behavioral DUT.
module tnca_range_grid_ooc #(
    parameter N = 3,

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
input wire clk,
input wire rst_n,
input wire [N*(1)-1:0] maint_valid,
output wire [N*(1)-1:0] maint_ready,
input wire [N*(ADDR_W)-1:0] maint_base,
input wire [N*(ADDR_W)-1:0] maint_size,
output wire [N*(1)-1:0] maint_done,
output wire [N*(1)-1:0] maint_busy,
output wire [N*(1)-1:0] req_ready,
input wire [N*(1)-1:0] req_valid,
input wire [N*(ADDR_W)-1:0] req_addr,
input wire [N*(1)-1:0] req_we,
input wire [N*(WORD_BYTES)-1:0] req_be,
input wire [N*(WORD_BITS)-1:0] req_wdata,
input wire [N*(SRC_W)-1:0] req_src,
input wire [N*(TAG_W)-1:0] req_tag,
input wire [N*(1)-1:0] req_amo,
input wire [N*(4)-1:0] req_aop,
input wire [N*(WORD_BITS)-1:0] req_cmp,
output wire [N*(1)-1:0] rsp_valid,
input wire [N*(1)-1:0] rsp_ready,
output wire [N*(SRC_W)-1:0] rsp_src,
output wire [N*(TAG_W)-1:0] rsp_tag,
output wire [N*(WORD_BITS)-1:0] rsp_rdata,
output wire [N*(1)-1:0] mem_req_valid,
input wire [N*(1)-1:0] mem_req_ready,
output wire [N*(ADDR_W)-1:0] mem_req_addr,
output wire [N*(1)-1:0] mem_req_we,
output wire [N*(LINE_BITS)-1:0] mem_req_wdata,
output wire [N*(MSHR_W)-1:0] mem_req_tag,
input wire [N*(1)-1:0] mem_rsp_valid,
input wire [N*(MSHR_W)-1:0] mem_rsp_tag,
input wire [N*(LINE_BITS)-1:0] mem_rsp_rdata,
input wire [16-1:0] dco_slow,
output wire [N*(32)-1:0] perf_bypass,
output wire [N*(32)-1:0] perf_evict,
output wire [N*(32)-1:0] perf_dead_evict,
output wire [N*(4)-1:0] perf_gear,
output wire [N*(32)-1:0] perf_hit,
output wire [N*(32)-1:0] perf_miss,
output wire [N*(32)-1:0] perf_merge,
output wire [N*(32)-1:0] perf_mshr_full,
output wire [N*(32)-1:0] perf_target_full,
output wire [N*(32)-1:0] perf_wb,
output wire [N*(32)-1:0] perf_hit_past_block
);
wire [(76)-1:0] dco_cfg;
wire [(1)-1:0] dco_owner_epoch;
wire [(DCO_T)-1:0] dco_t_valid;
wire [(DCO_T*64)-1:0] dco_t_base;
wire [(DCO_T*64)-1:0] dco_t_size;
wire [(DCO_T*65)-1:0] dco_t_end;
wire [(DCO_T)-1:0] dco_t_bypass;
wire [(DCO_T*6)-1:0] dco_t_lsb;
wire [(DCO_DEAD)-1:0] dco_dead_valid;
wire [(DCO_DEAD*DCO_ID_W)-1:0] dco_dead_id;
wire [N*(1)-1:0] tmu_wr_valid;
wire [N*(1)-1:0] tmu_wr_ready;
wire [N*(12)-1:0] tmu_wr_off;
wire [N*(64)-1:0] tmu_wr_data;
wire [N*(SRC_W)-1:0] tmu_wr_src;
wire [N*(12)-1:0] tmu_rd_off;
wire [N*(64)-1:0] tmu_rd_data;
wire [N*(1)-1:0] tmu_tl_valid;
wire [N*(1)-1:0] tmu_tl_ready;
wire [N*(64)-1:0] tmu_tl_addr;
wire [N*(1)-1:0] tmu_tl_drop;
wire [7:0] policy;
wire [3:0] bbits,ginit;
wire [5:0] granule;
wire [15:0] win,ub,lb;
assign dco_cfg={lb,ub,win,ginit,6'd0,granule,bbits,policy};
tnca_tmu #(.REGISTERED_RANGES(DCO_REGISTERED_RANGES),.N_SLICE(N),.N_SRC(N),
 .SRC_W(SRC_W),.TENSORS(DCO_T),.T_W(DCO_T>1?$clog2(DCO_T):1),
 .DEAD(DCO_DEAD),.ID_W(DCO_ID_W)) tmu(
 .clk(clk),.rst_n(rst_n),.mw_valid(tmu_wr_valid),.mw_ready(tmu_wr_ready),
 .mw_off(tmu_wr_off),.mw_data(tmu_wr_data),.mw_src(tmu_wr_src),
 .mr_off(tmu_rd_off),.mr_data(tmu_rd_data),.tl_valid(tmu_tl_valid),
 .tl_ready(tmu_tl_ready),.tl_addr(tmu_tl_addr),.tl_drop(tmu_tl_drop),
 .policy(policy),.b_bits(bbits),.d_lsb(granule),.gear_init(ginit),
 .gear_window(win),.gear_ub(ub),.gear_lb(lb),.owner_epoch(dco_owner_epoch),
 .t_valid(dco_t_valid),.t_base(dco_t_base),.t_size(dco_t_size),.t_end(dco_t_end),
 .t_bypass(dco_t_bypass),.t_lsb(dco_t_lsb),.dead_valid(dco_dead_valid),.dead_id(dco_dead_id));
genvar tile;
generate for(tile=0;tile<N;tile=tile+1)begin:g_tile
 // P1_CACHE_BOUNDARY_ATTRIBUTE: experiment varies only this instance attribute.
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
.SLICE_ID(tile),
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
) u_slice(
.clk(clk),
.rst_n(rst_n),
.maint_valid(maint_valid[tile*(1) +: (1)]),
.maint_ready(maint_ready[tile*(1) +: (1)]),
.maint_base(maint_base[tile*(ADDR_W) +: (ADDR_W)]),
.maint_size(maint_size[tile*(ADDR_W) +: (ADDR_W)]),
.maint_done(maint_done[tile*(1) +: (1)]),
.maint_busy(maint_busy[tile*(1) +: (1)]),
.req_ready(req_ready[tile*(1) +: (1)]),
.req_valid(req_valid[tile*(1) +: (1)]),
.req_addr(req_addr[tile*(ADDR_W) +: (ADDR_W)]),
.req_we(req_we[tile*(1) +: (1)]),
.req_be(req_be[tile*(WORD_BYTES) +: (WORD_BYTES)]),
.req_wdata(req_wdata[tile*(WORD_BITS) +: (WORD_BITS)]),
.req_src(req_src[tile*(SRC_W) +: (SRC_W)]),
.req_tag(req_tag[tile*(TAG_W) +: (TAG_W)]),
.req_amo(req_amo[tile*(1) +: (1)]),
.req_aop(req_aop[tile*(4) +: (4)]),
.req_cmp(req_cmp[tile*(WORD_BITS) +: (WORD_BITS)]),
.rsp_valid(rsp_valid[tile*(1) +: (1)]),
.rsp_ready(rsp_ready[tile*(1) +: (1)]),
.rsp_src(rsp_src[tile*(SRC_W) +: (SRC_W)]),
.rsp_tag(rsp_tag[tile*(TAG_W) +: (TAG_W)]),
.rsp_rdata(rsp_rdata[tile*(WORD_BITS) +: (WORD_BITS)]),
.mem_req_valid(mem_req_valid[tile*(1) +: (1)]),
.mem_req_ready(mem_req_ready[tile*(1) +: (1)]),
.mem_req_addr(mem_req_addr[tile*(ADDR_W) +: (ADDR_W)]),
.mem_req_we(mem_req_we[tile*(1) +: (1)]),
.mem_req_wdata(mem_req_wdata[tile*(LINE_BITS) +: (LINE_BITS)]),
.mem_req_tag(mem_req_tag[tile*(MSHR_W) +: (MSHR_W)]),
.mem_rsp_valid(mem_rsp_valid[tile*(1) +: (1)]),
.mem_rsp_tag(mem_rsp_tag[tile*(MSHR_W) +: (MSHR_W)]),
.mem_rsp_rdata(mem_rsp_rdata[tile*(LINE_BITS) +: (LINE_BITS)]),
.dco_cfg(dco_cfg),
.dco_owner_epoch(dco_owner_epoch),
.dco_t_valid(dco_t_valid),
.dco_t_base(dco_t_base),
.dco_t_size(dco_t_size),
.dco_t_end(dco_t_end),
.dco_t_bypass(dco_t_bypass),
.dco_t_lsb(dco_t_lsb),
.dco_dead_valid(dco_dead_valid),
.dco_dead_id(dco_dead_id),
.dco_slow(dco_slow),
.tmu_wr_valid(tmu_wr_valid[tile*(1) +: (1)]),
.tmu_wr_ready(tmu_wr_ready[tile*(1) +: (1)]),
.tmu_wr_off(tmu_wr_off[tile*(12) +: (12)]),
.tmu_wr_data(tmu_wr_data[tile*(64) +: (64)]),
.tmu_wr_src(tmu_wr_src[tile*(SRC_W) +: (SRC_W)]),
.tmu_rd_off(tmu_rd_off[tile*(12) +: (12)]),
.tmu_rd_data(tmu_rd_data[tile*(64) +: (64)]),
.tmu_tl_valid(tmu_tl_valid[tile*(1) +: (1)]),
.tmu_tl_ready(tmu_tl_ready[tile*(1) +: (1)]),
.tmu_tl_addr(tmu_tl_addr[tile*(64) +: (64)]),
.tmu_tl_drop(tmu_tl_drop[tile*(1) +: (1)]),
.perf_bypass(perf_bypass[tile*(32) +: (32)]),
.perf_evict(perf_evict[tile*(32) +: (32)]),
.perf_dead_evict(perf_dead_evict[tile*(32) +: (32)]),
.perf_gear(perf_gear[tile*(4) +: (4)]),
.perf_hit(perf_hit[tile*(32) +: (32)]),
.perf_miss(perf_miss[tile*(32) +: (32)]),
.perf_merge(perf_merge[tile*(32) +: (32)]),
.perf_mshr_full(perf_mshr_full[tile*(32) +: (32)]),
.perf_target_full(perf_target_full[tile*(32) +: (32)]),
.perf_wb(perf_wb[tile*(32) +: (32)]),
.perf_hit_past_block(perf_hit_past_block[tile*(32) +: (32)])
 );
end endgenerate
endmodule
