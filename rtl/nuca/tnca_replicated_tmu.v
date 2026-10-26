// One logical TMU, replicated state at each slice. No added visibility cycle.
// Every copy sees the identical global write/TLW stream and deterministic arbiter.
module tnca_replicated_tmu #(
 parameter N_SLICE=3,SRC_W=4,N_SRC=3,TENSORS=8,T_W=3,LIVE=256,LIVE_W=8,
 parameter DEAD=16,DEAD_W=4,ID_W=16,REGISTERED_RANGES=1,TLW_PIPE_STAGES=0,REGISTER_WRITES=0
)(
 input wire clk,rst_n,
 input wire[N_SLICE-1:0] mw_valid,output wire[N_SLICE-1:0] mw_ready,
 input wire[N_SLICE*12-1:0] mw_off,input wire[N_SLICE*64-1:0] mw_data,
 input wire[N_SLICE*SRC_W-1:0] mw_src,
 input wire[N_SLICE*12-1:0] mr_off,output wire[N_SLICE*64-1:0] mr_data,
 input wire[N_SLICE-1:0] tl_valid,output wire[N_SLICE-1:0] tl_ready,
 input wire[N_SLICE*64-1:0] tl_addr,input wire[N_SLICE-1:0] tl_drop,
 output wire[N_SLICE*8-1:0] policy,
 output wire[N_SLICE*4-1:0] b_bits,gear_init,
 output wire[N_SLICE*6-1:0] d_lsb,
 output wire[N_SLICE*16-1:0] gear_window,gear_ub,gear_lb,
 output wire[N_SLICE-1:0] owner_epoch,
 output wire[N_SLICE*TENSORS-1:0] t_valid,t_bypass,
 output wire[N_SLICE*TENSORS*64-1:0] t_base,t_size,
 output wire[N_SLICE*TENSORS*65-1:0] t_end,
 output wire[N_SLICE*TENSORS*6-1:0] t_lsb,
 output wire[N_SLICE*DEAD-1:0] dead_valid,
 output wire[N_SLICE*DEAD*ID_W-1:0] dead_id
);
 genvar local_id;
 generate for(local_id=0;local_id<N_SLICE;local_id=local_id+1)begin:g_replica
  wire[N_SLICE-1:0] write_ready,notice_ready;
  wire[N_SLICE*64-1:0] read_data;
  assign mw_ready[local_id]=write_ready[local_id];
  assign tl_ready[local_id]=notice_ready[local_id];
  assign mr_data[local_id*64+:64]=read_data[local_id*64+:64];
  // Preserve state identity only. Combinational logic remains optimizable.
  (* KEEP_HIERARCHY = "yes" *) tnca_tmu #(
   .REPLICA_STATE("true"),.REGISTER_WRITES(REGISTER_WRITES),.TLW_PIPE_STAGES(TLW_PIPE_STAGES),.N_SLICE(N_SLICE),.N_SRC(N_SRC),.SRC_W(SRC_W),
   .TENSORS(TENSORS),.T_W(T_W),.LIVE(LIVE),.LIVE_W(LIVE_W),
   .DEAD(DEAD),.DEAD_W(DEAD_W),.ID_W(ID_W),.REGISTERED_RANGES(REGISTERED_RANGES)) local_tmu(
   .clk(clk),.rst_n(rst_n),.mw_valid(mw_valid),.mw_ready(write_ready),
   .mw_off(mw_off),.mw_data(mw_data),.mw_src(mw_src),.mr_off(mr_off),.mr_data(read_data),
   .tl_valid(tl_valid),.tl_ready(notice_ready),.tl_addr(tl_addr),.tl_drop(tl_drop),
   .policy(policy[local_id*8+:8]),.b_bits(b_bits[local_id*4+:4]),
   .d_lsb(d_lsb[local_id*6+:6]),.gear_init(gear_init[local_id*4+:4]),
   .gear_window(gear_window[local_id*16+:16]),.gear_ub(gear_ub[local_id*16+:16]),
   .gear_lb(gear_lb[local_id*16+:16]),.owner_epoch(owner_epoch[local_id]),
   .t_valid(t_valid[local_id*TENSORS+:TENSORS]),
   .t_base(t_base[local_id*TENSORS*64+:TENSORS*64]),
   .t_size(t_size[local_id*TENSORS*64+:TENSORS*64]),
   .t_end(t_end[local_id*TENSORS*65+:TENSORS*65]),
   .t_bypass(t_bypass[local_id*TENSORS+:TENSORS]),
   .t_lsb(t_lsb[local_id*TENSORS*6+:TENSORS*6]),
   .dead_valid(dead_valid[local_id*DEAD+:DEAD]),.dead_id(dead_id[local_id*DEAD*ID_W+:DEAD*ID_W]));
 end endgenerate
endmodule
