// Version-2 register binding for the persistent physical-grid shell.
// Legacy register offsets remain recognizable; identity is deliberately
// different and opcodes are decoded in full, never truncated into another op.
// Host word accesses are coherent. Optional opcode 18 hands a byte range to
// DMA after cache writeback/invalidation and physical memory drain.
// One serialized request at a time; a busy/stale doorbell is dropped and must
// be explicitly rung again. Descriptor fields are snapshotted on admission.
// Memory routing/admission controls and the HBM aperture are quasi-static:
// they change only while fully idle, and consumers keep registered local
// copies in their own SLR. After any such change, admissions are held for
// SETTLE cycles (busy reads 1) so no request can meet a stale copy.
`default_nettype none
module tnca_multicore_mmio #(
 parameter N=3,PORTS=3,NUM_RCTX=4,PHYS_W=8,CODE_WORDS=16384,
 parameter MEMORY_CONTROLS=0,AXI_OUT_N=8,HBM_STRIPED=0,
 parameter MEM_BASE=64'h20000,REG_ADDR_W=8,CACHE_MAINTENANCE=0
)(
 input wire clk,rst_n,input wire [REG_ADDR_W-1:0] reg_addr,
 input wire [63:0] reg_wdata,input wire reg_we,reg_re,
 output reg [63:0] reg_rdata,
 output reg cmd_valid,input wire cmd_ready,
 output reg [63:0] cmd_id,cmd_pc,cmd_arg_base,cmd_arg_stride,cmd_deadline,
 output reg [31:0] cmd_gx,cmd_gy,cmd_gz,
 output reg [1:0] cmd_group_log2,
 output reg [N-1:0] cmd_cores,output reg [NUM_RCTX-1:0] cmd_roles,
 output reg [3:0] cmd_slice,
 input wire done_valid,output wire done_ready,input wire [3:0] done_status,
 input wire [63:0] done_id,done_cycles,
 input wire [N*32-1:0] issued,completed,
 input wire [31:0] peak_active,input wire [N*64-1:0] active_cycles,
 output wire host_valid,input wire host_ready,output reg host_we,
 output reg [63:0] host_addr,host_wdata,output reg [7:0] host_be,
 input wire host_rsp_valid,input wire [63:0] host_rsp_data,input wire [3:0] host_rsp_status,
 input wire all_idle,input wire [PORTS-1:0] axi_fault,
 output wire cache_maint_valid,input wire cache_maint_ready,cache_maint_done,
 output reg [63:0] cache_maint_base,cache_maint_size,
 input wire [PORTS*64-1:0] memory_reads,memory_writes,
 output reg [7:0] memory_active_ports,memory_outstanding,
 output reg [63:0] axi_base,axi_size
);
 localparam MAGIC=64'h00020000544e4341,CODE_BYTES=CODE_WORDS*8;
 reg [63:0] req_seq,req_op,a0,a1,a2,a3,a4,a5,deadline;
 reg [63:0] served,rsp_seq,rsp_status,rsp_r0,rsp_r1;
 reg [63:0] window_addr,window_data;
 reg [7:0] window_be;
 reg [2:0] window_status;
 reg [1:0] state,host_state;
 localparam IDLE=0,ISSUE=1,WAIT=2;
 reg [63:0] device_clock,launch_start,launch_end,launch_sequence;
 reg maintenance_cmd,maintenance_complete;
 // Longer than the one-cycle copies in tnca_memory_mode and tnca_line_axi.
 localparam SETTLE=2;
 reg [1:0] settle;
 wire admit_idle=all_idle && settle==0;
 reg [63:0] maintenance_sequence;
 wire [63:0] memory_end=MEM_BASE+axi_size;
 wire memory_valid=memory_end>=MEM_BASE;
 wire range_ok=a0[2:0]==0 && a1!=0 && memory_valid &&
  a1<=memory_end/8 && a0/8<=memory_end/8-a1;
 wire host_address_ok=window_addr[2:0]==0 &&
  ((window_addr<CODE_BYTES) ||
   (memory_valid && axi_size>=8 && window_addr>=MEM_BASE && window_addr<=memory_end-8));
 wire flags_ok=a2[63:32]==0 && (a2 & 64'h0ffffcf0)==0 && a2[9:8]!=3 &&
  a4[63:32]!=0 && (a4[63:32]>>N)==0 && ((a2 & 64'hf)>>NUM_RCTX)==0;
 assign host_valid=host_state==ISSUE;
 assign done_ready=state==WAIT && !maintenance_cmd;
 assign cache_maint_valid=state==ISSUE && maintenance_cmd;
 always @*begin
  reg_rdata=0;
  case(reg_addr)
   8'h00:reg_rdata=req_seq;8'h08:reg_rdata=req_op;
   8'h10:reg_rdata=a0;8'h18:reg_rdata=a1;8'h20:reg_rdata=a2;
   8'h28:reg_rdata=a3;8'h30:reg_rdata=a4;8'h38:reg_rdata=a5;
   8'h40:reg_rdata=deadline;
   8'h48:reg_rdata=(state!=IDLE || host_state!=IDLE || !admit_idle);
   8'h50:reg_rdata=rsp_seq;8'h58:reg_rdata=rsp_status;
   8'h60:reg_rdata=rsp_r0;8'h68:reg_rdata=rsp_r1;
   8'h70:reg_rdata=window_addr;8'h78:reg_rdata=window_data;
   8'h80:reg_rdata=window_be;8'h90:reg_rdata=window_status;
   8'h98:reg_rdata=MAGIC;8'ha0:reg_rdata=served;
   8'ha8,9'h100,9'h108:reg_rdata=axi_base;8'hb0:reg_rdata=axi_size;
   8'hb8:reg_rdata=((MEM_BASE>>12)<<32) | 64'd2 | ((|axi_fault)?64'd1:64'd0) |
                  (CACHE_MAINTENANCE!=0 ? 64'd16 : 64'd0) |
                  (MEMORY_CONTROLS!=0 ? 64'd32 : 64'd0) |
                  (HBM_STRIPED!=0 ? 64'd64 : 64'd0);
   // R_AXI_STAT 63:32 is LOCAL_LIMIT in 4 KiB pages, as in v1.
   9'h110:if(MEMORY_CONTROLS)reg_rdata={48'd0,memory_active_ports,memory_outstanding};
   9'h120:if(HBM_STRIPED)reg_rdata=64'h5234315300010c05;
   9'h118:if(MEMORY_CONTROLS)reg_rdata=64'h5234304d00010000 | (PORTS<<8) | AXI_OUT_N;
   8'he0:reg_rdata=1;8'he8:reg_rdata=launch_start;
   8'hf0:reg_rdata=launch_end;8'hf8:reg_rdata=launch_sequence;
   default:reg_rdata=0;
  endcase
 end
 generate
  if(HBM_STRIPED && REG_ADDR_W<9)begin:g_bad_striped_registers
   tnca_invalid_striped_register_geometry invalid_configuration();
  end
  if(MEMORY_CONTROLS && (REG_ADDR_W<9 || AXI_OUT_N<1 || AXI_OUT_N>255 || PORTS>255))begin:g_bad_memory_controls
   tnca_invalid_memory_control_geometry invalid_configuration();
  end
  if(N<1 || N>16 || PORTS<1 || PORTS>N)begin:g_bad_geometry
   tnca_invalid_mmio_grid_geometry invalid_configuration();
  end
  if(MEM_BASE<CODE_BYTES || MEM_BASE%4096!=0)begin:g_bad_memory
   tnca_invalid_mmio_memory_base invalid_configuration();
  end
 endgenerate
 always @(posedge clk)begin
  if(!rst_n)begin
   req_seq<=0;req_op<=0;a0<=0;a1<=0;a2<=0;a3<=0;a4<=0;a5<=0;deadline<=0;
   served<=0;rsp_seq<=0;rsp_status<=0;rsp_r0<=0;rsp_r1<=0;
   window_addr<=0;window_data<=0;window_be<=8'hff;window_status<=0;
   host_state<=IDLE;host_we<=0;host_addr<=0;host_wdata<=0;host_be<=8'hff;
   state<=IDLE;cmd_valid<=0;cmd_id<=0;cmd_pc<=0;cmd_arg_base<=0;cmd_arg_stride<=0;
   cmd_deadline<=0;cmd_gx<=0;cmd_gy<=0;cmd_gz<=0;cmd_group_log2<=0;
   cmd_cores<=0;cmd_roles<=0;cmd_slice<=0;
   memory_active_ports<=PORTS;memory_outstanding<=AXI_OUT_N;
   axi_base<=0;axi_size<=0;device_clock<=0;launch_start<=0;launch_end<=0;launch_sequence<=0;settle<=SETTLE;
   maintenance_cmd<=0;maintenance_complete<=0;maintenance_sequence<=0;cache_maint_base<=0;cache_maint_size<=0;
  end else begin
   device_clock<=device_clock+1;
   if(settle!=0)settle<=settle-1;
   if(reg_we)case(reg_addr)
    9'h110:if(MEMORY_CONTROLS && state==IDLE && host_state==IDLE && all_idle &&
      reg_wdata[63:16]==0 && (reg_wdata[15:8]==1 || reg_wdata[15:8]==PORTS) &&
      reg_wdata[7:0]>=1 && reg_wdata[7:0]<=AXI_OUT_N)begin
     memory_active_ports<=reg_wdata[15:8];memory_outstanding<=reg_wdata[7:0];settle<=SETTLE;
    end
    8'h00:req_seq<=reg_wdata;8'h08:req_op<=reg_wdata;
    8'h10:a0<=reg_wdata;8'h18:a1<=reg_wdata;8'h20:a2<=reg_wdata;
    8'h28:a3<=reg_wdata;8'h30:a4<=reg_wdata;8'h38:a5<=reg_wdata;
    8'h40:deadline<=reg_wdata;
    8'h70:window_addr<=reg_wdata;8'h78:window_data<=reg_wdata;8'h80:window_be<=reg_wdata[7:0];
    8'ha8,9'h100,9'h108:if(state==IDLE && host_state==IDLE && all_idle)begin axi_base<=reg_wdata;settle<=SETTLE;end
    8'hb0:if(state==IDLE && host_state==IDLE && all_idle)begin axi_size<=reg_wdata;settle<=SETTLE;end
    8'h88:if(reg_wdata[1] && host_state==IDLE)begin
     if(state!=IDLE || !admit_idle || !host_address_ok)window_status<=3'b100;
     else begin
      host_state<=ISSUE;window_status<=3'b001;
      host_we<=reg_wdata[0];host_addr<=window_addr;host_wdata<=window_data;host_be<=window_be;
     end
    end
    8'h48:if(reg_wdata[0] && state==IDLE && host_state==IDLE && admit_idle && req_seq>served)begin
     served<=req_seq;rsp_seq<=req_seq;rsp_status<=0;rsp_r0<=0;rsp_r1<=0;
     case(req_op)
      64'd1,64'd2:if(!range_ok)rsp_status<=3;
      64'd4:begin end // fully drained idle is required to admit this doorbell
      64'd5:begin end // no destructive board shutdown
      64'd6:rsp_r0<=req_seq;
      64'd8:rsp_status<=3; // legacy LAUNCH32 is not this binding's command
      64'd10:begin rsp_r0<=N;rsp_r1<=16777216;end
      64'd12:begin
       if(a0>=N)rsp_status<=3;
       else begin rsp_r0<=issued[a0*32+:32];rsp_r1<=completed[a0*32+:32];end
      end
      64'd13:begin
       if(a0>=N)rsp_status<=3;
       else begin rsp_r0<=peak_active;rsp_r1<=active_cycles[a0*64+:64];end
      end
      64'd14:begin rsp_r0<=1;rsp_r1<=NUM_RCTX>=4?4:(NUM_RCTX>=2?2:1);end
      64'd16:begin rsp_r0<=PORTS;rsp_r1<=1;end
      64'd17:begin
       if(a0>=PORTS)rsp_status<=3;
       else begin rsp_r0<=memory_reads[a0*64+:64];rsp_r1<=memory_writes[a0*64+:64];end
      end
      64'd18:begin
       if(CACHE_MAINTENANCE==0 || !memory_valid || a0<MEM_BASE || a0>memory_end || a1>memory_end-a0)
        rsp_status<=3;
       else if(|axi_fault)rsp_status<=1;
       else begin
        rsp_seq<=0;maintenance_cmd<=1;maintenance_complete<=0;maintenance_sequence<=req_seq;
        cache_maint_base<=a0;cache_maint_size<=a1;state<=ISSUE;
       end
      end
      64'd11:begin
       if(!flags_ok)rsp_status<=3;
       else begin
        // Publish the response sequence only after matching global completion.
        rsp_seq<=0;cmd_valid<=1;state<=ISSUE;
        cmd_id<=req_seq;cmd_pc<=a0;cmd_arg_base<=a1;cmd_arg_stride<=a5;cmd_deadline<=deadline;
        cmd_gx<=a3[31:0];cmd_gy<=a3[63:32];cmd_gz<=a4[31:0];cmd_cores<=a4[63:32];
        cmd_roles<=a2[NUM_RCTX-1:0];cmd_slice<={a2[28],a2[31:29]};cmd_group_log2<=a2[9:8];
        launch_start<=device_clock;launch_sequence<=0;
       end
      end
      default:rsp_status<=3;
     endcase
    end
    default:begin end
   endcase
   if(host_state==ISSUE && host_ready)host_state<=WAIT;
   if(host_state==WAIT && host_rsp_valid)begin
    host_state<=IDLE;window_status<=host_rsp_status==0?3'b010:3'b100;
    if(host_rsp_status==0)begin window_data<=host_rsp_data;window_addr<=host_addr+8;end
   end
   if(state==ISSUE && (maintenance_cmd ? cache_maint_ready : cmd_ready))begin cmd_valid<=0;state<=WAIT;end
   if(state==WAIT && maintenance_cmd && cache_maint_done)maintenance_complete<=1;
   if(state==WAIT && maintenance_cmd && (maintenance_complete || cache_maint_done) && all_idle)begin
    state<=IDLE;rsp_seq<=maintenance_sequence;rsp_status<=(|axi_fault)?4'd1:4'd0;
    rsp_r0<=0;rsp_r1<=0;maintenance_cmd<=0;
   end
   if(state==WAIT && !maintenance_cmd && done_valid)begin
    state<=IDLE;rsp_seq<=cmd_id;rsp_status<=done_id==cmd_id?done_status:4'd1;
    rsp_r0<=done_cycles[31:0];rsp_r1<=done_cycles[63:32];
    launch_end<=device_clock;launch_sequence<=cmd_id;
   end
  end
 end
endmodule
`default_nettype wire
