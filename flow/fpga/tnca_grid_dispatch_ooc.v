// Out-of-context timing harness for tnca_grid_dispatch (R41 F06), physical
// screening only. Every dispatcher input is driven from a register and every
// output lands in one, as in tnca_top/tnca_multicore_mmio, where the launch
// fields are MMIO registers and the commands enter registered elastic links.
// Without this the unit's ports are unconstrained in
// flow/synth/vivado_ooc_timing.tcl and the routed census path (launch
// register -> DSP product -> remaining/done) would not be timed at all.
`default_nettype none
module tnca_grid_dispatch_ooc #(
    parameter CORE_N=3,
    parameter CORE_W=(CORE_N>1 ? $clog2(CORE_N) : 1),
    parameter MAX_PROGRAMS=16777216
)(
    input wire clk, rst_n,
    input wire launch_valid, abort_valid, done_ready,
    input wire [3:0] abort_status,
    input wire [63:0] launch_id, launch_pc, launch_arg_base, launch_arg_stride,
    input wire [31:0] launch_gx, launch_gy, launch_gz,
    input wire [CORE_N-1:0] launch_cores, cmd_ready, core_done, core_abort,
    input wire [1:0] launch_group_log2,
    input wire [CORE_N*4-1:0] core_status,
    output reg [1023:0] observed
);
    reg lv, av, dr; reg [3:0] as;
    reg [63:0] lid, lpc, lab, las; reg [31:0] gx, gy, gz;
    reg [CORE_N-1:0] lc, cr, cd, ca; reg [1:0] lg; reg [CORE_N*4-1:0] cs;
    always @(posedge clk) begin
        lv<=launch_valid; av<=abort_valid; dr<=done_ready; as<=abort_status;
        lid<=launch_id; lpc<=launch_pc; lab<=launch_arg_base; las<=launch_arg_stride;
        gx<=launch_gx; gy<=launch_gy; gz<=launch_gz;
        lc<=launch_cores; cr<=cmd_ready; cd<=core_done; ca<=core_abort; lg<=launch_group_log2; cs<=core_status;
    end
    wire launch_ready, done_valid;
    wire [CORE_N-1:0] cmd_valid, cmd_cancel, core_busy;
    wire [63:0] cmd_id, cmd_pc, done_id, done_cycles;
    wire [CORE_N*64-1:0] cmd_arg_base, active_cycles;
    wire [CORE_N*32-1:0] cmd_x, cmd_y, cmd_z, issued, completed;
    wire [31:0] cmd_nx, cmd_ny, cmd_nz, peak_active;
    wire [3:0] done_status;
    wire [CORE_W-1:0] fault_core;
    tnca_grid_dispatch #(.CORE_N(CORE_N),.CORE_W(CORE_W),.MAX_PROGRAMS(MAX_PROGRAMS)) u_dispatch (
        .clk(clk),.rst_n(rst_n),.launch_valid(lv),.abort_valid(av),.abort_status(as),.launch_ready(launch_ready),
        .launch_id(lid),.launch_pc(lpc),.launch_arg_base(lab),.launch_arg_stride(las),
        .launch_gx(gx),.launch_gy(gy),.launch_gz(gz),.launch_cores(lc),.launch_group_log2(lg),
        .cmd_valid(cmd_valid),.cmd_ready(cr),.cmd_id(cmd_id),.cmd_pc(cmd_pc),.cmd_arg_base(cmd_arg_base),
        .cmd_x(cmd_x),.cmd_y(cmd_y),.cmd_z(cmd_z),.cmd_nx(cmd_nx),.cmd_ny(cmd_ny),.cmd_nz(cmd_nz),
        .core_done(cd),.core_abort(ca),.core_status(cs),.cmd_cancel(cmd_cancel),
        .done_valid(done_valid),.done_ready(dr),.done_status(done_status),.done_id(done_id),.done_cycles(done_cycles),
        .fault_core(fault_core),.core_busy(core_busy),.issued(issued),.completed(completed),
        .peak_active(peak_active),.active_cycles(active_cycles));
    // Fold every output into registers so no output logic is trimmed.
    wire [4095:0] all = {launch_ready, done_valid, cmd_valid, cmd_cancel, core_busy, cmd_id, cmd_pc,
                         done_id, done_cycles, cmd_arg_base, active_cycles, cmd_x, cmd_y, cmd_z,
                         issued, completed, cmd_nx, cmd_ny, cmd_nz, peak_active, done_status, fault_core};
    integer k;
    always @(posedge clk) for (k=0; k<1024; k=k+1) observed[k] <= all[k] ^ all[k+1024] ^ all[k+2048] ^ all[k+3072];
endmodule
`default_nettype wire
