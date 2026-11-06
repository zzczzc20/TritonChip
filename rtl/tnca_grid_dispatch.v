// One logical grid, one outstanding command per physical core. Coordinates
// advance only when reserved in a bounded per-core queue; no storage proportional to the grid size.
//
// ADMISSION (R41 F06). A launch is validated by four registered stages after
// the accepting edge, from the captured nx/ny/nz/pc/args/stride, before the
// grid may issue anything:
//   A  saturate every factor to SW = clog2(MAX_PROGRAMS+2) bits (a factor
//      above MAX_PROGRAMS becomes MAX_PROGRAMS+1), round x up to chunks,
//      check zero/alignment/argument range/core mask;
//   B  x*y and chunks(x)*y (SW x SW products);
//   C  (x*y)*z and (chunks*y)*z from the low SW bits, plus "x*y > MAX";
//   D  decide: refuse (status 3), complete an empty grid (status 0), or run
//      with `remaining` loaded from the registered chunk count.
// With every factor nonzero, the saturated product exceeds MAX_PROGRAMS
// exactly when the true 96-bit product does, and is exact otherwise; so
// the decision equals the old single-cycle 96-bit test. No multiplier output
// feeds dispatch state in the cycle it is produced. launch_ready stays low
// through admission (so host access and a second launch wait too), and an
// abort or stray completion during admission is recorded exactly as one in
// the first running cycle would be; a refused or empty grid still reports
// the decision as of the accepting edge (status 3 / 0, fault_core 0,
// done_cycles 0). Cost: four cycles per launch before the first command.
`default_nettype none
module tnca_grid_dispatch #(
    parameter CORE_N=3,
    parameter CORE_W=(CORE_N>1 ? $clog2(CORE_N) : 1),
    parameter MAX_PROGRAMS=16777216
)(
    input wire clk, rst_n,
    input wire launch_valid,
    input wire abort_valid,
    input wire [3:0] abort_status,
    output wire launch_ready,
    input wire [63:0] launch_id, launch_pc, launch_arg_base, launch_arg_stride,
    input wire [31:0] launch_gx, launch_gy, launch_gz,
    input wire [CORE_N-1:0] launch_cores,
    input wire [1:0] launch_group_log2,
    output reg [CORE_N-1:0] cmd_valid,
    input wire [CORE_N-1:0] cmd_ready,
    output wire [63:0] cmd_id, cmd_pc,
    output wire [CORE_N*64-1:0] cmd_arg_base,
    output reg [CORE_N*32-1:0] cmd_x, cmd_y, cmd_z,
    output wire [31:0] cmd_nx, cmd_ny, cmd_nz,
    input wire [CORE_N-1:0] core_done, core_abort,
    input wire [CORE_N*4-1:0] core_status,
    output wire [CORE_N-1:0] cmd_cancel,
    output reg done_valid,
    input wire done_ready,
    output reg [3:0] done_status,
    output reg [63:0] done_id, done_cycles,
    output reg [CORE_W-1:0] fault_core,
    output reg [CORE_N-1:0] core_busy,
    output reg [CORE_N*32-1:0] issued, completed,
    output reg [31:0] peak_active,
    output reg [CORE_N*64-1:0] active_cycles
);
    reg running, faulted;
    reg [CORE_N-1:0] active, pending;
    reg [63:0] id_q, pc_q, args_q, stride_q;
    reg [31:0] nx, ny, nz, x, y, z, remaining;
    reg [CORE_W-1:0] rr;
    reg [2:0] group_width;
    reg [CORE_N*3-1:0] accepted_width;
    wire [2:0] requested_width = launch_group_log2 === 2'd2 ? 3'd4 :
                                 (launch_group_log2 === 2'd1 ? 3'd2 : 3'd1);

    // ---- admission pipeline (see the header) ---------------------------------
    localparam SW = $clog2(MAX_PROGRAMS+2);
    localparam [SW-1:0] SAT = MAX_PROGRAMS+1;
    reg [3:0] adm;                      // one-hot: stage A, B, C, D in flight
    wire sizing = |adm;
    reg bad_group;                      // launch_group_log2 === 3, at the edge
    function [SW-1:0] sat;
        input [32:0] v;
        sat = (v > MAX_PROGRAMS) ? SAT : v[SW-1:0];
    endfunction
    // Every input below is a register written only by the accepting edge,
    // and nothing is accepted again until the decision, so the stages run
    // free and are consistent by stage D.
    wire [32:0] x_round = {1'b0,nx} + group_width - 1'b1;
    wire [32:0] chunks_x = group_width==3'd4 ? x_round >> 2 :
                           (group_width==3'd2 ? x_round >> 1 : x_round);
    wire [95:0] last_args = {32'd0,args_q} + {32'd0,stride_q} * (CORE_N-1);
    wire a_zero_now = nx==0 || ny==0 || nz==0;
    reg [SW-1:0] s_x, s_y, s_z, s_g;    // stage A
    reg a_zero, a_bad;
    reg [2*SW-1:0] p_xy, p_gy;          // stage B
    reg [2*SW-1:0] p_xyz, p_gyz;        // stage C
    reg c_over_xy;
    always @(posedge clk) begin
        s_x<=sat({1'b0,nx}); s_y<=sat({1'b0,ny}); s_z<=sat({1'b0,nz}); s_g<=sat(chunks_x);
        a_zero<=a_zero_now;
        a_bad<=bad_group || last_args[95:64]!=0 || pc_q[2:0]!=0 || args_q[2:0]!=0 ||
               stride_q[2:0]!=0 || (!a_zero_now && active==0);
        p_xy<=s_x*s_y; p_gy<=s_g*s_y;
        c_over_xy<=p_xy>MAX_PROGRAMS;
        p_xyz<=p_xy[SW-1:0]*s_z; p_gyz<=p_gy[SW-1:0]*s_z;
    end
    wire d_refuse = a_bad || (!a_zero && (c_over_xy || p_xyz>MAX_PROGRAMS));

    integer j, choice, candidate;
    reg found;
    wire unexpected_done = |(core_done & ~core_busy);
    reg fault_now;
    integer f;
    always @* begin
        fault_now=(abort_valid && (running || sizing)) | unexpected_done | (|(core_abort & core_busy));
        for(f=0;f<CORE_N;f=f+1)
            if(core_done[f] && core_busy[f] && core_status[f*4+:4]!=0) fault_now=1;
    end
    wire [CORE_N-1:0] accepted = cmd_valid & cmd_ready;
    reg [CORE_N-1:0] enqueue;
    wire [CORE_N-1:0] pending_after = (pending & ~accepted) | enqueue;
    wire [CORE_N-1:0] busy_after = (core_busy & ~core_done) | accepted;
    assign launch_ready = !running && !sizing && !done_valid;
    assign cmd_id=id_q;
    assign cmd_pc=pc_q;
    genvar a;
    generate for(a=0;a<CORE_N;a=a+1) begin: g_args
        assign cmd_arg_base[a*64+:64]=args_q + a * stride_q;
    end endgenerate
    assign cmd_cancel=pending & {CORE_N{faulted || fault_now}};
    assign cmd_nx=nx; assign cmd_ny=ny; assign cmd_nz=nz;

    always @* begin
        choice=0; found=0; enqueue=0;
        cmd_valid=pending & {CORE_N{running && !faulted && !fault_now}};
        for(j=0;j<CORE_N;j=j+1) begin
            candidate=rr+j;
            if(candidate>=CORE_N) candidate=candidate-CORE_N;
            if(!found && active[candidate] && !core_busy[candidate] && !pending[candidate]) begin
                choice=candidate; found=1;
            end
        end
        if(running && remaining!=0 && !faulted && !fault_now && found)
            enqueue[choice]=1'b1;
    end

    integer c;
    reg selected_fault;
    integer active_count;
    always @* begin
        active_count=0;
        for(integer a=0;a<CORE_N;a=a+1) active_count=active_count+core_busy[a];
    end
    always @(posedge clk) begin
        if(!rst_n) begin
            running<=0; faulted<=0; active<=0; core_busy<=0;pending<=0;cmd_x<=0;cmd_y<=0;cmd_z<=0;
            done_valid<=0; done_status<=0; done_id<=0; done_cycles<=0;
            fault_core<=0; issued<=0; completed<=0; peak_active<=0; active_cycles<=0;
            id_q<=0;pc_q<=0;args_q<=0;stride_q<=0;group_width<=1;accepted_width<=0;
            nx<=0;ny<=0;nz<=0;x<=0;y<=0;z<=0;remaining<=0;rr<=0;adm<=0;bad_group<=0;
        end else begin
            if(done_valid && done_ready) done_valid<=0;
            adm<={adm[2:0],1'b0};
            if(launch_valid && launch_ready) begin
                id_q<=launch_id;pc_q<=launch_pc;args_q<=launch_arg_base;stride_q<=launch_arg_stride;
                group_width<=requested_width;accepted_width<=0;bad_group<=launch_group_log2 === 2'd3;
                nx<=launch_gx;ny<=launch_gy;nz<=launch_gz;active<=launch_cores;
                x<=0;y<=0;z<=0;rr<=0;core_busy<=0;pending<=0;faulted<=0;fault_core<=0;
                issued<=0;completed<=0;peak_active<=0;active_cycles<=0;done_id<=launch_id;done_cycles<=0;done_status<=0;
                remaining<=0;running<=0;adm<=4'b0001;
            end else if(running || sizing) begin
                // During admission core_busy, pending and accepted are zero,
                // so only the fault/abort recording below has any effect.
                if(running) done_cycles<=done_cycles+1'b1;
                if(active_count>peak_active) peak_active<=active_count;
                core_busy<=busy_after;
                pending<=faulted || fault_now ? 0 : pending_after;
                selected_fault=0;
                for(c=0;c<CORE_N;c=c+1) begin
                    if(core_busy[c]) active_cycles[c*64+:64]<=active_cycles[c*64+:64]+1'b1;
                    if(accepted[c]) begin
                        accepted_width[c*3+:3]<=nx-cmd_x[c*32+:32]<group_width ? nx-cmd_x[c*32+:32] : group_width;
                        issued[c*32+:32]<=issued[c*32+:32]+(nx-cmd_x[c*32+:32]<group_width ? nx-cmd_x[c*32+:32] : group_width);
                    end
                    if(core_done[c] && core_busy[c]) completed[c*32+:32]<=completed[c*32+:32]+accepted_width[c*3+:3];
                    if(!faulted && !selected_fault && (core_done[c] || (core_abort[c] && core_busy[c])) &&
                       ((core_done[c] && !core_busy[c]) || core_abort[c] || core_status[c*4+:4]!=0)) begin
                        fault_core<=c;selected_fault=1;
                        done_status<=!core_busy[c] || core_status[c*4+:4]==0 ? 1 : core_status[c*4+:4];
                    end
                end
                if(!faulted && !selected_fault && abort_valid) begin
                    done_status<=abort_status==0 ? 4'd1 : abort_status;
                    fault_core<=0;
                end
                if(fault_now) faulted<=1;
                if(|enqueue) begin
                    cmd_x[choice*32+:32]<=x;cmd_y[choice*32+:32]<=y;cmd_z[choice*32+:32]<=z;
                    remaining<=remaining-1'b1;
                    rr<=choice==CORE_N-1 ? 0 : choice+1;
                    if({1'b0,x}+group_width>=nx) begin
                        x<=0;
                        if(y+1==ny) begin y<=0;z<=z+1'b1;end
                        else y<=y+1'b1;
                    end else x<=x+group_width;
                end
                if(running && (faulted || fault_now || remaining==0) && busy_after==0 && (faulted || fault_now || pending_after==0)) begin
                    running<=0;done_valid<=1;
                end
                // Stage D: the decision is the old acceptance-edge decision.
                if(adm[3]) begin
                    if(d_refuse || a_zero) begin
                        done_valid<=1;done_status<=d_refuse ? 4'd3 : 4'd0;fault_core<=0;faulted<=0;
                    end else begin
                        running<=1;remaining<=p_gyz[31:0];
                    end
                end
            end
        end
    end
endmodule
`default_nettype wire
