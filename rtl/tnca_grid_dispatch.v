// One logical grid, one outstanding command per physical core. Coordinates
// advance only when reserved in a bounded per-core queue; no storage proportional to the grid size.
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
    wire [63:0] groups_x=({32'd0,launch_gx}+requested_width-1) >> (requested_width==4 ? 2 : requested_width==2 ? 1 : 0);
    wire [95:0] groups_total=groups_x*{64'd0,launch_gy}*{64'd0,launch_gz};
    integer j, choice, candidate;
    reg found;
    wire unexpected_done = |(core_done & ~core_busy);
    reg fault_now;
    integer f;
    always @* begin
        fault_now=(abort_valid && running) | unexpected_done | (|(core_abort & core_busy));
        for(f=0;f<CORE_N;f=f+1)
            if(core_done[f] && core_busy[f] && core_status[f*4+:4]!=0) fault_now=1;
    end
    wire [CORE_N-1:0] accepted = cmd_valid & cmd_ready;
    reg [CORE_N-1:0] enqueue;
    wire [CORE_N-1:0] pending_after = (pending & ~accepted) | enqueue;
    wire [CORE_N-1:0] busy_after = (core_busy & ~core_done) | accepted;
    wire [95:0] total = {64'd0,launch_gx} * {64'd0,launch_gy} * {64'd0,launch_gz};
    wire [95:0] last_args = {32'd0,launch_arg_base} + {32'd0,launch_arg_stride} * (CORE_N-1);
    assign launch_ready = !running && !done_valid;
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
            nx<=0;ny<=0;nz<=0;x<=0;y<=0;z<=0;remaining<=0;rr<=0;
        end else begin
            if(done_valid && done_ready) done_valid<=0;
            if(launch_valid && launch_ready) begin
                id_q<=launch_id;pc_q<=launch_pc;args_q<=launch_arg_base;stride_q<=launch_arg_stride;
                group_width<=requested_width;accepted_width<=0;
                nx<=launch_gx;ny<=launch_gy;nz<=launch_gz;active<=launch_cores;
                x<=0;y<=0;z<=0;rr<=0;core_busy<=0;pending<=0;faulted<=0;fault_core<=0;
                issued<=0;completed<=0;peak_active<=0;active_cycles<=0;done_id<=launch_id;done_cycles<=0;done_status<=0;
                if(total>MAX_PROGRAMS || launch_group_log2 === 2'd3 || last_args[95:64]!=0 ||
                   launch_pc[2:0]!=0 || launch_arg_base[2:0]!=0 || launch_arg_stride[2:0]!=0 ||
                   (total!=0 && launch_cores==0)) begin
                    done_valid<=1;done_status<=3;running<=0;remaining<=0;
                end else if(total==0) begin
                    done_valid<=1;running<=0;remaining<=0;
                end else begin
                    remaining<=groups_total[31:0];running<=1;
                end
            end else if(running) begin
                done_cycles<=done_cycles+1'b1;
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
                if((faulted || fault_now || remaining==0) && busy_after==0 && (faulted || fault_now || pending_after==0)) begin
                    running<=0;done_valid<=1;
                end
            end
        end
    end
endmodule
`default_nettype wire
