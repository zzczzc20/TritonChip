// Observation only: a second simulation top beside tb/device/tnca_multicore_device.v.
// Never synthesize, never list in a product file set. It reads the elaborated
// device through hierarchical references and writes the traces defined in
// README.md: one core trace per core (TNCA_CORE_TRACE 1) and one uncore
// trace (TNCA_UNCORE_TRACE 1). Product RTL is untouched: the hooks live
// here, in the evaluator's bench, not in rtl/.
//
// Compile with the device bench and name both tops:
//   iverilog ... -s tnca_multicore_device -s tnca_core_trace \
//            -DTNCA_TRACE_CORES=3 tools/performance_harness/align/tnca_core_trace.v
// Run with +core_trace=<file> for core 0, +core_trace<c>=<file> for core c>0 and
// optionally +uncore_trace=<file>. TNCA_DEVICE_PLUSARGS carries them through
// the runtime.
`timescale 1ns/1ps
`ifndef TNCA_TRACE_CORES
`define TNCA_TRACE_CORES 3
`endif
`define TNCA_DEV tnca_multicore_device
`define TNCA_TILE(c) tnca_multicore_device.dut.u_system.g_tile[c].u_tile

module tnca_core_trace;
    integer cycle = 0;        // posedges with rst_n high; the first such edge is 0
    reg [4095:0] path;
    reg [4095:0] upath;
    integer ufile = 0;
    integer launches_all = 0;
    reg [`TNCA_TRACE_CORES-1:0] launch_bits = 0;   // a launch in flight on core c: the uncore records are kept only then (host traffic is not a launch's)
    wire clk   = `TNCA_TILE(0).u_core.clk;
    wire rst_n = `TNCA_TILE(0).u_core.rst_n;

    initial begin
        if (!$value$plusargs("core_trace=%s", path)) $fatal(1, "missing +core_trace");
        if ($value$plusargs("uncore_trace=%s", upath)) begin
            ufile = $fopen(upath, "w");
            if (!ufile) $fatal(1, "cannot create uncore trace");
            $fdisplay(ufile, "TNCA_UNCORE_TRACE 1 cores=%0d", `TNCA_TRACE_CORES);
        end
    end

    // Everything below samples the pre-edge value, which is the value the
    // core's own clocked logic sees at this edge.
    genvar c;
    generate for (c = 0; c < `TNCA_TRACE_CORES; c = c + 1) begin : g_core
        integer file = 0;
        integer launches = 0;
        reg     idle_q = 1'b1;
        reg     in_launch = 1'b0;
        reg [4095:0] cpath;
        reg got;
        initial begin
            // Each core reads its own plusarg (initial blocks run in no fixed
            // order, so nothing here depends on the top's `path`).
            got = 1'b0;
            case (c)
            0: got = $value$plusargs("core_trace=%s", cpath);
            1: got = $value$plusargs("core_trace1=%s", cpath);
            2: got = $value$plusargs("core_trace2=%s", cpath);
            3: got = $value$plusargs("core_trace3=%s", cpath);
            default: ;
            endcase
            if (!got) $fatal(1, "missing +core_trace<c> for core %0d", c);
            file = $fopen(cpath, "w");
            if (!file) $fatal(1, "cannot create core trace");
            $fdisplay(file, "TNCA_CORE_TRACE 1 core=%0d", c);
        end
        always @(posedge clk) begin
            if (rst_n) begin
                if (`TNCA_TILE(c).u_core.launch_valid) begin
                    launches = launches + 1;
                    in_launch = 1'b1;
                    launch_bits[c] = 1'b1;
                    $fdisplay(file, "L %0d %016h %0d", cycle,
                              `TNCA_TILE(c).u_core.launch_pc, `TNCA_TILE(c).u_core.launch_role_mask);
                    if (ufile) $fdisplay(ufile, "L %0d %0d %016h %0d", cycle, c,
                                         `TNCA_TILE(c).u_core.launch_pc, `TNCA_TILE(c).u_core.launch_role_mask);
                end
                if (`TNCA_TILE(c).u_core.x_valid)
                    $fdisplay(file, "I %0d %0d %016h", cycle,
                              `TNCA_TILE(c).u_core.x_rctx, `TNCA_TILE(c).u_core.x_pc);
                if (`TNCA_TILE(c).u_core.mem_req_valid && `TNCA_TILE(c).u_core.mem_req_ready)
                    $fdisplay(file, "Q %0d %0d %016h %02h %0d %0d %016h", cycle,
                              `TNCA_TILE(c).u_core.mem_req_we, `TNCA_TILE(c).u_core.mem_req_addr,
                              `TNCA_TILE(c).u_core.mem_req_be, `TNCA_TILE(c).u_core.mem_req_tag,
                              `TNCA_TILE(c).u_core.mem_req_amo, `TNCA_TILE(c).u_core.mem_req_wdata);
                if (`TNCA_TILE(c).u_core.mem_rsp_valid)
                    $fdisplay(file, "P %0d %0d %016h", cycle,
                              `TNCA_TILE(c).u_core.mem_rsp_tag, `TNCA_TILE(c).u_core.mem_rsp_rdata);
                if (in_launch && `TNCA_TILE(c).u_core.core_idle && !idle_q) begin
                    in_launch = 1'b0;
                    launch_bits[c] = 1'b0;
                    $fdisplay(file, "E %0d", cycle);
                end
                idle_q = `TNCA_TILE(c).u_core.core_idle;
                // the uncore: the home slice's port and the tile's memory port (while a launch runs)
                if (ufile && (|launch_bits)) begin
                    if (`TNCA_TILE(c).s_req_valid && `TNCA_TILE(c).s_req_ready)
                        $fdisplay(ufile, "S %0d %0d %0d %0d %0d %016h", cycle, c,
                                  `TNCA_TILE(c).s_req_src, `TNCA_TILE(c).s_req_tag,
                                  `TNCA_TILE(c).s_req_we, `TNCA_TILE(c).s_req_addr);
                    if (`TNCA_TILE(c).s_rsp_valid && `TNCA_TILE(c).s_rsp_ready)
                        $fdisplay(ufile, "R %0d %0d %0d %0d", cycle, c,
                                  `TNCA_TILE(c).s_rsp_src, `TNCA_TILE(c).s_rsp_tag);
                    if (`TNCA_TILE(c).mem_req_valid && `TNCA_TILE(c).mem_req_ready)
                        $fdisplay(ufile, "M %0d %0d %0d %016h %0d", cycle, c,
                                  `TNCA_TILE(c).mem_req_we, `TNCA_TILE(c).mem_req_addr, `TNCA_TILE(c).mem_req_tag);
                    if (`TNCA_TILE(c).mem_rsp_valid)
                        $fdisplay(ufile, "D %0d %0d %0d", cycle, c, `TNCA_TILE(c).mem_rsp_tag);
                end
            end
        end
        final begin
            if (file) begin
                $fdisplay(file, "END %0d %0d", launches, cycle);
                $fclose(file);
            end
        end
    end endgenerate

    // The device's memory port (PORTS=1): the outlet to the DRAM model, where
    // the validation replays responses.
    always @(posedge clk) begin
        if (rst_n && ufile && (|launch_bits)) begin
            if (`TNCA_DEV.mr && `TNCA_DEV.mready)
                $fdisplay(ufile, "X %0d %0d %016h %0d %0d", cycle, `TNCA_DEV.mwe, `TNCA_DEV.ma, `TNCA_DEV.ms, `TNCA_DEV.mt);
            if (`TNCA_DEV.mrv)
                $fdisplay(ufile, "Y %0d %0d %0d", cycle, `TNCA_DEV.mrs, `TNCA_DEV.mrt);
        end
    end
    // The cycle counter advances between edges, so every posedge block above
    // (they run in no fixed order) sees the same value at an edge.
    always @(negedge clk) if (rst_n) cycle = cycle + 1;
    final begin
        if (ufile) begin
            $fdisplay(ufile, "END %0d", cycle);
            $fclose(ufile);
        end
    end
endmodule
