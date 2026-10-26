// Observation only: a second simulation top beside tb/device/tnca_multicore_device.v.
// Never synthesize, never list in a product file set. It reads one core of the
// elaborated device through hierarchical references and writes the core trace
// defined in README.md (TNCA_CORE_TRACE 1). Product RTL is untouched: the
// trace hooks live here, in the evaluator's bench, not in rtl/.
//
// Compile with the device bench and name both tops:
//   iverilog ... -s tnca_multicore_device -s tnca_core_trace \
//            -DTNCA_TRACE_CORE=0 tools/performance_harness/align/tnca_core_trace.v
// Run with +core_trace=<file> (TNCA_DEVICE_PLUSARGS carries it through the runtime).
`timescale 1ns/1ps
`ifndef TNCA_TRACE_CORE
`define TNCA_TRACE_CORE 0
`endif
`define TNCA_TRACED_CORE tnca_multicore_device.dut.u_system.g_tile[`TNCA_TRACE_CORE].u_tile.u_core

module tnca_core_trace;
    integer file = 0;
    integer cycle = 0;        // posedges with rst_n high; the first such edge is 0
    integer launches = 0;
    reg     idle_q = 1'b1;    // core_idle at the previous edge
    reg     in_launch = 1'b0;
    reg [4095:0] path;

    initial begin
        if (!$value$plusargs("core_trace=%s", path)) $fatal(1, "missing +core_trace");
        file = $fopen(path, "w");
        if (!file) $fatal(1, "cannot create core trace");
        $fdisplay(file, "TNCA_CORE_TRACE 1 core=%0d", `TNCA_TRACE_CORE);
    end

    // Everything below samples the pre-edge value, which is the value the
    // core's own clocked logic sees at this edge: `x_valid` here is the
    // instruction whose retire effects the core applies at this edge, and a
    // request with valid && ready is the one the memory accepts at this edge.
    always @(posedge `TNCA_TRACED_CORE.clk) begin
        if (`TNCA_TRACED_CORE.rst_n) begin
            if (`TNCA_TRACED_CORE.launch_valid) begin
                launches = launches + 1;
                in_launch = 1'b1;
                $fdisplay(file, "L %0d %016h %0d", cycle,
                          `TNCA_TRACED_CORE.launch_pc, `TNCA_TRACED_CORE.launch_role_mask);
            end
            if (`TNCA_TRACED_CORE.x_valid)
                $fdisplay(file, "I %0d %0d %016h", cycle,
                          `TNCA_TRACED_CORE.x_rctx, `TNCA_TRACED_CORE.x_pc);
            if (`TNCA_TRACED_CORE.mem_req_valid && `TNCA_TRACED_CORE.mem_req_ready)
                $fdisplay(file, "Q %0d %0d %016h %02h %0d %0d %016h", cycle,
                          `TNCA_TRACED_CORE.mem_req_we, `TNCA_TRACED_CORE.mem_req_addr,
                          `TNCA_TRACED_CORE.mem_req_be, `TNCA_TRACED_CORE.mem_req_tag,
                          `TNCA_TRACED_CORE.mem_req_amo, `TNCA_TRACED_CORE.mem_req_wdata);
            if (`TNCA_TRACED_CORE.mem_rsp_valid)
                $fdisplay(file, "P %0d %0d %016h", cycle,
                          `TNCA_TRACED_CORE.mem_rsp_tag, `TNCA_TRACED_CORE.mem_rsp_rdata);
            // The launch ends when the core goes idle again. `core_idle` is
            // low from the launch edge on, so a rising edge after a launch is
            // its completion (ISA sec 7: every role exited, every engine done).
            if (in_launch && `TNCA_TRACED_CORE.core_idle && !idle_q) begin
                in_launch = 1'b0;
                $fdisplay(file, "E %0d", cycle);
            end
            idle_q = `TNCA_TRACED_CORE.core_idle;
            cycle = cycle + 1;
        end
    end

    final begin
        if (file) begin
            $fdisplay(file, "END %0d %0d", launches, cycle);
            $fclose(file);
        end
    end
endmodule
