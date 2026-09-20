//==============================================================================
// tnca_device -- a PERSISTENT single-core device for the R2 runtime
//
// doc/phase2_tasks_and_acceptance.md R2-08 RT-02 asks for thirty-two launches
// in one session with no restart, and R2-11 makes restarting an explicit,
// observable device-loss recovery rather than something that happens between
// kernels. A one-shot testbench per launch would make every launch a recovery,
// and would make "the intermediate result stayed in device memory" impossible
// to claim: A->B->C has nowhere to leave anything.
//
// So this does not exit. It comes up, says so, and then serves requests until
// it is told to stop.
//
// The protocol is spec/r2_transport_v1.json: a request/response mailbox with a
// monotonic sequence number, which is the shape a hardware doorbell has. The
// host writes a descriptor and rings; the device reads it, acts, and writes a
// completion. Swapping the two text files for two MMIO registers would change
// the medium and not the protocol -- which is the point, because R2-10 asks
// for a transport that could be implemented in hardware.
//
// Plusargs:
//   +req=<file> +rsp=<file>   the mailbox
//   +up=<file>  +down=<file>  the payload files
//   +hex=<file>               an initial image (optional)
//   +pollcycles=<n>           how often to look at the mailbox
//   +deadline=<n>             cycles a single launch may take before the
//                             device reports TIMEOUT. NOT a host wait: this
//                             is the device's own deadline (R2-08).
//   +trace                    one line per served request
//==============================================================================
`timescale 1ns/1ps

module tnca_device;

    parameter NUM_RCTX       = 2;
    parameter RCTX_W         = 1;
    parameter VLEN           = 16;
    parameter ELEN           = 32;
    parameter VREGS_PER_RCTX = 64;
    parameter PHYS_VREGS     = NUM_RCTX * VREGS_PER_RCTX;
    parameter PHYS_W         = 7;
    parameter WORDS          = 16384;
    parameter MEM_LATENCY    = 1;
    parameter RANDOM_STALL   = 0;
    parameter MEM_REORDER    = 0;
    parameter LSU_TAG_W      = 5;
    parameter HAS_SFU        = 1;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    wire [63:0] imem_addr, imem_data;
    wire        mreq_valid, mreq_ready, mreq_we;
    wire [63:0] mreq_addr;
    wire [7:0]  mreq_be;
    wire        mreq_amo;
    wire [3:0]  mreq_aop;
    wire [63:0] mreq_cmp, mreq_wdata;
    wire        mrsp_valid;
    wire [63:0] mrsp_rdata;
    wire [LSU_TAG_W-1:0] mreq_tag, mrsp_tag;
    wire [1:0]  mreq_hint;

    reg         launch_valid = 1'b0;
    reg [63:0]  launch_pc = 64'd0;
    reg [NUM_RCTX-1:0] launch_role_mask;
    reg [31:0]  pid_x = 0, pid_y = 0, pid_z = 0;
    reg [31:0]  npid_x = 1, npid_y = 1, npid_z = 1;
    reg [63:0]  arg_base = 64'd0;
    reg [NUM_RCTX*PHYS_W-1:0] launch_vbase;

    wire        core_idle;
    wire        fault_valid;
    wire [7:0]  fault_code;
    wire [63:0] fault_pc;
    wire [RCTX_W-1:0] fault_rctx;
    wire [23:0] fault_detail;

    wire [63:0] perf_cycles, perf_issued, perf_stall_hazard,
                perf_stall_lsu, perf_stall_wb, perf_squash;

    tnca_core #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .VLEN(VLEN), .ELEN(ELEN), .LW(4),
        .VREGS_PER_RCTX(VREGS_PER_RCTX), .PHYS_VREGS(PHYS_VREGS),
        .PHYS_W(PHYS_W), .CORE_ID(0), .NUM_CORES(1), .HAS_SFU(HAS_SFU)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .launch_valid(launch_valid), .launch_pc(launch_pc),
        .launch_role_mask(launch_role_mask),
        .launch_pid_x(pid_x), .launch_pid_y(pid_y), .launch_pid_z(pid_z),
        .launch_npid_x(npid_x), .launch_npid_y(npid_y), .launch_npid_z(npid_z),
        .launch_arg_base(arg_base),
        .launch_vbase(launch_vbase),
        .core_idle(core_idle),
        .fault_valid(fault_valid), .fault_code(fault_code),
        .fault_pc(fault_pc), .fault_rctx(fault_rctx),
        .fault_detail(fault_detail),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mreq_valid), .mem_req_ready(mreq_ready),
        .mem_req_amo(mreq_amo), .mem_req_aop(mreq_aop), .mem_req_cmp(mreq_cmp),
        .mem_req_addr(mreq_addr), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wdata),
        .mem_req_tag(mreq_tag), .mem_req_hint(mreq_hint),
        .mem_rsp_valid(mrsp_valid), .mem_rsp_tag(mrsp_tag),
        .mem_rsp_rdata(mrsp_rdata),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard), .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash)
    );

    tnca_mem_model #(
        .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
        .LATENCY(MEM_LATENCY), .RANDOM_STALL(RANDOM_STALL),
        .REORDER(MEM_REORDER)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .req_valid(mreq_valid), .req_ready(mreq_ready),
        .req_addr(mreq_addr), .req_we(mreq_we),
        .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
        .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
        .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag), .rsp_rdata(mrsp_rdata)
    );

    //==========================================================================
    // The mailbox
    //==========================================================================
    reg [1023:0] req_path, rsp_path, up_path, down_path, hexfile;
    integer pollcycles, deadline;
    reg do_trace;

    integer served;              // the highest sequence acted on
    integer fh, n, i;
    integer seq, op, a0, a1, a2, a3, a4, a5;
    integer status, r0, r1;
    integer cyc;

    task write_response;
        input integer rseq;
        input integer rstatus;
        input integer rr0;
        input integer rr1;
        integer f;
        begin
            // Written in one go and closed, so a host that reads a partial
            // line sees nothing rather than half an answer.
            f = $fopen(rsp_path, "w");
            $fwrite(f, "%0d %0d %0d %0d\n", rseq, rstatus, rr0, rr1);
            $fclose(f);
        end
    endtask

    // One launch, from idle to idle. Returns through `status`.
    //
    // All six grid values, not two. ISA sec 5.1 makes `program_id(axis)` the
    // logical index on THAT axis and `num_programs(axis)` the logical extent
    // of it; carrying only x meant a 2x3 grid handed every program a pid.x of
    // 0..5 and an npid.x of 6, which is the linear id and the total wearing
    // the names of the x index and the x extent. A kernel cannot tell the
    // difference and gets the wrong answer.
    task do_launch;
        input integer pc;
        input integer argb;
        input integer mask;
        input integer px;
        input integer py;
        input integer pz;
        input integer nx;
        input integer ny;
        input integer nz;
        begin
            launch_pc    = pc;
            arg_base     = argb;
            launch_role_mask = mask[NUM_RCTX-1:0];
            pid_x = px;  pid_y = py;  pid_z = pz;
            npid_x = (nx == 0) ? 1 : nx;
            npid_y = (ny == 0) ? 1 : ny;
            npid_z = (nz == 0) ? 1 : nz;

            // Stimulus one delta after the edge, never on it: a blocking
            // assignment in the active region of the edge the DUT samples is
            // a race, and it silently swallowed the launch pulse once.
            @(posedge clk);
            #1 launch_valid = 1'b1;
            @(posedge clk);
            #1 launch_valid = 1'b0;
            repeat (3) @(posedge clk);

            cyc = 0;
            while (!core_idle && cyc < deadline) begin
                @(posedge clk);
                cyc = cyc + 1;
            end

            if (cyc >= deadline) begin
                status = 2;                  // TIMEOUT: the DEVICE deadline
                r0 = cyc; r1 = 0;
            end else if (fault_valid) begin
                status = 1;                  // FAULT
                r0 = {24'd0, fault_code};
                r1 = fault_pc[31:0];
            end else begin
                status = 0;
                r0 = cyc;
                r1 = perf_issued[31:0];
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("req=%s", req_path)) begin
            $display("FATAL: pass +req=<file>"); $finish;
        end
        if (!$value$plusargs("rsp=%s", rsp_path)) begin
            $display("FATAL: pass +rsp=<file>"); $finish;
        end
        // Not `void'(...)`: that is SystemVerilog, and this compiles as
        // Verilog-2005 like the rest of the benches.
        if ($value$plusargs("up=%s", up_path)) ;
        if ($value$plusargs("down=%s", down_path)) ;
        if (!$value$plusargs("pollcycles=%d", pollcycles)) pollcycles = 20;
        if (!$value$plusargs("deadline=%d", deadline)) deadline = 2000000;
        do_trace = $test$plusargs("trace");

        if ($value$plusargs("hex=%s", hexfile))
            u_mem.load_hex(hexfile);

        launch_role_mask = {{(NUM_RCTX-1){1'b0}}, 1'b1};
        for (i = 0; i < NUM_RCTX; i = i + 1)
            launch_vbase[i*PHYS_W +: PHYS_W] = i * VREGS_PER_RCTX;

        served = 0;
        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // The handshake that proves the device came up. A host that never
        // sees this knows the difference between "slow" and "never started".
        $display("DEVICE READY words=%0d rctx=%0d", WORDS, NUM_RCTX);
        write_response(0, 0, 0, 0);

        forever begin
            repeat (pollcycles) @(posedge clk);

            fh = $fopen(req_path, "r");
            if (fh != 0) begin
                n = $fscanf(fh, "%d %d %d %d %d %d %d %d",
                            seq, op, a0, a1, a2, a3, a4, a5);
                $fclose(fh);
                if (n == 8 && seq > served) begin
                    served = seq;
                    status = 0; r0 = 0; r1 = 0;
                    case (op)
                        1: begin                       // UPLOAD
                            if (a0 % 8 != 0 || a1 <= 0 ||
                                (a0/8 + a1) > WORDS) status = 3;
                            else u_mem.load_range(up_path, a0/8, a1);
                        end
                        2: begin                       // DOWNLOAD
                            if (a0 % 8 != 0 || a1 <= 0 ||
                                (a0/8 + a1) > WORDS) status = 3;
                            else u_mem.dump_hex(down_path, a0/8, a1);
                        end
                        // a3 = pid_x | pid_y<<16, a4 = pid_z | npid_x<<16,
                        // a5 = npid_y | npid_z<<16. Sixteen bits per axis is
                        // ample: the contract caps a grid at 4096 programs.
                        3: do_launch(a0, a1, a2,
                                     a3 & 32'hFFFF, (a3 >> 16) & 32'hFFFF,
                                     a4 & 32'hFFFF, (a4 >> 16) & 32'hFFFF,
                                     a5 & 32'hFFFF, (a5 >> 16) & 32'hFFFF);
                        4: begin                       // QUIESCE
                            i = 0;
                            while (!core_idle && i < deadline) begin
                                @(posedge clk); i = i + 1;
                            end
                            status = (i >= deadline) ? 2 : 0;
                            r0 = i;
                        end
                        5: begin                       // SHUTDOWN
                            write_response(seq, 0, 0, 0);
                            $display("DEVICE SHUTDOWN after %0d requests", served);
                            $finish;
                        end
                        6: begin                       // PING
                            r0 = served;
                        end
                        default: status = 3;
                    endcase
                    if (do_trace)
                        $display("[dev] seq=%0d op=%0d a0=%0d a1=%0d -> status=%0d r0=%0d", seq, op, a0, a1, status, r0);
                    write_response(seq, status, r0, r1);
                end
            end
        end
    end

endmodule
