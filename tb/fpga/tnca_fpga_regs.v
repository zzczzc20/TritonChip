//==============================================================================
// tnca_fpga_regs -- the register server, for a host that is not a testbench
//
// tb/fpga/tb_tnca_fpga.v is a testbench. It knows the program, it knows the
// answers, and it decides pass or fail. That is what a protocol reference
// ought to be, and it is not something a host process can drive: there is no
// way to ask it "write this register and tell me what came back".
//
// This is that. The same device, the same register port, no opinion at all
// about what the accesses mean. python/tnca/backends/u50_sim.py drives it,
// and through it python/tnca/backends/u50.py -- the file that will drive the
// board.
//
// WHY A SECOND BENCH IS WORTH IT
//
// The backend that runs on the U50 does not get to differ from the one that
// runs here. So what is tested here has to be what goes on the board, which
// means the register protocol has to be exercised THROUGH the host code and
// not only through a bench written beside it and sharing its assumptions.
// That is R36-HAND-02's split in one sentence: the medium is replaced and
// the protocol is not.
//
// A bench and a server check different things and both are needed.
// tb_tnca_fpga.v checks values -- hand-computed ones, so that agreement
// between a bench and a device cannot be agreement about being wrong
// together. This one checks that the host's own protocol code drives the
// real port correctly, which no bench can check from the inside.
//
// PROTOCOL
//
// Two files in a directory the host owns, both named on the command line.
//
//   regs.cmd, written whole by the host and never appended to:
//       <tid>               the transaction id, one greater each round trip
//       w <offset> <value>  a 64-bit register write
//       r <offset>          a 64-bit register read
//
//   regs.rsp, written whole by this bench:
//       <tid>               echoed, so a host cannot read an old answer
//       <value>             one line per `r`, in the order they appeared
//       end                 the transaction is complete
//
// The tid is the mailbox's sequence number in another medium, for the same
// reason: a file written by one process and read by another has no other way
// to say which answer belongs to which question, and a host that took the
// first answer it found would read the previous transaction's. That is
// ERR-05 at the level below the mailbox.
//
// TIMING
//
// `intercycle` clocks pass between commands, so that the device's own state
// machines have edges to move on. Without it a whole batch would be applied
// in one delta and every status register would read the value it held before
// the accesses it is being asked about -- which is indistinguishable, from
// the host, from a device that ignores writes.
//
// `pollcycles` pass between one look for a command file and the next. The
// host pays that latency once per round trip and the simulation pays it in
// wasted cycles; both are why it is a plusarg and not a constant.
//==============================================================================
`default_nettype none

module tnca_fpga_regs;

    // The R39 hardware switches, as the board image has them (core-vrf-sync-v2;
    // rtl/fpga/tnca_fpga_top.v). The fpga-u50-sim backend overrides them with
    // -P (SimLink rtl_params) to build core-vrf-sync-v1 for an ablation.
    parameter VRF_BYPASS   = 1;
    parameter FE_NOBUBBLE  = 1;
    parameter VRF_PREFETCH = 1;

    // The device, exactly as the board would instantiate it: one clock, one
    // asynchronous reset, one flat register port. Nothing is added for the
    // benefit of the host -- if this bench needed a port the board does not
    // have, the backend it is testing would not be the backend that ships.
    localparam ADDR_BITS = 8;
    localparam WORDS     = 65536;
    localparam ADDR_W    = 16;
    localparam LSU_TAG_W = 5;

    reg clk = 1'b0;
    reg rst_n_async = 1'b0;

    reg  [ADDR_BITS-1:0] reg_addr;
    reg  [63:0]          reg_wdata;
    reg                  reg_we;
    reg                  reg_re;
    wire [63:0]          reg_rdata;

    always #5 clk = ~clk;

    tnca_fpga_top #(
        .WORDS(WORDS), .ADDR_W(ADDR_W), .LSU_TAG_W(LSU_TAG_W),
        .USE_BANKED_VRF(1), .VRF_BYPASS(VRF_BYPASS), .FE_NOBUBBLE(FE_NOBUBBLE),
        .VRF_PREFETCH(VRF_PREFETCH)
    ) u_top (
        .clk(clk), .rst_n_async(rst_n_async),
        .reg_addr(reg_addr), .reg_wdata(reg_wdata),
        .reg_we(reg_we), .reg_re(reg_re), .reg_rdata(reg_rdata)
    );

    //==========================================================================
    // Files and plusargs
    //==========================================================================
    reg [8*512-1:0] cmd_path, rsp_path;
    integer         pollcycles, intercycle, watchdog;

    integer      tid_max;
    reg [63:0]   last_tid;

    reg [8*512-1:0] line;
    reg [8*64-1:0]  kind;
    reg [63:0]      arg_off, arg_val;
    reg [63:0]      values [0:4095];
    integer         nvalues;

    integer fh, i, n, wd;
    reg     do_trace;

    //==========================================================================
    // One round trip
    //
    // The batch is applied in the order it was written and read back in the
    // order it was asked for. Nothing is reordered and nothing is cached: a
    // register read is a read of the device, every time, because a host that
    // cached one would be a host that cannot see the device move.
    //==========================================================================
    task run_batch;
        input [63:0] tid;
        begin
            nvalues = 0;
            fh = $fopen(cmd_path, "r");
            if (fh != 0) begin
                // Skip the id line; it was read to get here.
                n = $fscanf(fh, "%d", arg_off);
                while ($fgets(line, fh) != 0) begin
                    if ($sscanf(line, "%s %d %d", kind, arg_off, arg_val) >= 2) begin
                        reg_addr = arg_off[ADDR_BITS-1:0];
                        if (kind == "w") begin
                            reg_wdata = arg_val;
                            reg_we    = 1'b1;
                            reg_re    = 1'b0;
                            @(posedge clk);
                            #1;
                            reg_we    = 1'b0;
                            if (do_trace)
                                $display("  w %0d %016h", arg_off, arg_val);
                        end else if (kind == "r") begin
                            reg_we = 1'b0;
                            reg_re = 1'b1;
                            #1;
                            // Combinational: the value is beside the address
                            // in the same cycle, which is the contract
                            // rtl/fpga/tnca_mmio.v states for `reg_rdata`.
                            values[nvalues] = reg_rdata;
                            nvalues = nvalues + 1;
                            @(posedge clk);
                            #1;
                            reg_re = 1'b0;
                            if (do_trace)
                                $display("  r %0d -> %016h", arg_off,
                                         values[nvalues-1]);
                        end else begin
                            $display("tnca_fpga_regs: unknown command %0s",
                                     kind);
                        end
                        repeat (intercycle) @(posedge clk);
                        #1;
                    end
                end
                $fclose(fh);
            end

            fh = $fopen(rsp_path, "w");
            if (fh == 0) begin
                $display("tnca_fpga_regs: cannot write %0s", rsp_path);
                $finish;
            end
            $fwrite(fh, "%0d\n", tid);
            for (i = 0; i < nvalues; i = i + 1)
                $fwrite(fh, "%016x\n", values[i]);
            $fwrite(fh, "end\n");
            $fclose(fh);
            $fflush;

            last_tid = tid;
        end
    endtask

    //==========================================================================
    // Run
    //==========================================================================
    initial begin
        reg_addr  = {ADDR_BITS{1'b0}};
        reg_wdata = 64'd0;
        reg_we    = 1'b0;
        reg_re    = 1'b0;
        last_tid  = 64'hFFFF_FFFF_FFFF_FFFF;
        nvalues   = 0;
        do_trace  = $test$plusargs("trace");

        if (!$value$plusargs("cmd=%s", cmd_path)) begin
            $display("tnca_fpga_regs: pass +cmd=<file>");
            $finish;
        end
        if (!$value$plusargs("rsp=%s", rsp_path)) begin
            $display("tnca_fpga_regs: pass +rsp=<file>");
            $finish;
        end
        if (!$value$plusargs("pollcycles=%d", pollcycles)) pollcycles = 20;
        if (!$value$plusargs("intercycle=%d", intercycle)) intercycle = 8;
        // A host that dies leaves this running until something stops it. The
        // watchdog is that something, and it says so rather than exiting
        // silently -- a simulation that stops without a line is a simulation
        // nobody can tell from a slow one.
        if (!$value$plusargs("watchdog=%d", watchdog)) watchdog = 200000000;

        // Reset, asserted asynchronously and released synchronously, the same
        // shape rtl/fpga/tnca_fpga_top.v asks a board to supply.
        repeat (8) @(posedge clk);
        #1 rst_n_async = 1'b1;
        repeat (8) @(posedge clk);

        $display("tnca_fpga_regs: up (words=%0d)", WORDS);
        // A simulator with its stdout redirected to a file buffers it, so
        // without this the line above sits in a buffer and a host watching
        // the log for it sees nothing. That cost half an hour once; the
        // host no longer watches the log, and the flush stays because the
        // log is what a person reads when it goes wrong.
        $fflush;

        wd = 0;
        while (wd < watchdog) begin
            repeat (pollcycles) @(posedge clk);
            wd = wd + pollcycles;
            #1;
            fh = $fopen(cmd_path, "r");
            if (fh != 0) begin
                if ($fscanf(fh, "%d", tid_max) == 1) begin
                    $fclose(fh);
                    if (tid_max != last_tid)
                        run_batch(tid_max);
                end else begin
                    $fclose(fh);
                end
            end
        end

        $display("tnca_fpga_regs: watchdog at %0d cycles with no completion",
                 watchdog);
        $finish;
    end

endmodule

`default_nettype wire
