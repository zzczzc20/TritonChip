// Register-only driver for the actual coherent multicore FPGA RTL wrapper.
`timescale 1ns/1ps
`default_nettype none
module tnca_multicore_regs;
 parameter N=3,PORTS=3,MEMORY_BYTES=1048576,STAGES=2,NOC_PIPE=0,DISPATCH_PIPE=0,HOME_PIPE=0;
 localparam NX=N>4?4:N,ADDR_BITS=8,WORDS=(131072+MEMORY_BYTES)/8;
 reg clk=0,rst_n_async=0;always #5 clk=~clk;
 reg [7:0] reg_addr;reg [63:0] reg_wdata;reg reg_we,reg_re;
 wire [63:0] reg_rdata;
 wire [PORTS-1:0] m_awvalid;
 wire [PORTS-1:0] m_awready;
 wire [PORTS*64-1:0] m_awaddr;
 wire [PORTS*3-1:0] m_awid;
 wire [PORTS*8-1:0] m_awlen;
 wire [PORTS*3-1:0] m_awsize;
 wire [PORTS*2-1:0] m_awburst;
 wire [PORTS-1:0] m_wvalid;
 wire [PORTS-1:0] m_wready;
 wire [PORTS*256-1:0] m_wdata;
 wire [PORTS*256/8-1:0] m_wstrb;
 wire [PORTS-1:0] m_wlast;
 wire [PORTS-1:0] m_bvalid;
 wire [PORTS-1:0] m_bready;
 wire [PORTS*3-1:0] m_bid;
 wire [PORTS*2-1:0] m_bresp;
 wire [PORTS-1:0] m_arvalid;
 wire [PORTS-1:0] m_arready;
 wire [PORTS*64-1:0] m_araddr;
 wire [PORTS*3-1:0] m_arid;
 wire [PORTS*8-1:0] m_arlen;
 wire [PORTS*3-1:0] m_arsize;
 wire [PORTS*2-1:0] m_arburst;
 wire [PORTS-1:0] m_rvalid;
 wire [PORTS-1:0] m_rready;
 wire [PORTS*256-1:0] m_rdata;
 wire [PORTS*3-1:0] m_rid;
 wire [PORTS*2-1:0] m_rresp;
 wire [PORTS-1:0] m_rlast;
 tnca_multicore_fpga #(.NX(NX),.NY(N/NX),.MEM_PORTS(PORTS),.LINK_STAGES(STAGES),.NOC_LINK_STAGES(NOC_PIPE),.DISPATCH_STAGES(DISPATCH_PIPE),.MEM_HOME_STAGES(HOME_PIPE),.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)) dut(
 .clk(clk),.rst_n(rst_n_async),.reg_addr(reg_addr),.reg_wdata(reg_wdata),.reg_we(reg_we),.reg_re(reg_re),.reg_rdata(reg_rdata),
 .m_awvalid(m_awvalid),
 .m_awready(m_awready),
 .m_awaddr(m_awaddr),
 .m_awid(m_awid),
 .m_awlen(m_awlen),
 .m_awsize(m_awsize),
 .m_awburst(m_awburst),
 .m_wvalid(m_wvalid),
 .m_wready(m_wready),
 .m_wdata(m_wdata),
 .m_wstrb(m_wstrb),
 .m_wlast(m_wlast),
 .m_bvalid(m_bvalid),
 .m_bready(m_bready),
 .m_bid(m_bid),
 .m_bresp(m_bresp),
 .m_arvalid(m_arvalid),
 .m_arready(m_arready),
 .m_araddr(m_araddr),
 .m_arid(m_arid),
 .m_arlen(m_arlen),
 .m_arsize(m_arsize),
 .m_arburst(m_arburst),
 .m_rvalid(m_rvalid),
 .m_rready(m_rready),
 .m_rdata(m_rdata),
 .m_rid(m_rid),
 .m_rresp(m_rresp),
 .m_rlast(m_rlast));
 generate for(genvar p=0;p<PORTS;p=p+1)begin:g_ram
  tnca_axi_line_model #(.DATA_W(256),.OUT_N(8),.LINES((131072+MEMORY_BYTES)/64),.LATENCY(25+p*19)) ram(
  .clk(clk),.rst_n(rst_n_async),
 .m_awvalid(m_awvalid[p*(1)+:(1)]),
 .m_awready(m_awready[p*(1)+:(1)]),
 .m_awaddr(m_awaddr[p*(64)+:(64)]),
 .m_awid(m_awid[p*(3)+:(3)]),
 .m_awlen(m_awlen[p*(8)+:(8)]),
 .m_awsize(m_awsize[p*(3)+:(3)]),
 .m_awburst(m_awburst[p*(2)+:(2)]),
 .m_wvalid(m_wvalid[p*(1)+:(1)]),
 .m_wready(m_wready[p*(1)+:(1)]),
 .m_wdata(m_wdata[p*(256)+:(256)]),
 .m_wstrb(m_wstrb[p*(256/8)+:(256/8)]),
 .m_wlast(m_wlast[p*(1)+:(1)]),
 .m_bvalid(m_bvalid[p*(1)+:(1)]),
 .m_bready(m_bready[p*(1)+:(1)]),
 .m_bid(m_bid[p*(3)+:(3)]),
 .m_bresp(m_bresp[p*(2)+:(2)]),
 .m_arvalid(m_arvalid[p*(1)+:(1)]),
 .m_arready(m_arready[p*(1)+:(1)]),
 .m_araddr(m_araddr[p*(64)+:(64)]),
 .m_arid(m_arid[p*(3)+:(3)]),
 .m_arlen(m_arlen[p*(8)+:(8)]),
 .m_arsize(m_arsize[p*(3)+:(3)]),
 .m_arburst(m_arburst[p*(2)+:(2)]),
 .m_rvalid(m_rvalid[p*(1)+:(1)]),
 .m_rready(m_rready[p*(1)+:(1)]),
 .m_rdata(m_rdata[p*(256)+:(256)]),
 .m_rid(m_rid[p*(3)+:(3)]),
 .m_rresp(m_rresp[p*(2)+:(2)]),
 .m_rlast(m_rlast[p*(1)+:(1)]));
 end endgenerate
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
