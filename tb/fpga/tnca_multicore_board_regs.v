// Register-only driver for the actual coherent multicore FPGA RTL wrapper.
`timescale 1ns/1ps
`default_nettype none
module tnca_multicore_board_regs;
 parameter N=3,PORTS=3,MEMORY_BYTES=1048576,STAGES=2,NOC_PIPE=0,DISPATCH_PIPE=0,HOME_PIPE=0,CODE_PIPE=0,MAINTENANCE=0,MEMORY_CONTROLS=0,HYBRID_LSU=0,SFU_LANES=1,HOME_BACKGROUND_READS=0;
 // Stand-in BO address and optional sparse backing (tb/common/tnca_sparse_map.vh):
 // SPARSE=1 backs only SPARSE_MAP's ranges (SPARSE_LINES lines) of a window of
 // MEMORY_BYTES, so an 8 GiB window can be simulated. Defaults keep the dense
 // bench at 0x1_0000_0000 unchanged.
 parameter [63:0] WIN_BASE=64'h100000000;
 parameter SPARSE=0,SPARSE_LINES=0,SPARSE_MAP="none";
 // Cache geometry and convergence options; defaults keep the small, fast
 // bench. tools/r40/image_geometry_tests.py sets them from an image.tcl.
 parameter SETS=4,SET_W=2,WAYS=2,WAY_W=1,DCO_REGISTERED_RANGES=0,NUCA_REGISTER_VICTIM_SELECT=0,
  NUCA_CACHE_DATA_PIPELINE=0,NUCA_REGISTER_VICTIM_RANK=0,DCO_TLW_PIPE_STAGES=0,
  DCO_REGISTER_WRITES=0,DCO_LOCAL_REPLICAS=0,NUCA_SHARED_DATA_READ=0,AXI_LINE_LUTRAM=0;
 // The image's tensor-transpose option (core-vrf-sync-v5-x8 images build 2).
 parameter HAS_TENSOR_XPOSE=1;
 // R41 G2 resource sizing (tools/r40/image_geometry_tests.py sets them from an image.tcl).
 parameter NUM_RCTX=4,RCTX_W=2,PHYS_VREGS=256,PHYS_W=8,ACC_N=16,ACC_W=4,BAR_N=32,BAR_W=5,BAR_EPOCH_W=48,PHYS_ADDR_W=64;
 localparam NX=N>4?4:N,ADDR_BITS=9,WORDS=(131072+MEMORY_BYTES)/8;
 reg clk=0,rst_n_async=0;always #5 clk=~clk;
 reg [8:0] reg_addr;reg [63:0] reg_wdata;reg reg_we,reg_re;
 wire [63:0] reg_rdata;
 wire [3-1:0] m_awvalid;
 wire [3-1:0] m_awready;
 wire [3*64-1:0] m_awaddr;
 wire [3*3-1:0] m_awid;
 wire [3*8-1:0] m_awlen;
 wire [3*3-1:0] m_awsize;
 wire [3*2-1:0] m_awburst;
 wire [3-1:0] m_wvalid;
 wire [3-1:0] m_wready;
 wire [3*256-1:0] m_wdata;
 wire [3*256/8-1:0] m_wstrb;
 wire [3-1:0] m_wlast;
 wire [3-1:0] m_bvalid;
 wire [3-1:0] m_bready;
 wire [3*3-1:0] m_bid;
 wire [3*2-1:0] m_bresp;
 wire [3-1:0] m_arvalid;
 wire [3-1:0] m_arready;
 wire [3*64-1:0] m_araddr;
 wire [3*3-1:0] m_arid;
 wire [3*8-1:0] m_arlen;
 wire [3*3-1:0] m_arsize;
 wire [3*2-1:0] m_arburst;
 wire [3-1:0] m_rvalid;
 wire [3-1:0] m_rready;
 wire [3*256-1:0] m_rdata;
 wire [3*3-1:0] m_rid;
 wire [3*2-1:0] m_rresp;
 wire [3-1:0] m_rlast;
 reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
 reg [11:0] awaddr=0,araddr=0;reg [31:0] wdata=0;reg [3:0] wstrb=15;
 wire awready,wready,bvalid,arready,rvalid;wire [31:0] rdata;wire [1:0] bresp,rresp;
 tnca_multicore_hbm_xo #(.HOME_BACKGROUND_READS(HOME_BACKGROUND_READS),.HYBRID_LSU(HYBRID_LSU),.SFU_LANES(SFU_LANES),.MEMORY_CONTROLS(MEMORY_CONTROLS),.CACHE_MAINTENANCE(MAINTENANCE),.MEM_PORTS(PORTS),.LINK_STAGES(STAGES),.CODE_LINK_STAGES(CODE_PIPE),.NOC_LINK_STAGES(NOC_PIPE),.DISPATCH_STAGES(DISPATCH_PIPE),.MEM_HOME_STAGES(HOME_PIPE),.SETS(SETS),.SET_W(SET_W),.WAYS(WAYS),.WAY_W(WAY_W),
  .DCO_REGISTERED_RANGES(DCO_REGISTERED_RANGES),.NUCA_REGISTER_VICTIM_SELECT(NUCA_REGISTER_VICTIM_SELECT),
  .NUCA_CACHE_DATA_PIPELINE(NUCA_CACHE_DATA_PIPELINE),.NUCA_REGISTER_VICTIM_RANK(NUCA_REGISTER_VICTIM_RANK),
  .DCO_TLW_PIPE_STAGES(DCO_TLW_PIPE_STAGES),.DCO_REGISTER_WRITES(DCO_REGISTER_WRITES),
  .DCO_LOCAL_REPLICAS(DCO_LOCAL_REPLICAS),.NUCA_SHARED_DATA_READ(NUCA_SHARED_DATA_READ),.AXI_LINE_LUTRAM(AXI_LINE_LUTRAM),
  .HAS_TENSOR_XPOSE(HAS_TENSOR_XPOSE),.NUM_RCTX(NUM_RCTX),.RCTX_W(RCTX_W),.PHYS_VREGS(PHYS_VREGS),.PHYS_W(PHYS_W),
  .ACC_N(ACC_N),.ACC_W(ACC_W),.BAR_N(BAR_N),.BAR_W(BAR_W),.BAR_EPOCH_W(BAR_EPOCH_W),.PHYS_ADDR_W(PHYS_ADDR_W)) dut(
 .ap_clk(clk),.ap_rst_n(rst_n_async),
 .s_axi_tnca_awvalid(awvalid),
 .s_axi_tnca_awready(awready),
 .s_axi_tnca_awaddr(awaddr),
 .s_axi_tnca_wvalid(wvalid),
 .s_axi_tnca_wready(wready),
 .s_axi_tnca_wdata(wdata),
 .s_axi_tnca_wstrb(wstrb),
 .s_axi_tnca_bvalid(bvalid),
 .s_axi_tnca_bready(bready),
 .s_axi_tnca_bresp(bresp),
 .s_axi_tnca_arvalid(arvalid),
 .s_axi_tnca_arready(arready),
 .s_axi_tnca_araddr(araddr),
 .s_axi_tnca_rvalid(rvalid),
 .s_axi_tnca_rready(rready),
 .s_axi_tnca_rdata(rdata),
 .s_axi_tnca_rresp(rresp),
 .m_axi_hbm0_awvalid(m_awvalid[0+:1]),
 .m_axi_hbm0_awready(m_awready[0+:1]),
 .m_axi_hbm0_awaddr(m_awaddr[0+:64]),
 .m_axi_hbm0_awid(m_awid[0+:3]),
 .m_axi_hbm0_awlen(m_awlen[0+:8]),
 .m_axi_hbm0_awsize(m_awsize[0+:3]),
 .m_axi_hbm0_awburst(m_awburst[0+:2]),
 .m_axi_hbm0_wvalid(m_wvalid[0+:1]),
 .m_axi_hbm0_wready(m_wready[0+:1]),
 .m_axi_hbm0_wdata(m_wdata[0+:256]),
 .m_axi_hbm0_wstrb(m_wstrb[0+:32]),
 .m_axi_hbm0_wlast(m_wlast[0+:1]),
 .m_axi_hbm0_bvalid(m_bvalid[0+:1]),
 .m_axi_hbm0_bready(m_bready[0+:1]),
 .m_axi_hbm0_bid(m_bid[0+:3]),
 .m_axi_hbm0_bresp(m_bresp[0+:2]),
 .m_axi_hbm0_arvalid(m_arvalid[0+:1]),
 .m_axi_hbm0_arready(m_arready[0+:1]),
 .m_axi_hbm0_araddr(m_araddr[0+:64]),
 .m_axi_hbm0_arid(m_arid[0+:3]),
 .m_axi_hbm0_arlen(m_arlen[0+:8]),
 .m_axi_hbm0_arsize(m_arsize[0+:3]),
 .m_axi_hbm0_arburst(m_arburst[0+:2]),
 .m_axi_hbm0_rvalid(m_rvalid[0+:1]),
 .m_axi_hbm0_rready(m_rready[0+:1]),
 .m_axi_hbm0_rdata(m_rdata[0+:256]),
 .m_axi_hbm0_rid(m_rid[0+:3]),
 .m_axi_hbm0_rresp(m_rresp[0+:2]),
 .m_axi_hbm0_rlast(m_rlast[0+:1]),
 .m_axi_hbm1_awvalid(m_awvalid[1+:1]),
 .m_axi_hbm1_awready(m_awready[1+:1]),
 .m_axi_hbm1_awaddr(m_awaddr[64+:64]),
 .m_axi_hbm1_awid(m_awid[3+:3]),
 .m_axi_hbm1_awlen(m_awlen[8+:8]),
 .m_axi_hbm1_awsize(m_awsize[3+:3]),
 .m_axi_hbm1_awburst(m_awburst[2+:2]),
 .m_axi_hbm1_wvalid(m_wvalid[1+:1]),
 .m_axi_hbm1_wready(m_wready[1+:1]),
 .m_axi_hbm1_wdata(m_wdata[256+:256]),
 .m_axi_hbm1_wstrb(m_wstrb[32+:32]),
 .m_axi_hbm1_wlast(m_wlast[1+:1]),
 .m_axi_hbm1_bvalid(m_bvalid[1+:1]),
 .m_axi_hbm1_bready(m_bready[1+:1]),
 .m_axi_hbm1_bid(m_bid[3+:3]),
 .m_axi_hbm1_bresp(m_bresp[2+:2]),
 .m_axi_hbm1_arvalid(m_arvalid[1+:1]),
 .m_axi_hbm1_arready(m_arready[1+:1]),
 .m_axi_hbm1_araddr(m_araddr[64+:64]),
 .m_axi_hbm1_arid(m_arid[3+:3]),
 .m_axi_hbm1_arlen(m_arlen[8+:8]),
 .m_axi_hbm1_arsize(m_arsize[3+:3]),
 .m_axi_hbm1_arburst(m_arburst[2+:2]),
 .m_axi_hbm1_rvalid(m_rvalid[1+:1]),
 .m_axi_hbm1_rready(m_rready[1+:1]),
 .m_axi_hbm1_rdata(m_rdata[256+:256]),
 .m_axi_hbm1_rid(m_rid[3+:3]),
 .m_axi_hbm1_rresp(m_rresp[2+:2]),
 .m_axi_hbm1_rlast(m_rlast[1+:1]),
 .m_axi_hbm2_awvalid(m_awvalid[2+:1]),
 .m_axi_hbm2_awready(m_awready[2+:1]),
 .m_axi_hbm2_awaddr(m_awaddr[128+:64]),
 .m_axi_hbm2_awid(m_awid[6+:3]),
 .m_axi_hbm2_awlen(m_awlen[16+:8]),
 .m_axi_hbm2_awsize(m_awsize[6+:3]),
 .m_axi_hbm2_awburst(m_awburst[4+:2]),
 .m_axi_hbm2_wvalid(m_wvalid[2+:1]),
 .m_axi_hbm2_wready(m_wready[2+:1]),
 .m_axi_hbm2_wdata(m_wdata[512+:256]),
 .m_axi_hbm2_wstrb(m_wstrb[64+:32]),
 .m_axi_hbm2_wlast(m_wlast[2+:1]),
 .m_axi_hbm2_bvalid(m_bvalid[2+:1]),
 .m_axi_hbm2_bready(m_bready[2+:1]),
 .m_axi_hbm2_bid(m_bid[6+:3]),
 .m_axi_hbm2_bresp(m_bresp[4+:2]),
 .m_axi_hbm2_arvalid(m_arvalid[2+:1]),
 .m_axi_hbm2_arready(m_arready[2+:1]),
 .m_axi_hbm2_araddr(m_araddr[128+:64]),
 .m_axi_hbm2_arid(m_arid[6+:3]),
 .m_axi_hbm2_arlen(m_arlen[16+:8]),
 .m_axi_hbm2_arsize(m_arsize[6+:3]),
 .m_axi_hbm2_arburst(m_arburst[4+:2]),
 .m_axi_hbm2_rvalid(m_rvalid[2+:1]),
 .m_axi_hbm2_rready(m_rready[2+:1]),
 .m_axi_hbm2_rdata(m_rdata[512+:256]),
 .m_axi_hbm2_rid(m_rid[6+:3]),
 .m_axi_hbm2_rresp(m_rresp[4+:2]),
 .m_axi_hbm2_rlast(m_rlast[2+:1]));
 generate for(genvar p=0;p<PORTS;p=p+1)begin:g_ram
  tnca_axi_line_model #(.DATA_W(256),.OUT_N(8),.LINES(SPARSE ? SPARSE_LINES : (131072+MEMORY_BYTES)/64),.LATENCY(25+p*19),
   .WIN_BASE(WIN_BASE),.SPARSE(SPARSE),.SPARSE_MAP(SPARSE_MAP)) ram(
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
`include "tb/fpga/tnca_multicore_shared_hbm.vh"
`include "tb/fpga/tnca_multicore_dma.vh"
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
    integer bus_cycles;
    task bus_write;
        input [11:0] off;input [31:0] value;
        reg aw_done,w_done;
        begin
            @(negedge clk);awaddr=off;wdata=value;awvalid=1;wvalid=1;aw_done=0;w_done=0;bus_cycles=0;
            while(!aw_done || !w_done)begin
                @(posedge clk);if(awvalid && awready)aw_done=1;if(wvalid && wready)w_done=1;
                @(negedge clk);if(aw_done)awvalid=0;if(w_done)wvalid=0;
                bus_cycles=bus_cycles+1;if(bus_cycles>1000)$fatal(1,"AXI-Lite write admission timeout");
            end
            bready=1;while(!bvalid)begin @(negedge clk);bus_cycles=bus_cycles+1;if(bus_cycles>2000)$fatal(1,"AXI-Lite B timeout");end
            if(bresp!=0)$fatal(1,"AXI-Lite B error");@(posedge clk);@(negedge clk);bready=0;
        end
    endtask
    task bus_read;
        input [11:0] off;output [31:0] value;
        begin
            @(negedge clk);araddr=off;arvalid=1;bus_cycles=0;
            begin:wait_ar
                forever begin @(posedge clk);if(arready)disable wait_ar;bus_cycles=bus_cycles+1;if(bus_cycles>1000)$fatal(1,"AXI-Lite AR timeout");end
            end
            @(negedge clk);arvalid=0;rready=1;
            while(!rvalid)begin @(negedge clk);bus_cycles=bus_cycles+1;if(bus_cycles>2000)$fatal(1,"AXI-Lite R timeout");end
            if(rresp!=0)$fatal(1,"AXI-Lite R error");value=rdata;@(posedge clk);@(negedge clk);rready=0;
        end
    endtask
    reg [31:0] read_low,read_high;
    task run_batch;
        input [63:0] tid;
        begin
            nvalues = 0;
            journal_begin(tid);
            fh = $fopen(cmd_path, "r");
            if (fh != 0) begin
                // Skip the id line; it was read to get here.
                n = $fscanf(fh, "%d", arg_off);
                while ($fgets(line, fh) != 0) begin
                    if ($sscanf(line, "%s %d %d", kind, arg_off, arg_val) >= 2) begin
                        reg_addr = arg_off[ADDR_BITS-1:0];
                        if (kind == "w") begin
                            bus_write(arg_off[11:0],arg_val[31:0]);
                            bus_write(arg_off[11:0]+4,arg_val[63:32]);
                            if (do_trace)
                                $display("  w %0d %016h", arg_off, arg_val);
                        end else if (kind == "r") begin
                            bus_read(arg_off[11:0],read_low);
                            bus_read(arg_off[11:0]+4,read_high);
                            values[nvalues]={read_high,read_low};nvalues=nvalues+1;
                            if (do_trace)
                                $display("  r %0d -> %016h", arg_off,
                                         values[nvalues-1]);
                        end else if(kind=="b" || kind=="d" || kind=="B" || kind=="D")begin
                            if(!MAINTENANCE || arg_off>=MEMORY_BYTES || !dut.device.all_idle)
                                $fatal(1,"DMA attempted without quiescent supported memory");
                            dma_wide=(kind=="B" || kind=="D");
                            if(dma_wide && (arg_off[2:0]!=0 || arg_off>MEMORY_BYTES-8))$fatal(1,"unaligned/past-end BO word");
                            dma_offset=arg_off;dma_word=arg_val;#1;
                            if(dma_line<0)$fatal(1,"DMA at BO offset %h has no sparse backing",arg_off);
                            if(kind=="b" || kind=="B")begin
                                dma_byte=arg_val[7:0];dma_write=1;
                                #1;dma_write=0;
                            end else begin
                                values[nvalues]=dma_wide ? dma_read_word[dma_port*64+:64] : dma_read[dma_port*8+:8];nvalues=nvalues+1;
                            end
                        end else begin
                            $display("tnca_fpga_regs: unknown command %0s",
                                     kind);
                        end
                        if(kind=="w" || kind=="r")repeat (intercycle) @(posedge clk);
                        #1;
                    end
                end
                $fclose(fh);
            end

            journal_end(tid);
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
`include "tb/fpga/tnca_mmio_journal.vh"
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
        journal_init;

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

        journal_replay_run;
        wd = 0;
        while (wd < watchdog) begin
            repeat (pollcycles) @(posedge clk);
            wd = wd + pollcycles;
            #1;
            journal_poll_stop;
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

`define TNCA_PROBE_TOP dut.device.system.shell.u_system
`define TNCA_PROBE_START dut.device.cmd_valid && dut.device.cmd_ready
`define TNCA_PROBE_DONE dut.device.done_valid && dut.device.done_ready
`include "tb/fpga/tnca_stage_probes.vh"
`undef TNCA_PROBE_TOP
`undef TNCA_PROBE_START
`undef TNCA_PROBE_DONE
endmodule

`default_nettype wire
