//==============================================================================
// tb_mesh -- randomised all-to-all test of the 2x2 mesh
//
// Every node injects packets to random destinations on random virtual networks
// and every node checks what arrives. Each packet carries its source, its
// destination, its virtual network and a per-source sequence number, so the
// test can assert three things that a "it seems to work" test cannot:
//
//   delivery      every injected packet arrives, exactly once
//   correctness   at the destination it named, on the VN it was sent on, with
//                 its payload intact
//   ordering      packets from one source to one destination on one VN arrive
//                 in the order they were sent -- deterministic XY routing has
//                 no other legal behaviour, and if this ever fails the routing
//                 has become adaptive by accident
//
// Receivers apply random back-pressure, because a network that is only ever
// tested against an always-ready sink has not been tested.
//==============================================================================
`timescale 1ns/1ps

module tb_mesh;

    localparam NX = 2, NY = 2, N = NX*NY;
    localparam XW = 2, YW = 2;
    localparam NV = 2;
    localparam PAYLOAD_W = 64;
    localparam FW = XW + YW + PAYLOAD_W;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    reg  [N*NV-1:0]     li_valid;
    wire [N*NV-1:0]     li_ready;
    reg  [N*NV*FW-1:0]  li_flit;
    wire [N*NV-1:0]     lo_valid;
    wire [N*NV-1:0]     lo_ready;
    wire [N*NV*FW-1:0]  lo_flit;

    tnca_mesh #(
        .NX(NX), .NY(NY), .XW(XW), .YW(YW),
        .PAYLOAD_W(PAYLOAD_W), .NV(NV), .DEPTH(4), .AW(2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .l_in_valid(li_valid), .l_in_ready(li_ready), .l_in_flit(li_flit),
        .l_out_valid(lo_valid), .l_out_ready(lo_ready), .l_out_flit(lo_flit)
    );

    // payload = {seq[15:0], vn[3:0], dst[3:0], src[3:0], magic[35:0]}
    localparam [35:0] MAGIC = 36'h5A5_1234_9;

    integer sent [0:N-1][0:N-1][0:NV-1];   // next sequence number
    integer got  [0:N-1][0:N-1][0:NV-1];   // next expected sequence number
    integer n_sent, n_recv, fails, npkt;

    integer i, s, d, v, k;
    reg [FW-1:0] f;
    reg [63:0] rnd;
    reg [31:0] lfsr;

    // ---- seed, trace and coverage -------------------------------------------
    // R1-08 REG-07: the seed has to reach the PRNG that chooses destinations
    // and back-pressure, not merely appear in a log line, and the run has to
    // report what it actually exercised. Twenty thousand packets that never
    // met back-pressure are not a test of a network under back-pressure.
    integer seed, seed0;
    reg [63:0] trace_hash;
    integer pair_count [0:N-1][0:N-1][0:NV-1];   // src x dst x vn, injected
    integer n_pairs, n_backpressure, n_stall_cycles;

    function dx_of; input integer nd; dx_of = nd[0]; endfunction
    function dy_of; input integer nd; dy_of = nd[1]; endfunction

    // ---- receivers -----------------------------------------------------------
    integer rs, rd, rv, rseq;
    always @(posedge clk) begin
        if (rst_n) begin
            for (i = 0; i < N*NV; i = i + 1) begin
                if (lo_valid[i] && lo_ready[i]) begin
                    f    = lo_flit[i*FW +: FW];
                    rs   = f[36 +: 4];
                    rd   = f[40 +: 4];
                    rv   = f[44 +: 4];
                    rseq = f[48 +: 16];
                    n_recv = n_recv + 1;
                    if (f[35:0] !== MAGIC) begin
                        fails = fails + 1;
                        if (fails < 10) $display("FAIL corrupt payload at node %0d", i/NV);
                    end
                    if (rd !== (i/NV)) begin
                        fails = fails + 1;
                        if (fails < 10)
                            $display("FAIL misrouted: dst=%0d arrived at %0d", rd, i/NV);
                    end
                    if (rv !== (i%NV)) begin
                        fails = fails + 1;
                        if (fails < 10)
                            $display("FAIL wrong VN: sent on %0d arrived on %0d", rv, i%NV);
                    end
                    if (rseq !== got[rs][rd][rv]) begin
                        fails = fails + 1;
                        if (fails < 10)
                            $display("FAIL order src=%0d dst=%0d vn=%0d got seq %0d want %0d",
                                     rs, rd, rv, rseq, got[rs][rd][rv]);
                    end
                    got[rs][rd][rv] = got[rs][rd][rv] + 1;
                end
            end
        end
    end

    // ---- random back-pressure on every receiver ------------------------------
    // Continuous assignments, not an always @(*) with a shared loop variable:
    // that block re-triggers on its own loop counter and spins at time zero.
    always @(posedge clk) begin
        if (!rst_n) lfsr <= lfsr_init;
        else        lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
    end

    // The back-pressure pattern is part of the stimulus, so it moves with the
    // seed too. A fixed LFSR seed meant every "seed" met the same sequence of
    // stalls, which is the half of the traffic that decides whether a buffer
    // ever fills.
    reg [31:0] lfsr_init;

    // Count the cycles a receiver had something to take and refused it. If
    // this is zero the back-pressure never happened, whatever the LFSR did.
    integer bi;
    always @(posedge clk) begin
        if (rst_n) begin
            for (bi = 0; bi < N*NV; bi = bi + 1)
                if (lo_valid[bi] && !lo_ready[bi])
                    n_stall_cycles = n_stall_cycles + 1;
        end
    end
    genvar gi;
    generate
        for (gi = 0; gi < N*NV; gi = gi + 1) begin : g_bp
            assign lo_ready[gi] = lfsr[gi % 20] | lfsr[(gi+7) % 20];   // ~75% ready
        end
    endgenerate

    // ---- watchdog ------------------------------------------------------------
    integer wd;
    initial begin
        wd = 0;
        forever begin
            @(posedge clk);
            wd = wd + 1;
            if (wd > 500000) begin
                $display("FAIL: watchdog -- sent=%0d recv=%0d", n_sent, n_recv);
                $finish;
            end
        end
    end

    // ---- injectors -----------------------------------------------------------
    initial begin
        n_sent = 0; n_recv = 0; fails = 0;
        li_valid = {(N*NV){1'b0}};
        for (s = 0; s < N; s = s + 1)
            for (d = 0; d < N; d = d + 1)
                for (v = 0; v < NV; v = v + 1) begin
                    sent[s][d][v] = 0;
                    got[s][d][v]  = 0;
                end

        if (!$value$plusargs("npkt=%d", npkt)) npkt = 4000;
        if (!$value$plusargs("seed=%d", seed)) seed = 20260918;
        seed0 = seed;
        trace_hash = 64'hcbf29ce484222325;
        n_backpressure = 0; n_stall_cycles = 0;
        // Derive the back-pressure LFSR from the seed, never zero (an
        // all-zero LFSR is a constant, so the receivers would never stall).
        lfsr_init = (seed ^ 32'hCAFE_0001) | 32'd1;
        for (s = 0; s < N; s = s + 1)
            for (d = 0; d < N; d = d + 1)
                for (v = 0; v < NV; v = v + 1)
                    pair_count[s][d][v] = 0;

        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;

        while (n_sent < npkt) begin
            @(posedge clk);
            #1;
            for (i = 0; i < N*NV; i = i + 1) begin
                if (li_valid[i] && li_ready[i]) li_valid[i] = 1'b0;
            end
            for (i = 0; i < N*NV; i = i + 1) begin
                if (!li_valid[i] && (n_sent < npkt)) begin
                    rnd = {$random(seed), $random(seed)};
                    if (rnd[2:0] != 3'd0) begin      // inject most cycles
                        s = i / NV;
                        v = i % NV;
                        d = rnd[9:8] % N;
                        // flit = {dst_y[YW-1:0], dst_x[XW-1:0], payload[63:0]}
                        // For a 2x2 mesh node d: x = d[0], y = d[1].
                        f = { {(YW-1){1'b0}}, dy_of(d),
                              {(XW-1){1'b0}}, dx_of(d),
                              sent[s][d][v][15:0], v[3:0], d[3:0], s[3:0], MAGIC };
                        li_flit[i*FW +: FW] = f;
                        li_valid[i] = 1'b1;
                        sent[s][d][v] = sent[s][d][v] + 1;
                        n_sent = n_sent + 1;
                        pair_count[s][d][v] = pair_count[s][d][v] + 1;
                        trace_hash = (trace_hash ^ {52'd0, s[3:0], d[3:0], v[3:0]})
                                     * 64'd1099511628211;
                    end
                end
            end
        end
        // let the last presented flits be taken
        while (|li_valid) begin
            @(posedge clk); #1;
            for (i = 0; i < N*NV; i = i + 1)
                if (li_valid[i] && li_ready[i]) li_valid[i] = 1'b0;
        end

        // drain
        k = 0;
        while ((n_recv < n_sent) && (k < 20000)) begin
            @(posedge clk); #1;
            k = k + 1;
        end

        n_pairs = 0;
        for (s = 0; s < N; s = s + 1)
            for (d = 0; d < N; d = d + 1)
                for (v = 0; v < NV; v = v + 1)
                    if (pair_count[s][d][v] > 0) n_pairs = n_pairs + 1;

        $display("---- mesh %0dx%0d, %0d VNs ----", NX, NY, NV);
        $display("seed=%0d trace_hash=%016h", seed0, trace_hash);
        $display("sent=%0d received=%0d pairs_covered=%0d/%0d backpressure_cycles=%0d",
                 n_sent, n_recv, n_pairs, N*N*NV, n_stall_cycles);

        // Coverage, asserted rather than printed. "Twenty thousand packets"
        // says nothing about whether every source reached every destination
        // on both virtual networks, or whether a receiver ever pushed back.
        if (n_pairs != N*N*NV) begin
            fails = fails + 1;
            $display("FAIL: only %0d of %0d source/destination/VN combinations were used",
                     n_pairs, N*N*NV);
        end
        if (n_stall_cycles == 0) begin
            fails = fails + 1;
            $display("FAIL: no receiver ever applied back-pressure");
        end
        if (n_recv != n_sent) begin
            fails = fails + 1;
            $display("FAIL: %0d packets lost", n_sent - n_recv);
        end
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

endmodule
