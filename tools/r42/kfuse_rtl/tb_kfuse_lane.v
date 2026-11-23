// tb_kfuse_lane -- bit-exact bench of rtl/core/tnca_kfuse_lane.v against vectors from
// tools/r42/kfuse_rtl_vectors.py (the model tools/r42/kfuse_model.py, variant sum_fma, trunc).
//
//   vvp ... +VEC=file.hex +N=count [+BUBBLE=pct] [+MAXERR=n]
//
// Word format (one hex word per vector): {y[31:0], c[31:0], b[8G-1:0], a[8G-1:0]}.
// One vector is issued per cycle (with BUBBLE percent idle cycles, data garbage and valid_in = 0 in them);
// the result of a vector is compared when valid_out shows it, LAT cycles later. C_LATE = 1 drives c late.
`timescale 1ns/1ps
module tb_kfuse_lane;
    parameter integer G         = 8;
    parameter integer W         = 0;
    parameter [7:0]   CUTS      = 8'b00111011;
    parameter integer IN_REG    = 1;
    parameter integer C_LATE    = 0;
    parameter integer SIGN_MODE = 1;
    parameter integer TREE      = 0;
    parameter integer NMAX      = 1100000;

    localparam integer VW = 16*G + 64;
    reg [VW-1:0] mem [0:NMAX-1];

    reg clk = 0;
    reg rst_n = 0;
    reg valid_in = 0;
    reg [8*G-1:0] a = 0, b = 0;
    reg [31:0] c = 0;
    wire [31:0] y;
    wire valid_out;

`ifdef NETLIST
    tnca_kfuse_lane dut (
`else
    tnca_kfuse_lane #(.G(G), .W(W), .CUTS(CUTS), .IN_REG(IN_REG), .C_LATE(C_LATE),
                      .SIGN_MODE(SIGN_MODE), .TREE(TREE)) dut (
`endif
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a(a), .b(b), .c(c), .y(y), .valid_out(valid_out));

    always #5 clk = ~clk;

    integer lat, latf;
    integer n, nvec, bubble, maxerr;
    integer errs, checked, issued, cyc;
    reg [1023:0] vecfile;
    integer seed;
    reg [VW-1:0] w;
    reg [31:0] y_exp;
    integer idx;

    // issue schedule: iss[cycle mod 4096] = vector index + 1 (0 = bubble); c for C_LATE mode
    integer iss [0:4095];
    reg [31:0] cq [0:4095];

    // C_LATE = 1: c comes from a register on the lane's clock (the accumulator file's read), changing with the
    // group sum's arrival at the align stage: at posedge m it takes the c of the group issued m - latf + 1 cycles ago.
    integer pm, ps;
    always @(posedge clk) if (C_LATE != 0 && rst_n) begin
        pm = cyc - 1;
        ps = pm - latf + 1;
        c <= (ps >= 0 && iss[ps % 4096] != 0) ? cq[ps % 4096] : $urandom(seed);
    end

    initial begin
        if (!$value$plusargs("VEC=%s", vecfile)) begin
            $display("ERROR: +VEC missing"); $finish;
        end
        if (!$value$plusargs("N=%d", nvec)) nvec = NMAX;
        if (!$value$plusargs("BUBBLE=%d", bubble)) bubble = 0;
        if (!$value$plusargs("MAXERR=%d", maxerr)) maxerr = 100;
        if (!$value$plusargs("SEED=%d", seed)) seed = 12345;
        $readmemh(vecfile, mem, 0, nvec - 1);
`ifdef NETLIST
        // post-synthesis netlist: the lane's parameters are gone, the latency is given
        if (!$value$plusargs("LAT=%d", lat) || !$value$plusargs("LATF=%d", latf)) begin
            $display("ERROR: +LAT and +LATF are needed for a netlist"); $finish;
        end
`else
        lat  = dut.LAT;
        latf = dut.LAT_FRONT;
`endif
        $display("tb_kfuse_lane G=%0d W=%0d CUTS=%b IN_REG=%0d C_LATE=%0d SIGN_MODE=%0d TREE=%0d LAT=%0d LAT_FRONT=%0d N=%0d bubble=%0d%%",
                 G, W, CUTS, IN_REG, C_LATE, SIGN_MODE, TREE, lat, latf, nvec, bubble);
        errs = 0; checked = 0; issued = 0; cyc = 0;
        for (n = 0; n < 4096; n = n + 1) begin iss[n] = 0; cq[n] = 0; end
`ifdef NETLIST
        #250;   // a netlist run: glbl's GSR pulse (100 ns) holds every flop at INIT until then
`endif
        repeat (4) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        // main loop: at every negedge, compare what the lane shows, then drive the next cycle
        while (checked < nvec) begin
            // compare: the vector issued LAT cycles ago
            if (cyc >= lat) begin
                idx = iss[(cyc - lat) % 4096];
                if (idx != 0) begin
                    w = mem[idx - 1];
                    y_exp = w[VW-1 -: 32];
                    checked = checked + 1;
                    if (valid_out !== 1'b1 || y !== y_exp) begin
                        errs = errs + 1;
                        if (errs <= maxerr)
                            $display("MISMATCH vec=%0d a=%0h b=%0h c=%08h got=%08h exp=%08h valid=%b",
                                     idx - 1, w[8*G-1:0], w[16*G-1:8*G], w[16*G+31:16*G], y, y_exp, valid_out);
                    end
                end else if (valid_out !== 1'b0) begin
                    errs = errs + 1;
                    if (errs <= maxerr) $display("MISMATCH valid_out high in a bubble cycle %0d", cyc);
                end
            end
            // drive
            if (issued < nvec && ($urandom(seed) % 100) >= bubble) begin
                w = mem[issued];
                a <= w[8*G-1:0];
                b <= w[16*G-1:8*G];
                valid_in <= 1'b1;
                iss[cyc % 4096] = issued + 1;
                cq[cyc % 4096] = w[16*G+31:16*G];
                issued = issued + 1;
            end else begin
                a <= {$urandom(seed), $urandom(seed), $urandom(seed), $urandom(seed),
                      $urandom(seed), $urandom(seed), $urandom(seed), $urandom(seed)};
                b <= {$urandom(seed), $urandom(seed), $urandom(seed), $urandom(seed),
                      $urandom(seed), $urandom(seed), $urandom(seed), $urandom(seed)};
                valid_in <= 1'b0;
                iss[cyc % 4096] = 0;
                cq[cyc % 4096] = $urandom(seed);
            end
            if (C_LATE == 0)
                c <= (iss[cyc % 4096] != 0) ? cq[cyc % 4096] : $urandom(seed);
            cyc = cyc + 1;
            @(negedge clk);
        end
        $display("RESULT G=%0d W=%0d total=%0d mismatches=%0d", G, W, checked, errs);
        $finish;
    end
endmodule
