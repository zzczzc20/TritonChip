//==============================================================================
// tnca_router_fast -- tnca_router with the same ports and the same cycle
// behaviour, restructured for timing (R42 clk2, CLKFIX2 bit 11).
//
// What is different from tnca_router, and why it costs no cycle:
//
//  * The route of a flit (one-hot output port) is computed from the flit's
//    destination when it is pushed into the input FIFO and stored beside it,
//    instead of being recomputed from the FIFO output every cycle. (Not for the
//    local injection port: see g_loc.) The compare
//    sits on the push side (upstream link FIFO read -> compare -> FIFO write
//    data), not in front of the arbiter. The value is a pure function of the
//    flit, so the arbiter sees the same route in the same cycle.
//  * Occupancy flags are registered copies of (cnt != 0) and (cnt == DEPTH),
//    computed from the next count. They equal the combinational flags of
//    tnca_fifo in every cycle.
//  * The arbiter works on one-hot request / grant vectors with a one-hot
//    round-robin pointer. The priority order is the original one: rr, rr+1,
//    ..., rr+4 (mod 5), first requester wins; rr becomes (granted input + 1)
//    mod 5 when a granted flit is accepted by the output.
//  * out_flit is an AND-OR of the five input FIFO outputs under the one-hot
//    grant. While out_valid is low out_flit is zero (tnca_router drove input
//    port 0's head); no consumer looks at the payload without valid.
//
// tb/noc/tb_router_miter.v drives both routers with the same random traffic
// and compares the handshake and the flit of every transfer cycle by cycle.
//==============================================================================

module tnca_rfifo #(
    parameter W     = 32,
    parameter DEPTH = 4,
    parameter AW    = 2
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          in_valid,
    output wire          in_ready,
    input  wire [W-1:0]  in_data,
    output wire          out_valid,
    input  wire          out_ready,
    output wire [W-1:0]  out_data
);
    reg [W-1:0]  mem [0:DEPTH-1];
    reg [AW-1:0] rptr, wptr;
    reg [AW:0]   cnt;
    reg          nz_q, full_q;

    assign in_ready  = !full_q;
    assign out_valid = nz_q;
    assign out_data  = mem[rptr];

    wire push = in_valid  && !full_q;
    wire pop  = nz_q && out_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            rptr <= {AW{1'b0}};
            wptr <= {AW{1'b0}};
            cnt  <= {(AW+1){1'b0}};
            nz_q <= 1'b0;
            full_q <= 1'b0;
        end else begin
            if (push) begin
                mem[wptr] <= in_data;
                wptr <= (wptr == DEPTH-1) ? {AW{1'b0}} : (wptr + 1'b1);
            end
            if (pop)
                rptr <= (rptr == DEPTH-1) ? {AW{1'b0}} : (rptr + 1'b1);
            // same case structure as tnca_fifo (an unknown push or pop changes nothing)
            case ({push, pop})
            2'b10: begin cnt <= cnt + 1'b1; nz_q <= 1'b1; full_q <= (cnt + 1'b1 == DEPTH); end
            2'b01: begin cnt <= cnt - 1'b1; nz_q <= (cnt != 1);  full_q <= 1'b0; end
            default: ;
            endcase
        end
    end
endmodule

module tnca_router_fast #(
    parameter XW        = 2,
    parameter YW        = 2,
    parameter PAYLOAD_W = 144,
    parameter NV        = 2,
    parameter DEPTH     = 4,
    parameter AW        = 2,
    parameter X_HERE    = 0,
    parameter Y_HERE    = 0
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire [5*NV-1:0]             in_valid,
    output wire [5*NV-1:0]             in_ready,
    input  wire [5*NV*(XW+YW+PAYLOAD_W)-1:0] in_flit,
    output wire [5*NV-1:0]             out_valid,
    input  wire [5*NV-1:0]             out_ready,
    output wire [5*NV*(XW+YW+PAYLOAD_W)-1:0] out_flit
);
    localparam FW = XW + YW + PAYLOAD_W;
    localparam NC = 5 * NV;
    // output port order: 0 local, 1 north, 2 east, 3 south, 4 west

    // One-hot route of a flit at this router (same decision as tnca_router).
    function [4:0] route_oh;
        input [FW-1:0] fl;
        reg [XW-1:0] dx;
        reg [YW-1:0] dy;
        begin
            dx = fl[PAYLOAD_W +: XW];
            dy = fl[PAYLOAD_W+XW +: YW];
            if (dx > X_HERE[XW-1:0])      route_oh = 5'b00100;   // east
            else if (dx < X_HERE[XW-1:0]) route_oh = 5'b10000;   // west
            else if (dy > Y_HERE[YW-1:0]) route_oh = 5'b00010;   // north
            else if (dy < Y_HERE[YW-1:0]) route_oh = 5'b01000;   // south
            else                          route_oh = 5'b00001;   // local
        end
    endfunction

    // First requester in the rotated order rr, rr+1, ..., rr+4 (mod 5).
    function [4:0] arb5;
        input [4:0] req;
        input [4:0] rr;
        integer i, r, k, d;
        reg blk;
        begin
            arb5 = 5'b0;
            for (i = 0; i < 5; i = i + 1) begin
                blk = 1'b0;
                for (r = 0; r < 5; r = r + 1)
                    if (rr[r]) begin
                        d = (i - r + 5) % 5;
                        for (k = 0; k < d; k = k + 1)
                            if (req[(r + k) % 5]) blk = 1'b1;
                    end
                arb5[i] = req[i] & ~blk;
            end
        end
    endfunction

    genvar p, v, o, c;

    wire [NC-1:0]  f_valid;
    wire [NC-1:0]  f_ready;
    wire [FW-1:0]  f_data  [0:NC-1];
    wire [4:0]     f_dst   [0:NC-1];

    generate
        for (p = 0; p < 5; p = p + 1) begin : g_port
            for (v = 0; v < NV; v = v + 1) begin : g_vn
                localparam C = p*NV + v;
                if (p == 0) begin : g_loc
                    // Local injection port: its flit comes combinationally from the tile (address -> home slice hash -> destination
                    // fields), the longest path into the router, so the route is NOT computed on the push side here; it is computed
                    // from the FIFO output like in tnca_router.
                    wire [FW-1:0] q;
                    tnca_rfifo #(.W(FW), .DEPTH(DEPTH), .AW(AW)) u_in (
                        .clk(clk), .rst_n(rst_n),
                        .in_valid(in_valid[C]), .in_ready(in_ready[C]),
                        .in_data(in_flit[C*FW +: FW]),
                        .out_valid(f_valid[C]), .out_ready(f_ready[C]),
                        .out_data(q));
                    assign f_data[C] = q;
                    assign f_dst[C]  = route_oh(q);
                end else begin : g_lnk
                    // Link input ports: the flit comes from the neighbour's link FIFO read, a short path; the route is stored.
                    wire [FW+4:0] q;
                    tnca_rfifo #(.W(FW+5), .DEPTH(DEPTH), .AW(AW)) u_in (
                        .clk(clk), .rst_n(rst_n),
                        .in_valid(in_valid[C]), .in_ready(in_ready[C]),
                        .in_data({route_oh(in_flit[C*FW +: FW]), in_flit[C*FW +: FW]}),
                        .out_valid(f_valid[C]), .out_ready(f_ready[C]),
                        .out_data(q));
                    assign f_data[C] = q[FW-1:0];
                    assign f_dst[C]  = q[FW+4:FW];
                end
            end
        end
    endgenerate

    // Per (output port, vn): grant, indexed by input port.
    wire [4:0] gnt   [0:NC-1];
    wire [NC-1:0] fire;

    generate
        for (o = 0; o < 5; o = o + 1) begin : g_o
            for (v = 0; v < NV; v = v + 1) begin : g_v
                localparam OC = o*NV + v;
                wire [4:0] rq;
                reg  [4:0] rr_q;
                for (c = 0; c < 5; c = c + 1) begin : g_rq
                    assign rq[c] = f_valid[c*NV + v] & f_dst[c*NV + v][o];
                end
                assign gnt[OC] = arb5(rq, rr_q);
                assign out_valid[OC] = |rq;
                assign fire[OC] = (|rq) & out_ready[OC];

                // AND-OR output mux under the one-hot grant
                wire [FW-1:0] m0 = {FW{gnt[OC][0]}} & f_data[0*NV + v];
                wire [FW-1:0] m1 = {FW{gnt[OC][1]}} & f_data[1*NV + v];
                wire [FW-1:0] m2 = {FW{gnt[OC][2]}} & f_data[2*NV + v];
                wire [FW-1:0] m3 = {FW{gnt[OC][3]}} & f_data[3*NV + v];
                wire [FW-1:0] m4 = {FW{gnt[OC][4]}} & f_data[4*NV + v];
                assign out_flit[OC*FW +: FW] = m0 | m1 | m2 | m3 | m4;

                always @(posedge clk) begin
                    if (!rst_n) rr_q <= 5'b00001;
                    else if (fire[OC]) rr_q <= {gnt[OC][3:0], gnt[OC][4]};
                end
            end
        end
        // pop of input channel (i, v): its single granted output accepts
        for (p = 0; p < 5; p = p + 1) begin : g_pop
            for (v = 0; v < NV; v = v + 1) begin : g_pv
                assign f_ready[p*NV + v] =
                    (gnt[0*NV + v][p] & out_ready[0*NV + v]) |
                    (gnt[1*NV + v][p] & out_ready[1*NV + v]) |
                    (gnt[2*NV + v][p] & out_ready[2*NV + v]) |
                    (gnt[3*NV + v][p] & out_ready[3*NV + v]) |
                    (gnt[4*NV + v][p] & out_ready[4*NV + v]);
            end
        end
    endgenerate
endmodule
