//==============================================================================
// tnca_router -- 5-port mesh router, deterministic XY, N virtual networks
//
// Ports: 0 = local, 1 = north, 2 = east, 3 = south, 4 = west.
//
// VIRTUAL NETWORKS (arch overview sec 10.4)
//
// Requests and responses must not share buffers. Each virtual network here has
// its own input FIFO and its own arbiter at every output, so a backed-up
// request network cannot stop a response from making progress. That is the
// property the whole memory system's forward progress rests on: a slice can
// always drain a response, so the requests occupying it can always retire.
//
// XY routing is deadlock-free within a virtual network -- no cyclic channel
// dependency exists when every packet completes all its X hops before any Y hop
// (Dally & Seitz). But sec 10.4 is explicit that this is only half the problem:
// "无环路由只解决网络通道依赖的一部分问题；端点协议仍可能形成死锁". The endpoint
// rule that closes the other half lives in tnca_core_ni: a request is only
// injected when the requester already holds the resources to absorb its
// response.
//
// Single-flit packets. A real design serialises a wide packet over several
// flits and counts link bandwidth per hop (sec 10.3); doing that here would
// add a reassembly path without changing anything this stage is meant to
// establish, so it is Stage 4 work along with the performance model.
//==============================================================================

module tnca_router #(
    parameter XW        = 2,
    parameter YW        = 2,
    parameter PAYLOAD_W = 144,
    parameter NV        = 2,          // virtual networks
    parameter DEPTH     = 4,
    parameter AW        = 2,
    parameter X_HERE    = 0,
    parameter Y_HERE    = 0
) (
    input  wire                        clk,
    input  wire                        rst_n,

    // 5 ports x NV channels, flattened as (port*NV + vn)
    input  wire [5*NV-1:0]             in_valid,
    output wire [5*NV-1:0]             in_ready,
    input  wire [5*NV*(XW+YW+PAYLOAD_W)-1:0] in_flit,

    output wire [5*NV-1:0]             out_valid,
    input  wire [5*NV-1:0]             out_ready,
    output wire [5*NV*(XW+YW+PAYLOAD_W)-1:0] out_flit
);

    localparam FW = XW + YW + PAYLOAD_W;
    localparam NC = 5 * NV;

    localparam P_LOCAL = 0, P_NORTH = 1, P_EAST = 2, P_SOUTH = 3, P_WEST = 4;

    genvar p, v, q;

    // Every combinational block gets its OWN loop and index variables. Two
    // always @(*) blocks that share a module-level index each retrigger the
    // other through it, and the result is a zero-delay loop: simulation time
    // never advances and nothing prints.
    integer de_i;
    integer se_i, se_j, se_oc, se_k, se_cand;
    integer po_i, po_j, po_oc;

    //==========================================================================
    // Input FIFOs, one per (port, vn)
    //==========================================================================
    wire [NC-1:0]    f_valid;
    wire [NC-1:0]    f_ready;
    wire [FW-1:0]    f_data  [0:NC-1];

    generate
        for (p = 0; p < 5; p = p + 1) begin : g_port
            for (v = 0; v < NV; v = v + 1) begin : g_vn
                localparam C = p*NV + v;
                tnca_fifo #(.W(FW), .DEPTH(DEPTH), .AW(AW)) u_in (
                    .clk(clk), .rst_n(rst_n),
                    .in_valid(in_valid[C]), .in_ready(in_ready[C]),
                    .in_data(in_flit[C*FW +: FW]),
                    .out_valid(f_valid[C]), .out_ready(f_ready[C]),
                    .out_data(f_data[C]),
                    .count()
                );
            end
        end
    endgenerate

    //==========================================================================
    // Route computation: X first, then Y, then eject.
    //
    // flit = {dst_y, dst_x, payload}
    //==========================================================================
    wire [XW-1:0] dx [0:NC-1];
    wire [YW-1:0] dy [0:NC-1];
    generate
        for (q = 0; q < NC; q = q + 1) begin : g_dst
            assign dx[q] = f_data[q][PAYLOAD_W +: XW];
            assign dy[q] = f_data[q][PAYLOAD_W+XW +: YW];
        end
    endgenerate

    reg [2:0] dest [0:NC-1];
    always @(*) begin
        for (de_i = 0; de_i < NC; de_i = de_i + 1) begin
            if (dx[de_i] > X_HERE[XW-1:0])      dest[de_i] = P_EAST[2:0];
            else if (dx[de_i] < X_HERE[XW-1:0]) dest[de_i] = P_WEST[2:0];
            else if (dy[de_i] > Y_HERE[YW-1:0]) dest[de_i] = P_NORTH[2:0];
            else if (dy[de_i] < Y_HERE[YW-1:0]) dest[de_i] = P_SOUTH[2:0];
            else                                dest[de_i] = P_LOCAL[2:0];
        end
    end

    //==========================================================================
    // Output arbitration: round robin among the input channels of the same
    // virtual network that want this output port.
    //
    // `out_valid` is computed WITHOUT looking at `out_ready`. A valid that
    // depends on the downstream ready is a combinational path waiting to become
    // a loop the moment some consumer's ready depends on its valid; the input
    // FIFO here does not, but the discipline is not worth spending.
    //==========================================================================
    reg [2:0] rr [0:NC-1];          // per output channel: next input port to favour

    reg [2:0] sel_in [0:NC-1];
    reg       sel_ok [0:NC-1];

    always @(*) begin
        for (se_i = 0; se_i < 5; se_i = se_i + 1) begin      // output port
            for (se_j = 0; se_j < NV; se_j = se_j + 1) begin // virtual network
                se_oc = se_i*NV + se_j;
                sel_in[se_oc] = 3'd0;
                sel_ok[se_oc] = 1'b0;
                for (se_k = 4; se_k >= 0; se_k = se_k - 1) begin  // rotate
                    se_cand = (rr[se_oc] + se_k) % 5;
                    if (f_valid[se_cand*NV + se_j] &&
                        (dest[se_cand*NV + se_j] == se_i[2:0])) begin
                        sel_in[se_oc] = se_cand[2:0];
                        sel_ok[se_oc] = 1'b1;
                    end
                end
            end
        end
    end

    //==========================================================================
    // Outputs, and the pops they authorise.
    //
    // Each input channel routes to exactly one output port, so no two outputs
    // can select the same input and a granted flit is consumed exactly once.
    //==========================================================================
    reg [NC-1:0] pop;
    always @(*) begin
        pop = {NC{1'b0}};
        for (po_i = 0; po_i < 5; po_i = po_i + 1)
            for (po_j = 0; po_j < NV; po_j = po_j + 1) begin
                po_oc = po_i*NV + po_j;
                if (sel_ok[po_oc] && out_ready[po_oc])
                    pop[sel_in[po_oc]*NV + po_j] = 1'b1;
            end
    end

    generate
        for (q = 0; q < NC; q = q + 1) begin : g_out
            assign out_valid[q]         = sel_ok[q];
            assign out_flit[q*FW +: FW] = f_data[sel_in[q]*NV + (q % NV)];
            assign f_ready[q]           = pop[q];
        end
    endgenerate

    integer rr_i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (rr_i = 0; rr_i < NC; rr_i = rr_i + 1) rr[rr_i] <= 3'd0;
        end else begin
            for (rr_i = 0; rr_i < NC; rr_i = rr_i + 1)
                if (sel_ok[rr_i] && out_ready[rr_i])
                    rr[rr_i] <= (sel_in[rr_i] == 3'd4) ? 3'd0 : (sel_in[rr_i] + 3'd1);
        end
    end

endmodule
