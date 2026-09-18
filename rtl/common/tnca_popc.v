//==============================================================================
// tnca_popc -- population count, balanced adder tree
//
// Written as `for (i...) cnt = cnt + v[i];` this is a chain of W adders. The
// vector unit instantiates one per lane, so at VLEN=16 that was sixteen 32-deep
// carry chains sitting in parallel with everything else in the lane.
//
// Depth here is LOGW levels of narrow addition.
//==============================================================================

module tnca_popc #(
    parameter W    = 64,
    parameter LOGW = 6            // ceil(log2(W))
) (
    input  wire [W-1:0]  v,
    output wire [LOGW:0] count
);

    localparam PW = 1 << LOGW;
    localparam NW = LOGW + 1;     // every node is full width; the tool trims it

    // tree[lvl][idx] flattened: level lvl holds PW >> lvl live nodes.
    wire [(LOGW+1)*PW*NW-1:0] tree;

    genvar lvl, idx;
    generate
        for (idx = 0; idx < PW; idx = idx + 1) begin : g_leaf
            if (idx < W) begin : g_real
                assign tree[(0*PW + idx)*NW +: NW] = {{(NW-1){1'b0}}, v[idx]};
            end else begin : g_pad
                assign tree[(0*PW + idx)*NW +: NW] = {NW{1'b0}};
            end
        end
        for (lvl = 0; lvl < LOGW; lvl = lvl + 1) begin : g_lvl
            for (idx = 0; idx < (PW >> (lvl + 1)); idx = idx + 1) begin : g_node
                assign tree[((lvl+1)*PW + idx)*NW +: NW] =
                       tree[(lvl*PW + 2*idx)*NW     +: NW] +
                       tree[(lvl*PW + 2*idx + 1)*NW +: NW];
            end
        end
    endgenerate

    assign count = tree[(LOGW*PW)*NW +: NW];

endmodule
