//==============================================================================
// tnca_mesh -- NX x NY mesh of tnca_router, with the local ports exposed
//
// Node index is y*NX + x. Port convention inside a router:
//   0 local, 1 north (+y), 2 east (+x), 3 south (-y), 4 west (-x)
//
// Edge ports are tied off with valid low and ready high, which is safe because
// XY routing never produces a hop off the array: a flit's X hops finish before
// any Y hop, and both coordinates are in range by construction.
//==============================================================================

module tnca_mesh #(
    parameter NX        = 2,
    parameter NY        = 2,
    parameter XW        = 2,
    parameter YW        = 2,
    parameter PAYLOAD_W = 144,
    parameter NV        = 2,
    parameter DEPTH     = 4,
    parameter AW        = 2,
    parameter LINK_STAGES = 0
) (
    input  wire                                  clk,
    input  wire                                  rst_n,

    // local ports, flattened as (node*NV + vn)
    input  wire [NX*NY*NV-1:0]                   l_in_valid,
    output wire [NX*NY*NV-1:0]                   l_in_ready,
    input  wire [NX*NY*NV*(XW+YW+PAYLOAD_W)-1:0] l_in_flit,

    output wire [NX*NY*NV-1:0]                   l_out_valid,
    input  wire [NX*NY*NV-1:0]                   l_out_ready,
    output wire [NX*NY*NV*(XW+YW+PAYLOAD_W)-1:0] l_out_flit
);

    localparam FW = XW + YW + PAYLOAD_W;
    localparam N  = NX * NY;
    localparam NC = 5 * NV;

    localparam P_LOCAL = 0, P_NORTH = 1, P_EAST = 2, P_SOUTH = 3, P_WEST = 4;

    // per-router bundles
    wire [N*NC-1:0]     r_in_valid,  r_in_ready;
    wire [N*NC*FW-1:0]  r_in_flit;
    wire [N*NC-1:0]     r_out_valid, r_out_ready;
    wire [N*NC*FW-1:0]  r_out_flit;

    genvar x, y, v;

    generate
        for (y = 0; y < NY; y = y + 1) begin : g_y
            for (x = 0; x < NX; x = x + 1) begin : g_x
                localparam integer NODE = y*NX + x;

                tnca_router #(
                    .XW(XW), .YW(YW), .PAYLOAD_W(PAYLOAD_W), .NV(NV),
                    .DEPTH(DEPTH), .AW(AW), .X_HERE(x), .Y_HERE(y)
                ) u_r (
                    .clk(clk), .rst_n(rst_n),
                    .in_valid (r_in_valid [NODE*NC +: NC]),
                    .in_ready (r_in_ready [NODE*NC +: NC]),
                    .in_flit  (r_in_flit  [NODE*NC*FW +: NC*FW]),
                    .out_valid(r_out_valid[NODE*NC +: NC]),
                    .out_ready(r_out_ready[NODE*NC +: NC]),
                    .out_flit (r_out_flit [NODE*NC*FW +: NC*FW])
                );

                for (v = 0; v < NV; v = v + 1) begin : g_v
                    localparam integer LC = NODE*NC + P_LOCAL*NV + v;
                    localparam integer NC_ = NODE*NC + P_NORTH*NV + v;
                    localparam integer EC = NODE*NC + P_EAST*NV + v;
                    localparam integer SC = NODE*NC + P_SOUTH*NV + v;
                    localparam integer WC = NODE*NC + P_WEST*NV + v;

                    // ---- local ------------------------------------------
                    assign r_in_valid[LC] = l_in_valid[NODE*NV + v];
                    assign r_in_flit[LC*FW +: FW] = l_in_flit[(NODE*NV+v)*FW +: FW];
                    assign l_in_ready[NODE*NV + v] = r_in_ready[LC];

                    assign l_out_valid[NODE*NV + v] = r_out_valid[LC];
                    assign l_out_flit[(NODE*NV+v)*FW +: FW] = r_out_flit[LC*FW +: FW];
                    assign r_out_ready[LC] = l_out_ready[NODE*NV + v];

                    // ---- east / west --------------------------------------
                    if (x + 1 < NX) begin : g_e
                        localparam integer PEER = y*NX + (x+1);
                        localparam integer PW_ = PEER*NC + P_WEST*NV + v;
                        tnca_elastic_link #(.W(FW),.STAGES(LINK_STAGES)) link(
                            .clk(clk),.rst_n(rst_n),
                            .in_valid(r_out_valid[EC]),.in_ready(r_out_ready[EC]),.in_data(r_out_flit[EC*FW+:FW]),
                            .out_valid(r_in_valid[PW_]),.out_ready(r_in_ready[PW_]),.out_data(r_in_flit[PW_*FW+:FW]),.empty());
                    end else begin : g_e_edge
                        assign r_out_ready[EC] = 1'b1;
                    end

                    if (x == 0) begin : g_w_edge
                        assign r_in_valid[WC]         = 1'b0;
                        assign r_in_flit[WC*FW +: FW] = {FW{1'b0}};
                        assign r_out_ready[WC]        = 1'b1;
                    end else begin : g_w
                        localparam integer PEER = y*NX + (x-1);
                        localparam integer PE_ = PEER*NC + P_EAST*NV + v;
                        tnca_elastic_link #(.W(FW),.STAGES(LINK_STAGES)) link(
                            .clk(clk),.rst_n(rst_n),
                            .in_valid(r_out_valid[WC]),.in_ready(r_out_ready[WC]),.in_data(r_out_flit[WC*FW+:FW]),
                            .out_valid(r_in_valid[PE_]),.out_ready(r_in_ready[PE_]),.out_data(r_in_flit[PE_*FW+:FW]),.empty());
                    end

                    // ---- north / south ------------------------------------
                    if (y + 1 < NY) begin : g_n
                        localparam integer PEER = (y+1)*NX + x;
                        localparam integer PS_ = PEER*NC + P_SOUTH*NV + v;
                        tnca_elastic_link #(.W(FW),.STAGES(LINK_STAGES)) link(
                            .clk(clk),.rst_n(rst_n),
                            .in_valid(r_out_valid[NC_]),.in_ready(r_out_ready[NC_]),.in_data(r_out_flit[NC_*FW+:FW]),
                            .out_valid(r_in_valid[PS_]),.out_ready(r_in_ready[PS_]),.out_data(r_in_flit[PS_*FW+:FW]),.empty());
                    end else begin : g_n_edge
                        assign r_out_ready[NC_] = 1'b1;
                    end

                    if (y == 0) begin : g_s_edge
                        assign r_in_valid[SC]         = 1'b0;
                        assign r_in_flit[SC*FW +: FW] = {FW{1'b0}};
                        assign r_out_ready[SC]        = 1'b1;
                    end else begin : g_s
                        localparam integer PEER = (y-1)*NX + x;
                        localparam integer PN_ = PEER*NC + P_NORTH*NV + v;
                        tnca_elastic_link #(.W(FW),.STAGES(LINK_STAGES)) link(
                            .clk(clk),.rst_n(rst_n),
                            .in_valid(r_out_valid[SC]),.in_ready(r_out_ready[SC]),.in_data(r_out_flit[SC*FW+:FW]),
                            .out_valid(r_in_valid[PN_]),.out_ready(r_in_ready[PN_]),.out_data(r_in_flit[PN_*FW+:FW]),.empty());
                    end
                end
            end
        end
    endgenerate

endmodule
