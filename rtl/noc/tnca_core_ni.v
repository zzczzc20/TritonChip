//==============================================================================
// tnca_core_ni -- network interface on a core's memory port
//
// Turns the core's tagged word-granular memory requests into request flits
// addressed to the home slice, and returns response flits to the core.
//
// HOME SELECTION (arch overview sec 10.1, 10.2)
//
// home = H(physical line address), a pure function of the address and nothing
// else. Two different views of the same physical line must never compute
// different homes, which is why this takes no descriptor, no TensorID and no
// hint as input -- those are performance metadata and cannot be allowed to
// change where a line lives (sec 7.5).
//
// The hash interleaves at line granularity and folds two address bits in, so a
// stride that is a multiple of the slice count does not pile every access onto
// one slice. Page colouring and program-to-cluster affinity (sec 10.2) are
// placement policy and belong to the runtime; they change which physical pages
// a program gets, not this function.
//
// WHY THIS CANNOT DEADLOCK (sec 10.4, 10.5)
//
// XY routing removes cyclic channel dependencies inside a virtual network, but
// sec 10.4 is explicit that endpoint protocol can still deadlock. The rule that
// closes it here: the core's response path is unconditionally ready. The LSU
// reserved a beat-table slot before issuing, so the landing site for the
// response already exists and accepting it needs no further allocation
// (sec 10.5, rules 1 and 2). The response network therefore always drains,
// which lets the requests occupying the slices retire.
//==============================================================================
`include "tnca_noc_pkg.vh"

module tnca_core_ni #(
    parameter DATA_W = 64,
    parameter BYTE_W = DATA_W/8,
    parameter PAYLOAD_W = 216 + DATA_W-64 + BYTE_W-8,
    parameter XW      = 2,
    parameter YW      = 2,
    parameter NX      = 2,
    parameter NY      = 2,
    parameter TAG_W   = 5,
    parameter SRC_W   = 4,
    parameter MY_ID   = 0,
    parameter LINE_SH = 6            // log2(line bytes)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- core memory port ---------------------------------------------------
    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [63:0]              req_addr,
    input  wire                     req_we,
    input  wire [BYTE_W-1:0]        req_be,
    input  wire [DATA_W-1:0]        req_wdata,
    input  wire [TAG_W-1:0]         req_tag,
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [63:0]              req_cmp,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [DATA_W-1:0]        rsp_rdata,

    // ---- mesh local port ----------------------------------------------------
    output wire                                     net_in_valid_req,
    input  wire                                     net_in_ready_req,
    output wire [XW+YW+PAYLOAD_W-1:0]     net_in_flit_req,

    output wire                                     net_in_valid_rsp,
    input  wire                                     net_in_ready_rsp,
    output wire [XW+YW+PAYLOAD_W-1:0]     net_in_flit_rsp,

    input  wire                                     net_out_valid_req,
    output wire                                     net_out_ready_req,
    input  wire [XW+YW+PAYLOAD_W-1:0]     net_out_flit_req,

    input  wire                                     net_out_valid_rsp,
    output wire                                     net_out_ready_rsp,
    input  wire [XW+YW+PAYLOAD_W-1:0]     net_out_flit_rsp
);

    localparam PW = PAYLOAD_W;
    // Preserve every legacy bit position at DATA_W=64. Comparison operands
    // remain 64-bit scalar values; wide data/byte enables expand together.
    localparam BE_L=65, DATA_L=BE_L+BYTE_W, SRC_L=DATA_L+DATA_W;
    localparam TAG_L=SRC_L+4, AMO_L=TAG_L+5, AOP_L=AMO_L+1, CMP_L=AOP_L+4;

    // ---- home slice ---------------------------------------------------------
    wire [63:0] line = req_addr >> LINE_SH;
    // Keep the frozen 2x2 mapping bit-for-bit. Other topologies reduce a
    // bounded byte hash, rather than dividing a 64-bit address in the hot path.
    wire [7:0] hash = line[7:0] ^ line[15:8] ^ line[23:16] ^ line[31:24]
                    ^ line[39:32] ^ line[47:40] ^ line[55:48] ^ line[63:56];
    wire [7:0] home = (NX==2 && NY==2) ? {6'd0,(line[1:0] ^ line[3:2])}
                                                    : hash % (NX*NY);
    wire [XW-1:0] home_x = home % NX;
    wire [YW-1:0] home_y = home / NX;

    // ---- request injection --------------------------------------------------
    wire [PW-1:0] req_payload;
    assign req_payload[`TNCA_REQ_ADDR]  = req_addr;
    assign req_payload[`TNCA_REQ_WE]    = req_we;
    assign req_payload[BE_L+:BYTE_W]    = req_be;
    assign req_payload[DATA_L+:DATA_W] = req_wdata;
    assign req_payload[SRC_L+:4]   = MY_ID[3:0];
    assign req_payload[TAG_L+:5]   = req_tag;
    assign req_payload[AMO_L]   = req_amo;
    assign req_payload[AOP_L+:4]   = req_aop;
    assign req_payload[CMP_L+:64]   = req_cmp;
    assign req_payload[PW-1]            = 1'b0;

    assign net_in_valid_req = req_valid;
    assign net_in_flit_req  = {home_y, home_x, req_payload};
    assign req_ready        = net_in_ready_req;

    // A core never injects on the response network.
    assign net_in_valid_rsp = 1'b0;
    assign net_in_flit_rsp  = {(XW+YW+PW){1'b0}};

    // A core is not a destination for requests.
    assign net_out_ready_req = 1'b1;

    // ---- response delivery --------------------------------------------------
    // Unconditionally ready: this is the endpoint rule the network's freedom
    // from deadlock rests on.
    assign net_out_ready_rsp = 1'b1;

    always @(posedge clk) begin
        if (!rst_n) begin
            rsp_valid <= 1'b0;
        end else begin
            rsp_valid <= net_out_valid_rsp;
            rsp_tag   <= net_out_flit_rsp[DATA_W+:5];
            rsp_rdata <= net_out_flit_rsp[0+:DATA_W];
        end
    end

    wire _unused = &{1'b0, net_out_valid_req, net_out_flit_req, 1'b0};

`ifndef SYNTHESIS
    initial if ((DATA_W!=64 && DATA_W!=128 && DATA_W!=256 && DATA_W!=512)
                || BYTE_W!=DATA_W/8 || PAYLOAD_W!=216+DATA_W-64+BYTE_W-8)
        $fatal(1,"invalid NI data/mask/payload geometry");
`endif

endmodule
