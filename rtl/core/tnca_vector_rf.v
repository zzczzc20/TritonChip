//==============================================================================
// tnca_vector_rf -- vector register file, PHYS_VREGS x (VLEN * ELEN) bit
//
// The physical file is a core-wide pool and each RCTX sees VREGS_PER_RCTX
// logical names into it through a PROGRAMMABLE base, set at launch (arch
// overview sec 5.1: "物理 RF 在 launch 时静态分配").
//
// The base has to be programmable, not derived from the RCTX id. With
// `phys = rctx * VREGS_PER_RCTX + idx` two role contexts can never name the
// same physical register, so a producer role cannot hand a tile to a consumer
// role through a shared RF window -- and that is the preferred cross-role
// communication path for tensors. Every transfer would have to go out to the
// cache and back, which is the cost the whole design exists to avoid.
//
// Equal bases alias two contexts completely; disjoint bases reproduce the old
// behaviour. The kernel manifest states the mapping, so the compiler and the
// hardware agree on which logical name in which context is which physical
// register (doc/compiler_overview.md sec 13.1).
//
// Per-lane write enables implement the masked-write rule (ISA sec 2.2) without
// a read-modify-write through the datapath.
//
// Reads are asynchronous and write-first, matching tnca_scalar_rf.
//
// NOTE ON AREA: at the reference configuration (PHYS_VREGS=512, VLEN=16,
// ELEN=32) this is 32 KiB of state with 4 read ports -- three for the execute
// stage and one for the asynchronous engines. A flop-based
// implementation is only appropriate for simulation and small FPGA bring-up;
// silicon needs banked SRAM with an operand-collector front end (arch overview
// sec 5.3). The banked variant is tracked as Stage-2 work.
//==============================================================================
`include "tnca_defs.vh"

module tnca_vector_rf #(
    parameter PHYS_VREGS     = 128,
    parameter VREGS_PER_RCTX = 64,
    parameter NUM_RCTX       = 2,
    parameter VLEN           = 16,
    parameter ELEN           = 32,
    parameter RCTX_W         = 1,
    parameter PHYS_W         = 7     // $clog2(PHYS_VREGS)
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // per-RCTX physical base, flattened
    input  wire [NUM_RCTX*PHYS_W-1:0] vbase,

    input  wire [RCTX_W-1:0]         ra_rctx,
    input  wire [7:0]                ra_idx,
    output wire [VLEN*ELEN-1:0]      ra_data,

    input  wire [RCTX_W-1:0]         rb_rctx,
    input  wire [7:0]                rb_idx,
    output wire [VLEN*ELEN-1:0]      rb_data,

    input  wire [RCTX_W-1:0]         rc_rctx,
    input  wire [7:0]                rc_idx,
    output wire [VLEN*ELEN-1:0]      rc_data,

    // Fourth read port, for the matrix and transfer engines. They run
    // asynchronously alongside the pipeline, so sharing an operand port with
    // the execute stage would make every asynchronous command a source of
    // pipeline stalls -- which is the opposite of what it is for.
    input  wire [RCTX_W-1:0]         rd_rctx,
    input  wire [7:0]                rd_idx,
    output wire [VLEN*ELEN-1:0]      rd_data,

    // read port E -- the transfer engine's window read (t.st.async source)
    input  wire [RCTX_W-1:0]         re_rctx,
    input  wire [7:0]                re_idx,
    output wire [VLEN*ELEN-1:0]      re_data,

    input  wire                      w_en,
    input  wire [RCTX_W-1:0]         w_rctx,
    input  wire [7:0]                w_idx,
    input  wire [VLEN-1:0]           w_lane_en,
    input  wire [VLEN*ELEN-1:0]      w_data,

    // Second write port, for the transfer engine. A transfer lands over many
    // cycles while the pipeline keeps running; routing it through the execute
    // stage's write port would make every asynchronous load a source of
    // pipeline stalls, which is the opposite of what it is for. In a banked
    // implementation (arch overview sec 5.3) the two ports are usually
    // different banks and this is not an extra port at all.
    input  wire                      x_en,
    input  wire [RCTX_W-1:0]         x_rctx,
    input  wire [7:0]                x_idx,
    input  wire [VLEN-1:0]           x_lane_en,
    input  wire [VLEN*ELEN-1:0]      x_data
);

    reg [VLEN*ELEN-1:0] mem [0:PHYS_VREGS-1];

    // Written as continuous assignments, not a function: a function called from
    // a continuous assignment is only re-evaluated when its ARGUMENTS change,
    // so a base read from module scope inside one would go stale.
    wire [PHYS_W-1:0] base_a = vbase[ra_rctx*PHYS_W +: PHYS_W];
    wire [PHYS_W-1:0] base_b = vbase[rb_rctx*PHYS_W +: PHYS_W];
    wire [PHYS_W-1:0] base_c = vbase[rc_rctx*PHYS_W +: PHYS_W];
    wire [PHYS_W-1:0] base_d = vbase[rd_rctx*PHYS_W +: PHYS_W];
    wire [PHYS_W-1:0] base_e = vbase[re_rctx*PHYS_W +: PHYS_W];
    wire [PHYS_W-1:0] base_w = vbase[w_rctx*PHYS_W  +: PHYS_W];
    wire [PHYS_W-1:0] base_x = vbase[x_rctx*PHYS_W  +: PHYS_W];

    wire [PHYS_W-1:0] wa = base_w + w_idx;
    wire [PHYS_W-1:0] xa = base_x + x_idx;

    integer i, l;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < PHYS_VREGS; i = i + 1)
                mem[i] <= {(VLEN*ELEN){1'b0}};
        end else begin
            if (w_en) begin
                for (l = 0; l < VLEN; l = l + 1)
                    if (w_lane_en[l])
                        mem[wa][l*ELEN +: ELEN] <= w_data[l*ELEN +: ELEN];
            end
            if (x_en) begin
                for (l = 0; l < VLEN; l = l + 1)
                    if (x_lane_en[l])
                        mem[xa][l*ELEN +: ELEN] <= x_data[l*ELEN +: ELEN];
            end
        end
    end

    // Write-first read with per-lane granularity.
    wire [PHYS_W-1:0] aa = base_a + ra_idx;
    wire [PHYS_W-1:0] ab = base_b + rb_idx;
    wire [PHYS_W-1:0] ac = base_c + rc_idx;
    wire [PHYS_W-1:0] ad = base_d + rd_idx;
    wire [PHYS_W-1:0] ae = base_e + re_idx;

    wire hit_a = w_en && (aa == wa);
    wire hit_b = w_en && (ab == wa);
    wire hit_c = w_en && (ac == wa);
    wire hit_d = w_en && (ad == wa);
    wire hit_e = w_en && (ae == wa);

    genvar g;
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_bypass
            assign ra_data[g*ELEN +: ELEN] =
                (hit_a && w_lane_en[g]) ? w_data[g*ELEN +: ELEN] : mem[aa][g*ELEN +: ELEN];
            assign rb_data[g*ELEN +: ELEN] =
                (hit_b && w_lane_en[g]) ? w_data[g*ELEN +: ELEN] : mem[ab][g*ELEN +: ELEN];
            assign rc_data[g*ELEN +: ELEN] =
                (hit_c && w_lane_en[g]) ? w_data[g*ELEN +: ELEN] : mem[ac][g*ELEN +: ELEN];
            assign rd_data[g*ELEN +: ELEN] =
                (hit_d && w_lane_en[g]) ? w_data[g*ELEN +: ELEN] : mem[ad][g*ELEN +: ELEN];
            assign re_data[g*ELEN +: ELEN] =
                (hit_e && w_lane_en[g]) ? w_data[g*ELEN +: ELEN] : mem[ae][g*ELEN +: ELEN];
        end
    endgenerate

endmodule
