//==============================================================================
// tnca_mul64 -- pipelined 64x64 multiplier for the scalar unit
//
// Three register stages; the low, signed-high and unsigned-high products are
// all valid three cycles after the operands are presented, and a new multiply
// may be presented every cycle.
//
// The third cut exists because the high half needs two dependent additions --
// the unsigned high sum, then the signed correction subtracted from it -- and
// this library maps both to ripple carry (see flow/synth/calibration.v). Keeping
// them in one stage made it the deepest thing in the core.
//
// Why it left tnca_scalar_alu: combinational, this array measured 135 gate
// levels against a ~50-level budget at 1 GHz, while the rest of the scalar unit
// (add, shift, compare, select, clz, popc) sits comfortably under it. Making
// the whole ALU multi-cycle to accommodate one instruction would have slowed
// down address arithmetic, which is the scalar unit's actual job (arch overview
// sec 11.3). Splitting it out lets s.mul and s.mulh.* carry their own latency.
//
// ONE array, not two. The signed high half is derived from the unsigned
// product: the correction terms are multiples of 2^64, so they touch only the
// high half and cannot borrow out of the low half.
//
//   mulh.s(a,b) = mulh.u(a,b) - (a[63] ? b : 0) - (b[63] ? a : 0)
//==============================================================================

module tnca_mul64 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [63:0]  a,
    input  wire [63:0]  b,
    output wire [63:0]  lo,
    output wire [63:0]  hi_s,
    output wire [63:0]  hi_u
);

    // ---- S0: partial products -----------------------------------------------
    // Split into 32-bit halves so the stage boundary lands inside the array
    // rather than after it.
    wire [31:0] al = a[31:0],  ah = a[63:32];
    wire [31:0] bl = b[31:0],  bh = b[63:32];

    wire [63:0] s0_ll = al * bl;
    wire [63:0] s0_lh = al * bh;
    wire [63:0] s0_hl = ah * bl;
    wire [63:0] s0_hh = ah * bh;

    reg [63:0] p0_ll, p0_lh, p0_hl, p0_hh;
    reg        p0_asign, p0_bsign;
    reg [63:0] p0_a, p0_b;

    always @(posedge clk) begin
        if (!rst_n) begin
            p0_ll <= 64'd0; p0_lh <= 64'd0; p0_hl <= 64'd0; p0_hh <= 64'd0;
            p0_asign <= 1'b0; p0_bsign <= 1'b0;
            p0_a <= 64'd0; p0_b <= 64'd0;
        end else begin
            p0_ll <= s0_ll; p0_lh <= s0_lh; p0_hl <= s0_hl; p0_hh <= s0_hh;
            p0_asign <= a[63]; p0_bsign <= b[63];
            p0_a <= a; p0_b <= b;
        end
    end

    // ---- S1: accumulate the unsigned product ---------------------------------
    //   a*b = ll + (lh + hl) << 32 + hh << 64
    wire [64:0] mid     = {1'b0, p0_lh} + {1'b0, p0_hl};
    wire [64:0] lo_sum  = {1'b0, p0_ll} + {mid[31:0], 32'd0};
    wire [63:0] s1_lo   = lo_sum[63:0];
    wire [63:0] s1_hi_u = p0_hh + {31'd0, mid[64:32]} + {63'd0, lo_sum[64]};

    // The signed correction is independent of hi_u, so it is summed here and
    // subtracted in the next stage rather than chaining two adds.
    wire [63:0] s1_corr = (p0_asign ? p0_b : 64'd0) + (p0_bsign ? p0_a : 64'd0);

    reg [63:0] p1_lo, p1_hi_u, p1_corr;
    always @(posedge clk) begin
        if (!rst_n) begin
            p1_lo <= 64'd0; p1_hi_u <= 64'd0; p1_corr <= 64'd0;
        end else begin
            p1_lo <= s1_lo; p1_hi_u <= s1_hi_u; p1_corr <= s1_corr;
        end
    end

    // ---- S2: apply the signed correction ------------------------------------
    wire [63:0] s2_hi_s = p1_hi_u - p1_corr;

    reg [63:0] p2_lo, p2_hi_s, p2_hi_u;
    always @(posedge clk) begin
        if (!rst_n) begin
            p2_lo <= 64'd0; p2_hi_s <= 64'd0; p2_hi_u <= 64'd0;
        end else begin
            p2_lo <= p1_lo; p2_hi_s <= s2_hi_s; p2_hi_u <= p1_hi_u;
        end
    end

    assign lo   = p2_lo;
    assign hi_s = p2_hi_s;
    assign hi_u = p2_hi_u;

endmodule
