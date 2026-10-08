//==============================================================================
// tnca_fp32_fma -- IEEE 754 binary32 fused multiply-add, single rounding
//
//     y = round_RNE( a * b + (sub_c ? -c : c) )
//
// with mul_only=1 returning round_RNE(a * b) and the addend ignored entirely.
// The vector FP unit builds vf.add / vf.sub from this by passing b = 1.0, which
// is exact, and the matrix unit accumulates through it (tnca_fpu_multi), so
// every FP arithmetic result in the core goes through one rounding in one
// place. Subnormal inputs and results are handled; ftz=1 flushes them to a
// zero of the same sign -- an explicit mode, per ISA sec 8.2, never an
// implicit shortcut.
//
// THE ARITHMETIC IS GENERATED
//
// It is written once, in DSLX (xls/tnca_fma.x), and XLS places the pipeline
// registers (tools/xls/config.json). This module only picks the generated
// instance whose latency the core was elaborated for:
//
//     LAT = 2   rtl/gen/tnca_fma_xls.v      three stages, two registers
//     LAT = 3   rtl/gen/tnca_fma_l3_xls.v   four stages, three registers
//
// In both, `y` is combinational from the last stage -- the consumers (the
// vector FP writeback, the accumulator file) register it -- so `y` is valid
// LAT cycles after a/b/c are presented and a new operation may be presented
// every cycle. LAT is the core's FP_LAT. A clock that needs more stages is a
// new entry in config.json and a line here, not a redesign.
//
// Until R38 this file was the arithmetic itself, with two hand-placed register
// stages. Its U280 worst path was 60 logic levels through its normalise/round
// stage, and the reason it was hand-written was XLS's apfloat flushing
// subnormals; xls/tnca_fma.x does not use apfloat and keeps them. The
// generated version computes the same function bit for bit: tb_fp32_fma and
// fma-deep check it against the exact Fraction oracle (tools/model/fp32.py).
//==============================================================================

module tnca_fp32_fma #(
    parameter LAT = 2
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    input  wire        sub_c,       // negate the addend (vf.fms, vf.sub)
    input  wire        mul_only,    // ignore the addend (vf.mul)
    input  wire        ftz,
    output wire [31:0] y
);

    generate
        if (LAT == 2) begin : g_l2
            tnca_fma_xls u_xls (
                .clk(clk), .rst(!rst_n),
                .a(a), .b(b), .c(c),
                .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
                .out(y)
            );
        end else if (LAT == 3) begin : g_l3
            tnca_fma_l3_xls u_xls (
                .clk(clk), .rst(!rst_n),
                .a(a), .b(b), .c(c),
                .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
                .out(y)
            );
        end else begin : g_bad
            // No generated instance has this latency: refuse to elaborate
            // rather than build a unit whose results arrive at the wrong time.
            tnca_fp32_fma_LAT_has_no_generated_instance u_missing ();
        end
    endgenerate

endmodule
