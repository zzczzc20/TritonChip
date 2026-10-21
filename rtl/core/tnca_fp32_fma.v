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
// It is written once, in DSLX (xls/tnca_fma.x), as three stage functions --
// align, normalise, round -- and fma32, the three composed. This module picks
// the instance for the latency the core was elaborated for:
//
//     LAT = 2   rtl/gen/tnca_fma_{align,norm,round}_xls.v, combinational,
//               with the two registers placed here, between them
//     LAT = 3   rtl/gen/tnca_fma_l3_xls.v, fma32 scheduled by XLS into
//               four stages, three registers
//
// In both, `y` is combinational from the last stage -- the consumers (the
// vector FP writeback, the accumulator file) register it -- so `y` is valid
// LAT cycles after a/b/c are presented and a new operation may be presented
// every cycle. LAT is the core's FP_LAT.
//
// WHY THE LATENCY-2 CUTS ARE PLACED HERE (R41 Stage 2, F03/F08/F09)
//
// Until R41 the latency-2 instance was fma32 scheduled by XLS against asap7
// into three stages (tools/xls/config.json, `clock_period_ps: null`). On the
// U280 at 100 MHz each of its three stages was a census family of its own, all
// three within 0.05 ns of the design's -0.92 ns WNS: VFPU red_busy into the
// first stage (F03), the first stage's sticky bit into the second's shift
// amount (F08), and the second stage's shift amount into the accumulator file
// (F09). asap7 prices the 24x24 multiplier, which the FPGA maps to two cascaded
// DSP48E2 slices, at several times its FPGA cost, so the balance it finds is
// not the FPGA's; and once the arithmetic was shortened for the FPGA, XLS moved
// the alignment shifter into the middle stage, which made that the long one.
// So for this instance the two cuts are placed where the unit OOC measures the
// stages as balanced: after the alignment shift and its sticky bit
// (FmaAligned), and after the leading-zero count and what follows from it
// (FmaNormalised). What crosses each cut is a DSLX struct, so a stage's ports
// are whatever the DSLX says; the widths below are those structs' bit counts.
// The registers have the generated pipeline's synchronous reset.
//
// The function is unchanged, bit for bit: tb_fp32_fma and fma-deep check both
// instances against the exact Fraction oracle (tools/model/fp32.py), and
// evidence/r41/stage2/candidates/F03/miter runs both against the previous
// generated RTL.
//
// Until R38 this file was the arithmetic itself, with two hand-placed register
// stages. Its U280 worst path was 60 logic levels through its normalise/round
// stage, and the reason it was hand-written was XLS's apfloat flushing
// subnormals; xls/tnca_fma.x does not use apfloat and keeps them.
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
            // xls/tnca_fma.x: FmaAligned and FmaNormalised, flattened by XLS
            localparam W_ALIGN = 150;
            localparam W_NORM  = 113;
            wire [W_ALIGN-1:0] s0_d;
            reg  [W_ALIGN-1:0] s0_q;
            wire [W_NORM-1:0]  s1_d;
            reg  [W_NORM-1:0]  s1_q;

            tnca_fma_align_xls u_s0 (
                .a(a), .b(b), .c(c),
                .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz),
                .out(s0_d)
            );
            always @(posedge clk)
                if (!rst_n) s0_q <= {W_ALIGN{1'b0}};
                else        s0_q <= s0_d;

            tnca_fma_norm_xls u_s1 (.s(s0_q), .out(s1_d));
            always @(posedge clk)
                if (!rst_n) s1_q <= {W_NORM{1'b0}};
                else        s1_q <= s1_d;

            tnca_fma_round_xls u_s2 (.s(s1_q), .out(y));
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
            // The refusal is a port no instance has -- an error in every
            // elaborator -- and not an undefined module, which the hardware
            // export would have to list as a blackbox (tools/r36/export.py).
            tnca_fma_l3_xls u_missing (.LAT_must_be_2_or_3(1'b1));
        end
    endgenerate

endmodule
