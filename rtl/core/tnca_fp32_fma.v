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
    parameter LAT = 2,
    // R41 G2 area rebuild (xls/tnca_fmad.x, evidence/r41/hbm_bw/fma_dslx/):
    // 0 = the stages generated from xls/tnca_fma.x (the default, unchanged);
    // 1 = the same function, bit for bit, with each stage replaced by its
    //     proved-equal tnca_fmad.x counterpart where one exists.
    parameter IMPL = 0,
    // R42 area study (tnca_fmax_fma, tb/area, evidence/r42/area): 0 = the generated stages above, unchanged (default);
    // nonzero = the same function from the area stages, with (BE / 4) % 2 = 1 the staged window shifter in the rounding
    // stage and (BE / 8) % 2 = 1 the area multiply/align stage instead of the generated one. LAT 2: cuts behind the
    // alignment and behind the select (as above); LAT 3: one more register, behind the adders (inferred DSP48E2).
    // (IMPL / 16) % 16 is added to BE, so an image that sets only FMA_IMPL (the parameter that already reaches every FMA
    // and MAC lane) selects it: FMA_IMPL = 1 + 16 * BE (+ 256 for the vector FPU's area converter); IMPL % 16 is the
    // generated variant.
    parameter BE = 0
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

    localparam IMPL_G = IMPL % 16;                 // the generated variant
    localparam BE_X   = BE + (IMPL / 16) % 16;     // the area back end (IMPL / 256 odd is the area converter of tnca_vector_fpu)

    generate
        if (LAT == 2 && BE_X != 0) begin : g_l2x
            tnca_fmax_fma #(.CUTS(4'b1010), .SHIFT((BE_X / 4) % 2), .XLS_ALIGN(((BE_X / 8) % 2) ? 0 : 1)) u_fma_x (
                .clk(clk), .rst_n(rst_n), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .a_u(32'd0), .sel_u(1'b0), .y(y));
        end else if (LAT == 2) begin : g_l2
            // xls/tnca_fma.x: FmaAligned and FmaNormalised, flattened by XLS
            localparam W_ALIGN = 150;
            localparam W_NORM  = 122;
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

            if (IMPL_G == 1) begin : g_norm_d
                tnca_fmad_norm_xls u_s1 (.s(s0_q), .out(s1_d));
            end else begin : g_norm
                tnca_fma_norm_xls u_s1 (.s(s0_q), .out(s1_d));
            end
            always @(posedge clk)
                if (!rst_n) s1_q <= {W_NORM{1'b0}};
                else        s1_q <= s1_d;

            tnca_fma_round_xls u_s2 (.s(s1_q), .out(y));
        end else if (LAT == 3 && BE_X != 0) begin : g_l3x
            tnca_fmax_fma #(.CUTS(4'b1110), .SHIFT((BE_X / 4) % 2), .XLS_ALIGN(((BE_X / 8) % 2) ? 0 : 1)) u_fma_x (
                .clk(clk), .rst_n(rst_n), .a(a), .b(b), .c(c), .sub_c(sub_c), .mul_only(mul_only), .ftz(ftz), .a_u(32'd0), .sel_u(1'b0), .y(y));
        end else if (LAT == 3 && IMPL_G == 1) begin : g_l3d
            tnca_fmad_l3_xls u_xls (
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
            // The refusal is a port no instance has -- an error in every
            // elaborator -- and not an undefined module, which the hardware
            // export would have to list as a blackbox (tools/r36/export.py).
            tnca_fma_l3_xls u_missing (.LAT_must_be_2_or_3(1'b1));
        end
    endgenerate

endmodule
