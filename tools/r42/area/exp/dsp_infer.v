// dsp_infer -- which codings make Vivado 2023.2 map an adder / subtractor onto DSP48E2 (R42 area, experiment)
// t1: registered 48-bit add, attribute on the output register
module dsp_t1 (input clk, input [47:0] a, b, output reg [47:0] p);
    reg [47:0] ra, rb;
    (* use_dsp = "yes" *) reg [47:0] s;
    always @(posedge clk) begin ra <= a; rb <= b; s <= ra + rb; p <= s; end
endmodule
// t2: registered 49-bit dynamic add/sub with carry-in, attribute on the module
(* use_dsp = "yes" *)
module dsp_t2 (input clk, input [47:0] a, b, input sub, input cin, output reg [48:0] p);
    reg [47:0] ra, rb; reg rs, rc;
    always @(posedge clk) begin
        ra <= a; rb <= b; rs <= sub; rc <= cin;
        p <= rs ? ({1'b0, ra} - {1'b0, rb} - rc) : ({1'b0, ra} + {1'b0, rb} + rc);
    end
endmodule
// t3: purely combinational module wrapped, attribute on the instance's module, flattened by the top
(* use_dsp = "yes" *)
module dsp_comb (input [47:0] a, b, input cin, output [48:0] s);
    assign s = {1'b0, a} - {1'b0, b} - cin;
endmodule
module dsp_t3 (input clk, input [47:0] a, b, input cin, output reg [48:0] q);
    reg [47:0] ra, rb; reg rc; wire [48:0] s;
    always @(posedge clk) begin ra <= a; rb <= b; rc <= cin; q <= s; end
    dsp_comb u (.a(ra), .b(rb), .cin(rc), .s(s));
endmodule
// t4: 80-bit dynamic add/sub, registered, attribute on the module
(* use_dsp = "yes" *)
module dsp_t4 (input clk, input [79:0] a, b, input sub, output reg [80:0] p);
    reg [79:0] ra, rb; reg rs;
    always @(posedge clk) begin ra <= a; rb <= b; rs <= sub; p <= rs ? ({1'b0, ra} - {1'b0, rb}) : ({1'b0, ra} + {1'b0, rb}); end
endmodule
// t5: registered a - b - cin, 49-bit result, attribute on the output register (R1/H adders of the area norm stage)
module dsp_t5 (input clk, input [47:0] a, b, input cin, output reg [48:0] p);
    reg [47:0] ra, rb; reg rc;
    (* use_dsp = "yes" *) reg [48:0] s;
    always @(posedge clk) begin ra <= a; rb <= b; rc <= cin; s <= {1'b0, ra} - {1'b0, rb} - rc; p <= s; end
endmodule
// t6: three static registered adders sharing the operands: sum, a-b, b-a (49-bit results)
module dsp_t6 (input clk, input [47:0] a, b, output reg [48:0] p0, p1, p2);
    reg [47:0] ra, rb;
    (* use_dsp = "yes" *) reg [48:0] s0, s1, s2;
    always @(posedge clk) begin
        ra <= a; rb <= b;
        s0 <= {1'b0, ra} + {1'b0, rb}; s1 <= {1'b0, ra} - {1'b0, rb}; s2 <= {1'b0, rb} - {1'b0, ra};
        p0 <= s0; p1 <= s1; p2 <= s2;
    end
endmodule
// t7: the same without the output register after s (the P register is the last register): s feeds a comb mux
module dsp_t7 (input clk, input [47:0] a, b, input sel, output reg [48:0] p);
    reg [47:0] ra, rb; reg rsel;
    (* use_dsp = "yes" *) reg [48:0] s0, s1;
    always @(posedge clk) begin
        ra <= a; rb <= b; rsel <= sel;
        s0 <= {1'b0, ra} + {1'b0, rb}; s1 <= {1'b0, ra} - {1'b0, rb};
        p <= rsel ? s0 : s1;
    end
endmodule
