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
