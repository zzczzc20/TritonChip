`timescale 1ns/1ps
// tb_mul64 -- randomised differential test of the pipelined 64x64 multiplier
// against Verilog's own 128-bit multiply, with the emphasis on sign boundaries.
module tb_mul;
    reg clk=0, rst_n=0; always #5 clk=~clk;
    reg [63:0] a,b; wire [63:0] lo, hi_s, hi_u;
    tnca_mul64 dut(.clk(clk),.rst_n(rst_n),.a(a),.b(b),.lo(lo),.hi_s(hi_s),.hi_u(hi_u));
    reg [63:0] qa[0:7], qb[0:7]; reg [7:0] qv;
    integer i, kk, n, fails;
    reg [127:0] pu; reg signed [127:0] ps;
    task push; input [63:0] xa, xb; input v; begin
        for (kk=7;kk>0;kk=kk-1) begin qa[kk]=qa[kk-1]; qb[kk]=qb[kk-1]; qv[kk]=qv[kk-1]; end
        qa[0]=xa; qb[0]=xb; qv[0]=v; end
    endtask
    task chk; begin
        if (qv[2]) begin
            n = n+1;
            pu = qa[2]*qb[2];
            ps = $signed(qa[2])*$signed(qb[2]);
            if (lo !== pu[63:0])      begin fails=fails+1; if(fails<8) $display("FAIL lo  a=%h b=%h got=%h want=%h",qa[2],qb[2],lo,pu[63:0]); end
            if (hi_u !== pu[127:64])  begin fails=fails+1; if(fails<8) $display("FAIL hiu a=%h b=%h got=%h want=%h",qa[2],qb[2],hi_u,pu[127:64]); end
            if (hi_s !== ps[127:64])  begin fails=fails+1; if(fails<8) $display("FAIL his a=%h b=%h got=%h want=%h",qa[2],qb[2],hi_s,ps[127:64]); end
        end end
    endtask
    initial begin
        n=0; fails=0; qv=0;
        repeat(3) @(posedge clk); #1 rst_n=1;
        for (i=0;i<60000;i=i+1) begin
            case (i % 8)
              0: begin a=64'd0; b={$random,$random}; end
              1: begin a=64'hffffffffffffffff; b={$random,$random}; end
              2: begin a=64'h8000000000000000; b={$random,$random}; end
              3: begin a={$random,$random}; b=64'h8000000000000000; end
              4: begin a=64'h7fffffffffffffff; b=64'h7fffffffffffffff; end
              5: begin a={32'd0,$random}; b={32'd0,$random}; end
              default: begin a={$random,$random}; b={$random,$random}; end
            endcase
            @(posedge clk); #1; push(a,b,1'b1); chk;
        end
        for (i=0;i<4;i=i+1) begin @(posedge clk); #1; push(0,0,1'b0); chk; end
        $display("tb_mul64: %0d checks, %0d failures", n, fails);
        if (fails==0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
