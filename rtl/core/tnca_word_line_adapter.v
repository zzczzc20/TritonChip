// Position an eight-byte legacy request inside a 64-byte cache word. Addresses
// retain their original low bits for MMIO decoding. Tags own their offsets
// until a reply returns; the producer owns credit allocation as before.
`default_nettype none
module tnca_word_line_adapter #(
    parameter TAG_W=3
)(
    input wire clk,rst_n,
    input wire n_req_valid,
    output wire n_req_ready,
    input wire [63:0] n_req_addr,n_req_wdata,
    input wire n_req_we,n_req_amo,
    input wire [7:0] n_req_be,
    input wire [TAG_W-1:0] n_req_tag,
    input wire [1:0] n_req_hint,
    input wire [3:0] n_req_aop,
    input wire [63:0] n_req_cmp,
    output wire n_rsp_valid,
    output wire [TAG_W-1:0] n_rsp_tag,
    output wire [63:0] n_rsp_rdata,
    output wire w_req_valid,
    input wire w_req_ready,
    output wire [63:0] w_req_addr,w_req_be,
    output wire [511:0] w_req_wdata,
    output wire w_req_we,w_req_amo,
    output wire [TAG_W-1:0] w_req_tag,
    output wire [1:0] w_req_hint,
    output wire [3:0] w_req_aop,
    output wire [63:0] w_req_cmp,
    input wire w_rsp_valid,
    input wire [TAG_W-1:0] w_rsp_tag,
    input wire [511:0] w_rsp_rdata
);
    reg [2:0] word_offset[0:(1<<TAG_W)-1];
    reg [(1<<TAG_W)-1:0] active;
    wire take=n_req_valid && n_req_ready;
    assign w_req_valid=n_req_valid;
    assign n_req_ready=w_req_ready;
    assign w_req_addr=n_req_addr;
    assign w_req_be={56'd0,n_req_be}<<n_req_addr[5:0];
    assign w_req_wdata={448'd0,n_req_wdata}<<(n_req_addr[5:0]*8);
    assign w_req_we=n_req_we;
    assign w_req_amo=n_req_amo;
    assign w_req_tag=n_req_tag;
    assign w_req_hint=n_req_hint;
    assign w_req_aop=n_req_aop;
    assign w_req_cmp=n_req_cmp;
    assign n_rsp_valid=w_rsp_valid;
    assign n_rsp_tag=w_rsp_tag;
    assign n_rsp_rdata=w_rsp_rdata>>(word_offset[w_rsp_tag]*64);
    always @(posedge clk) begin
        if(!rst_n) active<=0;
        else begin
            if(w_rsp_valid) active[w_rsp_tag]<=0;
            if(take) begin active[n_req_tag]<=1;word_offset[n_req_tag]<=n_req_addr[5:3];end
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk) if(rst_n) begin
        if(take && n_req_addr[2:0]!=0) $fatal(1,"word adapter requires eight-byte aligned requests");
        if(take && active[n_req_tag] && !(w_rsp_valid && w_rsp_tag==n_req_tag)) $fatal(1,"word adapter tag reused");
        if(w_rsp_valid && !active[w_rsp_tag]) $fatal(1,"word adapter unowned response");
    end
`endif
endmodule
`default_nettype wire
