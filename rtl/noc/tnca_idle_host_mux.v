// Host word accesses use the same coherent NI path as a core. The enclosing
// command controller must prohibit launching while host_busy is asserted.
`default_nettype none
module tnca_idle_host_mux #(parameter TAG_W=5, parameter HOST_TAG=0)(
 input wire clk,rst_n,host_allow,
 input wire c_valid,output wire c_ready,
 input wire [63:0] c_addr,c_wdata,c_cmp,
 input wire c_we,c_amo,input wire [7:0] c_be,input wire [3:0] c_aop,
 input wire [TAG_W-1:0] c_tag,
 output wire c_rsp_valid,output wire [TAG_W-1:0] c_rsp_tag,output wire [63:0] c_rsp_data,
 input wire h_valid,output wire h_ready,input wire h_we,
 input wire [63:0] h_addr,h_wdata,input wire [7:0] h_be,
 output wire h_rsp_valid,output wire [63:0] h_rsp_data,output reg host_busy,
 output wire n_valid,input wire n_ready,
 output wire [63:0] n_addr,n_wdata,n_cmp,
 output wire n_we,n_amo,output wire [7:0] n_be,output wire [3:0] n_aop,
 output wire [TAG_W-1:0] n_tag,
 input wire n_rsp_valid,input wire [TAG_W-1:0] n_rsp_tag,input wire [63:0] n_rsp_data
);
 reg host_issued;
 wire offer_host=host_allow && !c_valid && h_valid;
 wire use_host=host_busy || offer_host;
 assign n_valid=use_host ? (!host_issued && h_valid) : c_valid;
 assign c_ready=n_ready && !use_host;
 assign h_ready=n_ready && use_host && !host_issued;
 assign n_addr=use_host ? h_addr : c_addr;
 assign n_wdata=use_host ? h_wdata : c_wdata;
 assign n_be=use_host ? h_be : c_be;
 assign n_we=use_host ? h_we : c_we;
 assign n_tag=use_host ? HOST_TAG : c_tag;
 assign n_amo=use_host ? 1'b0 : c_amo;
 assign n_aop=use_host ? 4'd0 : c_aop;
 assign n_cmp=use_host ? 64'd0 : c_cmp;
 // The enclosing idle gate ensures only a host response can be outstanding
 // when host_busy is set. Still match its tag rather than stealing a stray one.
 assign h_rsp_valid=n_rsp_valid && host_busy && n_rsp_tag==HOST_TAG;
 assign h_rsp_data=n_rsp_data;
 assign c_rsp_valid=n_rsp_valid && !host_busy;
 assign c_rsp_tag=n_rsp_tag;assign c_rsp_data=n_rsp_data;
 always @(posedge clk) begin
  if(!rst_n)begin host_busy<=0;host_issued<=0;end
  else begin
   if(offer_host)host_busy<=1;
   if(h_valid && h_ready)begin host_busy<=1;host_issued<=1;end
   if(h_rsp_valid)begin host_busy<=0;host_issued<=0;end
  end
 end
endmodule
`default_nettype wire
