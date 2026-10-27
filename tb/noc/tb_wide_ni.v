`timescale 1ns/1ps
module tb_wide_ni;
 parameter DATA_W=512, STALL=1, LINK_STAGES=2, TOTAL=768;
 localparam BYTES=DATA_W/8, PW=216+DATA_W-64+BYTES-8, FW=PW+4, N=3;
 reg clk=0; always #5 clk=~clk;
 reg rst_n=0;
 integer cycle=0,sent=0,received=0,first_accept=-1,last_accept=-1;
 reg [31:0] pending=0;
 reg [DATA_W-1:0] expected[0:31];
 wire [5:0] iv,ir,ov,orr;
 wire [6*FW-1:0] inf,outf;
 wire ready,cv;
 wire [4:0] ct;
 wire [DATA_W-1:0] cd;
 wire valid=sent<TOTAL && !pending[sent%32];
 wire [63:0] address=64'h1000+sent*64;
 wire [4:0] tag=sent%32;
 wire [3:0] operation=sent%12;
 wire [1:0] dest=home(address);
 wire [63:0] compare=64'hfedcba9876543210^sent;
 function [DATA_W-1:0] pattern;
  input integer seq;
  begin
   for(integer b=0;b<BYTES;b=b+1) pattern[b*8+:8]=(seq*17+b*31+5)%256;
  end
 endfunction
 function [BYTES-1:0] enables;
  input integer seq;
  begin for(integer b=0;b<BYTES;b=b+1) enables[b]=(seq+b)%3!=0; end
 endfunction
 function integer home;
  input [63:0] addr;
  reg [63:0] lineaddr;
  reg [7:0] hash;
  begin lineaddr=addr>>6; hash=0; for(integer b=0;b<8;b=b+1) hash=hash^lineaddr[b*8+:8]; home=hash%3; end
 endfunction
 tnca_mesh #(.NX(3),.NY(1),.XW(2),.YW(2),.PAYLOAD_W(PW),.LINK_STAGES(LINK_STAGES)) mesh(
  .clk(clk),.rst_n(rst_n),.l_in_valid(iv),.l_in_ready(ir),.l_in_flit(inf),
  .l_out_valid(ov),.l_out_ready(orr),.l_out_flit(outf));
 tnca_core_ni #(.NX(3),.NY(1),.XW(2),.YW(2),.DATA_W(DATA_W),.MY_ID(0)) core(
  .clk(clk),.rst_n(rst_n),.req_valid(valid),.req_ready(ready),.req_addr(address),
  .req_we(sent%2!=0),.req_be(enables(sent)),.req_wdata(pattern(sent)),.req_tag(tag),
  .req_amo(sent%5==0),.req_aop(operation),.req_cmp(64'hfedcba9876543210^sent),
  .rsp_valid(cv),.rsp_tag(ct),.rsp_rdata(cd),
  .net_in_valid_req(iv[0]),.net_in_ready_req(ir[0]),.net_in_flit_req(inf[0+:FW]),
  .net_out_valid_rsp(ov[1]),.net_out_flit_rsp(outf[FW+:FW]),.net_out_ready_rsp(orr[1]),
  .net_in_ready_rsp(1'b1),.net_out_valid_req(1'b0),.net_out_flit_req({FW{1'b0}}));
 // Source core uses only VN_REQ. Other nodes have no request producer.
 assign iv[2]=0; assign iv[4]=0;
 assign inf[2*FW+:FW]=0; assign inf[4*FW+:FW]=0;
 assign orr[3]=1; assign orr[5]=1;
 genvar node;
 generate for(node=0;node<N;node=node+1) begin:g_home
  wire qv,qr,qwe,qa;
  wire [63:0] qaddr,qcmp;
  wire [BYTES-1:0] qbe;
  wire [DATA_W-1:0] qdata;
  wire [3:0] qsrc,qaop;
  wire [4:0] qtag;
  reg response=0;
  reg [DATA_W-1:0] data=0;
  reg [4:0] rtag=0;
  reg [3:0] rsrc=0;
  wire response_ready;
  assign qr=(!response || response_ready) && (!STALL || ((cycle+node)%7!=0 && (cycle+node)%11!=0));
  tnca_slice_ni #(.NX(3),.NY(1),.XW(2),.YW(2),.DATA_W(DATA_W)) slice(
   .clk(clk),.rst_n(rst_n),.req_valid(qv),.req_ready(qr),.req_addr(qaddr),.req_we(qwe),
   .req_be(qbe),.req_wdata(qdata),.req_src(qsrc),.req_tag(qtag),.req_amo(qa),.req_aop(qaop),.req_cmp(qcmp),
   .rsp_valid(response),.rsp_ready(response_ready),.rsp_src(rsrc),.rsp_tag(rtag),.rsp_rdata(data),
   .net_out_valid_req(ov[node*2]),.net_out_ready_req(orr[node*2]),.net_out_flit_req(outf[node*2*FW+:FW]),
   .net_in_valid_rsp(iv[node*2+1]),.net_in_ready_rsp(ir[node*2+1]),.net_in_flit_rsp(inf[(node*2+1)*FW+:FW]),
   .net_in_ready_req(1'b1),.net_out_valid_rsp(1'b0),.net_out_flit_rsp({FW{1'b0}}));
  integer seq;
  always @(posedge clk) begin
   if(!rst_n) response<=0;
   else begin
    if(response_ready) response<=0;
    if(qv && qr) begin
     seq=(qaddr-64'h1000)/64;
     if(home(qaddr)!=node || qsrc!=0 || qtag!=seq%32 || qwe!=(seq%2!=0) || qa!=(seq%5==0) || qaop!=seq%12 ||
        qbe!==enables(seq) || qdata!==pattern(seq) || qcmp!==(64'hfedcba9876543210^seq))
      $fatal(1,"wide NI request mismatch node=%0d seq=%0d",node,seq);
     response<=1; rtag<=qtag; rsrc<=qsrc; data<=~qdata;
    end
   end
  end
 end endgenerate
 always @(posedge clk) if(rst_n) begin
  cycle<=cycle+1;
  // Independent canonical concatenation catches matching encoder/decoder bugs.
  // At 64 bits this is exactly the historical payload bit layout.
  if(valid && inf[0+:FW] !== {2'd0,dest,1'b0,compare,operation,(sent%5==0),tag,4'd0,pattern(sent),enables(sent),(sent%2!=0),address})
   $fatal(1,"canonical packet layout mismatch");
  if(valid && ready) begin
   if(first_accept<0) first_accept<=cycle;
   last_accept<=cycle; pending[tag]<=1; expected[tag]<=~pattern(sent); sent<=sent+1;
  end
  if(cv) begin
   if(!pending[ct] || cd!==expected[ct]) $fatal(1,"wide NI lost/duplicate/corrupt response tag=%0d",ct);
   pending[ct]<=0; received<=received+1;
  end
  if(received==TOTAL) begin
   if(sent!=TOTAL || pending!=0) $fatal(1,"wide NI did not drain");
   if(!STALL && last_accept-first_accept!=TOTAL-1) $fatal(1,"unsaturated wide injection span=%0d",last_accept-first_accept);
   $display("WIDE NI MESH PASS data=%0d payload=%0d links=%0d stall=%0d requests=%0d injection_span=%0d cycles=%0d",DATA_W,PW,LINK_STAGES,STALL,sent,last_accept-first_accept+1,cycle);
   $finish;
  end
  if(cycle>20000) $fatal(1,"wide NI timeout sent=%0d received=%0d",sent,received);
 end
 initial begin repeat(4) @(negedge clk); rst_n=1; end
endmodule
