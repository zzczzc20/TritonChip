`timescale 1ns/1ps
module tb_lsu_wide;
 reg clk=0; always #5 clk=~clk;
 reg rst_n=0,valid=0,store=0,vector=0,indexed=0,sgn=0;
 reg [2:0] size=0;
 reg [63:0] base=0,sd=0;
 reg [511:0] vd=0,indices=0;
 reg [15:0] mask=0;
 wire ready,cv,mv,mwe;
 wire [63:0] ma;
 wire [511:0] mw;
 wire [63:0] be;
 wire [1:0] tag;
 reg rv=0; reg [1:0] rt=0; reg [511:0] rd=0;
 wire [63:0] result;
 wire [511:0] vr;
 wire [15:0] lanes;
 reg [7:0] mem[0:8191];
 reg [7:0] oracle[0:8191];
 integer i,j,t,cy=0,cases=0;
 reg [63:0] expected,value,addr;
 reg [15:0] expmask;
 tnca_lsu #(.MEM_DW(512),.MEM_BYTES(64),.MEM_SHIFT(6),.NUM_BEATS(4),.BEAT_W(2)) dut(
  .clk(clk),.rst_n(rst_n),.req_ready(ready),.req_valid(valid),.req_store(store),
  .req_vector(vector),.req_indexed(indexed),.req_size(size),.req_signed(sgn),.req_hint(2'd0),
  .req_base(base),.req_offset(64'd0),.req_index(indices),.req_sdata(sd),.req_amo(1'b0),
  .req_aop(4'd0),.req_cmp(64'd0),.req_vdata(vd),.req_mask(mask),.req_rctx(1'b0),
  .req_dst_class(3'd1),.req_dst_idx(8'd0),.cpl_valid(cv),.cpl_ack(cv),
  .cpl_sdata(result),.cpl_vdata(vr),.cpl_lane_en(lanes),.mem_req_valid(mv),
  .mem_req_ready(1'b1),.mem_req_addr(ma),.mem_req_we(mwe),.mem_req_be(be),
  .mem_req_wdata(mw),.mem_req_tag(tag),.mem_rsp_valid(rv),.mem_rsp_tag(rt),.mem_rsp_rdata(rd));
 // One response per cycle; LSU must handle response and new issue together.
 always @(posedge clk) begin
  cy<=cy+1; rv<=0;
  if(rst_n && mv) begin
   if(ma[5:0]!=0 || ma+63>=8192) $fatal(1,"bad wide address");
   rt<=tag; rv<=1;
   for(integer b=0;b<64;b=b+1) begin
    rd[b*8+:8]<=mem[ma+b];
    if(mwe && be[b]) mem[ma+b]<=mw[b*8+:8];
   end
  end
 end
 task access;
  integer nbytes,nlanes,l,b;
  begin
   nbytes=1<<size; nlanes=vector?16:1;
   for(l=0;l<nlanes;l=l+1) if(mask[l]) begin
    addr=base+(vector?(indexed?$signed(indices[l*32+:32])*nbytes:l*nbytes):0);
    value=vector?{32'd0,vd[l*32+:32]}:sd;
    if(store) for(b=0;b<nbytes;b=b+1) oracle[addr+b]=(value>>(8*b))&255;
   end
   @(negedge clk); while(!ready) @(negedge clk);
   valid=1; @(negedge clk); valid=0;
   t=0; while(!cv && t<200) begin @(negedge clk); t=t+1; end
   if(!cv) $fatal(1,"wide access timeout");
   if(!store) for(l=0;l<nlanes;l=l+1) if(mask[l]) begin
    addr=base+(vector?(indexed?$signed(indices[l*32+:32])*nbytes:l*nbytes):0);
    expected=0;
    for(b=0;b<nbytes;b=b+1) expected=expected | ({56'd0,oracle[addr+b]}<<(8*b));
    if(sgn && nbytes<8 && (expected & (64'd1<<(8*nbytes-1)))) expected=expected | (~64'd0<<(8*nbytes));
    value=vector?{32'd0,vr[l*32+:32]}:result;
    if(vector) expected={32'd0,expected[31:0]};
    if(value!==expected) $fatal(1,"wide load mismatch case=%0d lane=%0d got=%h expected=%h",cases,l,value,expected);
   end
   @(negedge clk);
   for(l=0;l<8192;l=l+1) if(mem[l]!==oracle[l]) $fatal(1,"wide store mismatch byte=%0d",l);
   cases=cases+1;
  end
 endtask
 initial begin
  for(i=0;i<8192;i=i+1) begin mem[i]=i%256; oracle[i]=i%256; end
  repeat(3) @(negedge clk); rst_n=1;
  for(i=0;i<4;i=i+1) for(j=0;j<8;j=j+1) begin
   vector=0; store=0; indexed=0; size=i; sgn=j%2; base=4096+61+j; mask=1;
   access;
   store=1; sd=64'h89abcdef12345678; access;
  end
  for(i=0;i<3;i=i+1) for(j=0;j<8;j=j+1) begin
   vector=1; store=0; indexed=j%2; size=i; sgn=j%2; base=4096+61+j; mask=j%3?16'h5555:16'hffff;
   for(integer l=0;l<16;l=l+1) begin vd[l*32+:32]=32'h12345678+l; indices[l*32+:32]=l*3-12; end
   access; store=1; access;
  end
  $display("LSU WIDE PASS cases=%0d scalar/vector/gather/masks/sign/straddles/partial stores",cases); $finish;
 end
 initial begin #500000; $fatal(1,"wide timeout"); end
endmodule
