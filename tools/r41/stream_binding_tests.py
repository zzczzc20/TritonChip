#!/usr/bin/env python3
"""Exercise the generated CU through AXI-Lite and its named PC masters."""
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from generate_stream_binding import FIELDS, CONTROL

ROOT = Path(__file__).resolve().parents[2]


def bench():
    text = ['`timescale 1ns/1ps', 'module tb_stream_binding;',
            'reg clk=0,rst_n=0;always #5 clk=~clk;']
    for name, direction, width in CONTROL:
        text.append(f'{"reg" if direction=="input" else "wire"} [{width-1}:0] c_{name}'
                    + ('=0;' if direction=='input' else ';'))
    text += [f'wire [{32*w-1}:0] {n};' for n,_,w in FIELDS]
    text += ['tnca_stream_hbm_xo dut(.ap_clk(clk),.ap_rst_n(rst_n),']
    connections = [f'.s_axi_tnca_{n}(c_{n})' for n,_,_ in CONTROL]
    connections += [f'.m_axi_pc{p}_{n}({n}[{p*w}+:{w}])' for p in range(32) for n,_,w in FIELDS]
    text += [',\n'.join(connections), ');']
    for name, direction, width in FIELDS:
        if direction == 'input' and name not in ('awready','wready','bvalid'):
            text.append(f'assign {name}=' + ("32'hffffffff;" if name=='arready' else "'0;"))
    text += [r'''
integer target=0,requested_lines=0,request_id=0,tick=0;
reg [31:0] expected_pattern=0;reg new_window=0;
integer written[0:31],issued[0:31];
always @(posedge clk)begin tick<=tick+1;if(tick>10000)$fatal(1,"binding timeout");end
generate for(genvar p=0;p<32;p=p+1)begin:slave
 reg have=0,b=0;integer left=0,pos=0;
 assign awready[p]=!have && !b;assign wready[p]=have;assign bvalid[p]=b;
 always @(posedge clk)begin
  if(!rst_n)begin have<=0;b<=0;written[p]<=0;issued[p]<=0;end
  else begin
   if(new_window)begin
    if(have || b)$fatal(1,"binding window before drain");written[p]<=0;issued[p]<=0;
   end
   if(arvalid[p])$fatal(1,"fill issued a read");
   if(awvalid[p] && awready[p])begin
    if(p!=target || awaddr[p*64+:64]!==((64'(p)<<28)+4032+64*issued[p]) ||
       awid[p]!=0 || awsize[p*3+:3]!=6 || awburst[p*2+:2]!=1 ||
       awaddr[p*64+:12]+(awlen[p*8+:8]+1)*64>4096)$fatal(1,"named master mapping/geometry");
    have<=1;left<=awlen[p*8+:8]+1;pos<=0;issued[p]<=issued[p]+awlen[p*8+:8]+1;
   end
   if(wvalid[p] && wready[p])begin
    if(wdata[p*512+:512]!=={16{expected_pattern}} || wstrb[p*64+:64]!==64'hffffffffffffffff ||
       wlast[p]!=(pos==left-1))$fatal(1,"named master data/last");
    written[p]<=written[p]+1;
    if(wlast[p])begin have<=0;b<=1;end else pos<=pos+1;
   end
   if(b && bready[p])b<=0;
  end
 end
end endgenerate

task put32;
 input [11:0] a;input [31:0] d;
 reg ah,wh;
 begin
  @(negedge clk);c_awaddr=a;c_awvalid=1;c_wdata=d;c_wstrb=15;c_wvalid=1;c_bready=1;ah=0;wh=0;
  while(!ah || !wh)begin
   @(posedge clk);if(c_awvalid && c_awready)ah=1;if(c_wvalid && c_wready)wh=1;
   @(negedge clk);if(ah)c_awvalid=0;if(wh)c_wvalid=0;
  end
  wait(c_bvalid);if(c_bresp!=0)$fatal(1,"AXI-Lite write response");
  @(negedge clk);c_bready=0;
 end
endtask
task put64;
 input [11:0] a;input [63:0] d;
 begin put32(a,d[31:0]);put32(a+4,d[63:32]);end
endtask
task get32;
 input [11:0] a;output [31:0] d;
 begin
  @(negedge clk);c_araddr=a;c_arvalid=1;c_rready=1;
  @(posedge clk);while(!c_arready)@(posedge clk);
  @(negedge clk);c_arvalid=0;wait(c_rvalid);d=c_rdata;
  if(c_rresp!=0)$fatal(1,"AXI-Lite read response");
  @(negedge clk);c_rready=0;
 end
endtask
task get64;
 input [11:0] a;output [63:0] d;
 reg [31:0] lo,hi;
 begin get32(a,lo);get32(a+4,hi);d={hi,lo};end
endtask
reg [63:0] got;integer tries,i;
task fill;
 input integer pc,n;
 begin
  target=pc;requested_lines=n;request_id=request_id+1;expected_pattern=32'hf00d00ff+pc;
  @(negedge clk);new_window=1;@(negedge clk);new_window=0;
  put64(12'h040,64'h100000000|expected_pattern);put64(12'h050,pc);
  put64(12'h058,0);put64(12'h060,4032);put64(12'h068,n);
  put64(12'h048,64'd1<<pc);put64(12'h010,request_id);put64(12'h018,1);
  got=0;tries=0;
  while(got!=request_id && tries<100)begin get64(12'h028,got);tries=tries+1;end
  if(got!=request_id)$fatal(1,"wrapper completion timeout");
  get64(12'h020,got);if(got!=2)$fatal(1,"wrapper completion not drained");
  get64(12'h030,got);if(got!=0)$fatal(1,"wrapper command refused");
  get64(12'h078,got);if(got!=0)$fatal(1,"wrapper spurious reads");
  get64(12'h080,got);if(got!=n)$fatal(1,"wrapper counter mismatch");
  for(i=0;i<32;i=i+1)if(written[i]!=(i==pc?n:0))$fatal(1,"inactive named master traffic");
  $display("STREAM BINDING FILL PASS pc=%0d lines=%0d",pc,n);
 end
endtask
initial begin
 repeat(5)@(negedge clk);rst_n=1;repeat(8)@(negedge clk);
 get64(0,got);if(got!==64'h52343153424e4b01)$fatal(1,"wrapper identity");
 get64(8,got);if(got!==64'h0020020000100080)$fatal(1,"wrapper capability");
 put64(12'h088,64'h200000000);put64(12'h090,0);
 fill(0,17);fill(31,3);
 $display("STREAM BINDING AXI-LITE/PC0/PC31 PASS");$finish;
end
endmodule
''']
    return '\n'.join(text)


def main():
    subprocess.run(['python3','tools/r41/generate_stream_binding.py','--check'], cwd=ROOT, check=True)
    kernel = ET.parse(ROOT/'fpga/kernel_stream_hbm.xml').find('kernel')
    assert kernel.attrib['name'] == 'tnca_stream'
    ports = kernel.find('ports').findall('port')
    assert len(ports) == 33
    args = kernel.find('args').findall('arg')
    assert len(args) == 32
    for p, arg in enumerate(args):
        assert arg.attrib['id'] == str(p) and arg.attrib['port'] == f'm_axi_pc{p}'
    with tempfile.TemporaryDirectory(prefix='tnca-stream-binding-') as temporary:
        source = Path(temporary)/'tb.v';source.write_text(bench())
        binary = str(Path(temporary)/'binding.vvp')
        subprocess.run(['iverilog','-g2012','-s','tb_stream_binding','-o',binary,
            'rtl/board/tnca_stream_hbm_xo.v','rtl/fpga/tnca_stream_bank.v',
            'rtl/fpga/tnca_stream_axi.v','rtl/fpga/tnca_stream_fifo.v',
            'rtl/fpga/tnca_axi_lite.v','rtl/common/tnca_fifo.v',str(source)], cwd=ROOT, check=True, timeout=60)
        subprocess.run(['vvp','-i',binary], cwd=ROOT, check=True, timeout=120)
    print('STREAM CU BINDING FUNCTIONAL PASS', flush=True)


if __name__ == '__main__':
    main()
