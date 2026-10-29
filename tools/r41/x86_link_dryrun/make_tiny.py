#!/usr/bin/env python3
"""Create an isolated, non-executable-on-board AXI interface link probe.

It preserves the production kernel's top-level parameters/ports and AXI-Lite
adapter so package_xo and the real three-HBM connectivity see the same ABI.
It is only a link-flow probe; it is not TNCA and is never loaded on a card.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path, help='production multicore wrapper')
p.add_argument('output', type=Path, help='wrapper in isolated source copy')
a = p.parse_args()
s = a.source.read_text()
header = s.split('\n);', 1)[0] + '\n);\n'
start = s.index(' (* ASYNC_REG="TRUE" *)')
end = s.index('\n );', start) + len('\n );')
body = s[start:end]
body += '''
 // Probe only: issue one read on each port when the control bus is written.
 reg [63:0] probe_addr;
 reg [2:0] pending;
 reg [31:0] responses;
 always @(posedge ap_clk) begin
   if (!rst_n) begin probe_addr <= 0; pending <= 0; responses <= 0; end
   else begin
     if (reg_we) begin probe_addr <= reg_wdata; pending <= 3'b111; end
     else pending <= pending & ~{m_axi_hbm2_arready,m_axi_hbm1_arready,m_axi_hbm0_arready};
     if (m_axi_hbm0_rvalid) responses <= responses ^ m_axi_hbm0_rdata[31:0];
     if (m_axi_hbm1_rvalid) responses <= responses ^ m_axi_hbm1_rdata[31:0];
     if (m_axi_hbm2_rvalid) responses <= responses ^ m_axi_hbm2_rdata[31:0];
   end
 end
 assign reg_rdata = {32'b0, responses};
'''
for i in range(3):
    pre = f'm_axi_hbm{i}_'
    vals = dict(awvalid="1'b0", awaddr="64'b0", awid="{AXI_ID_W{1'b0}}",
                awlen="8'b0", awsize="3'd5", awburst="2'b01", wvalid="1'b0",
                wdata="{AXI_DATA_W{1'b0}}", wstrb="{(AXI_DATA_W/8){1'b0}}",
                wlast="1'b1", bready="1'b1", arvalid=f'pending[{i}]',
                araddr='probe_addr', arid="{AXI_ID_W{1'b0}}", arlen="8'b0",
                arsize="3'd5", arburst="2'b01", rready="1'b1")
    body += ''.join(f' assign {pre}{name} = {value};\n' for name, value in vals.items())
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(header + body + '\nendmodule\n`default_nettype wire\n')
