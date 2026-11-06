#!/usr/bin/env python3
"""Register wrapper for a combinational module (R41 G2 unit screens).

  regwrap.py MODULE.v MODULE > wrap.v

Every input and every output of MODULE gets a flop, clocked by `clk`, so an
out-of-context run measures the module register to register (top: MODULE_rw).
Ports are read from the `module NAME(...)` header (XLS emits `input wire
[N:0] x,` and `output wire [N:0] out`).
"""
import re
import sys

src, top = sys.argv[1], sys.argv[2]
text = open(src).read()
m = re.search(r'module\s+' + re.escape(top) + r'\s*(?:#\s*\(.*?\)\s*)?\((.*?)\);', text, re.S)
if not m:
    sys.exit(f'no module {top} in {src}')
ports = []
for dirn, width, name in re.findall(r'(input|output)\s+wire\s*(\[[^\]]*\])?\s*(\w+)', m.group(1)):
    if name not in ('clk', 'rst_n', 'rst'):
        ports.append((dirn, width or '', name))
has_clk = re.search(r'\bclk\b', m.group(1)) is not None
has_rstn = re.search(r'\brst_n\b', m.group(1)) is not None
out = [f'module {top}_rw(input wire clk,']
decl = []
for d, w, n in ports:
    decl.append(f'  {d} wire {w} {n}_p')
out.append(',\n'.join(decl) + ');')
for d, w, n in ports:
    out.append(f'  reg {w} {n}_q;')
    out.append(f'  wire {w} {n}_w;')
for d, w, n in ports:
    if d == 'input':
        out.append(f'  always @(posedge clk) {n}_q <= {n}_p;')
        out.append(f'  assign {n}_w = {n}_q;')
    else:
        out.append(f'  always @(posedge clk) {n}_q <= {n}_w;')
        out.append(f'  assign {n}_p = {n}_q;')
conn = ', '.join(f'.{n}({n}_w)' for d, w, n in ports)
if has_clk:
    conn = '.clk(clk), ' + conn
if has_rstn:
    conn = ".rst_n(1'b1), " + conn
out.append(f'  {top} u_dut({conn});')
out.append('endmodule')
print('\n'.join(out))
