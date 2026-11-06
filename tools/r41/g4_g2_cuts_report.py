#!/usr/bin/env python3
"""G2's capability cuts one at a time on the run-1 image (RTL link probes).

    python3 tools/r41/g4_g2_cuts_report.py evidence/r41/hbm_bw/g4/g2_cuts

Reads <dir>/<arm>.w*.jsonl (tools/r41/link_budget_rtl.py records) for the arms
base (the run-1 integration image), ctx2, bar8e38, epoch38, addr35, shread2,
lutram, and evidence/r41/hbm_bw/g4/rtl_s4/controls/g2.mode0*.jsonl for all of
G2's cuts together; prints device cycles and run-1/arm ratios (above 1 is
faster). Domain: RTL simulation with the testbench memory model.
"""
import glob,sys,json,statistics
sys.path.insert(0,'tools/r41')
from link_budget_report import rtl_rows
D=sys.argv[1]
arms=['base','ctx2','bar8e38','epoch38','addr35','shread2','lutram']
label={'base':'run-1 image (rerun)','ctx2':'contexts 4→2 (NUM_RCTX 2, RCTX_W 1, PHYS_VREGS 128, PHYS_W 7; profile -g2)','bar8e38':'barriers 32→8 with 38-bit epochs','epoch38':'38-bit epochs alone','addr35':'35-bit home address','shread2':'NUCA_SHARED_DATA_READ 2','lutram':'AXI_LINE_LUTRAM 1'}
data={a:{r['point']:r for r in rtl_rows(sorted(glob.glob(f'{D}/{a}.w*.jsonl')))} for a in arms}
g2={r['point']:r for r in rtl_rows(sorted(glob.glob('evidence/r41/hbm_bw/g4/rtl_s4/controls/g2.mode0*.jsonl')))}
pts=sorted(data['base'])
short=lambda p:p.split('/')[0][3:]+' '+p.split('/')[2].split('-small')[0]+(' strided' if 'strided' in p else '')
print('| Image (one cut on the run-1 image) | '+' | '.join(short(p) for p in pts)+' | geometric mean |')
print('|---|'+'---:|'*(len(pts)+1))
for a in arms+['g2all']:
    src=g2 if a=='g2all' else data[a]
    cells=[];rs=[]
    for p in pts:
        b=data['base'].get(p,{}).get('cycles');c=src.get(p,{}).get('cycles')
        if b and c: r=b/c;rs.append(r);cells.append(f'{c:,} ({r:.2f}x)')
        else: cells.append(src.get(p,{}).get('status') or '–')
    gm=f'{statistics.geometric_mean(rs):.3f}' if len(rs)==len(pts) else '–'
    print(f"| {label.get(a,'all G2 cuts (image_g2.tcl, rtl_s4/controls)')} | "+' | '.join(cells)+f' | {gm} |')
