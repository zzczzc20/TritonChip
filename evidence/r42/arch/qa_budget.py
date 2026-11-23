#!/usr/bin/env python3
"""Q-A: peak arithmetic and the resource budget of a wider dot-product engine.
Every input is a recorded number (path given); every derived number is labelled 'derived'."""
import gzip, re, json, math
R='../../../evidence/'
# ---- 1. peaks (rtl structural; evidence/r41/hbm_bw/BALANCE.md sec 2 table, tnca_matrix.v kstep) ----
LANES, CORES, F_MHZ = 16, 3, 100
print('PEAKS (3 cores x 16 lanes x G products x 2 FLOP x 100 MHz); bubble = 1 cycle per k-group of 16 row cycles')
for name,G in (('legacy m.mma.async / FP32 FMA',1),('m.mma.dp DP2 (FP16,BF16,mixed FP8)',2),('m.mma.dp DP4 (E4M3 x E4M3)',4),
               ('hypothetical G=8',8),('hypothetical G=11',11),('hypothetical G=16',16)):
    raw=CORES*LANES*G*2*F_MHZ/1e3
    print('  %-40s G=%2d  %7.1f GFLOP/s raw   %7.1f with the 16/17 bubble   %5.0f MAC/cycle/core'%(name,G,raw,raw*16/17,LANES*G))
need=100e9/(2*CORES*LANES*F_MHZ*1e6)
print('  G needed for 100 GFLOP/s peak at 100 MHz, 3 cores, 16 lanes: %.2f ; at 130 MHz: %.2f ; with 6 cores at 100 MHz: %.2f'%(need,need*100/130,need/2))
# ---- 2. SLR packing of the run-2 link (same RTL as run 3 plus one line, evidence/r41/hbm_bw/g4/run2/link_1f0a5aff) ----
txt=gzip.open('../../../evidence/r41/hbm_bw/g4/run2/link_1f0a5aff/utilization_slr.rpt.gz','rt').read()
def row(label):
    m=re.search(r'^\|\s*'+re.escape(label)+r'\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|',txt,re.M)
    return [int(m.group(i)) for i in (1,2,3)],[float(m.group(i)) for i in (4,5,6)]
clb,clbp=row('CLB'); lut,lutp=row('CLB LUTs'); dsp,dspp=row('DSPs'); ff,ffp=row('CLB Registers')
print('\nSLR packing, routed run-2 link (domain: routed image, not run 3 itself):')
ceil=96.74
tot_free=0
for i in range(3):
    clb_avail=clb[i]/clbp[i]*100; lut_avail=lut[i]/lutp[i]*100; dsp_avail=dsp[i]/dspp[i]*100
    dens=lut[i]/clb[i]
    free_clb=(ceil-clbp[i])/100*clb_avail
    print('  SLR%d: CLB %d (%.2f%%) of %.0f; LUT %d (%.2f%%) of %.0f; DSP %d (%.2f%%) of %.0f (derived); LUT/CLB %.2f'%(i,clb[i],clbp[i],clb_avail,lut[i],lutp[i],lut_avail,dsp[i],dspp[i],dsp_avail,dens))
    print('        headroom to the highest recorded routed packing (%.2f%%, run 1): %.0f CLB = %.0f LUT at the current density (derived)'%(ceil,free_clb,free_clb*dens))
    tot_free+=free_clb*dens
print('  total DSPs used in the three SLRs: %d (incl. platform); device 9,024'%sum(dsp))
# ---- 3. per-lane and per-product costs (recorded) ----
stage={'S1':689,'S2+F2':639,'B0':448,'norm':526,'round':490}   # fma_dslx/README.md sec 4, unit screens
lane=sum(stage.values()); backend=stage['B0']+stage['norm']+stage['round']
print('\nR41 shared lane (m.mma.dp): unit-screen stages %s = %d LUT, 8 DSP (RESOURCES.md sec 10: 2 -> 8 DSP per lane); back end (align with C, normalise, round) %d LUT = %.0f%%'%(stage,lane,backend,100*backend/lane))
V={'V0 FMA lane':2492,'V3 E4M3 DP4 (38-bit exact field)':2929,'V1 all-FP8 DP4 (66-bit field)':3433,'V2D FP16 DP2 + FP8 (64-bit field)':4142}
base=V['V0 FMA lane']
print('pmac prototype lanes (OOC routed, evidence/r41/pmac/README.md sec 7):')
perprod={}
for k,v in V.items():
    g={'V0 FMA lane':1,'V3 E4M3 DP4 (38-bit exact field)':4,'V1 all-FP8 DP4 (66-bit field)':4,'V2D FP16 DP2 + FP8 (64-bit field)':2}[k]
    extra=(v-base)/(g-1) if g>1 else 0
    perprod[k]=extra
    print('   %-38s %5d LUT  G=%d  marginal LUT per extra product %6.0f (derived)  LUT per MAC/cycle %6.0f (derived)'%(k,v,g,extra,v/g))
# ---- 4. what a wider group costs, per core (16 lanes), derived ----
print('\nBudget of raising G per lane (16 lanes per core), derived from the marginal costs above:')
hdr=[(0,'SLR0'),(1,'SLR1'),(2,'SLR2')]
free=[(ceil-clbp[i])/100*(clb[i]/clbp[i]*100)*(lut[i]/clb[i]) for i in range(3)]
for label,key,G0,G1 in (('E4M3 exact DSP cascade',list(perprod)[1],4,8),('E4M3 exact DSP cascade',list(perprod)[1],4,16),
                       ('mixed-FP8 66-bit field',list(perprod)[2],4,8),('mixed-FP8 66-bit field',list(perprod)[2],4,16),
                       ('FP16 exact 64-bit field',list(perprod)[3],2,4),('FP16 exact 64-bit field',list(perprod)[3],2,8)):
    c=perprod[key]*(G1-G0)*16
    print('  %-26s G %2d -> %2d: +%7.0f LUT per core; SLR headroom %s ; peak %.1f -> %.1f GFLOP/s'%(label,G0,G1,c,' / '.join('%s %.0f'%(n,free[i]) for i,n in hdr),CORES*LANES*G0*2*F_MHZ/1e3,CORES*LANES*G1*2*F_MHZ/1e3))
print('  headroom per SLR (derived): '+', '.join('SLR%d %.0f LUT'%(i,free[i]) for i in range(3)))

# ---- 5. operand supply of a wider group (derived from tnca_matrix.v: b_row is latched once per k-group and held for 16 row cycles;
#         A element word per cycle from ABUF; one 512-bit vector register = 64 B) ----
print('\nOperand supply per core for G products per lane per cycle (derived; the engine holds B for 16 row cycles + 1 load cycle):')
print('  %-8s %3s | %10s %10s %14s | %12s'%('format','G','B bytes/grp','VRF regs','B VRF B/cycle','A B/cycle (ABUF)'))
for fmt,eb in (('E4M3',1),('FP16',2)):
    for G in (2,4,8,16):
        if fmt=='E4M3' and G<4: continue
        b=LANES*G*eb
        print('  %-8s %3d | %10d %10.0f %14.1f | %12.0f'%(fmt,G,b,b/64,b/17,G*eb))
print('  today (DP4 E4M3, G=4): 64 B per group = 1 register, 3.8 B/cycle of the 64 B/cycle matrix read port (BALANCE.md sec 3.1 gives 5.6-6.8 B/cycle incl. the A fill).')
