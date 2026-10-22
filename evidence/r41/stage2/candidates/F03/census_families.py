#!/usr/bin/env python3
# census_families.py RESIDUAL_TSV OUT_JSON -- the census families of the FMA datapath
# (F03/F08/F09 and their neighbours) from attempt 3's residual_failing_endpoints.tsv
# (evidence/r40/v3/final_image_candidate/census/census.tgz): endpoints, TNS, worst slack
# and that endpoint's routed data-path delay.
import csv,re,sys,json
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
fams={
 'F03 red_busy->VFPU p0 (all)':(r'u_vfpu/red_busy_reg',r'u_vfpu/g_fma\[\d+\]\.u_fma/g_l2\.u_xls/p0_'),
 'F03 red_busy->p0_small_wins':(r'u_vfpu/red_busy_reg',r'u_vfpu/.*p0_small_wins_reg'),
 'F03 red_busy->p0_res_sign_mag':(r'u_vfpu/red_busy_reg',r'u_vfpu/.*p0_res_sign_mag_reg'),
 'F03 red_busy->p0_sub_671':(r'u_vfpu/red_busy_reg',r'u_vfpu/.*p0_sub_671_reg'),
 'F03-adj x_va/x_vb->VFPU p0':(r'u_core/g_vrf_banked\.x_v[ab]_q_reg',r'u_vfpu/.*u_xls/p0_'),
 'F03-adj x_fpctl->VFPU p0':(r'u_core/x_fpctl_reg',r'u_vfpu/.*u_xls/p0_'),
 'F08 VFPU p0->p1':(r'u_vfpu/.*u_xls/p0_',r'u_vfpu/.*u_xls/p1_'),
 'F08 VFPU p0_sticky_al->p1_sh_mag':(r'u_vfpu/.*u_xls/p0_sticky_al_reg',r'u_vfpu/.*u_xls/p1_sh_mag_reg'),
 'F08 MTX p0->p1':(r'u_mtx/.*u_xls/p0_',r'u_mtx/.*u_xls/p1_'),
 'F09 MTX p1_sh_mag->lane_mem':(r'u_mtx/.*u_xls/p1_sh_mag_reg',r'u_mtx/u_acc/.*lane_mem'),
 'F09 MTX p1->lane_mem (all)':(r'u_mtx/.*u_xls/p1_',r'u_mtx/u_acc/.*lane_mem'),
 'F09-adj VFPU p1_sh_mag->w_vdata':(r'u_vfpu/.*u_xls/p1_sh_mag_reg',r'u_core/w_vdata_reg'),
 'F09-adj VFPU p1->acc (reduction)':(r'u_vfpu/.*u_xls/p1_',r'u_vfpu/acc_reg'),
 'F09-adj VFPU p1->any':(r'u_vfpu/.*u_xls/p1_',r'.'),
 'MTX a_q->p0 (stage 0)':(r'u_mtx/g_mac\[\d+\]\.u_mac/g_in_reg\.[ab]_q_reg',r'u_mtx/.*u_xls/p0_'),
 'VFPU cvt->dly_v':(r'u_vfpu/cvt_',r'u_vfpu/dly_v_reg'),
 'VFPU x_va->dly_v':(r'u_core/g_vrf_banked\.x_va_q_reg',r'u_vfpu/dly_v_reg'),
}
out={}
for k,(sp,ep) in fams.items():
    sel=[r for r in rows if re.search(sp,r['startpoint']) and re.search(ep,r['endpoint'])]
    if not sel: out[k]=None; print(f"{k:42s} none"); continue
    w=min(sel,key=lambda r:float(r['slack_ns']))
    tns=sum(float(r['slack_ns']) for r in sel)
    out[k]=dict(endpoints=len(sel),tns=round(tns,3),worst_slack=float(w['slack_ns']),worst_datapath=float(w['datapath_ns']),levels=int(w['logic_levels']),start=w['startpoint'].split('u_core/')[-1],end=w['endpoint'].split('u_core/')[-1])
    print(f"{k:42s} n={len(sel):5d} tns={tns:9.2f} worst={float(w['slack_ns']):7.3f} dp={float(w['datapath_ns']):7.3f} lv={w['logic_levels']}")
json.dump(out,open(sys.argv[2],'w'),indent=1)
