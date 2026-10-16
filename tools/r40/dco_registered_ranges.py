#!/usr/bin/env python3
"""Source-bound endpoint/membership proof and owner selection composition.

A registered endpoint is usable only when it belongs to the same BASE/SIZE
commit. The sequential TMU bench checks that invariant separately. This tool
proves unsigned-64 membership, then abstracts that proven predicate and proves
the actual owner selection for every table size 1..16. No FPGA timing claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import dco_owner_order_proof as O

ROOT=Path(__file__).resolve().parents[2]
R=O.R
NEW="tv[k] && (a >= b) && (tend[k*65+64] || (a < tend[k*65 +: 64]))"


def extract_function(source,name):
    match=re.search(r'function(?:\s+automatic)?\s+\[11:0\]\s+'+name+r';.*?endfunction',source,re.S)
    if match is None:raise ValueError('function missing: '+name)
    return match[0]


def selection_model(body):
    body=re.sub(r'/\*.*?\*/|//[^\n]*','',body,flags=re.S)
    pattern=r'\s*'.join(map(re.escape,NEW.split()))
    body,count=re.subn(pattern,O.PREFIX,body)
    if count!=1:raise ValueError('registered membership predicate changed')
    body,count=re.subn(r'input\s*\[DCO_T\*65-1:0\]\s*tend;',
                       'input [DCO_T*64-1:0] tsize;',body)
    if count!=1 or re.search(r'\btend\b',body):raise ValueError('unexpected endpoint use')
    body=body.replace('owner_registered','owner')
    return O.abstract(body,1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--z3',default='z3')
    args=ap.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    slice_path=ROOT/'rtl/nuca/tnca_nuca_slice.v';tmu_path=ROOT/'rtl/nuca/tnca_tmu.v'
    old=extract_function(slice_path.read_text(),'owner')
    helper_path=ROOT/'rtl/common/tnca_dco_owner_registered.vh'
    new=extract_function(helper_path.read_text(),'owner_registered')
    compact=lambda value:re.sub(r'\s+','',value)
    slice_text=compact(slice_path.read_text());tmu_text=compact(tmu_path.read_text())
    bindings=(
        'owner_registered(cur_addr,dco_t_valid,dco_t_base,dco_t_end,dco_t_bypass,dco_t_lsb)',
        'owner_registered(line_addr(lookup_tag[owner_bank],cur_set),dco_t_valid,dco_t_base,dco_t_end,dco_t_bypass,dco_t_lsb)')
    if any(text not in slice_text for text in bindings):raise ValueError('slice endpoint consumer binding changed')
    if 'owner_registered(n_addr,t_valid,t_base,t_end,t_bypass,t_lsb)' not in tmu_text:
        raise ValueError('TMU endpoint consumer binding changed')
    assignment=re.search(r'te\[rg_slot\]\s*<=\s*(.*?);',tmu_path.read_text(),re.S)
    if assignment is None:raise ValueError('actual TMU endpoint assignment missing')
    endpoint=assignment[1].replace('st_base[w_src]','b').replace('st_size[w_src]','z')
    # default_nettype none and Icarus reject any residual unknown producer input.
    predicate="(a >= b) && (endpoint[64] || (a < endpoint[63:0]))"
    prefix="`default_nettype none\n"
    member=f'''module member_gold(input [63:0] a,b,z,output result);
assign result=(a >= b) && ((z == 64'd0) || (a-b < z));
endmodule
module member_gate(input [63:0] a,b,z,output result);
wire [64:0] endpoint={endpoint};
assign result={predicate};
endmodule
'''
    report=dict(schema='tnca-registered-range-proof/1',scope='unsigned-64 membership plus source-checked order composition; TMU commit invariant required',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (slice_path,tmu_path,helper_path,Path(__file__),Path(O.__file__),Path(O.screen.__file__))},
        yosys=subprocess.check_output(['yosys','-V'],text=True).strip(),
        z3=subprocess.check_output([args.z3,'-version'],text=True).strip(),
        contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest(),
        results=[],gate_status='not-measured',fmax_hz=None)
    def save():
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def proof(name,source,expected='unsat'):
        path=args.out/(name+'.v');path.write_text(prefix+source)
        result=R.run_step(['iverilog','-g2012','-s','member_gate' if name.startswith('member') else 'gate',
            '-o',str(args.out/(name+'.vvp')),str(path)],kind='build',case_dir=str(args.out),name=name+'_parse',cap_override=30)
        report['results'].append(dict(result));save()
        if result['status']!=R.PASS:return False
        gold,gate=('member_gold','member_gate') if name.startswith('member') else ('gold','gate')
        smt=args.out/(name+'.smt2');ys=args.out/(name+'.ys')
        ys.write_text(f'read_verilog -sv {json.dumps(str(path))}; proc; opt; memory_map; opt; '
            f'miter -equiv -make_assert -flatten {gold} {gate} miter; prep -top miter; write_smt2 {json.dumps(str(smt))}\n')
        result=R.run_step(['yosys','-Q','-T','-s',str(ys)],kind='build',case_dir=str(args.out),name=name+'_build',cap_override=120)
        report['results'].append(dict(result));save()
        if result['status']!=R.PASS:return False
        with smt.open('a') as stream:stream.write('\n(declare-fun state () |miter_s|)\n(assert (|miter_h| state))\n(assert (not (|miter_a| state)))\n(check-sat)\n')
        result=R.run_step([args.z3,str(smt)],kind='build',case_dir=str(args.out),name=name+'_solve',cap_override=120)
        report['results'].append(dict(result));save()
        passed=result['status']==R.PASS and R.read_text(result['stdout']).strip()==expected
        print(name+': '+('PROVEN' if passed and expected=='unsat' else 'MUTATION CAUGHT' if passed else 'UNPROVEN'),flush=True)
        return passed
    checks={}
    checks['membership']=proof('membership',member)
    checks['overflow_mutation']=proof('membership_overflow_mutation',member.replace('endpoint[64] ||',"1'b0 ||"),'sat')
    checks['zero_mutation']=proof('membership_zero_mutation',member.replace("{1'b1, 64'd0}","{1'b0, 64'd0}"),'sat')
    if not all(checks.values()):report['checks']=checks;save();return 1
    gold=O.abstract(old,2);gate=selection_model(new)
    for n in range(1,17):
        checks['order_'+str(n)]=proof('order_'+str(n),O.module('gold',gold,n)+O.module('gate',gate,n))
    mutant=new.replace('tbase[j*64 +: 64] >= b','tbase[j*64 +: 64] > b')
    if mutant==new:raise ValueError('tie mutation target disappeared')
    checks['tie_mutation']=proof('tie_mutation',O.module('gold',gold,8)+O.module('gate',selection_model(mutant),8),'sat')
    report['checks']=checks;report['status']='compositional-proof-pass' if all(checks.values()) else 'unproven-or-fail';save()
    return int(not all(checks.values()))


if __name__=='__main__':raise SystemExit(main())
