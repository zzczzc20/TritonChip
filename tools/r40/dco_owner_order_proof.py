#!/usr/bin/env python3
"""Compositional proof for the serial/rank owner selection implementations.

Membership arithmetic/load strides must be identical. After its abstraction,
base values may occur only in unsigned order comparisons, loads and the
serial winner's saved base. N bases plus the unsigned zero anchor have
<=N+1 distinct ranks; ceil(log2(N+1)) bits represent every ordering and tie.
Payloads remain unrestricted. This is an order-type proof with checked source
assumptions, not a claim that the unrestricted 64-bit SAT run completed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import dco_owner_screen as screen
R=screen.R

PREFIX="tv[k] && (a >= b) && ((z == 64'd0) || (a - b < z))"


def abstract(body, expected_memberships):
    body=re.sub(r'/\*.*?\*/|//[^\n]*','',body,flags=re.S)
    pattern=r'\s*'.join(map(re.escape,PREFIX.split()))
    body,count=re.subn(pattern,'tv[k]',body)
    if count!=expected_memberships:
        raise ValueError('membership expression changed; abstraction refused')
    compact=re.sub(r'\s+','',body)
    allowed=[
        'input[ADDR_W-1:0]a;',
        'input[DCO_T*64-1:0]tbase,tsize;',
        'input[DCO_T*64-1:0]tbase;', 'input[DCO_T*64-1:0]tsize;',
        'reg[63:0]best,b,z;', 'reg[63:0]b,z,best;',
        'b=tbase[k*64+:64];', 'z=tsize[k*64+:64];',
        "best=64'd0;", 'best=b;']
    for term in allowed:compact=compact.replace(term,'')
    for term in ('b>best','tbase[j*64+:64]>=b','tbase[j*64+:64]>b'):
        compact=re.sub(re.escape(term)+r'(?=\))','',compact)
    if re.search(r'\b(?:a|b|z|best|tbase|tsize)\b',compact):
        raise ValueError('unsupported address/base/size operation after membership cutpoint')
    if '[63:0]' in compact or re.search(r'\b64\b',compact):
        raise ValueError('unrecognized 64-bit operation would be narrowed by abstraction')
    # Only operand widths/strides change. Eligibility, payloads, control flow,
    # comparisons, slot indices and tie direction remain the actual RTL.
    body=body.replace('[63:0]','[RANK_BITS-1:0]')
    body=re.sub(r'DCO_T\s*\*\s*64','DCO_T*RANK_BITS',body)
    body=re.sub(r'\*\s*64\s*\+:\s*64','*RANK_BITS +: RANK_BITS',body)
    body=body.replace("64'd0","{RANK_BITS{1'b0}}")
    return body


def module(name,body,n,full_width=False):
    width=64 if full_width else max(1,math.ceil(math.log2(n+1)))
    return f'''module {name}(input [63:0] a,input [{n-1}:0] tv,tbyp,
input [{n*width-1}:0] tbase,tsize,input [{n*6-1}:0] tlsb,output [11:0] result);
localparam ADDR_W=64,DCO_T={n},RANK_BITS={width};
{body}
assign result=owner(a,tv,tbase,tsize,tbyp,tlsb);
endmodule
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-source',required=True,type=Path)
    ap.add_argument('--candidate-source',type=Path,default=ROOT/'rtl/nuca/tnca_nuca_slice.v')
    ap.add_argument('--entries',type=int,nargs='+',default=[1,3,7,8,16])
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--engine',choices=('sat','smt'),default='smt')
    ap.add_argument('--full-width',action='store_true',
                    help='compare actual 64-bit functions without eligibility/order abstraction')
    ap.add_argument('--z3',default='z3',help='SMT solver executable; required with --engine smt')
    args=ap.parse_args()
    if any(n<1 or n>16 for n in args.entries):ap.error('entries must be 1..16')
    if args.engine=='smt' and not shutil.which(args.z3):ap.error('Z3 executable unavailable; supply --z3 or use --engine sat')
    args.out.mkdir(parents=True,exist_ok=False)
    gold_source=args.reference_source.read_text();gate_source=args.candidate_source.read_text()
    gold=screen.function(gold_source);gate=screen.function(gate_source)
    report=dict(schema='tnca-owner-order-proof/1',scope='order-type composition with checked membership/base-operation assumptions',
        reference_sha256=hashlib.sha256(gold_source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(gate_source.encode()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        screen_dependency_sha256=hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest(),
        supervisor_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
        engine=args.engine,
        yosys_version=subprocess.check_output(['yosys','-V'],text=True,timeout=10).strip(),
        solver_version=(subprocess.check_output([args.z3,'-version'],text=True,timeout=10).strip()
                        if args.engine=='smt' else 'Yosys equiv_simple'),
        contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest(),
        gate_status='not-measured',fmax_hz=None,results=[])
    if args.full_width:report['scope']='unrestricted 64-bit function equivalence (no membership/order abstraction)'

    def proof(name,gold_model,gate_model,n):
        path=args.out/(name+'.v');path.write_text(module('gold',gold_model,n,args.full_width)+module('gate',gate_model,n,args.full_width))
        script=args.out/(name+'.ys')
        prefix=f'read_verilog -sv {json.dumps(str(path.resolve()))}; proc; opt; memory_map; opt; '
        smt=args.out/(name+'.smt2')
        command=('equiv_make gold gate equiv; hierarchy -top equiv; opt; equiv_simple; equiv_status -assert'
                 if args.engine=='sat' else
                 'miter -equiv -make_assert -flatten gold gate miter; prep -top miter; '
                 f'write_smt2 {json.dumps(str(smt.resolve()))}')
        # Lower processes before creating the miter: flattening unlowered
        # processes can discard their logic and produce a vacuous comparison.
        script.write_text(prefix+command+'\n')
        result=R.run_step(['yosys','-Q','-T','-s',str(script.resolve())],kind='build',
                         case_dir=str(args.out),name=name,cap_override=120)
        report['results'].append(dict(result))
        if args.engine=='smt' and result['status']==R.PASS:
            with smt.open('a') as stream:
                stream.write('\n(declare-fun state () |miter_s|)\n'
                             '(assert (|miter_h| state))\n'
                             '(assert (not (|miter_a| state)))\n(check-sat)\n')
            result=R.run_step([args.z3,str(smt.resolve())],kind='build',case_dir=str(args.out),
                              name=name+'_solve',cap_override=120)
            report['results'].append(dict(result))
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        text=R.read_text(result['stdout'])+R.read_text(result['stderr'])
        return result,text

    gold_model=abstract(gold,1);gate_model=abstract(gate,2)
    if args.full_width:gold_model,gate_model=gold,gate
    failed=False
    for n in args.entries:
        result,text=proof(f'order_{n}',gold_model,gate_model,n)
        passed=result['status']==R.PASS and (
            text.strip()=='unsat' if args.engine=='smt' else 'Equivalence successfully proven!' in text)
        failed|=not passed
        label='full-width' if args.full_width else 'order-type'
        print(f'{label} proof entries={n}: {"PROVEN" if passed else "UNPROVEN "+result["status"]}',flush=True)
    mutant=gate.replace('tbase[j*64 +: 64] >= b','tbase[j*64 +: 64] > b')
    if mutant==gate:raise ValueError('rank tie mutation target not found')
    result,text=proof('tie_mutation',gold_model,mutant if args.full_width else abstract(mutant,2),8)
    report['tie_mutation_caught']=(result['status']==R.PASS and text.strip()=='sat'
        if args.engine=='smt' else result['status']==R.FAIL and 'unproven' in text.lower())
    failed|=not report['tie_mutation_caught']
    for name,mutant in (
        ('membership',gate.replace('(a >= b)','(a > b)')),
        ('width-narrowing',gate.replace('integer k, j;', 'integer k, j; reg [63:0] extra;')),
        ('base-arithmetic',gate.replace('b = tbase[k*64 +: 64];',"b = tbase[k*64 +: 64]; b = b + 64'd1;"))):
        if mutant==gate:raise ValueError(name+' mutation target not found')
        try:abstract(mutant,2)
        except ValueError:report[name+'_mutation_refused']=True
        else:report[name+'_mutation_refused']=False;failed=True
    report['status']=(('full-width-proof-pass' if args.full_width else 'compositional-proof-pass')
                      if not failed else 'unproven-or-fail')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status']+'; no FPGA timing/route claim',flush=True)
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
