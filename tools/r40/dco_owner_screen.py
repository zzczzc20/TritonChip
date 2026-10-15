#!/usr/bin/env python3
"""Cheap DCO owner equivalence/depth screening before full FPGA synthesis.

Extracts actual old/new RTL functions. Generic LUT depth is diagnostic;
it cannot establish FPGA fmax, routability or R40 performance acceptance.
All subprocesses use the existing process-group deadline supervisor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/r1'))
import r1lib as R


def function(source):
    match=re.search(r'function(?:\s+automatic)?\s+\[11:0\]\s+owner;.*?endfunction',source,re.S)
    if match is None:raise ValueError('actual owner function not found')
    return match.group()


def module(name, body, entries, calls=1):
    assignments='\n'.join(f'assign result[{i*12} +: 12]=owner(a[{i*64} +: 64],tv,tbase,tsize,tbyp,tlsb);'
                          for i in range(calls))
    return f'''module {name}(input [{64*calls-1}:0] a, input [{entries-1}:0] tv,tbyp,
input [{entries*64-1}:0] tbase,tsize,input [{entries*6-1}:0] tlsb,output [{12*calls-1}:0] result);
localparam ADDR_W=64,DCO_T={entries};
localparam OWNER_LEAVES=(DCO_T>1)?(1<<$clog2(DCO_T)):1;
{body}
{assignments}
endmodule
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--reference',default='09ebe06')
    ap.add_argument('--reference-source',type=Path,help='exported reference RTL for an immutable archive without .git')
    ap.add_argument('--entries',type=int,nargs='+',default=[1,2,3,4,5,8,16])
    ap.add_argument('--shared-calls',type=int,default=1,
                    help='also map this many independent queries sharing an eight-entry configuration')
    ap.add_argument('--mapping-only',action='store_true',
                    help='skip bit-level equivalence; report mapping evidence without a proof claim')
    args=ap.parse_args()
    if any(n<1 or n>16 for n in args.entries):ap.error('entries must be 1..16')
    if not 1<=args.shared_calls<=16:ap.error('shared calls must be 1..16')
    args.out.mkdir(parents=True,exist_ok=False)
    path='rtl/nuca/tnca_nuca_slice.v'
    gold_source=(args.reference_source.read_text() if args.reference_source else
                 subprocess.check_output(['git','show',f'{args.reference}:{path}'],cwd=ROOT,text=True))
    gate_source=(ROOT/path).read_text()
    gold,gate=map(function,(gold_source,gate_source))
    report=dict(schema='tnca-dco-owner-screen/1',reference=args.reference,entries=args.entries,
                yosys_version=subprocess.check_output(['yosys','-V'],text=True,timeout=10).strip(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest(),
                mapping_only=args.mapping_only,
                reference_source_sha256=hashlib.sha256(gold_source.encode()).hexdigest(),
                candidate_source_sha256=hashlib.sha256(gate_source.encode()).hexdigest(),
                gate_status='not-measured',fmax_hz=None,route_status='not-measured',results=[])

    def step(name,commands,cap=120):
        script=args.out/(name+'.ys');script.write_text(commands+'\n')
        result=R.run_step(['yosys','-Q','-T','-s',str(script.resolve())],kind='build',
                          case_dir=str(args.out),name=name,cwd=str(ROOT),cap_override=cap)
        report['results'].append(dict(result))
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return result, R.read_text(result['stdout'])+R.read_text(result['stderr'])

    failed=False
    for entries in args.entries:
        source=args.out/f'owner_{entries}.v'
        source.write_text(module('owner_gold',gold,entries)+module('owner_gate',gate,entries))
        # Membership arithmetic is identical; compare complete owner payloads,
        # including no-match zeros, lower-index ties and per-tensor tile bits.
        if not args.mapping_only:
            result,text=step(f'equiv_{entries}',
                f'read_verilog -sv {json.dumps(str(source.resolve()))}; proc; opt; memory_map; opt; '
                'equiv_make owner_gold owner_gate equiv; hierarchy -top equiv; '
                'equiv_simple; equiv_status -assert')
            passed=result['status']==R.PASS and 'Equivalence successfully proven!' in text
            failed|=not passed
            print(f'owner entries={entries}: equivalence {"PROVEN" if passed else "UNPROVEN "+result["status"]}',flush=True)
            if not passed:continue
        for top in ('owner_gold','owner_gate'):
            result,text=step(f'depth_{entries}_{top}',
                f'read_verilog -sv {json.dumps(str(source.resolve()))}; synth -top {top} -flatten -noabc; '
                'abc -lut 6; clean; stat; ltp -noff',cap=180)
            match=re.search(r'longest topological path.*?length=(\d+)',text,re.I)
            cells=re.search(r'ABC RESULTS:\s+\$lut cells:\s+(\d+)',text)
            report['results'][-1]['lut_path_length']=int(match.group(1)) if match else None
            report['results'][-1]['generic_lut6_cells']=int(cells.group(1)) if cells else None
            failed|=result['status']!=R.PASS or match is None
            print(f'owner entries={entries} {top}: LUT depth '
                  f'{int(match.group(1)) if match else "unavailable"}',flush=True)
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Shared configuration comparisons may amortize across cache-way queries.
    # This generic wrapper is not a synthesized whole-slice area measurement.
    if args.shared_calls>1:
        source=args.out/'owner_shared.v'
        source.write_text(module('owner_gold',gold,8,args.shared_calls)+
                          module('owner_gate',gate,8,args.shared_calls))
        for top in ('owner_gold','owner_gate'):
            result,text=step(f'shared{args.shared_calls}_{top}',
                f'read_verilog -sv {json.dumps(str(source.resolve()))}; synth -top {top} -flatten -noabc; '
                'abc -lut 6; clean; stat; ltp -noff',cap=180)
            depth=re.search(r'longest topological path.*?length=(\d+)',text,re.I)
            cells=re.search(r'ABC RESULTS:\s+\$lut cells:\s+(\d+)',text)
            report['results'][-1].update(lut_path_length=int(depth.group(1)) if depth else None,
                generic_lut6_cells=int(cells.group(1)) if cells else None)
            failed|=result['status']!=R.PASS or depth is None or cells is None
            print(f'shared queries={args.shared_calls} {top}: depth '
                  f'{depth.group(1) if depth else "unknown"}, LUTs {cells.group(1) if cells else "unknown"}',flush=True)
    if args.mapping_only:
        report['status']='mapping-fail' if failed else 'mapping-only-equivalence-not-run'
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['status']+'; no FPGA timing/route claim',flush=True)
        return int(failed)
    # A tie-breaking mutation must not survive the same formal checker.
    if 'bases[right] > bases[left]' in gate:
        mutation=gate.replace('bases[right] > bases[left]','bases[right] >= bases[left]')
    else:
        mutation=gate.replace('tbase[j*64 +: 64] >= b','tbase[j*64 +: 64] > b')
    if mutation==gate:raise ValueError('tie mutation target disappeared')
    source=args.out/'tie_mutation.v'
    source.write_text(module('owner_gold',gold,8)+module('owner_mutant',mutation,8))
    result,text=step('tie_mutation',f'read_verilog -sv {json.dumps(str(source.resolve()))}; proc; opt; memory_map; opt; '
                     'equiv_make owner_gold owner_mutant equiv; hierarchy -top equiv; '
                     'equiv_simple; equiv_status -assert')
    caught=result['status']==R.FAIL and 'unproven' in text.lower()
    failed|=not caught
    report['mutation_caught']=caught
    report['status']='diagnostic-pass' if not failed else 'diagnostic-incomplete-or-fail'
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'tie mutation caught={caught}; FPGA timing/route remain not-measured',flush=True)
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
