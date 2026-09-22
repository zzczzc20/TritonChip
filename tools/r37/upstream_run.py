#!/usr/bin/env python3
"""Run original TileBench bodies on RTL; missing operators never count as passes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT.parent/'Tilebench')
    ap.add_argument('--only',nargs='+')
    ap.add_argument('--json',type=Path,default=ROOT/'sim/build/r37_upstream_run.json')
    ap.add_argument('--cycles',type=int,default=100000)
    args=ap.parse_args()
    import execute as E
    import upstream_cases
    table=upstream_cases.build(E.Case,args.source)
    spec=json.loads((ROOT/'spec/r37_tilebench.json').read_text())
    known={entry['id'] for entry in spec['operators']}
    if args.only and set(args.only)-known:
        ap.error('unknown operators: '+', '.join(sorted(set(args.only)-known)))
    rows=[]
    captured={}
    for entry in spec['operators']:
        oid=entry['id']
        if args.only and oid not in args.only:
            continue
        tests=[(name,c) for name,c in table.items() if c.kernel==oid]
        results=[]
        for name,case in tests:
            start=time.monotonic()
            try:
                for buf, feed in case.feeds.items():
                    previous, source_buf = feed[:2]
                    dtype, data = captured[previous][source_buf]
                    if len(feed) == 3:
                        # Match the upstream host's contiguous transpose.
                        nr,nc = feed[2]
                        data = [data[r*nc+c] for c in range(nc) for r in range(nr)]
                    case.buffers[buf] = (dtype, data)
                output={}
                ok,detail=E.run('upstream_'+name.replace('/','_'),case,
                               max_cycles=args.cycles,capture=output)
                if ok:
                    captured[name]=output
            except Exception as e:
                ok,detail=False,f'{type(e).__name__}: {e}'
            fn,sig,ce=case.kernel_spec
            result={'case':name,'ok':ok,'detail':detail,'seconds':round(time.monotonic()-start,2),
                    'source':fn.fn.__code__.co_filename,'source_sha256':hashlib.sha256(Path(fn.fn.__code__.co_filename).read_bytes()).hexdigest(),
                    'signature':sig,'constexprs':{k:str(v) for k,v in ce.items()}}
            result.update(scalars=case.scalars,program_id=case.program_id,
                          descriptors=case.descriptors,feeds=case.feeds,
                          buffers={b:{'dtype':t,'elements':len(v)} for b,(t,v) in case.buffers.items()},
                          rtol=case.rtol,atol=case.atol)
            results.append(result)
            print(f"{'ok' if ok else 'NO'} {name}: {detail}",flush=True)
        row={'id':oid,'ok':bool(results) and all(r['ok'] for r in results),'cases':results}
        if not tests:
            row['detail']='NOT RUN: numerical case not yet implemented'
            print('NO '+oid+': '+row['detail'],flush=True)
        rows.append(row)
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps({'operators':rows,'passed':sum(r['ok'] for r in rows),'total':len(rows)},indent=2)+'\n')
    passed=sum(r['ok'] for r in rows)
    print(f'Original TileBench RTL: {passed}/{len(rows)} operators pass')
    return 0 if rows and passed==len(rows) else 1


if __name__=='__main__':
    sys.exit(main())
