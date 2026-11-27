import json,sys,glob,os
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d,'*.jsonl'))):
        if f.endswith('digest.jsonl'): continue
        for l in open(f):
            r=json.loads(l)
            t=r.get('time') or {}
            c=r.get('correctness') or {}
            print(os.path.basename(d), r.get('point_id'), r.get('status'), t.get('device'), c.get('max_rel_err'), c.get('tolerance'), r.get('kernels') and r['kernels'][:3])
