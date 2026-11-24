#!/usr/bin/env python3
"""usage: xf_collect.py OUT.json DIR...   collect run_graphs results.json of TLM runs into {tag: {point: [cycles,status,err]}}"""
import json, sys, os
out = sys.argv[1]
res = {}
if os.path.exists(out):
    res = json.load(open(out))
for d in sys.argv[2:]:
    p = os.path.join(d, 'results.json')
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    res[os.path.basename(d.rstrip('/'))] = {x['point']: [x.get('device_cycles'), x.get('status'), x.get('max_rel_err'), x.get('problems')] for x in r['points']}
json.dump(res, open(out, 'w'), indent=1)
