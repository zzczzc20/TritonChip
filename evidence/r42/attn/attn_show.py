import json, sys
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l); t = r.get("time") or {}
        print(r["point_id"], r["status"], {k: v for k, v in t.items() if not isinstance(v, (dict, list))})
        pl = t.get("per_launch") or []
        if isinstance(pl, dict): pl = list(pl.items())
        for k in pl: print("   ", json.dumps(k)[:300])
