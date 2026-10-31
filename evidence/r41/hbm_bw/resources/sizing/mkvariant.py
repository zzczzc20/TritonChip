# Scratch only (R41 G2 sizing experiment): add a profile that is
# core-vrf-sync-v5-x8 with elaboration overrides, then regenerate targets.
import json, sys
name, overrides = sys.argv[1], json.loads(sys.argv[2])
p = 'spec/r1_profiles.json'
d = json.load(open(p))
base = json.loads(json.dumps(d['profiles']['core-vrf-sync-v5-x8']))
base['target_file'] = f'tnca-sim-{name}.json'
base['executes_artifacts_of'] = []
base['description'] = f'R41 G2 sizing experiment: core-vrf-sync-v5-x8 with {overrides}'
base['elaboration'].update(overrides)
d['profiles'][name] = base
json.dump(d, open(p, 'w'), indent=1)
open(p, 'a').write('\n')
