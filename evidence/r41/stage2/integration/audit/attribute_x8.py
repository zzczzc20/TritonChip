"""Locate the first changed compiler stage for every changed X8 kernel."""
import collections
import hashlib
import json
import re
from pathlib import Path
import sys
root, source = map(Path, sys.argv[1:3])
a = json.loads((root/'int-default/report.json').read_text())
b = json.loads((root/'int-x8/report.json').read_text())
comparison = json.loads((root/'x8-compare.json').read_text())
spec = json.loads((source/'spec/r39_sched_model.json').read_text())
models = {name: spec['by_target'].get(target, spec['default']) for name, target in
          [('v5', 'tnca-sim-vrf-v5'), ('x8', 'tnca-sim-vrf-v5-x8')]}
rows, totals = {}, collections.Counter()
for point, delta in sorted(comparison['points'].items()):
    kernels = {}
    for change in delta['changed_kernels']:
        name = change['kernel']
        paths = [root/label/'runs'/point.replace('/', '__')/run['points'][point]['kernels'][name]['asm']
                 for label, run in [('int-default', a), ('int-x8', b)]]
        stages = {}
        for suffix in ('tnir', 'opt.tnir', 'spmd.tnir', 'sched.tnir'):
            files = [path.with_name(name+'.'+suffix) for path in paths]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None for path in files]
            stages[suffix] = dict(same=hashes[0]==hashes[1], v5_sha256=hashes[0], x8_sha256=hashes[1])
        text = [path.read_text() for path in paths]
        opcode_counts = {opcode:[sum(bool(re.match(r'^\s*(?:@\S+\s+)?'+re.escape(opcode)+r'\s', line)) for line in assembly.splitlines())
                                for assembly in text]
                         for opcode in ('t.ld.async.b.t', 'v.ld.gather.b')}
        kernels[name] = dict(stages=stages, opcode_counts=opcode_counts,
                             change=change['change'], text_delta_bytes=change['text_delta_bytes'])
        totals['kernels'] += 1
        for stage, row in stages.items():
            totals[stage+('_absent' if row['v5_sha256'] is None and row['x8_sha256'] is None else '_same' if row['same'] else '_different')] += 1
        for opcode, counts in opcode_counts.items():
            totals[opcode+'_v5'] += counts[0]
            totals[opcode+'_x8'] += counts[1]
    rows[point] = dict(kernels=kernels, launch_sequence_same=delta['launch_sequence_same'])
summary = dict(totals)
summary.update(points=len(rows), source_same_kernels=sum(k['triton_source_same']
               for p in comparison['points'].values() for k in p['changed_kernels']),
               resource_changed_points=sum(any('launch structure' in k['change'] for k in p['changed_kernels'])
                                           for p in comparison['points'].values()))
output = dict(scheduler_models=models, summary=summary, points=rows)
(root/'x8-attribution.json').write_text(json.dumps(output, indent=1)+'\n')
print(json.dumps(dict(scheduler_models=models, summary=summary), indent=2))
print('| Family | Points with text changes | Points with resource changes |')
print('|---|---:|---:|')
for family in sorted({p.split('/')[0] for p in rows}):
    points = [p for p in rows if p.split('/')[0] == family]
    resources = sum(any('launch structure' in k['change'] for k in comparison['points'][p]['changed_kernels'])
                    for p in points)
    print(f'| `{family}` | {len(points)} | {resources} |')
