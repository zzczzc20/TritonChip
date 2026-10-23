"""Compare measured launches from both trees; do not infer missing samples."""
import json
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
out = {}
for case in ('features', 'pair', 'ports'):
    logs = {tree: (root/f'{tree}-{case}.log').read_text() for tree in ('base', 'int')}
    launches = {tree: json.loads(next(line.removeprefix('AUDIT_LAUNCHES ') for line in log.splitlines()
                                     if line.startswith('AUDIT_LAUNCHES '))) for tree, log in logs.items()}
    assert len(launches['base']) == len(launches['int'])
    rows = []
    for i, (a, b) in enumerate(zip(launches['base'], launches['int'])):
        assert a['kernel'] == b['kernel'] and a['grid'] == b['grid'], (a,b)
        assert a['device_cycles'] is not None and b['device_cycles'] is not None
        rows.append(dict(index=i, kernel=a['kernel'], grid=a['grid'], base_cycles=a['device_cycles'],
                         int_cycles=b['device_cycles'], delta=b['device_cycles']-a['device_cycles'],
                         percent=100*(b['device_cycles']/a['device_cycles']-1),
                         base_host_seconds=a['host_seconds'], int_host_seconds=b['host_seconds']))
    out[case] = dict(exit_codes={tree:int((root/f'{tree}-{case}.exit').read_text()) for tree in logs},
                     assertions_pass={tree:bool(re.search(r'NATIVE (MULTICORE FEATURES|MEMORY PORTS) PASS ', log))
                                      for tree,log in logs.items()}, launches=rows)
(root/'native-compare.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
