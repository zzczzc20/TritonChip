"""Keep native test output and resource records; omit compiler caches/builds."""
from pathlib import Path
import sys
import json
import tarfile
root = Path(sys.argv[1])
peaks, record = {}, {}
def save_record():
    name, value=record.get('Id'),record.get('MemoryPeak','')
    if name and value.isdigit(): peaks[name]=max(peaks.get(name,0),int(value))
    record.clear()
for line in (root/'resource-snapshots.log').read_text().splitlines():
    if not line.strip():
        save_record()
    elif '=' in line:
        key,value=line.split('=',1)
        if key in ('Id','MemoryCurrent') and key in record: save_record()
        record[key]=value
save_record()
(root/'resource-summary.json').write_text(json.dumps(dict(recorded_cgroup_peak_bytes=peaks,
    sum_of_recorded_peaks_bytes=sum(peaks.values()),
    note='Sum of recorded lane peaks is conservative: peaks occurred at different times; cgroup memory includes more than process RSS.'),indent=2)+'\n')
with tarfile.open(root/'native-logs.tgz', 'w:gz', compresslevel=6) as tar:
    for tree in ('base', 'int'):
        for case in ('features', 'pair', 'ports'):
            for suffix in ('.log', '.exit', '.memory.log'):
                path=root/(tree+'-'+case+suffix)
                tar.add(path, arcname=path.name)
    for name in ('native-compare.json', 'native-summary.log', 'resource-snapshots.log',
                 'resource-summary.json', 'previous-integration-lsu-miter.log',
                 'run_native.py', 'run_native_gb10.sh', 'summarize_native.py',
                 'wrapper-startup-failures', 'final-source-checks.log', 'simulator-logs'):
        path=root/name
        if path.exists(): tar.add(path, arcname=name)
