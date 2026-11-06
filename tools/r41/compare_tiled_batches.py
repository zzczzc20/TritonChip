#!/usr/bin/env python3
"""Compare complete same-image default and tiled board batches by frozen point."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path, batch):
    with tarfile.open(path) as archive:
        summary = json.load(archive.extractfile(f'{batch}/batch.json'))
        rows = [json.loads(line) for line in archive.extractfile(f'{batch}/formal.jsonl')]
    if (summary.get('status') != 'complete' or summary.get('failed_configurations') != []
            or summary.get('missing_formal') != [] or len(rows) != summary.get('configurations')):
        raise ValueError('batch is incomplete or contains failures: '+str(path))
    records = {r['point_id']: r for r in rows}
    if len(records) != len(rows) or any(r.get('status') != 'pass' or r.get('fallbacks') for r in rows):
        raise ValueError('duplicate or failing formal point: '+str(path))
    return records


def compare(default, tiled, batch):
    a, b = read(default,batch), read(tiled,batch)
    if set(a) != set(b):
        raise ValueError('point inventories differ')
    rows = []
    for name in sorted(a):
        x,y = a[name],b[name]
        px,py = x['platform'],y['platform']
        kx,ky = dict(px['knobs']),dict(py['knobs'])
        if ky.pop('TNCA_TILED_POINTWISE',None) != '2' or kx != ky:
            raise ValueError('knobs differ beyond tiled pointwise: '+name)
        for key in ('image_sha256','software_tree','profile','backend','backend_args','clock_hz'):
            if px.get(key) in (None, '', {}) or px.get(key) != py.get(key):
                raise ValueError('platform identity differs: '+name+' '+key)
        if x['point_sha256'] != y['point_sha256'] or x['demand'] != y['demand']:
            raise ValueError('point definition or logical demand differs: '+name)
        tx,ty = x['timings']['device-cycles-serial-grid'],y['timings']['device-cycles-serial-grid']
        if (tx['clock_hz'] != 100000000 or ty['clock_hz'] != 100000000 or tx['median'] <= 0
                or ty['median'] <= 0 or not tx.get('samples') or not ty.get('samples')):
            raise ValueError('invalid clock or cycle count: '+name)
        rows.append(dict(point=name,point_sha256=x['point_sha256'],
                         logical_bytes=x['demand']['read_bytes']+x['demand']['write_bytes'],
                         default_cycles=tx['median'],tiled2_cycles=ty['median'],
                         default_samples=tx['samples'],tiled2_samples=ty['samples'],
                         tiled2_over_default=ty['median']/tx['median']))
    return dict(schema='r41-board-tiled2-batch-comparison/1',batch=batch,
        scope='same image, software tree and 100 MHz; tiled pointwise knob 2 versus default; descriptive medians, not statistical or complete-suite acceptance',
        default_archive_sha256=sha(default),tiled2_archive_sha256=sha(tiled),points=len(rows),
        sum_cycles_speedup=sum(r['default_cycles'] for r in rows)/sum(r['tiled2_cycles'] for r in rows),
        geometric_mean_speedup=math.exp(statistics.mean(math.log(r['default_cycles']/r['tiled2_cycles']) for r in rows)),
        slower_points=sum(r['tiled2_over_default']>1 for r in rows),
        slower_over_5_percent=sum(r['tiled2_over_default']>1.05 for r in rows),
        slower_over_10_percent=sum(r['tiled2_over_default']>1.1 for r in rows),rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--default',type=Path,required=True)
    p.add_argument('--tiled',type=Path,required=True)
    p.add_argument('--batch',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=compare(a.default,a.tiled,a.batch)
    with a.out.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','scope')}))


if __name__ == '__main__':
    main()
