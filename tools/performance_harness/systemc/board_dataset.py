#!/usr/bin/env python3
"""Board timing dataset for calibrating and validating the SystemC model.

    board_dataset.py --records 'GLOB' [--records 'GLOB' ...] --root REPO --out dataset.json

Reads ArchBench protocol-2 board records (JSONL) and keeps, per record, what a
timing model is fitted and judged against: point and its definition hash,
configuration knobs, identity (software tree, image, measured clock), every
launch's kernel, grid and device cycles, the device graph makespan, and the
host-side counters ArchBench recorded. Only passing records without host
fallback are kept; every source file is listed with its sha256.

Each point is labelled with the frozen R40 split it belongs to
(spec/r40/suites/R40-04.calibration.json, R40-04.model-holdout.json). Fit only
on `calibration`; never on `holdout` (doc/r40_tasks_and_acceptance.md sec 3.4).
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path


def split_of(root):
    out = {}
    for name in ('calibration', 'model-holdout'):
        for p in json.loads((Path(root)/f'spec/r40/suites/R40-04.{name}.json').read_text())['points']:
            out[p['id']] = 'holdout' if name == 'model-holdout' else 'calibration'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records', action='append', required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    split = split_of(a.root)
    files = sorted({f for g in a.records for f in glob.glob(g, recursive=True)})
    rows, sources = [], {}
    for f in files:
        raw = Path(f).read_bytes()
        sources[f] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('status') != 'pass' or r.get('fallbacks'):
                continue
            plat, t = r.get('platform') or {}, r.get('time') or {}
            tims = r.get('timings') or {}
            graph = (tims.get('device-graph-makespan') or {}).get('median')
            rows.append(dict(
                point=r['point_id'], point_sha256=r.get('point_sha256'), family=r.get('family'),
                tier=r.get('tier'), split=split.get(r['point_id'], 'other'),
                knobs=plat.get('knobs') or {}, software_tree=plat.get('software_tree'),
                image_sha256=plat.get('image_sha256'), clock_hz=(r.get('provenance') or {}).get('clock_hz'),
                cores=((r.get('provenance') or {}).get('launch_map') or {}).get('cores'),
                serial_grid_cycles=(tims.get('device-cycles-serial-grid') or {}).get('median'),
                graph_makespan_cycles=graph,
                launches=[dict(kernel=l.get('kernel'), grid=l.get('grid'), cycles=l.get('cycles'))
                          for l in t.get('per_launch') or []],
                counters=r.get('counters'), source=f))
    doc = dict(schema='tnca-board-timing-dataset/1', records=len(rows), sources=sources,
               splits={s: sum(r['split'] == s for r in rows) for s in ('calibration', 'holdout', 'other')},
               rule='fit on calibration only; holdout is never used for fitting', rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc)+'\n')
    print(json.dumps({k: v for k, v in doc.items() if k in ('records', 'splits')}))


if __name__ == '__main__':
    main()
