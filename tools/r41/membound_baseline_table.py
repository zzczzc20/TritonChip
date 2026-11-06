#!/usr/bin/env python3
"""Per-point, per-family and per-size tables of the membound board baseline.

    python3 tools/r41/membound_baseline_table.py [--write]

Reads only the archived raw outputs
(evidence/r41/hbm_bw/membound_baseline/membound_baseline_e57b64c.tgz: every
slice's batch.json, state.json and formal.jsonl) and the frozen suite
archbench/suites/r41_membound.json. For every registered point it reports the
batch verdict, the formal device time (median of the formal serial-grid
samples over the record's clock), full-link GB/s (ArchBench demand reads plus
writes over that time) and eta (ArchBench bound / device cycles, from the
record). A point with no formal record, a failed run or a definition hash that
differs from the frozen suite is listed as such, never dropped.

Domain: board, run-1 image (sha256 30e79769..., UUID 49964b17-...), 100 MHz.
"""
import argparse
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence/r41/hbm_bw/membound_baseline'
ARCHIVE = OUT/'membound_baseline_e57b64c.tgz'
SUITE = ROOT/'archbench/suites/r41_membound.json'
SIZES = ('small', 'medium', 'large')


def load():
    batches, formal, runs = {}, {}, {}
    with tarfile.open(ARCHIVE) as tar:
        for m in tar.getmembers():
            name = m.name
            if name.endswith('/batch.json'):
                batches[name] = json.load(tar.extractfile(m))
            elif name.endswith('/state.json'):
                for stem, e in json.load(tar.extractfile(m))['runs'].items():
                    runs.setdefault(e['point'], []).append(dict(e, state=name))
            elif name.endswith('/formal.jsonl'):
                for line in tar.extractfile(m).read().decode().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        formal.setdefault(r['point_id'], []).append(r)
    return batches, formal, runs


def table():
    suite = json.loads(SUITE.read_text())
    batches, formal, runs = load()
    rows = []
    for p in suite['points']:
        pid = p['id']
        family, head = pid.split('/')[0][3:], pid.split('/')[2]
        size = next(s for s in SIZES if f'-{s}-' in head)
        layout = next(l for l in ('contig', 'strided', 'broadcast') if f'-{l}-' in head)
        row = dict(point=pid, family=family, dtype=p['dtype'], size=size, layout=layout,
                   runs=len(runs.get(pid, [])), failed_runs=sum(not e['ok'] for e in runs.get(pid, [])))
        recs = formal.get(pid, [])
        if len(recs) != 1:
            row.update(status='missing' if not recs else f'{len(recs)} formal records')
        else:
            r = recs[0]
            t = r['timings']['device-cycles-serial-grid']
            d = r['demand']
            moved = d['read_bytes'] + d['write_bytes']
            seconds = t['median'] / t['clock_hz']
            problems = []
            if r['point_sha256'] != p['sha256']:
                problems.append('definition differs from the frozen suite')
            if r['evidence_level'] != 'measured-board' or r['platform']['clock_hz'] != 100000000 \
                    or not r['platform']['image_sha256'].startswith('30e79769'):
                problems.append('not a run-1 100 MHz board record')
            row.update(status=r['status'] if not problems else 'rejected', problems=problems,
                       cycles=t['median'], samples=t['samples'], seconds=seconds, bytes=moved,
                       gbps=moved / seconds / 1e9, eta=r['eta'],
                       max_rel_err=r['correctness']['max_rel_err'], tol=r['correctness']['tol'],
                       launches=r['time']['launches'])
        rows.append(row)
    return rows, batches


def summary(rows, key):
    out = {}
    for r in rows:
        k = tuple(r[x] for x in key)
        out.setdefault(k, []).append(r)
    res = []
    for k, group in sorted(out.items(), key=lambda kv: kv[0]):
        ok = [r for r in group if r['status'] == 'pass']
        g = sorted(r['gbps'] for r in ok)
        best = max(ok, key=lambda r: r['gbps']) if ok else None
        worst = min(ok, key=lambda r: r['gbps']) if ok else None
        res.append(dict(zip(key, k), points=len(group), passed=len(ok),
                        best_gbps=g[-1] if g else None, best_point=best['point'] if best else None,
                        median_gbps=statistics.median(g) if g else None,
                        worst_gbps=g[0] if g else None, worst_point=worst['point'] if worst else None,
                        geomean_eta=statistics.geometric_mean([r['eta'] for r in ok]) if ok else None))
    return res


def markdown(rows, fam_size, size, layout):
    f = lambda v: '' if v is None else f'{v:.4g}'     # noqa: E731
    out = ['| Family | Size | Points | Pass | Best GB/s | Median GB/s | Worst GB/s | Worst point | eta geomean |',
           '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for s in fam_size:
        out.append(f"| {s['family']} | {s['size']} | {s['points']} | {s['passed']} | {f(s['best_gbps'])} | "
                   f"{f(s['median_gbps'])} | {f(s['worst_gbps'])} | {(s['worst_point'] or '').split('/')[-1]} | "
                   f"{f(s['geomean_eta'])} |")
    out += ['', '| Size | Points | Pass | Best GB/s | Median GB/s | Worst GB/s |', '|---|---:|---:|---:|---:|---:|']
    for s in size:
        out.append(f"| {s['size']} | {s['points']} | {s['passed']} | {f(s['best_gbps'])} | {f(s['median_gbps'])} | "
                   f"{f(s['worst_gbps'])} |")
    out += ['', '| Layout | Size | Points | Pass | Best GB/s | Median GB/s | Worst GB/s |', '|---|---|---:|---:|---:|---:|---:|']
    for s in layout:
        out.append(f"| {s['layout']} | {s['size']} | {s['points']} | {s['passed']} | {f(s['best_gbps'])} | "
                   f"{f(s['median_gbps'])} | {f(s['worst_gbps'])} |")
    out += ['', '| Point | Status | Device cycles | Device s | GB/s | eta | max rel err / tol |',
            '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        if r['status'] in ('pass', 'rejected') and 'cycles' in r:
            out.append(f"| {r['point']} | {r['status']} | {r['cycles']} | {r['seconds']:.6f} | {r['gbps']:.4f} | "
                       f"{r['eta']:.4f} | {r['max_rel_err']:.2e} / {r['tol']} |")
        else:
            out.append(f"| {r['point']} | {r['status']} | | | | | |")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    rows, batches = table()
    fam_size = summary(rows, ('family', 'size'))
    size = summary(rows, ('size',))
    layout = summary(rows, ('layout', 'size'))
    text = markdown(rows, fam_size, size, layout)
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    head = dict(domain='board', image_sha256='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de',
                clock_hz=100000000, archive=str(ARCHIVE.relative_to(ROOT)), points=len(rows), status_counts=counts,
                batches={k: dict(status=v['status'], configurations=v['configurations'],
                                 failed=v['failed_configurations'], missing=v['missing_formal'],
                                 board_wall_seconds=v['board_wall_seconds']) for k, v in sorted(batches.items())})
    print(json.dumps(head, indent=1))
    print(text)
    if a.write:
        (OUT/'table.json').write_text(json.dumps(dict(head, points_table=rows, family_size=fam_size, size=size,
                                                      layout_size=layout), indent=1) + '\n')
        (OUT/'tables.md').write_text('<!-- generated by tools/r41/membound_baseline_table.py; board, run-1 image, '
                                     '100 MHz -->\n\n' + text + '\n')


if __name__ == '__main__':
    main()
