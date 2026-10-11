"""Score and compare ArchBench records.

    python -m archbench.report results/tnca_tiny.jsonl results/gpu.jsonl [--tier tiny]
    python -m archbench.report gpu.jsonl --freeze-weights archbench/suites/tiny.gpu-weights.json --tier tiny

Per platform: every point's status, time, strict bound and eta; then E
(weighted harmonic eta) and A (weighted arithmetic eta) through the Stage 1
scorer (tools/r38/archbench/score.py), so a point that failed or was not
measured makes the full-suite score 0 and uncertified, with the
supported-only value beside it, labelled diagnostic (roadmap section 4.8).

Weights:
    balanced        each family the same mass, its points uniform inside it
                    (roadmap section 4.6 "balanced track")
    <file>.json     frozen reference-time weights, w_i = T_ref_i / sum T_ref,
                    written by --freeze-weights from a reference platform's
                    records; the file carries its own hash and is refused if
                    edited

Across platforms, per point: both etas, and -- only when both times are in
seconds -- the time ratio. A TNCA time in cycles has no duration until a
board supplies a clock, and none is invented here.
"""
import argparse
import json
from pathlib import Path
import sys

from .bounds import r38_score as S


def load(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    # The last record of a point on a platform wins (a rerun replaces).
    latest = {}
    for r in recs:
        latest[(platform_key(r), r['point_id'])] = r
    return list(latest.values())


def platform_key(r):
    p = r.get('platform') or {}
    if p.get('kind') == 'gpu':
        return f"gpu:{p.get('spec')}:{p.get('variant')}"
    if p.get('kind') == 'tnca':
        key = f"tnca:{p.get('profile')}@{p.get('backend')}:O{p.get('opt_level')}"
        # R39-00: the Inductor configuration, the ablation knobs and the
        # device build are part of what was measured. Without them, two runs
        # of one profile that differ in any of these collapsed into one key
        # and the later silently replaced the earlier.
        extra = {k: p.get(k) for k in ('inductor', 'knobs', 'device_id') if p.get(k) is not None}
        if extra:
            import hashlib
            blob = json.dumps(extra, sort_keys=True).encode()
            key += f"#{hashlib.sha256(blob).hexdigest()[:8]}"
        return key
    return str(p.get('name') or p.get('kind'))


def to_point(r):
    t = r.get('time') or {}
    b = r.get('bound') or {}
    return S.Point(id=r['point_id'], operator=r['family'], status=r['status'],
                   time_cycles=t.get('device'), bound_cycles=b.get('value'),
                   bound_version=b.get('version'), timing=t.get('timing'))


def balanced(ids_by_family):
    return S.balanced_weights(ids_by_family)


def freeze_weights(records, out, tier):
    ref = {r['point_id']: r['time']['device'] for r in records
           if r['status'] == 'pass' and (not tier or r['tier'] in tier)}
    w = S.reference_weights(ref)
    doc = dict(schema='archbench-weights/1', kind='reference-time',
               reference=sorted({platform_key(r) for r in records}),
               note='w_i = T_ref_i / sum_j T_ref_j over the reference records (roadmap 4.6)',
               weights=w, sha256=S.weights_hash(w))
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print(f'froze {len(w)} weights -> {out}')


def fmt_time(t, unit):
    if t is None:
        return '-'
    if unit == 's':
        return f'{t * 1e6:.3g}us'
    return f'{t:.0f}cyc'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('records', nargs='+')
    ap.add_argument('--tier', nargs='+')
    ap.add_argument('--track', choices=('llm', 'general'))
    ap.add_argument('--weights', default='balanced')
    ap.add_argument('--freeze-weights', metavar='OUT')
    ap.add_argument('--json', type=Path, help='write the summary here')
    args = ap.parse_args(argv)

    recs = [r for r in load(args.records)
            if (not args.tier or r['tier'] in args.tier) and (not args.track or args.track in r['track'])]
    if args.freeze_weights:
        freeze_weights(recs, args.freeze_weights, args.tier)
        return 0

    from . import registry
    pts = registry.points(tier=args.tier, track=args.track)
    suite_ids = {p.id for p in pts}
    by_family = {}
    for p in pts:
        by_family.setdefault(p.family, []).append(p.id)
    if args.weights == 'balanced':
        weights = balanced(by_family)
    else:
        weights = S.frozen_weights(json.loads(Path(args.weights).read_text()))

    plats = sorted({platform_key(r) for r in recs})
    summary = {}
    for pk in plats:
        mine = {r['point_id']: r for r in recs if platform_key(r) == pk}
        print(f'\n== {pk}  ({len(mine)} of {len(suite_ids)} points recorded)')
        for pid in sorted(suite_ids):
            r = mine.get(pid)
            if r is None:
                print(f'  {"not-measured":17s} {pid}')
                continue
            t = r.get('time') or {}
            b = r.get('bound') or {}
            eta = r.get('eta')
            err = (r.get('correctness') or {}).get('max_rel_err')
            print(f'  {r["status"]:17s} {pid:58s} T={fmt_time(t.get("device"), t.get("unit")):>12s} '
                  f'L={fmt_time(b.get("value"), b.get("unit")):>12s} ({b.get("binding")}) '
                  f'eta={"-" if eta is None else f"{eta:.4g}"} err={"-" if err is None else f"{err:.2g}"}')
        points = [to_point(mine[i]) if i in mine else S.Point(i, '', 'not-measured') for i in sorted(suite_ids)]
        try:
            E, A, etas = S.efficiency(points, {i: weights[i] for i in suite_ids if i in weights})
        except S.BoundViolation as e:
            print(f'  AUDIT FAILURE: {e}')
            summary[pk] = dict(audit=str(e))
            continue
        n_pass = sum(1 for p in points if p.status == 'pass')
        print(f'  pass {n_pass}/{len(points)}; E={E.value:.4g} ({"certified" if E.certified else "NOT CERTIFIED"}, '
              f'supported-only {E.supported_only:.4g}, weight covered {E.coverage_weight:.3f}); '
              f'A={A.value:.4g} (supported-only {A.supported_only:.4g})')
        summary[pk] = dict(passed=n_pass, total=len(points), E=E.value, E_supported=E.supported_only,
                           A=A.value, A_supported=A.supported_only, certified=E.certified,
                           missing=E.missing, etas=etas)

    if len(plats) > 1:
        print('\n== per point across platforms (eta; time ratio only when both are in seconds)')
        for pid in sorted(suite_ids):
            row = []
            secs = []
            for pk in plats:
                r = next((x for x in recs if platform_key(x) == pk and x['point_id'] == pid), None)
                eta = r.get('eta') if r else None
                row.append('-' if eta is None else f'{eta:.3g}')
                t = (r or {}).get('time') or {}
                secs.append(t.get('device') if t.get('unit') == 's' else t.get('device_seconds'))
            ratio = ''
            if len(plats) == 2 and all(s for s in secs):
                ratio = f'  T[{plats[0]}]/T[{plats[1]}] = {secs[0] / secs[1]:.3g}'
            print(f'  {pid:58s} ' + '  '.join(f'{x:>8s}' for x in row) + ratio)
        print('  columns: ' + ', '.join(plats))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1, default=str) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
