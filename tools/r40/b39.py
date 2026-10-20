#!/usr/bin/env python3
"""B39, the frozen R39 baseline, as R40's protocol can use it.

    python3 tools/r40/b39.py freeze --records R.jsonl [R.jsonl ...] --image <json> --out spec/r40
    python3 tools/r40/b39.py wrap --baseline spec/r40/baseline.json [--tier tiny] --out B39.jsonl

doc/r40_tasks_and_acceptance.md sec 3.1: B39 is R39's signed-off software, its
on-chip-RAM image and its configuration on the CPU migration path. R39's
records are protocol 1: deterministic, one launch sequence per point, device
cycles summed over serial-grid launches, and no provenance block. `freeze`
copies the board records of the final image into spec/r40/b39/<tier>.jsonl
(host log paths removed, as R39-00 froze its own) and writes
spec/r40/baseline.json: the commit and tree, every compile and RTL switch,
the image, its timing and clock, the toolchain, and the open items.

`wrap` gives each record the provenance block protocol 2 requires, filled
from baseline.json -- the one place those facts are recorded -- so
tools/r40/ablate.py can compare against B39 with --legacy-a. It invents
nothing: a fact baseline.json does not hold stays missing, and ablate.py
refuses the record.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def tree_digest(commit):
    """archbench.provenance.software_tree() of `commit`, exported clean: the
    name protocol-2 records give the same sources."""
    import tarfile
    import tempfile
    sys.path.insert(0, str(ROOT))
    from archbench.provenance import software_tree
    with tempfile.TemporaryDirectory() as d:
        tar = Path(d) / 'src.tar'
        subprocess.run(['git', 'archive', '-o', str(tar), commit], cwd=ROOT, check=True)
        with tarfile.open(tar) as t:
            t.extractall(Path(d) / 'src', filter='data')
        return software_tree(Path(d) / 'src')


def b4(board, sim_path):
    """Board against device simulation, point by point, in cycles."""
    sim = {}
    for line in Path(sim_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            sim[r['point_id']] = r
    same, diff = 0, []
    for pid, r in sorted(board.items()):
        if pid not in sim:
            continue
        a, b = (r.get('time') or {}).get('device'), (sim[pid].get('time') or {}).get('device')
        if a == b:
            same += 1
        else:
            diff.append(dict(point=pid, board=a, sim=b))
    return dict(identical=same, different=diff, compared=same + len(diff))


def freeze(args):
    out = Path(args.out)
    (out / 'b39').mkdir(parents=True, exist_ok=True)
    image = json.loads(Path(args.image).read_text())
    tiers = {}
    knobs = None
    for path in args.records:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            (r.get('host') or {}).pop('log', None)
            tiers.setdefault(r['tier'], {})[r['point_id']] = r
            k = (r.get('platform') or {}).get('knobs')
            if knobs is None:
                knobs = k
            elif k != knobs:
                sys.exit(f'{path}: {r["point_id"]} was measured with other knobs; B39 is one '
                         'configuration')
    counts = {}
    for tier, recs in sorted(tiers.items()):
        lines = [json.dumps(recs[pid]) for pid in sorted(recs)]
        (out / 'b39' / f'{tier}.jsonl').write_text('\n'.join(lines) + '\n')
        counts[tier] = dict(points=len(recs),
                            passing=sum(1 for r in recs.values() if r['status'] == 'pass'),
                            sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())
    commit = args.commit or git('rev-parse', 'HEAD')
    extra = {}
    if args.b4_sim:
        ref = out / 'b39' / 'b4_sim_tiny.jsonl'
        lines = []
        for line in Path(args.b4_sim).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                (r.get('host') or {}).pop('log', None)
                lines.append(json.dumps(r))
        ref.write_text('\n'.join(sorted(lines, key=lambda l: json.loads(l)['point_id'])) + '\n')
        extra['b4'] = dict(reference='spec/r40/b39/b4_sim_tiny.jsonl',
                           what='device-level RTL simulation of the same RTL and software, tiny, '
                                'each point from reset (archbench.run --jobs 6)',
                           **b4(tiers.get('tiny', {}), ref))
    for name, src in (('b1_conformance', args.b1), ('resources', args.resources)):
        if src:
            dst = out / 'b39' / f'{name}.json'
            dst.write_text(Path(src).read_text())
            extra[name] = f'spec/r40/b39/{name}.json'
    doc = dict(
        schema='tnca-r40-baseline/1',
        name='B39',
        what=('The frozen R39 baseline (doc/r40_tasks_and_acceptance.md sec 3.1): R39\'s '
              'signed-off software, its final board image and configuration, on the CPU '
              'migration path (torch CPU tensors -> Inductor cpu_backend="triton" -> tnca).'),
        software=dict(commit=commit,
                      tree=args.tree or git('rev-parse', f'{commit}^{{tree}}'),
                      tree_digest=tree_digest(commit),
                      merged=args.merged,
                      knobs=knobs,
                      defaults=json.loads(Path(args.defaults).read_text()) if args.defaults else None),
        image=image,
        toolchain=json.loads(Path(args.toolchain).read_text()) if args.toolchain else None,
        records={t: dict(path=f'spec/r40/b39/{t}.jsonl', **c) for t, c in counts.items()},
        timing=dict(kind='device-cycles-serial-grid',
                    note=('protocol 1: the sum over the call\'s launches of launch_accept_to_core_idle '
                          'in device cycles; programs of a grid run one after another; one deterministic '
                          'run per point (the first call); host transfers excluded. '
                          'Seconds = cycles / clock_hz.'),
                    boundaries=dict(
                        device='time.device, time.per_launch[].cycles: launch accepted -> core idle',
                        host_transfer='host.transfer_s: every launch\'s upload + download (first call)',
                        host_wall=('host.wall_s: the first call -- Dynamo, Inductor, Triton and the '
                                   'tnca lowering, then the launches; compile time = wall_s - '
                                   'transfer_s - device seconds - dispatch, not separately recorded '
                                   'in protocol 1'),
                        process='host.process_s: the child process, including the CPU reference')),
        open_items=json.loads(Path(args.open_items).read_text()) if args.open_items else [],
        **extra,
    )
    (out / 'baseline.json').write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(counts, indent=1))


def wrap(args):
    base = json.loads(Path(args.baseline).read_text())
    img, sw = base['image'], base['software']
    out = []
    for tier, meta in sorted(base['records'].items()):
        if args.tier and tier not in args.tier:
            continue
        for line in (ROOT / meta['path']).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get('platform') or {}
            r['provenance'] = dict(
                software_tree=sw.get('tree_digest') or sw.get('tree'), artifacts=[k.get('hash') or k.get('name')
                                                         for k in (r.get('kernels') or [])
                                                         if isinstance(k, dict)] or r.get('kernels'),
                profile=p.get('profile'), capabilities=img.get('capabilities'),
                image=f"xclbin:{img['xclbin_sha256'][:16]}", clock_hz=img.get('clock_hz'),
                launch_map=dict(timing='serial-grid', cores=[0],
                                launches=len((r.get('time') or {}).get('per_launch') or [])))
            r['time'] = dict(r.get('time') or {}, kind='device-cycles-serial-grid')
            # named as archbench/platforms/tnca.py names an image (device_id)
            p.update(image=f"xclbin:{img['xclbin_sha256'][:16]}", clock_policy=img.get('clock_policy'),
                     software_tree=sw.get('tree_digest') or sw.get('tree'))
            r['platform'] = p
            out.append(json.dumps(r))
    Path(args.out).write_text('\n'.join(out) + '\n')
    print(f'{len(out)} records -> {args.out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--records', nargs='+', required=True)
    f.add_argument('--image', required=True, help='a JSON file describing the image')
    f.add_argument('--toolchain')
    f.add_argument('--open-items')
    f.add_argument('--defaults', help='a JSON file: the compile/runtime defaults of the frozen software')
    f.add_argument('--b4-sim', help='the device-simulation records B4 compares the tiny board records with')
    f.add_argument('--b1', help='the image\'s B1 conformance JSON')
    f.add_argument('--resources', help='tools/r40/resources.py JSON of the image')
    f.add_argument('--commit', help='the frozen software commit (default HEAD)')
    f.add_argument('--merged', help='where it was merged (a commit), for the record')
    f.add_argument('--tree')
    f.add_argument('--out', default=str(ROOT / 'spec' / 'r40'))
    w = sub.add_parser('wrap')
    w.add_argument('--baseline', default=str(ROOT / 'spec' / 'r40' / 'baseline.json'))
    w.add_argument('--tier', nargs='*')
    w.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    return freeze(args) if args.cmd == 'freeze' else wrap(args)


if __name__ == '__main__':
    sys.exit(main())
