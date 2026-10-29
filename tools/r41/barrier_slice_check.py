#!/usr/bin/env python3
"""Check every compiled launch's barrier immediates against the slice the runtime gives it.

    barrier_slice_check.py --bar-n 8 [--manifests M.json ...] [--audit-report R.json --audit-runs DIR]

The runtime (python/tnca/runtime.py `_run_physical_grid`, and the per-program path) starts
`width = len(program_to_role)` programs per core with `roles_per_program` contexts each and,
when width > 1, sets the RSLICE flag and log2(width): every program then owns BAR_N // width
barriers, and tnca_core's guard (d_bar_index_oob) faults an mb.init/arm/arrive/inval immediate
>= BAR_N >> log2(width). A launch with width 1 is not sliced and owns all BAR_N. This tool takes
each launched artifact's mapping and the immediates in its code and reports the worst margin.

Sources:
  --manifests     JSON written by tools/r41/fp8_hub_compile_only.py (per artifact: execution,
                  resources, barrier_immediates read from the sealed code segment)
  --audit-report  tools/performance_harness/default_change_audit.py report; the immediates are
                  read from each kernel's dumped assembly under --audit-runs
Exit status 1 if any launch's largest immediate is >= its allowance.
"""
import argparse
import json
import re
import sys
from pathlib import Path

OPS = re.compile(r'^\s*mb\.(init|arm|arrive|inval)\b([^/\n]*)')


def asm_immediates(path):
    out = set()
    for line in Path(path).read_text().splitlines():
        m = OPS.match(line)
        if m:
            out.update(int(g) for g in re.findall(r'\bb(\d+)\b', m.group(2)))
    return sorted(out)


def allowance(bar_n, mapping):
    width = len(mapping)
    return (bar_n // width if width > 1 else bar_n), width, len(mapping['0'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bar-n', type=int, default=8)
    ap.add_argument('--manifests', nargs='*', default=[], type=Path)
    ap.add_argument('--audit-report', type=Path)
    ap.add_argument('--audit-runs', type=Path)
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    rows = []
    for mf in a.manifests:
        progress = mf.parent / 'compile_only.json'
        if progress.exists() and json.loads(progress.read_text()).get('status') == 'incomplete':
            raise SystemExit(f'{mf}: compile-only census is incomplete; refusing a slice verdict')
        for path, e in json.loads(mf.read_text()).items():
            lim, width, rpp = allowance(a.bar_n, e['execution'].get('program_to_role', {'0': ['role0']}))
            rows.append(dict(source=str(mf.parent.name), kernel=e.get('kernel'), width=width, rpp=rpp,
                             allowed=lim, immediates=e['barrier_immediates'],
                             manifest_barriers=e['resources'].get('barriers')))
    if a.audit_report:
        report = json.loads(a.audit_report.read_text())
        for pk, p in report['points'].items():
            for kn, k in p['kernels'].items():
                ex = k['launch_structure']['execution']
                lim, width, rpp = allowance(a.bar_n, ex.get('program_to_role', {'0': ['role0']}))
                imms = asm_immediates(a.audit_runs / pk.replace('/', '__') / k['asm']) if k.get('asm') else None
                if imms is None:
                    imms = [-1]
                rows.append(dict(source=pk, kernel=kn, width=width, rpp=rpp, allowed=lim, immediates=imms,
                                 manifest_barriers=k['launch_structure']['resources'].get('barriers')))
    bad = [r for r in rows if r['immediates'] and max(r['immediates']) >= r['allowed']]
    bad += [r for r in rows if isinstance(r['manifest_barriers'], int) and r['manifest_barriers'] > r['allowed']
            and r not in bad]
    by = {}
    for r in rows:
        key = (r['width'], r['rpp'], r['allowed'])
        d = by.setdefault(key, dict(launches=0, max_imm=-1, max_count=0))
        d['launches'] += 1
        d['max_imm'] = max(d['max_imm'], max(r['immediates'], default=-1))
        d['max_count'] = max(d['max_count'], len(r['immediates']))
    print(f'launches checked: {len(rows)}; BAR_N {a.bar_n}')
    print('width roles/prog allowed-per-program | launches max-immediate max-distinct')
    for (w, rpp, lim), d in sorted(by.items()):
        print(f'{w:5d} {rpp:9d} {lim:8d} | {d["launches"]:8d} {d["max_imm"]:13d} {d["max_count"]:12d}')
    for r in bad:
        print('VIOLATION', r['source'], r['kernel'], f'width {r["width"]} allowed {r["allowed"]} immediates {r["immediates"]}')
    print('VERDICT:', 'no launch exceeds its slice' if not bad else f'{len(bad)} launch(es) exceed their slice')
    if a.json:
        a.json.write_text(json.dumps(dict(bar_n=a.bar_n, rows=rows, violations=bad), indent=1))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
