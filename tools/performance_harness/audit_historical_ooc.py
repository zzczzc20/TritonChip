"""Verify source equality before reusing historical diagnostic OOC reports."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def audit(root, reference):
    expected = reference['rtl_git_blobs']
    rows, mismatches = {}, []
    for name, want in expected.items():
        path = root/name
        if not path.is_file():
            mismatches.append(dict(path=name, reason='missing'))
            continue
        raw = path.read_bytes()
        actual = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        rows[name] = actual
        if actual != want:
            mismatches.append(dict(path=name, reason='changed', expected=want, actual=actual))
    extra = [str(path.relative_to(root)) for path in (root/'rtl').rglob('*')
             if path.suffix in ('.v', '.vh') and str(path.relative_to(root)) not in expected]
    archives = {}
    for name in reference['raw_archives']:
        path = root/name
        records = {}
        with tarfile.open(path, 'r:gz') as archive:
            for item in archive.getmembers():
                if not item.isfile() or not item.name.endswith(('.rpt', '.tsv', '.log')):
                    continue
                if item.size > 16 << 20:
                    raise ValueError(f'unbounded diagnostic report: {item.name}')
                raw = archive.extractfile(item).read()
                record = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                if item.name.endswith(('_summary.rpt', '/timing.rpt')):
                    lines = raw.decode(errors='replace').splitlines()
                    for i, line in enumerate(lines):
                        if 'WNS(ns)' not in line or 'TNS(ns)' not in line:
                            continue
                        for candidate in lines[i+1:i+8]:
                            fields = candidate.split()
                            try:
                                if len(fields) < 8:
                                    continue
                                record['timing'] = dict(wns_ns=float(fields[0]), tns_ns=float(fields[1]),
                                    failing_setup_endpoints=int(fields[2]), whs_ns=float(fields[4]))
                                break
                            except ValueError:
                                continue
                        break
                records[item.name] = record
        archives[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), reports=records)
    return dict(schema='tnca-r41-historical-ooc-audit/1',
                status='historical-source-equivalence-pass' if not mismatches and not extra else 'refused-source-mismatch',
                reference_commit=reference['reference_commit'], rtl_tree=reference['reference_rtl_tree'],
                checked_rtl_files=len(rows), mismatches=mismatches, extra_rtl_sources=extra,
                rtl_git_blobs=rows, archives=archives,
                evidence_scope='reused unit OOC/kernel synthesis diagnostics; not fresh synthesis or route timing',
                remaining='fixed frequency candidate set and regional resource preflight still require audit')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = audit(args.root.resolve(), json.loads(args.reference.read_text()))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'checked_rtl_files', 'mismatches', 'extra_rtl_sources')}, indent=2))
    raise SystemExit(0 if report['status'] == 'historical-source-equivalence-pass' else 1)
