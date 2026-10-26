#!/usr/bin/env python3
"""Audit the archived R40 softmax compiler evidence; never optimize or run it.

This intentionally recognizes only these frozen fixtures. Static instruction
counts and loop trip counts are not measured latency or an alias-safety proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def audit(root):
    product = 'files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75'
    reports = {}
    report_manifest = json.loads((root / 'preflight_manifest.json').read_text())
    for mode in ('cpu', 'native-stub'):
        data = (root / (mode + '_preflight_report.json')).read_bytes()
        if hashlib.sha256(data).hexdigest() != report_manifest[mode]['sha256']:
            raise ValueError('preflight report hash mismatch: ' + mode)
        report = json.loads(data)
        if (report['status'] != 'complete' or report['device'] != mode
                or report['software_tree'] != product
                or report['software_tree_after'] != product):
            raise ValueError('unexpected preflight identity: ' + mode)
        reports[mode] = report
    manifest = json.loads((root / 'source_manifest.json').read_text())
    with tarfile.open(root / 'historical_compile_artifacts.tar.gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError('archive inventory differs from manifest')
        texts = {}
        for member in members:
            if not member.isfile():
                raise ValueError('unexpected non-file archive member')
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest[member.name]['sha256']:
                raise ValueError('artifact hash mismatch: ' + member.name)
            texts[member.name] = data.decode()
    rows = []
    for tier, cols in (('tiny', 37), ('small', 512), ('medium', 512)):
        pair = {}
        for mode in ('cpu', 'native-stub'):
            prefix = f'{mode}/op.softmax__{tier}__'
            irs = [k for k in texts if k.startswith(prefix) and k.endswith('.opt.ttir')]
            if len(irs) != 1:
                raise ValueError('expected exactly one TTIR for ' + prefix)
            key = irs[0]
            source_dir = str(Path(manifest[key]['source']).parents[2])
            runs = [r for r in reports[mode]['runs'].values()
                    if r.get('run_dir') == source_dir]
            if len(runs) != 1 or runs[0]['status'] != 'pass':
                raise ValueError('missing passing preflight run: ' + key)
            text = texts[key]
            loops = list(re.finditer(r'^    (?:%\S+ = )?scf.for .*$', text, re.M))
            if len(loops) != 3:
                raise ValueError('expected the frozen three-loop softmax')
            final_loop = text[loops[-1].start():].split('\n    } loc(', 1)[0]
            expected = f'to %c{cols}_i32 step %c16_i32'
            if expected not in loops[-1].group():
                raise ValueError('unrecognized final-loop bounds')
            barriers = final_loop.count('ttg.barrier all')
            if barriers != (1 if mode == 'native-stub' else 0):
                raise ValueError('unexpected final-loop barrier count')
            if 'tt.store ' not in final_loop:
                raise ValueError('expected a store in the barrier loop')
            if not any(op in final_loop for op in ('tt.load ', 'tt.descriptor_load ')):
                raise ValueError('expected a reload in the barrier loop')
            base = key[:-len('.opt.ttir')]
            assembly = texts[base + '.s']
            tnir = texts[base + '.tnir']
            pair[mode] = {
                'ttir': key,
                'compile_knobs': runs[0]['knobs'],
                'tnir_fences': tnir.count('tnca.fence()'),
                'assembly_fences': re.findall(r'^\s+(fence\.\S+)', assembly, re.M),
                'final_loop_barriers_per_logical_row': barriers * ((cols + 15) // 16),
                'final_loop_has_load_and_store': True,
            }
        if pair['native-stub']['tnir_fences'] != pair['cpu']['tnir_fences'] + 1:
            raise ValueError('expected one additional native TNCA fence')
        rows.append({'tier': tier, 'columns': cols, 'paths': pair})
    return {
        'schema': 'r40-followup-softmax-fence-audit/1',
        'evidence_level': 'historical-compile-only',
        'reported_product_tree': product,
        'artifact_hashes_verified': len(manifest),
        'points': rows,
        'decision': 'Do not apply straight-line redundant-fence elimination.',
        'reason': 'The added fence is inside a loop with loads and stores; '
                  'hoisting needs a separate inter-iteration dependence proof.',
        'not_measured': ['numerics', 'performance', 'board state', 'gate improvement'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, default=Path(__file__).resolve().parents[2]
                        / 'evidence/r40/followup/fence_audit')
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence_root), indent=2))
