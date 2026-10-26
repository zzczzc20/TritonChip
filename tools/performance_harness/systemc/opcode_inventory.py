#!/usr/bin/env python3
"""Static opcode inventory of compiled TNCA kernels, with model coverage.

    opcode_inventory.py --runs A=~/r40v3_fix2/A_fix2/runs --runs B=~/r40v3_fix2/B_fix2/runs \
        [--classifier BUILD/tnca-chip] --out inventory.json [--markdown inventory.md]

Reads the assembled images (dump/<kernel>/<kernel>.hex) the compile-only
preflights wrote for every (point, configuration). Opcode identities come
from the canonical ISA tables (tools/asm/tnca_isa.py); the variant is the
disassembler's full mnemonic (size, format, K/dtype, order/scope suffixes).
With --classifier the model binary decides, word by word, whether its
executor implements the semantics and which timing resource it charges:
the table is the model's own decode, not a hand-maintained list.
Static counts only: dynamic frequency needs execution.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'tools/asm'))
from tnca_isa import ISA, MAJ, decode_fields  # noqa: E402
from tnca_dis import disassemble  # noqa: E402

BY_OPCODE = {(MAJ[m], f): name for name, (m, f, _k, _s) in ISA.items()}


def point_of(run_dir):
    parts = run_dir.name.split('__')
    return '/'.join(parts[:3]), '__'.join(parts[3:])


def words(hexfile):
    for line in hexfile.read_text().split():
        yield int(line, 16)


# Acceptance logic of the native executor before M1 (src/chip.cpp at commit
# 0befd8e), transcribed from its inline decode: which words it executed
# instead of refusing. `inexact` marks groups it executed whose numerics were
# not the RTL's (host libm SFU, IEEE min/max folds, ISA-text float->int).
OLD_SPECIALS = {0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 32}


def baseline_verdict(w):
    f = decode_fields(w)
    major, func, mod = f['major'], f['func'], f['mod']
    if (major, func) not in BY_OPCODE:
        return 'refused', 'undefined'
    if major == 0x00:
        if func == 0:
            return 'implemented', ''
        if func == 1:
            return ('implemented', '') if mod in OLD_SPECIALS else ('refused', 'uncovered special register')
        return 'refused', 's.wrspecial'
    if major == 0x02:
        return ('refused', 'user trap') if func == 0x18 else ('implemented', '')
    if major in (0x01, 0x10, 0x13, 0x14, 0x30, 0x31, 0x50):
        return 'implemented', ''
    if major == 0x11:
        return ('refused', 'FTZ modifier') if mod & ~15 else ('implemented', '')
    if major == 0x12:
        return ('implemented-inexact', 'float->int corners differ from RTL') if func in (2, 3) else ('implemented', '')
    if major == 0x15:
        return ('implemented-inexact', 'FP max/min fold differs from RTL') if func in (9, 10) else ('implemented', '')
    if major == 0x16:
        return 'implemented-inexact', 'host libm, not the RTL SFU'
    if major == 0x20:
        return ('implemented', '') if (mod & 7) <= 3 and not (mod >> 6) and ((mod >> 4) & 3) != 3 else ('refused', 'modifier')
    return 'refused', 'uncovered instruction group'


def classify(binary, unique):
    """Ask the model's decoder about each distinct word (no execution)."""
    text = ''.join(f'{w:016x}\n' for w in unique)
    out = subprocess.run([str(binary), '--classify'], input=text, capture_output=True,
                         text=True, check=True, timeout=600).stdout.splitlines()
    result = {}
    for line in out:
        word, semantics, timing, *why = line.split('\t')
        result[int(word, 16)] = dict(semantics=semantics, timing=timing, reason=why[0] if why else '')
    missing = set(unique) - set(result)
    if missing:
        raise RuntimeError(f'classifier did not answer {len(missing)} words')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', action='append', required=True, help='LABEL=DIRECTORY of run directories')
    ap.add_argument('--classifier', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--markdown', type=Path)
    a = ap.parse_args()
    rows = defaultdict(lambda: dict(static=0, kernels=set(), points=set(), runs=set(), words=set()))
    scanned = dict(runs=0, kernels=0, words=0)
    run_words = defaultdict(set)
    kernel_hashes = {}
    for spec in a.runs:
        label, directory = spec.split('=', 1)
        for run in sorted(Path(directory).expanduser().iterdir()):
            hexes = sorted(run.glob('dump/*/*.hex'))
            if not hexes:
                continue
            scanned['runs'] += 1
            point, config = point_of(run)
            for hexfile in hexes:
                image = list(words(hexfile))
                scanned['kernels'] += 1
                scanned['words'] += len(image)
                kernel = hexfile.stem
                kernel_hashes.setdefault(kernel, set()).add(hash(tuple(image)))
                run_words[(label, point, config)].update(image)
                for w in image:
                    f = decode_fields(w)
                    base = BY_OPCODE.get((f['major'], f['func']), f'undefined.{f["major"]:#04x}.{f["func"]:#04x}')
                    tokens = disassemble(w).split()
                    variant = next(t for t in tokens if not t.startswith('@'))
                    if f['pred'] or f['pn']:
                        variant += ' (predicated)'
                    for key in (('opcode', base), ('variant', variant)):
                        r = rows[key]
                        r['static'] += 1
                        r['kernels'].add((label, kernel, hash(tuple(image))))
                        r['points'].add(point)
                        r['runs'].add(f'{label}:{run.name}')
                        r['words'].add(w)
                        r['major_func'] = f"{f['major']:#04x}.{f['func']:#04x}"
    verdict = {}
    if a.classifier:
        unique = sorted({w for r in rows.values() for w in r['words']})
        verdict = classify(a.classifier, unique)
    table = []
    for (kind, name), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[1]['static'])):
        entry = dict(kind=kind, name=name, major_func=r['major_func'], static=r['static'],
                     kernels=len(r['kernels']), points=len(r['points']), runs=len(r['runs']),
                     distinct_words=len(r['words']))
        before = sorted({baseline_verdict(w)[0] for w in r['words']})
        entry['before_m1'] = 'implemented' if before == ['implemented'] else (
            'refused' if before == ['refused'] else ('implemented-inexact' if 'refused' not in before else 'partial'))
        if verdict:
            sem = sorted({verdict[w]['semantics'] for w in r['words']})
            tim = sorted({verdict[w]['timing'] for w in r['words']})
            reasons = sorted({verdict[w]['reason'] for w in r['words'] if verdict[w]['reason']})
            refused = sum(1 for w in r['words'] if verdict[w]['semantics'] != 'implemented')
            entry.update(semantics='implemented' if sem == ['implemented'] else
                         ('refused' if 'implemented' not in sem else 'partial'),
                         refused_words=refused, timing=tim, refusal_reasons=reasons[:4])
        table.append(entry)
    def executable(word_set, ok):
        return all(ok(w) for w in word_set)
    runs_summary = defaultdict(lambda: dict(runs=0, before=0, after=0, before_bit_exact=0))
    for (label, point, config), word_set in run_words.items():
        row = runs_summary[label]
        row['runs'] += 1
        row['before'] += executable(word_set, lambda w: baseline_verdict(w)[0] != 'refused')
        row['before_bit_exact'] += executable(word_set, lambda w: baseline_verdict(w)[0] == 'implemented')
        if verdict:
            row['after'] += executable(word_set, lambda w: verdict[w]['semantics'] == 'implemented')
    scanned['point_runs'] = dict(runs_summary)
    report = dict(schema='tnca-opcode-inventory/1', sources=a.runs, scanned=scanned,
                  classifier=str(a.classifier) if a.classifier else None,
                  note='static instruction counts in assembled images; a kernel is a distinct (label, name, image)',
                  rows=table)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    if a.markdown:
        lines = [f"Scanned {scanned['runs']} (point, config) runs, {scanned['kernels']} kernel images, "
                 f"{scanned['words']} instruction words.", '',
                 '| source | (point, config) runs | every word executable before M1 | ... and RTL-exact before M1 | every word implemented now |',
                 '|---|---:|---:|---:|---:|']
        for label, row in sorted(runs_summary.items()):
            lines.append(f"| {label} | {row['runs']} | {row['before']} | {row['before_bit_exact']} | {row['after']} |")
        lines += ['',
                 '| opcode | major.func | static | kernels | points | before M1 | semantics | timing |',
                 '|---|---|---:|---:|---:|---|---|---|']
        for e in table:
            if e['kind'] != 'opcode':
                continue
            lines.append(f"| `{e['name']}` | {e['major_func']} | {e['static']} | {e['kernels']} | {e['points']} | "
                         f"{e['before_m1']} | {e.get('semantics', '?')}{' (' + str(e['refused_words']) + ' words refused)' if e.get('refused_words') else ''} | "
                         f"{', '.join(e.get('timing', [])) or '?'} |")
        lines += ['', '| variant | static | kernels | points | before M1 | semantics | reason |', '|---|---:|---:|---:|---|---|---|']
        for e in table:
            if e['kind'] != 'variant':
                continue
            lines.append(f"| `{e['name']}` | {e['static']} | {e['kernels']} | {e['points']} | {e['before_m1']} | "
                         f"{e.get('semantics', '?')} | {'; '.join(e.get('refusal_reasons', []))} |")
        a.markdown.write_text('\n'.join(lines)+'\n')
    print(json.dumps(scanned))


if __name__ == '__main__':
    main()
