#!/usr/bin/env python3
"""S1-00: the evidence registry claims no more than its evidence.

doc/r38_stage1_tasks_and_acceptance.md S1-00. Checks spec/r38/evidence_registry.json,
then checks the checker: each mutation below must be caught, or this fails.
"""
import copy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'spec' / 'r38' / 'evidence_registry.json'
MAKEFILE = (ROOT / 'Makefile').read_text()


def make_targets():
    return set(re.findall(r'^([A-Za-z0-9_.-]+):', MAKEFILE, re.M))


def command_exists(cmd):
    cmd = cmd.strip()
    if cmd.startswith('make '):
        return cmd.split()[1] in make_targets()
    return (ROOT / cmd.split()[0]).exists()


def problems(reg, root=ROOT):
    levels = reg['levels']
    bad, seen = [], set()
    rank = {l: i for i, l in enumerate(levels)}
    for e in reg['entries']:
        tag = e.get('id', '<no id>')
        if tag in seen:
            bad.append(f'{tag}: duplicate id')
        seen.add(tag)
        lvl = e.get('level')
        if lvl not in rank:
            bad.append(f'{tag}: level {lvl!r} is not one of {levels}')
            continue
        if lvl == 'board-measured':
            bad.append(f'{tag}: board-measured is impossible in Stage 1')
        if not e.get('artifacts'):
            bad.append(f'{tag}: no artifacts')
        for a in e.get('artifacts', []):
            if not (root / a).exists():
                bad.append(f'{tag}: artifact {a} does not exist')
        for g in e.get('generated', []):
            if not g.startswith('sim/build/'):
                bad.append(f'{tag}: generated evidence {g} is not under sim/build/')
        if rank[lvl] >= rank['functionally-validated']:
            if not e.get('command'):
                bad.append(f'{tag}: {lvl} without a command that re-establishes it')
            elif not command_exists(e['command']):
                bad.append(f'{tag}: command {e["command"]!r} does not exist')
        if lvl == 'synthesized' and e.get('structural_proxy') is not True:
            bad.append(f'{tag}: synthesized without structural_proxy: true')
    return bad


def mutations(reg):
    def edit(fn):
        r = copy.deepcopy(reg)
        fn(r)
        return r
    first = next(i for i, e in enumerate(reg['entries']) if e['level'] == 'functionally-validated')
    synth = next(i for i, e in enumerate(reg['entries']) if e['level'] == 'synthesized')
    return {
        'promoted to board-measured': edit(lambda r: r['entries'][0].update(level='board-measured')),
        'linked artifact removed': edit(lambda r: r['entries'][0]['artifacts'].append('doc/does_not_exist.md')),
        'validated without a command': edit(lambda r: r['entries'][first].pop('command')),
        'command that does not exist': edit(lambda r: r['entries'][first].update(command='make no-such-target')),
        'synthesis not marked proxy': edit(lambda r: r['entries'][synth].pop('structural_proxy')),
        'unknown level': edit(lambda r: r['entries'][0].update(level='supported')),
    }


def main():
    reg = json.loads(REGISTRY.read_text())
    bad = problems(reg)
    for line in bad:
        print('FAIL', line)
    missed = [name for name, m in mutations(reg).items() if not problems(m)]
    for name in missed:
        print('FAIL mutation not caught:', name)
    ok = not bad and not missed
    print(f'evidence registry: {len(reg["entries"])} entries, {"PASS" if ok else "FAIL"}'
          f' ({len(mutations(reg)) - len(missed)}/{len(mutations(reg))} mutations caught)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
