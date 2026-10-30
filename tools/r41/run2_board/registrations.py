#!/usr/bin/env python3
"""Register run-2 stage groups and their explicit workload result checks."""
import argparse
import json
from pathlib import Path

import auto
import manifests
import stage

ROOT = manifests.ROOT
OUT = Path('spec/r41/run2/stages')
SUITE = Path('archbench/suites/r41_membound.json')


def equal(path, field, value):
    return dict(path=path, field=[field], equals=value)


def experiment(name, checks):
    name = 'r2-' + name
    path = manifests.OUT / (name + '.json')
    return dict(id=name, manifest=str(path), sha256=stage.sha(ROOT/path), checks=checks)


def batches(mode):
    result = []
    for p in sorted((ROOT/manifests.OUT).glob('r2-membound-*-' + mode + '.json')):
        spec = json.loads(p.read_text())
        output = spec['steps'][0]['outputs'][0].replace('{out}/', '')
        result.append(experiment(p.stem.removeprefix('r2-'), [equal(output, 'status', 'complete')]))
    return result


def group(name):
    table = {
        'identity': [experiment('s2-identity', [equal('source.json', 'status', 'pass'),
            equal('clock.json', 'status', 'pass'), equal('identity/batch.json', 'status', 'complete'),
            equal('identity/batch.json', 'configurations', 3)])],
        'hbm8': [experiment(n, [equal('summary.json', 'qualified', True)])
                 for n in ('s2-hbm8', 's2-hbm8-big')],
        'original113': [experiment('s2-original-suite', [equal('original/batch.json', 'status', 'complete'),
                                                         equal('original/batch.json', 'configurations', 113)])],
        'membound-default': batches('default'), 'membound-tiled2': batches('tiled2'),
        'fp8-cases': [experiment('s2-fp8-cases', [equal('summary.json', 'result', 'pass')])],
        'fp8-blockwise': [experiment('s2-fp8-blockwise', [equal('summary.json', 'result', 'pass')])],
        'fp8-hub': [experiment('s2-fp8-hub', [equal('summary.json', 'result', 'pass')])],
        'qwen-layers': [experiment('s2-qwen-fp8', [equal('linears/result.json', 'total', 14),
                         equal('linears/result.json', 'passed', 14), equal('decode/result.json', 'pass', True)])],
        'endurance': [experiment('s2-endurance', [equal('mix/endurance.json', 'result', 'pass'),
                        dict(path='mix/endurance.json', field=['hours'], minimum=2)])],
    }
    for model, count in (('vl', 6), ('q35', 5)):
        quality, session = model + '-quality/quality.json', model + '-session/report.json'
        table['chat-' + model] = [experiment('s2-qwen-chat-v3-' + model, [
            equal(quality, 'status', 'pass'), equal(quality, 'model_scope', 'full-checkpoint'),
            equal(quality, 'reduced_proof', False), dict(path=quality, field=['prompts'], length_equals=count),
            equal(session, 'status', 'pass'),
            dict(path=session, field=['persistent_session_check', 'passed'], equals=True)])]
    result = dict(schema='tnca-run2-stage-registration/1', id=name, experiments=table[name])
    if name.startswith('membound-'):
        result['coverage'] = dict(suite=str(SUITE), suite_sha256=stage.sha(ROOT/SUITE), count=513,
                                  tiled_pointwise='2' if name.endswith('tiled2') else '0')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    for name in auto.REQUIRED:
        path = ROOT/OUT/(name + '.json')
        result = group(name)
        if a.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=1) + '\n')
        if a.check and (not path.is_file() or json.loads(path.read_text()) != result):
            raise SystemExit('stage registration differs: ' + str(path))
    print(f'{len(auto.REQUIRED)} stage groups ' + ('written' if a.write else 'checked' if a.check else 'derived'))


if __name__ == '__main__':
    main()
