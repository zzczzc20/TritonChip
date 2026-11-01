#!/usr/bin/env python3
"""Queue a gated opt-in equal-output board session on the already-loaded image.

The caller prepares a new frozen source copy and copies the completed TLM
reports into gate-dir. This script never programs an image. Its stage runner
uses the shared lock and checks card idleness, image UUID/hash and clock.
"""
import argparse
import json
import sys
from pathlib import Path

EXPECTED_FRONTEND = 'b275ea15924d0a2f7765906e9e7cdc15afc43666c94ead673a20e48fc9cdec03'


def load(path):
    return json.loads(path.read_text())


def validate_gates(gate):
    audit = load(gate / 'audit.json')
    if audit.get('summary') != {'points': 113, 'unchanged': 113, 'changed': 0, 'not_compared': 0}:
        raise ValueError('default-code audit is not 113/113 unchanged')
    precise = load(gate / 'suite_ieee.json')
    if (precise.get('summary') != {'pass': 16, 'fail': 0, 'error': 0}
            or precise.get('knobs', {}).get('TNCA_IEEE_DIV') != '1'
            or precise.get('knobs', {}).get('TNCA_X8_EQUAL_OUTPUT') != '1'):
        raise ValueError('exact IEEE suite did not pass 16/16')
    default = load(gate / 'suite_default.json')
    if (default.get('summary') != {'pass': 12, 'fail': 4, 'error': 0}
            or default.get('knobs', {}).get('TNCA_IEEE_DIV') != '0'
            or default.get('knobs', {}).get('TNCA_X8_EQUAL_OUTPUT') != '1'):
        raise ValueError('unexpected default suite outcome')
    cases = default.get('cases', [])
    if len(cases) != 16 or any(row.get('status') != ('fail' if 'fp8_act_quant' in row.get('case', '')
                                                  and 'matmul' not in row.get('case', '')
                                                  and 'GEMM' not in row.get('case', '') else 'pass')
                               for row in cases):
        raise ValueError('default-suite failures are not the recorded standalone quantizers')
    model = load(gate / 'model.json')
    routes = model.get('routes', [])
    if (len(routes) != 2 or [r.get('route') for r in routes] !=
            [f'r41_x8_equal_output_q35_{i}_20261101' for i in (0, 1)]
            or not model.get('same_launch_count') or model.get('metadata_mismatches')
            or not model.get('logits') or any(x.get('bit_mismatches') != 0 for x in model['logits'].values())):
        raise ValueError('model comparison is not bitwise equal with matching launches')
    for route in routes:
        if route.get('status') != 'pass' or route.get('cpu_numerical_fallback') is not False:
            raise ValueError('model comparison has failure or numerical host fallback')
        decode = [x for x in route.get('steps', []) if x.get('kind') == 'decode']
        if len(decode) != 1 or decode[0].get('host_read_calls') != 0:
            raise ValueError('model decode did not have zero host reads')
    if routes[0].get('generated_tokens') != routes[1].get('generated_tokens'):
        raise ValueError('Qwen3.5 tokens differ between model routes')
    old = next(x for x in routes[0]['steps'] if x['kind'] == 'decode')
    new = next(x for x in routes[1]['steps'] if x['kind'] == 'decode')
    if new['device_cycles'] >= old['device_cycles']:
        raise ValueError('TLM decode did not improve')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='new isolated board source copy')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gate-dir', type=Path, required=True)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if root != (Path.home() / 'r41run3/runtime_equal_output_20261101').resolve():
        raise ValueError('root must be the new isolated equal-output snapshot')
    validate_gates(a.gate_dir)
    sys.path.insert(0, str(root / 'tools/r41/run2_board'))
    import q35_timeout_recovery as recovery
    if recovery.digest(root / 'compiler/python/tnca_cc/frontend.py') != EXPECTED_FRONTEND:
        raise ValueError('candidate source checksum does not match the TLM candidate')
    original = load(root / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json')
    row = original['experiments'][0]
    source = root / row['manifest']
    if recovery.digest(source) != row['sha256']:
        raise ValueError('original session registration changed')
    spec = load(source)
    name = 'r41-q35-equal-output-session-20261101'
    spec['id'] = name
    spec['registration']['scope'] = ('Opt-in proven equal-row accumulator output; '
                                      'original session quality and no-host rules unchanged.')
    spec['steps'][0]['env']['TNCA_X8_EQUAL_OUTPUT'] = '1'
    manifest = root / 'spec/r41/run3/board' / (name + '.json')
    out.mkdir(parents=True, exist_ok=False)
    recovery.OUT = out
    recovery.ROOT = root
    with manifest.open('x') as f:
        json.dump(spec, f, indent=2); f.write('\n')
    row['id'], row['manifest'], row['sha256'] = name, str(manifest.relative_to(root)), recovery.digest(manifest)
    row['checks'].extend([
        dict(path='q35-session/report.json', field=['cpu_numerical_fallback'], equals=False),
        dict(path='q35-session/report.json', field=['compiled_wrapper_roles_covered'], equals=True),
        dict(path='q35-session/report.json', field=['steady_decode_host_reads_detected'], equals=False),
    ])
    original['id'], original['experiments'] = name, [row]
    registration = out / 'registration.json'
    registration.write_text(json.dumps(original, indent=2) + '\n')
    recovery.record('registered', registration_sha256=recovery.digest(registration),
                    gate_sha256={p.name: recovery.digest(p) for p in sorted(a.gate_dir.glob('*.json'))})
    if recovery.digest(root / 'compiler/python/tnca_cc/frontend.py') != EXPECTED_FRONTEND:
        raise ValueError('candidate source changed after registration')
    recovery.stage(root, out / 'session', registration, out / 'stage.log')
    recovery.record('finished')


if __name__ == '__main__':
    main()
