#!/usr/bin/env python3
"""Queue the gated opt-in FP8 hoist after the existing speed sessions."""
import json
import sys
from pathlib import Path

WORK = Path.home() / 'r41run3'
ROOT = WORK / 'runtime_hoist_precise_20261031'
OUT = WORK / 'chat_hoist_precise_20261031'
PREVIOUS = WORK / 'chat_speed_after_quality_failure_retry_20261031'
EXPECTED = {
    'compiler/python/tnca_cc/frontend.py': '9da757bdd887912596f3682f5e06e6ff3ef49a5fa5acdf78a2ea04d0ae3139c1',
    'compiler/python/tnca_cc/register_blocks.py': '7b2c5730fbbfd4f23741f587826fe30f84df842114b77fee404600bcf235af64',
    'python/tnca/fp8_hoist.py': 'f45a7a43e918a77ca5db4d3223d26cf175de356c3d57ffbb19cddcf4f5887bc6',
    'tools/models/qwen_fp8_lib.py': 'd4c814f65efe1869ebaceadd5f480e8e33278272479f19a7fd242209bd144754',
}
sys.path.insert(0, str(WORK / 'runtime_activation_equal_20261031/tools/r41/run2_board'))
import q35_timeout_recovery as recovery


def vendor_hashes():
    base = ROOT / 'vendor/finegrained-fp8-v1'
    return {str(p.relative_to(base)): recovery.digest(p) for p in sorted(base.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def main():
    OUT.mkdir(exist_ok=False)
    recovery.OUT = OUT
    for name, expected in EXPECTED.items():
        if recovery.digest(ROOT / name) != expected:
            raise ValueError('prepared candidate changed: ' + name)
    vendors = vendor_hashes()
    if not vendors:
        raise ValueError('pinned FP8 dependency is missing')
    (OUT / 'vendor-hashes.json').write_text(json.dumps(vendors, indent=2) + '\n')
    old = json.loads((ROOT / 'spec/r41/run3/stages/chat-q35-nohost-v5-20261031.json').read_text())
    row = old['experiments'][0]
    spec = json.loads((ROOT / row['manifest']).read_text())
    name = 'r41-q35-hoist-session-20261031'
    spec['id'] = name
    spec['registration']['scope'] = ('Gated shape-selected FP8 hoist; non-default '
        'TNCA_FP8_HOIST_V1_PATH with pinned vendor revision 48f42b23; original chat rules unchanged.')
    step = spec['steps'][0]
    step['env']['TNCA_FP8_HOIST_V1_PATH'] = '{root}/vendor/finegrained-fp8-v1'
    step.setdefault('inputs', []).append('{root}/vendor/finegrained-fp8-v1')
    manifest = ROOT / 'spec/r41/run3/board' / (name + '.json')
    with manifest.open('x') as f:
        json.dump(spec, f, indent=2); f.write('\n')
    row['id'] = name
    row['manifest'] = str(manifest.relative_to(ROOT))
    row['sha256'] = recovery.digest(manifest)
    row['checks'].extend([
        dict(path='q35-session/report.json', field=['fp8_route_in_use', 'kind'], equals='experimental-tnca-hoist'),
        dict(path='q35-session/report.json', field=['cpu_numerical_fallback'], equals=False),
        dict(path='q35-session/report.json', field=['compiled_wrapper_roles_covered'], equals=True),
        dict(path='q35-session/report.json', field=['steady_decode_host_reads_detected'], equals=False),
        dict(path='q35-session/report.json', field=['steps', 0, 'transfers_inside_step', 'by_kernel',
             '_precise_block_quant', 'launches'], minimum=1),
    ])
    old['id'] = name
    old['experiments'] = [row]
    reg = OUT / 'registration.json'
    reg.write_text(json.dumps(old, indent=2) + '\n')
    recovery.record('waiting', preceding_watcher_pid=1203557, registration_sha256=recovery.digest(reg),
                    vendor_revision='48f42b23bb4e0122d4775c6d16753dd2efc7f580')
    recovery.wait_pid(1203557)
    for item in ('event_index', 'equalrow_event', 'pipeline_equalrow', 'activation_equal'):
        p = PREVIOUS / ('speed-' + item) / 'stage.json'
        if not p.is_file() or json.loads(p.read_text()).get('status') != 'pass':
            raise RuntimeError('preceding acceleration stage did not pass: ' + item)
    if vendor_hashes() != vendors:
        raise RuntimeError('pinned FP8 dependency changed')
    for name, expected in EXPECTED.items():
        if recovery.digest(ROOT / name) != expected:
            raise ValueError('candidate changed while waiting: ' + name)
    recovery.stage(ROOT, OUT / 'session', reg, OUT / 'stage.log')
    recovery.record('finished')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            recovery.record('stopped', error=f'{type(exc).__name__}: {exc}')
        raise
