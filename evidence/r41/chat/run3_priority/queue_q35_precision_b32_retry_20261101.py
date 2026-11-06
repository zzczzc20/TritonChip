#!/usr/bin/env python3
"""Queue a separate Q35 whole-prefill precision diagnostic on the existing image.

The original quality score and failures stay unchanged. The source copy and
registration are created before acquiring the shared card lock. No programming.
"""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path.home() / 'r41run3'
OLD = WORK / 'runtime_q35_whole_prefill_20261031'
ROOT = WORK / 'runtime_q35_precision_b32_whole_v2_20261101'
OUT = WORK / 'chat_q35_precision_b32_whole_v2_20261101'
ENTRY = WORK / 'q35_precision_cast_entry_v2_20261101.py'
ORIGINAL = 'r41-q35-whole-prefill-multiturn-20261031'
NAME = 'r41-q35-precision-b32-whole-multiturn-v2-20261101'
UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
EXPECTED = {
    'compiler/python/tnca_cc/frontend.py': 'f1822fa71b712ea5b29423e2eace5a2c10105753ca89ca7404af63584f94a99c',
    'compiler/python/tnca_cc/register_blocks.py': '2620b2497cf47bada001021e3c5265811a6080b83e8f07ac1693150dac92fe18',
    'python/tnca/triton_backend/compiler.py': '2a29c8eb516aa46c5e1aec84019db425a92798c446c93ecb013b7d7d2d79aed6',
    'tools/models/chat.py': '7121408bfa7189aad2a80dcb2bccce0a9ddce656fdf6b7eb9f7de46feec601ed',
    'tools/models/chat_quality.py': 'cfaf18a5503c706f08d41323ff9b14ce90871ad3793ce3013849b0d924523eff',
    'spec/r41/run3/board/' + ORIGINAL + '.json': 'e4fcbd0bef22676ac797c9f11ada82649bd8fb354b7b8742b62b777b503b003f',
}
ENTRY_SHA = '214e82916d2ccda5894a6767b9c1c17332b33c3fa7a7177e6376a25ed986656c'


def main():
    sys.path.insert(0, str(WORK / 'runtime_activation_equal_20261031/tools/r41/run2_board'))
    import auto
    import q35_timeout_recovery as recovery
    if ROOT.exists() or OUT.exists():
        raise FileExistsError('precision diagnostic source or output already exists')
    for name, sha in EXPECTED.items():
        if recovery.digest(OLD / name) != sha:
            raise ValueError('registered source changed: ' + name)
    if recovery.digest(ENTRY) != ENTRY_SHA:
        raise ValueError('precision entrypoint differs from TLM')
    source_reg = WORK / 'chat_q35_whole_prefill_20261031/registration.json'
    registration = copy.deepcopy(json.loads(source_reg.read_text()))
    if (registration.get('id') != ORIGINAL or len(registration.get('experiments', [])) != 1
            or registration['experiments'][0].get('sha256') != EXPECTED['spec/r41/run3/board/' + ORIGINAL + '.json']):
        raise ValueError('unexpected original registration')
    OUT.mkdir()
    recovery.OUT = OUT
    try:
        subprocess.run(['cp', '-a', '--reflink=auto', str(OLD), str(ROOT)], check=True)
        for name, sha in EXPECTED.items():
            if recovery.digest(ROOT / name) != sha:
                raise ValueError('source copy differs: ' + name)
        entry = ROOT / 'tools/models/q35_precision_cast_entry_v2_20261101.py'
        shutil.copyfile(ENTRY, entry)
        if recovery.digest(entry) != ENTRY_SHA:
            raise ValueError('entrypoint copy differs')
        row = registration['experiments'][0]
        original = ROOT / row['manifest']
        spec = copy.deepcopy(json.loads(original.read_text()))
        if spec.get('id') != ORIGINAL or len(spec.get('steps', [])) != 1:
            raise ValueError('unexpected whole-prefill manifest')
        step = spec['steps'][0]
        argv = step['argv']
        script = '{root}/tools/models/chat_quality.py'
        if argv.count(script) != 1 or argv.count('--prefill-chunk') != 1 or argv[argv.index('--prefill-chunk') + 1] != '0':
            raise ValueError('registered whole-prefill command changed')
        argv.insert(argv.index(script), '{root}/tools/models/q35_precision_cast_entry_v2_20261101.py')
        step['env']['TNCA_MAX_BLOCK'] = '32'
        step['env']['TNCA_IEEE_DIV'] = '0'
        step['inputs'] = step.get('inputs', []) + ['{root}/tools/models/q35_precision_cast_entry_v2_20261101.py']
        spec['id'] = NAME
        spec['registration']['scope'] = ('Separate diagnostic: unchanged multi-turn prompt and CPU scoring, whole-prefill, '
            'Inductor precision casts enabled and TNCA_MAX_BLOCK=32. A first-token TLM result is not a full-answer pass. '
            'The prior grouped and whole-prefill failures and original timeout remain unchanged.')
        manifest = ROOT / 'spec/r41/run3/board' / (NAME + '.json')
        with manifest.open('x') as f:
            json.dump(spec, f, indent=2); f.write('\n')
        row['id'], row['manifest'], row['sha256'] = NAME, str(manifest.relative_to(ROOT)), recovery.digest(manifest)
        registration['id'] = NAME
        reg = OUT / 'registration.json'
        with reg.open('x') as f:
            json.dump(registration, f, indent=2); f.write('\n')
        checks = dict(EXPECTED)
        checks[str(entry.relative_to(ROOT))] = ENTRY_SHA
        checks[str(manifest.relative_to(ROOT))] = row['sha256']
        (OUT / 'source-hashes.json').write_text(json.dumps(checks, indent=2) + '\n')
        reg_hash = recovery.digest(reg)
        recovery.record('registered', registration_sha256=reg_hash, source_hashes=checks,
                        scope='whole-prefill precision-cast b32 reduced quality diagnostic')
        sysfs = Path('/sys/bus/pci/devices')
        while True:
            with (WORK / 'candidate_queue.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    state = auto.probe(sysfs)
                    if state['uuid'] != UUID or auto.clock_mhz(sysfs) != 100:
                        raise RuntimeError('board image or clock changed; no programming attempted')
                    if auto.idle(state):
                        if recovery.digest(WORK / 'tnca_u280.xclbin') != IMAGE_SHA:
                            raise RuntimeError('image file hash changed')
                        if recovery.digest(reg) != reg_hash or any(recovery.digest(ROOT / p) != sha for p, sha in checks.items()):
                            raise RuntimeError('registered source changed while waiting')
                        cmd = [str(Path.home() / 'tc_env/torchenv/bin/python'), str(ROOT / 'tools/r41/run2_board/stage.py'),
                               '--root', str(ROOT), '--out', str(OUT / 'multiturn'),
                               '--python', str(Path.home() / 'tc_env/torchenv/bin/python'), '--registration', str(reg)]
                        recovery.record('stage-start', root=str(ROOT), board=state)
                        with (OUT / 'stage.log').open('x') as log:
                            result = subprocess.run(cmd, cwd=ROOT, env=auto.board_environment(), stdout=log, stderr=subprocess.STDOUT)
                        stage = OUT / 'multiturn/stage.json'
                        if result.returncode or not stage.is_file() or json.loads(stage.read_text()).get('status') != 'pass':
                            raise RuntimeError('registered precision diagnostic failed')
                        quality = OUT / 'multiturn' / NAME / 'q35-quality/quality.json'
                        data = json.loads(quality.read_text())
                        if data.get('status') != 'pass' or data.get('reduced_proof') is not True or len(data.get('prompts', [])) != 1:
                            raise ValueError('diagnostic quality did not pass')
                        recovery.check_prompt(data['prompts'][0])
                        recovery.record('finished', quality=str(quality), sha256=recovery.digest(quality))
                        return
            time.sleep(600)
    except BaseException as exc:
        recovery.record('stopped', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
