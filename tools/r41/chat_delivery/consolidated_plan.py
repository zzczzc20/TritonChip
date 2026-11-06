#!/usr/bin/env python3
"""Prepare, but do not run, the one-session-per-model board campaign."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
IMAGE_UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
MODELS = {'vl': ('qwen3vl-2b-fp8', 6), 'q35': ('qwen3.5-4b-fp8', 5)}
MODEL_BUDGET_SECONDS = {'vl': 7200, 'q35': 20700}
SOURCE_DIRS = ('python/tnca', 'compiler/python', 'compiler/targets', 'compiler/include',
               'compiler/lib', 'compiler/tools', 'compiler/pin', 'tools/asm', 'tools/models',
               'tools/performance_harness', 'tools/r41/run2_board', 'tools/r41/chat_delivery',
               'archbench', 'isa', 'spec')
SOURCE_FILES = ('compiler/CMakeLists.txt',)
# These manifests are copied from the signed plan into the frozen checkout;
# they have their own hashes in the plan and must not change the source hash.
OVERLAY_MANIFESTS = frozenset(f'spec/r41/run3/board/r41-final-{model}-campaign-20261101.json'
                              for model in MODELS)
# These original run-3 registrations live in the board archive, not r41.
# Bind them separately, just like the generated campaign manifests.
EXTERNAL_INPUTS = ('spec/r41/run3/xclbin_hbm_map.json',
                   'spec/r41/run3/board/r3-s2-qwen-chat-v3-vl.json',
                   'spec/r41/run3/board/r3-s2-qwen-chat-v3-q35.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root):
    """Bind the source used for compilation and measurement across checkouts."""
    digest = hashlib.sha256()
    root = Path(root)
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f'campaign source directory is missing: {name}')
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and relative not in OVERLAY_MANIFESTS and relative not in EXTERNAL_INPUTS and '__pycache__' not in path.parts \
                    and not path.name.endswith(('.pyc', '.pyo')):
                digest.update(relative.encode() + b'\0')
                digest.update(bytes.fromhex(sha(path)))
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'campaign source file is missing: {name}')
        digest.update(name.encode() + b'\0')
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def replace_flag(argv, flag, expected, replacement):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise ValueError(f'unexpected {flag} in registered quality template')
    argv[argv.index(flag) + 1] = replacement


def prepare(root, templates, references, out, frozen_commit=None):
    root, templates, references, out = map(Path, (root, templates, references, out))
    if out.exists():
        raise FileExistsError('campaign output already exists')
    if frozen_commit is not None:
        if not re.fullmatch(r'[0-9a-f]{40}', frozen_commit):
            raise ValueError('frozen commit must be a full SHA-1')
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
        if git('rev-parse', 'HEAD') != frozen_commit or git('rev-parse', 'origin/r41') != frozen_commit:
            raise ValueError('frozen source must equal the fetched origin/r41 commit')
        if git('status', '--porcelain', '--untracked-files=all', '--', *SOURCE_DIRS, *SOURCE_FILES):
            raise ValueError('frozen compiler, runtime or harness source is dirty')
    spec_path = root / 'spec/r41/chat/quality-author-20261101.json'
    script_path = root / 'spec/r41/chat/script-text.txt'
    checks_path = root / 'spec/r41/chat/script-author-20261101.json'
    spec, checks = (json.loads(path.read_text()) for path in (spec_path, checks_path))
    if spec.get('scoring_rule') != 'author-20261101' or \
            checks.get('schema') != 'tnca-chat-script-checks/1' or checks.get('script_sha256') != sha(script_path):
        raise ValueError('registered quality or scripted-chat checks do not match')
    source_files = ('python/tnca/compile_cache.py', 'python/tnca/inductor.py', 'python/tnca/runtime.py',
                    'python/tnca/triton_backend/driver.py', 'tools/models/chat.py', 'tools/models/chat_quality.py',
                    'tools/models/chat_timing.py', 'spec/r41/chat/quality-author-20261101.json',
                    'spec/r41/chat/script-author-20261101.json', 'spec/r41/chat/script-text.txt')
    plan = dict(schema='r41-consolidated-campaign/1', status='frozen' if frozen_commit else 'provisional',
                frozen_commit=frozen_commit, image_sha256=IMAGE_SHA, image_uuid=IMAGE_UUID,
                clock_hz=100000000, not_before_utc='2026-11-01T17:50:00Z',
                campaign_deadline_utc='2026-11-02T00:50:00Z',
                identity_checks=1, models=[], estimated_hours=6.75,
                estimate_scope=('about 6-7 hours: VL up to two hours, Q35 up to 5h45m, '
                                'plus idle time, all capped by 00:50 UTC; retry time is not guaranteed'),
                source_tree_sha256=tree_digest(root),
                source_sha256={name: sha(root / name) for name in source_files},
                warm_scope='same-session repeat; fresh-process persistent-cache gain is not established')
    plan['external_source_sha256'] = {}
    for name in EXTERNAL_INPUTS:
        source = templates / name
        if not source.is_file():
            raise FileNotFoundError(f'registered board input missing: {source}')
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        plan['external_source_sha256'][name] = sha(target)
    for model, (model_name, count) in MODELS.items():
        reg0_path = templates / f'spec/r41/run3/stages/chat-{model}-nohost-v5-20261031.json'
        reg0 = json.loads(reg0_path.read_text())
        row0 = next(row for row in reg0['experiments'] if '-quality-' in row['id'])
        source = templates / row0['manifest']
        if sha(source) != row0['sha256']:
            raise ValueError(f'{model}: original registered manifest changed')
        reference_path = references / f'r41_author_ref_{model}_cpu_20261101/quality.json'
        reference = json.loads(reference_path.read_text())
        expected = [p for p in spec['prompts'] if model in p.get('models', ['vl', 'q35'])]
        prompts = reference.get('prompts') or []
        if reference.get('status') != 'reference-recorded' or reference.get('device') != 'cpu' or \
                reference.get('model') != model_name or reference.get('model_scope') != 'full-checkpoint' or \
                len(expected) != count or [p['id'] for p in prompts] != [p['id'] for p in expected]:
            raise ValueError(f'{model}: incomplete or wrong CPU reference')
        for prompt, registered in zip(prompts, expected, strict=True):
            if [t.get('user') for t in prompt['turns']] != registered['turns']:
                raise ValueError(f'{model}: CPU reference prompt changed')
            for turn in prompt['turns']:
                evidence = turn.get('first_token_reference') or {}
                if not turn.get('tokens') or evidence.get('top1_token') != turn['tokens'][0] or \
                        evidence.get('vocab_size', 0) <= 0 or not evidence.get('top5_tokens') or \
                        evidence.get('top1_token') not in evidence.get('within_half_logit_tokens', []):
                    raise ValueError(f'{model}: missing CPU first-token evidence')
        manifest = copy.deepcopy(json.loads(source.read_text()))
        board = manifest.get('board') or {}
        memory_map = board.get('memory_map') or {}
        if memory_map.get('source') == EXTERNAL_INPUTS[0] and \
                memory_map.get('source_sha256') != plan['external_source_sha256'][EXTERNAL_INPUTS[0]]:
            raise ValueError(f'{model}: board memory map differs from the registered image')
        if (board.get('image_sha256'), board.get('image_uuid'), board.get('clock_hz')) != \
                (IMAGE_SHA, IMAGE_UUID, 100000000) or len(manifest.get('steps', [])) != 1:
            raise ValueError(f'{model}: unexpected board identity or extra steps')
        name = f'r41-final-{model}-campaign-20261101'
        manifest['id'] = name
        manifest['registration']['scope'] = ('One full-checkpoint process covers registered quality, '
            'same-session warm repeat, no-fallback evidence, user-visible speed and scripted chat.')
        ref_relative = f'evidence/r41/chat/quality-ref/author-{model}-20261101.json'
        manifest['sources'] += ['spec/r41/chat/quality-author-20261101.json',
                                'spec/r41/chat/script-author-20261101.json', ref_relative]
        step = manifest['steps'][0]
        manifest['budget_seconds'] = MODEL_BUDGET_SECONDS[model]
        step['timeout_seconds'] = MODEL_BUDGET_SECONDS[model]
        argv = step['argv']
        replace_flag(argv, '--spec', '{root}/spec/r41/chat/quality-v1.json',
                     '{root}/spec/r41/chat/quality-author-20261101.json')
        replace_flag(argv, '--reference', f'{{root}}/evidence/r41/chat/quality-ref/{model}.json',
                     '{root}/' + ref_relative)
        argv += ['--warm-repeat', '--script', '{root}/spec/r41/chat/script-text.txt',
                 '--script-checks', '{root}/spec/r41/chat/script-author-20261101.json']
        step['env']['TNCA_COMPILE_CACHE_DIR'] = '{out}/caches/tnca-inductor'
        step['timing_boundary'] = ('first Python module entry and later prompt entry through flushed assistant output; '
                                   'excludes interpreter startup')
        step['cache_state'] = 'fresh persistent namespace initially; same-session first-prompt repeat is warm'
        relative = f'spec/r41/run3/board/{name}.json'
        manifest_path = out / relative
        put(manifest_path, manifest)
        ref_target = out / ref_relative
        ref_target.parent.mkdir(parents=True, exist_ok=True)
        with ref_target.open('xb') as stream:
            stream.write(reference_path.read_bytes())
        registration = dict(schema='tnca-run2-stage-registration/1', id=name,
            max_experiment_seconds=MODEL_BUDGET_SECONDS[model],
            image_identity=dict(sha256=IMAGE_SHA, uuid=IMAGE_UUID),
            experiments=[dict(id=name, manifest=relative, sha256=sha(manifest_path),
                budget_seconds=MODEL_BUDGET_SECONDS[model],
                checks=[dict(path=f'{model}-quality/quality.json', field=['status'], equals='pass'),
                        dict(path=f'{model}-quality/quality.json', field=['model_scope'], equals='full-checkpoint'),
                        dict(path=f'{model}-quality/quality.json', field=['reduced_proof'], equals=False),
                        dict(path=f'{model}-quality/quality.json', field=['prompts'], length_equals=count),
                        dict(path=f'{model}-quality/quality.json', field=['warm_repeat', 'passed'], equals=True),
                        dict(path=f'{model}-quality/quality.json', field=['scripted_chat', 'passed'], equals=True)])])
        reg_path = out / f'registrations/{name}.json'
        put(reg_path, registration)
        plan['models'].append(dict(model=model, manifest=relative, manifest_sha256=sha(manifest_path),
                                   registration=str(reg_path.relative_to(out)), registration_sha256=sha(reg_path),
                                   reference=ref_relative, reference_sha256=sha(ref_target)))
    put(out / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'templates', 'references', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frozen-commit', help='full origin/r41 commit after the author-directed freeze')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.templates, args.references, args.out, args.frozen_commit), indent=2))


if __name__ == '__main__':
    main()
