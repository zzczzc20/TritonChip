#!/usr/bin/env python3
"""Score recorded quality outputs against a later CPU reference without changing them."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/models'))
from chat_quality import assess_no_fallback, prior_assistant_text_matches, score_author_turn  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_identity(device, reference):
    """Bind current records directly and legacy records by their pinned snapshot."""
    repo, revision = reference.get('repo'), reference.get('revision')
    if not repo or not revision:
        raise ValueError('CPU checkpoint identity is missing')
    recorded = (device.get('repo'), device.get('revision'))
    if recorded == (repo, revision):
        return 'explicit repo and revision'
    if recorded != (None, None):
        raise ValueError('device/CPU checkpoint identity differs or is incomplete')
    actual = (device.get('load') or {}).get('snapshot_view') or {}
    expected = (reference.get('load') or {}).get('snapshot_view') or {}
    actual_hash, expected_hash = actual.get('original_index_sha256'), expected.get('original_index_sha256')
    name = repo.replace('/', '--') + '--' + revision
    if not re.fullmatch(r'[0-9a-f]{40}', revision) or \
            not isinstance(actual_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', actual_hash) or \
            actual_hash != expected_hash or Path(actual.get('view') or '').name != name:
        raise ValueError('legacy checkpoint lacks a matching pinned snapshot and index hash')
    return 'legacy pinned snapshot name and matching checkpoint index SHA-256'


def rescore(device, reference, spec):
    if spec.get('scoring_rule') != 'author-20261101':
        raise ValueError('the author-rule spec is required')
    if device.get('device') != 'tnca' or reference.get('device') != 'cpu' or \
            reference.get('status') != 'reference-recorded':
        raise ValueError('expected recorded device results and completed CPU reference')
    for key in ('model', 'model_scope', 'reduced_depth_layers'):
        if device.get(key) != reference.get(key):
            raise ValueError(f'device/CPU {key} differs or is missing')
    identity = checkpoint_identity(device, reference)
    if device.get('model_scope') != 'full-checkpoint':
        raise ValueError('reduced-depth evidence is not full-model quality')
    models = {'qwen3vl-2b-fp8': 'vl', 'qwen3.5-4b-fp8': 'q35'}
    if device.get('model') not in models:
        raise ValueError('unknown quality model')
    model = models[device['model']]
    expected = {p['id']: p for p in spec['prompts'] if model in p.get('models', ['vl', 'q35'])}
    ref = {p['id']: p for p in reference.get('prompts', [])}
    seen = set()
    results = []
    for item in device.get('prompts', []):
        name = item['id']
        if name in seen or name not in expected or name not in ref:
            raise ValueError(f'unexpected or duplicate prompt: {name}')
        seen.add(name)
        prompt = expected[name]
        if ([turn.get('user') for turn in item['turns']] != prompt['turns'] or
                [turn.get('user') for turn in ref[name]['turns']] != prompt['turns']):
            raise ValueError(f'{name}: device and CPU prompts differ')
        rows = []
        for index, (turn, cpu, check) in enumerate(zip(item['turns'], ref[name]['turns'],
                                                        prompt['content_checks'], strict=True)):
            keywords = prompt['keywords'] if index == len(item['turns']) - 1 else []
            score = score_author_turn(turn, cpu, check, keywords)
            rows.append(dict(score=score, no_fallback=assess_no_fallback(turn, vision_expected=bool(prompt.get('images'))),
                             prior_assistant_text_matches_cpu=prior_assistant_text_matches(
                                 item['turns'], ref[name]['turns'], index),
                             original_score=(item.get('scores') or [None] * len(item['turns']))[index]))
        results.append(dict(id=name, turns=rows))
    missing = sorted(set(expected) - seen)
    verdicts = [row['score']['passed'] for item in results for row in item['turns']]
    proof = [row['no_fallback'] for item in results for row in item['turns']]
    completed = device.get('status') in ('pass', 'failed-chat-quality', 'unverified-chat-quality')
    status = ('failed' if False in verdicts or False in proof else
              'unverified' if missing or not completed or None in verdicts or None in proof or not results else 'pass')
    return dict(schema='r41-deadline-quality-rescore/1', status=status, model=device['model'],
                checkpoint_identity=identity,
                original_status=device.get('status'), missing_prompts=missing,
                scope='separate rescore; original device and CPU records remain unchanged', prompts=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = rescore(*(json.loads(path.read_text()) for path in (args.device, args.reference, args.spec)))
    result['input_sha256'] = {str(path): sha(path) for path in (args.device, args.reference, args.spec)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'missing_prompts': result['missing_prompts']}))
    return 0 if result['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
