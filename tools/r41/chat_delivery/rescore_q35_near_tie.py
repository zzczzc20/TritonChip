#!/usr/bin/env python3
"""Rescore recorded Q35 board answers under the author decision, without modifying them."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_archive(path, name):
    with tarfile.open(path, 'r:gz') as archive:
        return json.load(archive.extractfile(name))

def has(text, word):
    return re.search(r'(?<!\w)' + re.escape(word) + r'(?!\w)', text, re.I) is not None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    evidence = ROOT / 'evidence/r41/chat/run3_priority'
    four_path = evidence / 'q35_nohost_v6_four_prompts_timeout.json'
    old_path = evidence / 'q35_multiturn_segment_failed_20261031.json'
    ref_path = ROOT / 'evidence/r41/chat/quality-ref/q35.json'
    archive_path = evidence / 'q35_multiturn_tlm_grouped_20261101.tgz'
    raw_archive = evidence / 'q35_multiturn_segment_failed_20261031.tgz'
    four, old, refs = (json.loads(path.read_text()) for path in (four_path, old_path, ref_path))
    raw = read_archive(raw_archive, 'multiturn/r41-q35-segment-multiturn-20261031/q35-quality/quality.json')
    comp = read_archive(archive_path, 'r41_q35_multiturn_tlm_grouped_compare_20261101.json')
    prompts = {x['id']: x for x in refs['prompts']}
    expected = {'fact': 'paris', 'arith': '42', 'reason': 'bob', 'instruct': 'ok'}
    rows = []
    if [x['id'] for x in four['prompts']] != list(expected):
        raise ValueError('unexpected four-prompt board snapshot')
    for prompt in four['prompts']:
        pid = prompt['id']; turn = prompt['turns'][0]; ref = prompts[pid]['turns'][0]
        if not turn['tokens'] or not ref['tokens'] or turn['tokens'][0] != ref['tokens'][0]:
            raise ValueError(f'{pid}: first token does not match recorded CPU greedy top token')
        content = has(turn['text'], expected[pid])
        rows.append(dict(id=pid, content_pass=content, first_token=turn['tokens'][0],
                         cpu_top1=ref['tokens'][0], first_token_rule='equal to CPU greedy top token',
                         old_pass=prompt['passed'], new_pass=content))
    original = raw['prompts'][0]
    if (original['id'] != 'multiturn' or old['passed'] is not False or len(original['turns']) != 2
            or original['turns'][0]['user'] != 'My name is Alice. Please remember it and reply with one short sentence.'
            or original['turns'][0]['prompt_tokens'] != len(comp['input_ids'][0])):
        raise ValueError('multi-turn board input or original failure is not the registered case')
    cpu_top5 = comp['comparisons']['grouped_chunk8']['reference_top5']['top5']
    first, second = original['turns']; ref_turns = prompts['multiturn']['turns']
    content1 = has(first['text'], 'alice') and bool(re.search(r'\b(remember\w*|noted?|acknowledg\w*)\b', first['text'], re.I))
    content2 = has(second['text'], 'alice')
    top5_ids = [x['id'] for x in cpu_top5]
    if not first['tokens'] or not second['tokens'] or not ref_turns[1]['tokens']:
        raise ValueError('missing first token')
    rule1 = first['tokens'][0] in top5_ids
    rule2 = second['tokens'][0] == ref_turns[1]['tokens'][0]
    turns = [dict(index=0,text=first['text'],content_pass=content1,first_token=first['tokens'][0],
                  cpu_top5=cpu_top5,first_token_rule='in CPU reference top five',first_token_pass=rule1,
                  old_score=original['scores'][0],new_pass=content1 and rule1),
             dict(index=1,text=second['text'],content_pass=content2,first_token=second['tokens'][0],
                  cpu_top1=ref_turns[1]['tokens'][0],first_token_rule='equal to CPU greedy top token',first_token_pass=rule2,
                  old_score=original['scores'][1],new_pass=content2 and rule2)]
    result = dict(schema='r41-q35-author-near-tie-rescore/1',decision_date='2026-11-01',
       scope='separate rescore of recorded outputs from original four-prompt run and a segmented multi-turn run',
       original_run='timeout after four passed prompts; unchanged',original_multiturn='failed under then-registered containment rule; unchanged',
       source_sha256={str(x.relative_to(ROOT)):sha(x) for x in (four_path,old_path,ref_path,archive_path,raw_archive)},
       first_four=rows,multiturn=dict(old_pass=original['passed'],new_pass=all(x['new_pass'] for x in turns),turns=turns),
       combined_rescore_pass=all(x['new_pass'] for x in rows) and all(x['new_pass'] for x in turns),
       limitations=['Separate sessions; not a replacement for the original timeout or old-rule failure.',
                    'Board logits were not captured; first-turn test uses CPU top-five membership, not a claimed board logit gap.'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'combined_rescore_pass':result['combined_rescore_pass'],'original_multiturn':original['passed'],'new_multiturn':result['multiturn']['new_pass']},indent=2))
if __name__ == '__main__':main()
