#!/usr/bin/env python3
"""Integrity-check the manually judged R41 Stage 2 ledger; never invent a pass.

Default exit 1 means not accepted. --integrity-only checks the audit artifacts,
not product acceptance. --overall also requires the archived frozen Stage 1
checker result. This does not execute board tests or independent reproduction.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = 'evidence/r41/acceptance/stage2_required_index.json'
REQUIRED_IDS = ('S2-00.1', 'S2-00.2', 'S2-00.3', 'S2-00.4', 'S2-00.5', 'S2-00.6', 'S2-00.7', 'S2-00.8', 'S2-00.process', 'S2-00.budget', 'S2-01.1', 'S2-01.2', 'S2-01.3', 'S2-01.4', 'S2-01.5', 'S2-01.6', 'S2-01.7', 'S2-01.8', 'S2-01.timing', 'S2-02.A', 'S2-02.B', 'S2-02.C', 'S2-02.C.formats', 'S2-02.D.hypothesis', 'S2-02.D.sensitivity', 'S2-02.D.semantics', 'S2-02.D.preflight', 'S2-02.D.decision', 'S2-02.D.board', 'S2-03.A.1', 'S2-03.A.2', 'S2-03.A.3', 'S2-03.A.4', 'S2-03.A.5', 'S2-03.B', 'S2-03.C', 'S2-04.A', 'S2-04.B.core', 'S2-04.B.tail', 'S2-04.C.1', 'S2-04.C.2', 'S2-04.C.3', 'S2-04.C.4', 'S2-04.C.5', 'S2-04.D.margin', 'S2-04.D.1', 'S2-04.D.2', 'S2-04.D.3', 'S2-04.D.4', 'S2-04.D.5', 'S2-04.D.6', 'S2-04.D.7', 'S2-04.D.8', 'S2-04.E.freeze', 'S2-04.E.1', 'S2-04.E.2', 'S2-04.E.3', 'S2-04.E.4', 'S2-04.E.5', 'S2-05.A.1', 'S2-05.A.2', 'S2-05.A.3', 'S2-05.A.4', 'S2-05.A.5', 'S2-05.A.6', 'S2-05.A.7', 'S2-05.A.8', 'S2-05.B', 'S2-05.C.1', 'S2-05.C.2', 'S2-05.C.3', 'S2-05.C.4', 'S2-05.C.5', 'S2-05.C.6', 'S2-05.D.1', 'S2-05.D.2', 'S2-05.D.3', 'S2-06.1', 'S2-06.2', 'S2-06.3', 'S2-06.4', 'S2-06.5', 'S2-06.6', 'S2-06.7', 'S2-06.8', 'S2-06.9', 'S2-06.10', 'DoD.Stage1', 'DoD.Stage2', 'DoD.R41', 'Common.2', 'Common.6.4', 'Common.6.5', 'Common.7')
STATUSES = {'passed', 'failed', 'not measured', 'blocked', 'descoped'}
AUTHOR_DIRECTIVE = 'evidence/r41/AUTHOR_DIRECTIVE_20261031.md'
CHAT_CHECKS = {'full_checkpoint', 'registered_quality', 'persistent_session', 'scripted_chat',
               'weights_activations_on_tnca', 'zero_cpu_numerical_fallback',
               'zero_steady_host_tensor_reads', 'board_prefill_decode_rates',
               'device_launch_cycles_host_share'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(root, index, overall=False):
    errors = []
    rows = index.get('items', [])
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(REQUIRED_IDS):
        errors.append('mandatory audit coverage changed or duplicate IDs')
    contract = root/index['contract']
    if digest(contract) != index['contract_sha256']:
        errors.append('acceptance contract hash mismatch')
    hashes = index.get('evidence_sha256', {})
    for name, expected in hashes.items():
        path = root/name
        if not path.is_file() or digest(path) != expected:
            errors.append(f'evidence missing or changed: {name}')
    # Review-only entries name the file version a reviewer read. They are not
    # re-verified against the current tree, and they never stand for a rerun.
    review_only = index.get('evidence_review_only', {})
    for name, entry in review_only.items():
        if name in hashes:
            errors.append(f'evidence both hashed and review-only: {name}')
        if not entry.get('reviewed_sha256') or not entry.get('note'):
            errors.append(f'review-only evidence lacks a reviewed hash or note: {name}')
    for row in rows + index.get('followups', []):
        if row['status'] not in STATUSES:
            errors.append(f"{row['id']}: invalid status")
        if not row.get('evidence') or not row.get('reason'):
            errors.append(f"{row['id']}: evidence/reason missing")
        if row['status'] != 'passed' and not row.get('closure_plan'):
            errors.append(f"{row['id']}: bounded closure plan missing")
        if row['status'] == 'descoped' and (row.get('prior_status') not in ('not measured', 'blocked')
                or not row.get('prior_reason') or AUTHOR_DIRECTIVE not in row.get('evidence', [])):
            errors.append(f"{row['id']}: descoping must preserve prior judgment and cite the author")
        for name in row.get('evidence', []):
            if name not in hashes and name not in review_only:
                errors.append(f"{row['id']}: unhashed evidence {name}")
    metrics = json.loads((root/'evidence/r41/acceptance/board_metrics.json').read_text())
    board = metrics['board']
    if board['clock_hz'] != 100000000 or board['image_sha256'] != '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de':
        errors.append('run-1 board identity mismatch')
    if metrics['original']['points'] != 113 or metrics['original']['large_points'] != 24:
        errors.append('original-suite inventory incomplete')
    author = index.get('author_scope')
    if author:
        if author.get('directive') != AUTHOR_DIRECTIVE or AUTHOR_DIRECTIVE not in hashes:
            errors.append('author scope lacks the hashed directive')
        models = author.get('models', [])
        if len(models) != 2 or {x.get('model') for x in models} != {'Qwen3-VL-2B-FP8', 'Qwen3.5-4B-FP8'}:
            errors.append('author scope must retain both chat models')
        for model in models:
            required = CHAT_CHECKS | ({'image_prompt'} if model.get('model') == 'Qwen3-VL-2B-FP8' else set())
            if set(model.get('checks', {})) != required or any(v not in ('not measured', 'failed', 'passed')
                                                               for v in model.get('checks', {}).values()):
                errors.append(f"{model.get('model')}: chat evidence inventory differs")
            if model.get('status') not in ('not measured', 'failed', 'passed'):
                errors.append(f"{model.get('model')}: invalid chat status")
            if model.get('status') == 'passed' and (not model.get('evidence')
                    or any(name not in hashes for name in model['evidence'])
                    or not model.get('checks') or any(v != 'passed' for v in model['checks'].values())):
                errors.append(f"{model.get('model')}: passing chat requires hashed evidence and passing checks")
        if author.get('status') == 'evidence complete' and any(m.get('status') != 'passed' for m in models):
            errors.append('author scope cannot be complete before both chat models pass')
        if author.get('status') not in ('pending', 'evidence complete'):
            errors.append('invalid author scope status')
    counts = dict(Counter(r['status'] for r in rows))
    accepted = not errors and all(r['status']=='passed' for r in rows)
    result = dict(scope=index['scope'], integrity='failed' if errors else 'passed',
                  status='passed' if accepted else 'not accepted', counts=counts,
                  errors=errors, review_only_evidence=sorted(review_only),
                  items=rows, followups=index.get('followups', []),
                  author_scope=index.get('author_scope'))
    if overall:
        s1 = json.loads((root/'evidence/r41/stage1/board/acceptance.json').read_text())
        s1counts = dict(Counter(r['status'] for r in s1['items']))
        result['stage1_frozen_snapshot_counts'] = s1counts
        if not all(r['status']=='pass' for r in s1['items']):
            result['status'] = 'not accepted'
    return result

def render(index):
    out = ['# Stage 2 and R41 acceptance item ledger', '',
           index['scope'], '',
           'Status is a manual evidence judgment, protected by the integrity checker. '
           'Passed candidate items do not imply final release acceptance. Study rows '
           'record mandatory study obligations; numerical and functional Required '
           'items cannot close through a negative result.', '',
           'Each non-pass names a bounded plan in [closure_plan.md](closure_plan.md). '
           'Identity boundaries are in [identity.md](identity.md).', '',
           '| Item | Type | Requirement | Judgment | Evidence and reason | Closure |',
           '|---|---|---|---|---|---|']
    review_only = index.get('evidence_review_only', {})
    for r in index['items']:
        evidence = '; '.join(('[author directive](../AUTHOR_DIRECTIVE_20261031.md)' if p == AUTHOR_DIRECTIVE
                              else '`'+p+'`'+(' (review-only hash)' if p in review_only else ''))
                             for p in r['evidence'])
        out.append('| '+ ' | '.join([r['id'],r['classification'],r['requirement'],r['status'],evidence+' — '+r['reason'],r['closure_plan'] or 'Closed for stated scope'])+' |')
    out += ['', '## Supplementary delivery follow-ups', '',
            '| Follow-up | Judgment | Evidence and scope |', '|---|---|---|']
    for row in index.get('followups', []):
        out.append('| '+row['id']+' | '+row['status']+' | '+row['reason']+' '+
                   '; '.join('`'+p+'`' for p in row['evidence'])+' |')
    out += ['', 'FP8 layer campaigns do not replace the mandatory BF16 '
            'Qwen3.5-2B full-model workload under the historical gate.', '']
    if index.get('author_scope'):
        scope = index['author_scope']
        out += ['## Author-defined chat closure', '',
                '[Author directive](../AUTHOR_DIRECTIVE_20261031.md): '+scope['rule'], '',
                'Status: **'+scope['status']+'**. Historical failures and measured results remain unchanged.', '',
                '| Model | Chat status | Required evidence |', '|---|---|---|']
        for model in scope['models']:
            out.append('| '+model['model']+' | '+model['status']+' | '+model['evidence_required']+' |')
        out.append('')
    return '\n'.join(out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--render', type=Path)
    parser.add_argument('--overall', action='store_true')
    parser.add_argument('--integrity-only', action='store_true')
    args = parser.parse_args()
    index = json.loads((args.index or args.root/INDEX).read_text())
    result = evaluate(args.root, index, args.overall)
    if args.render:
        args.render.write_text(render(index))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='items'}, indent=2))
    return 0 if result['integrity']=='passed' and (args.integrity_only or result['status']=='passed') else 1

if __name__ == '__main__':
    raise SystemExit(main())
