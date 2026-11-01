#!/usr/bin/env python3
"""Registered chat-quality check (spec/r41/chat/quality-v1.json): CPU reference answers, then device scoring.

  # 1. before any device run: the CPU W8A8 reference model's own answers
  python tools/models/chat_quality.py --model vl --device cpu --spec spec/r41/chat/quality-v1.json --out REF_DIR
  # 2. the device, scored against that archived reference
  python tools/models/chat_quality.py --model vl --device tnca --spec ... --reference REF_DIR/quality.json --out OUT

Each prompt is a fresh conversation in one persistent session (multi-turn prompts keep their history). Greedy
decoding, stop at EOS or the spec's new_tokens. The scoring rule is the one written in the spec; nothing here is
tuned to a model or an answer.
"""
import time
_MODULE_STARTED = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat  # noqa: E402


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def prior_assistant_text_matches(device_turns, reference_turns, index):
    """Disclose whether the CPU reference and device saw the same prior answers."""
    if type(index) is not int or index < 0 or index > min(len(device_turns), len(reference_turns)):
        raise ValueError('conversation history index is invalid')
    return all(isinstance(device.get('text'), str) and device['text'] == reference.get('text')
               for device, reference in zip(device_turns[:index], reference_turns[:index], strict=True))


def score_turn(device, reference, keywords):
    dn, rn = normalize(device["text"]), normalize(reference["text"])
    required = [k for k in keywords if k.lower() in reference["text"].lower()]
    missing_in_reference = [k for k in keywords if k not in required]
    keyword_ok = all(k.lower() in device["text"].lower() for k in required)
    contained = bool(rn) and (rn in dn or (dn and dn in rn))
    a, b = device["tokens"], reference["tokens"]
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(passed=bool(keyword_ok and contained), required_keywords=required,
                keywords_missing_in_reference=missing_in_reference, keywords_ok=keyword_ok, containment=contained,
                tokens_equal=a == b, common_prefix_tokens=prefix, device_tokens=len(a), reference_tokens=len(b))


def first_token_reference(logits):
    """Record enough CPU logits to decide either part of the author's rule."""
    import torch
    values = logits.detach().cpu().flatten().float()
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("CPU first-token logits must be nonempty and finite")
    top = torch.topk(values, min(5, values.numel()))
    maximum, top1 = values.max(dim=0)
    return dict(vocab_size=values.numel(), top1_token=int(top1), top1_logit=float(maximum),
                top5_tokens=[int(x) for x in top.indices],
                within_half_logit_tokens=[int(x) for x in torch.nonzero(values >= maximum - 0.5).flatten()])


def _has_term(text, term):
    return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.IGNORECASE) is not None


def content_check(text, check):
    required, alternatives, forbidden = (check.get('all_terms', []), check.get('any_terms', []),
                                         check.get('none_terms', []))
    if any(not isinstance(value, list) for value in (required, alternatives, forbidden)) or \
            not (required or alternatives) or not all(isinstance(x, str) and x
                                                for x in required + alternatives + forbidden):
        raise ValueError('scoring requires explicit nonempty content checks')
    return all(_has_term(text, term) for term in required) and \
        (not alternatives or any(_has_term(text, term) for term in alternatives)) and \
        not any(_has_term(text, term) for term in forbidden)


def score_author_turn(device, reference, check, legacy_keywords):
    """Keep the old verdict alongside the pre-registered author-rule verdict."""
    legacy = score_turn(device, reference, legacy_keywords)
    content = content_check(device['text'], check)
    tokens, ref_tokens = device.get('tokens') or [], reference.get('tokens') or []
    token = tokens[0] if tokens else None
    first = None
    reason = 'missing CPU first-token evidence'
    if token is not None and ref_tokens and token == ref_tokens[0]:
        first, reason = True, 'equal to CPU greedy first token'
    elif token is not None and reference.get('first_token_reference') is not None:
        evidence = reference['first_token_reference']
        top1 = evidence['top1_token']
        vocab = evidence['vocab_size']
        top5, near = evidence['top5_tokens'], evidence['within_half_logit_tokens']
        if not ref_tokens or top1 != ref_tokens[0] or top1 not in near or \
                not isinstance(vocab, int) or vocab <= 0 or \
                any(not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= vocab for x in top5 + near):
            raise ValueError('inconsistent CPU first-token evidence')
        first = token in top5 or token in near
        reason = 'CPU top five or within 0.5 logit' if first else 'outside CPU top five and 0.5 logit'
    elif token is None:
        first, reason = False, 'device returned no token'
    passed = False if not content or first is False else (True if first else None)
    return dict(passed=passed, content_pass=content, first_token=token,
                first_token_pass=first, first_token_rule=reason, old_score=legacy)


def assess_no_fallback(turn, vision_expected=False):
    """True, False, or None when recorded evidence is incomplete."""
    evidence = turn.get('device_evidence') or {}
    if evidence.get('cpu_numerical_fallback') is True or evidence.get('compiled_wrapper_roles_covered') is False:
        return False
    incomplete = evidence.get('cpu_numerical_fallback') is not False or \
        evidence.get('compiled_wrapper_roles_covered') is not True
    wrappers = evidence.get('compiled_wrapper_fallbacks')
    if not isinstance(wrappers, dict) or any(role not in wrappers for role in ('prefill', 'decode', 'selection')):
        incomplete = True
    if isinstance(wrappers, dict) and any(wrappers.values()):
        return False
    for kind in ('parameters', 'buffers'):
        placement = (evidence.get('model_tensor_placements') or {}).get(kind) or {}
        if not placement:
            incomplete = True
        if any(not device.startswith('tnca') or type(count) is not int or count <= 0
               for device, count in placement.items()):
            return False
    steps = [step for step in turn.get('steps', []) if step.get('kind') == 'decode']
    if not steps:
        incomplete = True
    for step in steps:
        transfers = step.get('transfers_inside_step') or {}
        reads, staging = transfers.get('host_read_calls'), transfers.get('staging_download_bytes')
        if reads is None or staging is None:
            incomplete = True
        if (reads is not None and reads != 0) or (staging is not None and staging != 0):
            return False
    vision = evidence.get('vision_calls') or []
    if vision_expected and not vision:
        incomplete = True
    for call in vision:
        fallbacks = call.get('compiled_wrapper_fallbacks')
        if fallbacks is None:
            incomplete = True
        if fallbacks:
            return False
    return None if incomplete else True


def combine_verdicts(verdicts):
    if not verdicts or any(value is None for value in verdicts):
        return False if any(value is False for value in verdicts) else None
    return not any(value is False for value in verdicts)


def bitwise_logits_equal(before, after):
    import torch
    if not before or len(before) != len(after):
        return False
    return all(a.shape == b.shape and a.dtype == b.dtype and
               torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
               for a, b in zip(before, after, strict=True))


def assess_script(transcript, checks, device):
    if len(transcript) != len(checks) or not transcript:
        return dict(passed=False, reason='script did not complete every registered turn', turns=[])
    rows = []
    for entry, check in zip(transcript, checks, strict=True):
        report = entry.get('report') or {}
        content = content_check(entry.get('assistant', ''), check)
        fallback = (assess_no_fallback(dict(device_evidence=report, steps=report.get('steps', [])),
                                       vision_expected=bool(report.get('images'))) if device == 'tnca' else True)
        complete = report.get('status') == 'pass'
        passed = combine_verdicts([content, fallback, complete])
        rows.append(dict(content_pass=content, no_fallback=fallback, passed=passed))
    return dict(passed=combine_verdicts([row['passed'] for row in rows]), turns=rows)


def publish_quality(path, result):
    """Keep the last completed prompt if a long session is interrupted."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=1) + '\n')
    temporary.replace(path)


def main():
    process_started = _MODULE_STARTED if __name__ == '__main__' else time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("vl", "q35"), required=True)
    ap.add_argument("--device", choices=("tnca", "cpu"), default="tnca")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="archived CPU reference quality.json (required for tnca)")
    ap.add_argument("--root", type=Path, default=HERE.parents[1], help="tree that image paths are relative to")
    ap.add_argument("--layers", type=int, help="reduced-depth proof only")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="passed to chat.py (bounds compiled prefill shapes)")
    ap.add_argument("--prompts", help="comma-separated prompt ids: a reduced proof run, labelled as such")
    ap.add_argument("--max-new-tokens", type=int, help="override new_tokens: a reduced proof run, labelled as such")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tnca-tree")
    ap.add_argument('--warm-repeat', action='store_true',
                    help='repeat the first prompt in the same session and compare logits bitwise')
    ap.add_argument('--script', type=Path, help='run registered scripted chat in the same session')
    ap.add_argument('--script-checks', type=Path, help='registered content checks and script hash')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    scoring = spec.get('scoring_rule', 'quality-v1')
    if scoring not in ('quality-v1', 'author-20261101'):
        raise ValueError(f'unknown quality scoring rule: {scoring}')
    if bool(a.script) != bool(a.script_checks):
        raise ValueError('script and registered script checks must be supplied together')
    script_lines = script_checks = script_hash = None
    if a.script:
        raw_script = a.script.read_bytes()
        script_hash = hashlib.sha256(raw_script).hexdigest()
        script_checks = json.loads(a.script_checks.read_text())
        if script_checks.get('schema') != 'tnca-chat-script-checks/1' or \
                script_checks.get('script_sha256') != script_hash or not script_checks.get('checks'):
            raise ValueError('script does not match its registered checks')
        script_lines = raw_script.decode('utf-8').splitlines()
        submitted = []
        for line in map(str.strip, script_lines):
            if line in ('/quit', '/exit'):
                break
            if line and not line.startswith('/'):
                submitted.append(line)
        if len(submitted) != len(script_checks['checks']):
            raise ValueError('script turn count does not match its registered checks')
        for check in script_checks['checks']:
            content_check('', check)
    if a.device == "tnca" and a.reference is None:
        raise SystemExit("--reference (the archived CPU answers) is required on the device")
    reference = json.loads(a.reference.read_text()) if a.reference else None
    new_tokens = a.max_new_tokens or spec["new_tokens"]
    selected = set(a.prompts.split(",")) if a.prompts else None
    args = chat.parser().parse_args(["--model", a.model, "--device", a.device, "--new-tokens", str(new_tokens),
                                     "--bucket", str(spec["bucket"]), "--threads", str(a.threads),
                                     "--prefill-chunk", str(a.prefill_chunk)]
                                    + (["--layers", str(a.layers)] if a.layers else [])
                                    + (["--tnca-tree", a.tnca_tree] if a.tnca_tree else []))
    load_started = time.perf_counter()
    app = chat.ChatApplication(args)
    load_seconds = time.perf_counter() - load_started
    a.out.mkdir(parents=True, exist_ok=True)
    result = dict(schema="tnca-chat-quality-result/1", spec=spec["id"], spec_path=str(a.spec), model=app.spec.name,
                  repo=app.spec.repo, revision=app.spec.revision,
                  device=a.device, model_scope="reduced-depth-proof" if a.layers else "full-checkpoint",
                  reference=str(a.reference) if a.reference else None, prompts=[], status="running",
                  prefill_chunk=a.prefill_chunk, new_tokens=new_tokens,
                  reduced_proof=bool(a.layers or selected or a.max_new_tokens), scoring_rule=scoring,
                  reduced_depth_layers=a.layers, model_load_seconds=load_seconds)
    first_prompt = True
    first_capture = None
    try:
        if reference is not None and scoring == 'author-20261101':
            if reference.get('device') != 'cpu' or reference.get('model') != app.spec.name or \
                    reference.get('status') != 'reference-recorded' or \
                    reference.get('model_scope') != result['model_scope'] or \
                    (reference.get('repo') is not None and reference['repo'] != app.spec.repo) or \
                    (reference.get('revision') is not None and reference['revision'] != app.spec.revision) or \
                    (a.layers and reference.get('reduced_depth_layers') != a.layers):
                raise ValueError('CPU reference does not match the model or depth')
        for prompt in spec["prompts"]:
            if a.model not in prompt.get("models", ["vl", "q35"]) or (selected and prompt["id"] not in selected):
                continue
            app.reset_conversation()
            images = chat.open_images([a.root / p for p in prompt.get("images", [])])
            turns = []
            checks = prompt.get('content_checks', [])
            if scoring == 'author-20261101' and len(checks) != len(prompt['turns']):
                raise ValueError(f"{prompt['id']}: one content check is required per turn")
            for i, text in enumerate(prompt["turns"]):
                prompt_started = process_started if first_prompt else time.perf_counter()
                first_prompt = False
                report = app.generate(text, images if i == 0 else (), keep_history=True,
                                      prompt_started=prompt_started)
                if report["status"] != "pass":
                    result["failed_generation"] = chat.public_report(report)
                    raise RuntimeError(f"generation failed or lacks device evidence: {report['status']}")
                if a.warm_repeat and first_capture is None:
                    first_capture = (text, images if i == 0 else (), list(report['generated_tokens']),
                                     [value.detach().clone() for value in report.get('logits', [])])
                print(json.dumps(dict(prompt=prompt["id"], turn=i, user=text, assistant=report["generated_text"])),
                      flush=True)
                from chat_timing import finalize_user_visible_metrics
                finalize_user_visible_metrics(report['user_visible_timing'], time.perf_counter() - prompt_started,
                                              'module entry through first output flush; excludes interpreter startup'
                                              if len(result['prompts']) == 0 and i == 0
                                              else 'prompt entry through flushed assistant output')
                turn = dict(user=text, text=report["generated_text"], tokens=report["generated_tokens"],
                                  prompt_tokens=report["prompt_tokens"],
                                  statistics=report["statistics"], session=report["session"],
                                  user_visible_timing=report['user_visible_timing'],
                                  device_evidence={k: report.get(k) for k in
                                                   ("model_tensor_placements", "cpu_numerical_fallback",
                                                    "compiled_wrapper_fallbacks", "compiled_wrapper_roles_covered",
                                                    "decode_warmup",
                                                    "vision_calls", "output_retrieval",
                                                    "output_retrieval_seconds", "executed_generation_steps",
                                                    "eos_overrun_steps", "eos_stop_observed")},
                                  steps=[{k: s.get(k) for k in ("kind", "input_tokens", "seconds", "new_graphs", "code_store",
                                                                 "reused_cached_tokens", "transfers_inside_step",
                                                                 "logits_download_bytes")} for s in report["steps"]])
                if a.device == 'cpu' and scoring == 'author-20261101':
                    if not report.get('logits') or not turn['tokens']:
                        raise ValueError('CPU reference lacks first-token logits or token')
                    turn['first_token_reference'] = first_token_reference(report['logits'][0])
                    if turn['first_token_reference']['top1_token'] != turn['tokens'][0]:
                        raise ValueError('CPU reference was not greedy at the first token')
                turns.append(turn)
            row = dict(id=prompt["id"], kind=prompt["kind"], keywords=prompt["keywords"], turns=turns)
            if reference is not None:
                ref = {p["id"]: p for p in reference["prompts"]}[prompt["id"]]
                if scoring == 'author-20261101' and \
                        [r.get('user') for r in ref['turns']] != list(prompt['turns']):
                    raise ValueError(f"{prompt['id']}: CPU reference prompt differs")
                if a.max_new_tokens:                 # compare against the reference's first tokens only
                    ref = dict(ref, turns=[dict(r, tokens=r["tokens"][:new_tokens]) for r in ref["turns"]])
                scores = []
                for j, (t, r) in enumerate(zip(turns, ref['turns'], strict=True)):
                    keywords = prompt['keywords'] if j == len(turns) - 1 else []
                    score = (score_author_turn(t, r, checks[j], keywords) if scoring == 'author-20261101'
                             else score_turn(t, r, keywords))
                    if scoring == 'author-20261101':
                        score['prior_assistant_text_matches_cpu'] = prior_assistant_text_matches(turns, ref['turns'], j)
                    scores.append(score)
                verdicts = [s['passed'] for s in scores]
                passed = False if False in verdicts else (None if None in verdicts else True)
                row.update(scores=scores, passed=passed)
            else:
                row["reference_keywords_present"] = [k for k in prompt["keywords"]
                                                     if k.lower() in turns[-1]["text"].lower()]
            result["prompts"].append(row)
            publish_quality(a.out / 'quality.json', result)
        extra_verdicts = []
        if a.warm_repeat:
            if first_capture is None:
                raise ValueError('warm repeat requires a first quality prompt')
            prompt, images, tokens, logits = first_capture
            app.reset_conversation()
            started = time.perf_counter()
            warm = app.generate(prompt, images, keep_history=True, prompt_started=started)
            print(json.dumps(dict(prompt='warm-repeat', user=prompt, assistant=warm['generated_text'])), flush=True)
            from chat_timing import finalize_user_visible_metrics
            finalize_user_visible_metrics(warm['user_visible_timing'], time.perf_counter() - started,
                                          'same-session repeated prompt through flushed output')
            fallback = (assess_no_fallback(dict(device_evidence=warm, steps=warm.get('steps', [])),
                                           vision_expected=bool(warm.get('images'))) if a.device == 'tnca' else True)
            same_tokens = warm['generated_tokens'] == tokens
            same_logits = bitwise_logits_equal(logits, warm.get('logits', []))
            passed = combine_verdicts([warm.get('status') == 'pass', fallback, same_tokens, same_logits])
            result['warm_repeat'] = dict(passed=passed, no_fallback=fallback, tokens_equal=same_tokens,
                                         logits_bitwise_equal=same_logits, report=chat.public_report(warm),
                                         scope='same process and session; not a fresh-process cache-hit proof')
            extra_verdicts.append(passed)
        if script_lines is not None:
            app.reset_conversation()
            transcript = []
            def checkpoint(_turn):
                result['scripted_chat'] = dict(status='running', script_sha256=script_hash,
                                               transcript=list(transcript))
                publish_quality(a.out / 'quality.json', result)
            completed = app.repl(lines=script_lines, transcript=transcript, on_turn=checkpoint)
            script = assess_script(transcript, script_checks['checks'], a.device)
            script['passed'] = combine_verdicts([bool(completed), script['passed']])
            script.update(script_sha256=script_hash, checks_sha256=hashlib.sha256(a.script_checks.read_bytes()).hexdigest(),
                          transcript=transcript)
            result['scripted_chat'] = script
            extra_verdicts.append(script['passed'])
        if reference is not None:
            verdicts = [p['passed'] for p in result['prompts']] + extra_verdicts
            result['passed'] = combine_verdicts(verdicts)
            result['status'] = ('pass' if result['passed'] is True else
                                'unverified-chat-quality' if result['passed'] is None else 'failed-chat-quality')
        else:
            extra = combine_verdicts(extra_verdicts) if extra_verdicts else True
            result['status'] = ('reference-recorded' if extra is True else
                                'unverified-chat-quality' if extra is None else 'failed-chat-quality')
    except BaseException:
        result["status"] = "failed-preserved"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["versions"] = chat.L.versions()
        result["load"] = app.load_report
        publish_quality(a.out / 'quality.json', result)
    print(json.dumps({k: result[k] for k in ("status", "model", "device")} | {"passed": result.get("passed")}))
    return 0 if result["status"] in ("pass", "reference-recorded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
