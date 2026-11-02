#!/usr/bin/env python3
"""Summarize completed chat reports without accessing the accelerator.

Inputs are existing quality.json files, not live queues. Unknown transfer or
fallback counters remain unknown rather than being reported as zero.
"""
import argparse
import json
from pathlib import Path


def _all_known_true(values):
    values = list(values)
    if any(value is False for value in values):
        return False
    if not values or any(value is None for value in values):
        return None
    return True


def _decode_reads_are_zero(turn):
    steps = [step for step in turn.get('steps', []) if step.get('kind') == 'decode']
    if not steps:
        return None
    checks = []
    for step in steps:
        transfers = step.get('transfers_inside_step') or {}
        reads, staging = transfers.get('host_read_calls'), transfers.get('staging_download_bytes')
        checks.append(reads == 0 and staging == 0 if reads is not None and staging is not None else None)
    return _all_known_true(checks)


def _timing(report):
    timing = report.get('user_visible_timing') or {}
    names = ('prompt_to_return_seconds', 'non_special_token_ids',
             'non_special_token_ids_per_second', 'compile_and_decode_warmup_seconds',
             'prefill_seconds', 'decode_seconds', 'decode_step_seconds',
             'output_retrieval_seconds', 'scope')
    return {name: timing.get(name) for name in names}


def summarize(report, source):
    prompts = report.get('prompts') or []
    turns = [turn for prompt in prompts for turn in prompt.get('turns', [])]
    fact = next((prompt for prompt in prompts if prompt.get('id') == 'fact'), None)
    fact_turns = fact.get('turns', []) if fact else []
    first = fact_turns[0] if fact_turns else {}
    warm = report.get('warm_repeat') or {}
    scripted = report.get('scripted_chat') or {}
    fallbacks = [(turn.get('device_evidence') or {}).get('cpu_numerical_fallback')
                 for turn in turns]
    return dict(source=str(source), model=report.get('model'), device=report.get('device'),
                model_scope=report.get('model_scope'), status=report.get('status'),
                quality=dict(passed=sum(prompt.get('passed') is True for prompt in prompts),
                             total=len(prompts),
                             prompts=[dict(id=prompt.get('id'), passed=prompt.get('passed'))
                                      for prompt in prompts],
                             warm_repeat_passed=warm.get('passed'),
                             scripted_chat_passed=scripted.get('passed')),
                quality_no_cpu_numerical_fallback=_all_known_true(
                    None if value is None else value is False for value in fallbacks),
                quality_zero_decode_host_reads=_all_known_true(_decode_reads_are_zero(turn) for turn in turns),
                parameter_and_buffer_placements=(first.get('device_evidence') or {}).get('model_tensor_placements'),
                fact_cold=_timing(first), fact_warm=_timing(warm.get('report') or {}),
                fact_executed_generation_steps=(first.get('device_evidence') or {}).get('executed_generation_steps'),
                fact_post_eos_steps=(first.get('device_evidence') or {}).get('eos_overrun_steps'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('quality_json', type=Path, nargs='+')
    args = parser.parse_args()
    results = [summarize(json.loads(path.read_text()), path) for path in args.quality_json]
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
