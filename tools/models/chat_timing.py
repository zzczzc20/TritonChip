"""Accounting for wall time from a chat prompt to returned output."""


def compiler_snapshot(metrics, counters):
    """Record process-local Dynamo compilation spans and FX cache counters."""
    entries = metrics.get('_compile.compile_inner', ())
    inductor = counters.get('inductor', {})
    return {'entries': len(entries), 'hits': int(inductor.get('fxgraph_cache_hit', 0)),
            'misses': int(inductor.get('fxgraph_cache_miss', 0))}


def compiler_delta(before, metrics, counters):
    entries = metrics.get('_compile.compile_inner', ())
    after = compiler_snapshot(metrics, counters)
    if any(after[key] < before[key] for key in ('entries', 'hits', 'misses')):
        return {'available': False, 'reason': 'process-wide compilation counters were reset'}
    return {'available': True,
            'dynamo_compile_seconds': sum(entries[before['entries']:]),
            'dynamo_compile_calls': after['entries'] - before['entries'],
            'fx_graph_cache_hits': after['hits'] - before['hits'],
            'fx_graph_cache_misses': after['misses'] - before['misses'],
            'scope': 'process-wide Dynamo compilation spans; overlaps warmup and prefill'}


def finalize_user_visible_metrics(timing, elapsed_seconds, scope):
    """Replace the provisional return time with the output-flush observation."""
    if elapsed_seconds < timing['prompt_to_return_seconds']:
        raise ValueError('output cannot precede report creation')
    timing['prompt_to_return_seconds'] = elapsed_seconds
    timing['returned_token_ids_per_second'] = (timing['returned_token_ids'] / elapsed_seconds
                                                if elapsed_seconds else None)
    timing['non_special_token_ids_per_second'] = (timing['non_special_token_ids'] / elapsed_seconds
                                                   if elapsed_seconds else None)
    measured = sum(timing[key] for key in ('compile_and_decode_warmup_seconds', 'prefill_seconds',
                                           'decode_seconds', 'output_retrieval_seconds',
                                           'reference_check_seconds'))
    timing['other_wall_seconds'] = max(0.0, elapsed_seconds - measured)
    timing['accounted_span_exceeds_wall'] = measured > elapsed_seconds + 1e-6
    timing['scope'] = scope


def user_visible_metrics(report, elapsed_seconds, special_ids=(), reference_seconds=0.0):
    """Summarize observed spans without equating compilation, warmup and device time."""
    if elapsed_seconds < 0 or reference_seconds < 0:
        raise ValueError('wall durations must be nonnegative')
    tokens = list(report.get('generated_tokens') or [])
    special = set(special_ids)
    visible = sum(token not in special for token in tokens)
    stats = report.get('statistics') or {}
    warmup = (report.get('decode_warmup') or {}).get('seconds', 0.0) or 0.0
    prefill = stats.get('prefill_seconds', 0.0) or 0.0
    decode = stats.get('decode_seconds', 0.0) or 0.0
    retrieval = report.get('output_retrieval_seconds', 0.0) or 0.0
    decode_steps = [row.get('seconds') for row in report.get('steps', [])
                    if row.get('kind') == 'decode']
    measured = warmup + prefill + decode + retrieval + reference_seconds
    return {
        'prompt_to_return_seconds': elapsed_seconds,
        'returned_token_ids': len(tokens),
        'non_special_token_ids': visible,
        'returned_token_ids_per_second': len(tokens) / elapsed_seconds if elapsed_seconds else None,
        'non_special_token_ids_per_second': visible / elapsed_seconds if elapsed_seconds else None,
        'compile_and_decode_warmup_seconds': warmup,
        'prefill_seconds': prefill,
        'decode_seconds': decode,
        'decode_step_seconds': decode_steps,
        'output_retrieval_seconds': retrieval,
        'reference_check_seconds': reference_seconds,
        'other_wall_seconds': max(0.0, elapsed_seconds - measured),
        'accounted_span_exceeds_wall': measured > elapsed_seconds + 1e-6,
        'scope': 'prompt entry through returned report; returned IDs include special tokens',
        'warmup_scope': 'recorded decode warmup includes device work; prefill may include compilation',
    }
