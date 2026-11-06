"""Analytical state/activation estimates from an audited language inventory.

This describes the dense full-attention plus gated-delta architecture in the
supplied public configuration. It is not an allocation trace or a capacity
pass, and it performs no model-name dispatch in the execution path.
"""
import argparse
import hashlib
import json
from pathlib import Path


def estimate(inventory, *, batch, prefill, decode, window):
    cfg = inventory['text_configuration']
    if cfg['mamba_ssm_dtype'] != 'float32' or cfg['dtype'] != 'bfloat16':
        raise ValueError('this estimate requires BF16 activations and FP32 recurrent state')
    layers = cfg['layer_types']
    if set(layers) - {'linear_attention', 'full_attention'}:
        raise ValueError('unknown layer state semantics')
    linear, full = layers.count('linear_attention'), layers.count('full_attention')
    keys, values = cfg['linear_num_key_heads'], cfg['linear_num_value_heads']
    kd, vd = cfg['linear_key_head_dim'], cfg['linear_value_head_dim']
    slots = prefill+decode
    state = dict(
        recurrent_fp32=linear*batch*values*kd*vd*4,
        conv_bf16=linear*batch*(2*keys*kd+values*vd)*cfg['linear_conv_kernel_dim']*2,
        full_attention_kv_bf16=full*2*batch*cfg['num_key_value_heads']*slots*cfg['head_dim']*2)
    weights = inventory['selected_distinct_bytes']
    base = weights+sum(state.values())
    activations = dict(
        hidden_bf16=batch*prefill*cfg['hidden_size']*2,
        one_mlp_expansion_bf16=batch*prefill*cfg['intermediate_size']*2,
        one_attention_score_fp32=batch*cfg['num_attention_heads']*prefill*slots*4,
        full_prefill_logits_bf16=batch*prefill*cfg['vocab_size']*2,
        last_token_logits_fp32=batch*cfg['vocab_size']*4)
    largest = max(inventory['selected_tensors'].items(), key=lambda row: row[1]['bytes'])
    return dict(schema='tnca-language-capacity-estimate/1',
        repository=inventory['repository'], revision=inventory['revision'],
        geometry=dict(batch=batch, prefill=prefill, decode=decode, kv_slots=slots,
                      linear_layers=linear, full_attention_layers=full),
        language_distinct_weight_bytes=weights, state_bytes=state,
        weights_plus_state_bytes=base, exposed_window_bytes=window,
        remaining_before_runtime_and_activations_bytes=window-base,
        analytical_individual_buffer_bytes=activations,
        largest_storage=dict(name=largest[0], bytes=largest[1]['bytes']),
        extra_copy_of_largest_weight_would_fit_base=(base+largest[1]['bytes'] <= window),
        verdict='not-measured: state formula and individual buffers do not bound runtime peak',
        exclusions=['allocator alignment/fragmentation and reserved address ranges',
                    'code, argument tables, descriptors and repack scratch',
                    'simultaneous live activation/cast/transpose buffers',
                    'temporary or duplicate weight packing',
                    'chunked recurrence solve/mask intermediates and cache recording/rollback'],
        assumptions=['single resident copy of each checkpoint storage',
                     'tied embedding and output projection remain one storage',
                     'conv cache retains K values per channel as in pinned official cache',
                     'no offload, cache history recording or speculative rollback',
                     'KV cache reserves prefill+decode slots; BF16 K/V and FP32 recurrent state'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--prefill', type=int, required=True)
    ap.add_argument('--decode', type=int, required=True)
    ap.add_argument('--window-bytes', type=int, default=1 << 32)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.batch <= 0 or args.prefill <= 0 or args.decode < 0 or args.window_bytes <= 0:
        ap.error('invalid capacity geometry')
    report = estimate(json.loads(args.inventory.read_text()), batch=args.batch,
                      prefill=args.prefill, decode=args.decode, window=args.window_bytes)
    report['inventory_sha256'] = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('weights_plus_state_bytes',
        'remaining_before_runtime_and_activations_bytes', 'extra_copy_of_largest_weight_would_fit_base', 'verdict')}, indent=2))
