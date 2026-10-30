#!/usr/bin/env python3
"""Bounded, teacher-forced Q41 pilot of the pinned full BF16 language model.

This records a pilot, not Stage 2 acceptance. It uses the official model and
independent FP32 references. The CPU mode validates the same execution and
comparison machinery without accessing an accelerator.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen35_reference as reference  # noqa: E402
import qwen_fp8_lib as runtime  # noqa: E402

IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
IMAGE_UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_card(root, sysfs=Path('/sys/bus/pci/devices')):
    sys.path.insert(0, str(root/'tools/r41/run2_board'))
    import auto
    board = auto.probe(sysfs)
    if not auto.idle(board) or board['uuid'] != IMAGE_UUID or auto.clock_mhz(sysfs) != 100:
        raise RuntimeError('card busy or candidate image/clock changed')
    return board


def board_preflight(root, env, guard, sysfs=Path('/sys/bus/pci/devices')):
    profile = 'core-vrf-sync-v5-x8-g2c4-dp'
    if env.get('TNCA_BACKEND') != 'fpga-u280-multicore' or env.get('TNCA_PROFILE') != profile:
        raise ValueError('pilot requires the registered candidate backend and profile')
    args = json.loads(env['TNCA_BACKEND_ARGS'])
    if (args.get('image_profile') != profile or args.get('core_count') != 3
            or args.get('memory_ports') != 3 or args.get('hbm_bytes') != 8*(1 << 30)
            or args.get('hbm_dma') is not True or args.get('device_clock') is not True):
        raise ValueError('pilot backend configuration differs')
    if sha(Path(args['xclbin'])) != IMAGE_SHA:
        raise ValueError('pilot candidate image hash differs')
    clock_path = Path(env['TNCA_OPERATING_CLOCK_EVIDENCE'])
    clock = json.loads(clock_path.read_text())
    if clock.get('image_sha256') != IMAGE_SHA or clock.get('clock_hz') != 100_000_000:
        raise ValueError('pilot clock evidence differs')
    lock_file = guard.enter_context((Path(args['xclbin']).parent/'candidate_queue.lock').open('a'))
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    board = check_card(root, sysfs)
    return dict(image_sha256=IMAGE_SHA, uuid=IMAGE_UUID, clock_hz=100_000_000,
                clock_evidence_sha256=sha(clock_path), preflight=board)


def load_model(snapshot, inventory, sources):
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    source_manifest = json.loads(sources.read_text())
    for name, row in source_manifest['files'].items():
        mod = importlib.import_module(name.removeprefix('src/').removesuffix('.py').replace('/', '.'))
        if sha(mod.__file__) != row['sha256']:
            raise ValueError('official model source differs: ' + name)
    inv = json.loads(inventory.read_text())
    if inv['revision'] != reference.REVISION or sha(snapshot/'config.json') != inv['source_sha256']['config']:
        raise ValueError('checkpoint revision or config differs')
    cfg = Qwen3_5TextConfig(**json.loads((snapshot/'config.json').read_text())['text_config'])
    cfg._attn_implementation = 'eager'
    if cfg.num_hidden_layers != 24 or cfg.vocab_size != 248320:
        raise ValueError('checkpoint architecture differs')
    with torch.device('meta'):
        model = modeling.Qwen3_5ForCausalLM(cfg).eval()
    state = {}
    for shard in sorted({v['shard'] for v in inv['selected_tensors'].values()}):
        if sha(snapshot/shard) != inv['shard_lfs'][shard]['sha256']:
            raise ValueError('checkpoint shard differs: ' + shard)
        with safe_open(snapshot/shard, framework='pt', device='cpu') as f:
            for key, entry in inv['selected_tensors'].items():
                if entry['shard'] == shard:
                    state[key.replace('model.language_model.', 'model.', 1)] = f.get_tensor(key)
    if len(state) != 320:
        raise ValueError('language storage inventory differs')
    state['lm_head.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, assign=True, strict=True)
    model.tie_weights()
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    if model.lm_head.weight is not model.model.embed_tokens.weight or any(b.is_meta for b in model.buffers()):
        raise ValueError('tied embeddings or model buffers differ')
    return model


class LanguageStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, cache):
        output = self.model(input_ids=input_ids, position_ids=position_ids,
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
        return output.logits


def compare(expected, logits, cache, limits, valid_length):
    actual = {'logits': logits.detach()}
    actual.update(reference.cache_tensors(cache))
    # StaticCache has explicit length tensors that DynamicCache did not
    # include in the independent reference. Check them rather than silently
    # discarding them; all reference tensors remain mandatory below.
    lengths = []
    for name in sorted(set(actual)-set(expected)):
        if not name.endswith('.cumulative_length') or name.removesuffix('.cumulative_length')+'.keys' not in expected:
            continue
        value = actual.pop(name).detach().cpu()
        if value.numel() != 1 or int(value.item()) != valid_length:
            raise ValueError('static cache length differs: ' + name + ' ' + str(value))
        lengths.append(dict(tensor=name, value=int(value.item()), passed=True))
    if set(actual) != set(expected):
        raise ValueError('state inventory differs: missing=' + str(sorted(set(expected)-set(actual))) +
                         ' extra=' + str(sorted(set(actual)-set(expected))))
    rows = []
    for name in sorted(expected):
        want = expected[name].detach().cpu()
        got = actual[name].detach().cpu()
        if name.endswith(('.keys', '.values')) and got.ndim == want.ndim and got.ndim >= 3:
            got = got[:, :, :want.shape[2]]
        if got.shape != want.shape:
            raise ValueError('state shape differs: ' + name + ' ' + str(got.shape) + ' ' + str(want.shape))
        if 'recurrent_states' in name and got.dtype != torch.float32:
            raise ValueError('recurrent state is not FP32: ' + name)
        x, y = want.double(), got.double()
        finite = bool(torch.isfinite(y).all())
        norm = float(torch.linalg.vector_norm(x))
        err = float(torch.linalg.vector_norm(y-x))
        rel = err / norm if norm else None
        max_abs = float((y-x).abs().max()) if y.numel() else 0.0
        limit = limits['logits_rel_l2'] if name == 'logits' else limits['state_rel_l2']
        ok = finite and (rel <= limit if norm else max_abs <= limits['zero_reference_max_abs'])
        rows.append(dict(tensor=name, shape=list(got.shape), dtype=str(got.dtype),
                         rel_l2=rel, max_abs=max_abs, finite=finite, passed=ok))
    return rows, lengths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'reference', 'root', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'tnca'), required=True)
    p.add_argument('--point', choices=('Q41-core', 'Q41-tail'), default='Q41-core')
    p.add_argument('--steps', type=int, default=1, help='prefill plus at most the registered decode steps')
    a = p.parse_args()
    root, snap, ref = a.root.resolve(), a.snapshot.resolve(), a.reference.resolve()
    points = json.loads((root/'spec/r41/stage2/model_points.json').read_text())['points']
    point = next(x for x in points if x['id'] == a.point)
    if not 1 <= a.steps <= point['decode'] + 1:
        raise ValueError('steps exceed the frozen point')
    lock = json.loads((root/'spec/r41/stage2/model_lock.json').read_text())
    cross = json.loads((root/'evidence/r41/acceptance/p0/cross-validation.json').read_text())
    if sha(ref/'report.json') != cross['reference_report_sha256']['fp32']:
        raise ValueError('independent FP32 reference report differs')
    report = json.loads((ref/'report.json').read_text())
    workload = report['workloads'][a.point]
    if workload['input_ids'] != point['input_ids'] or workload['prefill'] != point['prefill'] or workload['decode'] != point['decode']:
        raise ValueError('reference workload differs from frozen point')
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = dict(schema='r41-qwen35-bf16-pilot/1', status='running', device=a.device,
                  point=a.point, steps_requested=a.steps, total_registered_steps=point['decode']+1,
                  checkpoint_revision=reference.REVISION, reference_report_sha256=sha(ref/'report.json'),
                  script_sha256=sha(Path(__file__)), model_lock_sha256=sha(root/'spec/r41/stage2/model_lock.json'),
                  model_points_sha256=sha(root/'spec/r41/stage2/model_points.json'),
                  torch_version=str(torch.__version__), pilot_only=True, steps=[])
    output = a.out/'report.json'
    def save():
        output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    guard = contextlib.ExitStack()
    try:
        sys.path.insert(0, str(root))
        from archbench.provenance import software_tree
        result['software_tree'] = software_tree(root, refresh=True)
        if a.device == 'tnca':
            result['board'] = board_preflight(root, os.environ, guard)
            save()
        model = load_model(snap, root/'evidence/r41/stage1/kickoff/qwen-storage/inventory.json',
                           root/'evidence/r41/stage1/kickoff/transformers-reference/sources.json')
        if a.device == 'tnca':
            result['board']['before_device_open'] = check_card(root)
            save()
        dev = runtime.use_tnca(root) if a.device == 'tnca' else torch.device('cpu')
        stage = LanguageStage(model)
        if a.device == 'tnca':
            runtime.prepare_for_compile(stage)
            runtime.to_device_except(stage, dev)
            stage = torch.compile(stage, fullgraph=True, dynamic=False)
        from transformers import StaticCache
        cache = StaticCache(config=model.config, max_cache_len=point['prefill']+point['decode'])
        cfg = model.config
        cache.early_initialization(batch_size=1, num_heads=cfg.num_key_value_heads,
                                   head_dim=cfg.head_dim, dtype=torch.bfloat16, device=dev)
        from safetensors.torch import load_file
        with torch.inference_mode():
            for step in range(a.steps):
                row = workload['steps'][step]
                if sha(ref/row['file']) != row['sha256']:
                    raise ValueError('independent reference tensor file differs: ' + row['file'])
                seq = point['input_ids'][:point['prefill']] if step == 0 else [point['input_ids'][point['prefill']+step-1]]
                pos = 0 if step == 0 else point['prefill']+step-1
                ids = torch.tensor([seq], dtype=torch.int64).to(dev)
                positions = torch.arange(pos, pos+len(seq), dtype=torch.int64)[None, :].to(dev)
                t0 = time.monotonic()
                logits = stage(ids, positions, cache)
                checks, lengths = compare(load_file(str(ref/row['file']), device='cpu'), logits, cache,
                                          lock['numerics'], pos+len(seq))
                passed = all(r['passed'] for r in checks)
                result['steps'].append(dict(step=step, wall_seconds=time.monotonic()-t0, passed=passed,
                                            tensors=checks, cache_lengths=lengths))
                save()
                if not passed:
                    raise ValueError('numerical comparison failed at step ' + str(step))
        result['status'] = 'pass'
    except Exception as exc:
        result['status'] = 'fail'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        save()
        guard.close()
    print(json.dumps(dict(status=result['status'], steps=len(result['steps']), pilot_only=True)))


if __name__ == '__main__':
    main()
