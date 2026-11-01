#!/usr/bin/env python3
"""Compare profiled reduced-depth model runs without touching an FPGA board."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]


def load_run(path):
    report = json.loads((path / 'report.json').read_text())
    profile = json.loads((path / 'profile/launches.json').read_text())['launches']
    return report, [row for row in profile
                    if row.get('scope_kind') in ('prefill', 'decode') and not row.get('skipped')]


def summarize(report, rows):
    result = []
    for kind in ('prefill', 'decode'):
        launches = [row for row in rows if row['scope_kind'] == kind]
        if not launches or any(row.get('device_cycles') is None for row in launches):
            raise ValueError(f'{kind}: missing launches or device cycles')
        macs = instructions = 0
        for row in launches:
            if row['kernel'] != 'w8a8_block_dynamic_fp8_matmul_kernel':
                continue
            activation = next(x for x in row['args'] if x.get('role') == 'act' and x.get('dtype') == 'bfloat16')
            weight = next(x for x in row['args'] if x.get('role') == 'weight' and x.get('dtype') == 'float8_e4m3fn')
            macs += math.prod(activation['shape'][:-1]) * math.prod(weight['shape'])
            instructions += sum(value for name, value in row['counters'].items()
                                if name.startswith('core_') and name.endswith('_instructions'))
        steps = [(index, step) for index, step in enumerate(report['steps']) if step['kind'] == kind]
        if not steps:
            raise ValueError(f'{kind}: missing step report')
        result.append(dict(kind=kind, launches=len(launches),
                           device_cycles=sum(row['device_cycles'] for row in launches),
                           code_loads=sum(row['code_load_count'] for row in launches),
                           code_bytes=sum(row['code_upload_bytes'] for row in launches),
                           evictions=sum(row['evicted'] for row in launches),
                           host_read_calls=sum(step['transfers_inside_step']['host_read_calls'] for _, step in steps),
                           argument_uploads_by_step=[dict(step=step.get('step', index),
                               counters=step.get('argument_uploads')) for index, step in steps], fp8_macs=macs,
                           fp8_instructions=instructions,
                           fp8_instructions_per_mac=instructions / macs if macs else None))
    return result


def compare(control, candidate):
    import torch
    from safetensors.torch import load_file
    from tnca import artifact

    before, rows_before = load_run(control)
    after, rows_after = load_run(candidate)
    problems = []
    identity = ('model', 'repo', 'revision', 'model_scope', 'reduced_depth_layers', 'prompt_text',
                'prompt_tokens', 'bucket', 'prefill_chunk', 'requested_new_tokens', 'generated_tokens')
    for key in identity:
        if before.get(key) != after.get(key):
            problems.append(f'{key} differs')
    for name, report in (('control', before), ('candidate', after)):
        target = report.get('device_target') or {}
        if target.get('TNCA_BACKEND') != 'systemc-tlm-grid' or target.get('TNCA_PROFILE') != 'core-vrf-sync-v5-x8-g2c4-dp':
            problems.append(f'{name}: wrong TLM backend or profile')
        if report.get('status') != 'pass' or report.get('cpu_numerical_fallback') is not False:
            problems.append(f'{name}: status or no-host-fallback proof failed')
        if report.get('compiled_wrapper_roles_covered') is not True or report.get('steady_decode_host_reads_detected') is not False:
            problems.append(f'{name}: wrapper coverage or steady decode host-read proof failed')
        decode_steps = [step for step in report.get('steps', []) if step.get('kind') == 'decode']
        if not decode_steps or any((step.get('transfers_inside_step') or {}).get('host_read_calls') != 0 or
                                   (step.get('transfers_inside_step') or {}).get('staging_download_bytes') != 0
                                   for step in decode_steps):
            problems.append(f'{name}: decode transfer counters are missing or nonzero')
        placements = report.get('model_tensor_placements') or {}
        for kind in ('parameters', 'buffers'):
            if not placements.get(kind) or any(not device.startswith('tnca') for device in placements[kind]):
                problems.append(f'{name}: {kind} are not exclusively on TNCA')

    metadata_mismatches, code_mismatches = [], []
    if len(rows_before) != len(rows_after):
        problems.append('measured launch counts differ')
    text_hashes = {}
    def text_hash(path):
        if path not in text_hashes:
            text_hashes[path] = hashlib.sha256(artifact.load(path)[1]['text']).hexdigest()
        return text_hashes[path]
    for index, (old, new) in enumerate(zip(rows_before, rows_after)):
        if any(old.get(key) != new.get(key) for key in ('scope_kind', 'kernel', 'grid')):
            metadata_mismatches.append(index)
        if text_hash(old['artifact']) != text_hash(new['artifact']):
            code_mismatches.append(index)
    if metadata_mismatches or code_mismatches:
        problems.append('launch structure or device code differs')

    old_logits, new_logits = load_file(str(control / 'logits.safetensors')), load_file(str(candidate / 'logits.safetensors'))
    logits = {}
    if set(old_logits) != set(new_logits):
        problems.append('logit step sets differ')
    if not old_logits or len(old_logits) != len(before.get('generated_tokens') or ()) or \
            len(new_logits) != len(after.get('generated_tokens') or ()):
        problems.append('one or more returned tokens lack logit evidence')
    for name in sorted(set(old_logits) & set(new_logits)):
        old, new = old_logits[name].contiguous(), new_logits[name].contiguous()
        if old.shape != new.shape or old.dtype != new.dtype:
            problems.append(f'{name}: logit shape or dtype differs')
            continue
        width = old.element_size()
        mismatches = int((old.view(torch.uint8).reshape(-1, width) !=
                          new.view(torch.uint8).reshape(-1, width)).any(dim=1).sum())
        logits[name] = dict(elements=old.numel(), bit_mismatches=mismatches)
        if mismatches:
            problems.append(f'{name}: logits differ bitwise')
    control_steps, candidate_steps = summarize(before, rows_before), summarize(after, rows_after)
    if any(not row['fp8_macs'] for row in control_steps + candidate_steps):
        problems.append('a measured phase did not exercise the FP8 matmul path')
    return dict(status='fail' if problems else 'pass', problems=problems,
                scope='reduced-depth DP SystemC/TLM, not board or full-model quality',
                model=before.get('model'), control=control_steps,
                candidate=candidate_steps, metadata_mismatches=metadata_mismatches,
                code_mismatches=code_mismatches, logits=logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.control, args.candidate)
    except Exception as error:
        result = dict(status='fail', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result['status'] != 'pass'


if __name__ == '__main__':
    sys.exit(main())
