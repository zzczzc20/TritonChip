#!/usr/bin/env python3
"""Derive run-2 board manifests from the registered run-1 workloads.

This only registers workloads. It neither qualifies a clock nor accesses a
board. Frozen image sources and the runtime software tree are distinct.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import auto
from stage import PROFILE

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('spec/r41/run2/board')
MAP = 'evidence/r41/hbm_bw/g4/run2/link_1f0a5aff/xclbin_map.json'
MAP_SHA = 'c1e7b1a4f8fb4f05107fc231dcb6fbbb54361c568053e95107d34c03ae65b9fc'
IMAGE = '{home}/r41run2-auto/run2.xclbin'
CLOCK = '{home}/r41run2-auto/qualified_clock.json'
SOURCE = '{home}/r41run2-auto/source_review_fullflow.json'
SOURCE_COMMIT = '1f0a5aff4b5d2df6ef9f3805513d37638bf5bc0d'
RUN1_IMAGE = '{home}/tnca_images/r41s2_run1_c9f25787494a/tnca_u280.xclbin'
RUN1_CLOCK = '{home}/tnca_build/r41s2_run1_c9f25787494a/r41s2-board/qualified_clock.json'
RUN1_MAP = 'evidence/r41/stage2/board/xclbin_hbm_map.json'
RUN1_SOURCE = 'evidence/r41/stage2/board/image_source.json'
RUN1_SHA = '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
RUN1_UUID = '49964b17-e0b4-5bd2-dd40-7fc37b1205d3'
RUN1_PROFILE = 'core-vrf-sync-v5-x8'

STAGE2 = ('s2-identity', 's2-hbm8', 's2-hbm8-big', 's2-original-suite',
          's2-fp8-cases', 's2-fp8-blockwise', 's2-fp8-hub', 's2-qwen-fp8', 's2-endurance')
CHAT = ('s2-qwen-chat-v3-vl', 's2-qwen-chat-v3-q35')


def jobs():
    result = [(Path('spec/r41/stage2/board') / (name + '.json'), 'r2-' + name) for name in STAGE2]
    for group in ('membound', 'membound_tiled2'):
        for p in sorted((ROOT / 'spec/r41/hbm_bw' / group).glob('membound-*.json')):
            suffix = '-tiled2' if group.endswith('tiled2') else '-default'
            result.append((p.relative_to(ROOT), 'r2-' + p.stem + suffix))
    result += [(Path('spec/r41/chat') / (name + '.json'), 'r2-' + name) for name in CHAT]
    return result


def backend(hbm_bytes=8 << 30):
    return dict(xclbin=IMAGE, image_profile=PROFILE, core_count=3, memory_ports=3,
                hbm_dma=True, hbm_bytes=hbm_bytes, device_clock=True)


def replace(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, replacements) for k, v in value.items()}
    return value


def derive(path, name):
    original = ROOT / path
    raw = original.read_bytes()
    old = json.loads(raw)
    target = str(OUT / (name + '.json'))
    replacements = [
        (str(path), target), (RUN1_IMAGE, IMAGE), (RUN1_CLOCK, CLOCK),
        (RUN1_MAP, MAP), (RUN1_SOURCE, SOURCE), (RUN1_SHA, auto.IMAGE_SHA),
        (RUN1_UUID, auto.IMAGE_UUID), (RUN1_PROFILE, PROFILE),
    ]
    spec = replace(copy.deepcopy(old), replacements)
    spec['id'] = name
    board = spec['board']
    board.update(image=IMAGE, image_sha256=auto.IMAGE_SHA, image_uuid=auto.IMAGE_UUID,
                 clock_evidence=CLOCK, clock_hz=100_000_000, profile=PROFILE,
                 backend_args=backend())
    board['memory_map'].update(window_base=0, window_bytes=8 << 30, physical_pcs='PC0-PC31',
                               source=MAP, source_sha256=MAP_SHA)
    board['image_source'] = dict(commit=SOURCE_COMMIT, branch='r41-hbm-bw', record=SOURCE,
                                 parameters='evidence/r41/hbm_bw/g4/run2/image.tcl',
                                 run='evidence/r41/hbm_bw/g4/run2/BRINGUP.md')
    board['load_policy'] = 'approved run-2 image after 15 consecutive idle minutes; every FPGA step checks identity'
    spec['hypothesis'] = ('Run-2 board execution of the workload registered in ' + str(path) +
                          ' passes its explicit numerical and structural checks on the approved 100 MHz image.')
    spec['competing_explanations'] = [
        'the run-2 image, its striped HBM layout, or its four-context DP profile changes behavior',
        'a workload fails, falls back, times out, or is omitted from its registered checks',
        'the card identity or clock changes during the run']
    spec['expected_observations'] = [
        'the registered workload completes with its numerical and structural checks passing on run-2',
        'the experiment records the observed image, clock, software tree and raw results']
    spec['registration'] = dict(status='registered-not-run', derived_from=str(path),
                                derived_from_sha256=hashlib.sha256(raw).hexdigest(),
                                generator='tools/r41/run2_board/manifests.py',
                                image_source_commit=SOURCE_COMMIT,
                                source_scope='frozen image source separately verified; runtime tree recorded by experiment.py',
                                inherited_expected_observations=old['expected_observations'],
                                inherited_registration=old.get('registration', {}))
    if path.name == 's2-identity.json':
        spec['sources'].append('tools/r41/run2_board/source_review.py')
    for step in spec['steps']:
        if step['id'] == 'source' and path.name == 's2-identity.json':
            step['argv'] = ['{python}', 'tools/r41/run2_board/source_review.py',
                            '--archive', '{home}/r41run2-auto/src_1f0a5aff.tgz',
                            '--root', '{home}/r41run2-auto/src_1f0a5aff', '--out', '{out}/source.json']
        if 'TNCA_BACKEND_ARGS' in step.get('env', {}):
            args = json.loads(step['env']['TNCA_BACKEND_ARGS'])
            size = 16 << 30 if step['id'] == 'clamp' else 8 << 30
            args.update(backend(size))
            step['env']['TNCA_BACKEND_ARGS'] = json.dumps(args, separators=(',', ':'))
        if 'board' in step and 'backend_args' in step['board']:
            size = 16 << 30 if step['id'] == 'clamp' else 8 << 30
            step['board']['backend_args'] = backend(size)
        if step.get('id') == 'summary' and 'tools/performance_harness/stage2_board.py' in step['argv']:
            step['argv'] += ['--image-label', 'run-2']
            if path.name in ('s2-fp8-cases.json', 's2-fp8-blockwise.json'):
                step['argv'].append('--require-dp')
                if '--require-x8' not in step['argv']:
                    step['argv'].append('--require-x8')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    for path, name in jobs():
        target = ROOT / OUT / (name + '.json')
        spec = derive(path, name)
        if a.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(spec, indent=1) + '\n')
        if a.check and (not target.exists() or json.loads(target.read_text()) != spec):
            raise SystemExit('registered manifest differs: ' + str(target))
    print(f'{len(jobs())} run-2 manifests ' + ('written' if a.write else 'checked' if a.check else 'derived in memory'))


if __name__ == '__main__':
    main()
