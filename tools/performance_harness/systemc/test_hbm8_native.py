#!/usr/bin/env python3
"""One native artifact on a mapped 8 GiB window: PC0-PC31 boundaries and refusals.

The supplied BO/PC map (for R41 the PLANNED 8 GiB map: not a recorded
allocation) translates every DRAM line on the fast and the slow path. Launch 1
loads lines in PC0, on both sides of the 32-bit logical boundary, beside the
TMU page, in every PC16-PC31 and at the BO's last line (the last 8 bytes with
a scalar load), and stores them in a region crossing the PC15/PC16 boundary
(logical local+4 GiB). After maintenance, launch 2 loads that region back and
stores it near the BO top. Every output byte is checked against the input files
and every PC byte counter against the translation of dram_address_map.py.
Each negative case must refuse with its own message; nothing is substituted.
"""
import argparse
import json
import os
from pathlib import Path
import struct
from unittest.mock import patch

from dram_address_map import translate
from run import run, sha
from validate_chip import assemble

HOLE = (0x7fff0000, 0x7fff1000)
PC_BYTES = 1 << 28


def program(loads, stores, scalar_load, scalar_store):
    text = []
    for source, destination in zip(loads, stores):
        text += [f'.li s1, #{source}', 'v.ld.w v0, [s1, #0]', f'.li s2, #{destination}', 'v.st.w [s2, #0], v0']
    text += [f'.li s1, #{scalar_load}', 's.ld.d s3, [s1, #0]', f'.li s2, #{scalar_store}', 's.st.d [s2, #0], s3', 'exit']
    return text


def words(path):
    return len(path.read_text().split())


def config_file(base, path, **updates):
    values = dict(line.split('#')[0].strip().split('=', 1) for line in base.read_text().splitlines()
                  if '=' in line.split('#')[0])
    values.update({k: str(v) for k, v in updates.items()})
    path.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--baseline-binary', type=Path, help='unmodified model: its refusal of the mapped 8 GiB window is recorded')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--memory-bytes', type=lambda v: int(v, 0), default=0x200000000)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit.read_text())
    mapping = audit['mapping']
    assert sha(args.map) == audit['map_sha256']
    local, size = mapping['local_limit'], mapping['bo_size']
    bo_top = local + size
    top = args.memory_bytes
    assert local < top <= bo_top
    assert mapping['bo_base'] == 0 and size == 1 << 33 and [b['pc'] for b in mapping['banks']] == list(range(32)), mapping
    boundary = local + (1 << 32)                      # physical 4 GiB: PC15 | PC16
    loads = [local, (1 << 32) - 64, 1 << 32, HOLE[0] - 64, HOLE[1]]
    loads += [local + pc * PC_BYTES + 4096 for pc in range(16, 32)]
    loads += [top - 128]
    scalar_load = top - 8                             # the model window's last 8 bytes
    n = len(loads)
    region1 = [boundary - 64 * (n // 2) + 64 * i for i in range(n)]
    scalar1 = region1[-1] + 64
    region2 = [top - 64 * (n + 8) + 64 * i for i in range(n)]
    scalar2 = top - 64 * 6
    lines = loads + [scalar_load & ~63] + region1 + [scalar1] + region2 + [scalar2]
    assert len(set(a & ~63 for a in lines)) == len(lines) and all(local <= a < top for a in lines)
    assert any(translate(mapping, a)['pc'] == 15 for a in region1) and any(translate(mapping, a)['pc'] == 16 for a in region1)

    first = out/'launch1.hex'
    assemble('\n'.join(program(loads, region1, scalar_load, scalar1)) + '\n', first)
    offset = words(first) * 8
    image = out/'boundaries.hex'
    assemble('\n'.join(program(loads, region1, scalar_load, scalar1) +
                       program(region1, region2, scalar1, scalar2)) + '\n', image)
    assert image.read_text().split()[:words(first)] == first.read_text().split(), 'launch 2 offset'
    commands = []
    for i, source in enumerate(loads):
        path = out/f'input-{i}.bin'
        path.write_bytes(struct.pack('<16I', *[(i + 1) * 0x01010101 + j for j in range(16)]))
        commands.append(f'data {source} {path}')
    scalar_path = out/'input-scalar.bin'
    scalar_path.write_bytes(struct.pack('<Q', 0x5ca1ab1e0ddba11))
    commands.append(f'data {scalar_load} {scalar_path}')
    expects = [f'expect {a} {out/f"input-{i}.bin"}' for region in (region1, region2) for i, a in enumerate(region)]
    expects += [f'expect {scalar1} {scalar_path}', f'expect {scalar2} {scalar_path}']
    script = out/'boundaries.commands'
    script.write_text('\n'.join([f'code {image}', *commands, 'launch 1 0 0 1 1 1', 'fence',
                                 f'launch 2 {offset} 0 1 1 1', 'fence', *expects]) + '\n')
    checked = 64 * 2 * n + 16

    # Independent expectation: each load line misses once (64 B read); each
    # stored line write-allocates and is written back once (128 B).
    expected = {}
    for address, nbytes in [(a, 64) for a in loads + [scalar_load & ~63] + region1 + [scalar1]] + \
                           [(a, 128) for a in region1 + [scalar1] + region2 + [scalar2]]:
        pc = translate(mapping, address & ~63)['pc']
        expected[f'pc_{pc}_bytes'] = expected.get(f'pc_{pc}_bytes', 0) + nbytes
    config = config_file(args.config, out/'model.cfg', memory_bytes=top, dram_max_passes=8)
    report = dict(status='running', map_sha256=audit['map_sha256'], map_status=audit['status'],
                  window=dict(local_limit=local, bo_size=size, bo_top=bo_top, model_memory_bytes=top),
                  logical_loads=loads, scalar_load=scalar_load, region1=region1, region2=region2,
                  expected_pc_bytes=expected, positives=[], negatives=[])
    save = lambda: (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    environment = dict(TNCA_DRAM_MAP=str(args.map.resolve()))
    for path in ('fast', 'slow'):
        with patch.dict(os.environ, TNCA_DRAM_PATH=path, **environment):
            result = run(args.binary, config, script, out/path, timeout=600, mode='chip')
        summary = result['summary'] or {}
        counters = summary.get('counters', {})
        got = {k: v for k, v in counters.items() if k.startswith('pc_') and k.endswith('_bytes')}
        row = dict(path=path, returncode=result['returncode'], checked_bytes=summary.get('checked_bytes'),
                   address_map_domain=summary.get('address_map_domain'), address_map_sha256=summary.get('address_map_sha256'),
                   model_timing_status=result['model_timing_status'], visible_cycles=summary.get('visible_cycles'),
                   pc_bytes=got, host_elapsed_seconds=result['host_elapsed_seconds'])
        row['status'] = ('pass' if result['returncode'] == 0 and summary.get('checked_bytes') == checked and got == expected
                         and summary.get('address_map_sha256') == audit['map_sha256'] else 'fail')
        report['positives'].append(row)
        save()
        assert row['status'] == 'pass', (row, (out/path/'stderr.log').read_text()[-2000:])

    def refused(name, text, message, *, binary=None, map_=True, path='fast', **updates):
        case = out/'negative'/f'{name}-{path}'
        case.mkdir(parents=True)
        image_path = case/'case.hex'
        assemble(text + '\nexit\n', image_path)
        script_path = case/'case.commands'
        lines_ = [f'code {image_path}']
        if 'data' in updates:
            address, payload = updates.pop('data')
            data_path = case/'data.bin'
            data_path.write_bytes(payload)
            lines_.append(f'data {address} {data_path}')
        lines_ += ['launch 1 0 0 1 1 1', 'fence']
        script_path.write_text('\n'.join(lines_) + '\n')
        cfg = config_file(args.config, case/'model.cfg', **dict(dict(memory_bytes=top), **updates))
        env = dict(TNCA_DRAM_PATH=path, **(environment if map_ else {}))
        with patch.dict(os.environ, env):
            if not map_:
                os.environ.pop('TNCA_DRAM_MAP', None)
            result = run(binary or args.binary, cfg, script_path, case/'run', timeout=120, mode='chip')
        stderr = (case/'run'/'stderr.log').read_text()
        row = dict(case=name, path=path, expected_message=message, returncode=result['returncode'],
                   refusal_lines=[line.strip() for line in stderr.splitlines() if message in line or 'MODEL_' in line][:3],
                   binary=str(binary or args.binary), mapped=map_, config_updates={k: str(v) for k, v in updates.items()})
        row['status'] = 'refused-as-expected' if result['returncode'] != 0 and message in stderr else 'fail'
        report['negatives'].append(row)
        save()
        assert row['status'] == 'refused-as-expected', row

    for path in ('fast', 'slow'):
        # A device-side refusal inside a slow-path child pass is reported by
        # the parent only as a failed child (the child's stderr is closed);
        # the identical artifact's specific reason is the fast-path row.
        device = (lambda m: m) if path == 'fast' else (lambda m: 'slow DRAM path: child pass failed')
        refused('device-load-at-window-top', f'.li s1, #{top}\nv.ld.w v0, [s1, #0]', device('address outside model memory window'), path=path)
        refused('device-byte-one-past-bo-top', f'.li s1, #{bo_top}\ns.ld.b s3, [s1, #0]', device('address outside model memory window'), path=path, memory_bytes=bo_top)
        refused('device-word-crossing-bo-top', f'.li s1, #{bo_top - 4}\ns.ld.d s3, [s1, #0]', device('address outside model memory window'), path=path, memory_bytes=bo_top)
        refused('host-byte-one-past-bo-top', '.li s1, #0', 'host access outside model memory window', path=path, data=(bo_top, b'\x01'), memory_bytes=bo_top)
        refused('device-load-tmu-page', f'.li s1, #{HOLE[0]}\nv.ld.w v0, [s1, #0]', device('TMU MMIO is not ordinary model memory'), path=path)
        refused('device-byte-tmu-page-last', f'.li s1, #{HOLE[1] - 1}\ns.ld.b s3, [s1, #0]', device('TMU MMIO is not ordinary model memory'), path=path)
        refused('host-write-tmu-page', '.li s1, #0', 'TMU MMIO is not ordinary model memory', path=path, data=(HOLE[1] - 8, bytes(8)))
        refused('device-load-sram-below-local-limit', f'.li s1, #{local - 64}\nv.ld.w v0, [s1, #0]',
                device('DRAM address map: SRAM or address outside AXI BO'), path=path)
        refused('window-one-line-past-bo', '.li s1, #0', 'model window exceeds the mapped AXI BO', path=path, memory_bytes=bo_top + 64)
    refused('unmapped-legacy-logical-window', '.li s1, #0', 'window exceeds PC map', map_=False, memory_bytes=bo_top)
    if args.baseline_binary:
        refused('baseline-binary-mapped-8gib-window', '.li s1, #0', 'window exceeds PC map', binary=args.baseline_binary.resolve(), memory_bytes=bo_top)
    report['status'] = 'pass'
    report['scope'] = ('native model execution on a mapped window; the map is ' + audit['status'] +
                       '; bank/row mapping and timing are not calibrated; no board evidence')
    save()
    print(json.dumps(dict(status=report['status'], positives=[(r['path'], r['status'], r['visible_cycles']) for r in report['positives']],
                          negatives=len(report['negatives'])), indent=2))


if __name__ == '__main__':
    main()
