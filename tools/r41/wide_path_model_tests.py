#!/usr/bin/env python3
"""Explore coupled LSU/NoC/home widths with real native-chip numeric checks.

Predictions are uncalibrated; this does not execute an FPGA or qualify speedups.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import shlex
import hashlib
import struct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/performance_harness/systemc'))
import validate_chip


def exploratory_run(binary, config, trace, out, *, mode):
    """R41 standalone experiment; makes no frozen R40 qualification claim."""
    assert mode == 'chip'
    arity = {'code': 2, 'data': 3, 'expect': 3, 'launch': 7,
             'launch_roles': 9, 'launch_mask': 8, 'fence': 1}
    for line in trace.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and (fields[0] not in arity or len(fields) != arity[fields[0]]):
            raise ValueError('unsupported chip command or trailing fields')
    out.mkdir(parents=True, exist_ok=False)
    command = [str(binary), str(config), str(trace), str(out)]
    with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180)
    summary_path = out/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    (out/'exploratory_identity.json').write_text(json.dumps({
        'qualification': 'R41 exploratory only; not frozen R40 acceptance',
        'command': command, 'model_sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/performance_harness/systemc').rglob('*') if p.is_file() and p.suffix in ('.cpp', '.hpp')}, 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (binary, config, trace)}}, indent=2)+'\n')
    return {'status': 'completed-prediction' if result.returncode == 0 else 'failed',
            'returncode': result.returncode, 'summary': summary}


def streaming_add(binary, out, unroll=1):
    # Three dispatched programs, disjoint 16 KiB slices, real native integer
    # vector arithmetic. This is a model microbenchmark, not standard torch.
    assert unroll in (1, 2, 4, 8, 16, 32, 64)
    code = out/'stream-add.hex'
    body = []
    for j in range(unroll):
        body += [f'v.ld.w v{3*j}, [s1, #{64*j}]', f'v.ld.w v{3*j+1}, [s2, #{64*j}]']
    for j in range(unroll):
        body += [f'v.add v{3*j+2}, v{3*j}, v{3*j+1}', f'v.st.w [s3, #{64*j}], v{3*j+2}']
    body = '\n'.join(body)
    validate_chip.assemble(f"""
        s.rdspecial s8, pid.x
        .li s9, #16384
        s.mul s8, s8, s9
        .li s1, #0x40000
        .li s2, #0x80000
        .li s3, #0xc0000
        s.add s1, s1, s8
        s.add s2, s2, s8
        s.add s3, s3, s8
        .li s4, #{256//unroll}
    loop:
        {body}
        s.addi s1, s1, #{64*unroll}
        s.addi s2, s2, #{64*unroll}
        s.addi s3, s3, #{64*unroll}
        s.addi s4, s4, #-1
        br.nz s4, loop
        exit
    """, code)
    n = 3*4096
    x = out/'stream-x.bin'; y = out/'stream-y.bin'; expected = out/'stream-expected.bin'
    x.write_bytes(struct.pack('<'+'I'*n, *range(n)))
    y.write_bytes(struct.pack('<'+'I'*n, *(2*i+1 for i in range(n))))
    expected.write_bytes(struct.pack('<'+'I'*n, *(3*i+1 for i in range(n))))
    script = out/'stream-add.commands'
    script.write_text(f'code {code}\ndata 0x40000 {x}\ndata 0x80000 {y}\nlaunch 1 0 0 3 1 1\nfence\nexpect 0xc0000 {expected}\n')
    result = exploratory_run(binary, out/'chip.cfg', script, out/'stream-add', mode='chip')
    assert result['status'] == 'completed-prediction', out/'stream-add/stderr.log'
    return result['summary']


def unaligned_copy(binary, out):
    code = out/'unaligned.hex'
    validate_chip.assemble("""
        .li s1, #0x5003d
        .li s2, #0x6003f
        .li s3, #0x5555
        p.frombits p1, s3
        @p1 v.ld.w v0, [s1, #0]
        @p1 v.st.w [s2, #0], v0
        exit
    """, code)
    source = bytes((i*17+13)%256 for i in range(128))
    expected = bytearray([0xa5]*128)
    for lane in range(0,16,2):
        expected[63+lane*4:67+lane*4] = source[61+lane*4:65+lane*4]
    src = out/'unaligned-src.bin'; dst = out/'unaligned-dst.bin'; exp = out/'unaligned-expected.bin'
    src.write_bytes(source); dst.write_bytes(bytes([0xa5]*128)); exp.write_bytes(expected)
    script = out/'unaligned.commands'
    script.write_text(f'code {code}\ndata 0x50000 {src}\ndata 0x60000 {dst}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x60000 {exp}\n')
    result = exploratory_run(binary, out/'chip.cfg', script, out/'unaligned', mode='chip')
    assert result['status'] == 'completed-prediction', out/'unaligned/stderr.log'
    return result['summary']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    validate_chip.run = exploratory_run
    original = validate_chip.config
    rows = []
    variants = [('baseline', {}), ('explicit8', {'tl_core_beat_bytes': 8, 'tl_noc_payload_bytes': 8}),
                ('core64_noc8', {'tl_core_beat_bytes': 64, 'tl_noc_payload_bytes': 8}),
                ('core64_noc64', {'tl_core_beat_bytes': 64, 'tl_noc_payload_bytes': 64}),
                ('wide_home1', {'tl_core_beat_bytes': 64, 'tl_noc_payload_bytes': 64,
                                'tl_slice_hit_occupancy': 1, 'tl_slice_hit_latency': 2,
                                'tl_slice_wake_occupancy': 1})]
    for name, changes in variants:
        def configured(path, **kw):
            return original(path, memory_model=1, core_pipeline=1, **changes, **kw)
        validate_chip.config = configured
        result = validate_chip.validate(args.binary.resolve(), args.out/name)
        result['results']['unaligned'] = unaligned_copy(args.binary.resolve(), args.out/name)
        result['results']['stream-add'] = streaming_add(args.binary.resolve(), args.out/name)
        rows.append({'variant': name, 'hypotheses': changes, 'status': result['status'],
                     'results': result['results']})
        print(name, result['status'], flush=True)
    # Optional keys at their defaults must preserve ALL counters and outputs.
    # Ignore wall-clock metadata; compare deterministic device predictions.
    for key, base in rows[0]['results'].items():
        other = rows[1]['results'][key]
        for field in ('counters', 'visible_cycles', 'checked_bytes'):
            if field in base:
                assert base[field] == other[field], (key, field, 'default behavior changed')
    (args.out/'sweep.json').write_text(json.dumps({'evidence': 'model-prediction-uncalibrated',
        'whole_chip_qualified': False, 'rows': rows}, indent=2)+'\n')
    print('WIDE PATH MODEL NUMERICS AND DEFAULT EQUIVALENCE PASS')


if __name__ == '__main__':
    main()
