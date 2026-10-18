#!/usr/bin/env python3
"""Exercise native machine-word execution using actual assembler/compiler outputs."""
import argparse
import json
from pathlib import Path
import struct
import sys

from run import HERE, ROOT, run, sha
from validate import config

sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]
from tnca_asm import Assembler, build_image
from tnca_cc.target import TargetInfo
from tnca_cc.codegen import CodeGen
from tnca_cc.kernels import elementwise


def assemble(text, path):
    words = build_image(Assembler().assemble(text))
    path.write_text(''.join(f'{word:016x}\n' for word in words))


def validate(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = config(out/'chip.cfg', trace_events=10000)
    code = out/'grid.hex'
    assemble((ROOT/'tb/core/progs/r40_grid_mmio_memory.s').read_text(), code)
    expected = out/'grid.bin'
    expected.write_bytes(b''.join(struct.pack('<Q', 4096+i) for i in range(455)))
    script = out/'grid.commands'
    script.write_text(f'code {code}\nlaunch 1 0 0 13 7 5\nlaunch 2 0x100 0 13 7 5\nfence\nexpect 0x30000 {expected}\n')
    result = run(binary, cfg, script, out/'grid', mode='chip')
    assert result['status'] == 'completed-prediction', out/'grid/stderr.log'
    counters = result['summary']['counters']
    assert sum(counters.get(f'core_{i}_programs_completed', 0) for i in range(3)) == 910
    assert all(counters.get(f'core_{i}_programs_completed', 0) for i in range(3))
    results = dict(grid=result['summary'])
    # Actual TNCA IR -> product code generator -> canonical assembler -> model.
    # Both a masked tail and untouched sentinel lanes are checked independently.
    target = TargetInfo.load('tnca-sim-vrf-v5')
    for dtype in ('i32', 'f32'):
        kernel, _ = elementwise(target, dtype=dtype, n_elements=47,
            addresses={'x': 0x40000, 'y': 0x44000, 'out': 0x48000})
        assembly = CodeGen(kernel, target, check_protocol=False).run().text()+'\nexit\n'
        asmfile = out/f'{dtype}.s';asmfile.write_text(assembly)
        code = out/f'{dtype}.hex';assemble(assembly, code)
        x = [i*2-19 for i in range(48)];y = [i+3 for i in range(48)]
        fmt = '<i' if dtype == 'i32' else '<f'
        inputs=[]
        for name, values in [('x', x), ('y', y)]:
            path = out/f'{dtype}-{name}.bin';path.write_bytes(b''.join(struct.pack(fmt, v) for v in values));inputs.append(path)
        initial = out/f'{dtype}-initial.bin';initial.write_bytes(struct.pack('<I', 0xdeadbeef)*48)
        expected = out/f'{dtype}-expected.bin'
        expected.write_bytes(b''.join(struct.pack(fmt, a+b) for a, b in zip(x[:47], y[:47]))+struct.pack('<I', 0xdeadbeef))
        script = out/f'{dtype}.commands'
        script.write_text(f'code {code}\ndata 0x40000 {inputs[0]}\ndata 0x44000 {inputs[1]}\ndata 0x48000 {initial}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x48000 {expected}\n')
        result = run(binary, cfg, script, out/dtype, mode='chip')
        assert result['status'] == 'completed-prediction', out/dtype/'stderr.log'
        results[dtype] = result['summary']
    # An unsupported descriptor is a hard error, never a free operation.
    code = out/'unsupported.hex';assemble('td.release d0\nexit\n', code)
    script = out/'unsupported.commands';script.write_text(f'code {code}\nlaunch 3 0 0 1 1 1\n')
    result = run(binary, cfg, script, out/'unsupported', mode='chip')
    assert result['status'] == 'failed' and result['returncode'] != 0
    report = dict(schema='tnca-native-chip-validation/1', status='passed', results=results,
        uncovered=['multi-role contexts', 'DMA/descriptors/windows', 'MMA/accumulators',
                   'barriers', 'complete FP/CVT/SFU/shuffle/reduction semantics',
                   'RF bank/port contention', 'calibrated core timing', 'standard Torch graph adapter'],
        performance_gate='not-measured', whole_chip_qualified=False,
        compiler_sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'compiler/python/tnca_cc').glob('*.py')},
        isa_schema_sha256=sha(ROOT/'isa/tnca-isa-v0.1.json'))
    (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(validate(a.binary, a.out)['status'])
