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
from tnca_cc.kernels import elementwise, reduction
from tnca_cc.ir import Builder, make_kernel


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
    for dtype in ('i32', 'f32', 'f16'):
        n, padded = (13, 16) if dtype == 'f16' else (47, 48)
        kernel, _ = elementwise(target, dtype=dtype, n_elements=n,
            addresses={'x': 0x40000, 'y': 0x44000, 'out': 0x48000})
        if dtype == 'f16':
            # The historical elementwise constructor uses 32-bit storage even
            # when its arithmetic dtype is f16. Build the real narrow-memory
            # conversion path explicitly rather than mislabeling its bytes.
            kernel = make_kernel('narrow_add')
            b = Builder(kernel, kernel.role(0).region)
            xptr, yptr, optr = (b.const(a) for a in (0x40000, 0x44000, 0x48000))
            mask = b.vcmp('lt', b.arange(), b.splat(b.const(n)))
            xval = b.vcvt('f16', 'f32', b.load(xptr, mask=mask, dtype='i32', size='h'))
            yval = b.vcvt('f16', 'f32', b.load(yptr, mask=mask, dtype='i32', size='h'))
            value = b.vcvt('f32', 'f16', b.vbin('add', xval, yval, dtype='f32'))
            b.store(optr, value, mask=mask, size='h')
        assembly = CodeGen(kernel, target, check_protocol=False).run().text()+'\nexit\n'
        asmfile = out/f'{dtype}.s';asmfile.write_text(assembly)
        code = out/f'{dtype}.hex';assemble(assembly, code)
        x = [i*2-19 for i in range(padded)];y = [i+3 for i in range(padded)]
        fmt = '<i' if dtype == 'i32' else '<e' if dtype == 'f16' else '<f'
        inputs=[]
        for name, values in [('x', x), ('y', y)]:
            path = out/f'{dtype}-{name}.bin';path.write_bytes(b''.join(struct.pack(fmt, v) for v in values));inputs.append(path)
        sentinel = struct.pack('<H', 0x3555) if dtype == 'f16' else struct.pack('<I', 0xdeadbeef)
        initial = out/f'{dtype}-initial.bin';initial.write_bytes(sentinel*padded)
        expected = out/f'{dtype}-expected.bin'
        expected.write_bytes(b''.join(struct.pack(fmt, a+b) for a, b in zip(x[:n], y[:n]))+sentinel*(padded-n))
        script = out/f'{dtype}.commands'
        script.write_text(f'code {code}\ndata 0x40000 {inputs[0]}\ndata 0x44000 {inputs[1]}\ndata 0x48000 {initial}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x48000 {expected}\n')
        result = run(binary, cfg, script, out/dtype, mode='chip')
        assert result['status'] == 'completed-prediction', out/dtype/'stderr.log'
        results[dtype] = result['summary']
    for dtype in ('i32', 'f32'):
        kernel, _ = reduction(target, dtype=dtype, n_elements=13,
            addresses={'x': 0x40000, 'out': 0x48000})
        assembly = CodeGen(kernel, target, check_protocol=False).run().text()+'\nexit\n'
        code = out/f'reduce-{dtype}.hex';assemble(assembly, code)
        fmt = '<i' if dtype == 'i32' else '<f'
        values = [i-5 for i in range(13)]
        data = out/f'reduce-{dtype}-input.bin';data.write_bytes(b''.join(struct.pack(fmt, x) for x in values))
        reference = out/f'reduce-{dtype}-expected.bin';reference.write_bytes(struct.pack(fmt, sum(values)))
        script = out/f'reduce-{dtype}.commands'
        script.write_text(f'code {code}\ndata 0x40000 {data}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x48000 {reference}\n')
        result = run(binary, cfg, script, out/f'reduce-{dtype}', mode='chip')
        assert result['status'] == 'completed-prediction', out/f'reduce-{dtype}/stderr.log'
        results[f'reduce-{dtype}'] = result['summary']
    code = out/'shuffle.hex'
    assemble('''
        .li s1, #0x40000
        v.ld.w v0, [s1, #0]
        v.iota v1
        v.xori v1, v1, #15
        .li s2, #0x5555
        p.frombits p1, s2
        @p1 v.permute v0, v0, v1
        .li s3, #123
        @p1 v.insert v0, s3, #1
        @p1 v.insert v0, s3, #0
        v.movi v2, #-9
        @p1 v.sel v0, v0, v2
        .li s2, #5
        v.shuffle.xor v0, v0, s2
        v.st.w [s1, #0], v0
        exit
    ''', code)
    data = out/'shuffle-input.bin';data.write_bytes(struct.pack('<16I', *range(16)))
    values = [15-i if i % 2 == 0 else i for i in range(16)]
    values[0] = 123
    values = [v if i % 2 == 0 else 0xfffffff7 for i, v in enumerate(values)]
    reference = out/'shuffle-expected.bin';reference.write_bytes(struct.pack('<16I', *(values[i ^ 5] for i in range(16))))
    script = out/'shuffle.commands'
    script.write_text(f'code {code}\ndata 0x40000 {data}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x40000 {reference}\n')
    result = run(binary, cfg, script, out/'shuffle', mode='chip')
    assert result['status'] == 'completed-prediction', out/'shuffle/stderr.log'
    results['shuffle'] = result['summary']
    data=out/'sfu-input.bin'
    raw=[0,0x80000000,0x3f800000,0x40800000,0x7f800000,0xff800000,0x80000001,0x00000001]*2
    data.write_bytes(struct.pack('<16I',*raw))
    code=out/'sfu.hex'
    assemble('''
        .li s1, #0x40000
        v.ld.w v0, [s1, #0]
        v.exp2.approx v1, v0
        v.rcp.approx v2, v0
        v.rsqrt.approx v3, v0
        v.log2.approx v4, v0
        .li s1, #0x40100
        v.st.w [s1, #0], v1
        v.st.w [s1, #64], v2
        v.st.w [s1, #128], v3
        v.st.w [s1, #192], v4
        exit
    ''',code)
    reference=out/'sfu-expected.bin'
    rows=[
        [0x3f800000,0x3f800000,0x40000000,0x41800000,0x7f800000,0,0x3f800000,0x3f800000],
        [0x7f800000,0xff800000,0x3f800000,0x3e800000,0,0x80000000,0xff800000,0x7f800000],
        [0x7f800000,0xff800000,0x3f800000,0x3f000000,0,0x7fc00000,0xff800000,0x7f800000],
        [0xff800000,0xff800000,0,0x40000000,0x7f800000,0x7fc00000,0xff800000,0xff800000]]
    reference.write_bytes(b''.join(struct.pack('<16I',*(row*2)) for row in rows))
    script=out/'sfu.commands';script.write_text(f'code {code}\ndata 0x40000 {data}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x40100 {reference}\n')
    result=run(binary,cfg,script,out/'sfu',mode='chip')
    assert result['status']=='completed-prediction',out/'sfu/stderr.log'
    assert result['summary']['counters']['core_0_sfu_queue_cycles']>0
    results['sfu']=result['summary']
    # An uncovered engine operation is a hard error, never a free operation.
    code = out/'unsupported.hex';assemble('m.zero a0\nexit\n', code)
    script = out/'unsupported.commands';script.write_text(f'code {code}\nlaunch 3 0 0 1 1 1\n')
    result = run(binary, cfg, script, out/'unsupported', mode='chip')
    assert result['status'] == 'failed' and result['returncode'] != 0
    code = out/'reserved.hex';code.write_text(f'{(16<<56)|(22<<48):016x}\n')
    script = out/'reserved.commands';script.write_text(f'code {code}\nlaunch 4 0 0 1 1 1\n')
    result = run(binary, cfg, script, out/'reserved', mode='chip')
    assert result['status']=='failed' and 'undefined ISA opcode' in (out/'reserved/stderr.log').read_text()
    script = out/'trailing.commands';script.write_text('launch 1 0 0 1 1 1 ignored\n')
    try:
        run(binary, cfg, script, out/'trailing', mode='chip')
    except ValueError as e:
        assert 'trailing fields' in str(e)
    else:
        raise AssertionError('trailing launch fields were silently accepted')
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
