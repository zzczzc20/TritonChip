#!/usr/bin/env python3
"""Core RTL checks for immediate barrier indices at slice boundaries."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(cmd):
    print('COMMAND', ' '.join(map(str, cmd)), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'command exit {result.returncode}')
    return result.stdout


def main():
    failed = []
    with tempfile.TemporaryDirectory(prefix='tnca-barrier-index-') as temp:
        temp = Path(temp)
        for label, bars, width, limit in [('delivered', 32, 5, 8), ('g2c4', 8, 3, 2)]:
            exe = temp / (label + '.vvp')
            run(['iverilog', '-g2005', '-I', 'rtl/common', '-y', 'rtl/core',
                 '-y', 'rtl/common', '-y', 'rtl/gen', '-y', 'tb/common',
                 '-Ptb_tnca_core.NUM_RCTX=4', '-Ptb_tnca_core.RCTX_W=2',
                 '-Ptb_tnca_core.PHYS_VREGS=256', '-Ptb_tnca_core.PHYS_W=8',
                 '-Ptb_tnca_core.RSLICE=1', '-Ptb_tnca_core.RPAIR=1',
                 '-Ptb_tnca_core.MAC_DP=1',
                 f'-Ptb_tnca_core.BAR_N={bars}', f'-Ptb_tnca_core.BAR_W={width}',
                 '-o', str(exe), 'tb/core/tb_tnca_core.v'])
            for op in ('init', 'arm', 'arrive', 'inval'):
                for invalid in (False, True):
                    index = limit if invalid else limit - 1
                    pre = 'mb.arm s1, b0, #1, #0\n' if op == 'arrive' else ''
                    instr = {'init': f'mb.init b{index}',
                             'arm': f'mb.arm s2, b{index}, #0, #0',
                             'arrive': f'mb.arrive b{index}',
                             'inval': f'mb.inval b{index}'}[op]
                    if op == 'arrive' and not invalid:
                        pre = f'mb.arm s1, b{index}, #1, #0\n'
                    src = temp / 'test.s'; img = temp / 'test.hex'
                    src.write_text('.org 0\n' + pre + instr + '\nexit\n')
                    run(['python3', 'tools/asm/tnca_asm.py', str(src), '-o', str(img), '--size', '16384'])
                    cmd = ['vvp', str(exe), f'+hex={img}', '+maxcycles=10000', '+slice', '+slicek=2']
                    if invalid:
                        cmd.append('+expectfault=01')
                    output = run(cmd)
                    ok = ('PASS' in output and 'FAIL' not in output)
                    name = f'{label} {op} index={index} invalid={invalid}'
                    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
                    if not ok:
                        failed.append(name)
            # The optional source-free completion has its own token in the
            # scalar immediately following the ordinary completion token.
            for ident in (limit, bars, 65535):
                src = temp / 'test.s'; img = temp / 'test.hex'
                src.write_text(f'.org 0\n.li s1, #0\n.li s2, #{ident}\n'
                               'm.mma.async.sf a0, v0, v8, s1\nexit\n')
                run(['python3', 'tools/asm/tnca_asm.py', str(src), '-o', str(img), '--size', '16384'])
                output = run(['vvp', str(exe), f'+hex={img}', '+maxcycles=10000',
                              '+slice', '+slicek=2', '+expectfault=01'])
                name = f'{label} source-free token={ident}'
                ok = 'PASS' in output and 'FAIL' not in output
                print(('PASS ' if ok else 'FAIL ') + name, flush=True)
                if not ok:
                    failed.append(name)
            # A returned token contains the physical slice base. Exercise all
            # contexts so valid tokens from nonzero slices remain usable.
            for operation in ('mb.wait s1', 'mb.test s2, s1'):
                src = temp / 'test.s'; img = temp / 'test.hex'
                src.write_text('.org 0\nmb.arm s1, b0, #0, #0\n' + operation + '\nexit\n')
                run(['python3', 'tools/asm/tnca_asm.py', str(src), '-o', str(img), '--size', '16384'])
                output = run(['vvp', str(exe), f'+hex={img}', '+maxcycles=10000',
                              '+slice', '+slicek=2', '+allroles'])
                name = f'{label} returned token {operation}'
                ok = 'PASS' in output and 'FAIL' not in output
                print(('PASS ' if ok else 'FAIL ') + name, flush=True)
                if not ok:
                    failed.append(name)
            # These IDs are outside context zero's slice. Verify that token
            # consumers fault before dispatching a barrier/engine command.
            operations = ('mb.wait s1', 'mb.test s2, s1',
                          'm.mma.async a0, v0, v8, s1',
                          'm.mma.dp.async.k16.f16 a0, v0, v8, s1',
                          't.ld.async.h w0, d0, s10, s1',
                          't.st.async.h w0, d0, s10, s1')
            for operation in operations:
                for ident in (limit, bars, 65535):
                    src = temp / 'test.s'; img = temp / 'test.hex'
                    src.write_text(f'.org 0\n.li s1, #{ident}\n{operation}\nexit\n')
                    run(['python3', 'tools/asm/tnca_asm.py', str(src), '-o', str(img), '--size', '16384'])
                    output = run(['vvp', str(exe), f'+hex={img}', '+maxcycles=10000',
                                  '+slice', '+slicek=2', '+expectfault=01'])
                    name = f'{label} token={ident} {operation}'
                    ok = 'PASS' in output and 'FAIL' not in output
                    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
                    if not ok:
                        failed.append(name)
    print(f'BARRIER INDEX failures={len(failed)}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
