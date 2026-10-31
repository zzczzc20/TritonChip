#!/usr/bin/env python3
"""The transposed write's sequencing, RTL capture against the model (R41 X8).

    xpose_oracle.py --root PRODUCT_TREE --out FRESH_DIR [--latency 1 8 ...]

Offline only. iverilog runs tb_tnca_core (HAS_TENSOR_XPOSE=2) with
tb_xpose_capture.v as a second top over t30_xpose (16-bit) and t33_xpose8
(1-byte), and the capture is replayed through the model's core_pipeline=1
rule for a transposed command (include/tnca_tensor_engine.hpp, fast_transfer,
`transpose && rtl`):

    per column block: one request a cycle (t += 1 per issued unit), the
    response times are the RTL's own; w = max(t, last response) + 1; 8/esize
    window writes at w .. w+per-1; the next block starts at w+per.

Memory time is taken from the capture, so what is compared is the engine's
own sequencing: write cycles per block (must equal 8/esize), the first write
after the block's last response, and the gap to the next block. Writes the
per-block table and a verdict to FRESH_DIR/xpose_oracle.json.
"""
import argparse
import json
from pathlib import Path
import subprocess


def build(root, out, latency):
    vvp = out / f'tb_xpose8_l{latency}.vvp'
    subprocess.run(['iverilog', '-g2012', '-I', 'rtl/common', '-y', 'rtl/core', '-y', 'rtl/common',
                    '-y', 'rtl/gen', '-y', 'tb/common', '-P', 'tb_tnca_core.HAS_TENSOR_XPOSE=2',
                    '-P', f'tb_tnca_core.MEM_LATENCY={latency}', '-o', str(vvp),
                    'tb/core/tb_tnca_core.v',
                    'tools/performance_harness/systemc/rtl_oracle/tb_xpose_capture.v'],
                   cwd=root, check=True)
    return vvp


def commands(path):
    cmds, cur = [], []
    for line in path.read_text().splitlines():
        f = line.split()
        if f[0] != 'C':
            continue
        # Geometry is X in the setup cycle, before the snapshot is copied.
        cyc, esz, R, C, st, req, rsp, wr, done = (int(x) if x.isdigit() else -1 for x in f[1:])
        cur.append(dict(cycle=cyc, esz=esz, R=R, C=C, st=st, req=req, rsp=rsp, wr=wr))
        if done:
            if any(c['st'] in (14, 15, 16) for c in cur):
                cmds.append(cur)
            cur = []
    return cmds


def blocks(cmd):
    """Split one command into column blocks: XRUN entry to the last write."""
    # vw_en is registered: a write shows the cycle after its S_XWRITE cycle,
    # so the last write of a block is seen in the next block's first cycle.
    out, b, prev = [], None, None
    for c in cmd:
        if c['wr'] and b is not None and prev == 16:
            b['writes'].append(c['cycle'])
            c = dict(c, wr=0)
        if c['st'] == 14 and prev != 14:
            b = dict(start=c['cycle'], xrun=0, reqs=[], rsps=[], writes=[])
            out.append(b)
        prev = c['st']
        if b is None:
            continue
        b['xrun'] += c['st'] == 14
        if c['req'] and c['st'] == 14:
            b['reqs'].append(c['cycle'])
        if c['rsp']:
            b['rsps'].append(c['cycle'])
        if c['wr']:
            b['writes'].append(c['cycle'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--latency', type=int, nargs='+', default=[1, 8])
    a = p.parse_args()
    root = a.root.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    rows, ok = [], True
    for prog in ('t30_xpose', 't33_xpose8'):
        subprocess.run(['python3', 'tools/asm/tnca_asm.py', f'tb/core/progs/{prog}.s', '-o',
                        str(a.out / f'{prog}.hex'), '--size', '16384'], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
    for lat in a.latency:
        vvp = build(root, a.out, lat)
        for prog in ('t30_xpose', 't33_xpose8'):
            cap = a.out / f'{prog}_l{lat}.cap'
            r = subprocess.run(['vvp', '-n', str(vvp), f'+hex={a.out / (prog + ".hex")}',
                                '+maxcycles=200000', f'+xpose_capture={cap}'],
                               cwd=root, capture_output=True, text=True)
            assert 'PASS' in r.stdout, r.stdout[-2000:]
            for k, cmd in enumerate(commands(cap)):
                esz, R, C = cmd[-1]['esz'], cmd[-1]['R'], cmd[-1]['C']
                per = 8 >> esz
                bl = blocks(cmd)
                for j, b in enumerate(bl):
                    # The model's rule, fed the RTL's request and response
                    # cycles. A write is seen the cycle after its S_XWRITE
                    # cycle (vw_en is registered), so the RTL's write cycles
                    # are taken one earlier. The model skips an out-of-bounds
                    # unit in no time; the RTL walks it, one S_XRUN cycle.
                    drained = max(b['rsps'], default=b['start'])
                    w_model = max(b['start'] + len(b['reqs']), drained) + 1
                    # The RTL's own rule: S_XDRAIN sees the last tag clear the
                    # cycle after its response, and each unit takes one S_XRUN
                    # cycle whether or not it is fetched.
                    w_rtl = max(b['start'] + b['xrun'], drained + 1 if b['rsps'] else 0) + 1
                    first = b['writes'][0] - 1
                    nxt = bl[j + 1]['start'] if j + 1 < len(bl) else None
                    rows.append(dict(latency=lat, program=prog, command=k, esize=1 << esz, R=R, C=C,
                                     block=j, per=per, writes=len(b['writes']),
                                     requests=len(b['reqs']), oob_units=b['xrun'] - len(b['reqs']),
                                     responses=len(b['rsps']),
                                     first_write_rtl=first - b['start'],
                                     first_write_model=w_model - b['start'],
                                     error_model=first - w_model, error_rtl_rule=first - w_rtl,
                                     next_block_gap_rtl=(nxt - (b['writes'][-1] - 1)) if nxt else None,
                                     next_block_gap_model=1 if nxt else None))
                    ok &= len(b['writes']) == per
    # Expected: the RTL rule above exact for every block, the gap to the next
    # block the model's 1, and the model's own error (the drain cycle and the
    # walk of out-of-bounds units it leaves out) the same set for 16-bit and
    # 1-byte blocks.
    structural = all(r['error_rtl_rule'] == 0 for r in rows) and all(
        r['next_block_gap_rtl'] == r['next_block_gap_model'] for r in rows if r['next_block_gap_rtl'] is not None)
    verdict = dict(writes_per_block_match=ok, structure_match=structural,
                   error_model_by_esize={e: sorted({r['error_model'] for r in rows if r['esize'] == e})
                                         for e in (1, 2)},
                   blocks_by_esize={e: len([r for r in rows if r['esize'] == e]) for e in (1, 2)})
    verdict['pass'] = ok and structural
    verdict['model_error_note'] = ('RTL first write minus the model, cycles per column block: +1 when it '
                                   'waits on memory (S_XDRAIN), +1 per out-of-bounds unit walked after '
                                   'the last response; pre-existing 16-bit behaviour, unchanged')
    (a.out / 'xpose_oracle.json').write_text(json.dumps(dict(verdict=verdict, blocks=rows), indent=1))
    print(json.dumps(verdict))
    return 0 if verdict['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
