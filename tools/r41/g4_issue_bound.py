#!/usr/bin/env python3
"""Issue and SFU ceilings of the compiled memory-bound kernels (G4, analytic).

    $TORCH_PYTHON tools/r41/g4_issue_bound.py \
        --census evidence/r41/hbm_bw/link_budget/census/census.json \
        --kernels evidence/r41/hbm_bw/link_budget/census/census_kernels.tgz \
        --suite archbench/suites/r41_membound.json --out OUT.json

For every loop-free kernel of every registered point that G3's compile census
captured, this script counts the instructions an ideal general lowering can
NOT remove: the instructions whose operands depend, directly or through a
predicate, on data loaded from memory. Address arithmetic, index vectors,
constants and pointer loads are assumed free (hoisted out of a tiled loop),
and loads and stores are assumed free (moved by the tensor transfer engine).
What remains is a lower bound on the instructions the single-issue core must
execute per program of 16 elements (ISA: one 16 x 32-bit vector op per
issue slot, doc/architecture_overview.md).

From it, the per-core ceiling in B/cycle is
    demand bytes per program / max(k_data, n_sfu * ceil(16 / sfu_lanes))
where demand bytes per program are ArchBench demand (reads + writes) divided
by the number of programs (output elements / block elements). The SFU term
uses the SFU's issue occupancy of ceil(16 / slices) cycles per vector op
(rtl/core/tnca_sfu.v), ignoring its fixed pipeline overhead, so it is
optimistic. The result is an upper bound for any memory system. Every number
is computed from the census files; nothing is typed in. Domain: analytic on
compiled code.

Kernels with loops (reductions, normalisations) are not bounded this way;
they are reported with `bound: null` and the generic minimum stated in
G4_DESIGN.md (at least one vector op per 16 input elements).
"""
import argparse
import io
import json
import math
import re
import tarfile
from pathlib import Path

LINE = re.compile(r'^\s*(?:@(p\d+)\s+)?([a-z][a-z0-9_.]*)\s*(.*?)\s*(?://.*)?$')
REG = re.compile(r'\b([vsp]\d+)\b')
TARGET_PER_CORE = 320.0 / 3.0   # 32 GB/s / (3 cores x 100 MHz), BANDWIDTH_PLAN.md section 1


def parse(text):
    out = []
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().endswith(':') or raw.lstrip().startswith('//'):
            continue
        m = LINE.match(raw)
        if not m:
            continue
        pred, op, rest = m.groups()
        comment = raw.split('//', 1)[1] if '//' in raw else ''
        regs = REG.findall(rest.split('//')[0])
        out.append(dict(pred=pred, op=op, regs=regs, comment=comment, text=raw.strip()))
    return out


def is_store(op):
    return op.startswith('v.st') or op.startswith('s.st') or op.startswith('t.st')


def is_vload(op):
    return op.startswith('v.ld') or op.startswith('t.ld')


def data_instructions(insts):
    """Return (k_data, n_sfu, n_mem, data ops) for a loop-free kernel."""
    tainted = set()
    k = 0
    sfu = 0
    mem = 0
    ops = []
    for ins in insts:
        op, regs = ins['op'], ins['regs']
        if op.startswith('br') or op in ('exit', 'fence') or op.startswith('fence'):
            continue
        if is_store(op):
            mem += 1
            continue
        dest, srcs = (regs[0], regs[1:]) if regs else (None, [])
        if is_vload(op):
            mem += 1
            if dest:
                tainted.add(dest)
            continue
        if op.startswith('s.ld'):
            # argument pointers are not data; any other scalar load is
            if dest:
                (tainted.discard if 'device_ptr' in ins['comment'] or 'arg_base' in ins['comment']
                 else tainted.add)(dest)
            continue
        dep = any(r in tainted for r in srcs) or (ins['pred'] in tainted)
        if dest:
            (tainted.add if dep else tainted.discard)(dest)
        if dep:
            k += 1
            ops.append(op)
            if op.endswith('.approx'):
                sfu += 1
    return k, sfu, mem, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', required=True)
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--suite', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sfu-lanes', type=int, nargs='+', default=[1, 4, 16])
    a = ap.parse_args()

    import archbench.registry as reg   # needs torch (the membound tier builds inputs)
    census = json.loads(Path(a.census).read_text())['points']
    suite = json.loads(Path(a.suite).read_text())
    texts = {}
    with tarfile.open(a.kernels) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith('.s'):
                parts = Path(m.name).parts
                texts[(parts[-4], parts[-2])] = tf.extractfile(m).read().decode()

    rows = []
    for entry in suite['points']:
        pid = entry['id']
        p = reg.get(pid)
        key = pid.replace('/', '__')
        # the census compiled the small size; the kernel body per program is
        # size-independent (one 16-element block per program)
        small = re.sub(r'-(medium|large)-', '-small-', key)
        small = re.sub(r'-r\d+-c', '-rX-c', small)
        cands = [c for c in census if re.sub(r'-r\d+-c', '-rX-c', c) == small]
        row = dict(id=pid, family=p.family, dtype=p.dtype, census_point=cands[0] if cands else None,
                   demand_bytes=p.demand.read_bytes + p.demand.write_bytes)
        if not cands:
            row['bound'] = None
            row['why'] = 'no census kernel'
            rows.append(row)
            continue
        ks = census[cands[0]]['kernels']
        kern = []
        loops = False
        for name, info in ks.items():
            text = texts.get((cands[0], name))
            if info.get('loops'):
                loops = True
            if text is None:
                continue
            k, sfu, mem, ops = data_instructions(parse(text))
            kern.append(dict(kernel=name, block=info['block_elements'], k_data=k, n_sfu=sfu, n_mem=mem,
                             loops=info.get('loops', 0), data_ops=ops))
        row['kernels'] = kern
        if loops or len(kern) != 1 or kern[0]['block'] != [16]:
            row['bound'] = None
            row['why'] = 'loop or multi-kernel: generic bound only'
            rows.append(row)
            continue
        kd = kern[0]
        # programs = output elements / 16; output elements from the point params
        out_elems = _numel_out(p)
        per_prog = row['demand_bytes'] / (out_elems / 16.0)
        row['bytes_per_program'] = per_prog
        bound = {}
        for L in a.sfu_lanes:
            cyc = max(kd['k_data'], kd['n_sfu'] * math.ceil(16 / L))
            bound[f'sfu{L}'] = None if cyc == 0 else per_prog / cyc
        # the same kernel issuing its own loads and stores (line LSU, no TTE)
        cyc_lsu = kd['k_data'] + kd['n_mem']
        bound['sfu16_with_lsu_memory_ops'] = per_prog / max(cyc_lsu, kd['n_sfu'])
        row['bound'] = bound
        row['reaches_target'] = {k: (v is None or v >= TARGET_PER_CORE) for k, v in bound.items()}
        rows.append(row)
    Path(a.out).write_text(json.dumps(dict(
        schema='g4-issue-bound-v1', domain='analytic on compiled code',
        target_b_per_cycle_per_core=TARGET_PER_CORE, census=a.census, kernels=a.kernels,
        suite=a.suite, suite_sha256=suite.get('sha256'), rows=rows), indent=1))


def _numel_out(p):
    """Output elements of a point from its parameters (R x C for pointwise)."""
    par = p.params
    r, c = par.get('rows') or par.get('R'), par.get('cols') or par.get('C')
    if r and c:
        return r * c
    m = re.search(r'-r(\d+)-c(\d+)', p.id)
    return int(m.group(1)) * int(m.group(2))


if __name__ == '__main__':
    main()
