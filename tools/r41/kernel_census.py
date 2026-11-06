#!/usr/bin/env python3
"""What the standard path compiles for each point: block sizes and instruction mix.

    python3 tools/r41/kernel_census.py BUNDLE_DIR [BUNDLE_DIR ...] --out FILE.json

Reads the TNCA assembly that ArchBench dumps into a point's provenance bundle
(`--bundle-dir`; kernels/<name>/<name>.s, the code the device executes) and
counts, per kernel, its static instructions by class, the vector loads and
stores and their element width, tensor-transfer-engine instructions, SFU
operations, loops (backward branches) and the per-program block size the
compiler states (`xN as a shift`). `_copy_kernel` (host-side constant clones
made while compiling) is listed separately. Nothing is executed; the counts
are the generated code's, not a run's.
"""
import argparse
import json
from pathlib import Path
import re

INSN = re.compile(r'^\s+(@p\d+\s+)?([a-z][a-z0-9_.]*)\s*(.*?)\s*(//.*)?$')
LABEL = re.compile(r'^([A-Za-z_.][\w.]*):')
BLOCK = re.compile(r'x(\d+) as a shift')
WIDTH = {'b': 1, 'h': 2, 'w': 4, 'd': 8}


def classify(op):
    if op.startswith('t.') or op.startswith('td.') or op.startswith('mb.'):
        return 'tensor'
    if op.startswith('v.ld') or op.startswith('v.gather'):
        return 'vector_load'
    if op.startswith('v.st') or op.startswith('v.scatter'):
        return 'vector_store'
    if op in ('v.exp2.approx', 'v.rcp.approx', 'v.rsqrt.approx', 'v.log2.approx', 'v.tanh.approx') \
            or op.endswith('.approx'):
        return 'sfu'
    if op.startswith('vf.') or op.startswith('v.'):
        return 'vector_alu'
    if op.startswith('s.ld') or op.startswith('s.st'):
        return 'scalar_memory'
    if op.startswith('br') or op in ('exit', 'jmp', 'j', 'call', 'ret'):
        return 'control'
    return 'scalar_alu'


def kernel(path):
    labels, rows, blocks = {}, [], []
    for line in Path(path).read_text().splitlines():
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(rows)
            continue
        b = BLOCK.search(line)
        if b:
            blocks.append(int(b.group(1)))
        m = INSN.match(line)
        if m:
            rows.append((m.group(2), m.group(3)))
    counts = {}
    # pseudo-instructions (.li) expand to several machine words: the exact
    # static size is the assembled image, one 64-bit word per line of .hex
    hexfile = Path(path).with_suffix('.hex')
    words = len([l for l in hexfile.read_text().splitlines() if l.strip()]) if hexfile.exists() else None
    pseudo = sum(1 for line in Path(path).read_text().splitlines() if line.strip().startswith('.li'))
    load_bytes = store_bytes = 0
    loops = 0
    for i, (op, args) in enumerate(rows):
        cls = classify(op)
        counts[cls] = counts.get(cls, 0) + 1
        if cls in ('vector_load', 'vector_store'):
            suffix = op.rsplit('.', 1)[-1]
            width = WIDTH.get(suffix[0], 0) * 16          # VLEN = 16 lanes
            if cls == 'vector_load':
                load_bytes += width
            else:
                store_bytes += width
        if op.startswith('br'):
            target = args.split(',')[-1].strip()
            if target in labels and labels[target] <= i:
                loops += 1
    return dict(static_instructions=len(rows), static_machine_words=words, pseudo_li=pseudo, classes=counts, loops=loops,
                block_elements=sorted(set(blocks)), vector_load_bytes_static=load_bytes,
                vector_store_bytes_static=store_bytes, uses_tensor_engine=counts.get('tensor', 0) > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bundles', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    points = {}
    for root in a.bundles:
        for asm in sorted(root.rglob('kernels/*/*.s')):
            name = asm.parent.name
            point = asm.parents[2].name
            if point in ('cold_call',):
                continue
            entry = points.setdefault(point, dict(kernels={}, constant_clone_kernels=0))
            if name == '_copy_kernel':
                entry['constant_clone_kernels'] += 1
                continue
            entry['kernels'][name] = kernel(asm)
    a.out.write_text(json.dumps(dict(schema='tnca-kernel-census/1', source='generated TNCA assembly in '
                                     'ArchBench provenance bundles', points=points), indent=1) + '\n')
    print(f'{len(points)} points -> {a.out}')


if __name__ == '__main__':
    main()
