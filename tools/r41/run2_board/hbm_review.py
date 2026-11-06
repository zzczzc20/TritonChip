#!/usr/bin/env python3
"""Check the original run-2 xclbin's previously extracted HBM mapping."""
import argparse
import json
from pathlib import Path
import auto

MAP_SHA = 'c1e7b1a4f8fb4f05107fc231dcb6fbbb54361c568053e95107d34c03ae65b9fc'
PC_SIZE = 1 << 28


def review(path):
    path = Path(path)
    if auto.sha(path) != MAP_SHA:
        raise ValueError('original xclbin mapping evidence changed')
    data = json.loads(path.read_text())
    if data.get('image_sha256') != auto.IMAGE_SHA:
        raise ValueError('mapping belongs to another image')
    groups = data['groups']
    if len(groups) != 3 or any(g['tag'] != 'MBG[0:31]' or g['size'] != 32 * PC_SIZE
                               or g['base'] != 0 or g['used'] != 1 for g in groups):
        raise ValueError('three kernel ports must connect to the 8 GiB HBM group')
    banks = data['allocation_8gib']['physical_banks']
    if len(banks) != 32 or any(b['pc'] != i or b['tag'] != f'HBM[{i}]' or not b['used']
                               or b['base'] != i * PC_SIZE or b['size'] != PC_SIZE
                               or b['allocated_base'] != i * PC_SIZE or b['allocated_size'] != PC_SIZE
                               for i, b in enumerate(banks)):
        raise ValueError('8 GiB allocation does not span PC0 through PC31')
    return dict(schema='tnca-run2-hbm-review/1', kind='hbm-map', status='pass',
                image_sha256=auto.IMAGE_SHA, source_sha256=MAP_SHA,
                group='MBG[0:31]', ports=3, allocated_bytes=32 * PC_SIZE,
                scope='static xclbin mapping; board memory and alias tests remain required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(review(a.map), indent=2) + '\n')


if __name__ == '__main__':
    main()
