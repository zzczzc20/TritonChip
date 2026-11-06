#!/usr/bin/env python3
"""Prepare the same registered workloads for a corrected candidate xclbin.

This does not approve, program, or access the board. Run only after the actual
xclbin and its HBM map exist; it writes a fresh registration tree.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import uuid
import auto
import candidate_source
import manifests
import registrations
import stage

ROOT = manifests.ROOT
OUT = Path('spec/r41/run3')
IMAGE = '{home}/r41run3/tnca_u280.xclbin'
CLOCK = '{home}/r41run3/qualified_clock.json'
SOURCE = '{home}/r41run3/source_review.json'
MAP = str(OUT/'xclbin_hbm_map.json')


def prepare(image, image_sha, image_uuid, mapping, xclbinutil):
    image, mapping = Path(image), Path(mapping)
    if auto.sha(image) != image_sha:
        raise ValueError('candidate image hash mismatch')
    if str(uuid.UUID(image_uuid)) != image_uuid:
        raise ValueError('candidate UUID must use canonical lowercase form')
    info = subprocess.run([xclbinutil, '--info', '--input', str(image)],
                          capture_output=True, text=True, check=True).stdout
    actual_uuid = re.search(r'UUID\s*\(xclbin\):\s*(\S+)', info, re.I)
    if (not actual_uuid or actual_uuid.group(1).lower() != image_uuid or
            not re.search(r'Name:\s*DATA_CLK\s*\n\s*Index:.*\n\s*Type:.*\n\s*Frequency:\s*100 MHz', info)):
        raise ValueError('xclbin metadata does not show the candidate UUID and 100 MHz DATA_CLK')
    topology = json.loads(mapping.read_text())
    if topology.get('image_sha256') != image_sha:
        raise ValueError('HBM map belongs to another image')
    groups = topology['groups']
    if (len(groups) != 3 or len({x['index'] for x in groups}) != 1 or
            any(x['tag'] != 'MBG[0:31]' or x['size'] != 8 << 30 or x['base'] != 0 or not x['used'] for x in groups)):
        raise ValueError('candidate does not expose three 8 GiB HBM ports')
    banks = topology['allocation_8gib']['physical_banks']
    if len(banks) != 32 or any(x['pc'] != i or x['tag'] != f'HBM[{i}]' or not x['used']
                               or x['base'] != i * (1 << 28) or x['size'] != 1 << 28
                               or x['allocated_base'] != i * (1 << 28) or x['allocated_size'] != 1 << 28
                               for i,x in enumerate(banks)):
        raise ValueError('candidate HBM allocation does not cover PC0 through PC31')
    destination = ROOT/OUT
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'board').mkdir()
    (destination/'stages').mkdir()
    (ROOT/MAP).write_bytes(mapping.read_bytes())
    replacements = [(manifests.IMAGE, IMAGE), (manifests.CLOCK, CLOCK), (manifests.SOURCE, SOURCE),
                    (manifests.MAP, MAP), (manifests.MAP_SHA, stage.sha(mapping)),
                    (auto.IMAGE_SHA, image_sha), (auto.IMAGE_UUID, image_uuid),
                    (manifests.SOURCE_COMMIT, candidate_source.COMMIT)]
    paths = {}
    for original, old_name in manifests.jobs():
        name = 'r3-' + old_name.removeprefix('r2-')
        old_path = str(manifests.OUT/(old_name+'.json'))
        new_path = str(OUT/'board'/(name+'.json'))
        spec = manifests.replace(manifests.derive(original, old_name),
                                 [(old_path, new_path), *replacements])
        spec['id'] = name
        spec['board']['image_source']['commit'] = candidate_source.COMMIT
        spec['board']['load_policy'] = 'candidate load requires explicit author approval and 15 consecutive idle minutes'
        spec['registration']['image_source_commit'] = candidate_source.COMMIT
        if original.name == 's2-identity.json':
            step = next(s for s in spec['steps'] if s['id'] == 'source')
            step['argv'] = ['{python}', 'tools/r41/run2_board/candidate_source.py',
                            '--archive', '{home}/r41run3/src_8d237223.tgz',
                            '--root', '{home}/r41run3/src_8d237223', '--image', IMAGE,
                            '--image-sha256', image_sha, '--out', '{out}/source.json']
            spec['sources'].append('tools/r41/run2_board/candidate_source.py')
        target = ROOT/new_path
        target.write_text(json.dumps(spec, indent=1)+'\n')
        paths[old_name] = (name, new_path, stage.sha(target))
    for name in auto.REQUIRED:
        reg = registrations.group(name)
        reg['image_identity'] = dict(sha256=image_sha, uuid=image_uuid)
        for row in reg['experiments']:
            new_name, new_path, digest = paths[row['id']]
            row.update(id=new_name, manifest=new_path, sha256=digest)
        (destination/'stages'/(name+'.json')).write_text(json.dumps(reg, indent=1)+'\n')
    return len(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--image-sha256', required=True)
    p.add_argument('--uuid', required=True)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--xclbinutil', default='xclbinutil')
    a = p.parse_args()
    print(f'{prepare(a.image, a.image_sha256, a.uuid, a.map, a.xclbinutil)} candidate workloads registered; no board access')


if __name__ == '__main__':
    main()
