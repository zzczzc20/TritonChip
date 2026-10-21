#!/usr/bin/env python3
"""Audit an actual AXI BO/PC map and export a native-model mapping input.

PC identities come from recorded physical HBM tags and allocation ranges.
This establishes no bank/row map, controller equivalence or calibrated timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def number(value):
    if type(value) is str:
        value=int(value,0)
    if type(value) is not int or not 0<=value<1<<64:
        raise ValueError('mapping values must be unsigned 64-bit integers')
    return value


def pointer(document, path):
    if path=='':
        return document
    if not path.startswith('/'):
        raise ValueError('JSON pointer must start with /')
    for key in path[1:].split('/'):
        key=key.replace('~1','/').replace('~0','~')
        document=document[int(key)] if isinstance(document,list) else document[key]
    return document


def normalize(window):
    base,size,local=(number(window[k]) for k in ('base','size','local_limit'))
    if not size or base+size>=1<<64 or local+size>=1<<64 or (base|size|local)&63:
        raise ValueError('unaligned or overflowing AXI window')
    allocation=window['allocation']
    if allocation.get('mapping_error') or number(allocation['exposed_bytes'])!=size:
        raise ValueError('allocation is not a complete physical mapping of this BO')
    topology=allocation['topology_sha256']
    if not re.fullmatch('[0-9a-f]{64}',topology):
        raise ValueError('physical topology identity missing')
    banks=[]
    for raw in allocation['physical_banks']:
        pc=number(raw['pc']);start,capacity,lo,length=(number(raw[k]) for k in
            ('base','size','allocated_base','allocated_size'))
        if raw['tag']!=f'HBM[{pc}]' or raw.get('used') is not True:
            raise ValueError('PC ID is not certified by a used physical HBM tag')
        if (not capacity or not length or (start|capacity|lo|length)&63 or
                start+capacity>=1<<64 or lo<start or lo+length>start+capacity):
            raise ValueError('invalid physical PC allocation range')
        banks.append(dict(pc=pc,base=start,size=capacity,allocated_base=lo,allocated_size=length))
    banks.sort(key=lambda b:b['allocated_base'])
    cursor=base;previous_end=0
    for bank in banks:
        if bank['allocated_base']!=cursor or bank['base']<previous_end:
            raise ValueError('BO physical coverage has a gap or overlap')
        cursor+=bank['allocated_size']
        previous_end=bank['base']+bank['size']
    if cursor!=base+size or len({b['pc'] for b in banks})!=len(banks):
        raise ValueError('BO physical coverage is incomplete or repeats a PC')
    return dict(local_limit=local,bo_base=base,bo_size=size,banks=banks,topology_sha256=topology)


def translate(mapping,address,nbytes=64):
    address,nbytes=number(address),number(nbytes)
    local,size=mapping['local_limit'],mapping['bo_size']
    if not nbytes or address<local or nbytes>size or address-local>size-nbytes:
        raise ValueError('address is SRAM or outside the AXI BO')
    physical=mapping['bo_base']+address-local
    for bank in mapping['banks']:
        if bank['allocated_base']<=physical and physical+nbytes<=bank['allocated_base']+bank['allocated_size']:
            return dict(pc=bank['pc'],offset=physical-bank['base'],physical_address=physical)
    raise ValueError('request crosses or misses a physical PC range')


def export(mapping):
    rows=['TNCA_DRAM_MAP_V1',f"window {mapping['local_limit']} {mapping['bo_base']} {mapping['bo_size']}"]
    rows.extend('pc {pc} {base} {size} {allocated_base} {allocated_size}'.format(**b) for b in mapping['banks'])
    return '\n'.join(rows)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--window-pointer',required=True)
    ap.add_argument('--image-pointer',required=True)
    ap.add_argument('--image-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    data=args.source.read_bytes();document=json.loads(data)
    image=pointer(document,args.image_pointer)
    if not re.fullmatch('[0-9a-f]{64}',args.image_sha256) or image!=args.image_sha256:
        raise ValueError('mapping evidence belongs to a different physical image')
    mapping=normalize(pointer(document,args.window_pointer))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    text=export(mapping);(out/'map.txt').write_text(text)
    probes=[]
    for bank in mapping['banks']:
        for physical in (bank['allocated_base'],bank['allocated_base']+bank['allocated_size']-64):
            logical=mapping['local_limit']+physical-mapping['bo_base']
            probes.append(dict(logical_address=logical,**translate(mapping,logical)))
    report=dict(schema='tnca-dram-address-map/1',status='axi-bo-pc-map-audited',
        source=str(args.source),source_sha256=hashlib.sha256(data).hexdigest(),image_sha256=image,
        window_pointer=args.window_pointer,image_pointer=args.image_pointer,mapping=mapping,
        map_sha256=hashlib.sha256(text.encode()).hexdigest(),boundary_probes=probes,
        bank_row_mapping='unverified',calibration='not-measured',
        scope='recorded BO allocation only; a new Session allocation requires its own map')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],physical_pcs=[b['pc'] for b in mapping['banks']],
                          local_limit=mapping['local_limit'],bank_row_mapping=report['bank_row_mapping']),indent=2))


if __name__=='__main__':
    main()
